#include <cuda_runtime.h>
#include <cstdint>

#ifndef SSM_GROUP_SIZE
#define SSM_GROUP_SIZE 4
#endif

#ifndef SSM_DBLOCK
#define SSM_DBLOCK 16
#endif

#ifndef SSM_TIME_TILE
#define SSM_TIME_TILE 16
#endif

#ifndef SSM_THREADS
#define SSM_THREADS 64
#endif

namespace {

constexpr float kLog2e = 1.44269504088896340736f;
constexpr unsigned kFullMask = 0xffffffffu;

// Direct 128B warp-coalesced path (kGroup == 1): 1 lane = 1 channel d (32 contiguous channels/warp).
// u, delta, y are naturally 128-byte coalesced across every warp, so only B and C need shared memory
// prefetching (1 __syncthreads per tile instead of 3, and ZERO warp shuffles!).
// Prefetches 4 time steps of (u, delta) at once into registers to hide DRAM latency (MLP = 8).
template <int kThreads, int kTimeTile>
__global__ __launch_bounds__(kThreads, (kThreads <= 64 ? 8 : 4))
void ssm_scan_dstate16_g1_kernel(
    const float* __restrict__ u,
    const float* __restrict__ delta,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    const float* __restrict__ skip,
    float* __restrict__ y,
    int seq_len,
    int d_model) {
    constexpr int kDState = 16;
    constexpr int kBCVecs = kTimeTile * (kDState / 4);

    __shared__ __align__(16) float s_B[kTimeTile][kDState];
    __shared__ __align__(16) float s_C[kTimeTile][kDState];

    const int tid = threadIdx.x;
    const int d = blockIdx.x * kThreads + tid;
    const int b = blockIdx.y;
    const bool valid_d = (d < d_model);

    float skip_d = 0.0f;
    float a_log2e[16];
    float h[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        h[i] = 0.0f;
        a_log2e[i] = 0.0f;
    }

    if (valid_d) {
        skip_d = __ldg(&skip[d]);
        const float* __restrict__ a_row = A + static_cast<size_t>(d) * kDState;
        if ((reinterpret_cast<uintptr_t>(a_row) & 15u) == 0u) {
            const float4* __restrict__ a4 = reinterpret_cast<const float4*>(a_row);
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                const float4 av = __ldg(&a4[v]);
                a_log2e[v * 4 + 0] = av.x * kLog2e;
                a_log2e[v * 4 + 1] = av.y * kLog2e;
                a_log2e[v * 4 + 2] = av.z * kLog2e;
                a_log2e[v * 4 + 3] = av.w * kLog2e;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                a_log2e[i] = __ldg(&a_row[i]) * kLog2e;
            }
        }
    }

    const size_t batch_seq_base = static_cast<size_t>(b) * seq_len;
    const float* __restrict__ B_batch = B + batch_seq_base * kDState;
    const float* __restrict__ C_batch = C + batch_seq_base * kDState;
    const float* __restrict__ u_batch = u + batch_seq_base * d_model + d;
    const float* __restrict__ dt_batch = delta + batch_seq_base * d_model + d;
    float* __restrict__ y_batch = y + batch_seq_base * d_model + d;

    const bool bc_aligned =
        ((reinterpret_cast<uintptr_t>(B_batch) & 15u) == 0u) &&
        ((reinterpret_cast<uintptr_t>(C_batch) & 15u) == 0u);
    float4* __restrict__ s_B4 = reinterpret_cast<float4*>(&s_B[0][0]);
    float4* __restrict__ s_C4 = reinterpret_cast<float4*>(&s_C[0][0]);

    int t0 = 0;
    for (; t0 + kTimeTile <= seq_len; t0 += kTimeTile) {
        if (bc_aligned) {
            const float4* __restrict__ gB4 = reinterpret_cast<const float4*>(B_batch + static_cast<size_t>(t0) * kDState);
            const float4* __restrict__ gC4 = reinterpret_cast<const float4*>(C_batch + static_cast<size_t>(t0) * kDState);
            #pragma unroll
            for (int idx = tid; idx < kBCVecs; idx += kThreads) {
                s_B4[idx] = __ldg(&gB4[idx]);
                s_C4[idx] = __ldg(&gC4[idx]);
            }
        } else {
            const float* __restrict__ gB = B_batch + static_cast<size_t>(t0) * kDState;
            const float* __restrict__ gC = C_batch + static_cast<size_t>(t0) * kDState;
            for (int idx = tid; idx < kTimeTile * kDState; idx += kThreads) {
                (&s_B[0][0])[idx] = __ldg(&gB[idx]);
                (&s_C[0][0])[idx] = __ldg(&gC[idx]);
            }
        }
        __syncthreads();

        if (valid_d) {
            size_t u_off = static_cast<size_t>(t0) * d_model;
            #pragma unroll 1
            for (int s0 = 0; s0 < kTimeTile; s0 += 4) {
                // Issue 4 coalesced 128B warp loads of u and 4 of delta upfront (MLP = 8!)
                const size_t o0 = u_off;
                const size_t o1 = u_off + d_model;
                const size_t o2 = u_off + 2 * static_cast<size_t>(d_model);
                const size_t o3 = u_off + 3 * static_cast<size_t>(d_model);
                const float u0 = __ldg(&u_batch[o0]);
                const float dt0 = __ldg(&dt_batch[o0]);
                const float u1 = __ldg(&u_batch[o1]);
                const float dt1 = __ldg(&dt_batch[o1]);
                const float u2 = __ldg(&u_batch[o2]);
                const float dt2 = __ldg(&dt_batch[o2]);
                const float u3 = __ldg(&u_batch[o3]);
                const float dt3 = __ldg(&dt_batch[o3]);

                const float u_arr[4] = {u0, u1, u2, u3};
                const float dt_arr[4] = {dt0, dt1, dt2, dt3};
                const size_t o_arr[4] = {o0, o1, o2, o3};

                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int s = s0 + k;
                    const float uv = u_arr[k];
                    const float dtv = dt_arr[k];
                    const float du = dtv * uv;
                    float y_acc = skip_d * uv;
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        h[i] = fmaf(exp2f(dtv * a_log2e[i]), h[i], du * s_B[s][i]);
                        y_acc = fmaf(s_C[s][i], h[i], y_acc);
                    }
                    y_batch[o_arr[k]] = y_acc;
                }
                u_off += 4 * static_cast<size_t>(d_model);
            }
        }
        __syncthreads();
    }

    if (t0 < seq_len) {
        const int tile_len = seq_len - t0;
        const float* __restrict__ gB = B_batch + static_cast<size_t>(t0) * kDState;
        const float* __restrict__ gC = C_batch + static_cast<size_t>(t0) * kDState;
        for (int idx = tid; idx < tile_len * kDState; idx += kThreads) {
            (&s_B[0][0])[idx] = __ldg(&gB[idx]);
            (&s_C[0][0])[idx] = __ldg(&gC[idx]);
        }
        __syncthreads();

        if (valid_d) {
            size_t u_off = static_cast<size_t>(t0) * d_model;
            for (int s = 0; s < tile_len; ++s) {
                const float uv = __ldg(&u_batch[u_off]);
                const float dtv = __ldg(&dt_batch[u_off]);
                const float du = dtv * uv;
                float y_acc = skip_d * uv;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    h[i] = fmaf(exp2f(dtv * a_log2e[i]), h[i], du * s_B[s][i]);
                    y_acc = fmaf(s_C[s][i], h[i], y_acc);
                }
                y_batch[u_off] = y_acc;
                u_off += d_model;
            }
        }
    }
}

// High-occupancy shared-memory float4 tile-prefetched fast path for d_state == 16 when
// d_model % kDBlock == 0 and u/delta/B/C/y are 16-byte aligned (covers official 4x4096x512x16).
template <int kGroup, int kDBlock, int kTimeTile>
__global__ __launch_bounds__(
    kGroup * kDBlock,
    ((kGroup * kDBlock) <= 64 ? 8 : ((kGroup * kDBlock) <= 128 ? 6 : 4)))
void ssm_scan_dstate16_tiled_kernel(
    const float* __restrict__ u,
    const float* __restrict__ delta,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    const float* __restrict__ skip,
    float* __restrict__ y,
    int seq_len,
    int d_model) {
    static_assert(kGroup == 2 || kGroup == 4 || kGroup == 8 || kGroup == 16,
                  "kGroup must be 2, 4, 8, or 16");
    static_assert((kDBlock % 4) == 0, "kDBlock must be a multiple of 4");
    constexpr int kDState = 16;
    constexpr int kThreads = kGroup * kDBlock;
    constexpr int kStatesPerThread = kDState / kGroup;
    constexpr int kVecsPerUDRow = kDBlock / 4;
    constexpr int kUDVecs = kTimeTile * kVecsPerUDRow;
    constexpr int kBCVecs = kTimeTile * (kDState / 4);

    __shared__ __align__(16) float s_u[kTimeTile][kDBlock];
    __shared__ __align__(16) float s_dt[kTimeTile][kDBlock];
    __shared__ __align__(16) float s_B[kTimeTile][kDState];
    __shared__ __align__(16) float s_C[kTimeTile][kDState];

    const int tid = threadIdx.x;
    const int sub = tid & (kGroup - 1);
    const int d_local = tid / kGroup;
    const int d_block_base = blockIdx.x * kDBlock;
    const int d = d_block_base + d_local;
    const int b = blockIdx.y;

    const float skip_d = __ldg(&skip[d]);
    float a_log2e[kStatesPerThread];
    float h[kStatesPerThread];
    #pragma unroll
    for (int i = 0; i < kStatesPerThread; ++i) {
        h[i] = 0.0f;
    }

    const int a_offset = d * kDState + sub * kStatesPerThread;
    if (kStatesPerThread == 8) {
        const float4 av0 = __ldg(reinterpret_cast<const float4*>(A + a_offset));
        const float4 av1 = __ldg(reinterpret_cast<const float4*>(A + a_offset + 4));
        a_log2e[0] = av0.x * kLog2e;
        a_log2e[1] = av0.y * kLog2e;
        a_log2e[2] = av0.z * kLog2e;
        a_log2e[3] = av0.w * kLog2e;
        a_log2e[4] = av1.x * kLog2e;
        a_log2e[5] = av1.y * kLog2e;
        a_log2e[6] = av1.z * kLog2e;
        a_log2e[7] = av1.w * kLog2e;
    } else if (kStatesPerThread == 4) {
        const float4 av = __ldg(reinterpret_cast<const float4*>(A + a_offset));
        a_log2e[0] = av.x * kLog2e;
        a_log2e[1] = av.y * kLog2e;
        a_log2e[2] = av.z * kLog2e;
        a_log2e[3] = av.w * kLog2e;
    } else if (kStatesPerThread == 2) {
        const float2 av = __ldg(reinterpret_cast<const float2*>(A + a_offset));
        a_log2e[0] = av.x * kLog2e;
        a_log2e[1] = av.y * kLog2e;
    } else {
        a_log2e[0] = __ldg(&A[a_offset]) * kLog2e;
    }

    const size_t batch_seq_base = static_cast<size_t>(b) * seq_len;
    const int d_vec4 = d_model >> 2;
    const int d_base4 = d_block_base >> 2;

    const float4* __restrict__ u_batch4 =
        reinterpret_cast<const float4*>(u + batch_seq_base * d_model) + d_base4;
    const float4* __restrict__ dt_batch4 =
        reinterpret_cast<const float4*>(delta + batch_seq_base * d_model) + d_base4;
    float4* __restrict__ y_batch4 =
        reinterpret_cast<float4*>(y + batch_seq_base * d_model) + d_base4;
    const float4* __restrict__ B_batch4 =
        reinterpret_cast<const float4*>(B + batch_seq_base * kDState);
    const float4* __restrict__ C_batch4 =
        reinterpret_cast<const float4*>(C + batch_seq_base * kDState);

    float4* __restrict__ s_u4 = reinterpret_cast<float4*>(&s_u[0][0]);
    float4* __restrict__ s_dt4 = reinterpret_cast<float4*>(&s_dt[0][0]);
    float4* __restrict__ s_B4 = reinterpret_cast<float4*>(&s_B[0][0]);
    float4* __restrict__ s_C4 = reinterpret_cast<float4*>(&s_C[0][0]);

    int t0 = 0;
    for (; t0 + kTimeTile <= seq_len; t0 += kTimeTile) {
        const size_t t0_u4 = static_cast<size_t>(t0) * d_vec4;
        const size_t t0_bc4 = static_cast<size_t>(t0) * 4;

        if (kThreads == 256 && kUDVecs == 64 && kBCVecs == 64) {
            const int idx = tid & 63;
            const int r = idx / kVecsPerUDRow;
            const int c4 = idx & (kVecsPerUDRow - 1);
            if (tid < 64) {
                s_u4[idx] = __ldg(&u_batch4[t0_u4 + r * d_vec4 + c4]);
            } else if (tid < 128) {
                s_dt4[idx] = __ldg(&dt_batch4[t0_u4 + r * d_vec4 + c4]);
            } else if (tid < 192) {
                s_B4[idx] = __ldg(&B_batch4[t0_bc4 + idx]);
            } else {
                s_C4[idx] = __ldg(&C_batch4[t0_bc4 + idx]);
            }
        } else {
            #pragma unroll
            for (int idx = tid; idx < kUDVecs; idx += kThreads) {
                const int r = idx / kVecsPerUDRow;
                const int c4 = idx & (kVecsPerUDRow - 1);
                const size_t off4 = t0_u4 + static_cast<size_t>(r) * d_vec4 + c4;
                s_u4[idx] = __ldg(&u_batch4[off4]);
                s_dt4[idx] = __ldg(&dt_batch4[off4]);
            }
            #pragma unroll
            for (int idx = tid; idx < kBCVecs; idx += kThreads) {
                s_B4[idx] = __ldg(&B_batch4[t0_bc4 + idx]);
                s_C4[idx] = __ldg(&C_batch4[t0_bc4 + idx]);
            }
        }
        __syncthreads();

        #pragma unroll 4
        for (int s = 0; s < kTimeTile; ++s) {
            const float u_val = s_u[s][d_local];
            const float dt = s_dt[s][d_local];
            const float du = dt * u_val;

            float y_part = 0.0f;
            if (kStatesPerThread == 8) {
                const float4 bv0 = reinterpret_cast<const float4*>(&s_B[s][0])[sub * 2 + 0];
                const float4 bv1 = reinterpret_cast<const float4*>(&s_B[s][0])[sub * 2 + 1];
                const float4 cv0 = reinterpret_cast<const float4*>(&s_C[s][0])[sub * 2 + 0];
                const float4 cv1 = reinterpret_cast<const float4*>(&s_C[s][0])[sub * 2 + 1];
                h[0] = fmaf(exp2f(dt * a_log2e[0]), h[0], du * bv0.x);
                h[1] = fmaf(exp2f(dt * a_log2e[1]), h[1], du * bv0.y);
                h[2] = fmaf(exp2f(dt * a_log2e[2]), h[2], du * bv0.z);
                h[3] = fmaf(exp2f(dt * a_log2e[3]), h[3], du * bv0.w);
                h[4] = fmaf(exp2f(dt * a_log2e[4]), h[4], du * bv1.x);
                h[5] = fmaf(exp2f(dt * a_log2e[5]), h[5], du * bv1.y);
                h[6] = fmaf(exp2f(dt * a_log2e[6]), h[6], du * bv1.z);
                h[7] = fmaf(exp2f(dt * a_log2e[7]), h[7], du * bv1.w);
                y_part = fmaf(cv0.x, h[0], fmaf(cv0.y, h[1], fmaf(cv0.z, h[2], fmaf(cv0.w, h[3],
                         fmaf(cv1.x, h[4], fmaf(cv1.y, h[5], fmaf(cv1.z, h[6], cv1.w * h[7])))))));
            } else if (kStatesPerThread == 4) {
                const float4 bv = reinterpret_cast<const float4*>(&s_B[s][0])[sub];
                const float4 cv = reinterpret_cast<const float4*>(&s_C[s][0])[sub];
                h[0] = fmaf(exp2f(dt * a_log2e[0]), h[0], du * bv.x);
                h[1] = fmaf(exp2f(dt * a_log2e[1]), h[1], du * bv.y);
                h[2] = fmaf(exp2f(dt * a_log2e[2]), h[2], du * bv.z);
                h[3] = fmaf(exp2f(dt * a_log2e[3]), h[3], du * bv.w);
                y_part = fmaf(cv.x, h[0],
                         fmaf(cv.y, h[1],
                         fmaf(cv.z, h[2], cv.w * h[3])));
            } else if (kStatesPerThread == 2) {
                const float2 bv = reinterpret_cast<const float2*>(&s_B[s][0])[sub];
                const float2 cv = reinterpret_cast<const float2*>(&s_C[s][0])[sub];
                h[0] = fmaf(exp2f(dt * a_log2e[0]), h[0], du * bv.x);
                h[1] = fmaf(exp2f(dt * a_log2e[1]), h[1], du * bv.y);
                y_part = fmaf(cv.x, h[0], cv.y * h[1]);
            } else {
                const float b_val = s_B[s][sub];
                const float c_val = s_C[s][sub];
                h[0] = fmaf(exp2f(dt * a_log2e[0]), h[0], du * b_val);
                y_part = c_val * h[0];
            }

            if (kGroup >= 16) {
                y_part += __shfl_xor_sync(kFullMask, y_part, 8);
            }
            if (kGroup >= 8) {
                y_part += __shfl_xor_sync(kFullMask, y_part, 4);
            }
            if (kGroup >= 4) {
                y_part += __shfl_xor_sync(kFullMask, y_part, 2);
            }
            y_part += __shfl_xor_sync(kFullMask, y_part, 1);

            if (sub == 0) {
                s_u[s][d_local] = fmaf(skip_d, u_val, y_part);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int idx = tid; idx < kUDVecs; idx += kThreads) {
            const int r = idx / kVecsPerUDRow;
            const int c4 = idx & (kVecsPerUDRow - 1);
            y_batch4[t0_u4 + static_cast<size_t>(r) * d_vec4 + c4] = s_u4[idx];
        }
        __syncthreads();
    }

    if (t0 < seq_len) {
        const int tile_len = seq_len - t0;
        const int ud_vecs = tile_len * kVecsPerUDRow;
        const int bc_vecs = tile_len * 4;
        const size_t t0_u4 = static_cast<size_t>(t0) * d_vec4;
        const size_t t0_bc4 = static_cast<size_t>(t0) * 4;

        for (int idx = tid; idx < ud_vecs; idx += kThreads) {
            const int r = idx / kVecsPerUDRow;
            const int c4 = idx & (kVecsPerUDRow - 1);
            const size_t off4 = t0_u4 + static_cast<size_t>(r) * d_vec4 + c4;
            s_u4[idx] = __ldg(&u_batch4[off4]);
            s_dt4[idx] = __ldg(&dt_batch4[off4]);
        }
        for (int idx = tid; idx < bc_vecs; idx += kThreads) {
            s_B4[idx] = __ldg(&B_batch4[t0_bc4 + idx]);
            s_C4[idx] = __ldg(&C_batch4[t0_bc4 + idx]);
        }
        __syncthreads();

        for (int s = 0; s < tile_len; ++s) {
            const float u_val = s_u[s][d_local];
            const float dt = s_dt[s][d_local];
            const float du = dt * u_val;

            float y_part = 0.0f;
            const int base_n = sub * kStatesPerThread;
            #pragma unroll
            for (int i = 0; i < kStatesPerThread; ++i) {
                const float b_val = s_B[s][base_n + i];
                const float c_val = s_C[s][base_n + i];
                h[i] = fmaf(exp2f(dt * a_log2e[i]), h[i], du * b_val);
                y_part = fmaf(c_val, h[i], y_part);
            }

            if (kGroup >= 16) {
                y_part += __shfl_xor_sync(kFullMask, y_part, 8);
            }
            if (kGroup >= 8) {
                y_part += __shfl_xor_sync(kFullMask, y_part, 4);
            }
            if (kGroup >= 4) {
                y_part += __shfl_xor_sync(kFullMask, y_part, 2);
            }
            y_part += __shfl_xor_sync(kFullMask, y_part, 1);

            if (sub == 0) {
                s_u[s][d_local] = fmaf(skip_d, u_val, y_part);
            }
        }
        __syncthreads();

        for (int idx = tid; idx < ud_vecs; idx += kThreads) {
            const int r = idx / kVecsPerUDRow;
            const int c4 = idx & (kVecsPerUDRow - 1);
            y_batch4[t0_u4 + static_cast<size_t>(r) * d_vec4 + c4] = s_u4[idx];
        }
    }
}

// Fast path for d_state == 16 (official benchmark: batch=4, seq_len=4096, d_model=512, d_state=16).
// kGroup contiguous lanes within a warp cooperate on one (b, d) channel by splitting the 16 states
// into kStatesPerThread = 16 / kGroup registers per thread:
//   - kGroup = 1: 1 thread holds all 16 states
//   - kGroup = 2: 2 lanes hold 8 states each (1 shfl_xor_sync per step)
//   - kGroup = 4: 4 lanes hold 4 states (1 float4) each (2 shfl_xor_sync per step, 4x warps -> full 40 SMs)
//   - kGroup = 8: 8 lanes hold 2 states each (3 shfl_xor_sync per step, 8x warps)
template <int kGroup, int kThreads, int kTimeTile, bool kWarpUniformD>
__global__ __launch_bounds__(
    kThreads,
    (kThreads <= 64 ? (kGroup >= 4 ? 12 : 8) : (kThreads <= 128 ? (kGroup >= 4 ? 6 : 4) : 4)))
void ssm_scan_dstate16_kernel(
    const float* __restrict__ u,
    const float* __restrict__ delta,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    const float* __restrict__ skip,
    float* __restrict__ y,
    int seq_len,
    int d_model) {
    static_assert(kGroup == 1 || kGroup == 2 || kGroup == 4 || kGroup == 8,
                  "SSM_GROUP_SIZE must be 1, 2, 4, or 8");
    constexpr int kDState = 16;
    constexpr int kStatesPerThread = kDState / kGroup;
    constexpr int kChannelsPerBlock = kThreads / kGroup;
    constexpr int kVec4PerTile = (kTimeTile * kDState) / 4;

    __shared__ __align__(16) float s_B[kTimeTile][16];
    __shared__ __align__(16) float s_C[kTimeTile][16];

    const int tid = threadIdx.x;
    const int sub = (kGroup > 1) ? (tid & (kGroup - 1)) : 0;
    const int src_lane = (tid & 31) & ~(kGroup - 1);
    const unsigned shfl_mask =
        kWarpUniformD ? kFullMask : (((1u << kGroup) - 1u) << src_lane);
    const int d_local = tid / kGroup;
    const int d = blockIdx.x * kChannelsPerBlock + d_local;
    const int b = blockIdx.y;
    const bool valid_d = (d < d_model);

    float a_log2e[kStatesPerThread];
    float h[kStatesPerThread];
    #pragma unroll
    for (int i = 0; i < kStatesPerThread; ++i) {
        h[i] = 0.0f;
        a_log2e[i] = 0.0f;
    }

    float skip_d = 0.0f;
    if (valid_d) {
        skip_d = __ldg(&skip[d]);
        const int a_offset = d * kDState + sub * kStatesPerThread;
        if (kStatesPerThread == 4 && ((reinterpret_cast<uintptr_t>(A) & 15u) == 0u)) {
            const float4 av = __ldg(reinterpret_cast<const float4*>(A + a_offset));
            a_log2e[0] = av.x * kLog2e;
            a_log2e[1] = av.y * kLog2e;
            a_log2e[2] = av.z * kLog2e;
            a_log2e[3] = av.w * kLog2e;
        } else {
            #pragma unroll
            for (int i = 0; i < kStatesPerThread; ++i) {
                a_log2e[i] = __ldg(&A[a_offset + i]) * kLog2e;
            }
        }
    }

    const size_t batch_seq_base = static_cast<size_t>(b) * seq_len;
    const float* __restrict__ B_batch = B + batch_seq_base * kDState;
    const float* __restrict__ C_batch = C + batch_seq_base * kDState;
    const float* __restrict__ u_batch = u + batch_seq_base * d_model + d;
    const float* __restrict__ delta_batch = delta + batch_seq_base * d_model + d;
    float* __restrict__ y_batch = y + batch_seq_base * d_model + d;

    const bool bc_aligned =
        ((reinterpret_cast<uintptr_t>(B_batch) & 15u) == 0u) &&
        ((reinterpret_cast<uintptr_t>(C_batch) & 15u) == 0u);

    int t0 = 0;
    // Full kTimeTile loop (completely unrolled inner micro-tiles, zero step-bounds branch)
    for (; t0 + kTimeTile <= seq_len; t0 += kTimeTile) {
        const float* __restrict__ B_tile = B_batch + static_cast<size_t>(t0) * kDState;
        const float* __restrict__ C_tile = C_batch + static_cast<size_t>(t0) * kDState;
        float4* __restrict__ sB4 = reinterpret_cast<float4*>(&s_B[0][0]);
        float4* __restrict__ sC4 = reinterpret_cast<float4*>(&s_C[0][0]);

        if (bc_aligned) {
            const float4* __restrict__ gB4 = reinterpret_cast<const float4*>(B_tile);
            const float4* __restrict__ gC4 = reinterpret_cast<const float4*>(C_tile);
            #pragma unroll
            for (int idx = tid; idx < kVec4PerTile; idx += kThreads) {
                sB4[idx] = __ldg(&gB4[idx]);
                sC4[idx] = __ldg(&gC4[idx]);
            }
        } else {
            for (int idx = tid; idx < kTimeTile * kDState; idx += kThreads) {
                (&s_B[0][0])[idx] = __ldg(&B_tile[idx]);
                (&s_C[0][0])[idx] = __ldg(&C_tile[idx]);
            }
        }
        __syncthreads();

        if (valid_d) {
            size_t u_off = static_cast<size_t>(t0) * d_model;
            #pragma unroll 2
            for (int s = 0; s < kTimeTile; ++s) {
                float u_val = 0.0f;
                float dt = 0.0f;
                if (kGroup == 1) {
                    u_val = __ldg(&u_batch[u_off]);
                    dt = __ldg(&delta_batch[u_off]);
                } else {
                    // Only sub == 0 issues global loads for (b, t, d), then broadcasts via shfl_sync!
                    // Across the warp, the (32 / kGroup) leader lanes access adjacent d channels,
                    // cutting redundant L1 tag lookups by kGroup x while keeping 100% coalesced sector reads.
                    if (sub == 0) {
                        u_val = __ldg(&u_batch[u_off]);
                        dt = __ldg(&delta_batch[u_off]);
                    }
                    u_val = __shfl_sync(shfl_mask, u_val, src_lane);
                    dt = __shfl_sync(shfl_mask, dt, src_lane);
                }
                const float du = dt * u_val;

                float y_part = 0.0f;
                if (kStatesPerThread == 4) {
                    const float4 bv = reinterpret_cast<const float4*>(&s_B[s][0])[sub];
                    const float4 cv = reinterpret_cast<const float4*>(&s_C[s][0])[sub];
                    h[0] = fmaf(exp2f(dt * a_log2e[0]), h[0], du * bv.x);
                    h[1] = fmaf(exp2f(dt * a_log2e[1]), h[1], du * bv.y);
                    h[2] = fmaf(exp2f(dt * a_log2e[2]), h[2], du * bv.z);
                    h[3] = fmaf(exp2f(dt * a_log2e[3]), h[3], du * bv.w);
                    y_part = fmaf(cv.x, h[0],
                             fmaf(cv.y, h[1],
                             fmaf(cv.z, h[2], cv.w * h[3])));
                } else if (kStatesPerThread == 2) {
                    const float2 bv = reinterpret_cast<const float2*>(&s_B[s][0])[sub];
                    const float2 cv = reinterpret_cast<const float2*>(&s_C[s][0])[sub];
                    h[0] = fmaf(exp2f(dt * a_log2e[0]), h[0], du * bv.x);
                    h[1] = fmaf(exp2f(dt * a_log2e[1]), h[1], du * bv.y);
                    y_part = fmaf(cv.x, h[0], cv.y * h[1]);
                } else {
                    const int base_n = sub * kStatesPerThread;
                    #pragma unroll
                    for (int i = 0; i < kStatesPerThread; ++i) {
                        const float b_val = s_B[s][base_n + i];
                        const float c_val = s_C[s][base_n + i];
                        h[i] = fmaf(exp2f(dt * a_log2e[i]), h[i], du * b_val);
                        y_part = fmaf(c_val, h[i], y_part);
                    }
                }

                if (kGroup >= 8) {
                    y_part += __shfl_xor_sync(shfl_mask, y_part, 4);
                }
                if (kGroup >= 4) {
                    y_part += __shfl_xor_sync(shfl_mask, y_part, 2);
                }
                if (kGroup >= 2) {
                    y_part += __shfl_xor_sync(shfl_mask, y_part, 1);
                }

                if (sub == 0) {
                    y_batch[u_off] = fmaf(skip_d, u_val, y_part);
                }
                u_off += d_model;
            }
        }
        __syncthreads();
    }

    // Remainder steps (< kTimeTile)
    if (t0 < seq_len) {
        const int tile_len = seq_len - t0;
        const int elems = tile_len * kDState;
        const float* __restrict__ B_tile = B_batch + static_cast<size_t>(t0) * kDState;
        const float* __restrict__ C_tile = C_batch + static_cast<size_t>(t0) * kDState;

        if (bc_aligned) {
            const int vecs = tile_len * (kDState / 4);
            const float4* __restrict__ gB4 = reinterpret_cast<const float4*>(B_tile);
            const float4* __restrict__ gC4 = reinterpret_cast<const float4*>(C_tile);
            float4* __restrict__ sB4 = reinterpret_cast<float4*>(&s_B[0][0]);
            float4* __restrict__ sC4 = reinterpret_cast<float4*>(&s_C[0][0]);
            for (int idx = tid; idx < vecs; idx += kThreads) {
                sB4[idx] = __ldg(&gB4[idx]);
                sC4[idx] = __ldg(&gC4[idx]);
            }
        } else {
            for (int idx = tid; idx < elems; idx += kThreads) {
                (&s_B[0][0])[idx] = __ldg(&B_tile[idx]);
                (&s_C[0][0])[idx] = __ldg(&C_tile[idx]);
            }
        }
        __syncthreads();

        if (valid_d) {
            size_t u_off = static_cast<size_t>(t0) * d_model;
            for (int s = 0; s < tile_len; ++s) {
                float u_val = 0.0f;
                float dt = 0.0f;
                if (kGroup == 1) {
                    u_val = __ldg(&u_batch[u_off]);
                    dt = __ldg(&delta_batch[u_off]);
                } else {
                    if (sub == 0) {
                        u_val = __ldg(&u_batch[u_off]);
                        dt = __ldg(&delta_batch[u_off]);
                    }
                    u_val = __shfl_sync(shfl_mask, u_val, src_lane);
                    dt = __shfl_sync(shfl_mask, dt, src_lane);
                }
                const float du = dt * u_val;

                float y_part = 0.0f;
                const int base_n = sub * kStatesPerThread;
                #pragma unroll
                for (int i = 0; i < kStatesPerThread; ++i) {
                    const float b_val = s_B[s][base_n + i];
                    const float c_val = s_C[s][base_n + i];
                    h[i] = fmaf(exp2f(dt * a_log2e[i]), h[i], du * b_val);
                    y_part = fmaf(c_val, h[i], y_part);
                }

                if (kGroup >= 8) {
                    y_part += __shfl_xor_sync(shfl_mask, y_part, 4);
                }
                if (kGroup >= 4) {
                    y_part += __shfl_xor_sync(shfl_mask, y_part, 2);
                }
                if (kGroup >= 2) {
                    y_part += __shfl_xor_sync(shfl_mask, y_part, 1);
                }

                if (sub == 0) {
                    y_batch[u_off] = fmaf(skip_d, u_val, y_part);
                }
                u_off += d_model;
            }
        }
    }
}

// General path for arbitrary 1 <= d_state <= 64 (including odd d_state, d_state=1, d_state=64).
// Uses kGroup lanes per channel so every thread handles at most kMaxStatesPerThread (<= 8 or 16)
// compile-time indexed registers — guaranteeing zero local-memory register spills for all d_state <= 64!
template <int kGroup, int kMaxStatesPerThread, int kThreads, int kTimeTile, int kMaxState>
__global__ __launch_bounds__(kThreads, 4) void ssm_scan_general_kernel(
    const float* __restrict__ u,
    const float* __restrict__ delta,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    const float* __restrict__ skip,
    float* __restrict__ y,
    int seq_len,
    int d_model,
    int d_state) {
    static_assert(kGroup == 1 || kGroup == 2 || kGroup == 4 || kGroup == 8,
                  "kGroup must be 1, 2, 4, or 8");
    constexpr int kChannelsPerBlock = kThreads / kGroup;

    __shared__ float s_B[kTimeTile * kMaxState];
    __shared__ float s_C[kTimeTile * kMaxState];

    const int tid = threadIdx.x;
    const int sub = (kGroup > 1) ? (tid & (kGroup - 1)) : 0;
    const int src_lane = (tid & 31) & ~(kGroup - 1);
    const unsigned sub_mask = ((1u << kGroup) - 1u) << src_lane;
    const int d_local = tid / kGroup;
    const int d = blockIdx.x * kChannelsPerBlock + d_local;
    const int b = blockIdx.y;
    const bool valid_d = (d < d_model);

    float a_log2e[kMaxStatesPerThread];
    float h[kMaxStatesPerThread];
    #pragma unroll
    for (int i = 0; i < kMaxStatesPerThread; ++i) {
        h[i] = 0.0f;
        a_log2e[i] = 0.0f;
    }

    float skip_d = 0.0f;
    if (valid_d) {
        skip_d = __ldg(&skip[d]);
        const float* __restrict__ a_row = A + static_cast<size_t>(d) * d_state;
        #pragma unroll
        for (int i = 0; i < kMaxStatesPerThread; ++i) {
            const int n = sub + i * kGroup;
            if (n < d_state) {
                a_log2e[i] = __ldg(&a_row[n]) * kLog2e;
            }
        }
    }

    const size_t batch_seq_base = static_cast<size_t>(b) * seq_len;
    const float* __restrict__ B_batch = B + batch_seq_base * d_state;
    const float* __restrict__ C_batch = C + batch_seq_base * d_state;
    const float* __restrict__ u_batch = u + batch_seq_base * d_model + d;
    const float* __restrict__ delta_batch = delta + batch_seq_base * d_model + d;
    float* __restrict__ y_batch = y + batch_seq_base * d_model + d;

    for (int t0 = 0; t0 < seq_len; t0 += kTimeTile) {
        const int tile_len = (t0 + kTimeTile <= seq_len) ? kTimeTile : (seq_len - t0);
        const int elems = tile_len * d_state;
        const float* __restrict__ B_tile = B_batch + static_cast<size_t>(t0) * d_state;
        const float* __restrict__ C_tile = C_batch + static_cast<size_t>(t0) * d_state;

        for (int idx = tid; idx < elems; idx += kThreads) {
            s_B[idx] = __ldg(&B_tile[idx]);
            s_C[idx] = __ldg(&C_tile[idx]);
        }
        __syncthreads();

        if (valid_d) {
            size_t u_off = static_cast<size_t>(t0) * d_model;
            for (int s = 0; s < tile_len; ++s) {
                float u_val = 0.0f;
                float dt = 0.0f;
                if (kGroup == 1) {
                    u_val = __ldg(&u_batch[u_off]);
                    dt = __ldg(&delta_batch[u_off]);
                } else {
                    if (sub == 0) {
                        u_val = __ldg(&u_batch[u_off]);
                        dt = __ldg(&delta_batch[u_off]);
                    }
                    u_val = __shfl_sync(sub_mask, u_val, src_lane);
                    dt = __shfl_sync(sub_mask, dt, src_lane);
                }
                const float du = dt * u_val;
                const int row_base = s * d_state;

                float y_part = 0.0f;
                #pragma unroll
                for (int i = 0; i < kMaxStatesPerThread; ++i) {
                    const int n = sub + i * kGroup;
                    if (n < d_state) {
                        const float b_val = s_B[row_base + n];
                        const float c_val = s_C[row_base + n];
                        h[i] = fmaf(exp2f(dt * a_log2e[i]), h[i], du * b_val);
                        y_part = fmaf(c_val, h[i], y_part);
                    }
                }

                if (kGroup >= 8) {
                    y_part += __shfl_xor_sync(sub_mask, y_part, 4);
                }
                if (kGroup >= 4) {
                    y_part += __shfl_xor_sync(sub_mask, y_part, 2);
                }
                if (kGroup >= 2) {
                    y_part += __shfl_xor_sync(sub_mask, y_part, 1);
                }

                if (sub == 0) {
                    y_batch[u_off] = fmaf(skip_d, u_val, y_part);
                }
                u_off += d_model;
            }
        }
        __syncthreads();
    }
}

}  // namespace

extern "C" void solve(const float* u,
                      const float* delta,
                      const float* A,
                      const float* B,
                      const float* C,
                      const float* skip,
                      float* y,
                      int batch,
                      int seq_len,
                      int d_model,
                      int d_state) {
    if (batch <= 0 || seq_len <= 0 || d_model <= 0 || d_state <= 0) {
        return;
    }

    if (d_state == 16) {
        constexpr int kGroup = SSM_GROUP_SIZE;
        constexpr int kDBlock = SSM_DBLOCK;
        constexpr int kTimeTile = SSM_TIME_TILE;
        if (kGroup == 1) {
            const dim3 grid((d_model + kDBlock - 1) / kDBlock, batch);
            ssm_scan_dstate16_g1_kernel<kDBlock, kTimeTile>
                <<<grid, kDBlock>>>(u, delta, A, B, C, skip, y, seq_len, d_model);
            return;
        }
        const uintptr_t align_bits =
            reinterpret_cast<uintptr_t>(u) |
            reinterpret_cast<uintptr_t>(delta) |
            reinterpret_cast<uintptr_t>(A) |
            reinterpret_cast<uintptr_t>(B) |
            reinterpret_cast<uintptr_t>(C) |
            reinterpret_cast<uintptr_t>(y);
        if ((d_model % kDBlock == 0) && ((align_bits & 15u) == 0u)) {
            constexpr int kSafeGroup = (kGroup >= 2) ? kGroup : 2;
            const dim3 grid(d_model / kDBlock, batch);
            ssm_scan_dstate16_tiled_kernel<kSafeGroup, kDBlock, kTimeTile>
                <<<grid, kSafeGroup * kDBlock>>>(u, delta, A, B, C, skip, y, seq_len, d_model);
            return;
        }
        constexpr int kFbGroup = 4;
        constexpr int kFbThreads = 64;
        constexpr int kFbTile = 32;
        constexpr int kChannelsPerBlock = kFbThreads / kFbGroup;
        constexpr int kChannelsPerWarp = 32 / kFbGroup;
        const dim3 grid((d_model + kChannelsPerBlock - 1) / kChannelsPerBlock, batch);
        if ((d_model & (kChannelsPerWarp - 1)) == 0) {
            ssm_scan_dstate16_kernel<kFbGroup, kFbThreads, kFbTile, true>
                <<<grid, kFbThreads>>>(u, delta, A, B, C, skip, y, seq_len, d_model);
        } else {
            ssm_scan_dstate16_kernel<kFbGroup, kFbThreads, kFbTile, false>
                <<<grid, kFbThreads>>>(u, delta, A, B, C, skip, y, seq_len, d_model);
        }
        return;
    }

    if (d_state <= 8) {
        // 1 <= d_state <= 8: 2 lanes per channel, at most 4 states/thread
        constexpr int kGroup = 2;
        constexpr int kThreads = 64;
        constexpr int kChannelsPerBlock = kThreads / kGroup;
        const dim3 grid((d_model + kChannelsPerBlock - 1) / kChannelsPerBlock, batch);
        ssm_scan_general_kernel<kGroup, 4, kThreads, 32, 8>
            <<<grid, kThreads>>>(u, delta, A, B, C, skip, y, seq_len, d_model, d_state);
    } else if (d_state <= 32) {
        // 9 <= d_state <= 32: 4 lanes per channel, at most 8 states/thread
        constexpr int kGroup = 4;
        constexpr int kThreads = 128;
        constexpr int kChannelsPerBlock = kThreads / kGroup;
        const dim3 grid((d_model + kChannelsPerBlock - 1) / kChannelsPerBlock, batch);
        ssm_scan_general_kernel<kGroup, 8, kThreads, 32, 32>
            <<<grid, kThreads>>>(u, delta, A, B, C, skip, y, seq_len, d_model, d_state);
    } else {
        // 33 <= d_state <= 64: 8 lanes per channel, at most 8 states/thread (zero spill!)
        constexpr int kGroup = 8;
        constexpr int kThreads = 128;
        constexpr int kChannelsPerBlock = kThreads / kGroup;
        const dim3 grid((d_model + kChannelsPerBlock - 1) / kChannelsPerBlock, batch);
        ssm_scan_general_kernel<kGroup, 8, kThreads, 16, 64>
            <<<grid, kThreads>>>(u, delta, A, B, C, skip, y, seq_len, d_model, d_state);
    }
}
