#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kBr = 8;
constexpr int kBc = 32;
constexpr int kThreads = 128;  // 8x16: ty in [0..7] (1 Q row), tx in [0..15] (2 K cols / float4 O cols)
constexpr int kMaxElemsPerLane = 32;  // Fallback supports d up to 1024

__device__ __forceinline__ float subwarp16_max(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
    return val;
}

__device__ __forceinline__ float subwarp16_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 1);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    return val;
}

// ============================================================================
// Path 1: 3-stage 128x128 / 8x8 double-buffered FP32 SGEMM pipeline for large d
// ============================================================================

__device__ __forceinline__ void fma4x4(float4 a, float4 b, float c[4][4]) {
    c[0][0] = fmaf(a.x, b.x, c[0][0]); c[0][1] = fmaf(a.x, b.y, c[0][1]);
    c[0][2] = fmaf(a.x, b.z, c[0][2]); c[0][3] = fmaf(a.x, b.w, c[0][3]);
    c[1][0] = fmaf(a.y, b.x, c[1][0]); c[1][1] = fmaf(a.y, b.y, c[1][1]);
    c[1][2] = fmaf(a.y, b.z, c[1][2]); c[1][3] = fmaf(a.y, b.w, c[1][3]);
    c[2][0] = fmaf(a.z, b.x, c[2][0]); c[2][1] = fmaf(a.z, b.y, c[2][1]);
    c[2][2] = fmaf(a.z, b.z, c[2][2]); c[2][3] = fmaf(a.z, b.w, c[2][3]);
    c[3][0] = fmaf(a.w, b.x, c[3][0]); c[3][1] = fmaf(a.w, b.y, c[3][1]);
    c[3][2] = fmaf(a.w, b.z, c[3][2]); c[3][3] = fmaf(a.w, b.w, c[3][3]);
}

// Stage 1: S[M, N] = (Q[M, d] @ K[N, d]^T) * scale + alibi_alpha * (i - j)
__global__ void __launch_bounds__(256, 2)
alibi_qkt_128x128_kernel(const float* __restrict__ Q,
                         const float* __restrict__ K,
                         float* __restrict__ S,
                         int M,
                         int N,
                         int d,
                         float scale,
                         float alibi_alpha) {
    __shared__ float sA[2][8][132];
    __shared__ float sB[2][8][132];

    const int tid = threadIdx.x;
    const int ty = tid >> 4;   // 0..15
    const int tx = tid & 15;   // 0..15
    const int bm = blockIdx.y * 128;
    const int bn = blockIdx.x * 128;

    const int load_row = tid >> 1;         // 0..127
    const int load_col = (tid & 1) << 2;   // 0 or 4

    float c00[4][4] = {{0.0f}}, c01[4][4] = {{0.0f}};
    float c10[4][4] = {{0.0f}}, c11[4][4] = {{0.0f}};

    // Prefetch stage 0
    {
        float4 qv = *reinterpret_cast<const float4*>(&Q[static_cast<size_t>(bm + load_row) * d + load_col]);
        float4 kv = *reinterpret_cast<const float4*>(&K[static_cast<size_t>(bn + load_row) * d + load_col]);
        sA[0][load_col + 0][load_row] = qv.x;
        sA[0][load_col + 1][load_row] = qv.y;
        sA[0][load_col + 2][load_row] = qv.z;
        sA[0][load_col + 3][load_row] = qv.w;
        sB[0][load_col + 0][load_row] = kv.x;
        sB[0][load_col + 1][load_row] = kv.y;
        sB[0][load_col + 2][load_row] = kv.z;
        sB[0][load_col + 3][load_row] = kv.w;
    }
    __syncthreads();

    int stage = 0;
    for (int bk = 0; bk < d; bk += 8) {
        float4 next_q, next_k;
        const bool has_next = (bk + 8 < d);
        if (has_next) {
            next_q = *reinterpret_cast<const float4*>(&Q[static_cast<size_t>(bm + load_row) * d + bk + 8 + load_col]);
            next_k = *reinterpret_cast<const float4*>(&K[static_cast<size_t>(bn + load_row) * d + bk + 8 + load_col]);
        }

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float4 a0 = *reinterpret_cast<const float4*>(&sA[stage][k][ty * 4]);
            float4 a1 = *reinterpret_cast<const float4*>(&sA[stage][k][ty * 4 + 64]);
            float4 b0 = *reinterpret_cast<const float4*>(&sB[stage][k][tx * 4]);
            float4 b1 = *reinterpret_cast<const float4*>(&sB[stage][k][tx * 4 + 64]);
            fma4x4(a0, b0, c00);
            fma4x4(a0, b1, c01);
            fma4x4(a1, b0, c10);
            fma4x4(a1, b1, c11);
        }

        if (has_next) {
            stage ^= 1;
            sA[stage][load_col + 0][load_row] = next_q.x;
            sA[stage][load_col + 1][load_row] = next_q.y;
            sA[stage][load_col + 2][load_row] = next_q.z;
            sA[stage][load_col + 3][load_row] = next_q.w;
            sB[stage][load_col + 0][load_row] = next_k.x;
            sB[stage][load_col + 1][load_row] = next_k.y;
            sB[stage][load_col + 2][load_row] = next_k.z;
            sB[stage][load_col + 3][load_row] = next_k.w;
            __syncthreads();
        }
    }

    auto store_quad = [&](int r_off, int c_off, float c[4][4]) {
        const int col0 = bn + tx * 4 + c_off;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = bm + ty * 4 + r_off + i;
            float4 out = make_float4(
                fmaf(c[i][0], scale, alibi_alpha * static_cast<float>(row - (col0 + 0))),
                fmaf(c[i][1], scale, alibi_alpha * static_cast<float>(row - (col0 + 1))),
                fmaf(c[i][2], scale, alibi_alpha * static_cast<float>(row - (col0 + 2))),
                fmaf(c[i][3], scale, alibi_alpha * static_cast<float>(row - (col0 + 3))));
            *reinterpret_cast<float4*>(&S[static_cast<size_t>(row) * N + col0]) = out;
        }
    };
    store_quad(0, 0, c00);
    store_quad(0, 64, c01);
    store_quad(64, 0, c10);
    store_quad(64, 64, c11);
}

// Stage 2: Fused in-place row-wise softmax over S[M, N] (1 block of 256 threads per row)
__global__ void alibi_row_softmax_kernel(float* __restrict__ S, int N) {
    __shared__ float s_reduce[8];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int vec_n = N >> 2;
    float4* row_ptr = reinterpret_cast<float4*>(S + static_cast<size_t>(row) * N);

    // Fast register-cached path for N <= 2048 (at most 2 float4s per thread)
    float4 regs[2];
    float local_max = -FLT_MAX;
    if (vec_n <= 512) {
        if (tid < vec_n) {
            regs[0] = row_ptr[tid];
            local_max = fmaxf(local_max, fmaxf(fmaxf(regs[0].x, regs[0].y), fmaxf(regs[0].z, regs[0].w)));
        }
        if (tid + 256 < vec_n) {
            regs[1] = row_ptr[tid + 256];
            local_max = fmaxf(local_max, fmaxf(fmaxf(regs[1].x, regs[1].y), fmaxf(regs[1].z, regs[1].w)));
        }
    } else {
        for (int idx = tid; idx < vec_n; idx += 256) {
            float4 v = row_ptr[idx];
            local_max = fmaxf(local_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (lane == 0) s_reduce[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < 8) ? s_reduce[lane] : -FLT_MAX;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
        }
        if (lane == 0) s_reduce[0] = v;
    }
    __syncthreads();
    const float row_max = s_reduce[0];

    float local_sum = 0.0f;
    if (vec_n <= 512) {
        if (tid < vec_n) {
            regs[0].x = __expf(regs[0].x - row_max);
            regs[0].y = __expf(regs[0].y - row_max);
            regs[0].z = __expf(regs[0].z - row_max);
            regs[0].w = __expf(regs[0].w - row_max);
            local_sum += regs[0].x + regs[0].y + regs[0].z + regs[0].w;
        }
        if (tid + 256 < vec_n) {
            regs[1].x = __expf(regs[1].x - row_max);
            regs[1].y = __expf(regs[1].y - row_max);
            regs[1].z = __expf(regs[1].z - row_max);
            regs[1].w = __expf(regs[1].w - row_max);
            local_sum += regs[1].x + regs[1].y + regs[1].z + regs[1].w;
        }
    } else {
        for (int idx = tid; idx < vec_n; idx += 256) {
            float4 v = row_ptr[idx];
            v.x = __expf(v.x - row_max);
            v.y = __expf(v.y - row_max);
            v.z = __expf(v.z - row_max);
            v.w = __expf(v.w - row_max);
            local_sum += v.x + v.y + v.z + v.w;
            row_ptr[idx] = v;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane == 0) s_reduce[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane < 8) ? s_reduce[lane] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, offset);
        }
        if (lane == 0) s_reduce[0] = 1.0f / v;
    }
    __syncthreads();
    const float inv_sum = s_reduce[0];

    if (vec_n <= 512) {
        if (tid < vec_n) {
            row_ptr[tid] = make_float4(regs[0].x * inv_sum, regs[0].y * inv_sum,
                                       regs[0].z * inv_sum, regs[0].w * inv_sum);
        }
        if (tid + 256 < vec_n) {
            row_ptr[tid + 256] = make_float4(regs[1].x * inv_sum, regs[1].y * inv_sum,
                                             regs[1].z * inv_sum, regs[1].w * inv_sum);
        }
    } else {
        for (int idx = tid; idx < vec_n; idx += 256) {
            float4 v = row_ptr[idx];
            row_ptr[idx] = make_float4(v.x * inv_sum, v.y * inv_sum, v.z * inv_sum, v.w * inv_sum);
        }
    }
}

// Stage 3: O[M, d] = P[M, N] @ V[N, d]
__global__ void __launch_bounds__(256, 2)
alibi_pv_128x128_kernel(const float* __restrict__ P,
                        const float* __restrict__ V,
                        float* __restrict__ O,
                        int M,
                        int N,
                        int d) {
    __shared__ float sA[2][8][132];
    __shared__ float sB[2][8][132];

    const int tid = threadIdx.x;
    const int ty = tid >> 4;
    const int tx = tid & 15;
    const int bm = blockIdx.y * 128;
    const int bn = blockIdx.x * 128;

    const int load_row_a = tid >> 1;         // 0..127
    const int load_col_a = (tid & 1) << 2;   // 0 or 4
    const int load_row_b = tid >> 5;         // 0..7
    const int load_col_b = (tid & 31) << 2;  // 0..124 step 4

    float c00[4][4] = {{0.0f}}, c01[4][4] = {{0.0f}};
    float c10[4][4] = {{0.0f}}, c11[4][4] = {{0.0f}};

    // Prefetch stage 0
    {
        float4 pv = *reinterpret_cast<const float4*>(&P[static_cast<size_t>(bm + load_row_a) * N + load_col_a]);
        float4 vv = *reinterpret_cast<const float4*>(&V[static_cast<size_t>(load_row_b) * d + bn + load_col_b]);
        sA[0][load_col_a + 0][load_row_a] = pv.x;
        sA[0][load_col_a + 1][load_row_a] = pv.y;
        sA[0][load_col_a + 2][load_row_a] = pv.z;
        sA[0][load_col_a + 3][load_row_a] = pv.w;
        *reinterpret_cast<float4*>(&sB[0][load_row_b][load_col_b]) = vv;
    }
    __syncthreads();

    int stage = 0;
    for (int bk = 0; bk < N; bk += 8) {
        float4 next_p, next_v;
        const bool has_next = (bk + 8 < N);
        if (has_next) {
            next_p = *reinterpret_cast<const float4*>(&P[static_cast<size_t>(bm + load_row_a) * N + bk + 8 + load_col_a]);
            next_v = *reinterpret_cast<const float4*>(&V[static_cast<size_t>(bk + 8 + load_row_b) * d + bn + load_col_b]);
        }

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float4 a0 = *reinterpret_cast<const float4*>(&sA[stage][k][ty * 4]);
            float4 a1 = *reinterpret_cast<const float4*>(&sA[stage][k][ty * 4 + 64]);
            float4 b0 = *reinterpret_cast<const float4*>(&sB[stage][k][tx * 4]);
            float4 b1 = *reinterpret_cast<const float4*>(&sB[stage][k][tx * 4 + 64]);
            fma4x4(a0, b0, c00);
            fma4x4(a0, b1, c01);
            fma4x4(a1, b0, c10);
            fma4x4(a1, b1, c11);
        }

        if (has_next) {
            stage ^= 1;
            sA[stage][load_col_a + 0][load_row_a] = next_p.x;
            sA[stage][load_col_a + 1][load_row_a] = next_p.y;
            sA[stage][load_col_a + 2][load_row_a] = next_p.z;
            sA[stage][load_col_a + 3][load_row_a] = next_p.w;
            *reinterpret_cast<float4*>(&sB[stage][load_row_b][load_col_b]) = next_v;
            __syncthreads();
        }
    }

    auto store_quad = [&](int r_off, int c_off, float c[4][4]) {
        const int col0 = bn + tx * 4 + c_off;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = bm + ty * 4 + r_off + i;
            *reinterpret_cast<float4*>(&O[static_cast<size_t>(row) * d + col0]) =
                make_float4(c[i][0], c[i][1], c[i][2], c[i][3]);
        }
    };
    store_quad(0, 0, c00);
    store_quad(0, 64, c01);
    store_quad(64, 0, c10);
    store_quad(64, 64, c11);
}

// ============================================================================
// Path 2: High-occupancy 8x16 (128-thread) float4 FlashAttention (d <= 256)
// ============================================================================

template <int kVecCols>
__global__ void __launch_bounds__(128, 2)
alibi_attention_8x16_kernel(const float* __restrict__ Q,
                            const float* __restrict__ K,
                            const float* __restrict__ V,
                            float* __restrict__ output,
                            int M,
                            int N,
                            int d,
                            float scale,
                            float alibi_alpha) {
    extern __shared__ float smem[];
    const int d_stride = d + 4;
    float* sQ = smem;
    float* sKV = sQ + kBr * d_stride;
    float* sP = sKV + kBc * d_stride;  // 8 * 36

    const int tid = threadIdx.x;
    const int ty = tid >> 4;  // 0..7  -> 1 Q row
    const int tx = tid & 15;  // 0..15 -> 2 K cols / float4 O chunks
    const int q_block_row = blockIdx.x * kBr;
    const int q_len = min(kBr, M - q_block_row);
    const int vec_d = d >> 2;

    for (int idx = tid; idx < q_len * vec_d; idx += kThreads) {
        const int r = idx / vec_d;
        const int c4 = idx % vec_d;
        float4 v = *reinterpret_cast<const float4*>(&Q[static_cast<size_t>(q_block_row + r) * d + c4 * 4]);
        *reinterpret_cast<float4*>(&sQ[r * d_stride + c4 * 4]) =
            make_float4(v.x * scale, v.y * scale, v.z * scale, v.w * scale);
    }
    __syncthreads();

    float4 o_reg[kVecCols];
    #pragma unroll
    for (int c = 0; c < kVecCols; ++c) {
        o_reg[c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float m = -FLT_MAX;
    float l = 0.0f;
    const int q_global_row = q_block_row + ty;
    const bool valid_q = (ty < q_len);

    for (int k_start = 0; k_start < N; k_start += kBc) {
        const int kv_len = min(kBc, N - k_start);

        for (int idx = tid; idx < kv_len * vec_d; idx += kThreads) {
            const int r = idx / vec_d;
            const int c4 = idx % vec_d;
            *reinterpret_cast<float4*>(&sKV[r * d_stride + c4 * 4]) =
                *reinterpret_cast<const float4*>(&K[static_cast<size_t>(k_start + r) * d + c4 * 4]);
        }
        __syncthreads();

        float s0 = 0.0f, s1 = 0.0f;
        const int k0 = tx * 2;
        const int k1 = k0 + 1;
        if (valid_q) {
            const float* q_row = &sQ[ty * d_stride];
            for (int k = 0; k < d; k += 4) {
                float4 qv = *reinterpret_cast<const float4*>(&q_row[k]);
                if (k0 < kv_len) {
                    float4 kv = *reinterpret_cast<const float4*>(&sKV[k0 * d_stride + k]);
                    s0 = fmaf(qv.x, kv.x, fmaf(qv.y, kv.y, fmaf(qv.z, kv.z, fmaf(qv.w, kv.w, s0))));
                }
                if (k1 < kv_len) {
                    float4 kv = *reinterpret_cast<const float4*>(&sKV[k1 * d_stride + k]);
                    s1 = fmaf(qv.x, kv.x, fmaf(qv.y, kv.y, fmaf(qv.z, kv.z, fmaf(qv.w, kv.w, s1))));
                }
            }
            if (k0 < kv_len) {
                s0 = fmaf(alibi_alpha, static_cast<float>(q_global_row - (k_start + k0)), s0);
            }
            if (k1 < kv_len) {
                s1 = fmaf(alibi_alpha, static_cast<float>(q_global_row - (k_start + k1)), s1);
            }
        }

        float rmax = -FLT_MAX;
        if (k0 < kv_len) rmax = fmaxf(rmax, s0);
        if (k1 < kv_len) rmax = fmaxf(rmax, s1);

        float tile_max = subwarp16_max(rmax);
        float m_new = fmaxf(m, tile_max);
        float rescale = __expf(m - m_new);

        float p0 = (k0 < kv_len) ? __expf(s0 - m_new) : 0.0f;
        float p1 = (k1 < kv_len) ? __expf(s1 - m_new) : 0.0f;

        *reinterpret_cast<float2*>(&sP[ty * 36 + k0]) = make_float2(p0, p1);
        float tile_sum = subwarp16_sum(p0 + p1);
        l = l * rescale + tile_sum;
        m = m_new;

        #pragma unroll
        for (int c = 0; c < kVecCols; ++c) {
            o_reg[c].x *= rescale;
            o_reg[c].y *= rescale;
            o_reg[c].z *= rescale;
            o_reg[c].w *= rescale;
        }
        __syncthreads();

        for (int idx = tid; idx < kv_len * vec_d; idx += kThreads) {
            const int r = idx / vec_d;
            const int c4 = idx % vec_d;
            *reinterpret_cast<float4*>(&sKV[r * d_stride + c4 * 4]) =
                *reinterpret_cast<const float4*>(&V[static_cast<size_t>(k_start + r) * d + c4 * 4]);
        }
        __syncthreads();

        if (valid_q) {
            const float* p_row = &sP[ty * 36];
            for (int j = 0; j < kv_len; ++j) {
                float p = p_row[j];
                const float* v_row = &sKV[j * d_stride];
                #pragma unroll
                for (int c = 0; c < kVecCols; ++c) {
                    const int col = tx * 4 + c * 64;
                    if (col < d) {
                        float4 vv = *reinterpret_cast<const float4*>(&v_row[col]);
                        o_reg[c].x = fmaf(p, vv.x, o_reg[c].x);
                        o_reg[c].y = fmaf(p, vv.y, o_reg[c].y);
                        o_reg[c].z = fmaf(p, vv.z, o_reg[c].z);
                        o_reg[c].w = fmaf(p, vv.w, o_reg[c].w);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (valid_q) {
        float* o_ptr = output + static_cast<size_t>(q_global_row) * d;
        const float inv_l = 1.0f / l;
        #pragma unroll
        for (int c = 0; c < kVecCols; ++c) {
            const int col = tx * 4 + c * 64;
            if (col < d) {
                *reinterpret_cast<float4*>(&o_ptr[col]) = make_float4(
                    o_reg[c].x * inv_l, o_reg[c].y * inv_l, o_reg[c].z * inv_l, o_reg[c].w * inv_l);
            }
        }
    }
}

// ============================================================================
// Path 3: General warp-level Online Softmax fallback (any M, N, d <= 1024)
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void alibi_attention_fallback_kernel(const float* __restrict__ Q,
                                                const float* __restrict__ K,
                                                const float* __restrict__ V,
                                                float* __restrict__ output,
                                                int M,
                                                int N,
                                                int d,
                                                float scale,
                                                float alibi_alpha) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int q_row = blockIdx.x * 4 + warp_id;
    if (q_row >= M) return;

    const float* q_ptr = Q + static_cast<size_t>(q_row) * d;
    float q_reg[kMaxElemsPerLane];
    float o_reg[kMaxElemsPerLane];
    const int elems = (d + 31) >> 5;
    for (int i = 0; i < elems; ++i) {
        const int col = lane + i * 32;
        q_reg[i] = (col < d) ? q_ptr[col] * scale : 0.0f;
        o_reg[i] = 0.0f;
    }

    float m = -FLT_MAX;
    float l = 0.0f;
    for (int k_row = 0; k_row < N; ++k_row) {
        const float* k_ptr = K + static_cast<size_t>(k_row) * d;
        float dot = 0.0f;
        for (int i = 0; i < elems; ++i) {
            const int col = lane + i * 32;
            if (col < d) dot = fmaf(q_reg[i], k_ptr[col], dot);
        }
        float score = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0);
        score = fmaf(alibi_alpha, static_cast<float>(q_row - k_row), score);

        float m_new = fmaxf(m, score);
        float rescale = __expf(m - m_new);
        float p = __expf(score - m_new);
        l = l * rescale + p;

        const float* v_ptr = V + static_cast<size_t>(k_row) * d;
        for (int i = 0; i < elems; ++i) {
            const int col = lane + i * 32;
            if (col < d) o_reg[i] = fmaf(p, v_ptr[col], o_reg[i] * rescale);
        }
        m = m_new;
    }

    float* o_ptr = output + static_cast<size_t>(q_row) * d;
    const float inv_l = 1.0f / l;
    for (int i = 0; i < elems; ++i) {
        const int col = lane + i * 32;
        if (col < d) o_ptr[col] = o_reg[i] * inv_l;
    }
}

static float* g_scores = nullptr;
static size_t g_scores_cap = 0;

}  // namespace

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d, float alpha) {
    const float scale = rsqrtf(static_cast<float>(d));
    const bool aligned16 = ((reinterpret_cast<uintptr_t>(Q) |
                             reinterpret_cast<uintptr_t>(K) |
                             reinterpret_cast<uintptr_t>(V) |
                             reinterpret_cast<uintptr_t>(output)) & 15ULL) == 0;

    if (aligned16 && (M & 127) == 0 && (N & 127) == 0 && (d & 127) == 0 && d > 256) {
        const size_t need = static_cast<size_t>(M) * static_cast<size_t>(N);
        if (need > g_scores_cap) {
            if (g_scores) cudaFree(g_scores);
            cudaMalloc(&g_scores, need * sizeof(float));
            g_scores_cap = need;
        }
        alibi_qkt_128x128_kernel<<<dim3(N / 128, M / 128), 256>>>(Q, K, g_scores, M, N, d, scale, alpha);
        alibi_row_softmax_kernel<<<M, 256>>>(g_scores, N);
        alibi_pv_128x128_kernel<<<dim3(d / 128, M / 128), 256>>>(g_scores, V, output, M, N, d);
        return;
    }

    if (aligned16 && (d & 3) == 0 && d <= 256) {
        dim3 block(kThreads);
        dim3 grid((M + kBr - 1) / kBr);
        const int d_stride = d + 4;
        const size_t smem_bytes = (static_cast<size_t>(kBr + kBc) * d_stride + kBr * 36ULL) * sizeof(float);
        if (d <= 64) {
            alibi_attention_8x16_kernel<1><<<grid, block, smem_bytes>>>(Q, K, V, output, M, N, d, scale, alpha);
        } else if (d <= 128) {
            alibi_attention_8x16_kernel<2><<<grid, block, smem_bytes>>>(Q, K, V, output, M, N, d, scale, alpha);
        } else {
            alibi_attention_8x16_kernel<4><<<grid, block, smem_bytes>>>(Q, K, V, output, M, N, d, scale, alpha);
        }
        return;
    }

    dim3 block(128);
    dim3 grid((M + 3) / 4);
    alibi_attention_fallback_kernel<<<grid, block>>>(Q, K, V, output, M, N, d, scale, alpha);
}
