#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace {

#ifndef GAE_THREADS
#define GAE_THREADS 512
#endif

#ifndef GAE_VECS_PER_THREAD
#define GAE_VECS_PER_THREAD 2
#endif

static_assert(GAE_THREADS == 256 || GAE_THREADS == 512 || GAE_THREADS == 1024,
              "GAE_THREADS must be 256, 512, or 1024");
static_assert(GAE_VECS_PER_THREAD >= 1 && GAE_VECS_PER_THREAD <= 8,
              "GAE_VECS_PER_THREAD must be in [1, 8]");

constexpr int kWarp = 32;
constexpr unsigned kFullMask = 0xffffffffu;

struct DecayPowers {
    float d1;
    float d2;
    float d3;
    float d4;
    float d8;
    float d16;
    float d32;
    float d64;
    float d128;
    float d256;
    float d512;
    float d1024;
    float d2048;
    float d4096;
};

__host__ inline DecayPowers make_decay_powers(float gamma, float lam) {
    const double g = static_cast<double>(gamma);
    const double l = static_cast<double>(lam);
    const double c1 = g * l;
    const double c2 = c1 * c1;
    const double c3 = c2 * c1;
    const double c4 = c2 * c2;
    const double c8 = c4 * c4;
    const double c16 = c8 * c8;
    const double c32 = c16 * c16;
    const double c64 = c32 * c32;
    const double c128 = c64 * c64;
    const double c256 = c128 * c128;
    const double c512 = c256 * c256;
    const double c1024 = c512 * c512;
    const double c2048 = c1024 * c1024;
    const double c4096 = c2048 * c2048;

    DecayPowers p;
    p.d1 = static_cast<float>(c1);
    p.d2 = static_cast<float>(c2);
    p.d3 = static_cast<float>(c3);
    p.d4 = static_cast<float>(c4);
    p.d8 = static_cast<float>(c8);
    p.d16 = static_cast<float>(c16);
    p.d32 = static_cast<float>(c32);
    p.d64 = static_cast<float>(c64);
    p.d128 = static_cast<float>(c128);
    p.d256 = static_cast<float>(c256);
    p.d512 = static_cast<float>(c512);
    p.d1024 = static_cast<float>(c1024);
    p.d2048 = static_cast<float>(c2048);
    p.d4096 = static_cast<float>(c4096);
    return p;
}

__device__ __forceinline__ float decay_pow4_table(int exp4, const DecayPowers& p) {
    float g = 1.0f;
    if (exp4 & 1) g *= p.d4;
    if (exp4 & 2) g *= p.d8;
    if (exp4 & 4) g *= p.d16;
    if (exp4 & 8) g *= p.d32;
    if (exp4 & 16) g *= p.d64;
    return g;
}

__device__ __forceinline__ float decay_pow128_table(int exp128, const DecayPowers& p) {
    float g = 1.0f;
    if (exp128 & 1) g *= p.d128;
    if (exp128 & 2) g *= p.d256;
    if (exp128 & 4) g *= p.d512;
    if (exp128 & 8) g *= p.d1024;
    if (exp128 & 16) g *= p.d2048;
    return g;
}

__device__ __forceinline__ float warp_reverse_inclusive_fixed4(
    float val,
    int lane,
    const DecayPowers& p) {
    float nbr = __shfl_down_sync(kFullMask, val, 1);
    if (lane <= 30) val = fmaf(p.d4, nbr, val);
    nbr = __shfl_down_sync(kFullMask, val, 2);
    if (lane <= 29) val = fmaf(p.d8, nbr, val);
    nbr = __shfl_down_sync(kFullMask, val, 4);
    if (lane <= 27) val = fmaf(p.d16, nbr, val);
    nbr = __shfl_down_sync(kFullMask, val, 8);
    if (lane <= 23) val = fmaf(p.d32, nbr, val);
    nbr = __shfl_down_sync(kFullMask, val, 16);
    if (lane <= 15) val = fmaf(p.d64, nbr, val);
    return val;
}

template <int kWarps>
__device__ __forceinline__ float warp_reverse_inclusive_warps_fixed128(
    float val,
    int lane,
    const DecayPowers& p) {
    float nbr = __shfl_down_sync(kFullMask, val, 1);
    if (lane + 1 < kWarps) val = fmaf(p.d128, nbr, val);
    nbr = __shfl_down_sync(kFullMask, val, 2);
    if (lane + 2 < kWarps) val = fmaf(p.d256, nbr, val);
    nbr = __shfl_down_sync(kFullMask, val, 4);
    if (lane + 4 < kWarps) val = fmaf(p.d512, nbr, val);
    if (kWarps > 8) {
        nbr = __shfl_down_sync(kFullMask, val, 8);
        if (lane + 8 < kWarps) val = fmaf(p.d1024, nbr, val);
    }
    if (kWarps > 16) {
        nbr = __shfl_down_sync(kFullMask, val, 16);
        if (lane + 16 < kWarps) val = fmaf(p.d2048, nbr, val);
    }
    return val;
}

__device__ __forceinline__ void warp_reverse_inclusive_op(
    float& val,
    float& gain,
    int lane) {
    #pragma unroll
    for (int offset = 1; offset < kWarp; offset <<= 1) {
        const float nbr_val = __shfl_down_sync(kFullMask, val, offset);
        const float nbr_gain = __shfl_down_sync(kFullMask, gain, offset);
        if (lane + offset < kWarp) {
            val = fmaf(gain, nbr_val, val);
            gain *= nbr_gain;
        }
    }
}

// Specialized zero-loop fast path when 16-byte aligned and S == 4 * kThreads * kVecsPerThread
// (including S == 4096 with (1024, 1), (512, 2), or (256, 4)).
template <int kThreads, int kVecsPerThread>
__global__ __launch_bounds__(kThreads, 1) void gae_reverse_scan_exact_vec4_kernel(
    const float4* __restrict__ rewards4,
    const float4* __restrict__ values4,
    float4* __restrict__ advantages4,
    float gamma,
    DecayPowers p) {
    constexpr int kWarps = kThreads / kWarp;
    constexpr int kRowVecs = kThreads * kVecsPerThread;

    __shared__ float s_first_v[kWarps];
    __shared__ float s_warp_sum[kWarps];

    const int tid = threadIdx.x;
    const int lane = tid & (kWarp - 1);
    const int warp = tid >> 5;
    const int row_vec_offset = blockIdx.x * kRowVecs;

    float row_carry = 0.0f;
    float boundary_v = 0.0f;

    #pragma unroll
    for (int sub = kVecsPerThread - 1; sub >= 0; --sub) {
        const int vec_idx = row_vec_offset + sub * kThreads + tid;
        const float4 r4 = __ldg(&rewards4[vec_idx]);
        const float4 v4 = __ldg(&values4[vec_idx]);

        if (lane == 0) {
            s_first_v[warp] = v4.x;
        }
        __syncthreads();

        const float shfl_next_v = __shfl_down_sync(kFullMask, v4.x, 1);
        float v_after = shfl_next_v;
        if (lane == kWarp - 1) {
            v_after = (warp + 1 < kWarps) ? s_first_v[warp + 1] : boundary_v;
        }

        const float d0 = fmaf(gamma, v4.y, r4.x - v4.x);
        const float d1 = fmaf(gamma, v4.z, r4.y - v4.y);
        const float d2 = fmaf(gamma, v4.w, r4.z - v4.z);
        const float d3 = fmaf(gamma, v_after, r4.w - v4.w);

        const float g3 = d3;
        const float g2 = fmaf(p.d1, g3, d2);
        const float g1 = fmaf(p.d1, g2, d1);
        const float g0 = fmaf(p.d1, g1, d0);

        const float w_incl = warp_reverse_inclusive_fixed4(g0, lane, p);
        const float nbr_incl = __shfl_down_sync(kFullMask, w_incl, 1);
        if (lane == 0) {
            s_warp_sum[warp] = w_incl;
        }
        __syncthreads();

        if (warp == 0) {
            float w_val = (lane < kWarps) ? s_warp_sum[lane] : 0.0f;
            w_val = warp_reverse_inclusive_warps_fixed128<kWarps>(w_val, lane, p);
            const float next_w = __shfl_down_sync(kFullMask, w_val, 1);
            if (lane < kWarps) {
                const float excl = (lane + 1 < kWarps) ? next_w : 0.0f;
                const float warp_rem_gain = decay_pow128_table(kWarps - 1 - lane, p);
                s_warp_sum[lane] = (lane + 1 == kWarps)
                                       ? row_carry
                                       : fmaf(warp_rem_gain, row_carry, excl);
            }
            if (lane == 0 && kVecsPerThread > 1) {
                float subtile_gain = p.d4096;
                if (kWarps == 8) subtile_gain = p.d1024;
                if (kWarps == 16) subtile_gain = p.d2048;
                s_first_v[0] = fmaf(subtile_gain, row_carry, w_val);
            }
        }
        __syncthreads();

        const float warp_right_carry = s_warp_sum[warp];
        float c_in = warp_right_carry;
        if (lane < kWarp - 1) {
            const float rem_warp_gain = decay_pow4_table(kWarp - 1 - lane, p);
            c_in = fmaf(rem_warp_gain, warp_right_carry, nbr_incl);
        }

        float4 out4;
        out4.w = fmaf(p.d1, c_in, g3);
        out4.z = fmaf(p.d2, c_in, g2);
        out4.y = fmaf(p.d3, c_in, g1);
        out4.x = fmaf(p.d4, c_in, g0);
        advantages4[vec_idx] = out4;

        if (kVecsPerThread > 1) {
            row_carry = s_first_v[0];
            if (tid == 0) {
                s_warp_sum[0] = v4.x;
            }
            __syncthreads();
            boundary_v = s_warp_sum[0];
            __syncthreads();
        }
    }
}

// Fast multi-tile aligned float4 path when S is any positive multiple of 4 and pointers are 16B aligned.
template <int kThreads>
__global__ __launch_bounds__(kThreads, 1) void gae_reverse_scan_aligned_vec4_kernel(
    const float4* __restrict__ rewards4,
    const float4* __restrict__ values4,
    float4* __restrict__ advantages4,
    float gamma,
    DecayPowers p,
    int S4) {
    constexpr int kWarps = kThreads / kWarp;

    __shared__ float s_first_v[kWarps];
    __shared__ float s_warp_sum[kWarps];
    __shared__ float s_warp_gain[kWarps];
    __shared__ float s_next_tile_v;
    __shared__ float s_next_tile_carry;

    const int tid = threadIdx.x;
    const int lane = tid & (kWarp - 1);
    const int warp = tid >> 5;
    const int row_vec_offset = blockIdx.x * S4;

    float boundary_v = 0.0f;
    float row_carry = 0.0f;

    // Tile from right to left so the leftmost tile may be partial, or tile right-aligned:
    // Right-aligned tiling makes interior tiles full and puts any partial tile at index 0..rem-1,
    // or left-aligned puts the partial tile at the rightmost iteration (first iteration!).
    // Left-aligned: last_tile_start = ((S4 - 1) / kThreads) * kThreads.
    // Then ONLY the very first iteration can be partial, and every earlier tile is 100% full!
    const int last_tile_base = ((S4 - 1) / kThreads) * kThreads;
    for (int tile_base = last_tile_base; tile_base >= 0; tile_base -= kThreads) {
        const int vec_in_row = tile_base + tid;
        const bool valid = (vec_in_row < S4);

        float4 r4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 v4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (valid) {
            r4 = __ldg(&rewards4[row_vec_offset + vec_in_row]);
            v4 = __ldg(&values4[row_vec_offset + vec_in_row]);
        }

        if (lane == 0) {
            s_first_v[warp] = valid ? v4.x : boundary_v;
        }
        __syncthreads();

        const float shfl_next_v = __shfl_down_sync(kFullMask, v4.x, 1);
        float v_after = 0.0f;
        if (valid) {
            if (vec_in_row + 1 == S4) {
                v_after = 0.0f;
            } else if (tid + 1 == kThreads) {
                v_after = boundary_v;
            } else if (lane < kWarp - 1) {
                v_after = shfl_next_v;
            } else {
                v_after = s_first_v[warp + 1];
            }
        }

        float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
        float gain = 1.0f;
        if (valid) {
            const float d0 = fmaf(gamma, v4.y, r4.x - v4.x);
            const float d1 = fmaf(gamma, v4.z, r4.y - v4.y);
            const float d2 = fmaf(gamma, v4.w, r4.z - v4.z);
            const float d3 = fmaf(gamma, v_after, r4.w - v4.w);

            g3 = d3;
            g2 = fmaf(p.d1, g3, d2);
            g1 = fmaf(p.d1, g2, d1);
            g0 = fmaf(p.d1, g1, d0);
            gain = p.d4;
        }

        float w_val = g0;
        float w_gain = gain;
        warp_reverse_inclusive_op(w_val, w_gain, lane);

        const float nbr_val = __shfl_down_sync(kFullMask, w_val, 1);
        const float nbr_gain = __shfl_down_sync(kFullMask, w_gain, 1);

        if (lane == 0) {
            s_warp_sum[warp] = w_val;
            s_warp_gain[warp] = w_gain;
        }
        if (tid == 0) {
            s_next_tile_v = v4.x;
        }
        __syncthreads();

        if (warp == 0) {
            float b_val = (lane < kWarps) ? s_warp_sum[lane] : 0.0f;
            float b_gain = (lane < kWarps) ? s_warp_gain[lane] : 1.0f;
            warp_reverse_inclusive_op(b_val, b_gain, lane);

            const float next_b_val = __shfl_down_sync(kFullMask, b_val, 1);
            const float next_b_gain = __shfl_down_sync(kFullMask, b_gain, 1);
            if (lane < kWarps) {
                const float excl_val = (lane + 1 < kWarps) ? next_b_val : 0.0f;
                const float excl_gain = (lane + 1 < kWarps) ? next_b_gain : 1.0f;
                s_warp_sum[lane] = fmaf(excl_gain, row_carry, excl_val);
            }
            if (lane == 0) {
                s_next_tile_carry = fmaf(b_gain, row_carry, b_val);
            }
        }
        __syncthreads();

        if (valid) {
            const float warp_carry = s_warp_sum[warp];
            const float c_in = (lane < kWarp - 1)
                                   ? fmaf(nbr_gain, warp_carry, nbr_val)
                                   : warp_carry;

            float4 out4;
            out4.w = fmaf(p.d1, c_in, g3);
            out4.z = fmaf(p.d2, c_in, g2);
            out4.y = fmaf(p.d3, c_in, g1);
            out4.x = fmaf(p.d4, c_in, g0);
            advantages4[row_vec_offset + vec_in_row] = out4;
        }

        boundary_v = s_next_tile_v;
        row_carry = s_next_tile_carry;
        __syncthreads();
    }
}

// General fallback kernel for arbitrary S >= 1 (not divisible by 4) or misaligned base pointers.
// Handles every B >= 1, S >= 1 in a single launch with zero workspace.
template <int kThreads>
__global__ __launch_bounds__(kThreads, 1) void gae_reverse_scan_general_kernel(
    const float* __restrict__ rewards,
    const float* __restrict__ values,
    float* __restrict__ advantages,
    float gamma,
    DecayPowers p,
    int S) {
    constexpr int kWarps = kThreads / kWarp;
    constexpr int kTileElems = 4 * kThreads;

    __shared__ float s_first_v[kWarps];
    __shared__ float s_warp_sum[kWarps];
    __shared__ float s_warp_gain[kWarps];
    __shared__ float s_next_tile_v;
    __shared__ float s_next_tile_carry;

    const int tid = threadIdx.x;
    const int lane = tid & (kWarp - 1);
    const int warp = tid >> 5;

    const size_t row_offset = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(S);
    const float* __restrict__ row_r = rewards + row_offset;
    const float* __restrict__ row_v = values + row_offset;
    float* __restrict__ row_adv = advantages + row_offset;

    float boundary_v = 0.0f;
    float row_carry = 0.0f;

    const int last_tile_start = ((S - 1) / kTileElems) * kTileElems;
    for (int tile_start = last_tile_start; tile_start >= 0; tile_start -= kTileElems) {
        const int idx0 = tile_start + 4 * tid;
        const int rem = S - idx0;

        float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;

        if (rem >= 4) {
            r0 = __ldg(&row_r[idx0]);
            r1 = __ldg(&row_r[idx0 + 1]);
            r2 = __ldg(&row_r[idx0 + 2]);
            r3 = __ldg(&row_r[idx0 + 3]);
            v0 = __ldg(&row_v[idx0]);
            v1 = __ldg(&row_v[idx0 + 1]);
            v2 = __ldg(&row_v[idx0 + 2]);
            v3 = __ldg(&row_v[idx0 + 3]);
        } else if (rem > 0) {
            r0 = __ldg(&row_r[idx0]);
            v0 = __ldg(&row_v[idx0]);
            if (rem >= 2) {
                r1 = __ldg(&row_r[idx0 + 1]);
                v1 = __ldg(&row_v[idx0 + 1]);
            }
            if (rem >= 3) {
                r2 = __ldg(&row_r[idx0 + 2]);
                v2 = __ldg(&row_v[idx0 + 2]);
            }
        }

        if (lane == 0) {
            s_first_v[warp] = (rem > 0) ? v0 : boundary_v;
        }
        __syncthreads();

        const float shfl_next_v = __shfl_down_sync(kFullMask, v0, 1);
        float v_after = 0.0f;
        if (rem >= 4) {
            if (idx0 + 4 == S) {
                v_after = 0.0f;
            } else if (tid + 1 == kThreads) {
                v_after = boundary_v;
            } else if (lane < kWarp - 1) {
                v_after = shfl_next_v;
            } else {
                v_after = s_first_v[warp + 1];
            }
        }

        float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
        float gain = 1.0f;
        if (rem >= 4) {
            const float d0 = fmaf(gamma, v1, r0 - v0);
            const float d1 = fmaf(gamma, v2, r1 - v1);
            const float d2 = fmaf(gamma, v3, r2 - v2);
            const float d3 = fmaf(gamma, v_after, r3 - v3);
            g3 = d3;
            g2 = fmaf(p.d1, g3, d2);
            g1 = fmaf(p.d1, g2, d1);
            g0 = fmaf(p.d1, g1, d0);
            gain = p.d4;
        } else if (rem == 3) {
            const float d0 = fmaf(gamma, v1, r0 - v0);
            const float d1 = fmaf(gamma, v2, r1 - v1);
            const float d2 = r2 - v2;
            g2 = d2;
            g1 = fmaf(p.d1, g2, d1);
            g0 = fmaf(p.d1, g1, d0);
            gain = p.d3;
        } else if (rem == 2) {
            const float d0 = fmaf(gamma, v1, r0 - v0);
            const float d1 = r1 - v1;
            g1 = d1;
            g0 = fmaf(p.d1, g1, d0);
            gain = p.d2;
        } else if (rem == 1) {
            g0 = r0 - v0;
            gain = p.d1;
        }

        float w_val = g0;
        float w_gain = gain;
        warp_reverse_inclusive_op(w_val, w_gain, lane);

        const float nbr_val = __shfl_down_sync(kFullMask, w_val, 1);
        const float nbr_gain = __shfl_down_sync(kFullMask, w_gain, 1);

        if (lane == 0) {
            s_warp_sum[warp] = w_val;
            s_warp_gain[warp] = w_gain;
        }
        if (tid == 0) {
            s_next_tile_v = v0;
        }
        __syncthreads();

        if (warp == 0) {
            float b_val = (lane < kWarps) ? s_warp_sum[lane] : 0.0f;
            float b_gain = (lane < kWarps) ? s_warp_gain[lane] : 1.0f;
            warp_reverse_inclusive_op(b_val, b_gain, lane);

            const float next_b_val = __shfl_down_sync(kFullMask, b_val, 1);
            const float next_b_gain = __shfl_down_sync(kFullMask, b_gain, 1);
            if (lane < kWarps) {
                const float excl_val = (lane + 1 < kWarps) ? next_b_val : 0.0f;
                const float excl_gain = (lane + 1 < kWarps) ? next_b_gain : 1.0f;
                s_warp_sum[lane] = fmaf(excl_gain, row_carry, excl_val);
            }
            if (lane == 0) {
                s_next_tile_carry = fmaf(b_gain, row_carry, b_val);
            }
        }
        __syncthreads();

        if (rem >= 4) {
            const float warp_carry = s_warp_sum[warp];
            const float c_in = (lane < kWarp - 1)
                                   ? fmaf(nbr_gain, warp_carry, nbr_val)
                                   : warp_carry;
            row_adv[idx0 + 3] = fmaf(p.d1, c_in, g3);
            row_adv[idx0 + 2] = fmaf(p.d2, c_in, g2);
            row_adv[idx0 + 1] = fmaf(p.d3, c_in, g1);
            row_adv[idx0] = fmaf(p.d4, c_in, g0);
        } else if (rem == 3) {
            row_adv[idx0 + 2] = g2;
            row_adv[idx0 + 1] = g1;
            row_adv[idx0] = g0;
        } else if (rem == 2) {
            row_adv[idx0 + 1] = g1;
            row_adv[idx0] = g0;
        } else if (rem == 1) {
            row_adv[idx0] = g0;
        }

        boundary_v = s_next_tile_v;
        row_carry = s_next_tile_carry;
        __syncthreads();
    }
}

inline bool is_aligned_16(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 15u) == 0u;
}

}  // namespace

// rewards, values, advantages are device pointers
extern "C" void solve(const float* rewards, const float* values, float* advantages, float gamma,
                      float lam, int B, int S) {
    if (B <= 0 || S <= 0) {
        return;
    }

    const DecayPowers p = make_decay_powers(gamma, lam);
    const bool aligned_vec4 = ((S & 3) == 0) &&
                              is_aligned_16(rewards) &&
                              is_aligned_16(values) &&
                              is_aligned_16(advantages);

    if (aligned_vec4) {
        const float4* r4 = reinterpret_cast<const float4*>(rewards);
        const float4* v4 = reinterpret_cast<const float4*>(values);
        float4* a4 = reinterpret_cast<float4*>(advantages);

        constexpr int kExactS = 4 * GAE_THREADS * GAE_VECS_PER_THREAD;
        if (S == kExactS) {
            gae_reverse_scan_exact_vec4_kernel<GAE_THREADS, GAE_VECS_PER_THREAD>
                <<<B, GAE_THREADS>>>(r4, v4, a4, gamma, p);
            return;
        }

        if (S <= 1024) {
            gae_reverse_scan_aligned_vec4_kernel<256><<<B, 256>>>(r4, v4, a4, gamma, p, S >> 2);
            return;
        }
        gae_reverse_scan_aligned_vec4_kernel<GAE_THREADS>
            <<<B, GAE_THREADS>>>(r4, v4, a4, gamma, p, S >> 2);
        return;
    }

    if (S <= 1024) {
        gae_reverse_scan_general_kernel<256>
            <<<B, 256>>>(rewards, values, advantages, gamma, p, S);
    } else {
        gae_reverse_scan_general_kernel<GAE_THREADS>
            <<<B, GAE_THREADS>>>(rewards, values, advantages, gamma, p, S);
    }
}
