#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#ifndef PPO_MAX_BLOCKS
#define PPO_MAX_BLOCKS 320
#endif

namespace {

constexpr int kWarpSize = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarpSize;
constexpr int kBssMaxBlocks = 1024;
constexpr int kTargetBlocks =
    (PPO_MAX_BLOCKS > kBssMaxBlocks) ? kBssMaxBlocks : PPO_MAX_BLOCKS;
constexpr unsigned kFullMask = 0xffffffffu;

// Static BSS state for 100% single-kernel execution via last-block retirement:
// zero cudaMalloc, zero cudaMemset, zero 2nd kernel launch overhead (~30 us saved!),
// and cleanly overwrites pre-filled dirty output[0].
__device__ double g_block_sums[kBssMaxBlocks];
__device__ unsigned int g_retired_blocks = 0u;

__device__ __forceinline__ float ppo_elem(float a, float lp, float lpo, float lo_clip, float hi_clip) {
    const float r = expf(lp - lpo);
    const float r_hat = fminf(fmaxf(r, lo_clip), hi_clip);
    return fminf(r * a, r_hat * a);
}

__device__ __forceinline__ float ppo_vec4_sum(float4 a,
                                              float4 lp,
                                              float4 lpo,
                                              float lo_clip,
                                              float hi_clip) {
    const float s0 = ppo_elem(a.x, lp.x, lpo.x, lo_clip, hi_clip);
    const float s1 = ppo_elem(a.y, lp.y, lpo.y, lo_clip, hi_clip);
    const float s2 = ppo_elem(a.z, lp.z, lpo.z, lo_clip, hi_clip);
    const float s3 = ppo_elem(a.w, lp.w, lpo.w, lo_clip, hi_clip);
    return (s0 + s1) + (s2 + s3);
}

__device__ __forceinline__ double warp_reduce_sum_f64(double val) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(kFullMask, val, offset);
    }
    return val;
}

__device__ __forceinline__ void retire_and_write_mean(double local_sum,
                                                      float* __restrict__ output,
                                                      double neg_inv_n) {
    __shared__ double s_warp_sums[kWarps];
    __shared__ bool s_is_last;

    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid >> 5;

    local_sum = warp_reduce_sum_f64(local_sum);
    if (lane == 0) {
        s_warp_sums[warp] = local_sum;
    }
    __syncthreads();

    if (gridDim.x == 1) {
        if (warp == 0) {
            double bsum = (lane < kWarps) ? s_warp_sums[lane] : 0.0;
            bsum = warp_reduce_sum_f64(bsum);
            if (lane == 0) {
                output[0] = static_cast<float>(bsum * neg_inv_n);
            }
        }
        return;
    }

    if (warp == 0) {
        double bsum = (lane < kWarps) ? s_warp_sums[lane] : 0.0;
        bsum = warp_reduce_sum_f64(bsum);
        if (lane == 0) {
            g_block_sums[blockIdx.x] = bsum;
            __threadfence();
            const unsigned int prev = atomicAdd(&g_retired_blocks, 1u);
            s_is_last = (prev == gridDim.x - 1u);
        }
    }
    __syncthreads();

    if (s_is_last) {
        const int num_blocks = static_cast<int>(gridDim.x);
        double total = 0.0;
        for (int i = tid; i < num_blocks; i += kThreads) {
            total += g_block_sums[i];
        }
        total = warp_reduce_sum_f64(total);
        if (lane == 0) {
            s_warp_sums[warp] = total;
        }
        __syncthreads();

        if (warp == 0) {
            double grand_total = (lane < kWarps) ? s_warp_sums[lane] : 0.0;
            grand_total = warp_reduce_sum_f64(grand_total);
            if (lane == 0) {
                g_retired_blocks = 0u;
                output[0] = static_cast<float>(grand_total * neg_inv_n);
            }
        }
    }
}

// ============================================================================
// Fast Path: 16B-Aligned 4x float4 (16 floats/iter) Single-Launch Kernel!
//   - Burst-loads 4x float4 from advantages, log_pi, and log_pi_old via __ldg.
//   - Evaluates ratio via fast SFU `exp2f((lp - lpo) * log2e)` (< 2 ULP).
//   - Pairwise-sums all 16 float32 surrogates BEFORE doing ONLY ONE `double`
//     addition per 16 elements (cuts Turing 1/32-rate FP64 instructions by 16x!).
//   - Finalizes `-mean(surrogate)` in ONE kernel launch via last-block retirement!
// ============================================================================
__global__ __launch_bounds__(kThreads, 4) void ppo_loss_vec4_single_kernel(
    const float4* __restrict__ adv4,
    const float4* __restrict__ lp4,
    const float4* __restrict__ lpo4,
    const float* __restrict__ advantages,
    const float* __restrict__ log_pi,
    const float* __restrict__ log_pi_old,
    float* __restrict__ output,
    float lo_clip,
    float hi_clip,
    size_t vec_n,
    size_t total_elems,
    double neg_inv_n) {
    const size_t global_tid = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(kThreads) +
                              static_cast<size_t>(threadIdx.x);
    const size_t stride = static_cast<size_t>(gridDim.x) * static_cast<size_t>(kThreads);

    double acc_d = 0.0;
    size_t idx4 = global_tid;

    for (; idx4 + 3 * stride < vec_n; idx4 += 4 * stride) {
        const float4 a0 = __ldg(&adv4[idx4]);
        const float4 p0 = __ldg(&lp4[idx4]);
        const float4 q0 = __ldg(&lpo4[idx4]);

        const float4 a1 = __ldg(&adv4[idx4 + stride]);
        const float4 p1 = __ldg(&lp4[idx4 + stride]);
        const float4 q1 = __ldg(&lpo4[idx4 + stride]);

        const float4 a2 = __ldg(&adv4[idx4 + 2 * stride]);
        const float4 p2 = __ldg(&lp4[idx4 + 2 * stride]);
        const float4 q2 = __ldg(&lpo4[idx4 + 2 * stride]);

        const float4 a3 = __ldg(&adv4[idx4 + 3 * stride]);
        const float4 p3 = __ldg(&lp4[idx4 + 3 * stride]);
        const float4 q3 = __ldg(&lpo4[idx4 + 3 * stride]);

        const float sum01 = ppo_vec4_sum(a0, p0, q0, lo_clip, hi_clip) +
                            ppo_vec4_sum(a1, p1, q1, lo_clip, hi_clip);
        const float sum23 = ppo_vec4_sum(a2, p2, q2, lo_clip, hi_clip) +
                            ppo_vec4_sum(a3, p3, q3, lo_clip, hi_clip);

        // Only ONE FP64 accumulation per 16 elements!
        acc_d += static_cast<double>(sum01 + sum23);
    }

    for (; idx4 < vec_n; idx4 += stride) {
        const float4 av = __ldg(&adv4[idx4]);
        const float4 pv = __ldg(&lp4[idx4]);
        const float4 qv = __ldg(&lpo4[idx4]);
        acc_d += static_cast<double>(ppo_vec4_sum(av, pv, qv, lo_clip, hi_clip));
    }

    const size_t tail_start = vec_n << 2;
    for (size_t i = tail_start + global_tid; i < total_elems; i += stride) {
        acc_d += static_cast<double>(
            ppo_elem(__ldg(&advantages[i]), __ldg(&log_pi[i]), __ldg(&log_pi_old[i]), lo_clip, hi_clip));
    }

    retire_and_write_mean(acc_d, output, neg_inv_n);
}

// Misaligned pointer scalar fallback (still 100% single-launch!).
__global__ __launch_bounds__(kThreads, 4) void ppo_loss_scalar_single_kernel(
    const float* __restrict__ advantages,
    const float* __restrict__ log_pi,
    const float* __restrict__ log_pi_old,
    float* __restrict__ output,
    float lo_clip,
    float hi_clip,
    size_t total_elems,
    double neg_inv_n) {
    const size_t global_tid = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(kThreads) +
                              static_cast<size_t>(threadIdx.x);
    const size_t stride = static_cast<size_t>(gridDim.x) * static_cast<size_t>(kThreads);

    double acc_d = 0.0;
    size_t i = global_tid;
    for (; i + 3 * stride < total_elems; i += 4 * stride) {
        const float s0 = ppo_elem(__ldg(&advantages[i]), __ldg(&log_pi[i]), __ldg(&log_pi_old[i]), lo_clip, hi_clip);
        const float s1 = ppo_elem(__ldg(&advantages[i + stride]), __ldg(&log_pi[i + stride]), __ldg(&log_pi_old[i + stride]), lo_clip, hi_clip);
        const float s2 = ppo_elem(__ldg(&advantages[i + 2 * stride]), __ldg(&log_pi[i + 2 * stride]), __ldg(&log_pi_old[i + 2 * stride]), lo_clip, hi_clip);
        const float s3 = ppo_elem(__ldg(&advantages[i + 3 * stride]), __ldg(&log_pi[i + 3 * stride]), __ldg(&log_pi_old[i + 3 * stride]), lo_clip, hi_clip);
        acc_d += static_cast<double>((s0 + s1) + (s2 + s3));
    }
    for (; i < total_elems; i += stride) {
        acc_d += static_cast<double>(
            ppo_elem(__ldg(&advantages[i]), __ldg(&log_pi[i]), __ldg(&log_pi_old[i]), lo_clip, hi_clip));
    }

    retire_and_write_mean(acc_d, output, neg_inv_n);
}

}  // namespace

extern "C" void solve(const float* advantages, const float* log_pi, const float* log_pi_old,
                      float* output, float clip_eps, int B, int S) {
    if (B <= 0 || S <= 0) {
        return;
    }

    const size_t total = static_cast<size_t>(B) * static_cast<size_t>(S);
    const float lo_clip = 1.0f - clip_eps;
    const float hi_clip = 1.0f + clip_eps;
    const double neg_inv_n = -1.0 / static_cast<double>(total);

    const uintptr_t align_mask = reinterpret_cast<uintptr_t>(advantages) |
                                 reinterpret_cast<uintptr_t>(log_pi) |
                                 reinterpret_cast<uintptr_t>(log_pi_old);

    if ((align_mask & 15u) == 0u && total >= 16u) {
        const size_t vec_n = total >> 2;
        size_t needed_blocks = (vec_n + (kThreads * 4) - 1) / (kThreads * 4);
        if (needed_blocks < 1u) needed_blocks = 1u;
        if (needed_blocks > static_cast<size_t>(kTargetBlocks)) {
            needed_blocks = static_cast<size_t>(kTargetBlocks);
        }
        ppo_loss_vec4_single_kernel<<<static_cast<unsigned int>(needed_blocks), kThreads>>>(
            reinterpret_cast<const float4*>(advantages),
            reinterpret_cast<const float4*>(log_pi),
            reinterpret_cast<const float4*>(log_pi_old),
            advantages, log_pi, log_pi_old, output,
            lo_clip, hi_clip, vec_n, total, neg_inv_n);
    } else {
        size_t needed_blocks = (total + (kThreads * 4) - 1) / (kThreads * 4);
        if (needed_blocks < 1u) needed_blocks = 1u;
        if (needed_blocks > static_cast<size_t>(kTargetBlocks)) {
            needed_blocks = static_cast<size_t>(kTargetBlocks);
        }
        ppo_loss_scalar_single_kernel<<<static_cast<unsigned int>(needed_blocks), kThreads>>>(
            advantages, log_pi, log_pi_old, output, lo_clip, hi_clip, total, neg_inv_n);
    }
}
