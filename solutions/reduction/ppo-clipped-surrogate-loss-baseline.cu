#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

namespace {

constexpr int kWarpSize = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarpSize;
constexpr int kMaxBlocks = 512;
constexpr unsigned kFullMask = 0xffffffffu;

__device__ double g_base_block_sums[kMaxBlocks];

__device__ __forceinline__ double warp_reduce_sum_f64(double val) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(kFullMask, val, offset);
    }
    return val;
}

__global__ __launch_bounds__(kThreads) void ppo_loss_baseline_partial_kernel(
    const float* __restrict__ advantages,
    const float* __restrict__ log_pi,
    const float* __restrict__ log_pi_old,
    float lo_clip,
    float hi_clip,
    size_t total_elems) {
    __shared__ double s_warp_sums[kWarps];

    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid >> 5;
    const size_t global_tid = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(kThreads) +
                              static_cast<size_t>(tid);
    const size_t stride = static_cast<size_t>(gridDim.x) * static_cast<size_t>(kThreads);

    double local_sum = 0.0;
    for (size_t i = global_tid; i < total_elems; i += stride) {
        const float a = advantages[i];
        const float r = expf(log_pi[i] - log_pi_old[i]);
        const float r_hat = fminf(fmaxf(r, lo_clip), hi_clip);
        const float surrogate = fminf(r * a, r_hat * a);
        local_sum += static_cast<double>(surrogate);
    }

    local_sum = warp_reduce_sum_f64(local_sum);
    if (lane == 0) {
        s_warp_sums[warp] = local_sum;
    }
    __syncthreads();

    if (warp == 0) {
        double bsum = (lane < kWarps) ? s_warp_sums[lane] : 0.0;
        bsum = warp_reduce_sum_f64(bsum);
        if (lane == 0) {
            g_base_block_sums[blockIdx.x] = bsum;
        }
    }
}

__global__ __launch_bounds__(kThreads) void ppo_loss_baseline_reduce_kernel(
    float* __restrict__ output,
    int num_blocks,
    double neg_inv_n) {
    __shared__ double s_warp_sums[kWarps];

    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid >> 5;

    double sum = 0.0;
    for (int i = tid; i < num_blocks; i += kThreads) {
        sum += g_base_block_sums[i];
    }
    sum = warp_reduce_sum_f64(sum);
    if (lane == 0) {
        s_warp_sums[warp] = sum;
    }
    __syncthreads();

    if (warp == 0) {
        double total = (lane < kWarps) ? s_warp_sums[lane] : 0.0;
        total = warp_reduce_sum_f64(total);
        if (lane == 0) {
            output[0] = static_cast<float>(total * neg_inv_n);
        }
    }
}

}  // namespace

extern "C" void solve(const float* advantages, const float* log_pi, const float* log_pi_old,
                      float* output, float clip_eps, int B, int S) {
    if (B <= 0 || S <= 0) {
        return;
    }
    const size_t total = static_cast<size_t>(B) * static_cast<size_t>(S);
    int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    if (blocks > kMaxBlocks) {
        blocks = kMaxBlocks;
    }

    const float lo_clip = 1.0f - clip_eps;
    const float hi_clip = 1.0f + clip_eps;
    const double neg_inv_n = -1.0 / static_cast<double>(total);

    ppo_loss_baseline_partial_kernel<<<blocks, kThreads>>>(
        advantages, log_pi, log_pi_old, lo_clip, hi_clip, total);
    ppo_loss_baseline_reduce_kernel<<<1, kThreads>>>(output, blocks, neg_inv_n);
}
