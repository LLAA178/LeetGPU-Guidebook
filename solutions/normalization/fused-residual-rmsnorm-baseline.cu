#include <cuda_runtime.h>

#include <cstddef>

namespace {

constexpr int kWarpSize = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarpSize;
constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(kFullMask, val, offset);
    }
    return val;
}

__global__ __launch_bounds__(kThreads) void fused_residual_rmsnorm_baseline_kernel(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int C,
    float inv_c,
    float eps) {
    __shared__ float s_warp_sums[kWarps];
    __shared__ float s_inv_rms;

    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid >> 5;
    const size_t row_offset = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(C);

    const float* __restrict__ row_x = x + row_offset;
    const float* __restrict__ row_res = residual + row_offset;
    float* __restrict__ row_out = out + row_offset;

    float sum_sq = 0.0f;
    for (int j = tid; j < C; j += kThreads) {
        const float z = row_x[j] + row_res[j];
        sum_sq = fmaf(z, z, sum_sq);
    }

    sum_sq = warp_reduce_sum(sum_sq);
    if (lane == 0) {
        s_warp_sums[warp] = sum_sq;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum = (lane < kWarps) ? s_warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            s_inv_rms = rsqrtf(block_sum * inv_c + eps);
        }
    }
    __syncthreads();

    const float inv_rms = s_inv_rms;
    for (int j = tid; j < C; j += kThreads) {
        const float z = row_x[j] + row_res[j];
        row_out[j] = z * inv_rms * weight[j];
    }
}

}  // namespace

// x, residual, weight, out are device pointers
extern "C" void solve(const float* x,
                      const float* residual,
                      const float* weight,
                      float* out,
                      int N,
                      int C,
                      float eps) {
    if (N <= 0 || C <= 0) {
        return;
    }
    const float inv_c = 1.0f / static_cast<float>(C);
    fused_residual_rmsnorm_baseline_kernel<<<N, kThreads>>>(
        x, residual, weight, out, C, inv_c, eps);
}
