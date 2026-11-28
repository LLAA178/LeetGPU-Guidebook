#include <cuda_runtime.h>
#include <cmath>

namespace {

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarp;
constexpr int kMaxBlocks = 1024;
constexpr unsigned kFullMask = 0xffffffffu;

__device__ double g_partial_sq[kMaxBlocks];
__device__ float g_inv_rms_scale;

__device__ __forceinline__ double block_reduce_sum_f64(double val, double* s_warp) {
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x / kWarp;
    #pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(kFullMask, val, offset);
    }
    if (lane == 0) s_warp[warp] = val;
    __syncthreads();

    double out = 0.0;
    if (warp == 0) {
        out = (lane < kWarps) ? s_warp[lane] : 0.0;
        #pragma unroll
        for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
            out += __shfl_down_sync(kFullMask, out, offset);
        }
    }
    return out;
}

__global__ __launch_bounds__(kThreads) void rmsnorm_sumsq_pass1_kernel(
    const float* __restrict__ input,
    int n) {
    __shared__ double s_warp[kWarps];
    double sq = 0.0;
    for (int i = blockIdx.x * kThreads + threadIdx.x; i < n; i += gridDim.x * kThreads) {
        const double v = static_cast<double>(input[i]);
        sq += v * v;
    }
    sq = block_reduce_sum_f64(sq, s_warp);
    if (threadIdx.x == 0) {
        g_partial_sq[blockIdx.x] = sq;
    }
}

__global__ __launch_bounds__(kThreads) void rmsnorm_sumsq_pass2_kernel(int num_blocks,
                                                                       int n,
                                                                       float gamma,
                                                                       float eps) {
    __shared__ double s_warp[kWarps];
    double sq = 0.0;
    for (int i = threadIdx.x; i < num_blocks; i += kThreads) {
        sq += g_partial_sq[i];
    }
    sq = block_reduce_sum_f64(sq, s_warp);
    if (threadIdx.x == 0) {
        const float mean_sq = static_cast<float>(sq / static_cast<double>(n));
        g_inv_rms_scale = gamma * rsqrtf(mean_sq + eps);
    }
}

__global__ __launch_bounds__(kThreads) void rmsnorm_apply_baseline_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n,
    float beta) {
    const float scale = g_inv_rms_scale;
    for (int i = blockIdx.x * kThreads + threadIdx.x; i < n; i += gridDim.x * kThreads) {
        output[i] = fmaf(input[i], scale, beta);
    }
}

}  // namespace

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N,
                      float eps) {
    if (N <= 0) {
        return;
    }

    int blocks = (N + kThreads - 1) / kThreads;
    if (blocks > kMaxBlocks) {
        blocks = kMaxBlocks;
    }
    const unsigned grid = static_cast<unsigned>(blocks);

    rmsnorm_sumsq_pass1_kernel<<<grid, kThreads>>>(input, N);
    rmsnorm_sumsq_pass2_kernel<<<1, kThreads>>>(blocks, N, gamma, eps);
    rmsnorm_apply_baseline_kernel<<<grid, kThreads>>>(input, output, N, beta);
}
