#include <cuda_runtime.h>

namespace {

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarp;
constexpr unsigned kFull = 0xffffffffu;

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = kWarp / 2; off > 0; off >>= 1) {
        v += __shfl_down_sync(kFull, v, off);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float s_wsums[kWarps];
    __shared__ float s_total;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) s_wsums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float bsum = lane < kWarps ? s_wsums[lane] : 0.0f;
        bsum = warp_reduce_sum(bsum);
        if (lane == 0) s_total = bsum;
    }
    __syncthreads();
    return s_total;
}

__global__ __launch_bounds__(kThreads) void layernorm_baseline_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int C,
    float inv_c,
    float eps) {
    const int tid = threadIdx.x;
    const float* row_in = input + static_cast<size_t>(blockIdx.x) * C;
    float* row_out = output + static_cast<size_t>(blockIdx.x) * C;

    float local_sum = 0.0f;
    for (int j = tid; j < C; j += kThreads) {
        local_sum += row_in[j];
    }
    const float mean = block_reduce_sum(local_sum) * inv_c;

    float local_var = 0.0f;
    for (int j = tid; j < C; j += kThreads) {
        const float d = row_in[j] - mean;
        local_var = fmaf(d, d, local_var);
    }
    const float inv_std = rsqrtf(block_reduce_sum(local_var) * inv_c + eps);

    for (int j = tid; j < C; j += kThreads) {
        row_out[j] = fmaf((row_in[j] - mean) * inv_std, weight[j], bias[j]);
    }
}

}  // namespace

extern "C" void solve(const float* input, const float* weight, const float* bias,
                      float* output, int N, int C, float eps) {
    if (N <= 0 || C <= 0) return;
    layernorm_baseline_kernel<<<N, kThreads>>>(
        input, weight, bias, output, C, 1.0f / static_cast<float>(C), eps);
}
