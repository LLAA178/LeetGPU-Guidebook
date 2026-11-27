#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>

namespace {

constexpr int kBaselineThreads = 256;

__global__ void hadamard_stage_baseline_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int B,
    int N,
    int h,
    float post_scale) {
    const size_t pair_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t half_n = static_cast<size_t>(N >> 1);
    const size_t total_pairs = static_cast<size_t>(B) * half_n;
    if (pair_idx >= total_pairs) {
        return;
    }

    const int b = static_cast<int>(pair_idx / half_n);
    const int p = static_cast<int>(pair_idx % half_n);
    const int j = ((p / h) * (2 * h)) + (p % h);

    const size_t base = static_cast<size_t>(b) * static_cast<size_t>(N);
    const float u = src[base + j];
    const float v = src[base + j + h];
    dst[base + j] = (u + v) * post_scale;
    dst[base + j + h] = (u - v) * post_scale;
}

__global__ void hadamard_copy_n1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B) {
        output[idx] = input[idx];
    }
}

}  // namespace

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int B, int N) {
    if (B <= 0 || N <= 0) {
        return;
    }

    if (N == 1) {
        const int blocks = (B + kBaselineThreads - 1) / kBaselineThreads;
        hadamard_copy_n1_kernel<<<blocks, kBaselineThreads>>>(input, output, B);
        return;
    }

    const size_t total_bytes = static_cast<size_t>(B) * static_cast<size_t>(N) * sizeof(float);
    float* d_tmp = nullptr;
    cudaMalloc(&d_tmp, total_bytes);

    int stages = 0;
    for (int t = N; t > 1; t >>= 1) {
        ++stages;
    }

    // Ping-pong so the final stage writes directly into output
    const float* cur_src = input;
    float* cur_dst = (stages & 1) ? output : d_tmp;

    const size_t total_pairs = static_cast<size_t>(B) * static_cast<size_t>(N >> 1);
    const int blocks = static_cast<int>((total_pairs + kBaselineThreads - 1) / kBaselineThreads);
    const float inv_sqrt_n = static_cast<float>(1.0 / std::sqrt(static_cast<double>(N)));

    for (int s = 0, h = 1; h < N; ++s, h <<= 1) {
        const float scale = (s + 1 == stages) ? inv_sqrt_n : 1.0f;
        hadamard_stage_baseline_kernel<<<blocks, kBaselineThreads>>>(
            cur_src, cur_dst, B, N, h, scale);
        cur_src = cur_dst;
        cur_dst = (cur_dst == output) ? d_tmp : output;
    }

    cudaFree(d_tmp);
}
