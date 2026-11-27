#include <cuda_runtime.h>
#include <cstddef>

namespace {

constexpr int kBaselineThreads = 256;

__global__ void causal_depthwise_conv1d_baseline_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B,
    int L,
    int D,
    int K) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(B) * L * D;
    if (idx >= total) {
        return;
    }

    const int d = static_cast<int>(idx % static_cast<size_t>(D));
    const size_t bl = idx / static_cast<size_t>(D);
    const int l = static_cast<int>(bl % static_cast<size_t>(L));
    const int b = static_cast<int>(bl / static_cast<size_t>(L));

    const size_t batch_base = static_cast<size_t>(b) * L * D;
    const float* __restrict__ w_row = weight + static_cast<size_t>(d) * K;
    const int k_max = (K - 1 < l) ? (K - 1) : l;

    float acc = bias[d];
    for (int k = 0; k <= k_max; ++k) {
        acc = fmaf(w_row[k], x[batch_base + static_cast<size_t>(l - k) * D + d], acc);
    }
    output[idx] = acc;
}

}  // namespace

// x, weight, bias, output are device pointers
extern "C" void solve(const float* x, const float* weight, const float* bias, float* output, int B,
                      int L, int D, int K) {
    if (B <= 0 || L <= 0 || D <= 0 || K <= 0) {
        return;
    }

    const size_t total = static_cast<size_t>(B) * L * D;
    const int blocks = static_cast<int>((total + kBaselineThreads - 1) / kBaselineThreads);
    causal_depthwise_conv1d_baseline_kernel<<<blocks, kBaselineThreads>>>(
        x, weight, bias, output, B, L, D, K);
}
