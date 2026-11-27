#include <cuda_runtime.h>
#include <cstddef>

namespace {

constexpr int kBaselineThreads = 256;

__global__ void row_inclusive_scan_baseline_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int H,
    int W) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= H) {
        return;
    }

    const size_t row_base = static_cast<size_t>(r) * static_cast<size_t>(W);
    double acc = 0.0;
    for (int c = 0; c < W; ++c) {
        acc += static_cast<double>(input[row_base + c]);
        output[row_base + c] = static_cast<float>(acc);
    }
}

__global__ void col_inclusive_scan_baseline_kernel(
    float* __restrict__ output,
    int H,
    int W) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= W) {
        return;
    }

    double acc = 0.0;
    for (int r = 0; r < H; ++r) {
        const size_t idx = static_cast<size_t>(r) * static_cast<size_t>(W) + static_cast<size_t>(c);
        acc += static_cast<double>(output[idx]);
        output[idx] = static_cast<float>(acc);
    }
}

}  // namespace

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int H, int W) {
    if (H <= 0 || W <= 0) {
        return;
    }

    const int row_blocks = (H + kBaselineThreads - 1) / kBaselineThreads;
    row_inclusive_scan_baseline_kernel<<<row_blocks, kBaselineThreads>>>(input, output, H, W);

    const int col_blocks = (W + kBaselineThreads - 1) / kBaselineThreads;
    col_inclusive_scan_baseline_kernel<<<col_blocks, kBaselineThreads>>>(output, H, W);
}
