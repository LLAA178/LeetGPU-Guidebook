#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBins = 1024;
constexpr int kMaxBlocks = 4096;

__global__ __launch_bounds__(kThreads) void histogram_baseline_kernel(
    const int* __restrict__ input,
    int* __restrict__ histogram,
    int n,
    int num_bins) {
    __shared__ int s_hist[kMaxBins];

    for (int b = threadIdx.x; b < num_bins; b += kThreads) {
        s_hist[b] = 0;
    }
    __syncthreads();

    const int global_tid = blockIdx.x * kThreads + threadIdx.x;
    const int stride = gridDim.x * kThreads;
    const unsigned u_bins = static_cast<unsigned>(num_bins);

    for (int i = global_tid; i < n; i += stride) {
        const unsigned v = static_cast<unsigned>(input[i]);
        if (v < u_bins) {
            atomicAdd(&s_hist[v], 1);
        }
    }
    __syncthreads();

    for (int b = threadIdx.x; b < num_bins; b += kThreads) {
        const int count = s_hist[b];
        if (count != 0) {
            atomicAdd(&histogram[b], count);
        }
    }
}

}  // namespace

extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    if (num_bins <= 0) {
        return;
    }

    cudaMemsetAsync(histogram, 0, static_cast<size_t>(num_bins) * sizeof(int), 0);
    if (N <= 0) {
        return;
    }

    int blocks = (N + kThreads - 1) / kThreads;
    if (blocks > kMaxBlocks) {
        blocks = kMaxBlocks;
    }
    histogram_baseline_kernel<<<blocks, kThreads>>>(input, histogram, N, num_bins);
}
