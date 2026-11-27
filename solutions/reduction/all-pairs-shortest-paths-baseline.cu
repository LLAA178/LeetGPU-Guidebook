#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>

namespace {

constexpr int kBlockDim = 16;

__global__ __launch_bounds__(256) void fw_unblocked_step_kernel(float* __restrict__ d,
                                                                int n,
                                                                int k) {
    const int j = blockIdx.x * kBlockDim + threadIdx.x;
    const int i = blockIdx.y * kBlockDim + threadIdx.y;
    if (i >= n || j >= n) {
        return;
    }

    const size_t ij = static_cast<size_t>(i) * static_cast<size_t>(n) + j;
    const float dik = d[static_cast<size_t>(i) * static_cast<size_t>(n) + k];
    const float dkj = d[static_cast<size_t>(k) * static_cast<size_t>(n) + j];
    const float cur = d[ij];
    const float cand = dik + dkj;
    if (cand < cur) {
        d[ij] = cand;
    }
}

}  // namespace

// dist, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* dist, float* output, int N) {
    if (N <= 0) {
        return;
    }

    if (dist != output) {
        const size_t bytes = static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(float);
        cudaMemcpyAsync(output, dist, bytes, cudaMemcpyDeviceToDevice);
    }

    const dim3 block(kBlockDim, kBlockDim);
    const dim3 grid(static_cast<unsigned>((N + kBlockDim - 1) / kBlockDim),
                    static_cast<unsigned>((N + kBlockDim - 1) / kBlockDim));

    for (int k = 0; k < N; ++k) {
        fw_unblocked_step_kernel<<<grid, block>>>(output, N, k);
    }
}
