#include <cuda_runtime.h>

#include <cstddef>

namespace {

__global__ __launch_bounds__(256) void matrix_transpose_baseline_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols) {
    const int c = static_cast<int>(blockIdx.x) * 16 + static_cast<int>(threadIdx.x);
    const int r = static_cast<int>(blockIdx.y) * 16 + static_cast<int>(threadIdx.y);
    if (r < rows && c < cols) {
        output[static_cast<size_t>(c) * static_cast<size_t>(rows) + r] =
            input[static_cast<size_t>(r) * static_cast<size_t>(cols) + c];
    }
}

}  // namespace

extern "C" void solve(const float* input, float* output, int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        return;
    }

    const dim3 threads(16, 16);
    const dim3 blocks(static_cast<unsigned int>((cols + 15) / 16),
                      static_cast<unsigned int>((rows + 15) / 16));

    matrix_transpose_baseline_kernel<<<blocks, threads>>>(input, output, rows, cols);
}
