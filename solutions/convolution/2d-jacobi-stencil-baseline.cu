#include <cuda_runtime.h>

namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

__global__ void jacobi_2d_baseline_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int rows,
                                          int cols) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= rows || j >= cols) {
        return;
    }

    const size_t idx = static_cast<size_t>(i) * static_cast<size_t>(cols) + static_cast<size_t>(j);
    if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
        output[idx] = input[idx];
        return;
    }

    const float top = input[idx - static_cast<size_t>(cols)];
    const float bot = input[idx + static_cast<size_t>(cols)];
    const float left = input[idx - 1u];
    const float right = input[idx + 1u];
    output[idx] = 0.25f * (top + bot + left + right);
}

}  // namespace

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        return;
    }

    const dim3 block(kBlockX, kBlockY);
    const dim3 grid(static_cast<unsigned>((cols + kBlockX - 1) / kBlockX),
                    static_cast<unsigned>((rows + kBlockY - 1) / kBlockY));
    jacobi_2d_baseline_kernel<<<grid, block>>>(input, output, rows, cols);
}
