#include <cuda_runtime.h>
#include <cstddef>

namespace {

constexpr int kBlockDimX = 16;
constexpr int kBlockDimY = 16;

__global__ void gaussian_blur_baseline_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_rows,
    int input_cols,
    int kernel_rows,
    int kernel_cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= input_rows || col >= input_cols) {
        return;
    }

    const int pad_h = kernel_rows / 2;
    const int pad_w = kernel_cols / 2;

    float sum = 0.0f;
    for (int kr = 0; kr < kernel_rows; ++kr) {
        const int r = row + kr - pad_h;
        if (r >= 0 && r < input_rows) {
            const size_t in_row_base = static_cast<size_t>(r) * static_cast<size_t>(input_cols);
            const int k_row_base = kr * kernel_cols;
            for (int kc = 0; kc < kernel_cols; ++kc) {
                const int c = col + kc - pad_w;
                if (c >= 0 && c < input_cols) {
                    sum = fmaf(input[in_row_base + c], kernel[k_row_base + kc], sum);
                }
            }
        }
    }
    output[static_cast<size_t>(row) * static_cast<size_t>(input_cols) + col] = sum;
}

}  // namespace

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
    if (input_rows <= 0 || input_cols <= 0 || kernel_rows <= 0 || kernel_cols <= 0) {
        return;
    }

    const dim3 block(kBlockDimX, kBlockDimY);
    const dim3 grid(static_cast<unsigned>((input_cols + kBlockDimX - 1) / kBlockDimX),
                    static_cast<unsigned>((input_rows + kBlockDimY - 1) / kBlockDimY));
    gaussian_blur_baseline_kernel<<<grid, block>>>(
        input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
}
