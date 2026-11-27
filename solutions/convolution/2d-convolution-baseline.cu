#include <cuda_runtime.h>
#include <cstddef>

namespace {

constexpr int kBlockDimX = 16;
constexpr int kBlockDimY = 16;

__global__ void conv2d_valid_baseline_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_cols,
    int kernel_rows,
    int kernel_cols,
    int out_rows,
    int out_cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= out_rows || col >= out_cols) {
        return;
    }

    float sum = 0.0f;
    for (int kr = 0; kr < kernel_rows; ++kr) {
        const size_t in_row_base = static_cast<size_t>(row + kr) * static_cast<size_t>(input_cols);
        const int k_row_base = kr * kernel_cols;
        float row_sum = 0.0f;
        for (int kc = 0; kc < kernel_cols; ++kc) {
            row_sum = fmaf(input[in_row_base + col + kc], kernel[k_row_base + kc], row_sum);
        }
        sum += row_sum;
    }
    output[static_cast<size_t>(row) * static_cast<size_t>(out_cols) + col] = sum;
}

}  // namespace

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
    const int out_rows = input_rows - kernel_rows + 1;
    const int out_cols = input_cols - kernel_cols + 1;
    if (out_rows <= 0 || out_cols <= 0 || kernel_rows <= 0 || kernel_cols <= 0) {
        return;
    }

    const dim3 block(kBlockDimX, kBlockDimY);
    const dim3 grid(static_cast<unsigned>((out_cols + kBlockDimX - 1) / kBlockDimX),
                    static_cast<unsigned>((out_rows + kBlockDimY - 1) / kBlockDimY));
    conv2d_valid_baseline_kernel<<<grid, block>>>(
        input, kernel, output, input_cols, kernel_rows, kernel_cols, out_rows, out_cols);
}
