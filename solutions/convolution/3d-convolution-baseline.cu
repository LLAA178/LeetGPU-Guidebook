#include <cuda_runtime.h>

namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 8;
constexpr int kBlockZ = 2;

__global__ void conv3d_baseline_kernel(const float* __restrict__ input,
                                       const float* __restrict__ kernel,
                                       float* __restrict__ output,
                                       int input_rows,
                                       int input_cols,
                                       int kernel_depth,
                                       int kernel_rows,
                                       int kernel_cols,
                                       int out_d,
                                       int out_r,
                                       int out_c) {
    const int oc = blockIdx.x * blockDim.x + threadIdx.x;
    const int orow = blockIdx.y * blockDim.y + threadIdx.y;
    const int od = blockIdx.z * blockDim.z + threadIdx.z;

    if (oc >= out_c || orow >= out_r || od >= out_d) {
        return;
    }

    float sum = 0.0f;
    for (int kd = 0; kd < kernel_depth; ++kd) {
        const size_t in_d_off =
            static_cast<size_t>(od + kd) * static_cast<size_t>(input_rows);
        const size_t k_d_off =
            static_cast<size_t>(kd) * static_cast<size_t>(kernel_rows);
        for (int kr = 0; kr < kernel_rows; ++kr) {
            const float* in_row =
                input + (in_d_off + static_cast<size_t>(orow + kr)) * static_cast<size_t>(input_cols) +
                static_cast<size_t>(oc);
            const float* k_row =
                kernel + (k_d_off + static_cast<size_t>(kr)) * static_cast<size_t>(kernel_cols);
            for (int kc = 0; kc < kernel_cols; ++kc) {
                sum = fmaf(in_row[kc], k_row[kc], sum);
            }
        }
    }

    const size_t out_idx =
        (static_cast<size_t>(od) * static_cast<size_t>(out_r) + static_cast<size_t>(orow)) *
            static_cast<size_t>(out_c) +
        static_cast<size_t>(oc);
    output[out_idx] = sum;
}

}  // namespace

// input, kernel, output are device pointers
extern "C" void solve(const float* input,
                      const float* kernel,
                      float* output,
                      int input_depth,
                      int input_rows,
                      int input_cols,
                      int kernel_depth,
                      int kernel_rows,
                      int kernel_cols) {
    const int out_d = input_depth - kernel_depth + 1;
    const int out_r = input_rows - kernel_rows + 1;
    const int out_c = input_cols - kernel_cols + 1;
    if (out_d <= 0 || out_r <= 0 || out_c <= 0) {
        return;
    }

    const dim3 block(kBlockX, kBlockY, kBlockZ);
    const dim3 grid(static_cast<unsigned>((out_c + kBlockX - 1) / kBlockX),
                    static_cast<unsigned>((out_r + kBlockY - 1) / kBlockY),
                    static_cast<unsigned>((out_d + kBlockZ - 1) / kBlockZ));
    conv3d_baseline_kernel<<<grid, block>>>(
        input, kernel, output, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols, out_d, out_r, out_c);
}
