#include <cuda_runtime.h>

#include <cmath>

namespace {

__global__ __launch_bounds__(256) void conv1d_valid_baseline_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int output_size,
    int kernel_size) {
    const int i = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
    if (i >= output_size) {
        return;
    }

    float sum = 0.0f;
    const float* in_ptr = input + i;
    for (int j = 0; j < kernel_size; ++j) {
        sum = fmaf(in_ptr[j], kernel[j], sum);
    }
    output[i] = sum;
}

}  // namespace

extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_size, int kernel_size) {
    if (input_size <= 0 || kernel_size <= 0 || kernel_size > input_size) {
        return;
    }

    const int output_size = input_size - kernel_size + 1;
    const unsigned int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((output_size + 255) / 256);

    conv1d_valid_baseline_kernel<<<blocks, threads>>>(
        input, kernel, output, output_size, kernel_size);
}
