#include <cuda_runtime.h>
#include <cmath>

namespace {

constexpr int kThreads = 256;
constexpr float kInvSqrt2 = 0.70710678118654752440f;

__global__ __launch_bounds__(kThreads) void geglu_baseline_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int half_n) {
    const int idx = blockIdx.x * kThreads + threadIdx.x;
    if (idx >= half_n) {
        return;
    }

    const float x1 = input[idx];
    const float x2 = input[idx + half_n];
    const float gelu = 0.5f * x2 * (1.0f + erff(x2 * kInvSqrt2));
    output[idx] = x1 * gelu;
}

}  // namespace

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    const int half_n = N >> 1;
    if (half_n <= 0) {
        return;
    }

    const unsigned blocks = static_cast<unsigned>((half_n + kThreads - 1) / kThreads);
    geglu_baseline_kernel<<<blocks, kThreads>>>(input, output, half_n);
}
