#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;

__global__ void linear_recurrence_baseline_kernel(const float* __restrict__ a,
                                                  const float* __restrict__ x,
                                                  float* __restrict__ h,
                                                  int B,
                                                  int L) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) {
        return;
    }

    const size_t row_offset = static_cast<size_t>(b) * static_cast<size_t>(L);
    const float* row_a = a + row_offset;
    const float* row_x = x + row_offset;
    float* row_h = h + row_offset;

    float prev = row_x[0];
    row_h[0] = prev;
    for (int t = 1; t < L; ++t) {
        prev = row_a[t] * prev + row_x[t];
        row_h[t] = prev;
    }
}

}  // namespace

// a, x, h are device pointers
extern "C" void solve(const float* a, const float* x, float* h, int B, int L) {
    if (B <= 0 || L <= 0) {
        return;
    }

    const int blocks = (B + kThreads - 1) / kThreads;
    linear_recurrence_baseline_kernel<<<blocks, kThreads>>>(a, x, h, B, L);
}
