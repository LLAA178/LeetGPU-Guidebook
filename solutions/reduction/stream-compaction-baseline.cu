#include <cuda_runtime.h>

namespace {

__global__ void compact(const float* input, int n, float* output) {
    int next = 0;
    for (int i = 0; i < n; ++i) {
        const float value = input[i];
        if (value > 0.0f) output[next++] = value;
    }
}

}  // namespace

extern "C" void solve(const float* A, int N, float* out) {
    compact<<<1, 1>>>(A, N, out);
}
