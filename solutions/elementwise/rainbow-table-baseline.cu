#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ uint32_t fnv1a_hash_one(uint32_t x) {
    constexpr uint32_t kFnvPrime = 16777619u;
    constexpr uint32_t kOffsetBasis = 2166136261u;

    uint32_t h = kOffsetBasis;
    for (int byte_pos = 0; byte_pos < 4; ++byte_pos) {
        const uint32_t b = (x >> (byte_pos * 8)) & 0xffu;
        h = (h ^ b) * kFnvPrime;
    }
    return h;
}

__global__ __launch_bounds__(kThreads) void rainbow_table_baseline_kernel(
    const int* __restrict__ input,
    unsigned int* __restrict__ output,
    int n,
    int r) {
    const int idx = blockIdx.x * kThreads + threadIdx.x;
    if (idx >= n) {
        return;
    }

    uint32_t val = static_cast<uint32_t>(input[idx]);
    for (int round = 0; round < r; ++round) {
        val = fnv1a_hash_one(val);
    }
    output[idx] = val;
}

}  // namespace

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
    if (N <= 0) {
        return;
    }

    const unsigned blocks = static_cast<unsigned>((N + kThreads - 1) / kThreads);
    rainbow_table_baseline_kernel<<<blocks, kThreads>>>(input, output, N, R);
}
