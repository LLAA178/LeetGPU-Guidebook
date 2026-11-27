#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ uint32_t float_to_ordered_uint(float x) {
    if (x == 0.0f) {
        x = 0.0f;
    }
    const uint32_t u = __float_as_uint(x);
    return (u & 0x80000000u) ? (~u) : (u ^ 0x80000000u);
}

__device__ __forceinline__ float ordered_uint_to_float(uint32_t k) {
    const uint32_t u = (k & 0x80000000u) ? (k ^ 0x80000000u) : (~k);
    return __uint_as_float(u);
}

__global__ void init_padded_keys_kernel(const float* __restrict__ input,
                                        uint32_t* __restrict__ keys,
                                        int n,
                                        int padded_n) {
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < padded_n; i += stride) {
        keys[i] = (i < n) ? float_to_ordered_uint(input[i]) : 0u;
    }
}

__global__ void bitonic_step_kernel(uint32_t* __restrict__ keys, int padded_n, int j, int stage_k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= padded_n) {
        return;
    }
    const int ixj = i ^ j;
    if (ixj > i) {
        const uint32_t a = keys[i];
        const uint32_t b = keys[ixj];
        // Sort in descending order for stage_k-sized monotonic blocks when (i & stage_k) == 0
        const bool descending = ((i & stage_k) == 0);
        if ((descending && a < b) || (!descending && a > b)) {
            keys[i] = b;
            keys[ixj] = a;
        }
    }
}

__global__ void write_topk_kernel(const uint32_t* __restrict__ keys,
                                  float* __restrict__ output,
                                  int k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) {
        output[i] = ordered_uint_to_float(keys[i]);
    }
}

}  // namespace

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {
    if (N <= 0 || k <= 0) {
        return;
    }
    if (k > N) {
        k = N;
    }

    int padded_n = 1;
    while (padded_n < N) {
        padded_n <<= 1;
    }

    uint32_t* d_keys = nullptr;
    cudaMalloc(&d_keys, static_cast<size_t>(padded_n) * sizeof(uint32_t));

    int init_blocks = (padded_n + kThreads - 1) / kThreads;
    if (init_blocks > 4096) init_blocks = 4096;
    init_padded_keys_kernel<<<init_blocks, kThreads>>>(input, d_keys, N, padded_n);

    const int sort_blocks = (padded_n + kThreads - 1) / kThreads;
    for (int stage_k = 2; stage_k <= padded_n; stage_k <<= 1) {
        for (int j = stage_k >> 1; j > 0; j >>= 1) {
            bitonic_step_kernel<<<sort_blocks, kThreads>>>(d_keys, padded_n, j, stage_k);
        }
    }

    const int out_blocks = (k + kThreads - 1) / kThreads;
    write_topk_kernel<<<out_blocks, kThreads>>>(d_keys, output, k);

    cudaFree(d_keys);
}
