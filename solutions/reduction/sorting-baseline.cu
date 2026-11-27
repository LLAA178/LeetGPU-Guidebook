#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kMaxPow2 = 1 << 20;  // 1,048,576 >= max N = 1,000,000 (4.0 MiB static .bss)

__device__ uint32_t g_bitonic_keys[kMaxPow2];

__device__ __forceinline__ uint32_t float_to_key(float x) {
    const uint32_t u = __float_as_uint(x);
    const uint32_t mask = (u & 0x80000000u) ? 0xffffffffu : 0x80000000u;
    return u ^ mask;
}

__device__ __forceinline__ float key_to_float(uint32_t key) {
    const uint32_t mask = (key & 0x80000000u) ? 0x80000000u : 0xffffffffu;
    return __uint_as_float(key ^ mask);
}

// Single-block shared-memory bitonic sort for N <= 1024
__global__ __launch_bounds__(512) void bitonic_small_block_kernel(float* __restrict__ data,
                                                                  int n,
                                                                  int pow2) {
    __shared__ uint32_t s_keys[1024];

    const int tid = threadIdx.x;
    const int i0 = tid;
    const int i1 = tid + 512;
    if (i0 < pow2) {
        s_keys[i0] = (i0 < n) ? float_to_key(data[i0]) : 0xffffffffu;
    }
    if (i1 < pow2) {
        s_keys[i1] = (i1 < n) ? float_to_key(data[i1]) : 0xffffffffu;
    }
    __syncthreads();

    for (int k = 2; k <= pow2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            if (tid < (pow2 >> 1)) {
                const int lo = ((tid / j) * (j << 1)) + (tid & (j - 1));
                const int hi = lo ^ j;
                const bool asc = ((lo & k) == 0);
                const uint32_t a = s_keys[lo];
                const uint32_t b = s_keys[hi];
                if ((a > b) == asc) {
                    s_keys[lo] = b;
                    s_keys[hi] = a;
                }
            }
            __syncthreads();
        }
    }

    if (i0 < n) data[i0] = key_to_float(s_keys[i0]);
    if (i1 < n) data[i1] = key_to_float(s_keys[i1]);
}

// Initial 1,024-element block bitonic build from `data` into `g_bitonic_keys`
__global__ __launch_bounds__(512) void bitonic_init_block1024_kernel(
    const float* __restrict__ data,
    int n) {
    __shared__ uint32_t s_keys[1024];

    const int tid = threadIdx.x;
    const int base = blockIdx.x * 1024;
    const int g0 = base + tid;
    const int g1 = base + tid + 512;

    s_keys[tid] = (g0 < n) ? float_to_key(data[g0]) : 0xffffffffu;
    s_keys[tid + 512] = (g1 < n) ? float_to_key(data[g1]) : 0xffffffffu;
    __syncthreads();

    for (int k = 2; k <= 1024; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int lo = ((tid / j) * (j << 1)) + (tid & (j - 1));
            const int hi = lo ^ j;
            const bool asc = (((base + lo) & k) == 0);
            const uint32_t a = s_keys[lo];
            const uint32_t b = s_keys[hi];
            if ((a > b) == asc) {
                s_keys[lo] = b;
                s_keys[hi] = a;
            }
            __syncthreads();
        }
    }

    g_bitonic_keys[g0] = s_keys[tid];
    g_bitonic_keys[g1] = s_keys[tid + 512];
}

// Global compare-exchange step for stride j >= 1024
__global__ __launch_bounds__(256) void bitonic_global_step_kernel(int k, int j) {
    const int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lo = ((pair_id / j) * (j << 1)) + (pair_id & (j - 1));
    const int hi = lo ^ j;
    const bool asc = ((lo & k) == 0);

    const uint32_t a = g_bitonic_keys[lo];
    const uint32_t b = g_bitonic_keys[hi];
    if ((a > b) == asc) {
        g_bitonic_keys[lo] = b;
        g_bitonic_keys[hi] = a;
    }
}

// Fused shared-memory merge for trailing strides j = 512, 256, ..., 1
__global__ __launch_bounds__(512) void bitonic_shared_merge1024_kernel(int k) {
    __shared__ uint32_t s_keys[1024];

    const int tid = threadIdx.x;
    const int base = blockIdx.x * 1024;
    s_keys[tid] = g_bitonic_keys[base + tid];
    s_keys[tid + 512] = g_bitonic_keys[base + tid + 512];
    __syncthreads();

    const bool asc = ((base & k) == 0);
    #pragma unroll
    for (int j = 512; j > 0; j >>= 1) {
        const int lo = ((tid / j) * (j << 1)) + (tid & (j - 1));
        const int hi = lo ^ j;
        const uint32_t a = s_keys[lo];
        const uint32_t b = s_keys[hi];
        if ((a > b) == asc) {
            s_keys[lo] = b;
            s_keys[hi] = a;
        }
        __syncthreads();
    }

    g_bitonic_keys[base + tid] = s_keys[tid];
    g_bitonic_keys[base + tid + 512] = s_keys[tid + 512];
}

__global__ __launch_bounds__(256) void bitonic_writeback_kernel(float* __restrict__ data, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = key_to_float(g_bitonic_keys[idx]);
    }
}

}  // namespace

// data is device pointer
extern "C" void solve(float* data, int N) {
    if (N <= 1) {
        return;
    }

    int pow2 = 1;
    while (pow2 < N) {
        pow2 <<= 1;
    }

    if (pow2 <= 1024) {
        bitonic_small_block_kernel<<<1, 512>>>(data, N, pow2);
        return;
    }

    const unsigned blocks1024 = static_cast<unsigned>(pow2 / 1024);
    const unsigned pair_blocks = static_cast<unsigned>((pow2 / 2) / 256);

    bitonic_init_block1024_kernel<<<blocks1024, 512>>>(data, N);
    for (int k = 2048; k <= pow2; k <<= 1) {
        for (int j = k >> 1; j >= 1024; j >>= 1) {
            bitonic_global_step_kernel<<<pair_blocks, 256>>>(k, j);
        }
        bitonic_shared_merge1024_kernel<<<blocks1024, 512>>>(k);
    }
    bitonic_writeback_kernel<<<static_cast<unsigned>((N + 255) / 256), 256>>>(data, N);
}
