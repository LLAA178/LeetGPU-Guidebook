#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr uint32_t kFnvPrime = 16777619u;
constexpr uint32_t kOffsetBasis = 2166136261u;

__host__ __device__ __forceinline__ bool is_aligned_16b(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 15u) == 0u;
}

__device__ __forceinline__ void store_cs_u4(uint4* ptr, uint4 v) {
#if __CUDA_ARCH__ >= 700
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
#else
    *ptr = v;
#endif
}

__device__ __forceinline__ uint32_t fnv1a_step(uint32_t x) {
    uint32_t h = (kOffsetBasis ^ (x & 0xffu)) * kFnvPrime;
    h = (h ^ ((x >> 8) & 0xffu)) * kFnvPrime;
    h = (h ^ ((x >> 16) & 0xffu)) * kFnvPrime;
    h = (h ^ (x >> 24)) * kFnvPrime;
    return h;
}

// Interleaved 4-lane FNV-1a round: hides IMAD->XOR latency via 4x ILP!
__device__ __forceinline__ uint4 fnv1a_step_u4(uint4 v) {
    uint32_t h0 = (kOffsetBasis ^ (v.x & 0xffu)) * kFnvPrime;
    uint32_t h1 = (kOffsetBasis ^ (v.y & 0xffu)) * kFnvPrime;
    uint32_t h2 = (kOffsetBasis ^ (v.z & 0xffu)) * kFnvPrime;
    uint32_t h3 = (kOffsetBasis ^ (v.w & 0xffu)) * kFnvPrime;

    h0 = (h0 ^ ((v.x >> 8) & 0xffu)) * kFnvPrime;
    h1 = (h1 ^ ((v.y >> 8) & 0xffu)) * kFnvPrime;
    h2 = (h2 ^ ((v.z >> 8) & 0xffu)) * kFnvPrime;
    h3 = (h3 ^ ((v.w >> 8) & 0xffu)) * kFnvPrime;

    h0 = (h0 ^ ((v.x >> 16) & 0xffu)) * kFnvPrime;
    h1 = (h1 ^ ((v.y >> 16) & 0xffu)) * kFnvPrime;
    h2 = (h2 ^ ((v.z >> 16) & 0xffu)) * kFnvPrime;
    h3 = (h3 ^ ((v.w >> 16) & 0xffu)) * kFnvPrime;

    h0 = (h0 ^ (v.x >> 24)) * kFnvPrime;
    h1 = (h1 ^ (v.y >> 24)) * kFnvPrime;
    h2 = (h2 ^ (v.z >> 24)) * kFnvPrime;
    h3 = (h3 ^ (v.w >> 24)) * kFnvPrime;

    return make_uint4(h0, h1, h2, h3);
}

// Interleaved 8-lane (2 x uint4) FNV-1a round for 8x ILP saturation!
__device__ __forceinline__ void fnv1a_step_u8(uint4& a, uint4& b) {
    uint32_t a0 = (kOffsetBasis ^ (a.x & 0xffu)) * kFnvPrime;
    uint32_t a1 = (kOffsetBasis ^ (a.y & 0xffu)) * kFnvPrime;
    uint32_t a2 = (kOffsetBasis ^ (a.z & 0xffu)) * kFnvPrime;
    uint32_t a3 = (kOffsetBasis ^ (a.w & 0xffu)) * kFnvPrime;
    uint32_t b0 = (kOffsetBasis ^ (b.x & 0xffu)) * kFnvPrime;
    uint32_t b1 = (kOffsetBasis ^ (b.y & 0xffu)) * kFnvPrime;
    uint32_t b2 = (kOffsetBasis ^ (b.z & 0xffu)) * kFnvPrime;
    uint32_t b3 = (kOffsetBasis ^ (b.w & 0xffu)) * kFnvPrime;

    a0 = (a0 ^ ((a.x >> 8) & 0xffu)) * kFnvPrime;
    a1 = (a1 ^ ((a.y >> 8) & 0xffu)) * kFnvPrime;
    a2 = (a2 ^ ((a.z >> 8) & 0xffu)) * kFnvPrime;
    a3 = (a3 ^ ((a.w >> 8) & 0xffu)) * kFnvPrime;
    b0 = (b0 ^ ((b.x >> 8) & 0xffu)) * kFnvPrime;
    b1 = (b1 ^ ((b.y >> 8) & 0xffu)) * kFnvPrime;
    b2 = (b2 ^ ((b.z >> 8) & 0xffu)) * kFnvPrime;
    b3 = (b3 ^ ((b.w >> 8) & 0xffu)) * kFnvPrime;

    a0 = (a0 ^ ((a.x >> 16) & 0xffu)) * kFnvPrime;
    a1 = (a1 ^ ((a.y >> 16) & 0xffu)) * kFnvPrime;
    a2 = (a2 ^ ((a.z >> 16) & 0xffu)) * kFnvPrime;
    a3 = (a3 ^ ((a.w >> 16) & 0xffu)) * kFnvPrime;
    b0 = (b0 ^ ((b.x >> 16) & 0xffu)) * kFnvPrime;
    b1 = (b1 ^ ((b.y >> 16) & 0xffu)) * kFnvPrime;
    b2 = (b2 ^ ((b.z >> 16) & 0xffu)) * kFnvPrime;
    b3 = (b3 ^ ((b.w >> 16) & 0xffu)) * kFnvPrime;

    a.x = (a0 ^ (a.x >> 24)) * kFnvPrime;
    a.y = (a1 ^ (a.y >> 24)) * kFnvPrime;
    a.z = (a2 ^ (a.z >> 24)) * kFnvPrime;
    a.w = (a3 ^ (a.w >> 24)) * kFnvPrime;
    b.x = (b0 ^ (b.x >> 24)) * kFnvPrime;
    b.y = (b1 ^ (b.y >> 24)) * kFnvPrime;
    b.z = (b2 ^ (b.z >> 24)) * kFnvPrime;
    b.w = (b3 ^ (b.w >> 24)) * kFnvPrime;
}

template <int kFixedR>
__global__ __launch_bounds__(kThreads) void rainbow_vec4_ilp8_kernel(
    const int* __restrict__ input,
    unsigned int* __restrict__ output,
    int n,
    int runtime_r) {
    const int gtid = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
    const int gstride = static_cast<int>(gridDim.x) * kThreads;
    const int rounds = (kFixedR > 0) ? kFixedR : runtime_r;

    const uint4* __restrict__ in4 = reinterpret_cast<const uint4*>(input);
    uint4* __restrict__ out4 = reinterpret_cast<uint4*>(output);
    const int n4 = n >> 2;

    int i4 = gtid;
    for (; i4 + gstride < n4; i4 += 2 * gstride) {
        uint4 a = __ldg(&in4[i4]);
        uint4 b = __ldg(&in4[i4 + gstride]);

        if (kFixedR > 0) {
            #pragma unroll
            for (int r = 0; r < kFixedR; ++r) {
                fnv1a_step_u8(a, b);
            }
        } else {
            #pragma unroll 4
            for (int r = 0; r < rounds; ++r) {
                fnv1a_step_u8(a, b);
            }
        }

        store_cs_u4(&out4[i4], a);
        store_cs_u4(&out4[i4 + gstride], b);
    }

    for (; i4 < n4; i4 += gstride) {
        uint4 v = __ldg(&in4[i4]);
        if (kFixedR > 0) {
            #pragma unroll
            for (int r = 0; r < kFixedR; ++r) {
                v = fnv1a_step_u4(v);
            }
        } else {
            #pragma unroll 4
            for (int r = 0; r < rounds; ++r) {
                v = fnv1a_step_u4(v);
            }
        }
        store_cs_u4(&out4[i4], v);
    }

    for (int i = (n4 << 2) + gtid; i < n; i += gstride) {
        uint32_t x = static_cast<uint32_t>(__ldg(&input[i]));
        if (kFixedR > 0) {
            #pragma unroll
            for (int r = 0; r < kFixedR; ++r) {
                x = fnv1a_step(x);
            }
        } else {
            #pragma unroll 4
            for (int r = 0; r < rounds; ++r) {
                x = fnv1a_step(x);
            }
        }
        output[i] = x;
    }
}

__global__ __launch_bounds__(kThreads) void rainbow_scalar_ilp4_kernel(
    const int* __restrict__ input,
    unsigned int* __restrict__ output,
    int n,
    int rounds) {
    const int gtid = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
    const int gstride = static_cast<int>(gridDim.x) * kThreads;

    int i = gtid;
    for (; i + 3 * gstride < n; i += 4 * gstride) {
        uint4 v = make_uint4(static_cast<uint32_t>(__ldg(&input[i])),
                             static_cast<uint32_t>(__ldg(&input[i + gstride])),
                             static_cast<uint32_t>(__ldg(&input[i + 2 * gstride])),
                             static_cast<uint32_t>(__ldg(&input[i + 3 * gstride])));
        #pragma unroll 4
        for (int r = 0; r < rounds; ++r) {
            v = fnv1a_step_u4(v);
        }
        output[i] = v.x;
        output[i + gstride] = v.y;
        output[i + 2 * gstride] = v.z;
        output[i + 3 * gstride] = v.w;
    }
    for (; i < n; i += gstride) {
        uint32_t x = static_cast<uint32_t>(__ldg(&input[i]));
        #pragma unroll 4
        for (int r = 0; r < rounds; ++r) {
            x = fnv1a_step(x);
        }
        output[i] = x;
    }
}

}  // namespace

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
    if (N <= 0) {
        return;
    }

    if (is_aligned_16b(input) && is_aligned_16b(output)) {
        const int elems_per_block = kThreads * 8;
        int blocks = (N + elems_per_block - 1) / elems_per_block;
        if (blocks < 1) blocks = 1;
        if (blocks > 640) blocks = 640;  // 16 waves of 40 SMs -> ample MLP & zero grid overhead
        const unsigned grid = static_cast<unsigned>(blocks);

        if (R == 10) {
            rainbow_vec4_ilp8_kernel<10><<<grid, kThreads>>>(input, output, N, 10);
        } else if (R == 1) {
            rainbow_vec4_ilp8_kernel<1><<<grid, kThreads>>>(input, output, N, 1);
        } else {
            rainbow_vec4_ilp8_kernel<0><<<grid, kThreads>>>(input, output, N, R);
        }
        return;
    }

    int blocks = (N + (kThreads * 4) - 1) / (kThreads * 4);
    if (blocks < 1) blocks = 1;
    if (blocks > 640) blocks = 640;
    rainbow_scalar_ilp4_kernel<<<static_cast<unsigned>(blocks), kThreads>>>(input, output, N, R);
}
