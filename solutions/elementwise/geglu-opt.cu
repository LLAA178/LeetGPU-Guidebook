#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr float kInvSqrt2 = 0.70710678118654752440f;

__host__ __device__ __forceinline__ bool is_aligned_16b(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 15u) == 0u;
}

__device__ __forceinline__ void store_cs_f4(float4* ptr, float4 v) {
#if __CUDA_ARCH__ >= 700
    asm volatile("st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
                 : "memory");
#else
    *ptr = v;
#endif
}

__device__ __forceinline__ float eval_geglu(float x1, float x2) {
    return x1 * (0.5f * x2 * (1.0f + erff(x2 * kInvSqrt2)));
}

__device__ __forceinline__ float4 eval_geglu4(float4 a, float4 b) {
    return make_float4(eval_geglu(a.x, b.x), eval_geglu(a.y, b.y),
                       eval_geglu(a.z, b.z), eval_geglu(a.w, b.w));
}

__global__ __launch_bounds__(kThreads) void geglu_vec4_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int half4) {
    const int gtid = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
    const int gstride = static_cast<int>(gridDim.x) * kThreads;

    const float4* __restrict__ x1_4 = reinterpret_cast<const float4*>(input);
    const float4* __restrict__ x2_4 = x1_4 + half4;
    float4* __restrict__ out4 = reinterpret_cast<float4*>(output);

    int i4 = gtid;
    for (; i4 + gstride < half4; i4 += 2 * gstride) {
        const float4 a0 = __ldg(&x1_4[i4]);
        const float4 b0 = __ldg(&x2_4[i4]);
        const float4 a1 = __ldg(&x1_4[i4 + gstride]);
        const float4 b1 = __ldg(&x2_4[i4 + gstride]);

        store_cs_f4(&out4[i4], eval_geglu4(a0, b0));
        store_cs_f4(&out4[i4 + gstride], eval_geglu4(a1, b1));
    }
    for (; i4 < half4; i4 += gstride) {
        const float4 a = __ldg(&x1_4[i4]);
        const float4 b = __ldg(&x2_4[i4]);
        store_cs_f4(&out4[i4], eval_geglu4(a, b));
    }
}

__global__ __launch_bounds__(kThreads) void geglu_scalar_ilp4_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int half_n) {
    const int gtid = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
    const int gstride = static_cast<int>(gridDim.x) * kThreads;
    const float* __restrict__ x2 = input + half_n;

    int i = gtid;
    for (; i + 3 * gstride < half_n; i += 4 * gstride) {
        const float a0 = __ldg(&input[i]), b0 = __ldg(&x2[i]);
        const float a1 = __ldg(&input[i + gstride]), b1 = __ldg(&x2[i + gstride]);
        const float a2 = __ldg(&input[i + 2 * gstride]), b2 = __ldg(&x2[i + 2 * gstride]);
        const float a3 = __ldg(&input[i + 3 * gstride]), b3 = __ldg(&x2[i + 3 * gstride]);

        output[i] = eval_geglu(a0, b0);
        output[i + gstride] = eval_geglu(a1, b1);
        output[i + 2 * gstride] = eval_geglu(a2, b2);
        output[i + 3 * gstride] = eval_geglu(a3, b3);
    }
    for (; i < half_n; i += gstride) {
        output[i] = eval_geglu(__ldg(&input[i]), __ldg(&x2[i]));
    }
}

}  // namespace

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    const int half_n = N >> 1;
    if (half_n <= 0) {
        return;
    }

    if ((half_n & 3) == 0 && is_aligned_16b(input) && is_aligned_16b(output)) {
        const int half4 = half_n >> 2;
        int blocks = (half4 + 511) / 512;  // 245 -> clamped to 240 (6 exact waves of 40 SMs) at N = 1M!
        if (blocks < 1) blocks = 1;
        if (blocks > 240) blocks = 240;
        geglu_vec4_kernel<<<static_cast<unsigned>(blocks), kThreads>>>(input, output, half4);
        return;
    }

    int blocks = (half_n + (kThreads * 4) - 1) / (kThreads * 4);
    if (blocks < 1) blocks = 1;
    if (blocks > 240) blocks = 240;
    geglu_scalar_ilp4_kernel<<<static_cast<unsigned>(blocks), kThreads>>>(input, output, half_n);
}
