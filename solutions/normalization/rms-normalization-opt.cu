#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarp;     // 8 warps
constexpr int kSmallMaxN = 8192;             // 32 KiB shared memory single-block path
constexpr int kMaxPersistentBlocks = 160;
constexpr unsigned kFullMask = 0xffffffffu;

// Compact static .bss (~2.01 KiB << 64 KiB limit!)
__device__ double g_blk_sq[kThreads];
__device__ unsigned int g_retired_blocks = 0u;
__device__ float g_scale;

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

__device__ __forceinline__ double block_reduce_sum_f64(double val, double* s_warp) {
    const int tid = threadIdx.x;
    const int lane = tid & (kWarp - 1);
    const int warp = tid >> 5;

    #pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(kFullMask, val, offset);
    }
    if (lane == 0) {
        s_warp[warp] = val;
    }
    __syncthreads();

    double out = 0.0;
    if (warp == 0) {
        out = (lane < kWarps) ? s_warp[lane] : 0.0;
        #pragma unroll
        for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
            out += __shfl_down_sync(kFullMask, out, offset);
        }
    }
    return out;
}

// ============================================================================
// 1. Single-Launch Shared-Memory `float4` Kernel for `1 <= N <= 8192`
// Reads `input` ONCE into 32 KiB shared memory, computes `scale` in 1 block,
// and writes `output` ONCE (`1R + 1W` in 1 launch!)
// ============================================================================
__global__ __launch_bounds__(kThreads) void rmsnorm_small_single_block_kernel(
    const float* __restrict__ input,
    float gamma,
    float beta,
    float* __restrict__ output,
    int n,
    float eps) {
    __shared__ __align__(16) float s_data[kSmallMaxN];
    __shared__ double s_warp[kWarps];
    __shared__ float s_scale;

    const int tid = threadIdx.x;
    double thread_sq = 0.0;

    if (is_aligned_16b(input)) {
        const int n4 = n >> 2;
        const float4* __restrict__ in4 = reinterpret_cast<const float4*>(input);
        float4* s4 = reinterpret_cast<float4*>(s_data);
        float acc_f = 0.0f;
        for (int i4 = tid; i4 < n4; i4 += kThreads) {
            const float4 v = __ldg(&in4[i4]);
            s4[i4] = v;
            acc_f += fmaf(v.x, v.x, v.y * v.y) + fmaf(v.z, v.z, v.w * v.w);
        }
        thread_sq = static_cast<double>(acc_f);
        for (int i = (n4 << 2) + tid; i < n; i += kThreads) {
            const float v = __ldg(&input[i]);
            s_data[i] = v;
            thread_sq += static_cast<double>(v) * static_cast<double>(v);
        }
    } else {
        float acc_f = 0.0f;
        for (int i = tid; i < n; i += kThreads) {
            const float v = __ldg(&input[i]);
            s_data[i] = v;
            acc_f = fmaf(v, v, acc_f);
        }
        thread_sq = static_cast<double>(acc_f);
    }

    const double total_sq = block_reduce_sum_f64(thread_sq, s_warp);
    if (tid == 0) {
        const float mean_sq = static_cast<float>(total_sq / static_cast<double>(n));
        s_scale = gamma * rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    const float scale = s_scale;
    if (is_aligned_16b(output)) {
        const int n4 = n >> 2;
        const float4* s4 = reinterpret_cast<const float4*>(s_data);
        float4* __restrict__ out4 = reinterpret_cast<float4*>(output);
        for (int i4 = tid; i4 < n4; i4 += kThreads) {
            const float4 v = s4[i4];
            const float4 y = make_float4(fmaf(v.x, scale, beta), fmaf(v.y, scale, beta),
                                         fmaf(v.z, scale, beta), fmaf(v.w, scale, beta));
            store_cs_f4(&out4[i4], y);
        }
        for (int i = (n4 << 2) + tid; i < n; i += kThreads) {
            output[i] = fmaf(s_data[i], scale, beta);
        }
    } else {
        for (int i = tid; i < n; i += kThreads) {
            output[i] = fmaf(s_data[i], scale, beta);
        }
    }
}

// ============================================================================
// 2a. Large-N Pass 1: 8x-Unrolled `float4` Sum-of-Squares + Last-Block Finalize
// - Pure FP32 `fmaf` across 32 floats (`8` `float4`s), accumulating into `double`
//   only ONCE per 32 floats (zero Turing 1/32-rate FP64 bottleneck!).
// - Peels `0..3` misaligned head scalars for full `float4` alignment on any ptr!
// - Warms `N = 100,000` (`400 KB`) right into T4's `4 MiB` L2 cache!
// ============================================================================
__global__ __launch_bounds__(kThreads) void rmsnorm_sumsq_vec4_kernel(
    const float* __restrict__ input,
    float gamma,
    int n,
    float eps) {
    __shared__ double s_warp[kWarps];
    __shared__ bool s_is_last;

    const int tid = threadIdx.x;
    const int gtid = static_cast<int>(blockIdx.x) * kThreads + tid;
    const int gstride = static_cast<int>(gridDim.x) * kThreads;

    const uintptr_t addr = reinterpret_cast<uintptr_t>(input);
    const int word_mis = static_cast<int>((addr >> 2) & 3u);
    int head = (4 - word_mis) & 3;
    if (head > n) head = n;

    const int rem = n - head;
    const int n4 = rem >> 2;
    const int tail_start = head + (n4 << 2);
    const float4* __restrict__ in4 = reinterpret_cast<const float4*>(input + head);

    double thread_sq = 0.0;
    int i4 = gtid;

    // 8x-unrolled `float4` (`32` floats = 128 B per thread step), 4 independent FP32 accumulators
    for (; i4 + 7 * gstride < n4; i4 += 8 * gstride) {
        const float4 v0 = __ldg(&in4[i4]);
        const float4 v1 = __ldg(&in4[i4 + gstride]);
        const float4 v2 = __ldg(&in4[i4 + 2 * gstride]);
        const float4 v3 = __ldg(&in4[i4 + 3 * gstride]);
        const float4 v4 = __ldg(&in4[i4 + 4 * gstride]);
        const float4 v5 = __ldg(&in4[i4 + 5 * gstride]);
        const float4 v6 = __ldg(&in4[i4 + 6 * gstride]);
        const float4 v7 = __ldg(&in4[i4 + 7 * gstride]);

        float s0 = fmaf(v0.x, v0.x, v0.y * v0.y) + fmaf(v0.z, v0.z, v0.w * v0.w);
        float s1 = fmaf(v1.x, v1.x, v1.y * v1.y) + fmaf(v1.z, v1.z, v1.w * v1.w);
        float s2 = fmaf(v2.x, v2.x, v2.y * v2.y) + fmaf(v2.z, v2.z, v2.w * v2.w);
        float s3 = fmaf(v3.x, v3.x, v3.y * v3.y) + fmaf(v3.z, v3.z, v3.w * v3.w);

        s0 += fmaf(v4.x, v4.x, v4.y * v4.y) + fmaf(v4.z, v4.z, v4.w * v4.w);
        s1 += fmaf(v5.x, v5.x, v5.y * v5.y) + fmaf(v5.z, v5.z, v5.w * v5.w);
        s2 += fmaf(v6.x, v6.x, v6.y * v6.y) + fmaf(v6.z, v6.z, v6.w * v6.w);
        s3 += fmaf(v7.x, v7.x, v7.y * v7.y) + fmaf(v7.z, v7.z, v7.w * v7.w);

        thread_sq += static_cast<double>((s0 + s1) + (s2 + s3));
    }

    float rem_sq = 0.0f;
    for (; i4 < n4; i4 += gstride) {
        const float4 v = __ldg(&in4[i4]);
        rem_sq += fmaf(v.x, v.x, v.y * v.y) + fmaf(v.z, v.z, v.w * v.w);
    }
    thread_sq += static_cast<double>(rem_sq);

    if (blockIdx.x == 0u) {
        if (tid < head) {
            const float v = __ldg(&input[tid]);
            thread_sq += static_cast<double>(v) * static_cast<double>(v);
        }
        if (tail_start + tid < n) {
            const float v = __ldg(&input[tail_start + tid]);
            thread_sq += static_cast<double>(v) * static_cast<double>(v);
        }
    }

    const double blk_sq = block_reduce_sum_f64(thread_sq, s_warp);

    if (gridDim.x == 1u) {
        if (tid == 0) {
            const float mean_sq = static_cast<float>(blk_sq / static_cast<double>(n));
            g_scale = gamma * rsqrtf(mean_sq + eps);
        }
        return;
    }

    if (tid == 0) {
        g_blk_sq[blockIdx.x] = blk_sq;
        __threadfence();
        const unsigned prev = atomicAdd(&g_retired_blocks, 1u);
        s_is_last = (prev + 1u == gridDim.x);
    }
    __syncthreads();
    if (!s_is_last) {
        return;
    }

    __threadfence();
    __syncthreads();

    const double part = (tid < static_cast<int>(gridDim.x))
                            ? static_cast<const volatile double*>(g_blk_sq)[tid]
                            : 0.0;
    const double total_sq = block_reduce_sum_f64(part, s_warp);
    if (tid == 0) {
        const float mean_sq = static_cast<float>(total_sq / static_cast<double>(n));
        g_scale = gamma * rsqrtf(mean_sq + eps);
        atomicExch(&g_retired_blocks, 0u);
    }
}

// ============================================================================
// 2b. Large-N Pass 2: 4x-Unrolled `float4` FMA + `st.global.cs.v4.f32`
// ============================================================================
__global__ __launch_bounds__(kThreads) void rmsnorm_apply_vec4_kernel(
    const float* __restrict__ input,
    float beta,
    float* __restrict__ output,
    int n) {
    const float scale = g_scale;
    const int gtid = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
    const int gstride = static_cast<int>(gridDim.x) * kThreads;

    if (is_aligned_16b(input) && is_aligned_16b(output)) {
        const int n4 = n >> 2;
        const float4* __restrict__ in4 = reinterpret_cast<const float4*>(input);
        float4* __restrict__ out4 = reinterpret_cast<float4*>(output);

        int i4 = gtid;
        for (; i4 + 3 * gstride < n4; i4 += 4 * gstride) {
            const float4 x0 = __ldg(&in4[i4]);
            const float4 x1 = __ldg(&in4[i4 + gstride]);
            const float4 x2 = __ldg(&in4[i4 + 2 * gstride]);
            const float4 x3 = __ldg(&in4[i4 + 3 * gstride]);

            const float4 y0 = make_float4(fmaf(x0.x, scale, beta), fmaf(x0.y, scale, beta),
                                          fmaf(x0.z, scale, beta), fmaf(x0.w, scale, beta));
            const float4 y1 = make_float4(fmaf(x1.x, scale, beta), fmaf(x1.y, scale, beta),
                                          fmaf(x1.z, scale, beta), fmaf(x1.w, scale, beta));
            const float4 y2 = make_float4(fmaf(x2.x, scale, beta), fmaf(x2.y, scale, beta),
                                          fmaf(x2.z, scale, beta), fmaf(x2.w, scale, beta));
            const float4 y3 = make_float4(fmaf(x3.x, scale, beta), fmaf(x3.y, scale, beta),
                                          fmaf(x3.z, scale, beta), fmaf(x3.w, scale, beta));

            store_cs_f4(&out4[i4], y0);
            store_cs_f4(&out4[i4 + gstride], y1);
            store_cs_f4(&out4[i4 + 2 * gstride], y2);
            store_cs_f4(&out4[i4 + 3 * gstride], y3);
        }
        for (; i4 < n4; i4 += gstride) {
            const float4 x = __ldg(&in4[i4]);
            const float4 y = make_float4(fmaf(x.x, scale, beta), fmaf(x.y, scale, beta),
                                         fmaf(x.z, scale, beta), fmaf(x.w, scale, beta));
            store_cs_f4(&out4[i4], y);
        }
        for (int i = (n4 << 2) + gtid; i < n; i += gstride) {
            output[i] = fmaf(__ldg(&input[i]), scale, beta);
        }
    } else {
        for (int i = gtid; i < n; i += gstride) {
            output[i] = fmaf(__ldg(&input[i]), scale, beta);
        }
    }
}

}  // namespace

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N,
                      float eps) {
    if (N <= 0) {
        return;
    }

    if (N <= kSmallMaxN) {
        rmsnorm_small_single_block_kernel<<<1, kThreads>>>(input, gamma, beta, output, N, eps);
        return;
    }

    // Right-size grid: at N = 100,000, launches 40 blocks (1 block/SM across 40 T4 SMs!)
    int blocks = (N + 2559) / 2560;
    if (blocks < 1) blocks = 1;
    if (blocks > kMaxPersistentBlocks) blocks = kMaxPersistentBlocks;
    const unsigned grid = static_cast<unsigned>(blocks);

    rmsnorm_sumsq_vec4_kernel<<<grid, kThreads>>>(input, gamma, N, eps);
    rmsnorm_apply_vec4_kernel<<<grid, kThreads>>>(input, beta, output, N);
}
