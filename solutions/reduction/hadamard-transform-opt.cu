#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ void store_streaming_float4(float4* addr, float4 v) {
    asm volatile("st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(addr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
                 : "memory");
}

__device__ __forceinline__ void hadamard4_in_reg(float4& v) {
    const float s0 = v.x + v.y;
    const float d0 = v.x - v.y;
    const float s1 = v.z + v.w;
    const float d1 = v.z - v.w;
    v.x = s0 + s1;
    v.y = d0 + d1;
    v.z = s0 - s1;
    v.w = d0 - d1;
}

__device__ __forceinline__ void shfl_xor_butterfly_float4(float4& v, int lane_bit, int lane) {
    float4 p;
    p.x = __shfl_xor_sync(kFullMask, v.x, lane_bit);
    p.y = __shfl_xor_sync(kFullMask, v.y, lane_bit);
    p.z = __shfl_xor_sync(kFullMask, v.z, lane_bit);
    p.w = __shfl_xor_sync(kFullMask, v.w, lane_bit);
    if ((lane & lane_bit) == 0) {
        v.x += p.x;
        v.y += p.y;
        v.z += p.z;
        v.w += p.w;
    } else {
        v.x = p.x - v.x;
        v.y = p.y - v.y;
        v.z = p.z - v.z;
        v.w = p.w - v.w;
    }
}

__device__ __forceinline__ float4 add_f4(const float4& a, const float4& b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ __forceinline__ float4 sub_f4(const float4& a, const float4& b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__device__ __forceinline__ float4 scale_f4(const float4& a, float s) {
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

// ============================================================================
// Specialized Kronecker H_4096 = H_64 (x) H_64 Single-Pass Kernel (B x 4,096)
// - 256 threads/block, __launch_bounds__(256, 3) -> 768 threads/SM (75% occ),
//   EXACTLY 16,384 bytes (16.0 KiB) shared memory -> 3 blocks * 16 KiB = 48.0 KiB!
// - Stages h = 1..32 (all 6 stages of row-wise H_64): 100% in registers + half-warp shuffles!
// - Stages h = 64..2048 (all 6 stages of column-wise H_64): radix-8 x radix-8 over float4 columns
//   with ZERO bank conflicts and ONLY TWO __syncthreads() in the entire kernel!
// ============================================================================
__global__ __launch_bounds__(256, 3) void hadamard_4096_kronecker_vec4_kernel(
    const float4* __restrict__ in4,
    float4* __restrict__ out4) {
    __shared__ float4 s_v4[64][16];  // 64 * 16 * 16 B = 16,384 B exact!

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const size_t row_base4 = static_cast<size_t>(blockIdx.x) << 10;  // 1,024 float4s per row

    // 1. Coalesced load of 4 float4s per thread (16 floats) + row-wise H_64 (stages h=1..32)
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int idx4 = m * 256 + tid;
        float4 v = __ldg(&in4[row_base4 + idx4]);

        // h = 1, 2 inside float4
        hadamard4_in_reg(v);
        // h = 4, 8, 16, 32 across 16 contiguous threads (half-warp)
        shfl_xor_butterfly_float4(v, 1, lane);
        shfl_xor_butterfly_float4(v, 2, lane);
        shfl_xor_butterfly_float4(v, 4, lane);
        shfl_xor_butterfly_float4(v, 8, lane);

        s_v4[idx4 >> 4][idx4 & 15] = v;
    }
    __syncthreads();

    // 2. Column-wise stages h = 64, 128, 256 (lower 3 bits of r = 8*r_hi + 0..7)
    // Thread mapping: c4 = tid & 15 (16 contiguous float4s = bank-conflict-free!),
    //                 grp = tid >> 4 (0..15): r_hi = grp >> 1 (0..7), sub = grp & 1 (0..1)
    {
        const int c4 = tid & 15;
        const int grp = tid >> 4;
        const int r_base = (grp >> 1) * 8;
        const int sub = grp & 1;

        float4 a0 = s_v4[r_base + sub + 0][c4];
        float4 a1 = s_v4[r_base + sub + 2][c4];
        float4 a2 = s_v4[r_base + sub + 4][c4];
        float4 a3 = s_v4[r_base + sub + 6][c4];

        // h = 128 (stride 2 in r) and h = 256 (stride 4 in r) in registers:
        const float4 s0 = add_f4(a0, a1);
        const float4 d0 = sub_f4(a0, a1);
        const float4 s1 = add_f4(a2, a3);
        const float4 d1 = sub_f4(a2, a3);

        float4 b0 = add_f4(s0, s1);
        float4 b1 = add_f4(d0, d1);
        float4 b2 = sub_f4(s0, s1);
        float4 b3 = sub_f4(d0, d1);

        // h = 64 (stride 1 in r -> flips sub = grp & 1 -> lane ^ 16 inside the same warp!):
        shfl_xor_butterfly_float4(b0, 16, lane);
        shfl_xor_butterfly_float4(b1, 16, lane);
        shfl_xor_butterfly_float4(b2, 16, lane);
        shfl_xor_butterfly_float4(b3, 16, lane);

        s_v4[r_base + sub + 0][c4] = b0;
        s_v4[r_base + sub + 2][c4] = b1;
        s_v4[r_base + sub + 4][c4] = b2;
        s_v4[r_base + sub + 6][c4] = b3;
    }
    __syncthreads();

    // 3. Column-wise stages h = 512, 1024, 2048 (upper 3 bits of r = r_lo + 8 * {0..7})
    //    + exact 1/sqrt(4096) = 1/64 = 0.015625f scaling + coalesced streaming float4 stores!
    {
        constexpr float kScale4096 = 1.0f / 64.0f;
        const int c4 = tid & 15;
        const int grp = tid >> 4;
        const int r_lo = grp >> 1;      // 0..7
        const int sub = grp & 1;        // 0..1

        const int r0 = r_lo + (sub + 0) * 8;
        const int r1 = r_lo + (sub + 2) * 8;
        const int r2 = r_lo + (sub + 4) * 8;
        const int r3 = r_lo + (sub + 6) * 8;

        float4 a0 = s_v4[r0][c4];
        float4 a1 = s_v4[r1][c4];
        float4 a2 = s_v4[r2][c4];
        float4 a3 = s_v4[r3][c4];

        // h = 1024 (stride 16 in r) and h = 2048 (stride 32 in r) in registers:
        const float4 s0 = add_f4(a0, a1);
        const float4 d0 = sub_f4(a0, a1);
        const float4 s1 = add_f4(a2, a3);
        const float4 d1 = sub_f4(a2, a3);

        float4 b0 = add_f4(s0, s1);
        float4 b1 = add_f4(d0, d1);
        float4 b2 = sub_f4(s0, s1);
        float4 b3 = sub_f4(d0, d1);

        // h = 512 (stride 8 in r -> flips sub -> lane ^ 16 inside the same warp!):
        shfl_xor_butterfly_float4(b0, 16, lane);
        shfl_xor_butterfly_float4(b1, 16, lane);
        shfl_xor_butterfly_float4(b2, 16, lane);
        shfl_xor_butterfly_float4(b3, 16, lane);

        // Note c4 = 0..15 across adjacent lanes -> 16 threads write 64 contiguous floats (256 B coalesced)!
        store_streaming_float4(&out4[row_base4 + (r0 << 4) + c4], scale_f4(b0, kScale4096));
        store_streaming_float4(&out4[row_base4 + (r1 << 4) + c4], scale_f4(b1, kScale4096));
        store_streaming_float4(&out4[row_base4 + (r2 << 4) + c4], scale_f4(b2, kScale4096));
        store_streaming_float4(&out4[row_base4 + (r3 << 4) + c4], scale_f4(b3, kScale4096));
    }
}

// ============================================================================
// Zero-Shared-Memory Pure Warp-Shuffle Batch Kernel for 1 <= N <= 32
// Packs multiple rows per 256-thread block so wide-batch small-N shapes saturate SMs!
// ============================================================================
__global__ __launch_bounds__(256, 4) void hadamard_tiny_warp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B,
    int N,
    float scale) {
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(B) * static_cast<size_t>(N);
    const int lane = threadIdx.x & 31;

    float val = (gid < total) ? __ldg(&input[gid]) : 0.0f;
    const int elem_in_row = lane & (N - 1);

    for (int h = 1; h < N; h <<= 1) {
        const float partner = __shfl_xor_sync(kFullMask, val, h);
        val = ((elem_in_row & h) == 0) ? (val + partner) : (partner - val);
    }

    if (gid < total) {
        output[gid] = val * scale;
    }
}

// ============================================================================
// Single-Pass Shared-Memory + Warp-Shuffle FWHT for any power-of-two 64 <= N <= 8,192
// Works for both aligned and misaligned pointers!
// ============================================================================
template <int kThreads>
__global__ __launch_bounds__(kThreads) void hadamard_smem_1pass_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    float scale) {
    extern __shared__ float s_data[];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const size_t row_base = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(N);
    const float* __restrict__ in_row = input + row_base;
    float* __restrict__ out_row = output + row_base;

    // 1. Load row into shared memory while fusing stages h = 1..16 inside each warp!
    for (int base = 0; base < N; base += kThreads) {
        const int idx = base + tid;
        if (idx < N) {
            float v = __ldg(&in_row[idx]);
            #pragma unroll
            for (int h = 1; h <= 16; h <<= 1) {
                const float p = __shfl_xor_sync(kFullMask, v, h);
                v = ((lane & h) == 0) ? (v + p) : (p - v);
            }
            s_data[idx] = v;
        }
    }
    __syncthreads();

    // 2. Shared-memory butterfly stages for h = 32 .. N/2
    const int half_n = N >> 1;
    for (int h = 32; h < N; h <<= 1) {
        for (int p = tid; p < half_n; p += kThreads) {
            // Since h is a power of two >= 32: j = ((p & ~(h - 1)) << 1) | (p & (h - 1))
            const int low = p & (h - 1);
            const int j = ((p - low) << 1) | low;
            const float u = s_data[j];
            const float w = s_data[j + h];
            s_data[j] = u + w;
            s_data[j + h] = u - w;
        }
        __syncthreads();
    }

    // 3. Scale and write back to global memory
    for (int idx = tid; idx < N; idx += kThreads) {
        out_row[idx] = s_data[idx] * scale;
    }
}

// ============================================================================
// Multi-Stage Fallback for Ultra-Wide Rows (N >= 16,384, beyond 48 KiB default smem):
// Fuses 10 butterfly stages (H_1024) per shared-memory pass so even N = 32,768 or 65,536
// completes in just 2 passes using output itself for in-place updates (zero cudaMalloc!).
// ============================================================================
__global__ __launch_bounds__(256, 4) void hadamard_chunk1024_pass_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int N,
    float scale) {
    __shared__ float s_chunk[1024];

    const int chunk_start = blockIdx.x * 1024;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const size_t base = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(chunk_start);

    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int idx = m * 256 + tid;
        float v = __ldg(&src[base + idx]);
        #pragma unroll
        for (int h = 1; h <= 16; h <<= 1) {
            const float p = __shfl_xor_sync(kFullMask, v, h);
            v = ((lane & h) == 0) ? (v + p) : (p - v);
        }
        s_chunk[idx] = v;
    }
    __syncthreads();

    for (int h = 32; h < 1024; h <<= 1) {
        for (int p = tid; p < 512; p += 256) {
            const int low = p & (h - 1);
            const int j = ((p - low) << 1) | low;
            const float u = s_chunk[j];
            const float w = s_chunk[j + h];
            s_chunk[j] = u + w;
            s_chunk[j + h] = u - w;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int idx = m * 256 + tid;
        dst[base + idx] = s_chunk[idx] * scale;
    }
}

__global__ __launch_bounds__(256, 4) void hadamard_inplace_global_stage_kernel(
    float* __restrict__ data,
    int B,
    int N,
    int h,
    float post_scale) {
    const size_t pair_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t half_n = static_cast<size_t>(N >> 1);
    const size_t total_pairs = static_cast<size_t>(B) * half_n;
    if (pair_idx >= total_pairs) {
        return;
    }

    const size_t b = pair_idx / half_n;
    const int p = static_cast<int>(pair_idx % half_n);
    const int low = p & (h - 1);
    const int j = ((p - low) << 1) | low;

    const size_t base = b * static_cast<size_t>(N);
    const float u = data[base + j];
    const float w = data[base + j + h];
    data[base + j] = (u + w) * post_scale;
    data[base + j + h] = (u - w) * post_scale;
}

inline bool is_aligned_16(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 15u) == 0u;
}

}  // namespace

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int B, int N) {
    if (B <= 0 || N <= 0) {
        return;
    }

    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(N)));

    // 1. Specialized zero-smem-overhead Kronecker H_64 (x) H_64 fast path for official N = 4,096
    if (N == 4096 && is_aligned_16(input) && is_aligned_16(output)) {
        hadamard_4096_kronecker_vec4_kernel<<<B, 256>>>(
            reinterpret_cast<const float4*>(input),
            reinterpret_cast<float4*>(output));
        return;
    }

    // 2. Pure warp-shuffle multi-row kernel for N in {1, 2, 4, 8, 16, 32}
    if (N <= 32) {
        const size_t total = static_cast<size_t>(B) * static_cast<size_t>(N);
        const int blocks = static_cast<int>((total + 255) / 256);
        hadamard_tiny_warp_kernel<<<blocks, 256>>>(input, output, B, N, scale);
        return;
    }

    // 3. Single-pass shared-memory FWHT for N in {64, 128, 256, 512, 1024, 2048, 4096(unaligned), 8192}
    if (N <= 8192) {
        const size_t smem_bytes = static_cast<size_t>(N) * sizeof(float);
        if (N <= 256) {
            hadamard_smem_1pass_kernel<64><<<B, 64, smem_bytes>>>(input, output, N, scale);
        } else if (N <= 2048) {
            hadamard_smem_1pass_kernel<256><<<B, 256, smem_bytes>>>(input, output, N, scale);
        } else {
            hadamard_smem_1pass_kernel<512><<<B, 512, smem_bytes>>>(input, output, N, scale);
        }
        return;
    }

    // 4. Zero-workspace fallback for N >= 16,384: fuse stages h = 1..512 into output first,
    //    then run remaining stages h = 1024 .. N/2 in-place on output!
    const dim3 chunk_grid(static_cast<unsigned>(N / 1024), static_cast<unsigned>(B));
    hadamard_chunk1024_pass_kernel<<<chunk_grid, 256>>>(input, output, N, 1.0f);

    const size_t total_pairs = static_cast<size_t>(B) * static_cast<size_t>(N >> 1);
    const int stage_blocks = static_cast<int>((total_pairs + 255) / 256);
    for (int h = 1024; h < N; h <<= 1) {
        const float post_scale = ((h << 1) == N) ? scale : 1.0f;
        hadamard_inplace_global_stage_kernel<<<stage_blocks, 256>>>(output, B, N, h, post_scale);
    }
}
