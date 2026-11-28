#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#ifndef COUNT_MAX_BLOCKS
#define COUNT_MAX_BLOCKS 640
#endif

#ifndef COUNT_USE_CS
#define COUNT_USE_CS 0
#endif

#ifndef COUNT_CONTIGUOUS
#define COUNT_CONTIGUOUS 0
#endif

namespace {

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarp;
constexpr int kMaxBlocks = COUNT_MAX_BLOCKS;
static_assert(kMaxBlocks >= 1 && kMaxBlocks <= 1024, "COUNT_MAX_BLOCKS must be in [1, 1024]");
constexpr unsigned kFullMask = 0xffffffffu;

// Tiny 8.0 KB static .bss (referenced directly in device code, never passed from host)
__device__ long long g_partial_counts[1024];
__device__ unsigned int g_retired = 0u;

__device__ __forceinline__ int4 load_vec4(const int4* ptr) {
#if COUNT_USE_CS && defined(__CUDA_ARCH__)
    int4 ret;
    asm volatile("ld.global.cs.v4.s32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
                 : "l"(ptr));
    return ret;
#else
    return __ldg(ptr);
#endif
}

__device__ __forceinline__ int warp_reduce_sum_i32(int val) {
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(kFullMask, val, offset);
    }
    return val;
}

__device__ __forceinline__ long long warp_reduce_sum_i64(long long val) {
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(kFullMask, val, offset);
    }
    return val;
}

__device__ __forceinline__ int count_eq_int4(const int4& v, int target) {
    return static_cast<int>(v.x == target) + static_cast<int>(v.y == target) +
           static_cast<int>(v.z == target) + static_cast<int>(v.w == target);
}

// ============================================================================
// Flattened 2D Contiguous-Tile Streaming `ld.global.cs.v4.s32` * 8 + Retire!
// - Flattens row-major `(N, M)` matrix into `total_elems = N * M` ints (400 MB
//   at `10,000 x 10,000`), peeling 0..3 `head_elems` for strict 16B alignment.
// - Assigns each block a CONTIGUOUS 256-aligned range `[b_start, b_end)` so
//   every 8x-unrolled iteration sweeps a sequential `32 KiB` slice inside one
//   TLB page & GDDR6 DRAM row without multi-wave strided row-buffer thrashing!
// - Uses `ld.global.cs.v4.s32` streaming loads, accumulates branchlessly in
//   32-bit registers, reduces via warp shuffles once at exit, and retires the
//   last block to overwrite `output[0]` in ONE SINGLE KERNEL LAUNCH!
// ============================================================================
__global__ __launch_bounds__(kThreads) void count_2d_aligned_vec4_kernel(
    const int* __restrict__ input,
    int* __restrict__ output,
    uint32_t head_elems,
    uint32_t num_vec4,
    uint32_t total_elems,
    int target) {
    __shared__ long long s_warp[kWarps];
    __shared__ bool s_is_last;

    const uint32_t tid = threadIdx.x;
    const int lane = static_cast<int>(tid & 31u);
    const int warp = static_cast<int>(tid >> 5);
    const uint32_t num_blocks = gridDim.x;

    int acc = 0;

    if (blockIdx.x == 0u) {
        if (tid < head_elems) {
            acc += static_cast<int>(__ldg(&input[tid]) == target);
        }
        const uint32_t tail_start = head_elems + (num_vec4 << 2);
        const uint32_t tail_idx = tail_start + tid;
        if (tail_idx < total_elems) {
            acc += static_cast<int>(__ldg(&input[tail_idx]) == target);
        }
    }

    const int4* __restrict__ vec_ptr = reinterpret_cast<const int4*>(input + head_elems);

#if COUNT_CONTIGUOUS
    const uint32_t tiles = (num_vec4 + 255u) >> 8;
    const uint32_t tiles_per_block = (tiles + num_blocks - 1u) / num_blocks;
    const uint32_t chunk_vec4 = tiles_per_block << 8;
    const uint32_t b_start = blockIdx.x * chunk_vec4;
    uint32_t b_end = b_start + chunk_vec4;
    if (b_end > num_vec4) {
        b_end = num_vec4;
    }

    if (b_start < b_end) {
        const uint32_t step = static_cast<uint32_t>(kThreads);
        const uint32_t step8 = step * 8u;
        uint32_t idx = b_start + tid;

        for (; idx + 7u * step < b_end; idx += step8) {
            const int4 v0 = load_vec4(&vec_ptr[idx]);
            const int4 v1 = load_vec4(&vec_ptr[idx + step]);
            const int4 v2 = load_vec4(&vec_ptr[idx + 2u * step]);
            const int4 v3 = load_vec4(&vec_ptr[idx + 3u * step]);
            const int4 v4 = load_vec4(&vec_ptr[idx + 4u * step]);
            const int4 v5 = load_vec4(&vec_ptr[idx + 5u * step]);
            const int4 v6 = load_vec4(&vec_ptr[idx + 6u * step]);
            const int4 v7 = load_vec4(&vec_ptr[idx + 7u * step]);

            acc += (count_eq_int4(v0, target) + count_eq_int4(v1, target)) +
                   (count_eq_int4(v2, target) + count_eq_int4(v3, target)) +
                   (count_eq_int4(v4, target) + count_eq_int4(v5, target)) +
                   (count_eq_int4(v6, target) + count_eq_int4(v7, target));
        }
        for (; idx < b_end; idx += step) {
            acc += count_eq_int4(load_vec4(&vec_ptr[idx]), target);
        }
    }
#else
    const uint32_t stride = num_blocks * static_cast<uint32_t>(kThreads);
    const uint32_t stride8 = stride * 8u;
    uint32_t idx = blockIdx.x * static_cast<uint32_t>(kThreads) + tid;

    for (; idx + 7u * stride < num_vec4; idx += stride8) {
        const int4 v0 = load_vec4(&vec_ptr[idx]);
        const int4 v1 = load_vec4(&vec_ptr[idx + stride]);
        const int4 v2 = load_vec4(&vec_ptr[idx + 2u * stride]);
        const int4 v3 = load_vec4(&vec_ptr[idx + 3u * stride]);
        const int4 v4 = load_vec4(&vec_ptr[idx + 4u * stride]);
        const int4 v5 = load_vec4(&vec_ptr[idx + 5u * stride]);
        const int4 v6 = load_vec4(&vec_ptr[idx + 6u * stride]);
        const int4 v7 = load_vec4(&vec_ptr[idx + 7u * stride]);

        acc += (count_eq_int4(v0, target) + count_eq_int4(v1, target)) +
               (count_eq_int4(v2, target) + count_eq_int4(v3, target)) +
               (count_eq_int4(v4, target) + count_eq_int4(v5, target)) +
               (count_eq_int4(v6, target) + count_eq_int4(v7, target));
    }
    for (; idx < num_vec4; idx += stride) {
        acc += count_eq_int4(load_vec4(&vec_ptr[idx]), target);
    }
#endif

    const int warp_sum = warp_reduce_sum_i32(acc);
    if (lane == 0) {
        s_warp[warp] = static_cast<long long>(warp_sum);
    }
    __syncthreads();

    if (warp == 0) {
        long long bsum = (lane < kWarps) ? s_warp[lane] : 0;
        bsum = warp_reduce_sum_i64(bsum);
        if (lane == 0) {
            if (num_blocks == 1u) {
                output[0] = static_cast<int>(bsum);
                s_is_last = false;
            } else {
                static_cast<volatile long long*>(g_partial_counts)[blockIdx.x] = bsum;
                __threadfence();
                const unsigned int prev = atomicAdd(&g_retired, 1u);
                s_is_last = (prev == num_blocks - 1u);
            }
        }
    }
    __syncthreads();

    if (s_is_last) {
        __threadfence();
        const volatile long long* v_partials = g_partial_counts;
        long long sum = 0;
        for (uint32_t i = tid; i < num_blocks; i += static_cast<uint32_t>(kThreads)) {
            sum += v_partials[i];
        }
        sum = warp_reduce_sum_i64(sum);
        if (lane == 0) {
            s_warp[warp] = sum;
        }
        __syncthreads();

        if (warp == 0) {
            long long total = (lane < kWarps) ? s_warp[lane] : 0;
            total = warp_reduce_sum_i64(total);
            if (lane == 0) {
                atomicExch(&g_retired, 0u);
                output[0] = static_cast<int>(total);
            }
        }
    }
}

__global__ __launch_bounds__(kThreads) void count_2d_scalar_u64_kernel(
    const int* __restrict__ input,
    int* __restrict__ output,
    size_t total_elems,
    int target) {
    __shared__ long long s_warp[kWarps];
    __shared__ bool s_is_last;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_blocks = gridDim.x;

    long long acc = 0;
    const size_t stride = static_cast<size_t>(num_blocks) * kThreads;
    for (size_t i = static_cast<size_t>(blockIdx.x) * kThreads + static_cast<size_t>(tid);
         i < total_elems;
         i += stride) {
        acc += static_cast<long long>(__ldg(&input[i]) == target);
    }

    acc = warp_reduce_sum_i64(acc);
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        long long bsum = (lane < kWarps) ? s_warp[lane] : 0;
        bsum = warp_reduce_sum_i64(bsum);
        if (lane == 0) {
            if (num_blocks == 1) {
                output[0] = static_cast<int>(bsum);
                s_is_last = false;
            } else {
                static_cast<volatile long long*>(g_partial_counts)[blockIdx.x] = bsum;
                __threadfence();
                const unsigned int prev = atomicAdd(&g_retired, 1u);
                s_is_last = (prev == static_cast<unsigned int>(num_blocks - 1));
            }
        }
    }
    __syncthreads();

    if (s_is_last) {
        __threadfence();
        const volatile long long* v_partials = g_partial_counts;
        long long sum = 0;
        for (int i = tid; i < num_blocks; i += kThreads) {
            sum += v_partials[i];
        }
        sum = warp_reduce_sum_i64(sum);
        if (lane == 0) {
            s_warp[warp] = sum;
        }
        __syncthreads();

        if (warp == 0) {
            long long total = (lane < kWarps) ? s_warp[lane] : 0;
            total = warp_reduce_sum_i64(total);
            if (lane == 0) {
                atomicExch(&g_retired, 0u);
                output[0] = static_cast<int>(total);
            }
        }
    }
}

}  // namespace

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    if (N <= 0 || M <= 0) {
        return;
    }

    const size_t total_elems = static_cast<size_t>(N) * static_cast<size_t>(M);
    const uintptr_t addr = reinterpret_cast<uintptr_t>(input);

    if ((addr & 3u) == 0u && total_elems <= 0xf0000000ull) {
        uint32_t head_elems =
            static_cast<uint32_t>((4u - static_cast<unsigned>((addr >> 2) & 3u)) & 3u);
        const uint32_t total_u32 = static_cast<uint32_t>(total_elems);
        if (head_elems > total_u32) {
            head_elems = total_u32;
        }
        const uint32_t num_vec4 = (total_u32 - head_elems) >> 2;
        uint32_t blocks = (num_vec4 + 2047u) / 2048u;
        if (blocks < 1u) blocks = 1u;
        if (blocks > static_cast<uint32_t>(kMaxBlocks)) blocks = static_cast<uint32_t>(kMaxBlocks);

        count_2d_aligned_vec4_kernel<<<blocks, kThreads>>>(
            input, output, head_elems, num_vec4, total_u32, K);
    } else {
        size_t blocks_sz = (total_elems + 2047ull) / 2048ull;
        if (blocks_sz < 1ull) blocks_sz = 1ull;
        if (blocks_sz > static_cast<size_t>(kMaxBlocks)) blocks_sz = static_cast<size_t>(kMaxBlocks);
        const int blocks = static_cast<int>(blocks_sz);

        count_2d_scalar_u64_kernel<<<blocks, kThreads>>>(input, output, total_elems, K);
    }
}
