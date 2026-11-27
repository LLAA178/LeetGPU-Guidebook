#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#ifndef MERGE_ITEMS
// 256 threads * 16 items = 4096 floats = exactly 16.0 KiB smem/block (3 blocks/SM = 48 KiB on T4!)
// Also supports -DMERGE_ITEMS=8 (8 KiB/block, 4 blocks/SM) and -DMERGE_ITEMS=12 (12 KiB/block, 4 blocks/SM)
#define MERGE_ITEMS 16
#endif

namespace {

constexpr int kThreads = 256;
constexpr int kItems = MERGE_ITEMS;
static_assert(kItems >= 4 && (kItems % 4) == 0 && kItems <= 20,
              "MERGE_ITEMS must be a multiple of 4 in [4, 20]");
constexpr int kTile = kThreads * kItems;
constexpr int kTileVec4 = kTile / 4;

// Exact Merge-Path co-rank along diagonal `i + j = diag` with tie-breaking `A <= B`.
// Note: whenever `low < high`, `0 <= mid < M` and `0 <= diag - 1 - mid < N` hold
// strictly by loop invariants (`low >= max(0, diag - N)` and `high <= min(diag, M)`),
// eliminating all redundant bounds checks inside the binary search loop!
__device__ __forceinline__ int merge_path_global(const float* __restrict__ A,
                                                 int M,
                                                 const float* __restrict__ B,
                                                 int N,
                                                 int diag) {
    int low = (diag > N) ? (diag - N) : 0;
    int high = (diag < M) ? diag : M;

    while (low < high) {
        const int mid = low + ((high - low) >> 1);
        if (__ldg(&A[mid]) <= __ldg(&B[diag - 1 - mid])) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

__device__ __forceinline__ int merge_path_smem(const float* s_a,
                                               int la,
                                               const float* s_b,
                                               int lb,
                                               int diag) {
    int low = (diag > lb) ? (diag - lb) : 0;
    int high = (diag < la) ? diag : la;

    while (low < high) {
        const int mid = (low + high) >> 1;
        if (s_a[mid] <= s_b[diag - 1 - mid]) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

// Runtime byte-aligned 128-bit `float4` (`__ldg`) cooperative load into shared memory.
// Handles arbitrary start offsets (`a0`, `b0`) and misaligned base pointers safely.
__device__ __forceinline__ void load_range_aligned_vec4(const float* __restrict__ g_src,
                                                        float* s_dst,
                                                        int count) {
    if (count <= 0) {
        return;
    }
    const int tid = threadIdx.x;
    const uintptr_t addr = reinterpret_cast<uintptr_t>(g_src);

    if ((addr & 3u) == 0u && count >= 16) {
        const int head = (4 - static_cast<int>((addr >> 2) & 3u)) & 3;
        const int num_vec4 = (count - head) >> 2;
        const int tail_start = head + (num_vec4 << 2);

        if (tid < head) {
            s_dst[tid] = __ldg(&g_src[tid]);
        }
        if (tail_start + tid < count) {
            s_dst[tail_start + tid] = __ldg(&g_src[tail_start + tid]);
        }

        const float4* __restrict__ vec_src = reinterpret_cast<const float4*>(g_src + head);
        for (int v = tid; v < num_vec4; v += kThreads) {
            const float4 q = __ldg(&vec_src[v]);
            const int d = head + (v << 2);
            s_dst[d + 0] = q.x;
            s_dst[d + 1] = q.y;
            s_dst[d + 2] = q.z;
            s_dst[d + 3] = q.w;
        }
    } else {
        for (int i = tid; i < count; i += kThreads) {
            s_dst[i] = __ldg(&g_src[i]);
        }
    }
}

__device__ __forceinline__ void store_streaming_float4(float4* ptr, const float4& v) {
#ifdef __CUDA_ARCH__
    asm volatile("st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(ptr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
                 : "memory");
#else
    *ptr = v;
#endif
}

// ============================================================================
// Single-Kernel Merge-Path Tile Co-Rank + Register-Cached Shared Merge + float4
// - Zero static `.bss`, zero `cudaMalloc`, 1 single kernel launch!
// - Exact `kTile * 4` bytes of shared memory (16.000 KiB @ MERGE_ITEMS=16 ->
//   3 blocks/SM = 48 KiB on T4!) by reusing `s_buf4` for the 2 block co-ranks.
// - Only 2 threads (warp 0 lane 0 & warp 1 lane 0) run global binary search.
// - Register-cached `(va, vb)` serial merge halves shared-memory reads!
// - Stages sorted items back through `s_buf4` for 100% coalesced 128-bit
//   `st.global.cs.v4.f32` streaming global stores (preserving L2 pivot cache!).
// ============================================================================
__global__ __launch_bounds__(kThreads) void parallel_merge_opt_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N) {
    __shared__ float4 s_buf4[kTileVec4];
    float* s_buf = reinterpret_cast<float*>(s_buf4);
    int* s_corank = reinterpret_cast<int*>(s_buf4);

    const int tid = threadIdx.x;
    const int total = M + N;
    const int c0 = blockIdx.x * kTile;
    if (c0 >= total) {
        return;
    }
    int tile_count = total - c0;
    if (tile_count > kTile) {
        tile_count = kTile;
    }
    const int c1 = c0 + tile_count;

    // 1. Two independent warps compute the block's start and end co-ranks in parallel
    if (tid == 0) {
        s_corank[0] = (c0 == 0) ? 0 : merge_path_global(A, M, B, N, c0);
    }
    if (tid == 32) {
        s_corank[1] = (c1 == total) ? M : merge_path_global(A, M, B, N, c1);
    }
    __syncthreads();

    const int a0 = s_corank[0];
    const int a1 = s_corank[1];
    const int b0 = c0 - a0;
    const int la = a1 - a0;
    const int lb = tile_count - la;
    __syncthreads();

    // 2. Vectorized byte-aligned float4 load of A[a0..a1) and B[b0..b1) into shared memory
    float* s_a = s_buf;
    float* s_b = s_buf + la;
    load_range_aligned_vec4(A + a0, s_a, la);
    load_range_aligned_vec4(B + b0, s_b, lb);
    __syncthreads();

    // 3. Per-thread shared-memory co-rank + register-cached serial merge + coalesced store
    if (tile_count == kTile) {
        const int thread_diag = tid * kItems;
        int a_idx = merge_path_smem(s_a, la, s_b, lb, thread_diag);
        int b_idx = thread_diag - a_idx;

        float va = (a_idx < la) ? s_a[a_idx] : 0.0f;
        float vb = (b_idx < lb) ? s_b[b_idx] : 0.0f;
        float out[kItems];

#pragma unroll
        for (int item = 0; item < kItems; ++item) {
            const bool take_a = (b_idx >= lb) || ((a_idx < la) && (va <= vb));
            out[item] = take_a ? va : vb;
            if (take_a) {
                ++a_idx;
                if (item + 1 < kItems && a_idx < la) {
                    va = s_a[a_idx];
                }
            } else {
                ++b_idx;
                if (item + 1 < kItems && b_idx < lb) {
                    vb = s_b[b_idx];
                }
            }
        }
        __syncthreads();

        // Stage contiguous per-thread chunks as float4 into shared memory
        float4* thread_s4 = &s_buf4[tid * (kItems / 4)];
#pragma unroll
        for (int q = 0; q < kItems / 4; ++q) {
            float4 v;
            v.x = out[4 * q + 0];
            v.y = out[4 * q + 1];
            v.z = out[4 * q + 2];
            v.w = out[4 * q + 3];
            thread_s4[q] = v;
        }
        __syncthreads();

        float* out_ptr = C + c0;
        if ((reinterpret_cast<uintptr_t>(out_ptr) & 15u) == 0u) {
            float4* out_vec4 = reinterpret_cast<float4*>(out_ptr);
#pragma unroll
            for (int v = tid; v < kTileVec4; v += kThreads) {
                store_streaming_float4(&out_vec4[v], s_buf4[v]);
            }
        } else {
#pragma unroll
            for (int i = tid; i < kTile; i += kThreads) {
                out_ptr[i] = s_buf[i];
            }
        }
    } else {
        // Partial final block
        const int thread_diag = (tid * kItems < tile_count) ? (tid * kItems) : tile_count;
        int rem_items = tile_count - thread_diag;
        if (rem_items > kItems) {
            rem_items = kItems;
        }

        int a_idx = merge_path_smem(s_a, la, s_b, lb, thread_diag);
        int b_idx = thread_diag - a_idx;

        float va = (a_idx < la) ? s_a[a_idx] : 0.0f;
        float vb = (b_idx < lb) ? s_b[b_idx] : 0.0f;
        float out[kItems];

#pragma unroll
        for (int item = 0; item < kItems; ++item) {
            if (item < rem_items) {
                const bool take_a = (b_idx >= lb) || ((a_idx < la) && (va <= vb));
                out[item] = take_a ? va : vb;
                if (take_a) {
                    ++a_idx;
                    if (item + 1 < rem_items && a_idx < la) {
                        va = s_a[a_idx];
                    }
                } else {
                    ++b_idx;
                    if (item + 1 < rem_items && b_idx < lb) {
                        vb = s_b[b_idx];
                    }
                }
            } else {
                out[item] = 0.0f;
            }
        }
        __syncthreads();

#pragma unroll
        for (int item = 0; item < kItems; ++item) {
            if (item < rem_items) {
                s_buf[thread_diag + item] = out[item];
            }
        }
        __syncthreads();

        float* out_ptr = C + c0;
        for (int i = tid; i < tile_count; i += kThreads) {
            out_ptr[i] = s_buf[i];
        }
    }
}

}  // namespace

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    const int total = M + N;
    if (total <= 0) {
        return;
    }
    const int blocks = (total + kTile - 1) / kTile;
    parallel_merge_opt_kernel<<<blocks, kThreads>>>(A, B, C, M, N);
}
