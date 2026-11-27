#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#ifndef SUB3D_MAX_BLOCKS
#define SUB3D_MAX_BLOCKS 320
#endif

namespace {

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarp;
constexpr int kMaxBlocks = SUB3D_MAX_BLOCKS;
static_assert(kMaxBlocks >= 1 && kMaxBlocks <= 2048, "SUB3D_MAX_BLOCKS must be in [1, 2048]");
constexpr unsigned kFullMask = 0xffffffffu;

__device__ long long g_partial_sums[kMaxBlocks];
__device__ unsigned int g_retired = 0u;

__device__ __forceinline__ long long warp_reduce_sum_i64(long long val) {
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(kFullMask, val, offset);
    }
    return val;
}

__device__ __forceinline__ long long sum_int4(const int4& v) {
    return static_cast<long long>(v.x) + static_cast<long long>(v.y) +
           static_cast<long long>(v.z) + static_cast<long long>(v.w);
}

// Shared block-reduction + single-kernel last-block atomic retirement helper.
// MUST be called uniformly by all kThreads threads in the block.
__device__ __forceinline__ void reduce_block_and_retire(long long thread_acc,
                                                        int* __restrict__ output,
                                                        long long* s_warp,
                                                        bool* s_is_last) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_blocks = gridDim.x;

    long long acc = warp_reduce_sum_i64(thread_acc);
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
                *s_is_last = false;
            } else {
                static_cast<volatile long long*>(g_partial_sums)[blockIdx.x] = bsum;
                __threadfence();
                const unsigned int prev = atomicAdd(&g_retired, 1u);
                *s_is_last = (prev == static_cast<unsigned int>(num_blocks - 1));
            }
        }
    }
    __syncthreads();

    if (*s_is_last) {
        __threadfence();
        const volatile long long* v_partials = g_partial_sums;
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

// ============================================================================
// Path 1: Fully Collapsed 1D Contiguous Subarray (8x int4 unrolled streaming)
// Triggered when the 3D box collapses to a single contiguous 1D range.
// ============================================================================
__global__ __launch_bounds__(kThreads) void subarray_sum_3d_1d_vec4_kernel(
    const int* __restrict__ sub_in,
    int* __restrict__ output,
    int head_elems,
    int num_vec4,
    int len) {
    __shared__ long long s_warp[kWarps];
    __shared__ bool s_is_last;

    const int tid = threadIdx.x;
    const int num_blocks = gridDim.x;
    long long acc = 0;

    if (blockIdx.x == 0) {
        if (tid < head_elems) {
            acc += static_cast<long long>(__ldg(&sub_in[tid]));
        }
        const int tail_start = head_elems + (num_vec4 << 2);
        const int tail_idx = tail_start + tid;
        if (tail_idx < len) {
            acc += static_cast<long long>(__ldg(&sub_in[tail_idx]));
        }
    }

    const int4* __restrict__ vec_ptr = reinterpret_cast<const int4*>(sub_in + head_elems);
    const int stride = num_blocks * kThreads;
    const int stride8 = stride * 8;
    int idx = blockIdx.x * kThreads + tid;

    for (; idx + 7 * stride < num_vec4; idx += stride8) {
        const int4 v0 = __ldg(&vec_ptr[idx]);
        const int4 v1 = __ldg(&vec_ptr[idx + stride]);
        const int4 v2 = __ldg(&vec_ptr[idx + 2 * stride]);
        const int4 v3 = __ldg(&vec_ptr[idx + 3 * stride]);
        const int4 v4 = __ldg(&vec_ptr[idx + 4 * stride]);
        const int4 v5 = __ldg(&vec_ptr[idx + 5 * stride]);
        const int4 v6 = __ldg(&vec_ptr[idx + 6 * stride]);
        const int4 v7 = __ldg(&vec_ptr[idx + 7 * stride]);

        acc += (sum_int4(v0) + sum_int4(v1)) + (sum_int4(v2) + sum_int4(v3)) +
               (sum_int4(v4) + sum_int4(v5)) + (sum_int4(v6) + sum_int4(v7));
    }
    for (; idx < num_vec4; idx += stride) {
        acc += sum_int4(__ldg(&vec_ptr[idx]));
    }

    reduce_block_and_retire(acc, output, s_warp, &s_is_last);
}

// ============================================================================
// Path 2: Group-per-Row-Segment Byte-Aligned int4*4 Streaming (Zero Inner Div!)
// - kGroupSize = 32 (full warp per segment) for C >= 64 (incl. perf C = 489!)
// - kGroupSize = 8  (sub-warp per segment)  for 16 <= C < 64
// - Dynamically peels 0..3 head ints per segment using runtime byte address so
//   vec_ptr is GUARANTEED 16-byte aligned for every S_COL and K!
// - Advances segment pointer across (D, R) using precomputed (ptr_step, r_step,
//   wrap_extra) — ZERO integer division or modulo inside the segment loop!
// ============================================================================
template <int kGroupSize>
__global__ __launch_bounds__(kThreads) void subarray_sum_3d_group_vec4_kernel(
    const int* __restrict__ base_ptr,
    int* __restrict__ output,
    int D,
    int R,
    int C,
    long long stride_d,
    long long stride_r,
    long long ptr_step,
    long long wrap_extra,
    int r_step) {
    static_assert(kGroupSize == 32 || kGroupSize == 8, "kGroupSize must be 32 or 8");
    constexpr int kGroupsPerBlock = kThreads / kGroupSize;

    __shared__ long long s_warp[kWarps];
    __shared__ bool s_is_last;

    const int tid = threadIdx.x;
    const int sub_lane = tid & (kGroupSize - 1);
    const int group_in_block = tid / kGroupSize;
    const int total_segs = D * R;
    const int group_global = blockIdx.x * kGroupsPerBlock + group_in_block;
    const int group_stride = gridDim.x * kGroupsPerBlock;

    long long acc = 0;

    if (group_global < total_segs) {
        const int d_init = (D == 1) ? 0 : (group_global / R);
        int r_off = (D == 1) ? group_global : (group_global - d_init * R);
        long long seg_offset =
            static_cast<long long>(d_init) * stride_d + static_cast<long long>(r_off) * stride_r;

        for (int seg_idx = group_global; seg_idx < total_segs; seg_idx += group_stride) {
            const int* __restrict__ seg_ptr = base_ptr + seg_offset;
            const uintptr_t addr = reinterpret_cast<uintptr_t>(seg_ptr);
            int head_elems = (4 - static_cast<int>((addr >> 2) & 3u)) & 3;
            if (head_elems > C) {
                head_elems = C;
            }
            const int num_vec4 = (C - head_elems) >> 2;
            const int tail_start = head_elems + (num_vec4 << 2);

            if (sub_lane < head_elems) {
                acc += static_cast<long long>(__ldg(&seg_ptr[sub_lane]));
            }
            if (tail_start + sub_lane < C) {
                acc += static_cast<long long>(__ldg(&seg_ptr[tail_start + sub_lane]));
            }

            const int4* __restrict__ vec_ptr = reinterpret_cast<const int4*>(seg_ptr + head_elems);
            int base_v = 0;
            for (; base_v + 4 * kGroupSize <= num_vec4; base_v += 4 * kGroupSize) {
                const int4 q0 = __ldg(&vec_ptr[base_v + sub_lane]);
                const int4 q1 = __ldg(&vec_ptr[base_v + kGroupSize + sub_lane]);
                const int4 q2 = __ldg(&vec_ptr[base_v + 2 * kGroupSize + sub_lane]);
                const int4 q3 = __ldg(&vec_ptr[base_v + 3 * kGroupSize + sub_lane]);
                acc += (sum_int4(q0) + sum_int4(q1)) + (sum_int4(q2) + sum_int4(q3));
            }

            const int rem = num_vec4 - base_v;
            if (rem >= 3 * kGroupSize) {
                const int4 q0 = __ldg(&vec_ptr[base_v + sub_lane]);
                const int4 q1 = __ldg(&vec_ptr[base_v + kGroupSize + sub_lane]);
                const int4 q2 = __ldg(&vec_ptr[base_v + 2 * kGroupSize + sub_lane]);
                int4 q3 = {0, 0, 0, 0};
                if (sub_lane < rem - 3 * kGroupSize) {
                    q3 = __ldg(&vec_ptr[base_v + 3 * kGroupSize + sub_lane]);
                }
                acc += (sum_int4(q0) + sum_int4(q1)) + (sum_int4(q2) + sum_int4(q3));
            } else if (rem >= 2 * kGroupSize) {
                const int4 q0 = __ldg(&vec_ptr[base_v + sub_lane]);
                const int4 q1 = __ldg(&vec_ptr[base_v + kGroupSize + sub_lane]);
                int4 q2 = {0, 0, 0, 0};
                if (sub_lane < rem - 2 * kGroupSize) {
                    q2 = __ldg(&vec_ptr[base_v + 2 * kGroupSize + sub_lane]);
                }
                acc += (sum_int4(q0) + sum_int4(q1)) + sum_int4(q2);
            } else if (rem >= kGroupSize) {
                const int4 q0 = __ldg(&vec_ptr[base_v + sub_lane]);
                int4 q1 = {0, 0, 0, 0};
                if (sub_lane < rem - kGroupSize) {
                    q1 = __ldg(&vec_ptr[base_v + kGroupSize + sub_lane]);
                }
                acc += sum_int4(q0) + sum_int4(q1);
            } else if (sub_lane < rem) {
                acc += sum_int4(__ldg(&vec_ptr[base_v + sub_lane]));
            }

            seg_offset += ptr_step;
            r_off += r_step;
            if (r_off >= R) {
                r_off -= R;
                seg_offset += wrap_extra;
            }
        }
    }

    reduce_block_and_retire(acc, output, s_warp, &s_is_last);
}

// ============================================================================
// Path 3: Narrow-Column / Non-4B-Aligned Scalar Fallback (1 thread per segment)
// Handles C < 16 (e.g. C = 1 column slice) with zero wasted lanes or inner div!
// ============================================================================
__global__ __launch_bounds__(kThreads) void subarray_sum_3d_narrow_scalar_kernel(
    const int* __restrict__ base_ptr,
    int* __restrict__ output,
    int D,
    int R,
    int C,
    long long stride_d,
    long long stride_r,
    long long ptr_step,
    long long wrap_extra,
    int r_step) {
    __shared__ long long s_warp[kWarps];
    __shared__ bool s_is_last;

    const int tid = threadIdx.x;
    const int total_segs = D * R;
    const int global_tid = blockIdx.x * kThreads + tid;
    const int grid_stride = gridDim.x * kThreads;

    long long acc = 0;
    if (global_tid < total_segs) {
        const int d_init = (D == 1) ? 0 : (global_tid / R);
        int r_off = (D == 1) ? global_tid : (global_tid - d_init * R);
        long long seg_offset =
            static_cast<long long>(d_init) * stride_d + static_cast<long long>(r_off) * stride_r;

        for (int seg_idx = global_tid; seg_idx < total_segs; seg_idx += grid_stride) {
            const int* __restrict__ seg_ptr = base_ptr + seg_offset;
            for (int c = 0; c < C; ++c) {
                acc += static_cast<long long>(__ldg(&seg_ptr[c]));
            }
            seg_offset += ptr_step;
            r_off += r_step;
            if (r_off >= R) {
                r_off -= R;
                seg_offset += wrap_extra;
            }
        }
    }

    reduce_block_and_retire(acc, output, s_warp, &s_is_last);
}

}  // namespace

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input,
                      int* output,
                      int N,
                      int M,
                      int K,
                      int S_DEP,
                      int E_DEP,
                      int S_ROW,
                      int E_ROW,
                      int S_COL,
                      int E_COL) {
    if (N <= 0 || M <= 0 || K <= 0 ||
        S_DEP < 0 || E_DEP < S_DEP || E_DEP >= N ||
        S_ROW < 0 || E_ROW < S_ROW || E_ROW >= M ||
        S_COL < 0 || E_COL < S_COL || E_COL >= K) {
        return;
    }

    int D = E_DEP - S_DEP + 1;
    int R = E_ROW - S_ROW + 1;
    int C = E_COL - S_COL + 1;

    long long stride_d = static_cast<long long>(M) * K;
    long long stride_r = static_cast<long long>(K);
    const int* base_ptr =
        input + (static_cast<long long>(S_DEP) * M + S_ROW) * K + S_COL;

    // 1. Canonical contiguous dimension collapsing:
    // - If S_COL..E_COL covers all K columns AND (D == 1 or S_ROW..E_ROW covers all M rows),
    //   the entire 3D box collapses into ONE single 1D contiguous subarray!
    // - Also if S_COL..E_COL covers all K columns and K < 128, merge R into C so narrow K
    //   becomes wide R * K segments.
    if (S_COL == 0 && E_COL == K - 1) {
        if (D == 1 && static_cast<long long>(R) * C <= 1000000000ll) {
            C = R * C;
            R = 1;
            stride_r = stride_d;
        } else if (S_ROW == 0 && E_ROW == M - 1 &&
                   static_cast<long long>(D) * R * C <= 1000000000ll) {
            C = D * R * C;
            D = 1;
            R = 1;
        } else if (K < 128 && static_cast<long long>(R) * C <= 1000000000ll) {
            C = R * C;
            R = 1;
            stride_r = stride_d;
        }
    }

    // - Whenever R * stride_r == stride_d (e.g., S_ROW == 0 && E_ROW == M - 1 in the
    //   official 500x500x500 perf benchmark!), adjacent depths have uniform row stride,
    //   so (D, R) collapses into (1, D * R) with zero depth-wrap jumps!
    if (D > 1 && static_cast<long long>(R) * stride_r == stride_d &&
        static_cast<long long>(D) * R <= 1000000000ll) {
        R = D * R;
        D = 1;
    } else if (D > 1 && R == 1) {
        R = D;
        D = 1;
        stride_r = stride_d;
    }

    const uintptr_t addr = reinterpret_cast<uintptr_t>(base_ptr);
    const bool aligned_4b = ((addr & 3u) == 0u);

    // Dispatch Path 1: Single 1D Contiguous Range
    if (D == 1 && R == 1 && aligned_4b) {
        int head_elems = (4 - static_cast<int>((addr >> 2) & 3u)) & 3;
        if (head_elems > C) {
            head_elems = C;
        }
        const int num_vec4 = (C - head_elems) >> 2;
        int blocks = (num_vec4 + 2047) / 2048;
        if (blocks < 1) blocks = 1;
        if (blocks > kMaxBlocks) blocks = kMaxBlocks;

        subarray_sum_3d_1d_vec4_kernel<<<blocks, kThreads>>>(
            base_ptr, output, head_elems, num_vec4, C);
        return;
    }

    const int total_segs = D * R;

    // Dispatch Path 2a: Full Warp (32 lanes) per Row Segment for C >= 64
    if (aligned_4b && C >= 64) {
        int blocks = (total_segs + (kWarps * 4 - 1)) / (kWarps * 4);
        if (blocks < 1) blocks = 1;
        if (blocks > kMaxBlocks) blocks = kMaxBlocks;

        const int total_groups = blocks * kWarps;
        const int d_step = total_groups / R;
        const int r_step = total_groups - d_step * R;
        const long long ptr_step =
            static_cast<long long>(d_step) * stride_d + static_cast<long long>(r_step) * stride_r;
        const long long wrap_extra = stride_d - static_cast<long long>(R) * stride_r;

        subarray_sum_3d_group_vec4_kernel<32><<<blocks, kThreads>>>(
            base_ptr, output, D, R, C, stride_d, stride_r, ptr_step, wrap_extra, r_step);
        return;
    }

    // Dispatch Path 2b: Sub-Warp (8 lanes) per Row Segment for 16 <= C < 64
    if (aligned_4b && C >= 16) {
        constexpr int kGroupsPerBlock = kThreads / 8;  // 32
        int blocks = (total_segs + (kGroupsPerBlock * 4 - 1)) / (kGroupsPerBlock * 4);
        if (blocks < 1) blocks = 1;
        if (blocks > kMaxBlocks) blocks = kMaxBlocks;

        const int total_groups = blocks * kGroupsPerBlock;
        const int d_step = total_groups / R;
        const int r_step = total_groups - d_step * R;
        const long long ptr_step =
            static_cast<long long>(d_step) * stride_d + static_cast<long long>(r_step) * stride_r;
        const long long wrap_extra = stride_d - static_cast<long long>(R) * stride_r;

        subarray_sum_3d_group_vec4_kernel<8><<<blocks, kThreads>>>(
            base_ptr, output, D, R, C, stride_d, stride_r, ptr_step, wrap_extra, r_step);
        return;
    }

    // Dispatch Path 3: Narrow C < 16 or Non-4B-Aligned Pointer (1 thread per segment)
    {
        int blocks = (total_segs + kThreads - 1) / kThreads;
        if (blocks < 1) blocks = 1;
        if (blocks > kMaxBlocks) blocks = kMaxBlocks;

        const int total_threads = blocks * kThreads;
        const int d_step = total_threads / R;
        const int r_step = total_threads - d_step * R;
        const long long ptr_step =
            static_cast<long long>(d_step) * stride_d + static_cast<long long>(r_step) * stride_r;
        const long long wrap_extra = stride_d - static_cast<long long>(R) * stride_r;

        subarray_sum_3d_narrow_scalar_kernel<<<blocks, kThreads>>>(
            base_ptr, output, D, R, C, stride_d, stride_r, ptr_step, wrap_extra, r_step);
    }
}
