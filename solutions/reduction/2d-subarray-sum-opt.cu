#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarp;   // 8 warps per block
constexpr int kMaxPersistentBlocks = 160;  // 4 blocks/SM * 40 SMs on Tesla T4 = 100% occupancy
constexpr unsigned kFullMask = 0xffffffffu;

// Static device BSS state for single-launch last-block retirement
__device__ long long g_block_sums[kThreads];
__device__ unsigned int g_retired_blocks = 0u;

__device__ __forceinline__ long long warp_reduce_sum_i64(long long val) {
    #pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(kFullMask, val, offset);
    }
    return val;
}

__device__ __forceinline__ void retire_block_and_write_output(
    long long thread_sum,
    int* __restrict__ output,
    long long* s_warp_sums,
    bool* s_is_last) {
    const int tid = threadIdx.x;
    const int lane = tid & (kWarp - 1);
    const int warp = tid / kWarp;

    long long wsum = warp_reduce_sum_i64(thread_sum);
    if (lane == 0) {
        s_warp_sums[warp] = wsum;
    }
    __syncthreads();

    long long block_sum = 0;
    if (warp == 0) {
        block_sum = (lane < kWarps) ? s_warp_sums[lane] : 0LL;
        block_sum = warp_reduce_sum_i64(block_sum);
    }

    if (gridDim.x == 1u) {
        if (tid == 0) {
            output[0] = static_cast<int>(block_sum);
        }
        return;
    }

    if (tid == 0) {
        g_block_sums[blockIdx.x] = block_sum;
        __threadfence();
        const unsigned prev = atomicAdd(&g_retired_blocks, 1u);
        *s_is_last = (prev + 1u == gridDim.x);
    }
    __syncthreads();

    if (!(*s_is_last)) {
        return;
    }

    __threadfence();
    __syncthreads();

    const long long part =
        (tid < static_cast<int>(gridDim.x))
            ? static_cast<const volatile long long*>(g_block_sums)[tid]
            : 0LL;
    long long final_wsum = warp_reduce_sum_i64(part);
    if (lane == 0) {
        s_warp_sums[warp] = final_wsum;
    }
    __syncthreads();

    if (warp == 0) {
        long long total = (lane < kWarps) ? s_warp_sums[lane] : 0LL;
        total = warp_reduce_sum_i64(total);
        if (lane == 0) {
            atomicExch(&g_retired_blocks, 0u);
            output[0] = static_cast<int>(total);
        }
    }
}

// ============================================================================
// 1. Wide-Row Vectorized Kernel (`col_len >= 64`):
// Each warp (32 lanes) streams whole rows in [S_ROW .. E_ROW], dynamically
// peeling 0..3 head scalars per row so `row4` is 100% 16-byte aligned even
// when `S_COL % 4 != 0` (such as `S_COL = 1` on `10000 x 10000`), `M % 4 != 0`,
// or `input` has an unaligned base offset!
// ============================================================================
__global__ __launch_bounds__(kThreads, 4) void subarray_sum_wide_int4_kernel(
    const int* __restrict__ input,
    int* __restrict__ output,
    int m,
    int s_row,
    int e_row,
    int s_col,
    int col_len) {
    __shared__ long long s_warp_sums[kWarps];
    __shared__ bool s_is_last;

    const int tid = threadIdx.x;
    const int lane = tid & (kWarp - 1);
    const int warp = tid / kWarp;

    const int global_warp = static_cast<int>(blockIdx.x) * kWarps + warp;
    const int total_warps = static_cast<int>(gridDim.x) * kWarps;

    long long acc = 0;

    for (int r = s_row + global_warp; r <= e_row; r += total_warps) {
        const int* row_ptr =
            input + static_cast<size_t>(r) * static_cast<size_t>(m) + static_cast<size_t>(s_col);

        // Compute per-row head peel count (0..3 ints) to reach 16-byte boundary
        const uintptr_t addr = reinterpret_cast<uintptr_t>(row_ptr);
        const int word_mis = static_cast<int>((addr >> 2) & 3u);
        int head = (4 - word_mis) & 3;
        if (head > col_len) {
            head = col_len;
        }

        if (lane < head) {
            acc += __ldg(&row_ptr[lane]);
        }

        const int rem = col_len - head;
        const int vec4_cnt = rem >> 2;
        const int tail = rem & 3;
        const int4* __restrict__ row4 = reinterpret_cast<const int4*>(row_ptr + head);

        int i4 = lane;
        for (; i4 + 3 * kWarp < vec4_cnt; i4 += 4 * kWarp) {
            const int4 v0 = __ldg(&row4[i4]);
            const int4 v1 = __ldg(&row4[i4 + kWarp]);
            const int4 v2 = __ldg(&row4[i4 + 2 * kWarp]);
            const int4 v3 = __ldg(&row4[i4 + 3 * kWarp]);
            acc += static_cast<long long>(v0.x) + v0.y + v0.z + v0.w +
                   static_cast<long long>(v1.x) + v1.y + v1.z + v1.w +
                   static_cast<long long>(v2.x) + v2.y + v2.z + v2.w +
                   static_cast<long long>(v3.x) + v3.y + v3.z + v3.w;
        }
        for (; i4 < vec4_cnt; i4 += kWarp) {
            const int4 v = __ldg(&row4[i4]);
            acc += static_cast<long long>(v.x) + v.y + v.z + v.w;
        }

        const int* tail_ptr = row_ptr + head + (vec4_cnt << 2);
        if (lane < tail) {
            acc += __ldg(&tail_ptr[lane]);
        }
    }

    retire_block_and_write_output(acc, output, s_warp_sums, &s_is_last);
}

// ============================================================================
// 2. Narrow-Submatrix / Small-Submatrix Kernel (`col_len < 64` or tiny total):
// Uses power-of-2 column striping (`log2_sub_w` in 0..5) so zero integer
// divisions/modulos are executed and narrow columns (e.g. `col_len = 1`) use
// all 256 threads per block across 256 rows in parallel!
// ============================================================================
__global__ __launch_bounds__(kThreads, 4) void subarray_sum_narrow_2d_kernel(
    const int* __restrict__ input,
    int* __restrict__ output,
    int m,
    int s_row,
    int e_row,
    int s_col,
    int col_len,
    int log2_sub_w) {
    __shared__ long long s_warp_sums[kWarps];
    __shared__ bool s_is_last;

    const int tid = threadIdx.x;
    const int sub_w = 1 << log2_sub_w;
    const int sub_c = tid & (sub_w - 1);
    const int sub_r = tid >> log2_sub_w;
    const int rows_per_block = kThreads >> log2_sub_w;
    const int row_stride = static_cast<int>(gridDim.x) * rows_per_block;

    long long acc = 0;
    for (int r = s_row + static_cast<int>(blockIdx.x) * rows_per_block + sub_r;
         r <= e_row;
         r += row_stride) {
        const int* row_ptr =
            input + static_cast<size_t>(r) * static_cast<size_t>(m) + static_cast<size_t>(s_col);
        for (int c = sub_c; c < col_len; c += sub_w) {
            acc += __ldg(&row_ptr[c]);
        }
    }

    retire_block_and_write_output(acc, output, s_warp_sums, &s_is_last);
}

int pick_log2_sub_w(int col_len) {
    if (col_len <= 1) return 0;
    if (col_len <= 2) return 1;
    if (col_len <= 4) return 2;
    if (col_len <= 8) return 3;
    if (col_len <= 16) return 4;
    return 5;  // 32
}

}  // namespace

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int S_ROW, int E_ROW, int S_COL,
                      int E_COL) {
    (void)N;
    const int num_rows = E_ROW - S_ROW + 1;
    const int col_len = E_COL - S_COL + 1;
    if (num_rows <= 0 || col_len <= 0) {
        return;
    }

    const size_t total = static_cast<size_t>(num_rows) * static_cast<size_t>(col_len);

    // Use wide row-aligned int4 kernel whenever col_len >= 64 and input is 4-byte int aligned
    if (col_len >= 64 && ((reinterpret_cast<uintptr_t>(input) & 3u) == 0u)) {
        int blocks = (total <= 4096u) ? 1 : ((num_rows + kWarps - 1) / kWarps);
        if (blocks > kMaxPersistentBlocks) {
            blocks = kMaxPersistentBlocks;
        }
        subarray_sum_wide_int4_kernel<<<static_cast<unsigned>(blocks), kThreads>>>(
            input, output, M, S_ROW, E_ROW, S_COL, col_len);
        return;
    }

    const int log2_sub_w = pick_log2_sub_w(col_len);
    const int rows_per_block = kThreads >> log2_sub_w;
    int blocks = (total <= 4096u) ? 1 : ((num_rows + rows_per_block - 1) / rows_per_block);
    if (blocks > kMaxPersistentBlocks) {
        blocks = kMaxPersistentBlocks;
    }
    subarray_sum_narrow_2d_kernel<<<static_cast<unsigned>(blocks), kThreads>>>(
        input, output, M, S_ROW, E_ROW, S_COL, col_len, log2_sub_w);
}
