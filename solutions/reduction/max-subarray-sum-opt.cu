#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
constexpr int kThreads = 256;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;
constexpr int kTileElems = kThreads * 4;     // 1,024 int32 elements per block
constexpr int kSmallSingleBlockMax = 4096;   // 4,097 * 8 B = 32.8 KiB <= 48 KiB smem
constexpr int kMaxBssPrefixN = 2097152;      // 2M elements (16 MiB BSS prefix buffer)
constexpr int kMaxPrefixTiles = kMaxBssPrefixN / kTileElems;  // 2,048 tiles
constexpr int kMaxReduceBlocks = 512;
constexpr unsigned kFullMask = 0xffffffffu;

// Static BSS scratchpads: zero cudaMalloc overhead, zero cudaMemset overhead,
// and cleanly overwrites pre-filled dirty output[0].
__device__ long long g_prefix[kMaxBssPrefixN + 1];
__device__ long long g_tile_sums[kMaxPrefixTiles];
__device__ int g_block_max[kMaxReduceBlocks];

__device__ __forceinline__ long long warp_inclusive_scan_i64(long long val) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    #pragma unroll
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const long long up = __shfl_up_sync(kFullMask, val, offset);
        if (lane >= offset) {
            val += up;
        }
    }
    return val;
}

__device__ __forceinline__ int warp_reduce_max_i32(int val) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        const int other = __shfl_down_sync(kFullMask, val, offset);
        if (other > val) {
            val = other;
        }
    }
    return val;
}

template <int kBlockThreads>
__device__ __forceinline__ long long block_exclusive_scan_i64(long long val,
                                                              long long* s_warp_totals,
                                                              long long* block_total_out) {
    constexpr int kWarps = kBlockThreads / kWarpSize;
    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid >> 5;

    const long long inc = warp_inclusive_scan_i64(val);
    if (lane == kWarpSize - 1) {
        s_warp_totals[warp] = inc;
    }
    __syncthreads();

    if (warp == 0) {
        long long w_val = (lane < kWarps) ? s_warp_totals[lane] : 0LL;
        w_val = warp_inclusive_scan_i64(w_val);
        if (lane < kWarps) {
            s_warp_totals[lane] = w_val;
        }
    }
    __syncthreads();

    const long long warp_offset = (warp > 0) ? s_warp_totals[warp - 1] : 0LL;
    if (block_total_out != nullptr) {
        *block_total_out = s_warp_totals[kWarps - 1];
    }
    return warp_offset + inc - val;
}

template <int kBlockThreads>
__device__ __forceinline__ int block_reduce_max_i32(int val, int* s_warp_max) {
    constexpr int kWarps = kBlockThreads / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x >> 5;

    val = warp_reduce_max_i32(val);
    if (lane == 0) {
        s_warp_max[warp] = val;
    }
    __syncthreads();

    if (warp == 0) {
        int bmax = (lane < kWarps) ? s_warp_max[lane] : INT_MIN;
        bmax = warp_reduce_max_i32(bmax);
        return bmax;
    }
    return INT_MIN;
}

// ============================================================================
// Tier 1: Single-Block Shared-Memory Kernel for N <= 4,096 (1 launch total!).
// Computes exact int64 prefix sums in shared memory and reduces max(P[i] - P[i - W]).
// ============================================================================
__global__ __launch_bounds__(kThreads) void max_subarray_small_smem_kernel(
    const int* __restrict__ input,
    int* __restrict__ output,
    int N,
    int W) {
    __shared__ long long s_prefix[kSmallSingleBlockMax + 1];
    __shared__ long long s_warp_totals[kWarpsPerBlock];
    __shared__ int s_warp_max[kWarpsPerBlock];
    __shared__ long long s_running_base;

    const int tid = threadIdx.x;
    if (tid == 0) {
        s_prefix[0] = 0LL;
        s_running_base = 0LL;
    }
    __syncthreads();

    for (int chunk_base = 0; chunk_base < N; chunk_base += kTileElems) {
        const int base_idx = chunk_base + (tid << 2);
        long long x0 = 0LL;
        long long x1 = 0LL;
        long long x2 = 0LL;
        long long x3 = 0LL;
        if (base_idx < N) x0 = static_cast<long long>(__ldg(&input[base_idx]));
        if (base_idx + 1 < N) x1 = static_cast<long long>(__ldg(&input[base_idx + 1]));
        if (base_idx + 2 < N) x2 = static_cast<long long>(__ldg(&input[base_idx + 2]));
        if (base_idx + 3 < N) x3 = static_cast<long long>(__ldg(&input[base_idx + 3]));

        const long long thread_sum = (x0 + x1) + (x2 + x3);
        long long tile_total = 0LL;
        const long long excl =
            s_running_base + block_exclusive_scan_i64<kThreads>(thread_sum, s_warp_totals, &tile_total);

        if (base_idx < N) s_prefix[base_idx + 1] = excl + x0;
        if (base_idx + 1 < N) s_prefix[base_idx + 2] = excl + x0 + x1;
        if (base_idx + 2 < N) s_prefix[base_idx + 3] = excl + x0 + x1 + x2;
        if (base_idx + 3 < N) s_prefix[base_idx + 4] = excl + thread_sum;
        __syncthreads();

        if (tid == 0) {
            s_running_base += tile_total;
        }
        __syncthreads();
    }

    int local_max = INT_MIN;
    for (int end_idx = W + tid; end_idx <= N; end_idx += kThreads) {
        const int diff = static_cast<int>(s_prefix[end_idx] - s_prefix[end_idx - W]);
        if (diff > local_max) {
            local_max = diff;
        }
    }

    const int block_max = block_reduce_max_i32<kThreads>(local_max, s_warp_max);
    if (tid == 0) {
        output[0] = block_max;
    }
}

// ============================================================================
// Tier 2A (Single-Launch Multi-Block Medium Path, 4,096 < N <= 262,144):
// Wait-free 2-Kernel L2-resident int4 Prefix Sum + Fused Diff-Max Reduction!
// Kernel 1 computes per-tile int4 local scans and tile sums (for N = 50,000,
// only 49 tiles = 400 KB in L2 cache!).
// If num_tiles <= 1,024 (covering up to N = 1,048,576, including official N = 50,000!),
// Kernel 2 scans tile sums in shared memory AND adds tile offsets + computes
// P[i] - P[i - W] max reduction! Wait: P[i - W] needs global prefix P written first,
// so 3 tiny L2-resident kernels OR 2 kernels where Kernel 1 computes tile sums,
// Kernel 2 writes global P[0..N] in L2, and Kernel 3 reduces max(P[i] - P[i - W]).
// Even better: when N <= 1,048,576, can Kernel 1 write local inclusive prefix into
// g_prefix[1..N] AND g_tile_sums[b], then Kernel 2 scans g_tile_sums[0..num_tiles-1]
// INTO shared memory `s_tile_excl[0..num_tiles]` inside EVERY block (50 tiles = 50 int64s,
// scanned by 2 warps in 10 cycles!!) so Kernel 2 computes BOTH `P[i] = g_prefix[i] + s_tile_excl[i >> 10]`
// AND `P[i - W] = g_prefix[i - W] + s_tile_excl[(i - W) >> 10]` ON THE FLY WITHOUT A 3RD KERNEL?!
// ============================================================================

// YES! Look at that insight:
// Kernel 1 (`local_scan_int4_kernel`): each block `b` (`1,024` elements) scans its own tile in
// registers, writes tile-local prefix `L[1..1024]` to `g_prefix[b*1024 + 1 ..]` (with `g_prefix[0] = 0`)
// and writes `g_tile_sums[b] = tile_total`!
// Kernel 2 (`fused_offset_diff_max_kernel`): for `num_tiles <= 2048` (`N <= 2,097,152`),
// `num_tiles` is at most `2,048` `int64`s (`16 KiB` shared memory; for `N = 50,000`, ONLY `49` `int64`s = `392 B`!).
// Each block in Kernel 2 loads `g_tile_sums[0..num_tiles-1]` from L2 cache, scans it into `s_tile_excl[0..num_tiles]`
// in shared memory in ~1 us, and IMMEDIATELY evaluates exact global prefix at ANY index `k`:
//   `P(k) = (k == 0) ? 0 : (g_prefix[k] + s_tile_excl[(k - 1) >> 10])`!
// So Kernel 2 directly computes `diff = P(end_idx) - P(end_idx - W)` across all windows and reduces the
// global max into `output[0]`!
// TOTAL KERNEL LAUNCHES FOR N = 50,000: ONLY 2 TINY L2-RESIDENT KERNELS (plus 1-block reduce if blocks > 1)!
// Wait: if we launch `min(blocks, 64)` persistent blocks in Kernel 2 and reduce via `atomicMax`? No,
// `output[0]` can be negative and pre-filled with dirty `999999`! Wait: CUDA `atomicMax` works on signed `int`!
// If Kernel 1 thread `(blockIdx.x == 0 && threadIdx.x == 0)` sets `output[0] = INT_MIN` BEFORE Kernel 2 runs,
// then Kernel 2's blocks (`<= 80` blocks) each do ONE `atomicMax(&output[0], block_max)` at the end!!
// ZERO 3rd kernel! ONLY **2 KERNEL LAUNCHES TOTAL** FOR `4,096 < N <= 2,097,152`!!

__global__ __launch_bounds__(kThreads, 4) void local_scan_vec4_kernel(
    const int* __restrict__ input,
    int* __restrict__ output,
    int N,
    bool aligned_vec4) {
    __shared__ long long s_warp_totals[kWarpsPerBlock];

    const int tid = threadIdx.x;
    const int tile = blockIdx.x;
    const int tile_start = tile * kTileElems;
    const int idx = tile_start + (tid << 2);

    if (tile == 0 && tid == 0) {
        g_prefix[0] = 0LL;
        output[0] = INT_MIN;
    }

    long long x0 = 0LL;
    long long x1 = 0LL;
    long long x2 = 0LL;
    long long x3 = 0LL;

    if (aligned_vec4 && idx + 3 < N) {
        const int4 v = __ldg(reinterpret_cast<const int4*>(input) + (idx >> 2));
        x0 = static_cast<long long>(v.x);
        x1 = static_cast<long long>(v.y);
        x2 = static_cast<long long>(v.z);
        x3 = static_cast<long long>(v.w);
    } else {
        if (idx < N) x0 = static_cast<long long>(__ldg(&input[idx]));
        if (idx + 1 < N) x1 = static_cast<long long>(__ldg(&input[idx + 1]));
        if (idx + 2 < N) x2 = static_cast<long long>(__ldg(&input[idx + 2]));
        if (idx + 3 < N) x3 = static_cast<long long>(__ldg(&input[idx + 3]));
    }

    const long long thread_sum = (x0 + x1) + (x2 + x3);
    long long tile_total = 0LL;
    const long long excl =
        block_exclusive_scan_i64<kThreads>(thread_sum, s_warp_totals, &tile_total);

    if (idx + 3 < N) {
        // g_prefix[idx + 1] is at index (tile*1024 + 4*tid + 1); store scalar to avoid 16B alignment hazard on +1!
        g_prefix[idx + 1] = excl + x0;
        g_prefix[idx + 2] = excl + x0 + x1;
        g_prefix[idx + 3] = excl + x0 + x1 + x2;
        g_prefix[idx + 4] = excl + thread_sum;
    } else {
        if (idx < N) g_prefix[idx + 1] = excl + x0;
        if (idx + 1 < N) g_prefix[idx + 2] = excl + x0 + x1;
        if (idx + 2 < N) g_prefix[idx + 3] = excl + x0 + x1 + x2;
    }

    if (tid == 0) {
        g_tile_sums[tile] = tile_total;
    }
}

__global__ __launch_bounds__(kThreads, 4) void fused_offset_diff_max_kernel(
    int* __restrict__ output,
    int N,
    int W,
    int num_tiles) {
    __shared__ long long s_tile_excl[kMaxPrefixTiles];
    __shared__ long long s_warp_totals[kWarpsPerBlock];
    __shared__ int s_warp_max[kWarpsPerBlock];
    __shared__ long long s_running_base;

    const int tid = threadIdx.x;
    if (tid == 0) {
        s_running_base = 0LL;
    }
    __syncthreads();

    // Scan all tile sums into shared memory s_tile_excl[0..num_tiles-1].
    // For N = 50,000, num_tiles = 49, so this loop executes ONCE!
    for (int base = 0; base < num_tiles; base += kThreads) {
        const int t = base + tid;
        const long long val = (t < num_tiles) ? g_tile_sums[t] : 0LL;
        long long chunk_sum = 0LL;
        const long long excl =
            s_running_base + block_exclusive_scan_i64<kThreads>(val, s_warp_totals, &chunk_sum);
        if (t < num_tiles) {
            s_tile_excl[t] = excl;
        }
        __syncthreads();
        if (tid == 0) {
            s_running_base += chunk_sum;
        }
        __syncthreads();
    }

    const int num_windows = N - W + 1;
    const int global_tid = blockIdx.x * kThreads + tid;
    const int stride = gridDim.x * kThreads;

    int local_max = INT_MIN;
    for (int win = global_tid; win < num_windows; win += stride) {
        const int end_idx = win + W;
        const long long p_end = g_prefix[end_idx] + s_tile_excl[(end_idx - 1) >> 10];
        const long long p_start =
            (win == 0) ? 0LL : (g_prefix[win] + s_tile_excl[(win - 1) >> 10]);
        const int diff = static_cast<int>(p_end - p_start);
        if (diff > local_max) {
            local_max = diff;
        }
    }

    const int block_max = block_reduce_max_i32<kThreads>(local_max, s_warp_max);
    if (tid == 0) {
        atomicMax(output, block_max);
    }
}

// ============================================================================
// Tier 3: Zero-Allocation Chunked Sliding-Window Fallback for Huge N > 2,097,152.
// Partition the N - W + 1 windows into contiguous start-index ranges [s_begin, s_end).
// Each thread slides a single O(1)-update running window across its own contiguous slice:
//   initial window sum = O(W), followed by s_slice_len steps of `sum += in[s+W] - in[s]`.
// When W is large (e.g. N = 3,000,000, W = 50,000), block-level cooperative prefix-diff
// chunks of size B_CHUNK = 2,048 compute 2,048 windows per block in O(W + B_CHUNK) time!
// ============================================================================
__global__ __launch_bounds__(kThreads) void huge_chunked_sliding_kernel(
    const int* __restrict__ input,
    int* __restrict__ output,
    int N,
    int W,
    int windows_per_block) {
    __shared__ long long s_warp_totals[kWarpsPerBlock];
    __shared__ int s_warp_max[kWarpsPerBlock];
    __shared__ long long s_base_window_sum;

    const int tid = threadIdx.x;
    const int num_windows = N - W + 1;
    const int block_win_start = blockIdx.x * windows_per_block;
    if (block_win_start >= num_windows) {
        return;
    }
    int block_win_end = block_win_start + windows_per_block;
    if (block_win_end > num_windows) {
        block_win_end = num_windows;
    }

    // Step 1: Cooperatively compute the exact int64 sum of window `block_win_start`:
    //   sum_{k = 0}^{W - 1} input[block_win_start + k].
    long long partial = 0LL;
    for (int k = tid; k < W; k += kThreads) {
        partial += static_cast<long long>(__ldg(&input[block_win_start + k]));
    }
    long long base_win_sum = 0LL;
    block_exclusive_scan_i64<kThreads>(partial, s_warp_totals, &base_win_sum);

    int local_max = (tid == 0) ? static_cast<int>(base_win_sum) : INT_MIN;
    if (tid == 0) {
        s_base_window_sum = base_win_sum;
    }
    __syncthreads();

    // Step 2: Advance across [block_win_start + 1 .. block_win_end - 1] in tiles of 1,024 steps!
    // Each step `s` has delta `d[s] = input[s + W] - input[s]`, so window `s + 1` sum is
    // `cur_base + inclusive_scan(d)` — 100% O(N) work even when W = N / 2 and N = 50,000,000!!
    const int num_steps = block_win_end - block_win_start - 1;
    for (int step_base = 0; step_base < num_steps; step_base += kTileElems) {
        const int step0 = step_base + (tid << 2);
        long long d0 = 0LL;
        long long d1 = 0LL;
        long long d2 = 0LL;
        long long d3 = 0LL;

        if (step0 < num_steps) {
            const int s = block_win_start + step0;
            d0 = static_cast<long long>(__ldg(&input[s + W])) - static_cast<long long>(__ldg(&input[s]));
        }
        if (step0 + 1 < num_steps) {
            const int s = block_win_start + step0 + 1;
            d1 = static_cast<long long>(__ldg(&input[s + W])) - static_cast<long long>(__ldg(&input[s]));
        }
        if (step0 + 2 < num_steps) {
            const int s = block_win_start + step0 + 2;
            d2 = static_cast<long long>(__ldg(&input[s + W])) - static_cast<long long>(__ldg(&input[s]));
        }
        if (step0 + 3 < num_steps) {
            const int s = block_win_start + step0 + 3;
            d3 = static_cast<long long>(__ldg(&input[s + W])) - static_cast<long long>(__ldg(&input[s]));
        }

        const long long tsum = (d0 + d1) + (d2 + d3);
        long long tile_delta_total = 0LL;
        const long long excl =
            s_base_window_sum + block_exclusive_scan_i64<kThreads>(tsum, s_warp_totals, &tile_delta_total);

        if (step0 < num_steps) {
            const int w0 = static_cast<int>(excl + d0);
            if (w0 > local_max) local_max = w0;
        }
        if (step0 + 1 < num_steps) {
            const int w1 = static_cast<int>(excl + d0 + d1);
            if (w1 > local_max) local_max = w1;
        }
        if (step0 + 2 < num_steps) {
            const int w2 = static_cast<int>(excl + d0 + d1 + d2);
            if (w2 > local_max) local_max = w2;
        }
        if (step0 + 3 < num_steps) {
            const int w3 = static_cast<int>(excl + tsum);
            if (w3 > local_max) local_max = w3;
        }
        __syncthreads();
        if (tid == 0) {
            s_base_window_sum += tile_delta_total;
        }
        __syncthreads();
    }

    const int bmax = block_reduce_max_i32<kThreads>(local_max, s_warp_max);
    if (tid == 0) {
        g_block_max[blockIdx.x] = bmax;
    }
}

__global__ __launch_bounds__(kThreads) void reduce_block_max_kernel(
    int* __restrict__ output,
    int num_blocks) {
    __shared__ int s_warp_max[kWarpsPerBlock];
    const int tid = threadIdx.x;

    int local_max = INT_MIN;
    for (int i = tid; i < num_blocks; i += kThreads) {
        const int v = g_block_max[i];
        if (v > local_max) {
            local_max = v;
        }
    }
    const int bmax = block_reduce_max_i32<kThreads>(local_max, s_warp_max);
    if (tid == 0) {
        output[0] = bmax;
    }
}

}  // namespace

extern "C" void solve(const int* input, int* output, int N, int window_size) {
    if (N <= 0 || window_size <= 0 || window_size > N) {
        return;
    }

    // Tier 1: N <= 4,096 -> Single-block shared-memory scan + diff-max (1 launch!).
    if (N <= kSmallSingleBlockMax) {
        max_subarray_small_smem_kernel<<<1, kThreads>>>(input, output, N, window_size);
        return;
    }

    // Tier 2: 4,096 < N <= 2,097,152 -> 2-kernel L2-resident int4 prefix-sum + diff-max
    // (covers official N = 50,000, window_size = 25,000 in 49 tiles!).
    if (N <= kMaxBssPrefixN) {
        const bool aligned_vec4 = (reinterpret_cast<uintptr_t>(input) & 15u) == 0u;
        const int num_tiles = (N + kTileElems - 1) / kTileElems;
        local_scan_vec4_kernel<<<num_tiles, kThreads>>>(input, output, N, aligned_vec4);

        const int num_windows = N - window_size + 1;
        int diff_blocks = (num_windows + (kThreads * 4) - 1) / (kThreads * 4);
        if (diff_blocks < 1) {
            diff_blocks = 1;
        }
        if (diff_blocks > 160) {
            diff_blocks = 160;
        }
        fused_offset_diff_max_kernel<<<diff_blocks, kThreads>>>(
            output, N, window_size, num_tiles);
        return;
    }

    // Tier 3: Huge N > 2,097,152 -> O(N) chunked delta-scan sliding window (zero cudaMalloc!).
    const int num_windows = N - window_size + 1;
    int num_blocks = 160;
    if (num_windows < num_blocks) {
        num_blocks = num_windows;
    }
    const int windows_per_block = (num_windows + num_blocks - 1) / num_blocks;
    const int active_blocks = (num_windows + windows_per_block - 1) / windows_per_block;

    huge_chunked_sliding_kernel<<<active_blocks, kThreads>>>(
        input, output, N, window_size, windows_per_block);
    reduce_block_max_kernel<<<1, kThreads>>>(output, active_blocks);
}
