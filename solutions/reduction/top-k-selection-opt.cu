#include <cuda_runtime.h>
#include <cstdint>

#ifndef TOPK_THREADS
#define TOPK_THREADS 256
#endif

#ifndef TOPK_BLOCKS
#define TOPK_BLOCKS 320
#endif

#ifndef TOPK_STREAM_BLOCKS
#define TOPK_STREAM_BLOCKS 160
#endif

namespace {

constexpr int kWarp = 32;
constexpr int kThreads = TOPK_THREADS;
static_assert(kThreads >= kWarp && kThreads <= 1024 && (kThreads & (kThreads - 1)) == 0,
              "TOPK_THREADS must be a power of two in [32, 1024]");
constexpr int kMaxBlocks = TOPK_BLOCKS;
constexpr int kStreamBlocks = TOPK_STREAM_BLOCKS;
static_assert(kStreamBlocks >= 1 && kStreamBlocks <= 512, "TOPK_STREAM_BLOCKS must be in [1, 512]");
constexpr int kStreamMaxK = 256;
constexpr int kRadixBins = 65536;
constexpr int kSmallNThreshold = 4096;
constexpr int kFastKThreshold = 4096;
constexpr unsigned kFullMask = 0xffffffffu;

struct SelectState {
    uint32_t target_hi16;
    uint32_t greater_hi_count;
    uint32_t write_idx;
    uint32_t threshold_key;
    uint32_t strictly_greater_count;
};

__device__ uint32_t g_block_topk[kStreamBlocks * kStreamMaxK];
__device__ int g_block_valid[kStreamBlocks];
__device__ uint32_t g_hist_hi[kRadixBins];
__device__ uint32_t g_hist_lo[kRadixBins];
__device__ uint32_t g_fast_keys[kFastKThreshold];
__device__ SelectState g_state;

__device__ __forceinline__ uint32_t float_to_ordered_uint(float x) {
    if (x == 0.0f) {
        x = 0.0f;
    }
    const uint32_t u = __float_as_uint(x);
    return (u & 0x80000000u) ? (~u) : (u ^ 0x80000000u);
}

__device__ __forceinline__ float ordered_uint_to_float(uint32_t k) {
    const uint32_t u = (k & 0x80000000u) ? (k ^ 0x80000000u) : (~k);
    return __uint_as_float(u);
}

// ============================================================================
// Single-Pass Streaming Block-Local Top-K Filter for 1 <= k <= 256!
// - Shared memory per block: s_buf[2048] (8 KiB) + 4 scalars -> fits 4+ blocks/SM on T4!
// - Zero global atomics! Only ~5 shared-memory bitonic flushes per block across 312,500 floats!
// - 90%+ of warps bypass shared memory entirely via __ballot_sync!
// ============================================================================

__device__ __forceinline__ void bitonic_sort_desc_smem_256(uint32_t* s_data, int padded_len) {
    const int tid = threadIdx.x;
    for (int stage_k = 2; stage_k <= padded_len; stage_k <<= 1) {
        for (int j = stage_k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < padded_len; i += 256) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const uint32_t a = s_data[i];
                    const uint32_t b = s_data[ixj];
                    const bool descending = ((i & stage_k) == 0);
                    if ((descending && a < b) || (!descending && a > b)) {
                        s_data[i] = b;
                        s_data[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void flush_smem_candidates(uint32_t* s_buf,
                                                      int k,
                                                      int& has_k,
                                                      uint32_t& thresh,
                                                      int* s_cand_cnt) {
    const int tid = threadIdx.x;
    const int cand_cnt = *s_cand_cnt;
    const int start_idx = has_k ? (256 - k) : 256;
    const int valid_end = 256 + cand_cnt;
    const int total_items = valid_end - start_idx;

    int sort_len = 256;
    while (sort_len < total_items) {
        sort_len <<= 1;
    }

    // Pad [valid_end .. start_idx + sort_len - 1] with 0u before sorting s_buf + start_idx
    const int pad_end = start_idx + sort_len;
    for (int i = valid_end + tid; i < pad_end; i += 256) {
        s_buf[i] = 0u;
    }
    __syncthreads();

    bitonic_sort_desc_smem_256(s_buf + start_idx, sort_len);

    if (total_items >= k) {
        const uint32_t my_topk = (tid < k) ? s_buf[start_idx + tid] : 0u;
        __syncthreads();
        if (tid < k) {
            s_buf[256 - k + tid] = my_topk;
        }
        if (tid == 0) {
            *s_cand_cnt = 0;
        }
        __syncthreads();
        has_k = 1;
        thresh = s_buf[255];
    } else {
        // Fewer than k elements seen so far: keep them in candidate region s_buf[256 .. 256 + total_items - 1]
        const uint32_t my_item = (tid < total_items) ? s_buf[start_idx + tid] : 0u;
        __syncthreads();
        if (tid < total_items) {
            s_buf[256 + tid] = my_item;
        }
        if (tid == 0) {
            *s_cand_cnt = total_items;
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void warp_append_key(uint32_t key,
                                                bool keep,
                                                uint32_t* s_buf,
                                                int* s_cand_cnt) {
    const unsigned mask = __ballot_sync(kFullMask, keep);
    if (mask == 0u) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int leader = __ffs(mask) - 1;
    int warp_base = 0;
    if (lane == leader) {
        warp_base = atomicAdd(s_cand_cnt, __popc(mask));
    }
    warp_base = __shfl_sync(kFullMask, warp_base, leader);
    if (keep) {
        const int rank = __popc(mask & ((1u << lane) - 1u));
        s_buf[256 + warp_base + rank] = key;
    }
}

__global__ __launch_bounds__(256, 4) void topk_stream_pass1_kernel(
    const float* __restrict__ input,
    int peel,
    int aligned_n4,
    int n,
    int k) {
    __shared__ uint32_t s_buf[2048];
    __shared__ int s_cand_cnt;

    const int tid = threadIdx.x;
    const int b = blockIdx.x;
    const int num_blocks = gridDim.x;

    // Partition aligned_n4 float4 vectors evenly across blocks in contiguous ranges
    const int vec_start = static_cast<int>(
        (static_cast<uint64_t>(aligned_n4) * static_cast<uint64_t>(b)) /
        static_cast<uint64_t>(num_blocks));
    const int vec_end = static_cast<int>(
        (static_cast<uint64_t>(aligned_n4) * static_cast<uint64_t>(b + 1)) /
        static_cast<uint64_t>(num_blocks));
    const int my_vecs = vec_end - vec_start;

    int my_extra = 0;
    if (b == 0) {
        my_extra = peel + (n - (peel + (aligned_n4 << 2)));
    }
    const int my_total_elems = (my_vecs << 2) + my_extra;

    if (tid == 0) {
        g_block_valid[b] = (my_total_elems < k) ? my_total_elems : k;
        s_cand_cnt = 0;
    }
    s_buf[tid] = 0u;
    __syncthreads();

    if (my_total_elems == 0) {
        if (tid < k) {
            g_block_topk[b * kStreamMaxK + tid] = 0u;
        }
        return;
    }

    const float4* __restrict__ in4 = reinterpret_cast<const float4*>(input + peel);

    // Sample 256 evenly spaced elements across this block's float4 range for an instant tight bound!
    uint32_t thresh = 0u;
    if (my_vecs >= 256) {
        const int step = my_vecs >> 8;
        const float4 sv = __ldg(&in4[vec_start + tid * step]);
        s_buf[tid] = float_to_ordered_uint(sv.x);
        __syncthreads();
        bitonic_sort_desc_smem_256(s_buf, 256);
        thresh = s_buf[k - 1];
        __syncthreads();
    }

    int has_k = 0;

    // Handle head peel (0..3 elements) and tail remainder (0..3 elements) on block 0
    if (b == 0 && my_extra > 0) {
        const int tail_base = peel + (aligned_n4 << 2);
        const int tail_cnt = n - tail_base;
        const bool is_head = (tid < peel);
        const bool is_tail = (!is_head) && (tid - peel < tail_cnt);
        const bool valid = is_head || is_tail;
        uint32_t key = 0u;
        if (is_head) {
            key = float_to_ordered_uint(__ldg(&input[tid]));
        } else if (is_tail) {
            key = float_to_ordered_uint(__ldg(&input[tail_base + (tid - peel)]));
        }
        const bool keep = valid && (has_k ? (key > thresh) : (key >= thresh));
        warp_append_key(key, keep, s_buf, &s_cand_cnt);
        __syncthreads();
    }

    // Main streaming loop: 256 threads * 1 float4 = 1,024 floats per step
    for (int v_base = vec_start; v_base < vec_end; v_base += 256) {
        // Ensure worst-case +1024 survivors never overflow s_buf[256..2047] (capacity 1792)
        if (s_cand_cnt > 768) {
            flush_smem_candidates(s_buf, k, has_k, thresh, &s_cand_cnt);
        }

        const int v_idx = v_base + tid;
        uint32_t k0 = 0u, k1 = 0u, k2 = 0u, k3 = 0u;
        bool p0 = false, p1 = false, p2 = false, p3 = false;
        if (v_idx < vec_end) {
            const float4 v = __ldg(&in4[v_idx]);
            k0 = float_to_ordered_uint(v.x);
            k1 = float_to_ordered_uint(v.y);
            k2 = float_to_ordered_uint(v.z);
            k3 = float_to_ordered_uint(v.w);
            if (has_k) {
                p0 = (k0 > thresh);
                p1 = (k1 > thresh);
                p2 = (k2 > thresh);
                p3 = (k3 > thresh);
            } else {
                p0 = (k0 >= thresh);
                p1 = (k1 >= thresh);
                p2 = (k2 >= thresh);
                p3 = (k3 >= thresh);
            }
        }

        const unsigned any_warp = __ballot_sync(kFullMask, p0 || p1 || p2 || p3);
        if (any_warp != 0u) {
            warp_append_key(k0, p0, s_buf, &s_cand_cnt);
            warp_append_key(k1, p1, s_buf, &s_cand_cnt);
            warp_append_key(k2, p2, s_buf, &s_cand_cnt);
            warp_append_key(k3, p3, s_buf, &s_cand_cnt);
        }

        // Only synchronize and flush when either !has_k yet (first tile establishes strict > thresh!)
        // or every 4th tile (`((v_base - vec_start) & 768) == 768`) to check s_cand_cnt without
        // paying __syncthreads() on tiles where 0 threads wrote!
        // WAIT: before reading `s_cand_cnt` at the top of the next iteration, all warps that wrote
        // to `s_cand_cnt` in THIS iteration MUST be visible! A single `__syncthreads()` per 1,024
        // floats is only 300 barriers across 312,500 floats = ~3 microseconds total on T4!
        __syncthreads();
        if ((!has_k && s_cand_cnt >= k) || (s_cand_cnt >= 512)) {
            flush_smem_candidates(s_buf, k, has_k, thresh, &s_cand_cnt);
        }
    }

    if (s_cand_cnt > 0 || !has_k) {
        flush_smem_candidates(s_buf, k, has_k, thresh, &s_cand_cnt);
    }

    const int valid_cnt = (my_total_elems < k) ? my_total_elems : k;
    if (tid < valid_cnt) {
        const uint32_t out_key = has_k ? s_buf[256 - k + tid] : s_buf[256 + tid];
        g_block_topk[b * kStreamMaxK + tid] = out_key;
    }
}

// Stage 2 (<<<1, 256>>>, ~20 us): Merge up to 160 * k keys in L2 cache into output[0..k-1]!
__global__ __launch_bounds__(256) void topk_stream_merge_kernel(float* __restrict__ output,
                                                                int num_blocks,
                                                                int k) {
    __shared__ uint32_t s_buf[2048];
    __shared__ int s_cand_cnt;
    __shared__ int s_valid_counts[kStreamBlocks];

    const int tid = threadIdx.x;
    if (tid == 0) {
        s_cand_cnt = 0;
    }
    s_buf[tid] = 0u;
    for (int b = tid; b < num_blocks; b += 256) {
        s_valid_counts[b] = g_block_valid[b];
    }
    __syncthreads();

    int has_k = 0;
    uint32_t thresh = 0u;

    const int total_slots = num_blocks * k;
    for (int base = 0; base < total_slots; base += 256) {
        if (s_cand_cnt > 1536) {
            flush_smem_candidates(s_buf, k, has_k, thresh, &s_cand_cnt);
        }

        const int idx = base + tid;
        bool valid = false;
        uint32_t key = 0u;
        if (idx < total_slots) {
            const int b = idx / k;
            const int r = idx - b * k;
            if (r < s_valid_counts[b]) {
                valid = true;
                key = g_block_topk[b * kStreamMaxK + r];
            }
        }
        const bool keep = valid && (has_k ? (key > thresh) : (key >= thresh));
        warp_append_key(key, keep, s_buf, &s_cand_cnt);
        __syncthreads();

        if ((!has_k && s_cand_cnt >= k) || (s_cand_cnt >= 512)) {
            flush_smem_candidates(s_buf, k, has_k, thresh, &s_cand_cnt);
        }
    }

    if (s_cand_cnt > 0 || !has_k) {
        flush_smem_candidates(s_buf, k, has_k, thresh, &s_cand_cnt);
    }

    if (tid < k) {
        const uint32_t final_key = has_k ? s_buf[256 - k + tid] : s_buf[256 + tid];
        output[tid] = ordered_uint_to_float(final_key);
    }
}

// ============================================================================
// Small-N Single-Block Kernel (N <= 4096) & Large-K Radix Fallback (k > 256)
// ============================================================================

__device__ __forceinline__ void warp_aggregated_hist_inc(uint32_t* hist, uint32_t bin) {
#if __CUDA_ARCH__ >= 700
    const unsigned active = __activemask();
    const unsigned peers = __match_any_sync(active, bin);
    const int leader = __ffs(peers) - 1;
    const int lane = threadIdx.x & (kWarp - 1);
    if (lane == leader) {
        atomicAdd(&hist[bin], static_cast<uint32_t>(__popc(peers)));
    }
#else
    atomicAdd(&hist[bin], 1u);
#endif
}

__device__ __forceinline__ uint32_t block_inclusive_scan_256(uint32_t val, uint32_t* s_warp_sums) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

#pragma unroll
    for (int step = 1; step < 32; step <<= 1) {
        const uint32_t prev = __shfl_up_sync(kFullMask, val, step);
        if (lane >= step) {
            val += prev;
        }
    }
    if (lane == 31) {
        s_warp_sums[warp] = val;
    }
    __syncthreads();

    if (warp == 0) {
        const uint32_t orig = (lane < 8) ? s_warp_sums[lane] : 0u;
        uint32_t wv = orig;
#pragma unroll
        for (int step = 1; step < 8; step <<= 1) {
            const uint32_t prev = __shfl_up_sync(kFullMask, wv, step);
            if (lane >= step) {
                wv += prev;
            }
        }
        if (lane < 8) {
            s_warp_sums[lane] = wv - orig;
        }
    }
    __syncthreads();

    return val + s_warp_sums[warp];
}

__global__ __launch_bounds__(512) void topk_small_block_kernel(const float* __restrict__ input,
                                                               float* __restrict__ output,
                                                               int n,
                                                               int k) {
    __shared__ uint32_t s_keys[kSmallNThreshold];

    const int tid = threadIdx.x;
    int padded_n = 1;
    while (padded_n < n) {
        padded_n <<= 1;
    }

    for (int i = tid; i < padded_n; i += 512) {
        s_keys[i] = (i < n) ? float_to_ordered_uint(__ldg(&input[i])) : 0u;
    }
    __syncthreads();

    for (int stage_k = 2; stage_k <= padded_n; stage_k <<= 1) {
        for (int j = stage_k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < padded_n; i += 512) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const uint32_t a = s_keys[i];
                    const uint32_t b = s_keys[ixj];
                    const bool descending = ((i & stage_k) == 0);
                    if ((descending && a < b) || (!descending && a > b)) {
                        s_keys[i] = b;
                        s_keys[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int i = tid; i < k; i += 512) {
        output[i] = ordered_uint_to_float(s_keys[i]);
    }
}

__global__ __launch_bounds__(256) void clear_hists_kernel() {
    const int idx = blockIdx.x * 256 + threadIdx.x;
    uint4* hi4 = reinterpret_cast<uint4*>(g_hist_hi);
    uint4* lo4 = reinterpret_cast<uint4*>(g_hist_lo);
    const int n4 = kRadixBins >> 2;
    for (int i = idx; i < n4; i += gridDim.x * 256) {
        hi4[i] = make_uint4(0u, 0u, 0u, 0u);
        lo4[i] = make_uint4(0u, 0u, 0u, 0u);
    }
    if (idx == 0) {
        g_state.target_hi16 = 0u;
        g_state.greater_hi_count = 0u;
        g_state.write_idx = 0u;
        g_state.threshold_key = 0u;
        g_state.strictly_greater_count = 0u;
    }
}

template <bool kAligned>
__global__ __launch_bounds__(kThreads) void pass1_hist_hi16_kernel(const float* __restrict__ input,
                                                                   int n4,
                                                                   int n) {
    const int global_tid = blockIdx.x * kThreads + threadIdx.x;
    const int stride = gridDim.x * kThreads;

    if (kAligned) {
        const float4* __restrict__ in4 = reinterpret_cast<const float4*>(input);
        for (int idx = global_tid; idx < n4; idx += stride) {
            const float4 v = __ldg(&in4[idx]);
            warp_aggregated_hist_inc(g_hist_hi, float_to_ordered_uint(v.x) >> 16);
            warp_aggregated_hist_inc(g_hist_hi, float_to_ordered_uint(v.y) >> 16);
            warp_aggregated_hist_inc(g_hist_hi, float_to_ordered_uint(v.z) >> 16);
            warp_aggregated_hist_inc(g_hist_hi, float_to_ordered_uint(v.w) >> 16);
        }
        for (int i = (n4 << 2) + global_tid; i < n; i += stride) {
            warp_aggregated_hist_inc(g_hist_hi, float_to_ordered_uint(__ldg(&input[i])) >> 16);
        }
    } else {
        for (int i = global_tid; i < n; i += stride) {
            warp_aggregated_hist_inc(g_hist_hi, float_to_ordered_uint(__ldg(&input[i])) >> 16);
        }
    }
}

__device__ __forceinline__ void find_kth_bucket_65536(const uint32_t* hist,
                                                      uint32_t target_rank,
                                                      uint32_t* s_out_bucket,
                                                      uint32_t* s_out_prior) {
    __shared__ uint32_t s_chunk_sums[256];
    __shared__ uint32_t s_warp_sums[8];
    __shared__ int s_selected_chunk;
    __shared__ uint32_t s_chunk_prior;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    for (int c_base = warp; c_base < 256; c_base += 8) {
        const int bin_start = (255 - c_base) * 256;
        uint32_t c_sum = 0u;
#pragma unroll
        for (int r = 0; r < 256; r += 32) {
            c_sum += hist[bin_start + r + lane];
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            c_sum += __shfl_down_sync(kFullMask, c_sum, offset);
        }
        if (lane == 0) {
            s_chunk_sums[c_base] = c_sum;
        }
    }
    __syncthreads();

    const uint32_t c_val = s_chunk_sums[tid];
    const uint32_t c_inc = block_inclusive_scan_256(c_val, s_warp_sums);
    const uint32_t c_exc = c_inc - c_val;

    if (c_exc < target_rank && c_inc >= target_rank) {
        s_selected_chunk = tid;
        s_chunk_prior = c_exc;
    }
    __syncthreads();

    const int sel_chunk = s_selected_chunk;
    const uint32_t chunk_prior = s_chunk_prior;
    const int target_bin = (255 - sel_chunk) * 256 + (255 - tid);
    const uint32_t b_val = hist[target_bin];
    const uint32_t b_inc = block_inclusive_scan_256(b_val, s_warp_sums);
    const uint32_t b_exc = b_inc - b_val;

    const uint32_t rem = target_rank - chunk_prior;
    if (b_exc < rem && b_inc >= rem) {
        *s_out_bucket = static_cast<uint32_t>(target_bin);
        *s_out_prior = chunk_prior + b_exc;
    }
    __syncthreads();
}

__global__ __launch_bounds__(256) void select_hi16_bucket_kernel(uint32_t k) {
    __shared__ uint32_t s_bucket;
    __shared__ uint32_t s_prior;
    find_kth_bucket_65536(g_hist_hi, k, &s_bucket, &s_prior);
    if (threadIdx.x == 0) {
        g_state.target_hi16 = s_bucket;
        g_state.greater_hi_count = s_prior;
        g_state.write_idx = 0u;
    }
}

__device__ __forceinline__ void process_fast_key(uint32_t key, uint32_t target_hi) {
    const uint32_t hi = key >> 16;
    if (hi > target_hi) {
        const uint32_t pos = atomicAdd(&g_state.write_idx, 1u);
        g_fast_keys[pos] = key;
    } else if (hi == target_hi) {
        warp_aggregated_hist_inc(g_hist_lo, key & 0xffffu);
    }
}

template <bool kAligned>
__global__ __launch_bounds__(kThreads) void pass2_fast_collect_and_lo16_kernel(
    const float* __restrict__ input,
    int n4,
    int n) {
    const uint32_t target_hi = g_state.target_hi16;
    const int global_tid = blockIdx.x * kThreads + threadIdx.x;
    const int stride = gridDim.x * kThreads;

    if (kAligned) {
        const float4* __restrict__ in4 = reinterpret_cast<const float4*>(input);
        for (int idx = global_tid; idx < n4; idx += stride) {
            const float4 v = __ldg(&in4[idx]);
            process_fast_key(float_to_ordered_uint(v.x), target_hi);
            process_fast_key(float_to_ordered_uint(v.y), target_hi);
            process_fast_key(float_to_ordered_uint(v.z), target_hi);
            process_fast_key(float_to_ordered_uint(v.w), target_hi);
        }
        for (int i = (n4 << 2) + global_tid; i < n; i += stride) {
            process_fast_key(float_to_ordered_uint(__ldg(&input[i])), target_hi);
        }
    } else {
        for (int i = global_tid; i < n; i += stride) {
            process_fast_key(float_to_ordered_uint(__ldg(&input[i])), target_hi);
        }
    }
}

__global__ __launch_bounds__(256) void finalize_fast_topk_kernel(float* __restrict__ output,
                                                                 int k) {
    __shared__ uint32_t s_keys[kFastKThreshold];
    __shared__ uint32_t s_lo_bucket;
    __shared__ uint32_t s_lo_prior;
    __shared__ uint32_t s_fill_idx;

    const int tid = threadIdx.x;
    const uint32_t target_hi = g_state.target_hi16;
    const uint32_t greater_hi = g_state.greater_hi_count;
    const uint32_t rem = static_cast<uint32_t>(k) - greater_hi;

    int padded_k = 1;
    while (padded_k < k) {
        padded_k <<= 1;
    }
    for (int i = tid; i < padded_k; i += 256) {
        s_keys[i] = (i < static_cast<int>(greater_hi)) ? g_fast_keys[i] : 0u;
    }
    if (tid == 0) {
        s_fill_idx = greater_hi;
    }
    __syncthreads();

    if (rem > 0u) {
        find_kth_bucket_65536(g_hist_lo, rem, &s_lo_bucket, &s_lo_prior);

        const uint32_t thresh_lo = s_lo_bucket;
        const uint32_t strictly_above_lo = s_lo_prior;
        const uint32_t hi_prefix = target_hi << 16;

        for (int lo = static_cast<int>(thresh_lo) + 1 + tid; lo < kRadixBins; lo += 256) {
            const uint32_t cnt = g_hist_lo[lo];
            if (cnt > 0u) {
                const uint32_t base = atomicAdd(&s_fill_idx, cnt);
                const uint32_t full_key = hi_prefix | static_cast<uint32_t>(lo);
                for (uint32_t c = 0u; c < cnt; ++c) {
                    s_keys[base + c] = full_key;
                }
            }
        }
        __syncthreads();

        const uint32_t tie_start = greater_hi + strictly_above_lo;
        const uint32_t thresh_full_key = hi_prefix | thresh_lo;
        for (uint32_t pos = tie_start + static_cast<uint32_t>(tid);
             pos < static_cast<uint32_t>(k);
             pos += 256u) {
            s_keys[pos] = thresh_full_key;
        }
        __syncthreads();
    }

    bitonic_sort_desc_smem_256(s_keys, padded_k);

    for (int i = tid; i < k; i += 256) {
        output[i] = ordered_uint_to_float(s_keys[i]);
    }
}

template <bool kAligned>
__global__ __launch_bounds__(kThreads) void pass2_largek_lo16_kernel(
    const float* __restrict__ input,
    int n4,
    int n) {
    const uint32_t target_hi = g_state.target_hi16;
    const int global_tid = blockIdx.x * kThreads + threadIdx.x;
    const int stride = gridDim.x * kThreads;

    if (kAligned) {
        const float4* __restrict__ in4 = reinterpret_cast<const float4*>(input);
        for (int idx = global_tid; idx < n4; idx += stride) {
            const float4 v = __ldg(&in4[idx]);
            const uint32_t k0 = float_to_ordered_uint(v.x);
            const uint32_t k1 = float_to_ordered_uint(v.y);
            const uint32_t k2 = float_to_ordered_uint(v.z);
            const uint32_t k3 = float_to_ordered_uint(v.w);
            if ((k0 >> 16) == target_hi) warp_aggregated_hist_inc(g_hist_lo, k0 & 0xffffu);
            if ((k1 >> 16) == target_hi) warp_aggregated_hist_inc(g_hist_lo, k1 & 0xffffu);
            if ((k2 >> 16) == target_hi) warp_aggregated_hist_inc(g_hist_lo, k2 & 0xffffu);
            if ((k3 >> 16) == target_hi) warp_aggregated_hist_inc(g_hist_lo, k3 & 0xffffu);
        }
        for (int i = (n4 << 2) + global_tid; i < n; i += stride) {
            const uint32_t key = float_to_ordered_uint(__ldg(&input[i]));
            if ((key >> 16) == target_hi) warp_aggregated_hist_inc(g_hist_lo, key & 0xffffu);
        }
    } else {
        for (int i = global_tid; i < n; i += stride) {
            const uint32_t key = float_to_ordered_uint(__ldg(&input[i]));
            if ((key >> 16) == target_hi) warp_aggregated_hist_inc(g_hist_lo, key & 0xffffu);
        }
    }
}

__global__ __launch_bounds__(256) void select_lo16_bucket_kernel(uint32_t k) {
    __shared__ uint32_t s_lo_bucket;
    __shared__ uint32_t s_lo_prior;
    const uint32_t rem = k - g_state.greater_hi_count;
    if (rem > 0u) {
        find_kth_bucket_65536(g_hist_lo, rem, &s_lo_bucket, &s_lo_prior);
        if (threadIdx.x == 0) {
            g_state.threshold_key = (g_state.target_hi16 << 16) | s_lo_bucket;
            g_state.strictly_greater_count = g_state.greater_hi_count + s_lo_prior;
            g_state.write_idx = 0u;
        }
    } else if (threadIdx.x == 0) {
        g_state.threshold_key = (g_state.target_hi16 << 16);
        g_state.strictly_greater_count = g_state.greater_hi_count;
        g_state.write_idx = 0u;
    }
}

__global__ __launch_bounds__(kThreads) void collect_largek_keys_kernel(
    const float* __restrict__ input,
    uint32_t* __restrict__ out_keys,
    int n) {
    const uint32_t thresh = g_state.threshold_key;
    const int global_tid = blockIdx.x * kThreads + threadIdx.x;
    const int stride = gridDim.x * kThreads;

    for (int i = global_tid; i < n; i += stride) {
        const uint32_t key = float_to_ordered_uint(__ldg(&input[i]));
        if (key > thresh) {
            const uint32_t pos = atomicAdd(&g_state.write_idx, 1u);
            out_keys[pos] = key;
        }
    }
}

__global__ __launch_bounds__(256) void fill_largek_ties_kernel(uint32_t* __restrict__ out_keys,
                                                               int k) {
    const uint32_t start = g_state.strictly_greater_count;
    const uint32_t thresh = g_state.threshold_key;
    const int stride = blockDim.x * gridDim.x;
    for (uint32_t i = start + static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
         i < static_cast<uint32_t>(k);
         i += static_cast<uint32_t>(stride)) {
        out_keys[i] = thresh;
    }
}

__global__ __launch_bounds__(256) void bitonic_step_desc_kernel(uint32_t* __restrict__ keys,
                                                                int padded_k,
                                                                int j,
                                                                int stage_k) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= padded_k) {
        return;
    }
    const int ixj = i ^ j;
    if (ixj > i) {
        const uint32_t a = keys[i];
        const uint32_t b = keys[ixj];
        const bool descending = ((i & stage_k) == 0);
        if ((descending && a < b) || (!descending && a > b)) {
            keys[i] = b;
            keys[ixj] = a;
        }
    }
}

__global__ __launch_bounds__(256) void copy_and_pad_keys_kernel(const uint32_t* __restrict__ src,
                                                                uint32_t* __restrict__ dst,
                                                                int k,
                                                                int padded_k) {
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < padded_k; i += stride) {
        dst[i] = (i < k) ? src[i] : 0u;
    }
}

__global__ __launch_bounds__(256) void convert_keys_to_float_kernel(const uint32_t* __restrict__ keys,
                                                                    float* __restrict__ output,
                                                                    int k) {
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < k; i += stride) {
        output[i] = ordered_uint_to_float(keys[i]);
    }
}

}  // namespace

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {
    if (N <= 0 || k <= 0) {
        return;
    }
    if (k > N) {
        k = N;
    }

    // 1. Small-N single-block shared-memory sort (N <= 4096): 1 kernel launch total!
    if (N <= kSmallNThreshold) {
        topk_small_block_kernel<<<1, 512>>>(input, output, N, k);
        return;
    }

    // 2. Primary Single-Pass Streaming Block-Local Top-K Filter for k <= 256 (covers N=50M, k=100!):
    //    Reads the 200 MB input array ONCE at ~300 GB/s with zero global atomics!
    const uintptr_t in_addr = reinterpret_cast<uintptr_t>(input);
    if (k <= kStreamMaxK && ((in_addr & 3u) == 0u)) {
        const int mis_floats = static_cast<int>((in_addr & 15u) >> 2);
        const int peel = (4 - mis_floats) & 3;
        const int aligned_n4 = (N - peel) >> 2;
        int blocks = (aligned_n4 + 1023) / 1024;
        if (blocks < 1) blocks = 1;
        if (blocks > kStreamBlocks) blocks = kStreamBlocks;

        topk_stream_pass1_kernel<<<blocks, 256>>>(input, peel, aligned_n4, N, k);
        topk_stream_merge_kernel<<<1, 256>>>(output, blocks, k);
        return;
    }

    // 3. Fallback 2-pass 16-bit radix histogram path for k > 256 (up to k = N)
    const bool aligned = ((in_addr & 15u) == 0u);
    const int n4 = N >> 2;
    int scan_blocks = ((N + 3) / 4 + kThreads - 1) / kThreads;
    if (scan_blocks < 1) scan_blocks = 1;
    if (scan_blocks > kMaxBlocks) scan_blocks = kMaxBlocks;

    clear_hists_kernel<<<64, 256>>>();

    if (aligned) {
        pass1_hist_hi16_kernel<true><<<scan_blocks, kThreads>>>(input, n4, N);
    } else {
        pass1_hist_hi16_kernel<false><<<scan_blocks, kThreads>>>(input, n4, N);
    }

    select_hi16_bucket_kernel<<<1, 256>>>(static_cast<uint32_t>(k));

    if (k <= kFastKThreshold) {
        if (aligned) {
            pass2_fast_collect_and_lo16_kernel<true><<<scan_blocks, kThreads>>>(input, n4, N);
        } else {
            pass2_fast_collect_and_lo16_kernel<false><<<scan_blocks, kThreads>>>(input, n4, N);
        }
        finalize_fast_topk_kernel<<<1, 256>>>(output, k);
        return;
    }

    if (aligned) {
        pass2_largek_lo16_kernel<true><<<scan_blocks, kThreads>>>(input, n4, N);
    } else {
        pass2_largek_lo16_kernel<false><<<scan_blocks, kThreads>>>(input, n4, N);
    }
    select_lo16_bucket_kernel<<<1, 256>>>(static_cast<uint32_t>(k));

    uint32_t* out_u32 = reinterpret_cast<uint32_t*>(output);
    collect_largek_keys_kernel<<<scan_blocks, kThreads>>>(input, out_u32, N);

    int fill_blocks = (k + 255) / 256;
    if (fill_blocks > 1024) fill_blocks = 1024;
    fill_largek_ties_kernel<<<fill_blocks, 256>>>(out_u32, k);

    int padded_k = 1;
    while (padded_k < k) {
        padded_k <<= 1;
    }

    if (padded_k == k) {
        const int sort_blocks = (padded_k + 255) / 256;
        for (int stage_k = 2; stage_k <= padded_k; stage_k <<= 1) {
            for (int j = stage_k >> 1; j > 0; j >>= 1) {
                bitonic_step_desc_kernel<<<sort_blocks, 256>>>(out_u32, padded_k, j, stage_k);
            }
        }
        convert_keys_to_float_kernel<<<fill_blocks, 256>>>(out_u32, output, k);
    } else {
        uint32_t* d_pad = nullptr;
        cudaMalloc(&d_pad, static_cast<size_t>(padded_k) * sizeof(uint32_t));
        int pad_blocks = (padded_k + 255) / 256;
        if (pad_blocks > 4096) pad_blocks = 4096;
        copy_and_pad_keys_kernel<<<pad_blocks, 256>>>(out_u32, d_pad, k, padded_k);

        const int sort_blocks = (padded_k + 255) / 256;
        for (int stage_k = 2; stage_k <= padded_k; stage_k <<= 1) {
            for (int j = stage_k >> 1; j > 0; j >>= 1) {
                bitonic_step_desc_kernel<<<sort_blocks, 256>>>(d_pad, padded_k, j, stage_k);
            }
        }
        convert_keys_to_float_kernel<<<fill_blocks, 256>>>(d_pad, output, k);
        cudaFree(d_pad);
    }
}
