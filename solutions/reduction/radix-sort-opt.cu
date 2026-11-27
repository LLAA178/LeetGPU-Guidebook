#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

#ifndef ONESWEEP_ITEMS
#define ONESWEEP_ITEMS 16
#endif

static_assert(ONESWEEP_ITEMS >= 16 && ONESWEEP_ITEMS <= 24, "ONESWEEP_ITEMS must be in [16, 24]");

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarp;  // 8 warps
constexpr int kBins = 256;
constexpr int kSmallMaxN = 2048;

constexpr int kItemsPerThread = ONESWEEP_ITEMS;
constexpr int kWarpTile = kWarp * kItemsPerThread;
constexpr int kTileSize = kThreads * kItemsPerThread;  // 4,096 keys/tile when ONESWEEP_ITEMS=16

constexpr int kMaxN = 100000000;
constexpr int kHistBlocks = 160;                       // 4 resident blocks/SM across 40 T4 SMs
constexpr unsigned kFullMask = 0xffffffffu;

// Step execution modes planned dynamically on-device after single-pass global histogram
constexpr unsigned kModeSkip = 0u;
constexpr unsigned kModeSort = 1u;
constexpr unsigned kModeCopy = 2u;

// Buffer selectors: 0 = input, 1 = output, 2 = g_temp
struct __align__(16) StepPlan {
    unsigned mode;
    unsigned shift;
    unsigned src_sel;
    unsigned dst_sel;
    unsigned epoch;
    unsigned pad0;
    unsigned pad1;
    unsigned pad2;
};

struct __align__(8) LookbackWord {
    unsigned tag;
    unsigned val;
};

// Persistent heap workspace (zero static BSS bloat so shared-library dlopen inside
// memory-capped sandboxes always succeeds; allocated at most once on large N > 2048!)
__device__ uint32_t g_raw_hist[4][kBins];
__device__ uint32_t g_step_bin_base[4][kBins];
__device__ StepPlan g_step_plan[4];
__device__ unsigned g_step_tile_ctr[4];
__device__ unsigned g_hist_done = 0u;
__device__ unsigned g_epoch_base = 1u;

__device__ __forceinline__ LookbackWord load_lookback(const LookbackWord* ptr) {
    unsigned long long raw;
#if __CUDA_ARCH__ >= 700
    asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(raw) : "l"(ptr) : "memory");
#else
    raw = *reinterpret_cast<const volatile unsigned long long*>(ptr);
#endif
    LookbackWord w;
    w.tag = static_cast<unsigned>(raw & 0xffffffffull);
    w.val = static_cast<unsigned>(raw >> 32);
    return w;
}

__device__ __forceinline__ void store_lookback(LookbackWord* ptr, unsigned tag, unsigned val) {
    const unsigned long long raw =
        static_cast<unsigned long long>(tag) |
        (static_cast<unsigned long long>(val) << 32);
#if __CUDA_ARCH__ >= 700
    asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" :: "l"(ptr), "l"(raw) : "memory");
#else
    *reinterpret_cast<volatile unsigned long long*>(ptr) = raw;
#endif
}

__device__ __forceinline__ uint32_t warp_inclusive_scan_u32(uint32_t val, int lane) {
    #pragma unroll
    for (int step = 1; step < kWarp; step <<= 1) {
        const uint32_t prev = __shfl_up_sync(kFullMask, val, step);
        if (lane >= step) val += prev;
    }
    return val;
}

// ============================================================================
// 1. Single-Block Shared-Memory Radix Sort Fast Path for 1 <= N <= 4096
// Sorts all active 8-bit passes strictly inside shared memory in 1 launch!
// ============================================================================
__global__ __launch_bounds__(kThreads) void radix_sort_small_single_block_kernel(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    int n) {
    __shared__ uint32_t s_buf[2][kSmallMaxN];
    __shared__ uint32_t s_warp_hist[kWarps][kBins];
    __shared__ uint32_t s_warp_scan[kWarps];
    __shared__ uint32_t s_diff_bits;

    const int tid = threadIdx.x;
    const int lane = tid & (kWarp - 1);
    const int warp = tid / kWarp;

    if (tid == 0) {
        s_diff_bits = 0u;
    }
    __syncthreads();

    const uint32_t ref_key = __ldg(&input[0]);
    uint32_t thread_diff = 0u;
    #pragma unroll
    for (int i = tid; i < kSmallMaxN; i += kThreads) {
        if (i < n) {
            const uint32_t v = __ldg(&input[i]);
            s_buf[0][i] = v;
            thread_diff |= (v ^ ref_key);
        }
    }

    #pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        thread_diff |= __shfl_xor_sync(kFullMask, thread_diff, offset);
    }
    if (lane == 0 && thread_diff != 0u) {
        atomicOr(&s_diff_bits, thread_diff);
    }
    __syncthreads();

    const uint32_t diff_bits = s_diff_bits;
    int cur = 0;

    #pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = pass * 8;
        if (((diff_bits >> shift) & 0xffu) != 0u) {
            #pragma unroll
            for (int w = 0; w < kWarps; ++w) {
                s_warp_hist[w][tid] = 0u;
            }
            __syncthreads();

            uint32_t keys[8];
            uint16_t ranks[8];
            const int warp_base = warp * 256;

            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int idx = warp_base + r * kWarp + lane;
                const bool ok = (idx < n);
                const uint32_t key = ok ? s_buf[cur][idx] : 0u;
                keys[r] = key;
                const unsigned match_val =
                    ok ? ((key >> shift) & 0xffu) : (256u + static_cast<unsigned>(lane));
                const unsigned peer_mask = __match_any_sync(kFullMask, match_val);
                const unsigned peer_rank =
                    static_cast<unsigned>(__popc(peer_mask & ((1u << lane) - 1u)));
                const int leader = __ffs(peer_mask) - 1;
                unsigned wb = 0u;
                if (ok && lane == leader) {
                    wb = s_warp_hist[warp][match_val];
                    s_warp_hist[warp][match_val] = wb + static_cast<unsigned>(__popc(peer_mask));
                }
                wb = __shfl_sync(kFullMask, wb, leader);
                ranks[r] = static_cast<uint16_t>(wb + peer_rank);
            }
            __syncthreads();

            uint32_t warp_excl[kWarps];
            uint32_t bin_cnt = 0u;
            #pragma unroll
            for (int w = 0; w < kWarps; ++w) {
                const uint32_t c = s_warp_hist[w][tid];
                warp_excl[w] = bin_cnt;
                bin_cnt += c;
            }

            const uint32_t warp_incl = warp_inclusive_scan_u32(bin_cnt, lane);
            if (lane == kWarp - 1) {
                s_warp_scan[warp] = warp_incl;
            }
            __syncthreads();

            uint32_t block_excl = warp_incl - bin_cnt;
            #pragma unroll
            for (int w = 0; w < kWarps; ++w) {
                if (w < warp) block_excl += s_warp_scan[w];
            }
            #pragma unroll
            for (int w = 0; w < kWarps; ++w) {
                s_warp_hist[w][tid] = block_excl + warp_excl[w];
            }
            __syncthreads();

            const int nxt = cur ^ 1;
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int idx = warp_base + r * kWarp + lane;
                if (idx < n) {
                    const unsigned digit = (keys[r] >> shift) & 0xffu;
                    s_buf[nxt][s_warp_hist[warp][digit] + ranks[r]] = keys[r];
                }
            }
            cur = nxt;
            __syncthreads();
        }
    }

    for (int i = tid; i < n; i += kThreads) {
        output[i] = s_buf[cur][i];
    }
}

// ============================================================================
// 2. Single-Pass 4x256 Vectorized Histogram + On-Device Pass Planner
// Reads `input` ONCE (200 MB at N=50M), builds all 4 pass histograms, computes
// global bin offsets, skips any trivial pass, and sets ping-pong buffer plan!
// ============================================================================
__global__ __launch_bounds__(kThreads, 3) void radix_histogram_and_plan_kernel(
    const uint32_t* __restrict__ input,
    const uint32_t* __restrict__ output,
    int n) {
    // 2 privatized copies of 4 x 256 bins = 8 KiB shared memory -> 3-4 blocks/SM!
    __shared__ uint32_t s_hist[2][4][kBins];
    __shared__ uint32_t s_warp_scan[kWarps];
    __shared__ unsigned s_trivial_mask;
    __shared__ bool s_is_last;

    const int tid = threadIdx.x;
    const int lane = tid & (kWarp - 1);
    const int warp = tid / kWarp;
    const int copy_id = warp & 1;

    #pragma unroll
    for (int c = 0; c < 2; ++c) {
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            s_hist[c][p][tid] = 0u;
        }
    }
    __syncthreads();

    // Check 16B alignment of `input` so misaligned pointers fall back cleanly.
    // Round `n4_vec` down to a multiple of `kWarp` (32) so `__shfl_sync` / `__all_sync` are 100% warp-uniform!
    const uintptr_t addr = reinterpret_cast<uintptr_t>(input);
    if ((addr & 15u) == 0u) {
        const int n4 = n >> 2;
        const int n4_vec = n4 & ~(kWarp - 1);
        const uint4* __restrict__ in4 = reinterpret_cast<const uint4*>(input);
        const int gtid = blockIdx.x * kThreads + tid;
        const int gstride = gridDim.x * kThreads;

        int i4 = gtid;
        for (; i4 + 3 * gstride < n4_vec; i4 += 4 * gstride) {
            const uint4 v0 = __ldg(&in4[i4]);
            const uint4 v1 = __ldg(&in4[i4 + gstride]);
            const uint4 v2 = __ldg(&in4[i4 + 2 * gstride]);
            const uint4 v3 = __ldg(&in4[i4 + 3 * gstride]);

            const uint4 vs[4] = {v0, v1, v2, v3};
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const uint4 v = vs[u];
                const int same4 = (v.x == v.y) & (v.z == v.w) & (v.x == v.z);
                const uint32_t ref = __shfl_sync(kFullMask, v.x, 0);
                if (__all_sync(kFullMask, same4 & (v.x == ref))) {
                    if (lane == 0) {
                        atomicAdd(&s_hist[copy_id][0][ref & 0xffu], 128u);
                        atomicAdd(&s_hist[copy_id][1][(ref >> 8) & 0xffu], 128u);
                        atomicAdd(&s_hist[copy_id][2][(ref >> 16) & 0xffu], 128u);
                        atomicAdd(&s_hist[copy_id][3][ref >> 24], 128u);
                    }
                } else {
                    atomicAdd(&s_hist[copy_id][0][v.x & 0xffu], 1u);
                    atomicAdd(&s_hist[copy_id][0][v.y & 0xffu], 1u);
                    atomicAdd(&s_hist[copy_id][0][v.z & 0xffu], 1u);
                    atomicAdd(&s_hist[copy_id][0][v.w & 0xffu], 1u);

                    atomicAdd(&s_hist[copy_id][1][(v.x >> 8) & 0xffu], 1u);
                    atomicAdd(&s_hist[copy_id][1][(v.y >> 8) & 0xffu], 1u);
                    atomicAdd(&s_hist[copy_id][1][(v.z >> 8) & 0xffu], 1u);
                    atomicAdd(&s_hist[copy_id][1][(v.w >> 8) & 0xffu], 1u);

                    atomicAdd(&s_hist[copy_id][2][(v.x >> 16) & 0xffu], 1u);
                    atomicAdd(&s_hist[copy_id][2][(v.y >> 16) & 0xffu], 1u);
                    atomicAdd(&s_hist[copy_id][2][(v.z >> 16) & 0xffu], 1u);
                    atomicAdd(&s_hist[copy_id][2][(v.w >> 16) & 0xffu], 1u);

                    atomicAdd(&s_hist[copy_id][3][v.x >> 24], 1u);
                    atomicAdd(&s_hist[copy_id][3][v.y >> 24], 1u);
                    atomicAdd(&s_hist[copy_id][3][v.z >> 24], 1u);
                    atomicAdd(&s_hist[copy_id][3][v.w >> 24], 1u);
                }
            }
        }
        for (; i4 < n4_vec; i4 += gstride) {
            const uint4 v = __ldg(&in4[i4]);
            const int same4 = (v.x == v.y) & (v.z == v.w) & (v.x == v.z);
            const uint32_t ref = __shfl_sync(kFullMask, v.x, 0);
            if (__all_sync(kFullMask, same4 & (v.x == ref))) {
                if (lane == 0) {
                    atomicAdd(&s_hist[copy_id][0][ref & 0xffu], 128u);
                    atomicAdd(&s_hist[copy_id][1][(ref >> 8) & 0xffu], 128u);
                    atomicAdd(&s_hist[copy_id][2][(ref >> 16) & 0xffu], 128u);
                    atomicAdd(&s_hist[copy_id][3][ref >> 24], 128u);
                }
            } else {
                atomicAdd(&s_hist[copy_id][0][v.x & 0xffu], 1u);
                atomicAdd(&s_hist[copy_id][0][v.y & 0xffu], 1u);
                atomicAdd(&s_hist[copy_id][0][v.z & 0xffu], 1u);
                atomicAdd(&s_hist[copy_id][0][v.w & 0xffu], 1u);

                atomicAdd(&s_hist[copy_id][1][(v.x >> 8) & 0xffu], 1u);
                atomicAdd(&s_hist[copy_id][1][(v.y >> 8) & 0xffu], 1u);
                atomicAdd(&s_hist[copy_id][1][(v.z >> 8) & 0xffu], 1u);
                atomicAdd(&s_hist[copy_id][1][(v.w >> 8) & 0xffu], 1u);

                atomicAdd(&s_hist[copy_id][2][(v.x >> 16) & 0xffu], 1u);
                atomicAdd(&s_hist[copy_id][2][(v.y >> 16) & 0xffu], 1u);
                atomicAdd(&s_hist[copy_id][2][(v.z >> 16) & 0xffu], 1u);
                atomicAdd(&s_hist[copy_id][2][(v.w >> 16) & 0xffu], 1u);

                atomicAdd(&s_hist[copy_id][3][v.x >> 24], 1u);
                atomicAdd(&s_hist[copy_id][3][v.y >> 24], 1u);
                atomicAdd(&s_hist[copy_id][3][v.z >> 24], 1u);
                atomicAdd(&s_hist[copy_id][3][v.w >> 24], 1u);
            }
        }
        for (int i = (n4_vec << 2) + gtid; i < n; i += gstride) {
            const uint32_t x = __ldg(&input[i]);
            atomicAdd(&s_hist[copy_id][0][x & 0xffu], 1u);
            atomicAdd(&s_hist[copy_id][1][(x >> 8) & 0xffu], 1u);
            atomicAdd(&s_hist[copy_id][2][(x >> 16) & 0xffu], 1u);
            atomicAdd(&s_hist[copy_id][3][x >> 24], 1u);
        }
    } else {
        const int gtid = blockIdx.x * kThreads + tid;
        const int gstride = gridDim.x * kThreads;
        for (int i = gtid; i < n; i += gstride) {
            const uint32_t x = __ldg(&input[i]);
            atomicAdd(&s_hist[copy_id][0][x & 0xffu], 1u);
            atomicAdd(&s_hist[copy_id][1][(x >> 8) & 0xffu], 1u);
            atomicAdd(&s_hist[copy_id][2][(x >> 16) & 0xffu], 1u);
            atomicAdd(&s_hist[copy_id][3][x >> 24], 1u);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint32_t sum = s_hist[0][p][tid] + s_hist[1][p][tid];
        if (sum != 0u) {
            atomicAdd(&g_raw_hist[p][tid], sum);
        }
    }
    __syncthreads();

    if (tid == 0) {
        __threadfence();
        const unsigned prev = atomicAdd(&g_hist_done, 1u);
        s_is_last = (prev + 1u == gridDim.x);
        s_trivial_mask = 0u;
    }
    __syncthreads();
    if (!s_is_last) {
        return;
    }

    __threadfence();
    __syncthreads();

    // Last finishing block reads & zeroes g_raw_hist[0..3][tid] atomically at L2,
    // computes exclusive prefix sums, detects trivial passes, and writes g_step_plan[0..3]!
    uint32_t excl[4];
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        const uint32_t c = atomicExch(&g_raw_hist[p][tid], 0u);
        if (c == static_cast<uint32_t>(n)) {
            atomicOr(&s_trivial_mask, 1u << p);
        }
        const uint32_t warp_incl = warp_inclusive_scan_u32(c, lane);
        if (lane == kWarp - 1) {
            s_warp_scan[warp] = warp_incl;
        }
        __syncthreads();

        uint32_t b_excl = warp_incl - c;
        #pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            if (w < warp) b_excl += s_warp_scan[w];
        }
        excl[p] = b_excl;
        __syncthreads();
    }

    unsigned triv = s_trivial_mask;
    const bool in_place = (input == output);
    if (in_place) {
        // Keep active count even (0 or 2 or 4) if input == output so src != dst on every SORT step
        const int non_triv_cnt = 4 - __popc(triv & 0xfu);
        if (non_triv_cnt == 1 || non_triv_cnt == 3) {
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                if ((triv >> p) & 1u) {
                    triv &= ~(1u << p);
                    break;
                }
            }
        }
    }

    int active_passes[4];
    int num_active = 0;
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        if (((triv >> p) & 1u) == 0u) {
            active_passes[num_active++] = p;
        }
    }

    for (int s = 0; s < num_active; ++s) {
        g_step_bin_base[s][tid] = excl[active_passes[s]];
    }

    if (tid == 0) {
        const unsigned epoch0 = g_epoch_base;
        g_epoch_base = epoch0 + 4u;
        atomicExch(&g_hist_done, 0u);

        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            g_step_tile_ctr[s] = 0u;
            StepPlan plan{};
            plan.epoch = epoch0 + static_cast<unsigned>(s);
            if (num_active == 0) {
                if (s == 0 && !in_place) {
                    plan.mode = kModeCopy;
                    plan.src_sel = 0u;
                    plan.dst_sel = 1u;
                } else {
                    plan.mode = kModeSkip;
                }
            } else if (s < num_active) {
                const unsigned dst_sel = (((num_active - 1 - s) & 1) != 0) ? 2u : 1u;
                const unsigned src_sel = (s == 0) ? 0u : (3u - dst_sel);
                plan.mode = kModeSort;
                plan.shift = static_cast<unsigned>(active_passes[s] * 8);
                plan.src_sel = src_sel;
                plan.dst_sel = dst_sel;
            } else {
                plan.mode = kModeSkip;
            }
            g_step_plan[s] = plan;
        }
    }
}

// ============================================================================
// 3. Onesweep Pass Kernel (__match_any_sync + 64-bit Epoch Lookback + Smem Scatter)
// ============================================================================
template <int kStep>
__global__ __launch_bounds__(kThreads, 2) void radix_onesweep_step_kernel(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    uint32_t* __restrict__ g_temp,
    LookbackWord* __restrict__ g_lookback,
    int n,
    int num_tiles) {
    __shared__ uint32_t s_keys[kTileSize];
    __shared__ uint32_t s_warp_hist[kWarps][kBins];
    __shared__ uint32_t s_global_bin_offset[kBins];
    __shared__ uint32_t s_warp_scan[kWarps];
    __shared__ StepPlan s_plan;
    __shared__ int s_tile_id;

    const int tid = threadIdx.x;
    const int lane = tid & (kWarp - 1);
    const int warp = tid / kWarp;

    if (tid == 0) {
        const StepPlan p = g_step_plan[kStep];
        s_plan = p;
        if (p.mode == kModeSort) {
            s_tile_id = static_cast<int>(atomicAdd(&g_step_tile_ctr[kStep], 1u));
        } else {
            s_tile_id = static_cast<int>(blockIdx.x);
        }
    }
    __syncthreads();

    const unsigned mode = s_plan.mode;
    if (mode == kModeSkip) {
        return;
    }

    if (mode == kModeCopy) {
        const size_t base = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(kTileSize);
        for (int i = tid; i < kTileSize; i += kThreads) {
            const size_t idx = base + static_cast<size_t>(i);
            if (idx < static_cast<size_t>(n)) {
                output[idx] = __ldg(&input[idx]);
            }
        }
        return;
    }

    // kModeSort: Onesweep single-pass digit rank + decoupled lookback + coalesced scatter
    const int tile_id = s_tile_id;
    if (tile_id >= num_tiles) {
        return;
    }

    const uint32_t* src =
        (s_plan.src_sel == 0u) ? input : ((s_plan.src_sel == 1u) ? output : g_temp);
    uint32_t* dst = (s_plan.dst_sel == 1u) ? output : g_temp;
    const unsigned shift = s_plan.shift;
    const unsigned epoch = s_plan.epoch;
    const unsigned tag_agg = (epoch << 2) | 1u;
    const unsigned tag_pfx = (epoch << 2) | 2u;

    const int tile_base = tile_id * kTileSize;
    const int rem_n = n - tile_base;
    const int valid_n = (rem_n < kTileSize) ? rem_n : kTileSize;
    const bool full_tile = (valid_n == kTileSize);

    // Phase A: Zero warp-private histograms + issue all 16 stride-1 coalesced global loads upfront
    #pragma unroll
    for (int w = 0; w < kWarps; ++w) {
        s_warp_hist[w][tid] = 0u;
    }

    uint32_t keys[kItemsPerThread];
    uint16_t ranks[kItemsPerThread];
    const int warp_offset = warp * kWarpTile;

    if (full_tile) {
        const uint32_t* warp_src = src + static_cast<size_t>(tile_base + warp_offset + lane);
        #pragma unroll
        for (int r = 0; r < kItemsPerThread; ++r) {
            keys[r] = __ldg(&warp_src[r * kWarp]);
        }
    } else {
        #pragma unroll
        for (int r = 0; r < kItemsPerThread; ++r) {
            const int local_idx = warp_offset + r * kWarp + lane;
            keys[r] = (local_idx < valid_n) ? __ldg(&src[tile_base + local_idx]) : 0u;
        }
    }
    __syncthreads();

    // Rank digits stably inside each warp using hardware __match_any_sync
    const unsigned lt_mask = (1u << lane) - 1u;
    if (full_tile) {
        #pragma unroll
        for (int r = 0; r < kItemsPerThread; ++r) {
            const unsigned digit = (keys[r] >> shift) & 0xffu;
            const unsigned peer_mask = __match_any_sync(kFullMask, digit);
            const unsigned peer_rank = static_cast<unsigned>(__popc(peer_mask & lt_mask));
            const int leader = __ffs(peer_mask) - 1;
            unsigned wb = 0u;
            if (lane == leader) {
                wb = s_warp_hist[warp][digit];
                s_warp_hist[warp][digit] = wb + static_cast<unsigned>(__popc(peer_mask));
            }
            wb = __shfl_sync(kFullMask, wb, leader);
            ranks[r] = static_cast<uint16_t>(wb + peer_rank);
        }
    } else {
        #pragma unroll
        for (int r = 0; r < kItemsPerThread; ++r) {
            const int local_idx = warp_offset + r * kWarp + lane;
            const bool ok = (local_idx < valid_n);
            const unsigned match_val =
                ok ? ((keys[r] >> shift) & 0xffu) : (256u + static_cast<unsigned>(lane));
            const unsigned peer_mask = __match_any_sync(kFullMask, match_val);
            const unsigned peer_rank = static_cast<unsigned>(__popc(peer_mask & lt_mask));
            const int leader = __ffs(peer_mask) - 1;
            unsigned wb = 0u;
            if (ok && lane == leader) {
                wb = s_warp_hist[warp][match_val];
                s_warp_hist[warp][match_val] = wb + static_cast<unsigned>(__popc(peer_mask));
            }
            wb = __shfl_sync(kFullMask, wb, leader);
            ranks[r] = static_cast<uint16_t>(wb + peer_rank);
        }
    }
    __syncthreads();

    // Phase B: Thread `tid` owns bin `tid` (0..255) -> scan warp counts + publish lookback aggregate
    uint32_t warp_excl[kWarps];
    uint32_t tile_bin_count = 0u;
    #pragma unroll
    for (int w = 0; w < kWarps; ++w) {
        const uint32_t c = s_warp_hist[w][tid];
        warp_excl[w] = tile_bin_count;
        tile_bin_count += c;
    }

    // Publish lookback state IMMEDIATELY before block scan to minimize downstream lookback latency!
    LookbackWord* my_lb = &g_lookback[static_cast<size_t>(tile_id) * kBins + tid];
    if (tile_id == 0) {
        store_lookback(my_lb, tag_pfx, tile_bin_count);
    } else {
        store_lookback(my_lb, tag_agg, tile_bin_count);
    }

    const uint32_t warp_incl = warp_inclusive_scan_u32(tile_bin_count, lane);
    if (lane == kWarp - 1) {
        s_warp_scan[warp] = warp_incl;
    }
    __syncthreads();

    uint32_t tile_bin_excl = warp_incl - tile_bin_count;
    #pragma unroll
    for (int w = 0; w < kWarps; ++w) {
        if (w < warp) tile_bin_excl += s_warp_scan[w];
    }
    #pragma unroll
    for (int w = 0; w < kWarps; ++w) {
        s_warp_hist[w][tid] = tile_bin_excl + warp_excl[w];
    }

    // Phase C: Parallel 256-bin decoupled lookback across preceding tiles
    uint32_t excl_tile_prefix = 0u;
    if (tile_id > 0) {
        int prev = tile_id - 1;
        while (prev >= 0) {
            const LookbackWord* prev_lb = &g_lookback[static_cast<size_t>(prev) * kBins + tid];
            LookbackWord st = load_lookback(prev_lb);
            while (st.tag != tag_pfx && st.tag != tag_agg) {
                st = load_lookback(prev_lb);
            }
            excl_tile_prefix += st.val;
            if (st.tag == tag_pfx) {
                break;
            }
            --prev;
        }
        store_lookback(my_lb, tag_pfx, excl_tile_prefix + tile_bin_count);
    }

    // Pre-subtract tile_bin_excl (mod 2^32) so Phase D only needs 1 add (`offset + i`)!
    s_global_bin_offset[tid] = g_step_bin_base[kStep][tid] + excl_tile_prefix - tile_bin_excl;
    __syncthreads();

    // Phase D: Stage keys in shared memory in sorted bin order, then write coalesced runs to dst!
    if (full_tile) {
        #pragma unroll
        for (int r = 0; r < kItemsPerThread; ++r) {
            const unsigned digit = (keys[r] >> shift) & 0xffu;
            s_keys[s_warp_hist[warp][digit] + ranks[r]] = keys[r];
        }
    } else {
        #pragma unroll
        for (int r = 0; r < kItemsPerThread; ++r) {
            const int local_idx = warp_offset + r * kWarp + lane;
            if (local_idx < valid_n) {
                const unsigned digit = (keys[r] >> shift) & 0xffu;
                s_keys[s_warp_hist[warp][digit] + ranks[r]] = keys[r];
            }
        }
    }
    __syncthreads();

    if (full_tile) {
        #pragma unroll
        for (int r = 0; r < kItemsPerThread; ++r) {
            const int i = r * kThreads + tid;
            const uint32_t key = s_keys[i];
            const unsigned digit = (key >> shift) & 0xffu;
            dst[s_global_bin_offset[digit] + static_cast<uint32_t>(i)] = key;
        }
    } else {
        for (int i = tid; i < valid_n; i += kThreads) {
            const uint32_t key = s_keys[i];
            const unsigned digit = (key >> shift) & 0xffu;
            dst[s_global_bin_offset[digit] + static_cast<uint32_t>(i)] = key;
        }
    }
}

}  // namespace

// input, output are device pointers
extern "C" void solve(const unsigned int* input, unsigned int* output, int N) {
    if (N <= 0) {
        return;
    }

    if (N <= kSmallMaxN) {
        radix_sort_small_single_block_kernel<<<1, kThreads>>>(input, output, N);
        return;
    }

    const int num_tiles = (N + kTileSize - 1) / kTileSize;
    const int hist_blocks = (num_tiles < kHistBlocks) ? num_tiles : kHistBlocks;

    // Lazy persistent workspace (zero static BSS -> instant shared-library dlopen,
    // allocated at most once when N grows, never freed inside timed loops!)
    static uint32_t* s_temp = nullptr;
    static size_t s_temp_cap = 0;
    static LookbackWord* s_lookback = nullptr;
    static size_t s_lb_cap = 0;

    if (static_cast<size_t>(N) > s_temp_cap) {
        if (s_temp != nullptr) {
            cudaFree(s_temp);
            s_temp = nullptr;
        }
        const size_t alloc_n = (static_cast<size_t>(N) < 50000000u) ? 50000000u : static_cast<size_t>(N);
        if (cudaMalloc(&s_temp, alloc_n * sizeof(uint32_t)) != cudaSuccess) {
            cudaGetLastError();
            if (cudaMalloc(&s_temp, static_cast<size_t>(N) * sizeof(uint32_t)) != cudaSuccess) {
                return;
            }
            s_temp_cap = static_cast<size_t>(N);
        } else {
            s_temp_cap = alloc_n;
        }
    }

    const size_t need_lb = static_cast<size_t>(num_tiles) * static_cast<size_t>(kBins);
    if (need_lb > s_lb_cap) {
        if (s_lookback != nullptr) {
            cudaFree(s_lookback);
            s_lookback = nullptr;
        }
        const size_t min_lb = static_cast<size_t>((50000000 + kTileSize - 1) / kTileSize) * kBins;
        const size_t alloc_lb = (need_lb < min_lb) ? min_lb : need_lb;
        if (cudaMalloc(&s_lookback, alloc_lb * sizeof(LookbackWord)) != cudaSuccess) {
            cudaGetLastError();
            if (cudaMalloc(&s_lookback, need_lb * sizeof(LookbackWord)) != cudaSuccess) {
                return;
            }
            s_lb_cap = need_lb;
        } else {
            s_lb_cap = alloc_lb;
        }
        cudaMemsetAsync(s_lookback, 0, s_lb_cap * sizeof(LookbackWord));
    }

    radix_histogram_and_plan_kernel<<<static_cast<unsigned>(hist_blocks), kThreads>>>(
        input, output, N);
    radix_onesweep_step_kernel<0><<<static_cast<unsigned>(num_tiles), kThreads>>>(
        input, output, s_temp, s_lookback, N, num_tiles);
    radix_onesweep_step_kernel<1><<<static_cast<unsigned>(num_tiles), kThreads>>>(
        input, output, s_temp, s_lookback, N, num_tiles);
    radix_onesweep_step_kernel<2><<<static_cast<unsigned>(num_tiles), kThreads>>>(
        input, output, s_temp, s_lookback, N, num_tiles);
    radix_onesweep_step_kernel<3><<<static_cast<unsigned>(num_tiles), kThreads>>>(
        input, output, s_temp, s_lookback, N, num_tiles);
}
