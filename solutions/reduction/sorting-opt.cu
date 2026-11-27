#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

#ifndef SORT_ITEMS
#define SORT_ITEMS 8
#endif

static_assert(SORT_ITEMS == 8 || SORT_ITEMS == 12 || SORT_ITEMS == 16,
              "SORT_ITEMS must be 8, 12, or 16");

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarp;  // 8 warps
constexpr int kBins = 256;
constexpr int kSmallMaxN = 2048;          // 24.0 KiB shared memory single-block path

constexpr int kItemsPerThread = SORT_ITEMS;
constexpr int kWarpTile = kWarp * kItemsPerThread;
constexpr int kTileSize = kThreads * kItemsPerThread;  // 4,096 (or 2,048 when SORT_ITEMS=8)

// Static device .bss kept compact (~5.0 MiB << 16 MiB sandbox limit!) for 1 <= N <= 1,000,000
constexpr int kMaxN = 1048576;
constexpr int kMaxTiles = (kMaxN + 2047) / 2048 + 16;  // 528 tiles -> 1.08 MiB lookback table
constexpr int kMaxHistBlocks = 80;                     // 2 blocks/SM across 40 T4 SMs
constexpr unsigned kFullMask = 0xffffffffu;

constexpr unsigned kModeSkip = 0u;
constexpr unsigned kModeSort = 1u;
constexpr unsigned kModeCopyK2F = 2u;  // Copy g_temp -> data converting sortable uint32 -> float

struct __align__(16) StepPlan {
    unsigned mode;
    unsigned shift;
    unsigned src_sel;  // 0 = data, 1 = g_temp
    unsigned dst_sel;  // 0 = data, 1 = g_temp
    unsigned src_f2k;  // 1 on first active SORT step (converts float -> sortable key on load)
    unsigned dst_k2f;  // 1 on final active SORT step writing directly to data (key -> float)
    unsigned epoch;
    unsigned pad0;
};

struct __align__(8) LookbackWord {
    unsigned tag;
    unsigned val;
};

__device__ __align__(16) uint32_t g_temp[kMaxN];
__device__ __align__(8) LookbackWord g_lookback[static_cast<size_t>(kMaxTiles) * kBins];
__device__ uint32_t g_raw_hist[4][kBins];
__device__ uint32_t g_step_bin_base[4][kBins];
__device__ StepPlan g_step_plan[4];
__device__ unsigned g_step_tile_ctr[4];
__device__ unsigned g_hist_done = 0u;
__device__ unsigned g_epoch_base = 1u;

// Order-preserving bijection: IEEE-754 float <-> unsigned 32-bit integer
__device__ __forceinline__ uint32_t float_to_key(float x) {
    const uint32_t u = __float_as_uint(x);
    const uint32_t mask = (u & 0x80000000u) ? 0xffffffffu : 0x80000000u;
    return u ^ mask;
}

__device__ __forceinline__ uint32_t bits_to_key(uint32_t u) {
    const uint32_t mask = (u & 0x80000000u) ? 0xffffffffu : 0x80000000u;
    return u ^ mask;
}

__device__ __forceinline__ uint32_t key_to_bits(uint32_t key) {
    const uint32_t mask = (key & 0x80000000u) ? 0x80000000u : 0xffffffffu;
    return key ^ mask;
}

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

__device__ __forceinline__ unsigned warp_match_9bit(unsigned val) {
#if __CUDA_ARCH__ >= 700
    return __match_any_sync(kFullMask, val);
#else
    unsigned mask = kFullMask;
    #pragma unroll
    for (int b = 0; b < 9; ++b) {
        const unsigned bit = (val >> b) & 1u;
        const unsigned bmask = __ballot_sync(kFullMask, bit);
        mask &= bit ? bmask : ~bmask;
    }
    return mask;
#endif
}

__device__ __forceinline__ unsigned warp_match_8bit(unsigned val) {
#if __CUDA_ARCH__ >= 700
    return __match_any_sync(kFullMask, val);
#else
    unsigned mask = kFullMask;
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        const unsigned bit = (val >> b) & 1u;
        const unsigned bmask = __ballot_sync(kFullMask, bit);
        mask &= bit ? bmask : ~bmask;
    }
    return mask;
#endif
}

// ============================================================================
// 1. Single-Block In-Place Shared-Memory Radix Sort for 1 <= N <= 2048
// Uses only 24.04 KiB shared memory (<= 32 KiB) and executes in 1 launch!
// ============================================================================
__global__ __launch_bounds__(kThreads) void float_sort_small_single_block_kernel(
    float* __restrict__ data,
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

    const uint32_t ref_key = float_to_key(__ldg(&data[0]));
    uint32_t thread_diff = 0u;
    #pragma unroll
    for (int i = tid; i < kSmallMaxN; i += kThreads) {
        if (i < n) {
            const uint32_t k = float_to_key(__ldg(&data[i]));
            s_buf[0][i] = k;
            thread_diff |= (k ^ ref_key);
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
    if (diff_bits == 0u) {
        return;  // Already sorted in-place!
    }

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
                const unsigned peer_mask = warp_match_9bit(match_val);
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

    uint32_t* __restrict__ udata = reinterpret_cast<uint32_t*>(data);
    for (int i = tid; i < n; i += kThreads) {
        udata[i] = key_to_bits(s_buf[cur][i]);
    }
}

// ============================================================================
// 2. Fused Float->Key 4x256 Vectorized Histogram + In-Place Pass Planner
// ============================================================================
__global__ __launch_bounds__(kThreads) void float_histogram_and_plan_kernel(
    const float* __restrict__ data,
    int n) {
    __shared__ uint32_t s_hist[2][4][kBins];  // 8 KiB shared memory
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

    const uint32_t* __restrict__ udata = reinterpret_cast<const uint32_t*>(data);
    const uintptr_t addr = reinterpret_cast<uintptr_t>(udata);

    if ((addr & 15u) == 0u) {
        const int n4 = n >> 2;
        const int n4_vec = n4 & ~(kWarp - 1);  // Exact multiple of 32 for uniform warp intrinsics
        const uint4* __restrict__ in4 = reinterpret_cast<const uint4*>(udata);
        const int gtid = blockIdx.x * kThreads + tid;
        const int gstride = gridDim.x * kThreads;

        int i4 = gtid;
        for (; i4 + 3 * gstride < n4_vec; i4 += 4 * gstride) {
            const uint4 r0 = __ldg(&in4[i4]);
            const uint4 r1 = __ldg(&in4[i4 + gstride]);
            const uint4 r2 = __ldg(&in4[i4 + 2 * gstride]);
            const uint4 r3 = __ldg(&in4[i4 + 3 * gstride]);

            const uint4 rs[4] = {r0, r1, r2, r3};
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const uint32_t kx = bits_to_key(rs[u].x);
                const uint32_t ky = bits_to_key(rs[u].y);
                const uint32_t kz = bits_to_key(rs[u].z);
                const uint32_t kw = bits_to_key(rs[u].w);

                const int same4 = (kx == ky) & (kz == kw) & (kx == kz);
                const uint32_t ref = __shfl_sync(kFullMask, kx, 0);
                if (__all_sync(kFullMask, same4 & (kx == ref))) {
                    if (lane == 0) {
                        atomicAdd(&s_hist[copy_id][0][ref & 0xffu], 128u);
                        atomicAdd(&s_hist[copy_id][1][(ref >> 8) & 0xffu], 128u);
                        atomicAdd(&s_hist[copy_id][2][(ref >> 16) & 0xffu], 128u);
                        atomicAdd(&s_hist[copy_id][3][ref >> 24], 128u);
                    }
                } else {
                    atomicAdd(&s_hist[copy_id][0][kx & 0xffu], 1u);
                    atomicAdd(&s_hist[copy_id][0][ky & 0xffu], 1u);
                    atomicAdd(&s_hist[copy_id][0][kz & 0xffu], 1u);
                    atomicAdd(&s_hist[copy_id][0][kw & 0xffu], 1u);

                    atomicAdd(&s_hist[copy_id][1][(kx >> 8) & 0xffu], 1u);
                    atomicAdd(&s_hist[copy_id][1][(ky >> 8) & 0xffu], 1u);
                    atomicAdd(&s_hist[copy_id][1][(kz >> 8) & 0xffu], 1u);
                    atomicAdd(&s_hist[copy_id][1][(kw >> 8) & 0xffu], 1u);

                    atomicAdd(&s_hist[copy_id][2][(kx >> 16) & 0xffu], 1u);
                    atomicAdd(&s_hist[copy_id][2][(ky >> 16) & 0xffu], 1u);
                    atomicAdd(&s_hist[copy_id][2][(kz >> 16) & 0xffu], 1u);
                    atomicAdd(&s_hist[copy_id][2][(kw >> 16) & 0xffu], 1u);

                    atomicAdd(&s_hist[copy_id][3][kx >> 24], 1u);
                    atomicAdd(&s_hist[copy_id][3][ky >> 24], 1u);
                    atomicAdd(&s_hist[copy_id][3][kz >> 24], 1u);
                    atomicAdd(&s_hist[copy_id][3][kw >> 24], 1u);
                }
            }
        }
        for (; i4 < n4_vec; i4 += gstride) {
            const uint4 rv = __ldg(&in4[i4]);
            const uint32_t kx = bits_to_key(rv.x);
            const uint32_t ky = bits_to_key(rv.y);
            const uint32_t kz = bits_to_key(rv.z);
            const uint32_t kw = bits_to_key(rv.w);

            const int same4 = (kx == ky) & (kz == kw) & (kx == kz);
            const uint32_t ref = __shfl_sync(kFullMask, kx, 0);
            if (__all_sync(kFullMask, same4 & (kx == ref))) {
                if (lane == 0) {
                    atomicAdd(&s_hist[copy_id][0][ref & 0xffu], 128u);
                    atomicAdd(&s_hist[copy_id][1][(ref >> 8) & 0xffu], 128u);
                    atomicAdd(&s_hist[copy_id][2][(ref >> 16) & 0xffu], 128u);
                    atomicAdd(&s_hist[copy_id][3][ref >> 24], 128u);
                }
            } else {
                atomicAdd(&s_hist[copy_id][0][kx & 0xffu], 1u);
                atomicAdd(&s_hist[copy_id][0][ky & 0xffu], 1u);
                atomicAdd(&s_hist[copy_id][0][kz & 0xffu], 1u);
                atomicAdd(&s_hist[copy_id][0][kw & 0xffu], 1u);

                atomicAdd(&s_hist[copy_id][1][(kx >> 8) & 0xffu], 1u);
                atomicAdd(&s_hist[copy_id][1][(ky >> 8) & 0xffu], 1u);
                atomicAdd(&s_hist[copy_id][1][(kz >> 8) & 0xffu], 1u);
                atomicAdd(&s_hist[copy_id][1][(kw >> 8) & 0xffu], 1u);

                atomicAdd(&s_hist[copy_id][2][(kx >> 16) & 0xffu], 1u);
                atomicAdd(&s_hist[copy_id][2][(ky >> 16) & 0xffu], 1u);
                atomicAdd(&s_hist[copy_id][2][(kz >> 16) & 0xffu], 1u);
                atomicAdd(&s_hist[copy_id][2][(kw >> 16) & 0xffu], 1u);

                atomicAdd(&s_hist[copy_id][3][kx >> 24], 1u);
                atomicAdd(&s_hist[copy_id][3][ky >> 24], 1u);
                atomicAdd(&s_hist[copy_id][3][kz >> 24], 1u);
                atomicAdd(&s_hist[copy_id][3][kw >> 24], 1u);
            }
        }
        for (int i = (n4_vec << 2) + gtid; i < n; i += gstride) {
            const uint32_t k = bits_to_key(__ldg(&udata[i]));
            atomicAdd(&s_hist[copy_id][0][k & 0xffu], 1u);
            atomicAdd(&s_hist[copy_id][1][(k >> 8) & 0xffu], 1u);
            atomicAdd(&s_hist[copy_id][2][(k >> 16) & 0xffu], 1u);
            atomicAdd(&s_hist[copy_id][3][k >> 24], 1u);
        }
    } else {
        const int gtid = blockIdx.x * kThreads + tid;
        const int gstride = gridDim.x * kThreads;
        for (int i = gtid; i < n; i += gstride) {
            const uint32_t k = bits_to_key(__ldg(&udata[i]));
            atomicAdd(&s_hist[copy_id][0][k & 0xffu], 1u);
            atomicAdd(&s_hist[copy_id][1][(k >> 8) & 0xffu], 1u);
            atomicAdd(&s_hist[copy_id][2][(k >> 16) & 0xffu], 1u);
            atomicAdd(&s_hist[copy_id][3][k >> 24], 1u);
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

    const unsigned triv = s_trivial_mask;
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
                plan.mode = kModeSkip;  // All keys identical -> already sorted in-place!
            } else if (s < num_active) {
                plan.mode = kModeSort;
                plan.shift = static_cast<unsigned>(active_passes[s] * 8);
                plan.src_sel = static_cast<unsigned>(s & 1);
                plan.dst_sel = static_cast<unsigned>((s & 1) ^ 1);
                plan.src_f2k = (s == 0) ? 1u : 0u;
                plan.dst_k2f = (((num_active & 1) == 0) && (s == num_active - 1)) ? 1u : 0u;
            } else if ((num_active & 1) == 1 && s == num_active) {
                plan.mode = kModeCopyK2F;  // Odd active passes ended in g_temp -> convert back to data
            } else {
                plan.mode = kModeSkip;
            }
            g_step_plan[s] = plan;
        }
    }
}

// ============================================================================
// 3. In-Place Fused Onesweep Step Kernel (<= 25.1 KiB shared memory)
// ============================================================================
template <int kStep>
__global__ __launch_bounds__(kThreads) void float_onesweep_step_kernel(
    float* __restrict__ data,
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

    uint32_t* __restrict__ udata = reinterpret_cast<uint32_t*>(data);

    if (mode == kModeCopyK2F) {
        const size_t base = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(kTileSize);
        for (int i = tid; i < kTileSize; i += kThreads) {
            const size_t idx = base + static_cast<size_t>(i);
            if (idx < static_cast<size_t>(n)) {
                udata[idx] = key_to_bits(__ldg(&g_temp[idx]));
            }
        }
        return;
    }

    const int tile_id = s_tile_id;
    if (tile_id >= num_tiles) {
        return;
    }

    const uint32_t* src = (s_plan.src_sel == 0u) ? udata : g_temp;
    uint32_t* dst = (s_plan.dst_sel == 0u) ? udata : g_temp;
    const bool src_f2k = (s_plan.src_f2k != 0u);
    const bool dst_k2f = (s_plan.dst_k2f != 0u);
    const unsigned shift = s_plan.shift;
    const unsigned epoch = s_plan.epoch;
    const unsigned tag_agg = (epoch << 2) | 1u;
    const unsigned tag_pfx = (epoch << 2) | 2u;

    const int tile_base = tile_id * kTileSize;
    const int rem_n = n - tile_base;
    const int valid_n = (rem_n < kTileSize) ? rem_n : kTileSize;
    const bool full_tile = (valid_n == kTileSize);

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
            const uint32_t raw = __ldg(&warp_src[r * kWarp]);
            keys[r] = src_f2k ? bits_to_key(raw) : raw;
        }
    } else {
        #pragma unroll
        for (int r = 0; r < kItemsPerThread; ++r) {
            const int local_idx = warp_offset + r * kWarp + lane;
            if (local_idx < valid_n) {
                const uint32_t raw = __ldg(&src[tile_base + local_idx]);
                keys[r] = src_f2k ? bits_to_key(raw) : raw;
            } else {
                keys[r] = 0u;
            }
        }
    }
    __syncthreads();

    const unsigned lt_mask = (1u << lane) - 1u;
    if (full_tile) {
        #pragma unroll
        for (int r = 0; r < kItemsPerThread; ++r) {
            const unsigned digit = (keys[r] >> shift) & 0xffu;
            const unsigned peer_mask = warp_match_8bit(digit);
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
            const unsigned peer_mask = warp_match_9bit(match_val);
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

    uint32_t warp_excl[kWarps];
    uint32_t tile_bin_count = 0u;
    #pragma unroll
    for (int w = 0; w < kWarps; ++w) {
        const uint32_t c = s_warp_hist[w][tid];
        warp_excl[w] = tile_bin_count;
        tile_bin_count += c;
    }

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

    s_global_bin_offset[tid] = g_step_bin_base[kStep][tid] + excl_tile_prefix - tile_bin_excl;
    __syncthreads();

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
        if (dst_k2f) {
            #pragma unroll
            for (int r = 0; r < kItemsPerThread; ++r) {
                const int i = r * kThreads + tid;
                const uint32_t key = s_keys[i];
                const unsigned digit = (key >> shift) & 0xffu;
                dst[s_global_bin_offset[digit] + static_cast<uint32_t>(i)] = key_to_bits(key);
            }
        } else {
            #pragma unroll
            for (int r = 0; r < kItemsPerThread; ++r) {
                const int i = r * kThreads + tid;
                const uint32_t key = s_keys[i];
                const unsigned digit = (key >> shift) & 0xffu;
                dst[s_global_bin_offset[digit] + static_cast<uint32_t>(i)] = key;
            }
        }
    } else {
        for (int i = tid; i < valid_n; i += kThreads) {
            const uint32_t key = s_keys[i];
            const unsigned digit = (key >> shift) & 0xffu;
            dst[s_global_bin_offset[digit] + static_cast<uint32_t>(i)] =
                dst_k2f ? key_to_bits(key) : key;
        }
    }
}

}  // namespace

// data is device pointer
extern "C" void solve(float* data, int N) {
    if (N <= 1) {
        return;
    }

    if (N <= kSmallMaxN) {
        float_sort_small_single_block_kernel<<<1, kThreads>>>(data, N);
        return;
    }

    const int num_tiles = (N + kTileSize - 1) / kTileSize;
    int hist_blocks = (N + 8191) / 8192;
    if (hist_blocks < 1) hist_blocks = 1;
    if (hist_blocks > kMaxHistBlocks) hist_blocks = kMaxHistBlocks;

    float_histogram_and_plan_kernel<<<static_cast<unsigned>(hist_blocks), kThreads>>>(data, N);
    float_onesweep_step_kernel<0><<<static_cast<unsigned>(num_tiles), kThreads>>>(
        data, N, num_tiles);
    float_onesweep_step_kernel<1><<<static_cast<unsigned>(num_tiles), kThreads>>>(
        data, N, num_tiles);
    float_onesweep_step_kernel<2><<<static_cast<unsigned>(num_tiles), kThreads>>>(
        data, N, num_tiles);
    float_onesweep_step_kernel<3><<<static_cast<unsigned>(num_tiles), kThreads>>>(
        data, N, num_tiles);
}
