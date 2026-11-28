#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

// Official constraints: 1 <= N <= 100,000; interaction radius r = 5.0 (r^2 = 25.0), alpha = 0.05.
// Total static .bss = ~2.85 MiB (<= 4 MB sandbox limit, fits inside T4's 4 MiB L2 cache!).
// All kernels strictly use 1-arg `__launch_bounds__(...)` for PTX JIT compatibility.
constexpr int kMaxN = 100000;
constexpr int kMaxBuckets = 65536;  // 256 x 256 max buckets
constexpr int kSmallThreshold = 512;
constexpr int kMaxBboxBlocks = 256;

constexpr float kRadiusSq = 25.0f;
constexpr float kAlpha = 0.05f;

// Module-scope static device workspace (referenced ONLY inside device kernels!)
__device__ float4 g_sorted_agents[kMaxN];      // 1.60 MB
__device__ int g_sorted_orig[kMaxN];           // 0.40 MB
__device__ uint16_t g_agent_bucket[kMaxN];     // 0.20 MB
__device__ int g_bucket_counts[kMaxBuckets];   // 0.26 MB
__device__ int g_bucket_starts[kMaxBuckets + 1];
__device__ int g_bucket_write[kMaxBuckets];

__device__ float g_bbox_partials[kMaxBboxBlocks * 4];
__device__ unsigned int g_bbox_done;

__device__ float g_origin_x;
__device__ float g_origin_y;
__device__ float g_inv_cell;
__device__ int g_grid_w;
__device__ int g_grid_h;
__device__ int g_num_buckets;
__device__ int g_use_hash;
__device__ int g_fallback;

__device__ __forceinline__ float exact_dist2d_sq(float x0, float y0, float x1, float y1) {
    const float dx = __fsub_rn(x0, x1);
    const float dy = __fsub_rn(y0, y1);
    return __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
}

__device__ __forceinline__ int hash_cell_2d(int cx, int cy, int mask) {
    const uint32_t ux = static_cast<uint32_t>(cx);
    const uint32_t uy = static_cast<uint32_t>(cy);
    const uint32_t h = (ux * 73856093u) ^ (uy * 19349663u);
    return static_cast<int>(h & static_cast<uint32_t>(mask));
}

// ============================================================================
// Shared-Memory Coalesced Tiled All-Pairs Device Helper (`128` threads)
//   - Used directly as a 1-launch fast path when `N < 512` (or `N > kMaxN`),
//     AND as a zero-host-sync fallback on GPU if all points are tightly
//     collapsed inside a single cell or span non-finite/extreme coordinates!
// ============================================================================
constexpr int kTileThreads = 128;

__device__ __forceinline__ void run_tiled_allpairs_block(
    const float* __restrict__ agents,
    float* __restrict__ agents_next,
    int N,
    bool aligned_out) {
    __shared__ float s_tile[kTileThreads * 4];  // 2 KiB shared staging buffer

    const int tid = static_cast<int>(threadIdx.x);
    const int i = static_cast<int>(blockIdx.x) * kTileThreads + tid;

    float px = 0.0f, py = 0.0f, vx = 0.0f, vy = 0.0f;
    if (i < N) {
        px = __ldg(&agents[i * 4 + 0]);
        py = __ldg(&agents[i * 4 + 1]);
        vx = __ldg(&agents[i * 4 + 2]);
        vy = __ldg(&agents[i * 4 + 3]);
    }

    double sum_vx = 0.0;
    double sum_vy = 0.0;
    int cnt = 0;

    for (int tile_base = 0; tile_base < N; tile_base += kTileThreads) {
        const int count = ((N - tile_base) < kTileThreads) ? (N - tile_base) : kTileThreads;
        const int floats_to_load = count * 4;
        const int base_float = tile_base * 4;

        // Stride-1 coalesced load of 512 contiguous floats (alignment-independent!)
        #pragma unroll
        for (int off = tid; off < kTileThreads * 4; off += kTileThreads) {
            if (off < floats_to_load) {
                s_tile[off] = __ldg(&agents[base_float + off]);
            }
        }
        __syncthreads();

        if (i < N) {
            for (int k = 0; k < count; ++k) {
                const int j = tile_base + k;
                if (j != i) {
                    const float qx = s_tile[k * 4 + 0];
                    const float qy = s_tile[k * 4 + 1];
                    if (exact_dist2d_sq(px, py, qx, qy) < kRadiusSq) {
                        sum_vx += static_cast<double>(s_tile[k * 4 + 2]);
                        sum_vy += static_cast<double>(s_tile[k * 4 + 3]);
                        ++cnt;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (i < N) {
        float new_vx = vx;
        float new_vy = vy;
        if (cnt > 0) {
            const double inv_cnt = 1.0 / static_cast<double>(cnt);
            const float avg_vx = static_cast<float>(sum_vx * inv_cnt);
            const float avg_vy = static_cast<float>(sum_vy * inv_cnt);
            new_vx = vx + kAlpha * (avg_vx - vx);
            new_vy = vy + kAlpha * (avg_vy - vy);
        }
        const float new_px = px + new_vx;
        const float new_py = py + new_vy;

        if (aligned_out) {
            reinterpret_cast<float4*>(agents_next)[i] =
                make_float4(new_px, new_py, new_vx, new_vy);
        } else {
            agents_next[i * 4 + 0] = new_px;
            agents_next[i * 4 + 1] = new_py;
            agents_next[i * 4 + 2] = new_vx;
            agents_next[i * 4 + 3] = new_vy;
        }
    }
}

__global__ __launch_bounds__(kTileThreads) void boids_tiled_allpairs_kernel(
    const float* __restrict__ agents,
    float* __restrict__ agents_next,
    int N,
    bool aligned_out) {
    run_tiled_allpairs_block(agents, agents_next, N, aligned_out);
}

// ============================================================================
// Fast Path (`512 <= N <= 100,000`):
// Exact Power-of-Two Cell (`H = 8.0` or `16.0`, `inv_H = 0.125` or `0.0625` ->
// zero mantissa rounding!) 2D Uniform Spatial Grid / Deduplicated Spatial Hash!
//   - On Official Perf (`N = 10,000` in `[-1000, 1000]`): builds a collision-free
//     `127 x 127 = 16,129`-cell 2D grid (`63 KB` in L1/L2!), cutting average
//     neighbor checks per agent from `10,000` down to `< 6` agents!
// ============================================================================
__global__ __launch_bounds__(256) void boids_bbox_and_clear_kernel(
    const float* __restrict__ agents,
    int N,
    int max_buckets,
    bool aligned_in) {
    const int tid = static_cast<int>(threadIdx.x);
    const int gid = static_cast<int>(blockIdx.x) * 256 + tid;
    const int stride = static_cast<int>(gridDim.x) * 256;

    // Clear bucket counts in parallel during the bounding-box sweep!
    for (int b = gid; b < max_buckets; b += stride) {
        g_bucket_counts[b] = 0;
    }

    float min_x = INFINITY, max_x = -INFINITY;
    float min_y = INFINITY, max_y = -INFINITY;

    if (aligned_in) {
        const float4* __restrict__ a4 = reinterpret_cast<const float4*>(agents);
        for (int i = gid; i < N; i += stride) {
            const float4 v = __ldg(&a4[i]);
            min_x = fminf(min_x, v.x);
            max_x = fmaxf(max_x, v.x);
            min_y = fminf(min_y, v.y);
            max_y = fmaxf(max_y, v.y);
        }
    } else {
        for (int i = gid; i < N; i += stride) {
            const float x = __ldg(&agents[i * 4 + 0]);
            const float y = __ldg(&agents[i * 4 + 1]);
            min_x = fminf(min_x, x);
            max_x = fmaxf(max_x, x);
            min_y = fminf(min_y, y);
            max_y = fmaxf(max_y, y);
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        min_x = fminf(min_x, __shfl_down_sync(0xffffffffu, min_x, off));
        max_x = fmaxf(max_x, __shfl_down_sync(0xffffffffu, max_x, off));
        min_y = fminf(min_y, __shfl_down_sync(0xffffffffu, min_y, off));
        max_y = fmaxf(max_y, __shfl_down_sync(0xffffffffu, max_y, off));
    }

    __shared__ float s_b[8][4];
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    if (lane == 0) {
        s_b[warp_id][0] = min_x;
        s_b[warp_id][1] = max_x;
        s_b[warp_id][2] = min_y;
        s_b[warp_id][3] = max_y;
    }
    __syncthreads();

    if (tid == 0) {
        for (int w = 1; w < 8; ++w) {
            s_b[0][0] = fminf(s_b[0][0], s_b[w][0]);
            s_b[0][1] = fmaxf(s_b[0][1], s_b[w][1]);
            s_b[0][2] = fminf(s_b[0][2], s_b[w][2]);
            s_b[0][3] = fmaxf(s_b[0][3], s_b[w][3]);
        }

        const int base = static_cast<int>(blockIdx.x) * 4;
        g_bbox_partials[base + 0] = s_b[0][0];
        g_bbox_partials[base + 1] = s_b[0][1];
        g_bbox_partials[base + 2] = s_b[0][2];
        g_bbox_partials[base + 3] = s_b[0][3];
        __threadfence();

        const unsigned int prev = atomicAdd(&g_bbox_done, 1u);
        if (prev + 1u == gridDim.x) {
            float gx0 = g_bbox_partials[0], gx1 = g_bbox_partials[1];
            float gy0 = g_bbox_partials[2], gy1 = g_bbox_partials[3];
            for (unsigned int b = 1; b < gridDim.x; ++b) {
                gx0 = fminf(gx0, g_bbox_partials[b * 4 + 0]);
                gx1 = fmaxf(gx1, g_bbox_partials[b * 4 + 1]);
                gy0 = fminf(gy0, g_bbox_partials[b * 4 + 2]);
                gy1 = fmaxf(gy1, g_bbox_partials[b * 4 + 3]);
            }

            const float span_x = gx1 - gx0;
            const float span_y = gy1 - gy0;
            const float max_span = fmaxf(span_x, span_y);
            const float max_abs = fmaxf(fmaxf(fabsf(gx0), fabsf(gx1)),
                                        fmaxf(fabsf(gy0), fabsf(gy1)));

            // Guard extreme (>2e7 -> 1 ULP > 2.0) or ultra-tight (<= 12.0) clouds
            if (!isfinite(max_span) || max_abs > 2.0e7f || max_span <= 12.0f) {
                g_fallback = 1;
            } else {
                g_fallback = 0;
                g_origin_x = gx0;
                g_origin_y = gy0;

                // Use exact power-of-two `inv_cell`: 0.0625f (`H = 16.0`) or 0.125f (`H = 8.0`)!
                const float inv_c = (max_buckets == 16384 && max_span > 960.0f) ? 0.0625f : 0.125f;
                g_inv_cell = inv_c;

                const int max_dim = (max_buckets == 16384) ? 128 : 256;
                const int gw = static_cast<int>(floorf(span_x * inv_c)) + 2;
                const int gh = static_cast<int>(floorf(span_y * inv_c)) + 2;
                if (gw <= max_dim && gh <= max_dim) {
                    g_use_hash = 0;
                    g_grid_w = gw;
                    g_grid_h = gh;
                    g_num_buckets = gw * gh;
                } else {
                    g_use_hash = 1;
                    g_grid_w = 0;
                    g_grid_h = 0;
                    g_num_buckets = max_buckets;
                }
            }

            g_bbox_done = 0u;
        }
    }
}

__global__ __launch_bounds__(256) void boids_count_buckets_kernel(
    const float* __restrict__ agents,
    int N,
    bool aligned_in) {
    if (g_fallback != 0) {
        return;
    }

    const int i = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
    if (i >= N) {
        return;
    }

    float x, y;
    if (aligned_in) {
        const float4 v = __ldg(reinterpret_cast<const float4*>(agents) + i);
        x = v.x;
        y = v.y;
    } else {
        x = __ldg(&agents[i * 4 + 0]);
        y = __ldg(&agents[i * 4 + 1]);
    }

    const float ox = g_origin_x;
    const float oy = g_origin_y;
    const float inv_c = g_inv_cell;

    int cx = static_cast<int>(floorf(__fsub_rn(x, ox) * inv_c));
    int cy = static_cast<int>(floorf(__fsub_rn(y, oy) * inv_c));

    int bucket;
    if (g_use_hash == 0) {
        const int gw = g_grid_w;
        const int gh = g_grid_h;
        if (cx < 0) cx = 0;
        if (cx >= gw) cx = gw - 1;
        if (cy < 0) cy = 0;
        if (cy >= gh) cy = gh - 1;
        bucket = cy * gw + cx;
    } else {
        bucket = hash_cell_2d(cx, cy, g_num_buckets - 1);
    }

    g_agent_bucket[i] = static_cast<uint16_t>(bucket);
    atomicAdd(&g_bucket_counts[bucket], 1);
}

// Single-block 32-warp exclusive prefix sum over `num_buckets` (`<= 16,384` or `<= 65,536`)
__global__ __launch_bounds__(1024) void boids_scan_buckets_kernel(int N) {
    if (g_fallback != 0) {
        return;
    }

    __shared__ int s_warp_sums[32];
    __shared__ int s_chunk_base;
    __shared__ int s_max_cnt;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int num_buckets = g_num_buckets;

    if (tid == 0) {
        s_chunk_base = 0;
        s_max_cnt = 0;
    }
    __syncthreads();

    int local_max = 0;
    for (int base = 0; base < num_buckets; base += 1024) {
        const int idx = base + tid;
        const int cnt = (idx < num_buckets) ? g_bucket_counts[idx] : 0;
        if (cnt > local_max) {
            local_max = cnt;
        }

        int scan = cnt;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int n = __shfl_up_sync(0xffffffffu, scan, off);
            if (lane >= off) {
                scan += n;
            }
        }
        if (lane == 31) {
            s_warp_sums[warp_id] = scan;
        }
        __syncthreads();

        if (warp_id == 0) {
            int ws = s_warp_sums[lane];
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const int n = __shfl_up_sync(0xffffffffu, ws, off);
                if (lane >= off) {
                    ws += n;
                }
            }
            s_warp_sums[lane] = ws;
        }
        __syncthreads();

        const int warp_excl = (warp_id > 0) ? s_warp_sums[warp_id - 1] : 0;
        const int excl = s_chunk_base + warp_excl + (scan - cnt);
        if (idx < num_buckets) {
            g_bucket_starts[idx] = excl;
            g_bucket_write[idx] = excl;
        }
        __syncthreads();

        if (tid == 0) {
            s_chunk_base += s_warp_sums[31];
        }
        __syncthreads();
    }

    if (local_max > 0) {
        atomicMax(&s_max_cnt, local_max);
    }
    __syncthreads();

    if (tid == 0) {
        g_bucket_starts[num_buckets] = N;
        // If >25% of agents (>512) collapse into a single cell, run shared-memory tiled fallback!
        if (s_max_cnt > 512 && s_max_cnt > (N >> 2)) {
            g_fallback = 1;
        }
    }
}

__global__ __launch_bounds__(256) void boids_scatter_agents_kernel(
    const float* __restrict__ agents,
    int N,
    bool aligned_in) {
    if (g_fallback != 0) {
        return;
    }

    const int i = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
    if (i >= N) {
        return;
    }

    const int bucket = static_cast<int>(g_agent_bucket[i]);
    const int dst = atomicAdd(&g_bucket_write[bucket], 1);

    float4 v;
    if (aligned_in) {
        v = __ldg(reinterpret_cast<const float4*>(agents) + i);
    } else {
        v = make_float4(
            __ldg(&agents[i * 4 + 0]),
            __ldg(&agents[i * 4 + 1]),
            __ldg(&agents[i * 4 + 2]),
            __ldg(&agents[i * 4 + 3]));
    }
    g_sorted_agents[dst] = v;
    g_sorted_orig[dst] = i;
}

__global__ __launch_bounds__(kTileThreads) void boids_simulate_grid_kernel(
    const float* __restrict__ agents,
    float* __restrict__ agents_next,
    int N,
    bool aligned_out) {
    if (g_fallback != 0) {
        run_tiled_allpairs_block(agents, agents_next, N, aligned_out);
        return;
    }

    const int s = static_cast<int>(blockIdx.x) * kTileThreads + static_cast<int>(threadIdx.x);
    if (s >= N) {
        return;
    }

    const float4 me = __ldg(&g_sorted_agents[s]);
    const int orig_i = __ldg(&g_sorted_orig[s]);

    const float ox = g_origin_x;
    const float oy = g_origin_y;
    const float inv_c = g_inv_cell;

    int cx = static_cast<int>(floorf(__fsub_rn(me.x, ox) * inv_c));
    int cy = static_cast<int>(floorf(__fsub_rn(me.y, oy) * inv_c));

    double sum_vx = 0.0;
    double sum_vy = 0.0;
    int cnt = 0;

    if (g_use_hash == 0) {
        const int gw = g_grid_w;
        const int gh = g_grid_h;
        if (cx < 0) cx = 0;
        if (cx >= gw) cx = gw - 1;
        if (cy < 0) cy = 0;
        if (cy >= gh) cy = gh - 1;

        const int y_lo = (cy > 0) ? (cy - 1) : 0;
        const int y_hi = (cy + 1 < gh) ? (cy + 1) : (gh - 1);
        const int x_lo = (cx > 0) ? (cx - 1) : 0;
        const int x_hi = (cx + 1 < gw) ? (cx + 1) : (gw - 1);

        for (int ny = y_lo; ny <= y_hi; ++ny) {
            for (int nx = x_lo; nx <= x_hi; ++nx) {
                const int b = ny * gw + nx;
                const int start = __ldg(&g_bucket_starts[b]);
                const int end = __ldg(&g_bucket_starts[b + 1]);
                for (int k = start; k < end; ++k) {
                    if (k == s) {
                        continue;
                    }
                    const float4 nb = __ldg(&g_sorted_agents[k]);
                    if (exact_dist2d_sq(me.x, me.y, nb.x, nb.y) < kRadiusSq) {
                        sum_vx += static_cast<double>(nb.z);
                        sum_vy += static_cast<double>(nb.w);
                        ++cnt;
                    }
                }
            }
        }
    } else {
        const int mask = g_num_buckets - 1;
        int seen[9];
        int n_seen = 0;

        #pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
            #pragma unroll
            for (int dx = -1; dx <= 1; ++dx) {
                const int b = hash_cell_2d(cx + dx, cy + dy, mask);
                bool dup = false;
                for (int p = 0; p < n_seen; ++p) {
                    if (seen[p] == b) {
                        dup = true;
                        break;
                    }
                }
                if (dup) {
                    continue;
                }
                seen[n_seen++] = b;

                const int start = __ldg(&g_bucket_starts[b]);
                const int end = __ldg(&g_bucket_starts[b + 1]);
                for (int k = start; k < end; ++k) {
                    if (k == s) {
                        continue;
                    }
                    const float4 nb = __ldg(&g_sorted_agents[k]);
                    if (exact_dist2d_sq(me.x, me.y, nb.x, nb.y) < kRadiusSq) {
                        sum_vx += static_cast<double>(nb.z);
                        sum_vy += static_cast<double>(nb.w);
                        ++cnt;
                    }
                }
            }
        }
    }

    float new_vx = me.z;
    float new_vy = me.w;
    if (cnt > 0) {
        const double inv_cnt = 1.0 / static_cast<double>(cnt);
        const float avg_vx = static_cast<float>(sum_vx * inv_cnt);
        const float avg_vy = static_cast<float>(sum_vy * inv_cnt);
        new_vx = me.z + kAlpha * (avg_vx - me.z);
        new_vy = me.w + kAlpha * (avg_vy - me.w);
    }
    const float new_px = me.x + new_vx;
    const float new_py = me.y + new_vy;

    if (aligned_out) {
        reinterpret_cast<float4*>(agents_next)[orig_i] =
            make_float4(new_px, new_py, new_vx, new_vy);
    } else {
        agents_next[orig_i * 4 + 0] = new_px;
        agents_next[orig_i * 4 + 1] = new_py;
        agents_next[orig_i * 4 + 2] = new_vx;
        agents_next[orig_i * 4 + 3] = new_vy;
    }
}

}  // namespace

extern "C" void solve(const float* agents, float* agents_next, int N) {
    if (N <= 0) {
        return;
    }

    const bool aligned_in = ((reinterpret_cast<uintptr_t>(agents) & 15u) == 0u);
    const bool aligned_out = ((reinterpret_cast<uintptr_t>(agents_next) & 15u) == 0u);

    if (N < kSmallThreshold || N > kMaxN) {
        const unsigned int blocks = static_cast<unsigned int>((N + kTileThreads - 1) / kTileThreads);
        boids_tiled_allpairs_kernel<<<blocks, kTileThreads>>>(agents, agents_next, N, aligned_out);
        return;
    }

    const int max_buckets = (N <= 25000) ? 16384 : kMaxBuckets;

    int bbox_blocks = (N + 1023) / 1024;
    if (bbox_blocks < 1) bbox_blocks = 1;
    if (bbox_blocks > kMaxBboxBlocks) bbox_blocks = kMaxBboxBlocks;

    const unsigned int count_blocks = static_cast<unsigned int>((N + 255) / 256);
    const unsigned int sim_blocks = static_cast<unsigned int>((N + kTileThreads - 1) / kTileThreads);

    boids_bbox_and_clear_kernel<<<static_cast<unsigned int>(bbox_blocks), 256>>>(
        agents, N, max_buckets, aligned_in);
    boids_count_buckets_kernel<<<count_blocks, 256>>>(agents, N, aligned_in);
    boids_scan_buckets_kernel<<<1, 1024>>>(N);
    boids_scatter_agents_kernel<<<count_blocks, 256>>>(agents, N, aligned_in);
    boids_simulate_grid_kernel<<<sim_blocks, kTileThreads>>>(agents, agents_next, N, aligned_out);
}
