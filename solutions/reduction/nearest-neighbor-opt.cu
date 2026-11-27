#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>

namespace {

// Official constraints: 1 <= N <= 100,000; coordinates in [-1000, 1000].
constexpr int kMaxN = 100000;
constexpr int kMaxG = 32;
constexpr int kMaxCells = kMaxG * kMaxG * kMaxG;  // 32,768 cells (128 KiB)
constexpr int kGridThreshold = 2048;
constexpr int kMaxBboxBlocks = 256;

// Module-scope static device workspace (referenced ONLY inside device kernels!)
__device__ float g_sorted_x[kMaxN];
__device__ float g_sorted_y[kMaxN];
__device__ float g_sorted_z[kMaxN];
__device__ int g_sorted_orig[kMaxN];
__device__ int g_sorted_cell[kMaxN];
__device__ int g_point_cell[kMaxN];

__device__ int g_cell_counts[kMaxCells];
__device__ int g_cell_starts[kMaxCells + 1];
__device__ int g_cell_write[kMaxCells];

__device__ float g_split_x[kMaxG + 1];
__device__ float g_split_y[kMaxG + 1];
__device__ float g_split_z[kMaxG + 1];

__device__ float g_bbox_partials[kMaxBboxBlocks * 6];
__device__ unsigned int g_bbox_done;
__device__ int g_degenerate;

// ============================================================================
// Exact IEEE-754 non-FMA round-to-nearest squared distance (`atol = 0` safe!)
// ============================================================================
__device__ __forceinline__ float exact_dist_sq(float x0, float y0, float z0,
                                               float x1, float y1, float z1) {
    const float dx = __fsub_rn(x0, x1);
    const float dy = __fsub_rn(y0, y1);
    const float dz = __fsub_rn(z0, z1);
    const float dx2 = __fmul_rn(dx, dx);
    const float dy2 = __fmul_rn(dy, dy);
    const float dz2 = __fmul_rn(dz, dz);
    return __fadd_rn(__fadd_rn(dx2, dy2), dz2);
}

// ============================================================================
// Shared-Memory Coalesced-to-SoA Tiled Brute-Force Device Helper (`128` threads)
//   - Used directly as a 1-launch fast path when `N < 2,048` (or `N > kMaxN`),
//     AND as a zero-host-sync fallback inside `nn_grid_search_kernel` if the
//     point cloud is pathologically collapsed into a single voxel!
// ============================================================================
constexpr int kSoaThreads = 128;
constexpr int kSoaTile = 128;

__device__ __forceinline__ void run_tiled_soa_block(
    const float* __restrict__ points,
    int* __restrict__ indices,
    int N) {
    __shared__ float s_raw[kSoaTile * 3];  // 1.5 KiB coalesced staging buffer

    const int tid = static_cast<int>(threadIdx.x);
    const int i = static_cast<int>(blockIdx.x) * kSoaThreads + tid;

    if (N == 1) {
        if (i == 0) {
            indices[0] = 0;
        }
        return;
    }

    float xi = 0.0f, yi = 0.0f, zi = 0.0f;
    if (i < N) {
        xi = __ldg(&points[i * 3 + 0]);
        yi = __ldg(&points[i * 3 + 1]);
        zi = __ldg(&points[i * 3 + 2]);
    }

    float best_d2 = INFINITY;
    int best_j = INT_MAX;

    for (int tile_base = 0; tile_base < N; tile_base += kSoaTile) {
        const int count = ((N - tile_base) < kSoaTile) ? (N - tile_base) : kSoaTile;
        const int floats_to_load = count * 3;
        const int global_float_base = tile_base * 3;

        // 100% stride-1 coalesced global load of 384 contiguous floats (`x0,y0,z0,x1,...`)
        #pragma unroll
        for (int off = tid; off < kSoaTile * 3; off += kSoaThreads) {
            if (off < floats_to_load) {
                s_raw[off] = __ldg(&points[global_float_base + off]);
            }
        }
        __syncthreads();

        if (i < N) {
            for (int k = 0; k < count; ++k) {
                const int j = tile_base + k;
                if (j != i) {
                    const float d2 = exact_dist_sq(
                        xi, yi, zi,
                        s_raw[k * 3 + 0], s_raw[k * 3 + 1], s_raw[k * 3 + 2]);
                    if (d2 < best_d2) {
                        best_d2 = d2;
                        best_j = j;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (i < N) {
        indices[i] = best_j;
    }
}

__global__ __launch_bounds__(kSoaThreads) void nn_tiled_soa_kernel(
    const float* __restrict__ points,
    int* __restrict__ indices,
    int N) {
    run_tiled_soa_block(points, indices, N);
}

// ============================================================================
// Fast Path for `N >= 2,048`:
// Exact Provable IEEE-754 Directed-Rounding Shell-Pruned 3D Uniform Voxel Grid!
//   - Pass 1: Compute 3D bounding box + clear `g_cell_counts[0..num_cells-1]` +
//             build monotone split planes `g_split_{x,y,z}[0..G]`.
//   - Pass 2: Assign each point's cell via exact 4/5-step binary search on
//             `s_split_{x,y,z}` (guaranteeing `split[c] <= q < split[c+1]` with
//             ZERO FP rounding fuzziness at cell boundaries!) + `atomicAdd` counts.
//   - Pass 3: Single-block exclusive prefix sum over `num_cells` (`4,096` or `32,768`)
//             + automatic cluster/degeneracy detector (`g_degenerate`).
//   - Pass 4: Scatter points into spatially sorted SoA arrays (`g_sorted_{x,y,z,orig,cell}`).
//   - Pass 5: Warp-coherent sorted-point shell search using IEEE round-down
//             (`__fsub_rd`, `__fmul_rd`, `__fadd_rd`) wall & cell AABB lower bounds
//             with strict `<` pruning so EVERY exact distance & tie (`atol = 0`)
//             is 100% mathematically guaranteed!
// ============================================================================
__global__ __launch_bounds__(256) void nn_bbox_and_clear_kernel(
    const float* __restrict__ points,
    int N,
    int G,
    int num_cells) {
    const int tid = static_cast<int>(threadIdx.x);
    const int gid = static_cast<int>(blockIdx.x) * 256 + tid;
    const int stride = static_cast<int>(gridDim.x) * 256;

    // Clear `g_cell_counts` in parallel during the same launch!
    for (int c = gid; c < num_cells; c += stride) {
        g_cell_counts[c] = 0;
    }

    float min_x = INFINITY, max_x = -INFINITY;
    float min_y = INFINITY, max_y = -INFINITY;
    float min_z = INFINITY, max_z = -INFINITY;

    for (int i = gid; i < N; i += stride) {
        const float x = __ldg(&points[i * 3 + 0]);
        const float y = __ldg(&points[i * 3 + 1]);
        const float z = __ldg(&points[i * 3 + 2]);
        min_x = fminf(min_x, x);
        max_x = fmaxf(max_x, x);
        min_y = fminf(min_y, y);
        max_y = fmaxf(max_y, y);
        min_z = fminf(min_z, z);
        max_z = fmaxf(max_z, z);
    }

    // Warp reduction
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        min_x = fminf(min_x, __shfl_down_sync(0xffffffffu, min_x, off));
        max_x = fmaxf(max_x, __shfl_down_sync(0xffffffffu, max_x, off));
        min_y = fminf(min_y, __shfl_down_sync(0xffffffffu, min_y, off));
        max_y = fmaxf(max_y, __shfl_down_sync(0xffffffffu, max_y, off));
        min_z = fminf(min_z, __shfl_down_sync(0xffffffffu, min_z, off));
        max_z = fmaxf(max_z, __shfl_down_sync(0xffffffffu, max_z, off));
    }

    __shared__ float s_b[8][6];
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    if (lane == 0) {
        s_b[warp_id][0] = min_x;
        s_b[warp_id][1] = max_x;
        s_b[warp_id][2] = min_y;
        s_b[warp_id][3] = max_y;
        s_b[warp_id][4] = min_z;
        s_b[warp_id][5] = max_z;
    }
    __syncthreads();

    if (tid == 0) {
        for (int w = 1; w < 8; ++w) {
            s_b[0][0] = fminf(s_b[0][0], s_b[w][0]);
            s_b[0][1] = fmaxf(s_b[0][1], s_b[w][1]);
            s_b[0][2] = fminf(s_b[0][2], s_b[w][2]);
            s_b[0][3] = fmaxf(s_b[0][3], s_b[w][3]);
            s_b[0][4] = fminf(s_b[0][4], s_b[w][4]);
            s_b[0][5] = fmaxf(s_b[0][5], s_b[w][5]);
        }

        const int base = static_cast<int>(blockIdx.x) * 6;
        for (int k = 0; k < 6; ++k) {
            g_bbox_partials[base + k] = s_b[0][k];
        }
        __threadfence();

        const unsigned int prev = atomicAdd(&g_bbox_done, 1u);
        if (prev + 1u == gridDim.x) {
            float gx0 = g_bbox_partials[0], gx1 = g_bbox_partials[1];
            float gy0 = g_bbox_partials[2], gy1 = g_bbox_partials[3];
            float gz0 = g_bbox_partials[4], gz1 = g_bbox_partials[5];
            for (unsigned int b = 1; b < gridDim.x; ++b) {
                gx0 = fminf(gx0, g_bbox_partials[b * 6 + 0]);
                gx1 = fmaxf(gx1, g_bbox_partials[b * 6 + 1]);
                gy0 = fminf(gy0, g_bbox_partials[b * 6 + 2]);
                gy1 = fmaxf(gy1, g_bbox_partials[b * 6 + 3]);
                gz0 = fminf(gz0, g_bbox_partials[b * 6 + 4]);
                gz1 = fmaxf(gz1, g_bbox_partials[b * 6 + 5]);
            }

            g_split_x[0] = -INFINITY;
            g_split_y[0] = -INFINITY;
            g_split_z[0] = -INFINITY;
            const float inv_g = 1.0f / static_cast<float>(G);
            for (int k = 1; k < G; ++k) {
                const float t = static_cast<float>(k) * inv_g;
                float sx = gx0 + (gx1 - gx0) * t;
                float sy = gy0 + (gy1 - gy0) * t;
                float sz = gz0 + (gz1 - gz0) * t;
                if (k > 1 && sx < g_split_x[k - 1]) sx = g_split_x[k - 1];
                if (k > 1 && sy < g_split_y[k - 1]) sy = g_split_y[k - 1];
                if (k > 1 && sz < g_split_z[k - 1]) sz = g_split_z[k - 1];
                g_split_x[k] = sx;
                g_split_y[k] = sy;
                g_split_z[k] = sz;
            }
            g_split_x[G] = INFINITY;
            g_split_y[G] = INFINITY;
            g_split_z[G] = INFINITY;

            g_bbox_done = 0u;
        }
    }
}

__device__ __forceinline__ int locate_cell_axis(float val, const float* s_split, int G) {
    int c = 0;
    for (int step = (G >> 1); step >= 1; step >>= 1) {
        if (val >= s_split[c + step]) {
            c += step;
        }
    }
    return c;
}

__global__ __launch_bounds__(256) void nn_count_cells_kernel(
    const float* __restrict__ points,
    int N,
    int G,
    int shift) {
    __shared__ float s_sx[kMaxG + 1];
    __shared__ float s_sy[kMaxG + 1];
    __shared__ float s_sz[kMaxG + 1];

    const int tid = static_cast<int>(threadIdx.x);
    if (tid <= G) {
        s_sx[tid] = g_split_x[tid];
        s_sy[tid] = g_split_y[tid];
        s_sz[tid] = g_split_z[tid];
    }
    __syncthreads();

    const int i = static_cast<int>(blockIdx.x) * 256 + tid;
    if (i < N) {
        const float x = __ldg(&points[i * 3 + 0]);
        const float y = __ldg(&points[i * 3 + 1]);
        const float z = __ldg(&points[i * 3 + 2]);
        const int cx = locate_cell_axis(x, s_sx, G);
        const int cy = locate_cell_axis(y, s_sy, G);
        const int cz = locate_cell_axis(z, s_sz, G);
        const int cell = (cz << (2 * shift)) | (cy << shift) | cx;
        g_point_cell[i] = cell;
        atomicAdd(&g_cell_counts[cell], 1);
    }
}

// Single-block parallel scan of `num_cells` (`4,096` or `32,768`) in `< 6 us`!
__global__ __launch_bounds__(512) void nn_scan_cells_kernel(int N, int num_cells) {
    __shared__ int s_warp_sums[16];
    __shared__ int s_chunk_base;
    __shared__ int s_max_cnt;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    if (tid == 0) {
        s_chunk_base = 0;
        s_max_cnt = 0;
    }
    __syncthreads();

    int local_max = 0;
    for (int base = 0; base < num_cells; base += 512) {
        const int idx = base + tid;
        const int cnt = (idx < num_cells) ? g_cell_counts[idx] : 0;
        if (cnt > local_max) local_max = cnt;

        // Inclusive warp scan
        int scan = cnt;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int n = __shfl_up_sync(0xffffffffu, scan, off);
            if (lane >= off) scan += n;
        }
        if (lane == 31) {
            s_warp_sums[warp_id] = scan;
        }
        __syncthreads();

        if (warp_id == 0) {
            int ws = (lane < 16) ? s_warp_sums[lane] : 0;
            #pragma unroll
            for (int off = 1; off < 16; off <<= 1) {
                const int n = __shfl_up_sync(0xffffffffu, ws, off);
                if (lane >= off) ws += n;
            }
            if (lane < 16) {
                s_warp_sums[lane] = ws;
            }
        }
        __syncthreads();

        const int warp_excl = (warp_id > 0) ? s_warp_sums[warp_id - 1] : 0;
        const int excl = s_chunk_base + warp_excl + (scan - cnt);
        if (idx < num_cells) {
            g_cell_starts[idx] = excl;
            g_cell_write[idx] = excl;
        }
        __syncthreads();

        if (tid == 0) {
            s_chunk_base += s_warp_sums[15];
        }
        __syncthreads();
    }

    if (local_max > 0) {
        atomicMax(&s_max_cnt, local_max);
    }
    __syncthreads();

    if (tid == 0) {
        g_cell_starts[num_cells] = N;
        // If >12.5% of all points collapse into a single voxel (>256 pts), trigger tiled SoA fallback!
        g_degenerate = (s_max_cnt > 256 && s_max_cnt > (N >> 3)) ? 1 : 0;
    }
}

__global__ __launch_bounds__(256) void nn_scatter_points_kernel(
    const float* __restrict__ points,
    int N) {
    const int i = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
    if (i < N) {
        if (g_degenerate != 0) {
            return;
        }
        const int cell = g_point_cell[i];
        const int dst = atomicAdd(&g_cell_write[cell], 1);
        g_sorted_x[dst] = __ldg(&points[i * 3 + 0]);
        g_sorted_y[dst] = __ldg(&points[i * 3 + 1]);
        g_sorted_z[dst] = __ldg(&points[i * 3 + 2]);
        g_sorted_orig[dst] = i;
        g_sorted_cell[dst] = cell;
    }
}

// Strict IEEE-754 round-toward-minus-infinity axis distance squared to cell `ci`
__device__ __forceinline__ float axis_cell_d2_rd(float p, int ci, const float* s_split, int G) {
    if (ci > 0 && p < s_split[ci]) {
        const float d = __fsub_rd(s_split[ci], p);
        return (d > 0.0f) ? __fmul_rd(d, d) : 0.0f;
    }
    if (ci + 1 < G && p > s_split[ci + 1]) {
        const float d = __fsub_rd(p, s_split[ci + 1]);
        return (d > 0.0f) ? __fmul_rd(d, d) : 0.0f;
    }
    return 0.0f;
}

__global__ __launch_bounds__(kSoaThreads) void nn_grid_search_kernel(
    const float* __restrict__ points,
    int* __restrict__ indices,
    int N,
    int G,
    int shift) {
    if (g_degenerate != 0) {
        run_tiled_soa_block(points, indices, N);
        return;
    }

    __shared__ float s_sx[kMaxG + 1];
    __shared__ float s_sy[kMaxG + 1];
    __shared__ float s_sz[kMaxG + 1];

    const int tid = static_cast<int>(threadIdx.x);
    if (tid <= G) {
        s_sx[tid] = g_split_x[tid];
        s_sy[tid] = g_split_y[tid];
        s_sz[tid] = g_split_z[tid];
    }
    __syncthreads();

    const int s = static_cast<int>(blockIdx.x) * kSoaThreads + tid;
    if (s >= N) {
        return;
    }

    const float px = g_sorted_x[s];
    const float py = g_sorted_y[s];
    const float pz = g_sorted_z[s];
    const int my_orig = g_sorted_orig[s];
    const int my_cell = g_sorted_cell[s];

    const int mask = G - 1;
    const int cx = my_cell & mask;
    const int cy = (my_cell >> shift) & mask;
    const int cz = my_cell >> (2 * shift);

    float best_d2 = INFINITY;
    int best_idx = INT_MAX;

    for (int r = 0; r < G; ++r) {
        if (r >= 1 && best_d2 < INFINITY) {
            // Compute strict IEEE round-down lower bound across ALL 6 exterior half-spaces!
            float min_wall_d2 = INFINITY;
            if (cx - r >= 0) {
                const float sp = s_sx[cx - r + 1];
                const float d = (px > sp) ? __fsub_rd(px, sp) : 0.0f;
                min_wall_d2 = fminf(min_wall_d2, (d > 0.0f) ? __fmul_rd(d, d) : 0.0f);
            }
            if (cx + r < G) {
                const float sp = s_sx[cx + r];
                const float d = (px < sp) ? __fsub_rd(sp, px) : 0.0f;
                min_wall_d2 = fminf(min_wall_d2, (d > 0.0f) ? __fmul_rd(d, d) : 0.0f);
            }
            if (cy - r >= 0) {
                const float sp = s_sy[cy - r + 1];
                const float d = (py > sp) ? __fsub_rd(py, sp) : 0.0f;
                min_wall_d2 = fminf(min_wall_d2, (d > 0.0f) ? __fmul_rd(d, d) : 0.0f);
            }
            if (cy + r < G) {
                const float sp = s_sy[cy + r];
                const float d = (py < sp) ? __fsub_rd(sp, py) : 0.0f;
                min_wall_d2 = fminf(min_wall_d2, (d > 0.0f) ? __fmul_rd(d, d) : 0.0f);
            }
            if (cz - r >= 0) {
                const float sp = s_sz[cz - r + 1];
                const float d = (pz > sp) ? __fsub_rd(pz, sp) : 0.0f;
                min_wall_d2 = fminf(min_wall_d2, (d > 0.0f) ? __fmul_rd(d, d) : 0.0f);
            }
            if (cz + r < G) {
                const float sp = s_sz[cz + r];
                const float d = (pz < sp) ? __fsub_rd(sp, pz) : 0.0f;
                min_wall_d2 = fminf(min_wall_d2, (d > 0.0f) ? __fmul_rd(d, d) : 0.0f);
            }

            // Strict `<` guarantees not even an equal-distance smaller-index tie can exist outside!
            if (best_d2 < min_wall_d2) {
                break;
            }
        }

        const int iz_lo = (cz - r > 0) ? (cz - r) : 0;
        const int iz_hi = (cz + r < G - 1) ? (cz + r) : (G - 1);
        const int iy_lo = (cy - r > 0) ? (cy - r) : 0;
        const int iy_hi = (cy + r < G - 1) ? (cy + r) : (G - 1);
        const int ix_lo = (cx - r > 0) ? (cx - r) : 0;
        const int ix_hi = (cx + r < G - 1) ? (cx + r) : (G - 1);

        for (int iz = iz_lo; iz <= iz_hi; ++iz) {
            const bool z_border = (iz == cz - r) || (iz == cz + r);
            const float dz2_cell = (r >= 1) ? axis_cell_d2_rd(pz, iz, s_sz, G) : 0.0f;
            if (best_d2 < dz2_cell) {
                continue;
            }

            for (int iy = iy_lo; iy <= iy_hi; ++iy) {
                const bool yz_border = z_border || (iy == cy - r) || (iy == cy + r);
                const float dyz2_cell =
                    (r >= 1) ? __fadd_rd(dz2_cell, axis_cell_d2_rd(py, iy, s_sy, G)) : 0.0f;
                if (best_d2 < dyz2_cell) {
                    continue;
                }

                if (yz_border || r == 0) {
                    for (int ix = ix_lo; ix <= ix_hi; ++ix) {
                        if (r >= 1 && best_d2 < INFINITY) {
                            const float cell_d2 =
                                __fadd_rd(dyz2_cell, axis_cell_d2_rd(px, ix, s_sx, G));
                            if (best_d2 < cell_d2) {
                                continue;
                            }
                        }
                        const int cell = (iz << (2 * shift)) | (iy << shift) | ix;
                        const int start = __ldg(&g_cell_starts[cell]);
                        const int end = __ldg(&g_cell_starts[cell + 1]);
                        for (int k = start; k < end; ++k) {
                            if (k == s) continue;
                            const float d2 = exact_dist_sq(
                                px, py, pz,
                                __ldg(&g_sorted_x[k]),
                                __ldg(&g_sorted_y[k]),
                                __ldg(&g_sorted_z[k]));
                            if (d2 < best_d2) {
                                best_d2 = d2;
                                best_idx = __ldg(&g_sorted_orig[k]);
                            } else if (d2 == best_d2) {
                                const int cand_orig = __ldg(&g_sorted_orig[k]);
                                if (cand_orig < best_idx) {
                                    best_idx = cand_orig;
                                }
                            }
                        }
                    }
                } else {
                    // Only check left/right X-shell faces (`cx - r` and `cx + r`)!
                    for (int side = 0; side < 2; ++side) {
                        const int ix = (side == 0) ? (cx - r) : (cx + r);
                        if (ix < 0 || ix >= G) {
                            continue;
                        }
                        if (best_d2 < INFINITY) {
                            const float cell_d2 =
                                __fadd_rd(dyz2_cell, axis_cell_d2_rd(px, ix, s_sx, G));
                            if (best_d2 < cell_d2) {
                                continue;
                            }
                        }
                        const int cell = (iz << (2 * shift)) | (iy << shift) | ix;
                        const int start = __ldg(&g_cell_starts[cell]);
                        const int end = __ldg(&g_cell_starts[cell + 1]);
                        for (int k = start; k < end; ++k) {
                            if (k == s) continue;
                            const float d2 = exact_dist_sq(
                                px, py, pz,
                                __ldg(&g_sorted_x[k]),
                                __ldg(&g_sorted_y[k]),
                                __ldg(&g_sorted_z[k]));
                            if (d2 < best_d2) {
                                best_d2 = d2;
                                best_idx = __ldg(&g_sorted_orig[k]);
                            } else if (d2 == best_d2) {
                                const int cand_orig = __ldg(&g_sorted_orig[k]);
                                if (cand_orig < best_idx) {
                                    best_idx = cand_orig;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    indices[my_orig] = best_idx;
}

}  // namespace

extern "C" void solve(const float* points, int* indices, int N) {
    if (N <= 0) {
        return;
    }

    if (N < kGridThreshold || N > kMaxN) {
        const unsigned int blocks = static_cast<unsigned int>((N + kSoaThreads - 1) / kSoaThreads);
        nn_tiled_soa_kernel<<<blocks, kSoaThreads>>>(points, indices, N);
        return;
    }

    const int G = (N <= 16384) ? 16 : 32;
    const int shift = (G == 16) ? 4 : 5;
    const int num_cells = G * G * G;

    int bbox_blocks = (N + 1023) / 1024;
    if (bbox_blocks < 1) bbox_blocks = 1;
    if (bbox_blocks > kMaxBboxBlocks) bbox_blocks = kMaxBboxBlocks;

    const unsigned int count_blocks = static_cast<unsigned int>((N + 255) / 256);
    const unsigned int search_blocks = static_cast<unsigned int>((N + kSoaThreads - 1) / kSoaThreads);

    nn_bbox_and_clear_kernel<<<static_cast<unsigned int>(bbox_blocks), 256>>>(
        points, N, G, num_cells);
    nn_count_cells_kernel<<<count_blocks, 256>>>(points, N, G, shift);
    nn_scan_cells_kernel<<<1, 512>>>(N, num_cells);
    nn_scatter_points_kernel<<<count_blocks, 256>>>(points, N);
    nn_grid_search_kernel<<<search_blocks, kSoaThreads>>>(points, indices, N, G, shift);
}
