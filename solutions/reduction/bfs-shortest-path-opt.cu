#include <cuda_runtime.h>

#include <cstdint>

namespace {

// Official constraints: 1 <= rows, cols <= 1,000 (`W = (cols + 31) / 32 <= 32` words/row!).
// Notice EVERY row (`<= 1,000` cols) fits in AT MOST 32 `uint32_t` words = 1 single 32-lane warp!
// Using Boolean identity `next_front_s & Vis_T == next_front_s & (~unvis_t)` (since `next_front_s`
// is already masked by `unvis_s <= free_mask`!), we only need `unvis_s`, `unvis_t`, and ping-pong
// `front_s[2]`, `front_t[2]` -> 6 * 128 KB = 768 KB total static .bss (<< 2 MB limit, 100% L2-resident!).
constexpr int kMaxRows = 1000;
constexpr int kMaxWordsPerRow = 32;
constexpr int kMaxPlaneWords = kMaxRows * kMaxWordsPerRow;  // 32,000 words = 128 KB/plane
constexpr int kSmemMaxWords = 1792;                         // 6 * 1792 * 4 B = 42.0 KiB <= 48 KiB smem!

__device__ uint32_t g_unvis_s[kMaxPlaneWords];
__device__ uint32_t g_unvis_t[kMaxPlaneWords];
__device__ uint32_t g_front_s[2][kMaxPlaneWords];
__device__ uint32_t g_front_t[2][kMaxPlaneWords];

__global__ __launch_bounds__(32) void bfs_write_zero_kernel(
    const int* __restrict__ grid,
    int* __restrict__ result,
    int idx) {
    if (threadIdx.x == 0) {
        result[0] = (__ldg(&grid[idx]) == 0) ? 0 : -1;
    }
}

// ============================================================================
// Warp-wide 4-connected horizontal dilation (`(f << 1) | (f >> 1)` across 32-bit
// word boundaries for `cols <= 1,000` in 2 uniform `__shfl_*_sync` instructions!)
// ============================================================================
__device__ __forceinline__ uint32_t dilate_horiz_warp(uint32_t f, int lane, int W) {
    uint32_t from_left = __shfl_up_sync(0xffffffffu, f >> 31, 1);
    if (lane == 0) {
        from_left = 0u;
    }
    uint32_t from_right = __shfl_down_sync(0xffffffffu, f & 1u, 1);
    if (lane + 1 >= W) {
        from_right = 0u;
    }
    return (f << 1) | from_left | (f >> 1) | (from_right << 31);
}

// ============================================================================
// Fast Path 1 (`rows * W <= 1,792`, e.g., all grids up to `238 x 238`):
// Single-Launch All-In-Shared-Memory Bidirectional Bitboard Wavefront BFS!
// ============================================================================
__global__ __launch_bounds__(512) void bfs_smem_bidir_kernel(
    const int* __restrict__ grid,
    int* __restrict__ result,
    int rows,
    int cols,
    int W,
    int start_row,
    int start_col,
    int end_row,
    int end_col) {
    __shared__ uint32_t s_unvis_s[kSmemMaxWords];
    __shared__ uint32_t s_unvis_t[kSmemMaxWords];
    __shared__ uint32_t s_front_s[2][kSmemMaxWords];
    __shared__ uint32_t s_front_t[2][kSmemMaxWords];

    __shared__ int s_w_min[16];
    __shared__ int s_w_max[16];
    __shared__ int s_state[9];  // [s_lo, s_hi, t_lo, t_hi, s_cur, t_cur, dist, status, expand_s]

    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;  // 0..15
    const int lane = tid & 31;     // 0..31

    // 1. Pack `grid` into shared-memory bitboards via `__ballot_sync` (1 warp per row)
    for (int r = warp_id; r < rows; r += 16) {
        uint32_t my_free = 0u;
        for (int w = 0; w < W; ++w) {
            const int c = (w << 5) + lane;
            const bool is_free = (c < cols) && (__ldg(&grid[r * cols + c]) == 0);
            const uint32_t mask = __ballot_sync(0xffffffffu, is_free);
            if (lane == w) {
                my_free = mask;
            }
        }
        if (lane < W) {
            uint32_t us = my_free;
            uint32_t ut = my_free;
            uint32_t fs = 0u;
            uint32_t ft = 0u;
            if (r == start_row && lane == (start_col >> 5)) {
                const uint32_t sbit = 1u << (start_col & 31);
                if ((my_free & sbit) != 0u) {
                    us &= ~sbit;
                    fs = sbit;
                }
            }
            if (r == end_row && lane == (end_col >> 5)) {
                const uint32_t ebit = 1u << (end_col & 31);
                if ((my_free & ebit) != 0u) {
                    ut &= ~ebit;
                    ft = ebit;
                }
            }
            const int off = r * W + lane;
            s_unvis_s[off] = us;
            s_unvis_t[off] = ut;
            s_front_s[0][off] = fs;
            s_front_t[0][off] = ft;
        }
    }

    if (tid == 0) {
        s_state[0] = start_row;  // s_lo
        s_state[1] = start_row;  // s_hi
        s_state[2] = end_row;    // t_lo
        s_state[3] = end_row;    // t_hi
        s_state[4] = 0;          // s_cur
        s_state[5] = 0;          // t_cur
        s_state[6] = 0;          // dist
        s_state[7] = 0;          // status: 0 = running, 1 = found, -1 = unreachable
        s_state[8] = 1;          // expand_s: expand smaller active-row span next!
    }
    __syncthreads();

    const int max_steps = rows * cols;
    for (int step = 0; step < max_steps; ++step) {
        const bool expand_s = (s_state[8] != 0);
        const int cur_lo = expand_s ? s_state[0] : s_state[2];
        const int cur_hi = expand_s ? s_state[1] : s_state[3];
        const int cur_buf = expand_s ? s_state[4] : s_state[5];
        const int nxt_buf = cur_buf ^ 1;

        const int exp_lo = (cur_lo > 0) ? (cur_lo - 1) : 0;
        const int exp_hi = (cur_hi + 1 < rows) ? (cur_hi + 1) : (rows - 1);

        const uint32_t* f_cur = expand_s ? s_front_s[cur_buf] : s_front_t[cur_buf];
        uint32_t* f_nxt = expand_s ? s_front_s[nxt_buf] : s_front_t[nxt_buf];
        uint32_t* u_own = expand_s ? s_unvis_s : s_unvis_t;
        const uint32_t* u_opp = expand_s ? s_unvis_t : s_unvis_s;

        int local_min = rows;
        int local_max = -1;
        int local_hit = 0;

        for (int r = exp_lo + warp_id; r <= exp_hi; r += 16) {
            const uint32_t f_mid =
                (lane < W && r >= cur_lo && r <= cur_hi) ? f_cur[r * W + lane] : 0u;
            const uint32_t f_up =
                (lane < W && (r - 1) >= cur_lo && (r - 1) <= cur_hi) ? f_cur[(r - 1) * W + lane] : 0u;
            const uint32_t f_dn =
                (lane < W && (r + 1) >= cur_lo && (r + 1) <= cur_hi) ? f_cur[(r + 1) * W + lane] : 0u;

            const uint32_t nbr = dilate_horiz_warp(f_mid, lane, W) | f_up | f_dn;
            const uint32_t uo = (lane < W) ? u_own[r * W + lane] : 0u;
            const uint32_t next_f = nbr & uo;

            if (lane < W) {
                f_nxt[r * W + lane] = next_f;
                if (next_f != 0u) {
                    u_own[r * W + lane] = uo & (~next_f);
                }
            }

            if (__ballot_sync(0xffffffffu, next_f != 0u) != 0u) {
                if (r < local_min) local_min = r;
                if (r > local_max) local_max = r;
                const uint32_t u_other = (lane < W) ? u_opp[r * W + lane] : ~0u;
                if (__ballot_sync(0xffffffffu, (next_f & (~u_other)) != 0u) != 0u) {
                    local_hit = 1;
                }
            }
        }

        if (lane == 0) {
            s_w_min[warp_id] = local_min;
            s_w_max[warp_id] = local_max;
        }
        if (local_hit && lane == 0) {
            s_state[7] = 1;
        }
        __syncthreads();

        if (warp_id == 0) {
            int mn = (lane < 16) ? s_w_min[lane] : rows;
            int mx = (lane < 16) ? s_w_max[lane] : -1;
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) {
                const int mn_o = __shfl_down_sync(0xffffffffu, mn, off);
                const int mx_o = __shfl_down_sync(0xffffffffu, mx, off);
                if (mn_o < mn) mn = mn_o;
                if (mx_o > mx) mx = mx_o;
            }
            if (lane == 0) {
                const int new_dist = s_state[6] + 1;
                s_state[6] = new_dist;
                if (s_state[7] == 1) {
                    result[0] = new_dist;
                } else if (mn > mx) {
                    s_state[7] = -1;
                    result[0] = -1;
                } else {
                    if (expand_s) {
                        s_state[0] = mn;
                        s_state[1] = mx;
                        s_state[4] = nxt_buf;
                    } else {
                        s_state[2] = mn;
                        s_state[3] = mx;
                        s_state[5] = nxt_buf;
                    }
                    // Expand whichever frontier currently spans fewer active rows!
                    const int span_s = s_state[1] - s_state[0];
                    const int span_t = s_state[3] - s_state[2];
                    s_state[8] = (span_s <= span_t) ? 1 : 0;
                }
            }
        }
        __syncthreads();

        if (s_state[7] != 0) {
            return;
        }
    }

    if (tid == 0) {
        result[0] = -1;
    }
}

// ============================================================================
// Fast Path 2 (`rows * W > 1,792`, including Official Perf `500 x 500` & `1000 x 1000`):
//   Launch 1: All-SM Coalesced `__ballot_sync` Bitboard Pack (`~4 us` at 300+ GB/s!)
//   Launch 2: 32-Warp (`<<<1, 1024>>>`) L1/L2-Resident Dynamic Active-Band
//             Smaller-Frontier-First Bidirectional Bitboard Wavefront BFS!
// ============================================================================
__global__ __launch_bounds__(256) void bfs_pack_bitboard_kernel(
    const int* __restrict__ grid,
    int rows,
    int cols,
    int W,
    int start_row,
    int start_col,
    int end_row,
    int end_col) {
    const int warp_global = (static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x)) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (warp_global >= rows) {
        return;
    }

    const int r = warp_global;
    const int row_base = r * cols;

    uint32_t my_free = 0u;
    for (int w = 0; w < W; ++w) {
        const int c = (w << 5) + lane;
        const bool is_free = (c < cols) && (__ldg(&grid[row_base + c]) == 0);
        const uint32_t mask = __ballot_sync(0xffffffffu, is_free);
        if (lane == w) {
            my_free = mask;
        }
    }

    if (lane < W) {
        uint32_t us = my_free;
        uint32_t ut = my_free;
        uint32_t fs = 0u;
        uint32_t ft = 0u;

        if (r == start_row && lane == (start_col >> 5)) {
            const uint32_t sbit = 1u << (start_col & 31);
            if ((my_free & sbit) != 0u) {
                us &= ~sbit;
                fs = sbit;
            }
        }
        if (r == end_row && lane == (end_col >> 5)) {
            const uint32_t ebit = 1u << (end_col & 31);
            if ((my_free & ebit) != 0u) {
                ut &= ~ebit;
                ft = ebit;
            }
        }

        const int off = r * W + lane;
        g_unvis_s[off] = us;
        g_unvis_t[off] = ut;
        g_front_s[0][off] = fs;
        g_front_t[0][off] = ft;
    }
}

__global__ __launch_bounds__(1024) void bfs_l2_bidir_kernel(
    int* __restrict__ result,
    int rows,
    int cols,
    int W,
    int start_row,
    int end_row) {
    __shared__ int s_w_min[32];
    __shared__ int s_w_max[32];
    __shared__ int s_state[9];  // [s_lo, s_hi, t_lo, t_hi, s_cur, t_cur, dist, status, expand_s]

    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;  // 0..31
    const int lane = tid & 31;     // 0..31

    if (tid == 0) {
        s_state[0] = start_row;
        s_state[1] = start_row;
        s_state[2] = end_row;
        s_state[3] = end_row;
        s_state[4] = 0;
        s_state[5] = 0;
        s_state[6] = 0;
        s_state[7] = 0;
        s_state[8] = 1;
    }
    __syncthreads();

    const int max_steps = rows * cols;
    for (int step = 0; step < max_steps; ++step) {
        const bool expand_s = (s_state[8] != 0);
        const int cur_lo = expand_s ? s_state[0] : s_state[2];
        const int cur_hi = expand_s ? s_state[1] : s_state[3];
        const int cur_buf = expand_s ? s_state[4] : s_state[5];
        const int nxt_buf = cur_buf ^ 1;

        const int exp_lo = (cur_lo > 0) ? (cur_lo - 1) : 0;
        const int exp_hi = (cur_hi + 1 < rows) ? (cur_hi + 1) : (rows - 1);

        const uint32_t* f_cur = expand_s ? g_front_s[cur_buf] : g_front_t[cur_buf];
        uint32_t* f_nxt = expand_s ? g_front_s[nxt_buf] : g_front_t[nxt_buf];
        uint32_t* u_own = expand_s ? g_unvis_s : g_unvis_t;
        const uint32_t* u_opp = expand_s ? g_unvis_t : g_unvis_s;

        int local_min = rows;
        int local_max = -1;
        int local_hit = 0;

        for (int r = exp_lo + warp_id; r <= exp_hi; r += 32) {
            const uint32_t f_mid =
                (lane < W && r >= cur_lo && r <= cur_hi) ? f_cur[r * W + lane] : 0u;
            const uint32_t f_up =
                (lane < W && (r - 1) >= cur_lo && (r - 1) <= cur_hi) ? f_cur[(r - 1) * W + lane] : 0u;
            const uint32_t f_dn =
                (lane < W && (r + 1) >= cur_lo && (r + 1) <= cur_hi) ? f_cur[(r + 1) * W + lane] : 0u;

            const uint32_t nbr = dilate_horiz_warp(f_mid, lane, W) | f_up | f_dn;
            const uint32_t uo = (lane < W) ? u_own[r * W + lane] : 0u;
            const uint32_t next_f = nbr & uo;

            if (lane < W) {
                f_nxt[r * W + lane] = next_f;
                if (next_f != 0u) {
                    u_own[r * W + lane] = uo & (~next_f);
                }
            }

            if (__ballot_sync(0xffffffffu, next_f != 0u) != 0u) {
                if (r < local_min) local_min = r;
                if (r > local_max) local_max = r;
                const uint32_t u_other = (lane < W) ? u_opp[r * W + lane] : ~0u;
                if (__ballot_sync(0xffffffffu, (next_f & (~u_other)) != 0u) != 0u) {
                    local_hit = 1;
                }
            }
        }

        if (lane == 0) {
            s_w_min[warp_id] = local_min;
            s_w_max[warp_id] = local_max;
        }
        if (local_hit && lane == 0) {
            s_state[7] = 1;
        }
        __syncthreads();

        if (warp_id == 0) {
            int mn = s_w_min[lane];
            int mx = s_w_max[lane];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const int mn_o = __shfl_down_sync(0xffffffffu, mn, off);
                const int mx_o = __shfl_down_sync(0xffffffffu, mx, off);
                if (mn_o < mn) mn = mn_o;
                if (mx_o > mx) mx = mx_o;
            }
            if (lane == 0) {
                const int new_dist = s_state[6] + 1;
                s_state[6] = new_dist;
                if (s_state[7] == 1) {
                    result[0] = new_dist;
                } else if (mn > mx) {
                    s_state[7] = -1;
                    result[0] = -1;
                } else {
                    if (expand_s) {
                        s_state[0] = mn;
                        s_state[1] = mx;
                        s_state[4] = nxt_buf;
                    } else {
                        s_state[2] = mn;
                        s_state[3] = mx;
                        s_state[5] = nxt_buf;
                    }
                    const int span_s = s_state[1] - s_state[0];
                    const int span_t = s_state[3] - s_state[2];
                    s_state[8] = (span_s <= span_t) ? 1 : 0;
                }
            }
        }
        __syncthreads();

        if (s_state[7] != 0) {
            return;
        }
    }

    if (tid == 0) {
        result[0] = -1;
    }
}

}  // namespace

extern "C" void solve(const int* grid, int* result, int rows, int cols,
                      int start_row, int start_col, int end_row, int end_col) {
    if (rows <= 0 || cols <= 0) {
        return;
    }

    if (start_row == end_row && start_col == end_col) {
        bfs_write_zero_kernel<<<1, 32>>>(grid, result, start_row * cols + start_col);
        return;
    }

    const int W = (cols + 31) >> 5;
    if (rows <= kMaxRows && W <= kMaxWordsPerRow && rows * W <= kSmemMaxWords) {
        bfs_smem_bidir_kernel<<<1, 512>>>(
            grid, result, rows, cols, W, start_row, start_col, end_row, end_col);
        return;
    }

    const unsigned int pack_blocks = static_cast<unsigned int>((rows + 7) / 8);
    bfs_pack_bitboard_kernel<<<pack_blocks, 256>>>(
        grid, rows, cols, W, start_row, start_col, end_row, end_col);
    bfs_l2_bidir_kernel<<<1, 1024>>>(result, rows, cols, W, start_row, end_row);
}
