#include <cuda_runtime.h>

#include <cstdint>

namespace {

// Official constraints: 1 <= rows, cols <= 1,000 (max 1,000,000 cells = 31,250 uint32_t words).
// Total static .bss = 3 * 125 KB = 375 KB (<< 2 MB limit!).
constexpr int kMaxCells = 1000000;
constexpr int kMaxWords = (kMaxCells + 31) / 32;

__device__ uint32_t g_base_vis[kMaxWords];
__device__ uint32_t g_base_cur[kMaxWords];
__device__ uint32_t g_base_nxt[kMaxWords];

// ============================================================================
// Baseline: Unidirectional Level-Synchronous Frontier BFS (1-arg launch_bounds)
// ============================================================================
__global__ __launch_bounds__(256) void bfs_baseline_kernel(
    const int* __restrict__ grid,
    int* __restrict__ result,
    int rows,
    int cols,
    int start_row,
    int start_col,
    int end_row,
    int end_col) {
    __shared__ int s_found;
    __shared__ int s_any_next;

    const int tid = static_cast<int>(threadIdx.x);
    const int total = rows * cols;
    const int num_words = (total + 31) >> 5;
    const int start_idx = start_row * cols + start_col;
    const int end_idx = end_row * cols + end_col;

    if (start_idx == end_idx) {
        if (tid == 0) {
            result[0] = (grid[start_idx] == 0) ? 0 : -1;
        }
        return;
    }

    for (int w = tid; w < num_words; w += 256) {
        g_base_vis[w] = 0u;
        g_base_cur[w] = 0u;
        g_base_nxt[w] = 0u;
    }
    __syncthreads();

    if (tid == 0) {
        if (grid[start_idx] == 0 && grid[end_idx] == 0) {
            const uint32_t bit = 1u << (start_idx & 31);
            g_base_vis[start_idx >> 5] = bit;
            g_base_cur[start_idx >> 5] = bit;
        }
    }
    __syncthreads();

    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = {0, 0, -1, 1};

    for (int dist = 1; dist <= total; ++dist) {
        if (tid == 0) {
            s_found = 0;
            s_any_next = 0;
        }
        __syncthreads();

        int local_any = 0;
        int local_found = 0;

        for (int w = tid; w < num_words; w += 256) {
            uint32_t mask = g_base_cur[w];
            if (mask == 0u) {
                continue;
            }
            g_base_cur[w] = 0u;
            while (mask != 0u) {
                const int b = __ffs(static_cast<int>(mask)) - 1;
                mask &= (mask - 1u);
                const int idx = (w << 5) + b;
                if (idx >= total) {
                    break;
                }
                const int r = idx / cols;
                const int c = idx - r * cols;

                #pragma unroll
                for (int d = 0; d < 4; ++d) {
                    const int nr = r + dr[d];
                    const int nc = c + dc[d];
                    if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                        const int nidx = nr * cols + nc;
                        const uint32_t nbit = 1u << (nidx & 31);
                        const int nw = nidx >> 5;
                        if ((g_base_vis[nw] & nbit) == 0u && grid[nidx] == 0) {
                            const uint32_t old_v = atomicOr(&g_base_vis[nw], nbit);
                            if ((old_v & nbit) == 0u) {
                                if (nidx == end_idx) {
                                    local_found = 1;
                                }
                                atomicOr(&g_base_nxt[nw], nbit);
                                local_any = 1;
                            }
                        }
                    }
                }
            }
        }

        if (local_found) {
            atomicOr(&s_found, 1);
        }
        if (local_any) {
            atomicOr(&s_any_next, 1);
        }
        __syncthreads();

        if (s_found != 0) {
            if (tid == 0) {
                result[0] = dist;
            }
            return;
        }
        if (s_any_next == 0) {
            if (tid == 0) {
                result[0] = -1;
            }
            return;
        }

        for (int w = tid; w < num_words; w += 256) {
            g_base_cur[w] = g_base_nxt[w];
            g_base_nxt[w] = 0u;
        }
        __syncthreads();
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
    bfs_baseline_kernel<<<1, 256>>>(
        grid, result, rows, cols, start_row, start_col, end_row, end_col);
}
