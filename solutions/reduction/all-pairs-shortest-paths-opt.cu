#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kTile = 64;
constexpr int kBlockDim = 16;
constexpr int kThreads = kBlockDim * kBlockDim;  // 256 threads, 4x4 subtile per thread

__device__ __forceinline__ float pos_inf() {
    return __int_as_float(0x7f800000);
}

__host__ __device__ __forceinline__ bool is_aligned_16b(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 15u) == 0u;
}

// ============================================================================
// Fast Single-Launch Kernel for 1 <= N <= 64
// Holds the entire N x N matrix in 256 threads' 4x4 registers + 1 KiB
// double-buffered pivot row/col broadcast buffers (zero global intermediate I/O!)
// ============================================================================
__global__ __launch_bounds__(256) void fw_single_tile_le64_kernel(
    const float* __restrict__ dist,
    float* __restrict__ output,
    int n) {
    __shared__ float s_row[2][kTile];
    __shared__ float s_col[2][kTile];

    const int tx = threadIdx.x & 15;
    const int ty = threadIdx.x >> 4;
    const int r0 = ty * 4;
    const int c0 = tx * 4;
    const float inf = pos_inf();

    float reg[4][4];
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int gr = r0 + m;
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int gc = c0 + q;
            reg[m][q] = (gr < n && gc < n) ? __ldg(&dist[gr * n + gc]) : inf;
        }
    }

    if (ty == 0) {
        s_row[0][c0 + 0] = reg[0][0];
        s_row[0][c0 + 1] = reg[0][1];
        s_row[0][c0 + 2] = reg[0][2];
        s_row[0][c0 + 3] = reg[0][3];
    }
    if (tx == 0) {
        s_col[0][r0 + 0] = reg[0][0];
        s_col[0][r0 + 1] = reg[1][0];
        s_col[0][r0 + 2] = reg[2][0];
        s_col[0][r0 + 3] = reg[3][0];
    }
    __syncthreads();

    for (int k = 0; k < n; ++k) {
        const int cur = k & 1;
        const float a0 = s_col[cur][r0 + 0];
        const float a1 = s_col[cur][r0 + 1];
        const float a2 = s_col[cur][r0 + 2];
        const float a3 = s_col[cur][r0 + 3];

        const float b0 = s_row[cur][c0 + 0];
        const float b1 = s_row[cur][c0 + 1];
        const float b2 = s_row[cur][c0 + 2];
        const float b3 = s_row[cur][c0 + 3];

        reg[0][0] = fminf(reg[0][0], a0 + b0);
        reg[0][1] = fminf(reg[0][1], a0 + b1);
        reg[0][2] = fminf(reg[0][2], a0 + b2);
        reg[0][3] = fminf(reg[0][3], a0 + b3);

        reg[1][0] = fminf(reg[1][0], a1 + b0);
        reg[1][1] = fminf(reg[1][1], a1 + b1);
        reg[1][2] = fminf(reg[1][2], a1 + b2);
        reg[1][3] = fminf(reg[1][3], a1 + b3);

        reg[2][0] = fminf(reg[2][0], a2 + b0);
        reg[2][1] = fminf(reg[2][1], a2 + b1);
        reg[2][2] = fminf(reg[2][2], a2 + b2);
        reg[2][3] = fminf(reg[2][3], a2 + b3);

        reg[3][0] = fminf(reg[3][0], a3 + b0);
        reg[3][1] = fminf(reg[3][1], a3 + b1);
        reg[3][2] = fminf(reg[3][2], a3 + b2);
        reg[3][3] = fminf(reg[3][3], a3 + b3);

        const int kn = k + 1;
        if (kn < n) {
            const int nxt = cur ^ 1;
            const int target_ty = kn >> 2;
            const int target_m = kn & 3;
            if (ty == target_ty) {
                s_row[nxt][c0 + 0] = reg[target_m][0];
                s_row[nxt][c0 + 1] = reg[target_m][1];
                s_row[nxt][c0 + 2] = reg[target_m][2];
                s_row[nxt][c0 + 3] = reg[target_m][3];
            }
            if (tx == target_ty) {
                s_col[nxt][r0 + 0] = reg[0][target_m];
                s_col[nxt][r0 + 1] = reg[1][target_m];
                s_col[nxt][r0 + 2] = reg[2][target_m];
                s_col[nxt][r0 + 3] = reg[3][target_m];
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int gr = r0 + m;
        if (gr < n) {
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int gc = c0 + q;
                if (gc < n) {
                    output[gr * n + gc] = reg[m][q];
                }
            }
        }
    }
}

// ============================================================================
// Blocked FW Phase 1: Diagonal Pivot Tile D[r, r] (64 x 64)
// ============================================================================
__global__ __launch_bounds__(256) void fw_phase1_diag_kernel(float* __restrict__ d,
                                                             int n,
                                                             int r) {
    __shared__ float s_row[2][kTile];
    __shared__ float s_col[2][kTile];

    const int tx = threadIdx.x & 15;
    const int ty = threadIdx.x >> 4;
    const int r0 = ty * 4;
    const int c0 = tx * 4;
    const int base = r * kTile;
    const int rem = n - base;
    const int valid_k = (rem < kTile) ? rem : kTile;
    const float inf = pos_inf();

    float reg[4][4];
    const bool vec_ok = ((n & 3) == 0) && (base + kTile <= n) && is_aligned_16b(d);

    if (vec_ok) {
        const float4* d4 = reinterpret_cast<const float4*>(d);
        const size_t stride4 = static_cast<size_t>(n >> 2);
        const size_t col4 = static_cast<size_t>(base >> 2) + tx;
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const float4 v = __ldg(&d4[static_cast<size_t>(base + r0 + m) * stride4 + col4]);
            reg[m][0] = v.x;
            reg[m][1] = v.y;
            reg[m][2] = v.z;
            reg[m][3] = v.w;
        }
    } else {
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int gr = base + r0 + m;
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int gc = base + c0 + q;
                reg[m][q] =
                    (gr < n && gc < n) ? __ldg(&d[static_cast<size_t>(gr) * n + gc]) : inf;
            }
        }
    }

    if (ty == 0) {
        s_row[0][c0 + 0] = reg[0][0];
        s_row[0][c0 + 1] = reg[0][1];
        s_row[0][c0 + 2] = reg[0][2];
        s_row[0][c0 + 3] = reg[0][3];
    }
    if (tx == 0) {
        s_col[0][r0 + 0] = reg[0][0];
        s_col[0][r0 + 1] = reg[1][0];
        s_col[0][r0 + 2] = reg[2][0];
        s_col[0][r0 + 3] = reg[3][0];
    }
    __syncthreads();

    for (int k = 0; k < valid_k; ++k) {
        const int cur = k & 1;
        const float a0 = s_col[cur][r0 + 0];
        const float a1 = s_col[cur][r0 + 1];
        const float a2 = s_col[cur][r0 + 2];
        const float a3 = s_col[cur][r0 + 3];

        const float b0 = s_row[cur][c0 + 0];
        const float b1 = s_row[cur][c0 + 1];
        const float b2 = s_row[cur][c0 + 2];
        const float b3 = s_row[cur][c0 + 3];

        reg[0][0] = fminf(reg[0][0], a0 + b0);
        reg[0][1] = fminf(reg[0][1], a0 + b1);
        reg[0][2] = fminf(reg[0][2], a0 + b2);
        reg[0][3] = fminf(reg[0][3], a0 + b3);

        reg[1][0] = fminf(reg[1][0], a1 + b0);
        reg[1][1] = fminf(reg[1][1], a1 + b1);
        reg[1][2] = fminf(reg[1][2], a1 + b2);
        reg[1][3] = fminf(reg[1][3], a1 + b3);

        reg[2][0] = fminf(reg[2][0], a2 + b0);
        reg[2][1] = fminf(reg[2][1], a2 + b1);
        reg[2][2] = fminf(reg[2][2], a2 + b2);
        reg[2][3] = fminf(reg[2][3], a2 + b3);

        reg[3][0] = fminf(reg[3][0], a3 + b0);
        reg[3][1] = fminf(reg[3][1], a3 + b1);
        reg[3][2] = fminf(reg[3][2], a3 + b2);
        reg[3][3] = fminf(reg[3][3], a3 + b3);

        const int kn = k + 1;
        if (kn < valid_k) {
            const int nxt = cur ^ 1;
            const int target_ty = kn >> 2;
            const int target_m = kn & 3;
            if (ty == target_ty) {
                s_row[nxt][c0 + 0] = reg[target_m][0];
                s_row[nxt][c0 + 1] = reg[target_m][1];
                s_row[nxt][c0 + 2] = reg[target_m][2];
                s_row[nxt][c0 + 3] = reg[target_m][3];
            }
            if (tx == target_ty) {
                s_col[nxt][r0 + 0] = reg[0][target_m];
                s_col[nxt][r0 + 1] = reg[1][target_m];
                s_col[nxt][r0 + 2] = reg[2][target_m];
                s_col[nxt][r0 + 3] = reg[3][target_m];
            }
            __syncthreads();
        }
    }

    if (vec_ok) {
        float4* d4 = reinterpret_cast<float4*>(d);
        const size_t stride4 = static_cast<size_t>(n >> 2);
        const size_t col4 = static_cast<size_t>(base >> 2) + tx;
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            d4[static_cast<size_t>(base + r0 + m) * stride4 + col4] =
                make_float4(reg[m][0], reg[m][1], reg[m][2], reg[m][3]);
        }
    } else {
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int gr = base + r0 + m;
            if (gr < n) {
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int gc = base + c0 + q;
                    if (gc < n) {
                        d[static_cast<size_t>(gr) * n + gc] = reg[m][q];
                    }
                }
            }
        }
    }
}

// ============================================================================
// Blocked FW Phase 2: Fused Pivot Row & Pivot Col Panels
// - blockIdx.y == 0 -> Row panel D[r, t_other] = min(D[r, t_other], D[r, r] (+) D[r, t_other])
// - blockIdx.y == 1 -> Col panel D[t_other, r] = min(D[t_other, r], D[t_other, r] (+) D[r, r])
// Uses transposed `s_AT[64][68]` + `s_B[64][64]` (33.0 KiB smem) for LDS.128!
// ============================================================================
__global__ __launch_bounds__(256) void fw_phase2_panels_kernel(float* __restrict__ d,
                                                               int n,
                                                               int r) {
    __shared__ __align__(16) float s_AT[kTile][kTile + 4];
    __shared__ __align__(16) float s_B[kTile][kTile];

    const int tid = threadIdx.x;
    const int tx = tid & 15;
    const int ty = tid >> 4;
    const int r0 = ty * 4;
    const int c0 = tx * 4;

    const int bx = static_cast<int>(blockIdx.x);
    const int t_other = (bx < r) ? bx : (bx + 1);
    const bool is_row_panel = (blockIdx.y == 0u);

    const int tile_i = is_row_panel ? r : t_other;
    const int tile_j = is_row_panel ? t_other : r;

    const int base_i = tile_i * kTile;
    const int base_r = r * kTile;
    const int base_j = tile_j * kTile;

    const int rem_k = n - base_r;
    const int valid_k = (rem_k < kTile) ? rem_k : kTile;
    const float inf = pos_inf();

    // Load A = D[tile_i, r] transposed into s_AT[k][row], and B = D[r, tile_j] into s_B[k][col]
    const bool vec_ab = ((n & 3) == 0) && is_aligned_16b(d) && (base_i + kTile <= n) &&
                        (base_r + kTile <= n) && (base_j + kTile <= n);

    if (vec_ab) {
        const float4* d4 = reinterpret_cast<const float4*>(d);
        const size_t stride4 = static_cast<size_t>(n >> 2);
        const size_t a_col4 = static_cast<size_t>(base_r >> 2) + tx;
        const size_t b_col4 = static_cast<size_t>(base_j >> 2) + tx;
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int sr = r0 + m;
            const float4 av = __ldg(&d4[static_cast<size_t>(base_i + sr) * stride4 + a_col4]);
            s_AT[c0 + 0][sr] = av.x;
            s_AT[c0 + 1][sr] = av.y;
            s_AT[c0 + 2][sr] = av.z;
            s_AT[c0 + 3][sr] = av.w;

            const float4 bv = __ldg(&d4[static_cast<size_t>(base_r + sr) * stride4 + b_col4]);
            reinterpret_cast<float4*>(s_B[sr])[tx] = bv;
        }
    } else {
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int sr = r0 + m;
            const int gi_a = base_i + sr;
            const int gi_b = base_r + sr;
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int sc = c0 + q;
                const int gj_a = base_r + sc;
                const int gj_b = base_j + sc;
                s_AT[sc][sr] =
                    (gi_a < n && gj_a < n) ? __ldg(&d[static_cast<size_t>(gi_a) * n + gj_a]) : inf;
                s_B[sr][sc] =
                    (gi_b < n && gj_b < n) ? __ldg(&d[static_cast<size_t>(gi_b) * n + gj_b]) : inf;
            }
        }
    }
    __syncthreads();

    // C_old is already in s_B (if row panel) or s_AT (if col panel)!
    float c00, c01, c02, c03;
    float c10, c11, c12, c13;
    float c20, c21, c22, c23;
    float c30, c31, c32, c33;

    if (is_row_panel) {
        const float4 v0 = reinterpret_cast<const float4*>(s_B[r0 + 0])[tx];
        const float4 v1 = reinterpret_cast<const float4*>(s_B[r0 + 1])[tx];
        const float4 v2 = reinterpret_cast<const float4*>(s_B[r0 + 2])[tx];
        const float4 v3 = reinterpret_cast<const float4*>(s_B[r0 + 3])[tx];
        c00 = v0.x; c01 = v0.y; c02 = v0.z; c03 = v0.w;
        c10 = v1.x; c11 = v1.y; c12 = v1.z; c13 = v1.w;
        c20 = v2.x; c21 = v2.y; c22 = v2.z; c23 = v2.w;
        c30 = v3.x; c31 = v3.y; c32 = v3.z; c33 = v3.w;
    } else {
        const float4 t0 = reinterpret_cast<const float4*>(s_AT[c0 + 0])[ty];
        const float4 t1 = reinterpret_cast<const float4*>(s_AT[c0 + 1])[ty];
        const float4 t2 = reinterpret_cast<const float4*>(s_AT[c0 + 2])[ty];
        const float4 t3 = reinterpret_cast<const float4*>(s_AT[c0 + 3])[ty];
        c00 = t0.x; c01 = t1.x; c02 = t2.x; c03 = t3.x;
        c10 = t0.y; c11 = t1.y; c12 = t2.y; c13 = t3.y;
        c20 = t0.z; c21 = t1.z; c22 = t2.z; c23 = t3.z;
        c30 = t0.w; c31 = t1.w; c32 = t2.w; c33 = t3.w;
    }

    if (valid_k == kTile) {
        #pragma unroll 16
        for (int k = 0; k < kTile; ++k) {
            const float4 a = reinterpret_cast<const float4*>(s_AT[k])[ty];
            const float4 b = reinterpret_cast<const float4*>(s_B[k])[tx];

            c00 = fminf(c00, a.x + b.x);
            c01 = fminf(c01, a.x + b.y);
            c02 = fminf(c02, a.x + b.z);
            c03 = fminf(c03, a.x + b.w);

            c10 = fminf(c10, a.y + b.x);
            c11 = fminf(c11, a.y + b.y);
            c12 = fminf(c12, a.y + b.z);
            c13 = fminf(c13, a.y + b.w);

            c20 = fminf(c20, a.z + b.x);
            c21 = fminf(c21, a.z + b.y);
            c22 = fminf(c22, a.z + b.z);
            c23 = fminf(c23, a.z + b.w);

            c30 = fminf(c30, a.w + b.x);
            c31 = fminf(c31, a.w + b.y);
            c32 = fminf(c32, a.w + b.z);
            c33 = fminf(c33, a.w + b.w);
        }
    } else {
        for (int k = 0; k < valid_k; ++k) {
            const float4 a = reinterpret_cast<const float4*>(s_AT[k])[ty];
            const float4 b = reinterpret_cast<const float4*>(s_B[k])[tx];

            c00 = fminf(c00, a.x + b.x);
            c01 = fminf(c01, a.x + b.y);
            c02 = fminf(c02, a.x + b.z);
            c03 = fminf(c03, a.x + b.w);

            c10 = fminf(c10, a.y + b.x);
            c11 = fminf(c11, a.y + b.y);
            c12 = fminf(c12, a.y + b.z);
            c13 = fminf(c13, a.y + b.w);

            c20 = fminf(c20, a.z + b.x);
            c21 = fminf(c21, a.z + b.y);
            c22 = fminf(c22, a.z + b.z);
            c23 = fminf(c23, a.z + b.w);

            c30 = fminf(c30, a.w + b.x);
            c31 = fminf(c31, a.w + b.y);
            c32 = fminf(c32, a.w + b.z);
            c33 = fminf(c33, a.w + b.w);
        }
    }

    if (vec_ab) {
        float4* d4 = reinterpret_cast<float4*>(d);
        const size_t stride4 = static_cast<size_t>(n >> 2);
        const size_t c_col4 = static_cast<size_t>(base_j >> 2) + tx;
        d4[static_cast<size_t>(base_i + r0 + 0) * stride4 + c_col4] =
            make_float4(c00, c01, c02, c03);
        d4[static_cast<size_t>(base_i + r0 + 1) * stride4 + c_col4] =
            make_float4(c10, c11, c12, c13);
        d4[static_cast<size_t>(base_i + r0 + 2) * stride4 + c_col4] =
            make_float4(c20, c21, c22, c23);
        d4[static_cast<size_t>(base_i + r0 + 3) * stride4 + c_col4] =
            make_float4(c30, c31, c32, c33);
    } else {
        const float rows[4][4] = {
            {c00, c01, c02, c03},
            {c10, c11, c12, c13},
            {c20, c21, c22, c23},
            {c30, c31, c32, c33},
        };
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int gr = base_i + r0 + m;
            if (gr < n) {
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int gc = base_j + c0 + q;
                    if (gc < n) {
                        d[static_cast<size_t>(gr) * n + gc] = rows[m][q];
                    }
                }
            }
        }
    }
}

// ============================================================================
// Blocked FW Phase 3: Remaining (num_tiles - 1) x (num_tiles - 1) Tiles (94% of FLOPs!)
// C[i, j] = min(C[i, j], A[i, r] (+) B[r, j])
// Pure barrier-free 4x4 register-tiled LDS.128 min-plus GEMM + dirty-write skip!
// ============================================================================
__global__ __launch_bounds__(256) void fw_phase3_remain_kernel(float* __restrict__ d,
                                                               int n,
                                                               int r) {
    __shared__ __align__(16) float s_AT[kTile][kTile + 4];
    __shared__ __align__(16) float s_B[kTile][kTile];

    const int tid = threadIdx.x;
    const int tx = tid & 15;
    const int ty = tid >> 4;
    const int r0 = ty * 4;
    const int c0 = tx * 4;

    const int bx = static_cast<int>(blockIdx.x);
    const int by = static_cast<int>(blockIdx.y);
    const int tile_i = (by < r) ? by : (by + 1);
    const int tile_j = (bx < r) ? bx : (bx + 1);

    const int base_i = tile_i * kTile;
    const int base_r = r * kTile;
    const int base_j = tile_j * kTile;

    const int rem_k = n - base_r;
    const int valid_k = (rem_k < kTile) ? rem_k : kTile;
    const float inf = pos_inf();

    const bool vec_ok = ((n & 3) == 0) && is_aligned_16b(d) && (base_i + kTile <= n) &&
                        (base_r + kTile <= n) && (base_j + kTile <= n);

    float4 o0, o1, o2, o3;

    if (vec_ok) {
        const float4* d4 = reinterpret_cast<const float4*>(d);
        const size_t stride4 = static_cast<size_t>(n >> 2);
        const size_t a_col4 = static_cast<size_t>(base_r >> 2) + tx;
        const size_t b_col4 = static_cast<size_t>(base_j >> 2) + tx;

        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int sr = r0 + m;
            const float4 av = __ldg(&d4[static_cast<size_t>(base_i + sr) * stride4 + a_col4]);
            s_AT[c0 + 0][sr] = av.x;
            s_AT[c0 + 1][sr] = av.y;
            s_AT[c0 + 2][sr] = av.z;
            s_AT[c0 + 3][sr] = av.w;

            const float4 bv = __ldg(&d4[static_cast<size_t>(base_r + sr) * stride4 + b_col4]);
            reinterpret_cast<float4*>(s_B[sr])[tx] = bv;
        }

        o0 = __ldg(&d4[static_cast<size_t>(base_i + r0 + 0) * stride4 + b_col4]);
        o1 = __ldg(&d4[static_cast<size_t>(base_i + r0 + 1) * stride4 + b_col4]);
        o2 = __ldg(&d4[static_cast<size_t>(base_i + r0 + 2) * stride4 + b_col4]);
        o3 = __ldg(&d4[static_cast<size_t>(base_i + r0 + 3) * stride4 + b_col4]);
    } else {
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int sr = r0 + m;
            const int gi_a = base_i + sr;
            const int gi_b = base_r + sr;
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int sc = c0 + q;
                const int gj_a = base_r + sc;
                const int gj_b = base_j + sc;
                s_AT[sc][sr] =
                    (gi_a < n && gj_a < n) ? __ldg(&d[static_cast<size_t>(gi_a) * n + gj_a]) : inf;
                s_B[sr][sc] =
                    (gi_b < n && gj_b < n) ? __ldg(&d[static_cast<size_t>(gi_b) * n + gj_b]) : inf;
            }
        }

        float tmp[4][4];
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int gr = base_i + r0 + m;
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int gc = base_j + c0 + q;
                tmp[m][q] =
                    (gr < n && gc < n) ? __ldg(&d[static_cast<size_t>(gr) * n + gc]) : inf;
            }
        }
        o0 = make_float4(tmp[0][0], tmp[0][1], tmp[0][2], tmp[0][3]);
        o1 = make_float4(tmp[1][0], tmp[1][1], tmp[1][2], tmp[1][3]);
        o2 = make_float4(tmp[2][0], tmp[2][1], tmp[2][2], tmp[2][3]);
        o3 = make_float4(tmp[3][0], tmp[3][1], tmp[3][2], tmp[3][3]);
    }
    __syncthreads();

    float c00 = o0.x, c01 = o0.y, c02 = o0.z, c03 = o0.w;
    float c10 = o1.x, c11 = o1.y, c12 = o1.z, c13 = o1.w;
    float c20 = o2.x, c21 = o2.y, c22 = o2.z, c23 = o2.w;
    float c30 = o3.x, c31 = o3.y, c32 = o3.z, c33 = o3.w;

    if (valid_k == kTile) {
        #pragma unroll 16
        for (int k = 0; k < kTile; ++k) {
            const float4 a = reinterpret_cast<const float4*>(s_AT[k])[ty];
            const float4 b = reinterpret_cast<const float4*>(s_B[k])[tx];

            c00 = fminf(c00, a.x + b.x);
            c01 = fminf(c01, a.x + b.y);
            c02 = fminf(c02, a.x + b.z);
            c03 = fminf(c03, a.x + b.w);

            c10 = fminf(c10, a.y + b.x);
            c11 = fminf(c11, a.y + b.y);
            c12 = fminf(c12, a.y + b.z);
            c13 = fminf(c13, a.y + b.w);

            c20 = fminf(c20, a.z + b.x);
            c21 = fminf(c21, a.z + b.y);
            c22 = fminf(c22, a.z + b.z);
            c23 = fminf(c23, a.z + b.w);

            c30 = fminf(c30, a.w + b.x);
            c31 = fminf(c31, a.w + b.y);
            c32 = fminf(c32, a.w + b.z);
            c33 = fminf(c33, a.w + b.w);
        }
    } else {
        for (int k = 0; k < valid_k; ++k) {
            const float4 a = reinterpret_cast<const float4*>(s_AT[k])[ty];
            const float4 b = reinterpret_cast<const float4*>(s_B[k])[tx];

            c00 = fminf(c00, a.x + b.x);
            c01 = fminf(c01, a.x + b.y);
            c02 = fminf(c02, a.x + b.z);
            c03 = fminf(c03, a.x + b.w);

            c10 = fminf(c10, a.y + b.x);
            c11 = fminf(c11, a.y + b.y);
            c12 = fminf(c12, a.y + b.z);
            c13 = fminf(c13, a.y + b.w);

            c20 = fminf(c20, a.z + b.x);
            c21 = fminf(c21, a.z + b.y);
            c22 = fminf(c22, a.z + b.z);
            c23 = fminf(c23, a.z + b.w);

            c30 = fminf(c30, a.w + b.x);
            c31 = fminf(c31, a.w + b.y);
            c32 = fminf(c32, a.w + b.z);
            c33 = fminf(c33, a.w + b.w);
        }
    }

    if (vec_ok) {
        float4* d4 = reinterpret_cast<float4*>(d);
        const size_t stride4 = static_cast<size_t>(n >> 2);
        const size_t c_col4 = static_cast<size_t>(base_j >> 2) + tx;

        if (c00 != o0.x || c01 != o0.y || c02 != o0.z || c03 != o0.w) {
            d4[static_cast<size_t>(base_i + r0 + 0) * stride4 + c_col4] =
                make_float4(c00, c01, c02, c03);
        }
        if (c10 != o1.x || c11 != o1.y || c12 != o1.z || c13 != o1.w) {
            d4[static_cast<size_t>(base_i + r0 + 1) * stride4 + c_col4] =
                make_float4(c10, c11, c12, c13);
        }
        if (c20 != o2.x || c21 != o2.y || c22 != o2.z || c23 != o2.w) {
            d4[static_cast<size_t>(base_i + r0 + 2) * stride4 + c_col4] =
                make_float4(c20, c21, c22, c23);
        }
        if (c30 != o3.x || c31 != o3.y || c32 != o3.z || c33 != o3.w) {
            d4[static_cast<size_t>(base_i + r0 + 3) * stride4 + c_col4] =
                make_float4(c30, c31, c32, c33);
        }
    } else {
        const float rows[4][4] = {
            {c00, c01, c02, c03},
            {c10, c11, c12, c13},
            {c20, c21, c22, c23},
            {c30, c31, c32, c33},
        };
        const float orig[4][4] = {
            {o0.x, o0.y, o0.z, o0.w},
            {o1.x, o1.y, o1.z, o1.w},
            {o2.x, o2.y, o2.z, o2.w},
            {o3.x, o3.y, o3.z, o3.w},
        };
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int gr = base_i + r0 + m;
            if (gr < n) {
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int gc = base_j + c0 + q;
                    if (gc < n && rows[m][q] != orig[m][q]) {
                        d[static_cast<size_t>(gr) * n + gc] = rows[m][q];
                    }
                }
            }
        }
    }
}

}  // namespace

// dist, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* dist, float* output, int N) {
    if (N <= 0) {
        return;
    }

    if (N <= kTile) {
        fw_single_tile_le64_kernel<<<1, kThreads>>>(dist, output, N);
        return;
    }

    if (dist != output) {
        const size_t bytes = static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(float);
        cudaMemcpyAsync(output, dist, bytes, cudaMemcpyDeviceToDevice);
    }

    const int num_tiles = (N + kTile - 1) / kTile;
    const dim3 panel_grid(static_cast<unsigned>(num_tiles - 1), 2u);
    const dim3 remain_grid(static_cast<unsigned>(num_tiles - 1),
                           static_cast<unsigned>(num_tiles - 1));

    for (int r = 0; r < num_tiles; ++r) {
        fw_phase1_diag_kernel<<<1, kThreads>>>(output, N, r);
        fw_phase2_panels_kernel<<<panel_grid, kThreads>>>(output, N, r);
        fw_phase3_remain_kernel<<<remain_grid, kThreads>>>(output, N, r);
    }
}
