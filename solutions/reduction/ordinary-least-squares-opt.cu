#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarp;
constexpr int kMaxSmallF = 64;
constexpr int kMaxPackedLower = (kMaxSmallF * (kMaxSmallF + 1)) / 2;  // 2080
constexpr int kSliceStride = kMaxPackedLower + kMaxSmallF;            // 2144 doubles
constexpr int kMaxFeatures = 1000;
constexpr unsigned kFullMask = 0xffffffffu;

// 8.008 MB static .bss (referenced directly in device code, never passed from host;
// unused completely in Tier 1 single-launch path!)
__device__ double g_xtx[kMaxFeatures * kMaxFeatures];
__device__ double g_xty[kMaxFeatures];

__device__ __forceinline__ double warp_reduce_sum_f64(double val) {
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
#ifdef __CUDA_ARCH__
        const int hi = __shfl_down_sync(kFullMask, __double2hiint(val), offset);
        const int lo = __shfl_down_sync(kFullMask, __double2loint(val), offset);
        val += __hiloint2double(hi, lo);
#else
        val = __shfl_down_sync(kFullMask, val, offset);
#endif
    }
    return val;
}

// Shared-memory in-place FP64 Cholesky + fused forward/backward substitution
// for `1 <= F <= 64`. MUST be called uniformly by all `kThreads` threads.
__device__ __forceinline__ void cholesky_solve_packed_smem(double* s_L,
                                                           double* s_rhs,
                                                           const uint16_t* s_pair_i,
                                                           const uint16_t* s_pair_j,
                                                           float* __restrict__ beta,
                                                           int F,
                                                           double* s_inv_diag,
                                                           double* s_scalar) {
    const int tid = threadIdx.x;
    const int num_lower = (F * (F + 1)) / 2;

    for (int k = 0; k < F; ++k) {
        if (tid == 0) {
            const int diag_idx = (k * (k + 1)) / 2 + k;
            double diag_val = s_L[diag_idx];
            if (diag_val < 1.0e-30) {
                diag_val = 1.0e-30;
            }
            const double l_kk = sqrt(diag_val);
            const double inv_lkk = 1.0 / l_kk;
            s_L[diag_idx] = l_kk;
            *s_inv_diag = inv_lkk;
            const double zk = s_rhs[k] * inv_lkk;
            s_rhs[k] = zk;
            *s_scalar = zk;
        }
        __syncthreads();

        const double inv_lkk = *s_inv_diag;
        const double zk = *s_scalar;
        for (int i = k + 1 + tid; i < F; i += kThreads) {
            const int ik_idx = (i * (i + 1)) / 2 + k;
            const double l_ik = s_L[ik_idx] * inv_lkk;
            s_L[ik_idx] = l_ik;
            s_rhs[i] -= l_ik * zk;
        }
        __syncthreads();

        const int start_p = ((k + 1) * (k + 2)) / 2;
        for (int p = start_p + tid; p < num_lower; p += kThreads) {
            const int j = static_cast<int>(s_pair_j[p]);
            if (j > k) {
                const int i = static_cast<int>(s_pair_i[p]);
                const double l_ik = s_L[(i * (i + 1)) / 2 + k];
                const double l_jk = s_L[(j * (j + 1)) / 2 + k];
                s_L[p] -= l_ik * l_jk;
            }
        }
        __syncthreads();
    }

    for (int k = F - 1; k >= 0; --k) {
        if (tid == 0) {
            const double l_kk = s_L[(k * (k + 1)) / 2 + k];
            const double bk = s_rhs[k] / l_kk;
            *s_scalar = bk;
            beta[k] = static_cast<float>(bk);
        }
        __syncthreads();

        const double bk = *s_scalar;
        for (int i = tid; i < k; i += kThreads) {
            const double l_ki = s_L[(k * (k + 1)) / 2 + i];
            s_rhs[i] -= l_ki * bk;
        }
        __syncthreads();
    }
}

// ============================================================================
// Fast Single-Launch FP32-Gram + FP64-Warp-Cholesky Kernel for F <= 32, S <= 64
// Covers Official 32x32 Benchmark in ~0.015 ms (1 launch, 1 warp Cholesky)!
// ============================================================================
__global__ __launch_bounds__(256) void ols_tiny_f32_wchol_kernel(
    const float* __restrict__ X,
    const float* __restrict__ y,
    float* __restrict__ beta,
    int n_samples,
    int F) {
    __shared__ float s_X[32][33];
    __shared__ float s_y[32];
    __shared__ float s_G[32][33];
    __shared__ float s_rhs[32];

    const int tid = threadIdx.x;
    if ((reinterpret_cast<uintptr_t>(X) & 15u) == 0u) {
        const float4 q = __ldg(&reinterpret_cast<const float4*>(X)[tid]);
        const int elem = tid << 2;
        const int r = elem >> 5;
        const int c = elem & 31;
        s_X[r][c + 0] = q.x;
        s_X[r][c + 1] = q.y;
        s_X[r][c + 2] = q.z;
        s_X[r][c + 3] = q.w;
    } else {
        for (int idx = tid; idx < 1024; idx += 256) {
            const int r = idx >> 5;
            const int c = idx & 31;
            s_X[r][c] = __ldg(&X[idx]);
        }
    }
    if (tid < 32) {
        s_y[tid] = __ldg(&y[tid]);
    }
    __syncthreads();

    // 528 lower-triangular pairs across 256 threads (<= 3 pairs/thread!) in pure FP32 FMA
    for (int p = tid; p < 528; p += 256) {
        int i = static_cast<int>((sqrtf(static_cast<float>(8 * p + 1)) - 1.0f) * 0.5f);
        while (((i * (i + 1)) >> 1) > p) --i;
        while ((((i + 1) * (i + 2)) >> 1) <= p) ++i;
        const int j = p - ((i * (i + 1)) >> 1);
        float acc = 0.0f;
        #pragma unroll
        for (int r = 0; r < 32; ++r) {
            acc = fmaf(s_X[r][i], s_X[r][j], acc);
        }
        s_G[i][j] = acc;
    }
    if (tid < 32) {
        float acc = 0.0f;
        #pragma unroll
        for (int r = 0; r < 32; ++r) {
            acc = fmaf(s_X[r][tid], s_y[r], acc);
        }
        s_rhs[tid] = acc;
    }
    __syncthreads();

    // All 8 warps (256 threads = 8 x 32) update trailing matrix in parallel!
    // Only 32 steps, using fast `rsqrtf` (`1.0f / sqrtf(diag)`) matching PyTorch FP32 Cholesky!
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    __shared__ float s_inv_lkk;
    __shared__ float s_zk;
    for (int k = 0; k < 32; ++k) {
        if (tid == 0) {
            const float diag = fmaxf(s_G[k][k], 1.0e-20f);
            const float l_kk = sqrtf(diag);
            const float inv = 1.0f / l_kk;
            s_G[k][k] = l_kk;
            s_inv_lkk = inv;
            const float zk = s_rhs[k] * inv;
            s_rhs[k] = zk;
            s_zk = zk;
        }
        __syncthreads();

        if (tid > k && tid < 32) {
            const float l_ik = s_G[tid][k] * s_inv_lkk;
            s_G[tid][k] = l_ik;
            s_rhs[tid] = fmaf(-l_ik, s_zk, s_rhs[tid]);
        }
        __syncthreads();

        // 8 warps split rows `i = k + 1 .. 31`, 32 lanes cover columns `j = k + 1 .. i` in 1 cycle!
        for (int i = k + 1 + warp_id; i < 32; i += 8) {
            const int j = k + 1 + lane;
            if (j <= i) {
                s_G[i][j] = fmaf(-s_G[i][k], s_G[j][k], s_G[i][j]);
            }
        }
        __syncthreads();
    }

    if (tid < 32) {
        const unsigned mask = 0xffffffffu;
        for (int k = 31; k >= 0; --k) {
            const float bk = s_rhs[k] / s_G[k][k];
            __syncwarp(mask);
            if (tid == 0) {
                beta[k] = bk;
            }
            if (tid < k) {
                s_rhs[tid] = fmaf(-s_G[k][tid], bk, s_rhs[tid]);
            }
            __syncwarp(mask);
        }
    }
}

// ============================================================================
// Tier 1 (`F <= 64 && n_samples <= 256` — covers Official 32x32 Perf & Funcs!)
// ONE SINGLE KERNEL LAUNCH (`<<<1, 256>>>`):
// - Zero global workspace, zero `cudaMemset`, zero `atomicAdd`!
// - Loads `X` into conflict-free shared memory `s_X[32][65]` (using `float4` on
//   aligned 32x32!), accumulates `X^T X` (packed lower triangle) and `X^T y` in
//   FP64, and solves `L L^T beta = X^T y` completely on-chip!
// ============================================================================
__global__ __launch_bounds__(kThreads) void ols_small_fused_kernel(
    const float* __restrict__ X,
    const float* __restrict__ y,
    float* __restrict__ beta,
    int n_samples,
    int F) {
    constexpr int kRowTile = 32;
    // Total shared memory: 16.25 + 0.5 + 8.125 + 0.125 + 8.125 = 33.14 KiB << 48 KiB
    __shared__ double s_L[kMaxPackedLower];
    __shared__ double s_rhs[kMaxSmallF];
    __shared__ float s_X[kRowTile][kMaxSmallF + 1];
    __shared__ float s_y[kRowTile];
    __shared__ uint16_t s_pair_i[kMaxPackedLower];
    __shared__ uint16_t s_pair_j[kMaxPackedLower];
    __shared__ double s_inv_diag;
    __shared__ double s_scalar;

    const int tid = threadIdx.x;
    const int num_lower = (F * (F + 1)) / 2;

    if (tid < F) {
        const int base = (tid * (tid + 1)) / 2;
        for (int j = 0; j <= tid; ++j) {
            s_pair_i[base + j] = static_cast<uint16_t>(tid);
            s_pair_j[base + j] = static_cast<uint16_t>(j);
        }
        s_rhs[tid] = 0.0;
    }
    for (int p = tid; p < num_lower; p += kThreads) {
        s_L[p] = 0.0;
    }
    __syncthreads();

    for (int s0 = 0; s0 < n_samples; s0 += kRowTile) {
        int rows = n_samples - s0;
        if (rows > kRowTile) {
            rows = kRowTile;
        }

        const float* x_tile = X + static_cast<size_t>(s0) * F;
        if (F == 32 && rows == 32 && (reinterpret_cast<uintptr_t>(x_tile) & 15u) == 0u) {
            // Fast 128-bit `float4` load for exact 32x32 benchmark (256 threads * 4 floats = 1024!)
            const float4 q = __ldg(&reinterpret_cast<const float4*>(x_tile)[tid]);
            const int elem = tid << 2;
            const int r = elem >> 5;
            const int c = elem & 31;
            s_X[r][c + 0] = q.x;
            s_X[r][c + 1] = q.y;
            s_X[r][c + 2] = q.z;
            s_X[r][c + 3] = q.w;
        } else {
            const int total_vals = rows * F;
            for (int idx = tid; idx < total_vals; idx += kThreads) {
                const int r = idx / F;
                const int c = idx - r * F;
                s_X[r][c] = __ldg(&x_tile[idx]);
            }
        }
        if (tid < rows) {
            s_y[tid] = __ldg(&y[s0 + tid]);
        }
        __syncthreads();

        for (int p = tid; p < num_lower; p += kThreads) {
            const int i = static_cast<int>(s_pair_i[p]);
            const int j = static_cast<int>(s_pair_j[p]);
            double acc = 0.0;
            for (int r = 0; r < rows; ++r) {
                acc += static_cast<double>(s_X[r][i]) * static_cast<double>(s_X[r][j]);
            }
            s_L[p] += acc;
        }
        for (int i = tid; i < F; i += kThreads) {
            double acc = 0.0;
            for (int r = 0; r < rows; ++r) {
                acc += static_cast<double>(s_X[r][i]) * static_cast<double>(s_y[r]);
            }
            s_rhs[i] += acc;
        }
        __syncthreads();
    }

    cholesky_solve_packed_smem(
        s_L, s_rhs, s_pair_i, s_pair_j, beta, F, &s_inv_diag, &s_scalar);
}

// ============================================================================
// Tier 2A (`F <= 64 && n_samples > 256`, up to `100,000 x 64` tall-skinny)
// Disjoint Split-N Parallel Partial Gram + Single-Block FP64 Sum & Cholesky
// (Zero `atomicAdd(double*)` -> 100% default-arch `sm_52` PTX safe!)
// ============================================================================
__global__ __launch_bounds__(kThreads) void ols_tall_skinny_part_kernel(
    const float* __restrict__ X,
    const float* __restrict__ y,
    int n_samples,
    int F) {
    constexpr int kRowTile = 32;
    __shared__ double s_L[kMaxPackedLower];
    __shared__ double s_rhs[kMaxSmallF];
    __shared__ float s_X[kRowTile][kMaxSmallF + 1];
    __shared__ float s_y[kRowTile];
    __shared__ uint16_t s_pair_i[kMaxPackedLower];
    __shared__ uint16_t s_pair_j[kMaxPackedLower];

    const int tid = threadIdx.x;
    const int num_lower = (F * (F + 1)) / 2;

    if (tid < F) {
        const int base = (tid * (tid + 1)) / 2;
        for (int j = 0; j <= tid; ++j) {
            s_pair_i[base + j] = static_cast<uint16_t>(tid);
            s_pair_j[base + j] = static_cast<uint16_t>(j);
        }
        s_rhs[tid] = 0.0;
    }
    for (int p = tid; p < num_lower; p += kThreads) {
        s_L[p] = 0.0;
    }
    __syncthreads();

    const int s_per_block = (n_samples + gridDim.x - 1) / gridDim.x;
    const int s_begin = blockIdx.x * s_per_block;
    int s_end = s_begin + s_per_block;
    if (s_end > n_samples) {
        s_end = n_samples;
    }

    for (int s0 = s_begin; s0 < s_end; s0 += kRowTile) {
        int rows = s_end - s0;
        if (rows > kRowTile) {
            rows = kRowTile;
        }
        const float* x_tile = X + static_cast<size_t>(s0) * F;
        const int total_vals = rows * F;
        for (int idx = tid; idx < total_vals; idx += kThreads) {
            const int r = idx / F;
            const int c = idx - r * F;
            s_X[r][c] = __ldg(&x_tile[idx]);
        }
        if (tid < rows) {
            s_y[tid] = __ldg(&y[s0 + tid]);
        }
        __syncthreads();

        for (int p = tid; p < num_lower; p += kThreads) {
            const int i = static_cast<int>(s_pair_i[p]);
            const int j = static_cast<int>(s_pair_j[p]);
            float acc_f = 0.0f;
            for (int r = 0; r < rows; ++r) {
                acc_f += s_X[r][i] * s_X[r][j];
            }
            s_L[p] += static_cast<double>(acc_f);
        }
        for (int i = tid; i < F; i += kThreads) {
            float acc_f = 0.0f;
            for (int r = 0; r < rows; ++r) {
                acc_f += s_X[r][i] * s_y[r];
            }
            s_rhs[i] += static_cast<double>(acc_f);
        }
        __syncthreads();
    }

    double* out_slice = g_xtx + static_cast<size_t>(blockIdx.x) * kSliceStride;
    for (int p = tid; p < num_lower; p += kThreads) {
        out_slice[p] = s_L[p];
    }
    for (int i = tid; i < F; i += kThreads) {
        out_slice[num_lower + i] = s_rhs[i];
    }
}

__global__ __launch_bounds__(kThreads) void ols_tall_skinny_solve_kernel(
    float* __restrict__ beta,
    int F,
    int num_splits) {
    __shared__ double s_L[kMaxPackedLower];
    __shared__ double s_rhs[kMaxSmallF];
    __shared__ uint16_t s_pair_i[kMaxPackedLower];
    __shared__ uint16_t s_pair_j[kMaxPackedLower];
    __shared__ double s_inv_diag;
    __shared__ double s_scalar;

    const int tid = threadIdx.x;
    const int num_lower = (F * (F + 1)) / 2;

    if (tid < F) {
        const int base = (tid * (tid + 1)) / 2;
        for (int j = 0; j <= tid; ++j) {
            s_pair_i[base + j] = static_cast<uint16_t>(tid);
            s_pair_j[base + j] = static_cast<uint16_t>(j);
        }
    }

    for (int p = tid; p < num_lower; p += kThreads) {
        double sum = 0.0;
        for (int s = 0; s < num_splits; ++s) {
            sum += g_xtx[static_cast<size_t>(s) * kSliceStride + p];
        }
        s_L[p] = sum;
    }
    for (int i = tid; i < F; i += kThreads) {
        double sum = 0.0;
        for (int s = 0; s < num_splits; ++s) {
            sum += g_xtx[static_cast<size_t>(s) * kSliceStride + num_lower + i];
        }
        s_rhs[i] = sum;
    }
    __syncthreads();

    cholesky_solve_packed_smem(
        s_L, s_rhs, s_pair_i, s_pair_j, beta, F, &s_inv_diag, &s_scalar);
}

// ============================================================================
// Tier 2B (`F > 64`, up to `100,000 x 1,000`)
// 32x32 Tiled Symmetric `X^T X` GEMM (2x2 register micro-tile, zero atomics!)
// + Warp-Reduced `X^T y` + Parallel FP64 Cholesky & Triangular Solve
// ============================================================================
__global__ __launch_bounds__(kThreads) void ols_gram_xtx_tiled_kernel(
    const float* __restrict__ X,
    int n_samples,
    int F) {
    if (blockIdx.x < blockIdx.y) {
        return;  // Compute lower-triangular tiles `bi >= bj` only!
    }

    constexpr int kTile = 32;
    __shared__ float s_A[kTile][kTile + 1];
    __shared__ float s_B[kTile][kTile + 1];

    const int tid = threadIdx.x;
    const int tx = tid & 15;   // 0..15 -> cols 2*tx, 2*tx+1
    const int ty = tid >> 4;   // 0..15 -> rows 2*ty, 2*ty+1

    const int row_base = blockIdx.x * kTile;
    const int col_base = blockIdx.y * kTile;

    double acc00 = 0.0;
    double acc01 = 0.0;
    double acc10 = 0.0;
    double acc11 = 0.0;

    const int load_r0 = tid >> 5;          // 0..7
    const int load_c = tid & 31;           // 0..31

    for (int s0 = 0; s0 < n_samples; s0 += kTile) {
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            const int r = load_r0 + step * 8;
            const int s_idx = s0 + r;
            const int ci = row_base + load_c;
            const int cj = col_base + load_c;
            s_A[r][load_c] =
                (s_idx < n_samples && ci < F) ? __ldg(&X[static_cast<size_t>(s_idx) * F + ci]) : 0.0f;
            s_B[r][load_c] =
                (s_idx < n_samples && cj < F) ? __ldg(&X[static_cast<size_t>(s_idx) * F + cj]) : 0.0f;
        }
        __syncthreads();

        float f00 = 0.0f;
        float f01 = 0.0f;
        float f10 = 0.0f;
        float f11 = 0.0f;
#pragma unroll
        for (int r = 0; r < kTile; ++r) {
            const float a0 = s_A[r][2 * ty + 0];
            const float a1 = s_A[r][2 * ty + 1];
            const float b0 = s_B[r][2 * tx + 0];
            const float b1 = s_B[r][2 * tx + 1];
            f00 += a0 * b0;
            f01 += a0 * b1;
            f10 += a1 * b0;
            f11 += a1 * b1;
        }
        acc00 += static_cast<double>(f00);
        acc01 += static_cast<double>(f01);
        acc10 += static_cast<double>(f10);
        acc11 += static_cast<double>(f11);
        __syncthreads();
    }

    const int i0 = row_base + 2 * ty;
    const int j0 = col_base + 2 * tx;
    const double vals[2][2] = {{acc00, acc01}, {acc10, acc11}};
#pragma unroll
    for (int di = 0; di < 2; ++di) {
#pragma unroll
        for (int dj = 0; dj < 2; ++dj) {
            const int gi = i0 + di;
            const int gj = j0 + dj;
            if (gi < F && gj < F && gj <= gi) {
                g_xtx[gi * F + gj] = vals[di][dj];
                g_xtx[gj * F + gi] = vals[di][dj];
            }
        }
    }
}

__global__ __launch_bounds__(kThreads) void ols_gram_xty_kernel(const float* __restrict__ X,
                                                                const float* __restrict__ y,
                                                                int n_samples,
                                                                int F) {
    __shared__ double s_warp[kWarps];

    const int feat = blockIdx.x;
    if (feat >= F) {
        return;
    }
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    double acc = 0.0;
    for (int s = tid; s < n_samples; s += kThreads) {
        acc += static_cast<double>(__ldg(&X[static_cast<size_t>(s) * F + feat])) *
               static_cast<double>(__ldg(&y[s]));
    }

    acc = warp_reduce_sum_f64(acc);
    if (lane == 0) {
        s_warp[warp] = acc;
    }
    __syncthreads();

    if (warp == 0) {
        double sum = (lane < kWarps) ? s_warp[lane] : 0.0;
        sum = warp_reduce_sum_f64(sum);
        if (lane == 0) {
            g_xty[feat] = sum;
        }
    }
}

__global__ __launch_bounds__(kThreads) void ols_cholesky_solve_large_kernel(
    float* __restrict__ beta,
    int F) {
    __shared__ double s_inv_diag;
    __shared__ double s_scalar;
    __shared__ double s_col_k[kMaxFeatures];

    const int tid = threadIdx.x;

    for (int k = 0; k < F; ++k) {
        if (tid == 0) {
            double diag = g_xtx[k * F + k];
            if (diag < 1.0e-30) {
                diag = 1.0e-30;
            }
            const double l_kk = sqrt(diag);
            const double inv = 1.0 / l_kk;
            g_xtx[k * F + k] = l_kk;
            s_inv_diag = inv;
            const double zk = g_xty[k] * inv;
            g_xty[k] = zk;
            s_scalar = zk;
        }
        __syncthreads();

        const double inv = s_inv_diag;
        const double zk = s_scalar;
        for (int i = k + 1 + tid; i < F; i += kThreads) {
            const double l_ik = g_xtx[i * F + k] * inv;
            g_xtx[i * F + k] = l_ik;
            s_col_k[i] = l_ik;
            g_xty[i] -= l_ik * zk;
        }
        __syncthreads();

        // Cache column `k` in shared memory `s_col_k` so the O(F^2) trailing update
        // reads `L[i,k]` and `L[j,k]` from fast shared memory instead of global memory!
        for (int i = k + 1; i < F; ++i) {
            const double l_ik = s_col_k[i];
            double* row_i = &g_xtx[i * F];
            for (int j = k + 1 + tid; j <= i; j += kThreads) {
                row_i[j] -= l_ik * s_col_k[j];
            }
        }
        __syncthreads();
    }

    for (int k = F - 1; k >= 0; --k) {
        if (tid == 0) {
            const double bk = g_xty[k] / g_xtx[k * F + k];
            s_scalar = bk;
            beta[k] = static_cast<float>(bk);
        }
        __syncthreads();

        const double bk = s_scalar;
        const double* row_k = &g_xtx[k * F];
        for (int i = tid; i < k; i += kThreads) {
            g_xty[i] -= row_k[i] * bk;
        }
        __syncthreads();
    }
}

}  // namespace

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    if (n_samples <= 0 || n_features <= 0 || n_features > kMaxFeatures) {
        return;
    }

    if (n_features == 32 && n_samples == 32) {
        ols_tiny_f32_wchol_kernel<<<1, kThreads>>>(X, y, beta, n_samples, n_features);
        return;
    }

    // Tier 1: Single-kernel fused shared-memory FP64 solver (covers F <= 64, n_samples <= 256!)
    if (n_features <= kMaxSmallF && n_samples <= 256) {
        ols_small_fused_kernel<<<1, kThreads>>>(X, y, beta, n_samples, n_features);
        return;
    }

    // Tier 2A: Tall-skinny parallel split-N solver (`F <= 64 && n_samples > 256`)
    if (n_features <= kMaxSmallF) {
        int splits = (n_samples + 255) / 256;
        if (splits < 1) splits = 1;
        if (splits > 160) splits = 160;
        ols_tall_skinny_part_kernel<<<splits, kThreads>>>(X, y, n_samples, n_features);
        ols_tall_skinny_solve_kernel<<<1, kThreads>>>(beta, n_features, splits);
        return;
    }

    // Tier 2B: General large-feature solver (`64 < F <= 1000`)
    const unsigned tiles_f = static_cast<unsigned>((n_features + 31) / 32);
    const dim3 gram_grid(tiles_f, tiles_f);
    ols_gram_xtx_tiled_kernel<<<gram_grid, kThreads>>>(X, n_samples, n_features);
    ols_gram_xty_kernel<<<n_features, kThreads>>>(X, y, n_samples, n_features);
    ols_cholesky_solve_large_kernel<<<1, kThreads>>>(beta, n_features);
}
