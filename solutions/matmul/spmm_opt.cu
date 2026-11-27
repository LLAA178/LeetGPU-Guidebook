#include <cuda_runtime.h>
#include <stdint.h>

// Optimized SpMM v2 for SM 7.5 (Turing) and general GPUs:
// 1. All-zero fast path (nnz == 0 -> cudaMemset).
// 2. For moderate sparsity (e.g., 60-70% sparse, i.e., 30-40% non-zero, such as the
//    official 4096x2048x512 benchmark), a double-buffered 128x128 / 8x8 register-tiled
//    FP32 SGEMM kernel amortizes B loads 128x per block with zero warp-control overhead
//    and saturates FP32 FMA throughput (~5+ TFLOP/s on T4).
// 3. For ultra-sparse matrices (density < 10%) or shapes not divisible by 128, a
//    shared-memory-B + warp-uniform __ballot_sync float4 SpMM kernel skips zeros with
//    zero warp divergence.
// 4. General fallback for arbitrary unaligned shapes.

namespace {

// ============================================================================
// Kernel 1: Double-buffered 128x128 / 8x8 register-tiled FP32 SGEMM
// A: [M, N], B: [N, K], C: [M, K] (N is reduction dim, K is output col dim)
// Requires: M % 128 == 0, K % 128 == 0, N % 8 == 0, 16-byte aligned A, B, C
// ============================================================================
__global__ __launch_bounds__(256, 2) void spmm_sgemm_128x128_db_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
    // sA[2][8][132]: transposed A tile (k, m), padded stride 132 (multiple of 4)
    // sB[2][8][128]: B tile (k, n)
    __shared__ float sA[2][8][132];
    __shared__ float sB[2][8][128];

    const int tid = threadIdx.x;  // 0..255
    const int ty = tid >> 4;      // 0..15 (M-subtile)
    const int tx = tid & 15;      // 0..15 (K-subtile)

    const int block_m = blockIdx.y * 128;
    const int block_k = blockIdx.x * 128;

    // Cooperative load indices (1 float4 per thread for A, 1 float4 per thread for B)
    const int a_row = tid >> 1;   // 0..127
    const int a_col4 = tid & 1;   // 0..1 (cols 0..3 or 4..7)
    const int b_row = tid >> 5;   // 0..7
    const int b_col4 = tid & 31;  // 0..31 (cols 0..127 in float4)

    const float* A_ptr = A + (size_t)(block_m + a_row) * N + a_col4 * 4;
    const float* B_ptr = B + (size_t)b_row * K + block_k + b_col4 * 4;

    // 8x8 register accumulators per thread
    float4 c0[4], c1[4], c2[4], c3[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        c0[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        c1[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        c2[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        c3[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Prefetch first tile (n_base = 0) into buffer 0
    {
        float4 a_ld = *reinterpret_cast<const float4*>(A_ptr);
        float4 b_ld = *reinterpret_cast<const float4*>(B_ptr);
        sA[0][a_col4 * 4 + 0][a_row] = a_ld.x;
        sA[0][a_col4 * 4 + 1][a_row] = a_ld.y;
        sA[0][a_col4 * 4 + 2][a_row] = a_ld.z;
        sA[0][a_col4 * 4 + 3][a_row] = a_ld.w;
        *reinterpret_cast<float4*>(&sB[0][b_row][b_col4 * 4]) = b_ld;
    }
    __syncthreads();

    int cur = 0;
    for (int n_base = 0; n_base < N; n_base += 8) {
        const int next = cur ^ 1;
        float4 a_next = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 b_next = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const bool has_next = (n_base + 8 < N);
        if (has_next) {
            A_ptr += 8;
            B_ptr += (size_t)8 * K;
            a_next = *reinterpret_cast<const float4*>(A_ptr);
            b_next = *reinterpret_cast<const float4*>(B_ptr);
        }

        // Compute 8 outer products from buffer `cur`
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            float4 a0 = *reinterpret_cast<const float4*>(&sA[cur][kk][ty * 4]);
            float4 a1 = *reinterpret_cast<const float4*>(&sA[cur][kk][ty * 4 + 64]);
            float4 b0 = *reinterpret_cast<const float4*>(&sB[cur][kk][tx * 4]);
            float4 b1 = *reinterpret_cast<const float4*>(&sB[cur][kk][tx * 4 + 64]);

            // Quadrant (0, 0) and (0, 1)
            c0[0].x = fmaf(a0.x, b0.x, c0[0].x); c0[0].y = fmaf(a0.x, b0.y, c0[0].y);
            c0[0].z = fmaf(a0.x, b0.z, c0[0].z); c0[0].w = fmaf(a0.x, b0.w, c0[0].w);
            c1[0].x = fmaf(a0.x, b1.x, c1[0].x); c1[0].y = fmaf(a0.x, b1.y, c1[0].y);
            c1[0].z = fmaf(a0.x, b1.z, c1[0].z); c1[0].w = fmaf(a0.x, b1.w, c1[0].w);

            c0[1].x = fmaf(a0.y, b0.x, c0[1].x); c0[1].y = fmaf(a0.y, b0.y, c0[1].y);
            c0[1].z = fmaf(a0.y, b0.z, c0[1].z); c0[1].w = fmaf(a0.y, b0.w, c0[1].w);
            c1[1].x = fmaf(a0.y, b1.x, c1[1].x); c1[1].y = fmaf(a0.y, b1.y, c1[1].y);
            c1[1].z = fmaf(a0.y, b1.z, c1[1].z); c1[1].w = fmaf(a0.y, b1.w, c1[1].w);

            c0[2].x = fmaf(a0.z, b0.x, c0[2].x); c0[2].y = fmaf(a0.z, b0.y, c0[2].y);
            c0[2].z = fmaf(a0.z, b0.z, c0[2].z); c0[2].w = fmaf(a0.z, b0.w, c0[2].w);
            c1[2].x = fmaf(a0.z, b1.x, c1[2].x); c1[2].y = fmaf(a0.z, b1.y, c1[2].y);
            c1[2].z = fmaf(a0.z, b1.z, c1[2].z); c1[2].w = fmaf(a0.z, b1.w, c1[2].w);

            c0[3].x = fmaf(a0.w, b0.x, c0[3].x); c0[3].y = fmaf(a0.w, b0.y, c0[3].y);
            c0[3].z = fmaf(a0.w, b0.z, c0[3].z); c0[3].w = fmaf(a0.w, b0.w, c0[3].w);
            c1[3].x = fmaf(a0.w, b1.x, c1[3].x); c1[3].y = fmaf(a0.w, b1.y, c1[3].y);
            c1[3].z = fmaf(a0.w, b1.z, c1[3].z); c1[3].w = fmaf(a0.w, b1.w, c1[3].w);

            // Quadrant (1, 0) and (1, 1)
            c2[0].x = fmaf(a1.x, b0.x, c2[0].x); c2[0].y = fmaf(a1.x, b0.y, c2[0].y);
            c2[0].z = fmaf(a1.x, b0.z, c2[0].z); c2[0].w = fmaf(a1.x, b0.w, c2[0].w);
            c3[0].x = fmaf(a1.x, b1.x, c3[0].x); c3[0].y = fmaf(a1.x, b1.y, c3[0].y);
            c3[0].z = fmaf(a1.x, b1.z, c3[0].z); c3[0].w = fmaf(a1.x, b1.w, c3[0].w);

            c2[1].x = fmaf(a1.y, b0.x, c2[1].x); c2[1].y = fmaf(a1.y, b0.y, c2[1].y);
            c2[1].z = fmaf(a1.y, b0.z, c2[1].z); c2[1].w = fmaf(a1.y, b0.w, c2[1].w);
            c3[1].x = fmaf(a1.y, b1.x, c3[1].x); c3[1].y = fmaf(a1.y, b1.y, c3[1].y);
            c3[1].z = fmaf(a1.y, b1.z, c3[1].z); c3[1].w = fmaf(a1.y, b1.w, c3[1].w);

            c2[2].x = fmaf(a1.z, b0.x, c2[2].x); c2[2].y = fmaf(a1.z, b0.y, c2[2].y);
            c2[2].z = fmaf(a1.z, b0.z, c2[2].z); c2[2].w = fmaf(a1.z, b0.w, c2[2].w);
            c3[2].x = fmaf(a1.z, b1.x, c3[2].x); c3[2].y = fmaf(a1.z, b1.y, c3[2].y);
            c3[2].z = fmaf(a1.z, b1.z, c3[2].z); c3[2].w = fmaf(a1.z, b1.w, c3[2].w);

            c2[3].x = fmaf(a1.w, b0.x, c2[3].x); c2[3].y = fmaf(a1.w, b0.y, c2[3].y);
            c2[3].z = fmaf(a1.w, b0.z, c2[3].z); c2[3].w = fmaf(a1.w, b0.w, c2[3].w);
            c3[3].x = fmaf(a1.w, b1.x, c3[3].x); c3[3].y = fmaf(a1.w, b1.y, c3[3].y);
            c3[3].z = fmaf(a1.w, b1.z, c3[3].z); c3[3].w = fmaf(a1.w, b1.w, c3[3].w);
        }

        if (has_next) {
            sA[next][a_col4 * 4 + 0][a_row] = a_next.x;
            sA[next][a_col4 * 4 + 1][a_row] = a_next.y;
            sA[next][a_col4 * 4 + 2][a_row] = a_next.z;
            sA[next][a_col4 * 4 + 3][a_row] = a_next.w;
            *reinterpret_cast<float4*>(&sB[next][b_row][b_col4 * 4]) = b_next;
            __syncthreads();
        }
        cur = next;
    }

    // Store 8x8 tile back to C using coalesced float4 writes
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int r0 = block_m + ty * 4 + i;
        int r1 = r0 + 64;
        int col0 = block_k + tx * 4;
        int col1 = col0 + 64;
        *reinterpret_cast<float4*>(C + (size_t)r0 * K + col0) = c0[i];
        *reinterpret_cast<float4*>(C + (size_t)r0 * K + col1) = c1[i];
        *reinterpret_cast<float4*>(C + (size_t)r1 * K + col0) = c2[i];
        *reinterpret_cast<float4*>(C + (size_t)r1 * K + col1) = c3[i];
    }
}

// ============================================================================
// Kernel 2: Shared-memory-B + warp-uniform __ballot_sync float4 SpMM
// Used for ultra-sparse matrices (density < 10%) or non-128-multiple shapes
// ============================================================================
constexpr int WARPS_PER_BLOCK = 8;
constexpr int ROWS_PER_WARP = 4;
constexpr int BM_SP = WARPS_PER_BLOCK * ROWS_PER_WARP;  // 32 rows of A per block
constexpr int BN_SP = 256;                               // 256 cols of B/C per block
constexpr int BK_SP = 32;                                // 32 K-steps per tile

__global__ void spmm_smem_ballot_f4_kernel(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int M, int N, int K) {
    __shared__ float4 B_smem[BK_SP][64];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int block_row = blockIdx.y * BM_SP;
    const int block_col4 = blockIdx.x * 64;
    const int K4 = K >> 2;
    const int warp_row_base = block_row + warp_id * ROWS_PER_WARP;

    float4 acc0[ROWS_PER_WARP];
    float4 acc1[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        acc0[r] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        acc1[r] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);

    const int col4_0 = block_col4 + lane;
    const int col4_1 = block_col4 + lane + 32;
    const bool valid0 = (col4_0 < K4);
    const bool valid1 = (col4_1 < K4);

    for (int k_base = 0; k_base < N; k_base += BK_SP) {
        #pragma unroll
        for (int s = 0; s < BK_SP; s += WARPS_PER_BLOCK) {
            int bk = s + warp_id;
            int gk = k_base + bk;
            float4 v0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 v1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (gk < N) {
                if (valid0) v0 = B4[(size_t)gk * K4 + col4_0];
                if (valid1) v1 = B4[(size_t)gk * K4 + col4_1];
            }
            B_smem[bk][lane] = v0;
            B_smem[bk][lane + 32] = v1;
        }
        __syncthreads();

        const bool k_lane_valid = (k_base + lane < N);
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            int row = warp_row_base + r;
            if (row < M) {
                float a_val = k_lane_valid ? A[(size_t)row * N + (k_base + lane)] : 0.0f;
                unsigned mask = __ballot_sync(0xffffffffu, a_val != 0.0f);
                while (mask) {
                    int bit = __ffs(mask) - 1;
                    mask &= (mask - 1u);
                    float a = __shfl_sync(0xffffffffu, a_val, bit);
                    float4 b0 = B_smem[bit][lane];
                    float4 b1 = B_smem[bit][lane + 32];
                    acc0[r].x = fmaf(a, b0.x, acc0[r].x);
                    acc0[r].y = fmaf(a, b0.y, acc0[r].y);
                    acc0[r].z = fmaf(a, b0.z, acc0[r].z);
                    acc0[r].w = fmaf(a, b0.w, acc0[r].w);
                    acc1[r].x = fmaf(a, b1.x, acc1[r].x);
                    acc1[r].y = fmaf(a, b1.y, acc1[r].y);
                    acc1[r].z = fmaf(a, b1.z, acc1[r].z);
                    acc1[r].w = fmaf(a, b1.w, acc1[r].w);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int row = warp_row_base + r;
        if (row < M) {
            if (valid0) C4[(size_t)row * K4 + col4_0] = acc0[r];
            if (valid1) C4[(size_t)row * K4 + col4_1] = acc1[r];
        }
    }
}

// ============================================================================
// Kernel 3: General fallback for arbitrary unaligned shapes
// ============================================================================
__global__ void spmm_general_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= K) return;

    float sum = 0.0f;
    const float* a_row = A + (size_t)row * N;
    for (int k = 0; k < N; ++k) {
        float a = a_row[k];
        if (a != 0.0f) {
            sum = fmaf(a, B[(size_t)k * K + col], sum);
        }
    }
    C[(size_t)row * K + col] = sum;
}

}  // namespace

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K, int nnz) {
    if (M <= 0 || K <= 0) return;

    if (nnz == 0) {
        cudaMemset(C, 0, (size_t)M * K * sizeof(float));
        return;
    }

    const bool aligned16_A = ((reinterpret_cast<uintptr_t>(A) & 15u) == 0u);
    const bool aligned16_BC =
        ((reinterpret_cast<uintptr_t>(B) & 15u) == 0u) &&
        ((reinterpret_cast<uintptr_t>(C) & 15u) == 0u);

    if ((M & 127) == 0 && (K & 127) == 0 && (N & 7) == 0 &&
        aligned16_A && aligned16_BC &&
        (size_t)nnz * 10 >= (size_t)M * (size_t)N) {
        dim3 block(256);
        dim3 grid(K / 128, M / 128);
        spmm_sgemm_128x128_db_kernel<<<grid, block>>>(A, B, C, M, N, K);
    } else if ((K & 3) == 0 && aligned16_BC) {
        dim3 block(WARPS_PER_BLOCK * 32);
        dim3 grid((K + BN_SP - 1) / BN_SP, (M + BM_SP - 1) / BM_SP);
        spmm_smem_ballot_f4_kernel<<<grid, block>>>(A, B, C, M, N, K);
    } else {
        dim3 block(16, 16);
        dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        spmm_general_kernel<<<grid, block>>>(A, B, C, M, N, K);
    }
}
