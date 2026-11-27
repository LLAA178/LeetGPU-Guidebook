#include <cuda_runtime.h>
#include <stdint.h>

// Optimized SpMM for SM 7.5 (Turing) and general GPUs:
// - Stages a BK x BN (32 x 128) tile of dense matrix B in shared memory using float4 loads,
//   amortizing B traffic across BM = 32 rows of A per block.
// - Each of the 8 warps in a block processes 4 rows of A. Within each warp, all 32 lanes
//   cooperatively load 32 elements of A[row, k_base..k_base+31], use __ballot_sync to find
//   non-zeros, and iterate ONLY over non-zero columns with __ffs + __shfl_sync.
// - Because all 32 lanes in a warp process the same row of A, skipping zeros has zero warp
//   divergence, and each lane accumulates a float4 (4 columns of C) from conflict-free
//   shared memory loads of B.

namespace {

constexpr int WARPS_PER_BLOCK = 8;
constexpr int ROWS_PER_WARP = 4;
constexpr int BM = WARPS_PER_BLOCK * ROWS_PER_WARP;  // 32 rows of A per block
constexpr int BN = 256;                               // 256 cols of B/C per block (64 * float4)
constexpr int BK = 32;                                // 32 K-steps per shared-memory tile

__global__ void spmm_smem_ballot_f4_kernel(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int M, int N, int K) {
    // B_smem holds BK x (BN/4) float4s = 32 x 64 float4s = 32 KB
    __shared__ float4 B_smem[BK][64];

    const int tid = threadIdx.x;        // 0..255
    const int warp_id = tid >> 5;       // 0..7
    const int lane = tid & 31;          // 0..31

    const int block_row = blockIdx.y * BM;
    const int block_col4 = blockIdx.x * 64;  // in float4 units
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

    for (int k_base = 0; k_base < N; k_base += BK) {
        // 1. Cooperatively load BK x 64 float4s of B into shared memory (8 loads per thread)
        #pragma unroll
        for (int s = 0; s < BK; s += WARPS_PER_BLOCK) {
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

        // 2. Each warp processes its 4 rows of A, skipping zero entries via __ballot_sync
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

    // 3. Write back float4 results
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int row = warp_row_base + r;
        if (row < M) {
            if (valid0) C4[(size_t)row * K4 + col4_0] = acc0[r];
            if (valid1) C4[(size_t)row * K4 + col4_1] = acc1[r];
        }
    }
}

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

    // Fast path: all-zero sparse matrix
    if (nnz == 0) {
        cudaMemset(C, 0, (size_t)M * K * sizeof(float));
        return;
    }

    const bool aligned16 =
        ((reinterpret_cast<uintptr_t>(B) & 15u) == 0u) &&
        ((reinterpret_cast<uintptr_t>(C) & 15u) == 0u);

    if ((K & 3) == 0 && aligned16) {
        dim3 block(WARPS_PER_BLOCK * 32);
        dim3 grid((K + BN - 1) / BN, (M + BM - 1) / BM);
        spmm_smem_ballot_f4_kernel<<<grid, block>>>(A, B, C, M, N, K);
    } else {
        dim3 block(16, 16);
        dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        spmm_general_kernel<<<grid, block>>>(A, B, C, M, N, K);
    }
}
