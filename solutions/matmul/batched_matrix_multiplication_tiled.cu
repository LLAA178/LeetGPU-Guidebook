#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kFallbackTile = 16;

__global__ void bmm_fallback_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int BATCH,
                                    int M,
                                    int N,
                                    int K) {
    const int b = blockIdx.z;
    const int row = blockIdx.y * kFallbackTile + threadIdx.y;
    const int col = blockIdx.x * kFallbackTile + threadIdx.x;
    if (b >= BATCH || row >= M || col >= N) {
        return;
    }

    const float* a_row = A + (static_cast<size_t>(b) * M + row) * K;
    const float* b_mat = B + static_cast<size_t>(b) * K * N;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc = fmaf(a_row[k], b_mat[static_cast<size_t>(k) * N + col], acc);
    }
    C[(static_cast<size_t>(b) * M + row) * N + col] = acc;
}

// -----------------------------------------------------------------------------
// Double-Buffered Register-Tiled FP32 Batched SGEMM
// BM = 64, BN = 64, BK = 16, 16x16 = 256 threads (each thread computes 4x4)
// -----------------------------------------------------------------------------
constexpr int kBM = 64;
constexpr int kBN = 64;
constexpr int kBK = 16;
constexpr int kSmemPad = 68;  // 64 + 4 floats padding to avoid bank conflicts

__global__ void __launch_bounds__(256, 3)
bmm_tiled_64x64_kernel(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C,
                       int M,
                       int N,
                       int K) {
    __shared__ float sA[2][kBK][kSmemPad];  // transposed: [k][m]
    __shared__ float sB[2][kBK][kSmemPad];  // row-major:  [k][n]

    const int b = blockIdx.z;
    const int block_m = blockIdx.y * kBM;
    const int block_n = blockIdx.x * kBN;

    const int tid = threadIdx.x;
    const int ty = tid >> 4;   // 0..15 -> 4 rows: ty*4 .. ty*4+3
    const int tx = tid & 15;   // 0..15 -> 4 cols: tx*4 .. tx*4+3

    const float* A_b = A + static_cast<size_t>(b) * M * K;
    const float* B_b = B + static_cast<size_t>(b) * K * N;
    float* C_b = C + static_cast<size_t>(b) * M * N;

    // 256 threads load 64x16 A (256 float4s -> 1/thread) and 16x64 B (256 float4s -> 1/thread)
    const int a_r = tid >> 2;        // 0..63
    const int a_c = (tid & 3) * 4;   // 0, 4, 8, 12
    const int b_r = tid >> 4;        // 0..15
    const int b_c = (tid & 15) * 4;  // 0, 4, ..., 60

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // Load stage 0
    float4 reg_a = *reinterpret_cast<const float4*>(
        &A_b[static_cast<size_t>(block_m + a_r) * K + a_c]);
    float4 reg_b = *reinterpret_cast<const float4*>(
        &B_b[static_cast<size_t>(b_r) * N + block_n + b_c]);

    sA[0][a_c + 0][a_r] = reg_a.x;
    sA[0][a_c + 1][a_r] = reg_a.y;
    sA[0][a_c + 2][a_r] = reg_a.z;
    sA[0][a_c + 3][a_r] = reg_a.w;
    *reinterpret_cast<float4*>(&sB[0][b_r][b_c]) = reg_b;
    __syncthreads();

    const int num_tiles = K / kBK;
    for (int t = 0; t < num_tiles; ++t) {
        const int cur = t & 1;
        const int nxt = cur ^ 1;

        if (t + 1 < num_tiles) {
            const int k_next = (t + 1) * kBK;
            reg_a = *reinterpret_cast<const float4*>(
                &A_b[static_cast<size_t>(block_m + a_r) * K + k_next + a_c]);
            reg_b = *reinterpret_cast<const float4*>(
                &B_b[static_cast<size_t>(k_next + b_r) * N + block_n + b_c]);
        }

        #pragma unroll
        for (int k = 0; k < kBK; ++k) {
            float4 a_vec = *reinterpret_cast<const float4*>(&sA[cur][k][ty * 4]);
            float4 b_vec = *reinterpret_cast<const float4*>(&sB[cur][k][tx * 4]);

            acc[0][0] = fmaf(a_vec.x, b_vec.x, acc[0][0]);
            acc[0][1] = fmaf(a_vec.x, b_vec.y, acc[0][1]);
            acc[0][2] = fmaf(a_vec.x, b_vec.z, acc[0][2]);
            acc[0][3] = fmaf(a_vec.x, b_vec.w, acc[0][3]);

            acc[1][0] = fmaf(a_vec.y, b_vec.x, acc[1][0]);
            acc[1][1] = fmaf(a_vec.y, b_vec.y, acc[1][1]);
            acc[1][2] = fmaf(a_vec.y, b_vec.z, acc[1][2]);
            acc[1][3] = fmaf(a_vec.y, b_vec.w, acc[1][3]);

            acc[2][0] = fmaf(a_vec.z, b_vec.x, acc[2][0]);
            acc[2][1] = fmaf(a_vec.z, b_vec.y, acc[2][1]);
            acc[2][2] = fmaf(a_vec.z, b_vec.z, acc[2][2]);
            acc[2][3] = fmaf(a_vec.z, b_vec.w, acc[2][3]);

            acc[3][0] = fmaf(a_vec.w, b_vec.x, acc[3][0]);
            acc[3][1] = fmaf(a_vec.w, b_vec.y, acc[3][1]);
            acc[3][2] = fmaf(a_vec.w, b_vec.z, acc[3][2]);
            acc[3][3] = fmaf(a_vec.w, b_vec.w, acc[3][3]);
        }

        if (t + 1 < num_tiles) {
            sA[nxt][a_c + 0][a_r] = reg_a.x;
            sA[nxt][a_c + 1][a_r] = reg_a.y;
            sA[nxt][a_c + 2][a_r] = reg_a.z;
            sA[nxt][a_c + 3][a_r] = reg_a.w;
            *reinterpret_cast<float4*>(&sB[nxt][b_r][b_c]) = reg_b;
            __syncthreads();
        }
    }

    // Vectorized float4 writeback
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = block_m + ty * 4 + i;
        const int c = block_n + tx * 4;
        *reinterpret_cast<float4*>(&C_b[static_cast<size_t>(r) * N + c]) =
            make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    }
}

}  // namespace

extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    const bool aligned = ((reinterpret_cast<uintptr_t>(A) & 15) == 0) &&
                         ((reinterpret_cast<uintptr_t>(B) & 15) == 0) &&
                         ((reinterpret_cast<uintptr_t>(C) & 15) == 0);
    if (aligned && (M % kBM == 0) && (N % kBN == 0) && (K % kBK == 0)) {
        dim3 block(256);
        dim3 grid(N / kBN, M / kBM, BATCH);
        bmm_tiled_64x64_kernel<<<grid, block>>>(A, B, C, M, N, K);
        return;
    }

    dim3 block(kFallbackTile, kFallbackTile);
    dim3 grid((N + kFallbackTile - 1) / kFallbackTile,
              (M + kFallbackTile - 1) / kFallbackTile,
              BATCH);
    bmm_fallback_kernel<<<grid, block>>>(A, B, C, BATCH, M, N, K);
}
