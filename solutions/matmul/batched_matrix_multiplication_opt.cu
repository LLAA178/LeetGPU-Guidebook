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
// Double-Buffered 8x8 Register-Tiled FP32 Batched SGEMM
// BM = 128, BN = 128, BK = 8, 16x16 = 256 threads (each thread computes 8x8)
// Warp-tiled thread mapping (4x8 warp layout) for coalesced LDS & 0 bank conflicts
// -----------------------------------------------------------------------------
constexpr int kBM = 128;
constexpr int kBN = 128;
constexpr int kBK = 8;
constexpr int kSmemPad = 132;  // 128 + 4 floats padding

__global__ void __launch_bounds__(256, 2)
bmm_sgemm_128x128_kernel(const float* __restrict__ A,
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
    const int ty = tid >> 4;  // 0..15 -> 2x4 rows: ty*4 and ty*4 + 64
    const int tx = tid & 15;  // 0..15 -> 2x4 cols: tx*4 and tx*4 + 64

    const float* A_b = A + static_cast<size_t>(b) * M * K;
    const float* B_b = B + static_cast<size_t>(b) * K * N;
    float* C_b = C + static_cast<size_t>(b) * M * N;

    // 256 threads load 128x8 A (256 float4s -> 1/thread) and 8x128 B (256 float4s -> 1/thread)
    const int a_r = tid >> 1;         // 0..127
    const int a_c = (tid & 1) * 4;    // 0, 4
    const int b_r = tid >> 5;         // 0..7
    const int b_c = (tid & 31) * 4;   // 0, 4, ..., 124

    float acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
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
            float4 a0 = *reinterpret_cast<const float4*>(&sA[cur][k][ty * 4]);
            float4 a1 = *reinterpret_cast<const float4*>(&sA[cur][k][ty * 4 + 64]);
            float4 b0 = *reinterpret_cast<const float4*>(&sB[cur][k][tx * 4]);
            float4 b1 = *reinterpret_cast<const float4*>(&sB[cur][k][tx * 4 + 64]);

            float a_vals[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            float b_vals[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    acc[i][j] = fmaf(a_vals[i], b_vals[j], acc[i][j]);
                }
            }
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

    // Vectorized float4 writeback (2x2 quadrants of 4x4)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r0 = block_m + ty * 4 + i;
        const int r1 = r0 + 64;
        const int c0 = block_n + tx * 4;
        const int c1 = c0 + 64;
        *reinterpret_cast<float4*>(&C_b[static_cast<size_t>(r0) * N + c0]) =
            make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        *reinterpret_cast<float4*>(&C_b[static_cast<size_t>(r0) * N + c1]) =
            make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
        *reinterpret_cast<float4*>(&C_b[static_cast<size_t>(r1) * N + c0]) =
            make_float4(acc[i + 4][0], acc[i + 4][1], acc[i + 4][2], acc[i + 4][3]);
        *reinterpret_cast<float4*>(&C_b[static_cast<size_t>(r1) * N + c1]) =
            make_float4(acc[i + 4][4], acc[i + 4][5], acc[i + 4][6], acc[i + 4][7]);
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
        bmm_sgemm_128x128_kernel<<<grid, block>>>(A, B, C, M, N, K);
        return;
    }

    dim3 block(kFallbackTile, kFallbackTile);
    dim3 grid((N + kFallbackTile - 1) / kFallbackTile,
              (M + kFallbackTile - 1) / kFallbackTile,
              BATCH);
    bmm_fallback_kernel<<<grid, block>>>(A, B, C, BATCH, M, N, K);
}
