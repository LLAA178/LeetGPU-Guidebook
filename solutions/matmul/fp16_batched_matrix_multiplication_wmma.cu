#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

namespace {

namespace wmma = nvcuda::wmma;

__global__ void warp_bmm(const half* A, const half* B, half* C, int M, int N, int K) {
    // Four independent warps cover a 64 x 64 output tile.
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int row = blockIdx.y * 64 + (warp / 2) * 32;
    const int col = blockIdx.x * 64 + (warp % 2) * 32;
    if (row >= M || col >= N) return;

    const int batch = blockIdx.z;
    A += (batch * M + row) * K;
    B += batch * K * N + col;
    C += (batch * M + row) * N + col;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    }
    for (int k = 0; k < K; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            wmma::load_matrix_sync(a[i], A + i * 16 * K + k, K);
            wmma::load_matrix_sync(b[i], B + k * N + i * 16, N);
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
    }

    __shared__ __align__(32) float tile[4][16][16];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(&tile[warp][0][0], acc[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            #pragma unroll
            for (int p = lane; p < 256; p += 32) {
                C[(i * 16 + p / 16) * N + j * 16 + p % 16] =
                    __float2half_rn(tile[warp][p / 16][p % 16]);
            }
            __syncwarp();
        }
    }
}

__global__ void scalar_bmm(const half* A, const half* B, half* C, int M, int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    const int batch = blockIdx.z;
    A += (batch * M + row) * K;
    B += batch * K * N + col;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum = fmaf(__half2float(A[k]), __half2float(B[k * N]), sum);
    }
    C[(batch * M + row) * N + col] = __float2half_rn(sum);
}

}  // namespace

extern "C" void solve(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
    const uintptr_t inputs = reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B);
    if (M % 32 == 0 && N % 32 == 0 && K % 16 == 0 && inputs % 32 == 0) {
        warp_bmm<<<dim3((N + 63) / 64, (M + 63) / 64, BATCH), dim3(32, 4)>>>(A, B, C, M, N, K);
    } else {
        scalar_bmm<<<dim3((N + 31) / 32, (M + 7) / 8, BATCH), dim3(32, 8)>>>(A, B, C, M, N, K);
    }
}
