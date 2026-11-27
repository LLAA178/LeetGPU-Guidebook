#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kTile = 16;

__global__ void gemm_baseline_kernel(const half* __restrict__ A,
                                     const half* __restrict__ B,
                                     half* __restrict__ C,
                                     int M,
                                     int N,
                                     int K,
                                     float alpha,
                                     float beta) {
    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }

    float acc = 0.0f;
    const half* a_row = A + static_cast<size_t>(row) * K;
    for (int k = 0; k < K; ++k) {
        acc = fmaf(__half2float(a_row[k]), __half2float(B[static_cast<size_t>(k) * N + col]), acc);
    }

    const size_t c_idx = static_cast<size_t>(row) * N + col;
    float out = alpha * acc;
    if (beta != 0.0f) {
        out = fmaf(beta, __half2float(C[c_idx]), out);
    }
    C[c_idx] = __float2half(out);
}

}  // namespace

extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta) {
    dim3 block(kTile, kTile);
    dim3 grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
    gemm_baseline_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
}
