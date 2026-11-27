#include <cuda_runtime.h>

namespace {

constexpr int kTile = 16;

__global__ void bmm_baseline_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int BATCH,
                                    int M,
                                    int N,
                                    int K) {
    const int b = blockIdx.z;
    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;
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

}  // namespace

extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    dim3 block(kTile, kTile);
    dim3 grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile, BATCH);
    bmm_baseline_kernel<<<grid, block>>>(A, B, C, BATCH, M, N, K);
}
