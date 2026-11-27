#include <cuda_runtime.h>

// Baseline Sparse Matrix-Dense Matrix Multiplication (SpMM)
// A: M x N (row-major, ~60-70% sparse), B: N x K (row-major), C: M x K (row-major)
__global__ void spmm_baseline_kernel(const float* __restrict__ A,
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

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K, int nnz) {
    (void)nnz;
    if (M <= 0 || K <= 0) return;
    dim3 block(16, 16);
    dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    spmm_baseline_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
