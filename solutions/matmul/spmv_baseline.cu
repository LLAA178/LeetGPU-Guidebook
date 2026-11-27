#include <cuda_runtime.h>

__global__ void spmv_baseline_kernel(const float* __restrict__ A,
                                     const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    const float* row_ptr = A + static_cast<size_t>(row) * N;
    float sum = 0.0f;
    for (int col = 0; col < N; ++col) {
        float a = row_ptr[col];
        if (a != 0.0f) {
            sum += a * x[col];
        }
    }
    y[row] = sum;
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    (void)nnz;
    if (M <= 0 || N <= 0) return;

    constexpr int kThreads = 256;
    int blocks = (M + kThreads - 1) / kThreads;
    spmv_baseline_kernel<<<blocks, kThreads>>>(A, x, y, M, N);
}
