#include <cuda_runtime.h>

__global__ void grouped_gemm_baseline_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const int* __restrict__ group_offsets,
    float* __restrict__ C,
    int G,
    int M_total,
    int K,
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M_total || col >= N) return;

    // Binary search for group g such that group_offsets[g] <= row < group_offsets[g + 1]
    int lo = 0, hi = G;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if (group_offsets[mid] <= row) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    int g = lo;

    const float* a_row = A + (size_t)row * K;
    const float* b_mat = B + (size_t)g * K * N + col;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += a_row[k] * b_mat[(size_t)k * N];
    }
    C[(size_t)row * N + col] = sum;
}

extern "C" void solve(const float* A, const float* B, const int* group_offsets, float* C, int G,
                      int M_total, int K, int N) {
    if (G <= 0 || M_total <= 0 || K <= 0 || N <= 0) return;

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M_total + block.y - 1) / block.y);
    grouped_gemm_baseline_kernel<<<grid, block>>>(A, B, group_offsets, C, G, M_total, K, N);
}
