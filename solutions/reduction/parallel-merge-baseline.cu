#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ int merge_path_corank(const float* __restrict__ A,
                                                 int M,
                                                 const float* __restrict__ B,
                                                 int N,
                                                 int diag) {
    int low = (diag > N) ? (diag - N) : 0;
    int high = (diag < M) ? diag : M;

    while (low < high) {
        const int mid = low + ((high - low) >> 1);
        const int j = diag - 1 - mid;
        if (A[mid] <= B[j]) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

__global__ void parallel_merge_baseline_kernel(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int M,
                                               int N) {
    const int total = M + N;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= total) {
        return;
    }

    const int i = merge_path_corank(A, M, B, N, k);
    const int j = k - i;
    const bool take_a = (j >= N) || (i < M && A[i] <= B[j]);
    C[k] = take_a ? A[i] : B[j];
}

}  // namespace

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    const int total = M + N;
    if (total <= 0) {
        return;
    }
    const int blocks = (total + kThreads - 1) / kThreads;
    parallel_merge_baseline_kernel<<<blocks, kThreads>>>(A, B, C, M, N);
}
