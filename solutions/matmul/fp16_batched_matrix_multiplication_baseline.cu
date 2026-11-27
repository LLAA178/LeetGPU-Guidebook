#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

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
    const dim3 block(32, 8);
    scalar_bmm<<<dim3((N + 31) / 32, (M + 7) / 8, BATCH), block>>>(A, B, C, M, N, K);
}
