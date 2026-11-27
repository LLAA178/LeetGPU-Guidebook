#include <cuda_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

// Fast path: N % 4 == 0 and 16-byte aligned A, x.
// 1 warp (32 threads) processes 2 rows simultaneously, reusing x4 in registers.
__global__ void spmv_vec4_2rows_kernel(const float4* __restrict__ A4,
                                       const float4* __restrict__ x4,
                                       float* __restrict__ y,
                                       int M, int N4) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int row0 = warp_id << 1;
    if (row0 >= M) return;

    int row1 = row0 + 1;
    bool has_row1 = (row1 < M);

    const float4* r0 = A4 + static_cast<size_t>(row0) * N4;
    const float4* r1 = has_row1 ? (A4 + static_cast<size_t>(row1) * N4) : r0;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (int col4 = lane; col4 < N4; col4 += 32) {
        float4 xv = __ldg(x4 + col4);
        float4 a0 = __ldg(r0 + col4);
        sum0 = fmaf(a0.x, xv.x, sum0);
        sum0 = fmaf(a0.y, xv.y, sum0);
        sum0 = fmaf(a0.z, xv.z, sum0);
        sum0 = fmaf(a0.w, xv.w, sum0);

        if (has_row1) {
            float4 a1 = __ldg(r1 + col4);
            sum1 = fmaf(a1.x, xv.x, sum1);
            sum1 = fmaf(a1.y, xv.y, sum1);
            sum1 = fmaf(a1.z, xv.z, sum1);
            sum1 = fmaf(a1.w, xv.w, sum1);
        }
    }

    sum0 = warp_reduce_sum(sum0);
    if (lane == 0) {
        y[row0] = sum0;
    }

    if (has_row1) {
        sum1 = warp_reduce_sum(sum1);
        if (lane == 0) {
            y[row1] = sum1;
        }
    }
}

// General path: any M, N and alignment. 1 warp per row.
__global__ void spmv_scalar_warp_kernel(const float* __restrict__ A,
                                        const float* __restrict__ x,
                                        float* __restrict__ y,
                                        int M, int N) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (row >= M) return;

    const float* row_ptr = A + static_cast<size_t>(row) * N;
    float sum = 0.0f;
    for (int col = lane; col < N; col += 32) {
        sum = fmaf(__ldg(row_ptr + col), __ldg(x + col), sum);
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        y[row] = sum;
    }
}

}  // namespace

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    (void)nnz;
    if (M <= 0 || N <= 0) return;

    const bool aligned16 =
        ((N & 3) == 0) &&
        ((reinterpret_cast<uintptr_t>(A) & 15u) == 0u) &&
        ((reinterpret_cast<uintptr_t>(x) & 15u) == 0u);

    if (aligned16) {
        constexpr int kWarpsPerBlock = 4;
        constexpr int kThreads = kWarpsPerBlock * 32;
        int total_warps = (M + 1) >> 1;
        int blocks = (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
        spmv_vec4_2rows_kernel<<<blocks, kThreads>>>(
            reinterpret_cast<const float4*>(A),
            reinterpret_cast<const float4*>(x),
            y,
            M,
            N >> 2);
    } else {
        constexpr int kWarpsPerBlock = 4;
        constexpr int kThreads = kWarpsPerBlock * 32;
        int blocks = (M + kWarpsPerBlock - 1) / kWarpsPerBlock;
        spmv_scalar_warp_kernel<<<blocks, kThreads>>>(A, x, y, M, N);
    }
}
