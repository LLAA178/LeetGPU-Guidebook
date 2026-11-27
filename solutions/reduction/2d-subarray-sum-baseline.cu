#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarp;
constexpr int kMaxBlocks = 1024;
constexpr unsigned kFullMask = 0xffffffffu;

__device__ long long g_partial_sums[kMaxBlocks];

__global__ __launch_bounds__(kThreads) void subarray_sum_partial_baseline_kernel(
    const int* __restrict__ input,
    int m,
    int s_row,
    int s_col,
    int num_rows,
    int col_len) {
    __shared__ long long s_warp_sums[kWarps];

    const size_t total = static_cast<size_t>(num_rows) * static_cast<size_t>(col_len);
    const size_t gtid = static_cast<size_t>(blockIdx.x) * kThreads + threadIdx.x;
    const size_t gstride = static_cast<size_t>(gridDim.x) * kThreads;

    long long local_sum = 0;
    for (size_t idx = gtid; idx < total; idx += gstride) {
        const int r = s_row + static_cast<int>(idx / static_cast<size_t>(col_len));
        const int c = s_col + static_cast<int>(idx % static_cast<size_t>(col_len));
        local_sum += input[static_cast<size_t>(r) * static_cast<size_t>(m) + c];
    }

    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x / kWarp;
    #pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(kFullMask, local_sum, offset);
    }
    if (lane == 0) {
        s_warp_sums[warp] = local_sum;
    }
    __syncthreads();

    if (warp == 0) {
        long long block_sum = (lane < kWarps) ? s_warp_sums[lane] : 0LL;
        #pragma unroll
        for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(kFullMask, block_sum, offset);
        }
        if (lane == 0) {
            g_partial_sums[blockIdx.x] = block_sum;
        }
    }
}

__global__ __launch_bounds__(kThreads) void subarray_sum_final_baseline_kernel(
    int* __restrict__ output,
    int num_blocks) {
    __shared__ long long s_warp_sums[kWarps];

    long long sum = 0;
    for (int i = threadIdx.x; i < num_blocks; i += kThreads) {
        sum += g_partial_sums[i];
    }

    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x / kWarp;
    #pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(kFullMask, sum, offset);
    }
    if (lane == 0) {
        s_warp_sums[warp] = sum;
    }
    __syncthreads();

    if (warp == 0) {
        long long total = (lane < kWarps) ? s_warp_sums[lane] : 0LL;
        #pragma unroll
        for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(kFullMask, total, offset);
        }
        if (lane == 0) {
            output[0] = static_cast<int>(total);
        }
    }
}

}  // namespace

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int S_ROW, int E_ROW, int S_COL,
                      int E_COL) {
    (void)N;
    const int num_rows = E_ROW - S_ROW + 1;
    const int col_len = E_COL - S_COL + 1;
    if (num_rows <= 0 || col_len <= 0) {
        return;
    }

    const size_t total = static_cast<size_t>(num_rows) * static_cast<size_t>(col_len);
    const size_t raw_blocks = (total + kThreads - 1) / kThreads;
    const int num_blocks =
        (raw_blocks > static_cast<size_t>(kMaxBlocks)) ? kMaxBlocks : static_cast<int>(raw_blocks);

    subarray_sum_partial_baseline_kernel<<<static_cast<unsigned>(num_blocks), kThreads>>>(
        input, M, S_ROW, S_COL, num_rows, col_len);
    subarray_sum_final_baseline_kernel<<<1, kThreads>>>(output, num_blocks);
}
