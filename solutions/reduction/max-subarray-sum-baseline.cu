#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
constexpr int kThreads = 256;
constexpr int kMaxBlocks = 512;
constexpr unsigned kFullMask = 0xffffffffu;

__device__ int g_baseline_block_max[kMaxBlocks];

__device__ __forceinline__ int warp_reduce_max(int val) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        const int other = __shfl_down_sync(kFullMask, val, offset);
        if (other > val) {
            val = other;
        }
    }
    return val;
}

__global__ __launch_bounds__(kThreads) void max_subarray_baseline_window_kernel(
    const int* __restrict__ input,
    int* __restrict__ output,
    int num_windows,
    int window_size) {
    constexpr int kWarps = kThreads / kWarpSize;
    __shared__ int s_warp_max[kWarps];

    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid >> 5;
    const int global_tid = blockIdx.x * kThreads + tid;
    const int stride = gridDim.x * kThreads;

    int local_max = INT_MIN;
    for (int s = global_tid; s < num_windows; s += stride) {
        long long sum = 0;
        for (int k = 0; k < window_size; ++k) {
            sum += static_cast<long long>(input[s + k]);
        }
        const int win_sum = static_cast<int>(sum);
        if (win_sum > local_max) {
            local_max = win_sum;
        }
    }

    local_max = warp_reduce_max(local_max);
    if (lane == 0) {
        s_warp_max[warp] = local_max;
    }
    __syncthreads();

    if (warp == 0) {
        int bmax = (lane < kWarps) ? s_warp_max[lane] : INT_MIN;
        bmax = warp_reduce_max(bmax);
        if (lane == 0) {
            if (gridDim.x == 1) {
                output[0] = bmax;
            } else {
                g_baseline_block_max[blockIdx.x] = bmax;
            }
        }
    }
}

__global__ __launch_bounds__(kThreads) void max_subarray_baseline_reduce_kernel(
    int* __restrict__ output,
    int num_blocks) {
    constexpr int kWarps = kThreads / kWarpSize;
    __shared__ int s_warp_max[kWarps];

    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid >> 5;

    int local_max = INT_MIN;
    for (int i = tid; i < num_blocks; i += kThreads) {
        const int v = g_baseline_block_max[i];
        if (v > local_max) {
            local_max = v;
        }
    }

    local_max = warp_reduce_max(local_max);
    if (lane == 0) {
        s_warp_max[warp] = local_max;
    }
    __syncthreads();

    if (warp == 0) {
        int bmax = (lane < kWarps) ? s_warp_max[lane] : INT_MIN;
        bmax = warp_reduce_max(bmax);
        if (lane == 0) {
            output[0] = bmax;
        }
    }
}

}  // namespace

extern "C" void solve(const int* input, int* output, int N, int window_size) {
    if (N <= 0 || window_size <= 0 || window_size > N) {
        return;
    }
    const int num_windows = N - window_size + 1;
    int num_blocks = (num_windows + kThreads - 1) / kThreads;
    if (num_blocks > kMaxBlocks) {
        num_blocks = kMaxBlocks;
    }
    max_subarray_baseline_window_kernel<<<num_blocks, kThreads>>>(
        input, output, num_windows, window_size);
    if (num_blocks > 1) {
        max_subarray_baseline_reduce_kernel<<<1, kThreads>>>(output, num_blocks);
    }
}
