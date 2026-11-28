#include <cuda_runtime.h>
#include <cstddef>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 1024;

__device__ long long g_base_partials[kMaxBlocks];

__global__ __launch_bounds__(kThreads) void count_2d_stage1_kernel(const int* __restrict__ input,
                                                                   size_t total_elems,
                                                                   int K) {
    __shared__ long long s_cnt[kThreads];

    const int tid = threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * kThreads;

    long long cnt = 0;
    for (size_t i = static_cast<size_t>(blockIdx.x) * kThreads + static_cast<size_t>(tid);
         i < total_elems;
         i += stride) {
        if (input[i] == K) {
            ++cnt;
        }
    }

    s_cnt[tid] = cnt;
    __syncthreads();

    for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_cnt[tid] += s_cnt[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_base_partials[blockIdx.x] = s_cnt[0];
    }
}

__global__ __launch_bounds__(kThreads) void count_2d_stage2_kernel(int* __restrict__ output,
                                                                   int num_blocks) {
    __shared__ long long s_cnt[kThreads];

    const int tid = threadIdx.x;
    long long cnt = 0;
    for (int i = tid; i < num_blocks; i += kThreads) {
        cnt += g_base_partials[i];
    }

    s_cnt[tid] = cnt;
    __syncthreads();

    for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_cnt[tid] += s_cnt[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = static_cast<int>(s_cnt[0]);
    }
}

}  // namespace

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    if (N <= 0 || M <= 0) {
        return;
    }

    const size_t total_elems = static_cast<size_t>(N) * static_cast<size_t>(M);
    size_t blocks_sz = (total_elems + kThreads - 1) / kThreads;
    if (blocks_sz < 1ull) blocks_sz = 1ull;
    if (blocks_sz > static_cast<size_t>(kMaxBlocks)) blocks_sz = static_cast<size_t>(kMaxBlocks);
    const int blocks = static_cast<int>(blocks_sz);

    count_2d_stage1_kernel<<<blocks, kThreads>>>(input, total_elems, K);
    count_2d_stage2_kernel<<<1, kThreads>>>(output, blocks);
}
