#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 1024;

__device__ long long g_baseline_partials[kMaxBlocks];

__global__ void subarray_sum_3d_stage1_kernel(const int* __restrict__ input,
                                              int M,
                                              int K,
                                              int S_DEP,
                                              int S_ROW,
                                              int S_COL,
                                              int R_len,
                                              int C_len,
                                              long long total_elems) {
    __shared__ long long s_sum[kThreads];

    const int tid = threadIdx.x;
    const long long global_tid = static_cast<long long>(blockIdx.x) * kThreads + tid;
    const long long stride = static_cast<long long>(gridDim.x) * kThreads;
    const long long plane_elems = static_cast<long long>(R_len) * C_len;

    long long acc = 0;
    for (long long idx = global_tid; idx < total_elems; idx += stride) {
        const int d_off = static_cast<int>(idx / plane_elems);
        const int rem = static_cast<int>(idx - static_cast<long long>(d_off) * plane_elems);
        const int r_off = rem / C_len;
        const int c_off = rem - r_off * C_len;

        const long long d = S_DEP + d_off;
        const long long r = S_ROW + r_off;
        const long long c = S_COL + c_off;
        acc += static_cast<long long>(input[(d * M + r) * K + c]);
    }

    s_sum[tid] = acc;
    __syncthreads();

    for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum[tid] += s_sum[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_baseline_partials[blockIdx.x] = s_sum[0];
    }
}

__global__ void subarray_sum_3d_stage2_kernel(int* __restrict__ output, int num_blocks) {
    __shared__ long long s_sum[kThreads];

    const int tid = threadIdx.x;
    long long acc = 0;
    for (int i = tid; i < num_blocks; i += kThreads) {
        acc += g_baseline_partials[i];
    }
    s_sum[tid] = acc;
    __syncthreads();

    for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum[tid] += s_sum[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = static_cast<int>(s_sum[0]);
    }
}

}  // namespace

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input,
                      int* output,
                      int N,
                      int M,
                      int K,
                      int S_DEP,
                      int E_DEP,
                      int S_ROW,
                      int E_ROW,
                      int S_COL,
                      int E_COL) {
    if (N <= 0 || M <= 0 || K <= 0 ||
        S_DEP < 0 || E_DEP < S_DEP || E_DEP >= N ||
        S_ROW < 0 || E_ROW < S_ROW || E_ROW >= M ||
        S_COL < 0 || E_COL < S_COL || E_COL >= K) {
        return;
    }

    const int D_len = E_DEP - S_DEP + 1;
    const int R_len = E_ROW - S_ROW + 1;
    const int C_len = E_COL - S_COL + 1;
    const long long total_elems =
        static_cast<long long>(D_len) * static_cast<long long>(R_len) * C_len;

    long long blocks_ll = (total_elems + kThreads - 1) / kThreads;
    if (blocks_ll < 1) blocks_ll = 1;
    if (blocks_ll > kMaxBlocks) blocks_ll = kMaxBlocks;
    const int blocks = static_cast<int>(blocks_ll);

    subarray_sum_3d_stage1_kernel<<<blocks, kThreads>>>(
        input, M, K, S_DEP, S_ROW, S_COL, R_len, C_len, total_elems);
    subarray_sum_3d_stage2_kernel<<<1, kThreads>>>(output, blocks);
}
