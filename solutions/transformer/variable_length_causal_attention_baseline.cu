#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreads = kWarpSize * kWarpsPerBlock;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

__device__ __forceinline__ int find_seq_start(const int* __restrict__ cu_seqlens,
                                              int S,
                                              int token_idx) {
    int lo = 0;
    int hi = S;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (cu_seqlens[mid] <= token_idx) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return cu_seqlens[lo];
}

__global__ void variable_length_causal_attention_baseline_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const int* __restrict__ cu_seqlens,
    float* __restrict__ output,
    int T,
    int d,
    int S,
    float scale) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    if (warp_id >= T) return;

    const int q_row = warp_id;
    int seq_start = 0;
    if (lane == 0) {
        seq_start = find_seq_start(cu_seqlens, S, q_row);
    }
    seq_start = __shfl_sync(0xffffffff, seq_start, 0);

    const int elems = (d + kWarpSize - 1) / kWarpSize;
    float q_reg[8];
    float o_reg[8];
    const float* q_ptr = Q + static_cast<size_t>(q_row) * d;
    for (int i = 0; i < elems; ++i) {
        const int col = lane + i * kWarpSize;
        q_reg[i] = (col < d) ? q_ptr[col] * scale : 0.0f;
        o_reg[i] = 0.0f;
    }

    float m = -FLT_MAX;
    float l = 0.0f;

    for (int k_row = seq_start; k_row <= q_row; ++k_row) {
        const float* k_ptr = K + static_cast<size_t>(k_row) * d;
        float dot = 0.0f;
        for (int i = 0; i < elems; ++i) {
            const int col = lane + i * kWarpSize;
            if (col < d) {
                dot = fmaf(q_reg[i], k_ptr[col], dot);
            }
        }
        const float score = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0);
        const float m_new = fmaxf(m, score);
        const float alpha = __expf(m - m_new);
        const float p = __expf(score - m_new);
        l = l * alpha + p;

        const float* v_ptr = V + static_cast<size_t>(k_row) * d;
        for (int i = 0; i < elems; ++i) {
            const int col = lane + i * kWarpSize;
            if (col < d) {
                o_reg[i] = fmaf(p, v_ptr[col], o_reg[i] * alpha);
            }
        }
        m = m_new;
    }

    const float inv_l = 1.0f / l;
    float* out_ptr = output + static_cast<size_t>(q_row) * d;
    for (int i = 0; i < elems; ++i) {
        const int col = lane + i * kWarpSize;
        if (col < d) {
            out_ptr[col] = o_reg[i] * inv_l;
        }
    }
}

}  // namespace

// Q, K, V, cu_seqlens, output are device pointers
extern "C" void solve(const float* Q,
                      const float* K,
                      const float* V,
                      const int* cu_seqlens,
                      float* output,
                      int T,
                      int d,
                      int S) {
    if (T <= 0 || d <= 0 || S <= 0) return;
    const float scale = rsqrtf(static_cast<float>(d));
    const int blocks = (T + kWarpsPerBlock - 1) / kWarpsPerBlock;
    variable_length_causal_attention_baseline_kernel<<<blocks, kThreads>>>(
        Q, K, V, cu_seqlens, output, T, d, S, scale);
}
