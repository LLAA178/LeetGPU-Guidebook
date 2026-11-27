#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void decaying_causal_attention_baseline_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int seq_len,
    int d_model,
    float gamma,
    float inv_sqrt_d) {
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int n = blockIdx.x * kWarpsPerBlock + warp_id;
    if (n >= seq_len) {
        return;
    }

    const int elems = (d_model + kWarpSize - 1) / kWarpSize;
    float q_reg[8];
    float o_reg[8];

    const float* q_ptr = Q + static_cast<size_t>(n) * d_model;
    for (int i = 0; i < elems; ++i) {
        const int col = lane + i * kWarpSize;
        q_reg[i] = (col < d_model) ? q_ptr[col] * inv_sqrt_d : 0.0f;
        o_reg[i] = 0.0f;
    }

    float decay = 1.0f;
    for (int m = n; m >= 0; --m) {
        const float* k_ptr = K + static_cast<size_t>(m) * d_model;
        const float* v_ptr = V + static_cast<size_t>(m) * d_model;

        float dot = 0.0f;
        for (int i = 0; i < elems; ++i) {
            const int col = lane + i * kWarpSize;
            if (col < d_model) {
                dot = fmaf(q_reg[i], k_ptr[col], dot);
            }
        }
        float score = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0);
        float weight = score * decay;

        for (int i = 0; i < elems; ++i) {
            const int col = lane + i * kWarpSize;
            if (col < d_model) {
                o_reg[i] = fmaf(weight, v_ptr[col], o_reg[i]);
            }
        }
        decay *= gamma;
    }

    float* out_ptr = output + static_cast<size_t>(n) * d_model;
    for (int i = 0; i < elems; ++i) {
        const int col = lane + i * kWarpSize;
        if (col < d_model) {
            out_ptr[col] = o_reg[i];
        }
    }
}

}  // namespace

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int seq_len,
                      int d_model, float gamma) {
    if (seq_len <= 0 || d_model <= 0) {
        return;
    }
    const float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(d_model));
    const int blocks = (seq_len + kWarpsPerBlock - 1) / kWarpsPerBlock;
    decaying_causal_attention_baseline_kernel<<<blocks, kThreadsPerBlock>>>(
        Q, K, V, output, seq_len, d_model, gamma, inv_sqrt_d);
}
