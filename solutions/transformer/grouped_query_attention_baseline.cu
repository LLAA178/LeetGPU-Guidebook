#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreads = kWarpSize * kWarpsPerBlock;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void gqa_baseline_kernel(const float* __restrict__ Q,
                                    const float* __restrict__ K,
                                    const float* __restrict__ V,
                                    float* __restrict__ output,
                                    int num_q_heads,
                                    int num_kv_heads,
                                    int seq_len,
                                    int head_dim,
                                    float scale) {
    const int warp_in_block = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int q_row = blockIdx.x * kWarpsPerBlock + warp_in_block;
    const int q_head = blockIdx.y;
    if (q_row >= seq_len || q_head >= num_q_heads) {
        return;
    }

    const int group_size = num_q_heads / num_kv_heads;
    const int kv_head = q_head / group_size;

    const float* q_ptr = Q + (static_cast<size_t>(q_head) * seq_len + q_row) * head_dim;
    const float* k_base = K + static_cast<size_t>(kv_head) * seq_len * head_dim;
    const float* v_base = V + static_cast<size_t>(kv_head) * seq_len * head_dim;
    float* o_ptr = output + (static_cast<size_t>(q_head) * seq_len + q_row) * head_dim;

    // Each lane holds up to 8 output elements (head_dim <= 256, 256 / 32 = 8).
    float q_reg[8];
    float o_reg[8];
    const int elems_per_lane = (head_dim + kWarpSize - 1) / kWarpSize;
    for (int i = 0; i < elems_per_lane; ++i) {
        const int d = lane + i * kWarpSize;
        q_reg[i] = (d < head_dim) ? q_ptr[d] * scale : 0.0f;
        o_reg[i] = 0.0f;
    }

    float m = -FLT_MAX;
    float l = 0.0f;

    for (int k_row = 0; k_row < seq_len; ++k_row) {
        const float* k_ptr = k_base + static_cast<size_t>(k_row) * head_dim;
        float dot = 0.0f;
        for (int i = 0; i < elems_per_lane; ++i) {
            const int d = lane + i * kWarpSize;
            if (d < head_dim) {
                dot = fmaf(q_reg[i], k_ptr[d], dot);
            }
        }
        float score = warp_reduce_sum(dot);
        score = __shfl_sync(0xffffffff, score, 0);

        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float p = expf(score - m_new);
        l = l * alpha + p;

        const float* v_ptr = v_base + static_cast<size_t>(k_row) * head_dim;
        for (int i = 0; i < elems_per_lane; ++i) {
            const int d = lane + i * kWarpSize;
            if (d < head_dim) {
                o_reg[i] = fmaf(p, v_ptr[d], o_reg[i] * alpha);
            }
        }
        m = m_new;
    }

    float inv_l = 1.0f / l;
    for (int i = 0; i < elems_per_lane; ++i) {
        const int d = lane + i * kWarpSize;
        if (d < head_dim) {
            o_ptr[d] = o_reg[i] * inv_l;
        }
    }
}

}  // namespace

extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int num_q_heads, int num_kv_heads, int seq_len, int head_dim) {
    dim3 block(kThreads);
    dim3 grid((seq_len + kWarpsPerBlock - 1) / kWarpsPerBlock, num_q_heads);
    float scale = rsqrtf(static_cast<float>(head_dim));
    gqa_baseline_kernel<<<grid, block>>>(
        Q, K, V, output, num_q_heads, num_kv_heads, seq_len, head_dim, scale);
}
