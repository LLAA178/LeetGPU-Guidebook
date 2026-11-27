#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void softmax_attention_baseline_kernel(const float* __restrict__ Q,
                                                  const float* __restrict__ K,
                                                  const float* __restrict__ V,
                                                  float* __restrict__ output,
                                                  int M,
                                                  int N,
                                                  int d,
                                                  float scale) {
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int q_row = blockIdx.x * kWarpsPerBlock + warp_id;
    if (q_row >= M) {
        return;
    }

    const float* q_ptr = Q + static_cast<size_t>(q_row) * d;
    float q_reg[4];
    float o_reg[4];
    const int elems = (d + kWarpSize - 1) / kWarpSize;  // d <= 128 -> elems <= 4
    for (int i = 0; i < elems; ++i) {
        const int col = lane + i * kWarpSize;
        q_reg[i] = (col < d) ? q_ptr[col] * scale : 0.0f;
        o_reg[i] = 0.0f;
    }

    float m = -FLT_MAX;
    float l = 0.0f;

    for (int k_row = 0; k_row < N; ++k_row) {
        const float* k_ptr = K + static_cast<size_t>(k_row) * d;
        float dot = 0.0f;
        for (int i = 0; i < elems; ++i) {
            const int col = lane + i * kWarpSize;
            if (col < d) {
                dot = fmaf(q_reg[i], k_ptr[col], dot);
            }
        }
        float score = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0);

        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float p = expf(score - m_new);
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

    float* o_ptr = output + static_cast<size_t>(q_row) * d;
    const float inv_l = 1.0f / l;
    for (int i = 0; i < elems; ++i) {
        const int col = lane + i * kWarpSize;
        if (col < d) {
            o_ptr[col] = o_reg[i] * inv_l;
        }
    }
}

}  // namespace

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d) {
    dim3 block(kWarpsPerBlock * kWarpSize);
    dim3 grid((M + kWarpsPerBlock - 1) / kWarpsPerBlock);
    float scale = rsqrtf(static_cast<float>(d));
    softmax_attention_baseline_kernel<<<grid, block>>>(Q, K, V, output, M, N, d, scale);
}
