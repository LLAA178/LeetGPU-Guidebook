#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

namespace {

constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;
constexpr int kMaxElemsPerLane = 32;  // Supports d up to 1024

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void alibi_attention_baseline_kernel(const float* __restrict__ Q,
                                                const float* __restrict__ K,
                                                const float* __restrict__ V,
                                                float* __restrict__ output,
                                                int M,
                                                int N,
                                                int d,
                                                float scale,
                                                float alibi_alpha) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int q_row = blockIdx.x * kWarpsPerBlock + warp_id;
    if (q_row >= M) return;

    const float* q_ptr = Q + static_cast<size_t>(q_row) * d;
    float q_reg[kMaxElemsPerLane];
    float o_reg[kMaxElemsPerLane];

    const int elems = (d + 31) >> 5;
    for (int i = 0; i < elems; ++i) {
        const int col = lane + i * 32;
        q_reg[i] = (col < d) ? q_ptr[col] * scale : 0.0f;
        o_reg[i] = 0.0f;
    }

    float m = -FLT_MAX;
    float l = 0.0f;

    for (int k_row = 0; k_row < N; ++k_row) {
        const float* k_ptr = K + static_cast<size_t>(k_row) * d;
        float dot = 0.0f;
        for (int i = 0; i < elems; ++i) {
            const int col = lane + i * 32;
            if (col < d) {
                dot = fmaf(q_reg[i], k_ptr[col], dot);
            }
        }
        float score = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0);
        score += alibi_alpha * static_cast<float>(q_row - k_row);

        float m_new = fmaxf(m, score);
        float rescale = __expf(m - m_new);
        float p = __expf(score - m_new);
        l = l * rescale + p;

        const float* v_ptr = V + static_cast<size_t>(k_row) * d;
        for (int i = 0; i < elems; ++i) {
            const int col = lane + i * 32;
            if (col < d) {
                o_reg[i] = fmaf(p, v_ptr[col], o_reg[i] * rescale);
            }
        }
        m = m_new;
    }

    float* o_ptr = output + static_cast<size_t>(q_row) * d;
    const float inv_l = 1.0f / l;
    for (int i = 0; i < elems; ++i) {
        const int col = lane + i * 32;
        if (col < d) {
            o_ptr[col] = o_reg[i] * inv_l;
        }
    }
}

}  // namespace

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d, float alpha) {
    const float scale = rsqrtf(static_cast<float>(d));
    dim3 block(kThreadsPerBlock);
    dim3 grid((M + kWarpsPerBlock - 1) / kWarpsPerBlock);
    alibi_attention_baseline_kernel<<<grid, block>>>(Q, K, V, output, M, N, d, scale, alpha);
}
