#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

namespace {

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Baseline: 1 warp per (m, h) query row, Online Softmax over N keys.
// Supports all valid constraints: 1 <= M, N <= 4096, 1 <= H <= 64, 1 <= D <= 256.
__global__ void cross_attention_baseline_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int M, int N, int H, int D, float scale)
{
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int m = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    int h = blockIdx.y;

    if (m >= M || h >= H) return;

    // Max D is 256, so each lane holds at most 256 / 32 = 8 elements.
    float q_reg[8];
    float o_reg[8];
    int num_elems = (D + WARP_SIZE - 1) / WARP_SIZE;

    size_t q_offset = ((size_t)m * H + h) * D;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int d = lane + i * WARP_SIZE;
        q_reg[i] = (d < D) ? Q[q_offset + d] : 0.0f;
        o_reg[i] = 0.0f;
    }

    float m_val = -FLT_MAX;
    float l_val = 0.0f;

    for (int n = 0; n < N; ++n) {
        size_t kv_offset = ((size_t)n * H + h) * D;
        float dot = 0.0f;
        for (int i = 0; i < num_elems; ++i) {
            int d = lane + i * WARP_SIZE;
            if (d < D) {
                dot += q_reg[i] * K[kv_offset + d];
            }
        }
        float s = warp_reduce_sum(dot) * scale;

        float m_new = fmaxf(m_val, s);
        float alpha = expf(m_val - m_new);
        float p = expf(s - m_new);
        l_val = l_val * alpha + p;

        for (int i = 0; i < num_elems; ++i) {
            int d = lane + i * WARP_SIZE;
            if (d < D) {
                o_reg[i] = o_reg[i] * alpha + p * V[kv_offset + d];
            }
        }
        m_val = m_new;
    }

    float inv_l = 1.0f / l_val;
    for (int i = 0; i < num_elems; ++i) {
        int d = lane + i * WARP_SIZE;
        if (d < D) {
            output[q_offset + d] = o_reg[i] * inv_l;
        }
    }
}

}  // namespace

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int H, int D) {
    if (M <= 0 || N <= 0 || H <= 0 || D <= 0) return;

    float scale = 1.0f / sqrtf(static_cast<float>(D));
    dim3 block(WARPS_PER_BLOCK * WARP_SIZE);
    dim3 grid((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, H);
    cross_attention_baseline_kernel<<<grid, block>>>(Q, K, V, output, M, N, H, D, scale);
}
