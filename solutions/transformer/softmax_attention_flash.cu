#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 2;
constexpr int kRowsPerWarp = 4;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Fast 4-row-per-warp Online Softmax kernel:
// Reuses every loaded K[r] and V[r] across 4 query rows in registers (4x less K/V traffic)
// with zero shared memory for maximum occupancy and L1/L2 cache residency.
template <int kElems>
__global__ void __launch_bounds__(64, 8)
softmax_attention_4row_kernel(const float* __restrict__ Q,
                              const float* __restrict__ K,
                              const float* __restrict__ V,
                              float* __restrict__ output,
                              int M,
                              int N,
                              int d,
                              float scale) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int q_base_row = (blockIdx.x * kWarpsPerBlock + warp_id) * kRowsPerWarp;
    if (q_base_row >= M) {
        return;
    }

    float q_reg[kRowsPerWarp][kElems];
    float o_reg[kRowsPerWarp][kElems];
    float m[kRowsPerWarp];
    float l[kRowsPerWarp];

    #pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        const int qr = q_base_row + r;
        const bool valid = (qr < M);
        const float* q_ptr = Q + static_cast<size_t>(qr) * d;
        #pragma unroll
        for (int i = 0; i < kElems; ++i) {
            const int col = lane + i * kWarpSize;
            q_reg[r][i] = (valid && col < d) ? q_ptr[col] * scale : 0.0f;
            o_reg[r][i] = 0.0f;
        }
        m[r] = -FLT_MAX;
        l[r] = 0.0f;
    }

    for (int k_row = 0; k_row < N; ++k_row) {
        const float* k_ptr = K + static_cast<size_t>(k_row) * d;
        const float* v_ptr = V + static_cast<size_t>(k_row) * d;

        float k_val[kElems];
        float v_val[kElems];
        #pragma unroll
        for (int i = 0; i < kElems; ++i) {
            const int col = lane + i * kWarpSize;
            k_val[i] = (col < d) ? __ldg(&k_ptr[col]) : 0.0f;
            v_val[i] = (col < d) ? __ldg(&v_ptr[col]) : 0.0f;
        }

        float dot[kRowsPerWarp];
        #pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            float acc = 0.0f;
            #pragma unroll
            for (int i = 0; i < kElems; ++i) {
                acc = fmaf(q_reg[r][i], k_val[i], acc);
            }
            dot[r] = __shfl_sync(0xffffffff, warp_reduce_sum(acc), 0);
        }

        #pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            float m_new = fmaxf(m[r], dot[r]);
            float alpha = __expf(m[r] - m_new);
            float p = __expf(dot[r] - m_new);
            l[r] = l[r] * alpha + p;
            #pragma unroll
            for (int i = 0; i < kElems; ++i) {
                o_reg[r][i] = fmaf(p, v_val[i], o_reg[r][i] * alpha);
            }
            m[r] = m_new;
        }
    }

    #pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        const int qr = q_base_row + r;
        if (qr < M) {
            float* o_ptr = output + static_cast<size_t>(qr) * d;
            const float inv_l = 1.0f / l[r];
            #pragma unroll
            for (int i = 0; i < kElems; ++i) {
                const int col = lane + i * kWarpSize;
                if (col < d) {
                    o_ptr[col] = o_reg[r][i] * inv_l;
                }
            }
        }
    }
}

}  // namespace

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d) {
    dim3 block(kWarpsPerBlock * kWarpSize);
    dim3 grid((M + kWarpsPerBlock * kRowsPerWarp - 1) / (kWarpsPerBlock * kRowsPerWarp));
    const float scale = rsqrtf(static_cast<float>(d));

    if (d <= 32) {
        softmax_attention_4row_kernel<1><<<grid, block>>>(Q, K, V, output, M, N, d, scale);
    } else if (d <= 64) {
        softmax_attention_4row_kernel<2><<<grid, block>>>(Q, K, V, output, M, N, d, scale);
    } else if (d <= 96) {
        softmax_attention_4row_kernel<3><<<grid, block>>>(Q, K, V, output, M, N, d, scale);
    } else {
        softmax_attention_4row_kernel<4><<<grid, block>>>(Q, K, V, output, M, N, d, scale);
    }
}
