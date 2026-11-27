#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

namespace {

constexpr int kBr = 8;
constexpr int kBc = 32;
constexpr int kThreads = 128;  // 8x16: ty in [0..7] (1 Q row), tx in [0..15] (2 K cols / float4 O cols)
constexpr int kMaxElemsPerLane = 32;  // Fallback supports d up to 1024

__device__ __forceinline__ float subwarp16_max(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
    return val;
}

__device__ __forceinline__ float subwarp16_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 1);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    return val;
}

// High-occupancy 8x16 (128-thread) float4-vectorized FlashAttention (Br=8, Bc=32) with ALiBi bias.
template <int kVecCols>
__global__ void __launch_bounds__(128, 2)
alibi_attention_8x16_kernel(const float* __restrict__ Q,
                            const float* __restrict__ K,
                            const float* __restrict__ V,
                            float* __restrict__ output,
                            int M,
                            int N,
                            int d,
                            float scale,
                            float alibi_alpha) {
    extern __shared__ float smem[];
    const int d_stride = d + 4;
    float* sQ = smem;
    float* sKV = sQ + kBr * d_stride;
    float* sP = sKV + kBc * d_stride;  // 8 * 36

    const int tid = threadIdx.x;
    const int ty = tid >> 4;  // 0..7  -> 1 Q row
    const int tx = tid & 15;  // 0..15 -> 2 K cols / float4 O chunks
    const int q_block_row = blockIdx.x * kBr;
    const int q_len = min(kBr, M - q_block_row);
    const int vec_d = d >> 2;

    for (int idx = tid; idx < q_len * vec_d; idx += kThreads) {
        const int r = idx / vec_d;
        const int c4 = idx % vec_d;
        float4 v = *reinterpret_cast<const float4*>(&Q[static_cast<size_t>(q_block_row + r) * d + c4 * 4]);
        *reinterpret_cast<float4*>(&sQ[r * d_stride + c4 * 4]) =
            make_float4(v.x * scale, v.y * scale, v.z * scale, v.w * scale);
    }
    __syncthreads();

    float4 o_reg[kVecCols];
    #pragma unroll
    for (int c = 0; c < kVecCols; ++c) {
        o_reg[c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float m = -FLT_MAX;
    float l = 0.0f;
    const int q_global_row = q_block_row + ty;
    const bool valid_q = (ty < q_len);

    for (int k_start = 0; k_start < N; k_start += kBc) {
        const int kv_len = min(kBc, N - k_start);

        for (int idx = tid; idx < kv_len * vec_d; idx += kThreads) {
            const int r = idx / vec_d;
            const int c4 = idx % vec_d;
            *reinterpret_cast<float4*>(&sKV[r * d_stride + c4 * 4]) =
                *reinterpret_cast<const float4*>(&K[static_cast<size_t>(k_start + r) * d + c4 * 4]);
        }
        __syncthreads();

        // 1. Q @ K^T + ALiBi bias (1x2 per thread, float4 vectorized)
        float s0 = 0.0f, s1 = 0.0f;
        const int k0 = tx * 2;
        const int k1 = k0 + 1;
        if (valid_q) {
            const float* q_row = &sQ[ty * d_stride];
            for (int k = 0; k < d; k += 4) {
                float4 qv = *reinterpret_cast<const float4*>(&q_row[k]);
                if (k0 < kv_len) {
                    float4 kv = *reinterpret_cast<const float4*>(&sKV[k0 * d_stride + k]);
                    s0 = fmaf(qv.x, kv.x, fmaf(qv.y, kv.y, fmaf(qv.z, kv.z, fmaf(qv.w, kv.w, s0))));
                }
                if (k1 < kv_len) {
                    float4 kv = *reinterpret_cast<const float4*>(&sKV[k1 * d_stride + k]);
                    s1 = fmaf(qv.x, kv.x, fmaf(qv.y, kv.y, fmaf(qv.z, kv.z, fmaf(qv.w, kv.w, s1))));
                }
            }
            if (k0 < kv_len) {
                s0 = fmaf(alibi_alpha, static_cast<float>(q_global_row - (k_start + k0)), s0);
            }
            if (k1 < kv_len) {
                s1 = fmaf(alibi_alpha, static_cast<float>(q_global_row - (k_start + k1)), s1);
            }
        }

        // 2. 16-thread sub-warp Online Softmax
        float rmax = -FLT_MAX;
        if (k0 < kv_len) rmax = fmaxf(rmax, s0);
        if (k1 < kv_len) rmax = fmaxf(rmax, s1);

        float tile_max = subwarp16_max(rmax);
        float m_new = fmaxf(m, tile_max);
        float rescale = __expf(m - m_new);

        float p0 = (k0 < kv_len) ? __expf(s0 - m_new) : 0.0f;
        float p1 = (k1 < kv_len) ? __expf(s1 - m_new) : 0.0f;

        *reinterpret_cast<float2*>(&sP[ty * 36 + k0]) = make_float2(p0, p1);
        float tile_sum = subwarp16_sum(p0 + p1);
        l = l * rescale + tile_sum;
        m = m_new;

        #pragma unroll
        for (int c = 0; c < kVecCols; ++c) {
            o_reg[c].x *= rescale;
            o_reg[c].y *= rescale;
            o_reg[c].z *= rescale;
            o_reg[c].w *= rescale;
        }
        __syncthreads();

        // 3. Load V tile into sKV
        for (int idx = tid; idx < kv_len * vec_d; idx += kThreads) {
            const int r = idx / vec_d;
            const int c4 = idx % vec_d;
            *reinterpret_cast<float4*>(&sKV[r * d_stride + c4 * 4]) =
                *reinterpret_cast<const float4*>(&V[static_cast<size_t>(k_start + r) * d + c4 * 4]);
        }
        __syncthreads();

        // 4. float4-vectorized P @ V
        if (valid_q) {
            const float* p_row = &sP[ty * 36];
            for (int j = 0; j < kv_len; ++j) {
                float p = p_row[j];
                const float* v_row = &sKV[j * d_stride];
                #pragma unroll
                for (int c = 0; c < kVecCols; ++c) {
                    const int col = tx * 4 + c * 64;
                    if (col < d) {
                        float4 vv = *reinterpret_cast<const float4*>(&v_row[col]);
                        o_reg[c].x = fmaf(p, vv.x, o_reg[c].x);
                        o_reg[c].y = fmaf(p, vv.y, o_reg[c].y);
                        o_reg[c].z = fmaf(p, vv.z, o_reg[c].z);
                        o_reg[c].w = fmaf(p, vv.w, o_reg[c].w);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (valid_q) {
        float* o_ptr = output + static_cast<size_t>(q_global_row) * d;
        const float inv_l = 1.0f / l;
        #pragma unroll
        for (int c = 0; c < kVecCols; ++c) {
            const int col = tx * 4 + c * 64;
            if (col < d) {
                *reinterpret_cast<float4*>(&o_ptr[col]) = make_float4(
                    o_reg[c].x * inv_l, o_reg[c].y * inv_l, o_reg[c].z * inv_l, o_reg[c].w * inv_l);
            }
        }
    }
}

// General fallback for d % 4 != 0 or d > 256
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void alibi_attention_fallback_kernel(const float* __restrict__ Q,
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
    const int q_row = blockIdx.x * 4 + warp_id;
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
            if (col < d) dot = fmaf(q_reg[i], k_ptr[col], dot);
        }
        float score = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0);
        score = fmaf(alibi_alpha, static_cast<float>(q_row - k_row), score);

        float m_new = fmaxf(m, score);
        float rescale = __expf(m - m_new);
        float p = __expf(score - m_new);
        l = l * rescale + p;

        const float* v_ptr = V + static_cast<size_t>(k_row) * d;
        for (int i = 0; i < elems; ++i) {
            const int col = lane + i * 32;
            if (col < d) o_reg[i] = fmaf(p, v_ptr[col], o_reg[i] * rescale);
        }
        m = m_new;
    }

    float* o_ptr = output + static_cast<size_t>(q_row) * d;
    const float inv_l = 1.0f / l;
    for (int i = 0; i < elems; ++i) {
        const int col = lane + i * 32;
        if (col < d) o_ptr[col] = o_reg[i] * inv_l;
    }
}

}  // namespace

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d, float alpha) {
    const float scale = rsqrtf(static_cast<float>(d));
    if ((d & 3) == 0 && d <= 256) {
        dim3 block(kThreads);
        dim3 grid((M + kBr - 1) / kBr);
        const int d_stride = d + 4;
        const size_t smem_bytes = (static_cast<size_t>(kBr + kBc) * d_stride + kBr * 36ULL) * sizeof(float);
        if (d <= 64) {
            alibi_attention_8x16_kernel<1><<<grid, block, smem_bytes>>>(Q, K, V, output, M, N, d, scale, alpha);
        } else if (d <= 128) {
            alibi_attention_8x16_kernel<2><<<grid, block, smem_bytes>>>(Q, K, V, output, M, N, d, scale, alpha);
        } else {
            alibi_attention_8x16_kernel<4><<<grid, block, smem_bytes>>>(Q, K, V, output, M, N, d, scale, alpha);
        }
        return;
    }
    dim3 block(128);
    dim3 grid((M + 3) / 4);
    alibi_attention_fallback_kernel<<<grid, block>>>(Q, K, V, output, M, N, d, scale, alpha);
}
