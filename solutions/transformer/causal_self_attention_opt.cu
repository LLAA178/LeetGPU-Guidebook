#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

namespace {

constexpr int kBr = 64;
constexpr int kBc = 32;
constexpr int kThreads = 256;  // 16x16: ty in [0..15] (4 Q rows/thread), tx in [0..15] (2 K cols / float4 O cols)

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

// High-reuse 16x16 (256-thread, 8-warp) float4-vectorized FlashAttention (Br=64, Bc=32):
// - 59.9 KB dynamic shared memory (fits in 64 KB on sm_75+).
// - 64 Q rows share each K/V global load (2x less K/V gmem traffic than Br=32).
// - 8 warps/SM hide LDS & math latency; float4 loads from sP cut P-matrix LDS by 4x.
// - Reverse block scheduling + branch-free interior tiles.
template <int kVecCols>
__global__ void __launch_bounds__(256, 1)
causal_self_attention_br64_kernel(const float* __restrict__ Q,
                                  const float* __restrict__ K,
                                  const float* __restrict__ V,
                                  float* __restrict__ output,
                                  int M,
                                  int d,
                                  float scale) {
    extern __shared__ float smem[];
    const int d_stride = d + 4;
    float* sQ = smem;
    float* sKV = sQ + kBr * d_stride;
    float* sP = sKV + kBc * d_stride;  // 64 * 36

    const int tid = threadIdx.x;
    const int ty = tid >> 4;  // 0..15 -> 4 Q rows (ty*4 .. ty*4+3)
    const int tx = tid & 15;  // 0..15 -> 2 K cols / float4 O chunks

    // Reverse block scheduling: heaviest blocks (largest q_block_row) launch first.
    const int q_block_idx = gridDim.x - 1 - blockIdx.x;
    const int q_block_row = q_block_idx * kBr;
    const int q_len = min(kBr, M - q_block_row);
    const int q_max_row = q_block_row + q_len - 1;
    const int vec_d = d >> 2;

    for (int idx = tid; idx < q_len * vec_d; idx += kThreads) {
        const int r = idx / vec_d;
        const int c4 = idx % vec_d;
        float4 v = *reinterpret_cast<const float4*>(
            &Q[static_cast<size_t>(q_block_row + r) * d + c4 * 4]);
        *reinterpret_cast<float4*>(&sQ[r * d_stride + c4 * 4]) =
            make_float4(v.x * scale, v.y * scale, v.z * scale, v.w * scale);
    }
    __syncthreads();

    float4 o_reg[4][kVecCols];
    float m[4];
    float l[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        m[i] = -FLT_MAX;
        l[i] = 0.0f;
        #pragma unroll
        for (int c = 0; c < kVecCols; ++c) {
            o_reg[i][c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    const int r_base = ty * 4;
    const int k0 = tx * 2;
    const int k1 = k0 + 1;

    for (int k_start = 0; k_start <= q_max_row; k_start += kBc) {
        const int kv_len = min(kBc, M - k_start);

        for (int idx = tid; idx < kv_len * vec_d; idx += kThreads) {
            const int r = idx / vec_d;
            const int c4 = idx % vec_d;
            *reinterpret_cast<float4*>(&sKV[r * d_stride + c4 * 4]) =
                *reinterpret_cast<const float4*>(
                    &K[static_cast<size_t>(k_start + r) * d + c4 * 4]);
        }
        __syncthreads();

        const bool is_full_tile = (k_start + kBc - 1 <= q_block_row) && (q_len == kBr);

        float s0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float s1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        if (is_full_tile) {
            for (int k = 0; k < d; k += 4) {
                float4 kv0 = *reinterpret_cast<const float4*>(&sKV[k0 * d_stride + k]);
                float4 kv1 = *reinterpret_cast<const float4*>(&sKV[k1 * d_stride + k]);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float4 qv = *reinterpret_cast<const float4*>(&sQ[(r_base + i) * d_stride + k]);
                    s0[i] = fmaf(qv.x, kv0.x, fmaf(qv.y, kv0.y, fmaf(qv.z, kv0.z, fmaf(qv.w, kv0.w, s0[i]))));
                    s1[i] = fmaf(qv.x, kv1.x, fmaf(qv.y, kv1.y, fmaf(qv.z, kv1.z, fmaf(qv.w, kv1.w, s1[i]))));
                }
            }

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float tile_max = subwarp16_max(fmaxf(s0[i], s1[i]));
                float m_new = fmaxf(m[i], tile_max);
                float alpha = __expf(m[i] - m_new);
                float p0 = __expf(s0[i] - m_new);
                float p1 = __expf(s1[i] - m_new);
                *reinterpret_cast<float2*>(&sP[(r_base + i) * 36 + k0]) = make_float2(p0, p1);
                float tile_sum = subwarp16_sum(p0 + p1);
                l[i] = l[i] * alpha + tile_sum;
                m[i] = m_new;
                #pragma unroll
                for (int c = 0; c < kVecCols; ++c) {
                    o_reg[i][c].x *= alpha;
                    o_reg[i][c].y *= alpha;
                    o_reg[i][c].z *= alpha;
                    o_reg[i][c].w *= alpha;
                }
            }
        } else {
            const bool k0_in = (k0 < kv_len);
            const bool k1_in = (k1 < kv_len);
            for (int k = 0; k < d; k += 4) {
                float4 kv0 = k0_in ? *reinterpret_cast<const float4*>(&sKV[k0 * d_stride + k])
                                   : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 kv1 = k1_in ? *reinterpret_cast<const float4*>(&sKV[k1 * d_stride + k])
                                   : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (r_base + i < q_len) {
                        float4 qv = *reinterpret_cast<const float4*>(&sQ[(r_base + i) * d_stride + k]);
                        s0[i] = fmaf(qv.x, kv0.x, fmaf(qv.y, kv0.y, fmaf(qv.z, kv0.z, fmaf(qv.w, kv0.w, s0[i]))));
                        s1[i] = fmaf(qv.x, kv1.x, fmaf(qv.y, kv1.y, fmaf(qv.z, kv1.z, fmaf(qv.w, kv1.w, s1[i]))));
                    }
                }
            }

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int q_row_g = q_block_row + r_base + i;
                const bool valid_q = (r_base + i < q_len);
                const bool v0 = valid_q && k0_in && (k_start + k0 <= q_row_g);
                const bool v1 = valid_q && k1_in && (k_start + k1 <= q_row_g);

                float rmax = -FLT_MAX;
                if (v0) rmax = fmaxf(rmax, s0[i]);
                if (v1) rmax = fmaxf(rmax, s1[i]);

                float tile_max = subwarp16_max(rmax);
                float m_new = fmaxf(m[i], tile_max);
                float alpha = (m_new > -FLT_MAX) ? __expf(m[i] - m_new) : 1.0f;
                float p0 = v0 ? __expf(s0[i] - m_new) : 0.0f;
                float p1 = v1 ? __expf(s1[i] - m_new) : 0.0f;

                *reinterpret_cast<float2*>(&sP[(r_base + i) * 36 + k0]) = make_float2(p0, p1);
                float tile_sum = subwarp16_sum(p0 + p1);
                l[i] = l[i] * alpha + tile_sum;
                m[i] = m_new;
                #pragma unroll
                for (int c = 0; c < kVecCols; ++c) {
                    o_reg[i][c].x *= alpha;
                    o_reg[i][c].y *= alpha;
                    o_reg[i][c].z *= alpha;
                    o_reg[i][c].w *= alpha;
                }
            }
        }
        __syncthreads();

        // Load V tile into sKV
        for (int idx = tid; idx < kv_len * vec_d; idx += kThreads) {
            const int r = idx / vec_d;
            const int c4 = idx % vec_d;
            *reinterpret_cast<float4*>(&sKV[r * d_stride + c4 * 4]) =
                *reinterpret_cast<const float4*>(
                    &V[static_cast<size_t>(k_start + r) * d + c4 * 4]);
        }
        __syncthreads();

        // P @ V: float4-vectorized sP loads (4x fewer sP LDS) + 4x V register reuse
        if (kv_len == kBc) {
            for (int j = 0; j < kBc; j += 4) {
                float4 pv0 = *reinterpret_cast<const float4*>(&sP[(r_base + 0) * 36 + j]);
                float4 pv1 = *reinterpret_cast<const float4*>(&sP[(r_base + 1) * 36 + j]);
                float4 pv2 = *reinterpret_cast<const float4*>(&sP[(r_base + 2) * 36 + j]);
                float4 pv3 = *reinterpret_cast<const float4*>(&sP[(r_base + 3) * 36 + j]);

                const float ps[4][4] = {
                    {pv0.x, pv1.x, pv2.x, pv3.x},
                    {pv0.y, pv1.y, pv2.y, pv3.y},
                    {pv0.z, pv1.z, pv2.z, pv3.z},
                    {pv0.w, pv1.w, pv2.w, pv3.w},
                };

                #pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const float* v_row = &sKV[(j + jj) * d_stride];
                    #pragma unroll
                    for (int c = 0; c < kVecCols; ++c) {
                        const int col = tx * 4 + c * 64;
                        if (col < d) {
                            float4 vv = *reinterpret_cast<const float4*>(&v_row[col]);
                            #pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                o_reg[i][c].x = fmaf(ps[jj][i], vv.x, o_reg[i][c].x);
                                o_reg[i][c].y = fmaf(ps[jj][i], vv.y, o_reg[i][c].y);
                                o_reg[i][c].z = fmaf(ps[jj][i], vv.z, o_reg[i][c].z);
                                o_reg[i][c].w = fmaf(ps[jj][i], vv.w, o_reg[i][c].w);
                            }
                        }
                    }
                }
            }
        } else {
            for (int j = 0; j < kv_len; ++j) {
                const float p0 = sP[(r_base + 0) * 36 + j];
                const float p1 = sP[(r_base + 1) * 36 + j];
                const float p2 = sP[(r_base + 2) * 36 + j];
                const float p3 = sP[(r_base + 3) * 36 + j];
                const float* v_row = &sKV[j * d_stride];
                #pragma unroll
                for (int c = 0; c < kVecCols; ++c) {
                    const int col = tx * 4 + c * 64;
                    if (col < d) {
                        float4 vv = *reinterpret_cast<const float4*>(&v_row[col]);
                        o_reg[0][c].x = fmaf(p0, vv.x, o_reg[0][c].x);
                        o_reg[0][c].y = fmaf(p0, vv.y, o_reg[0][c].y);
                        o_reg[0][c].z = fmaf(p0, vv.z, o_reg[0][c].z);
                        o_reg[0][c].w = fmaf(p0, vv.w, o_reg[0][c].w);

                        o_reg[1][c].x = fmaf(p1, vv.x, o_reg[1][c].x);
                        o_reg[1][c].y = fmaf(p1, vv.y, o_reg[1][c].y);
                        o_reg[1][c].z = fmaf(p1, vv.z, o_reg[1][c].z);
                        o_reg[1][c].w = fmaf(p1, vv.w, o_reg[1][c].w);

                        o_reg[2][c].x = fmaf(p2, vv.x, o_reg[2][c].x);
                        o_reg[2][c].y = fmaf(p2, vv.y, o_reg[2][c].y);
                        o_reg[2][c].z = fmaf(p2, vv.z, o_reg[2][c].z);
                        o_reg[2][c].w = fmaf(p2, vv.w, o_reg[2][c].w);

                        o_reg[3][c].x = fmaf(p3, vv.x, o_reg[3][c].x);
                        o_reg[3][c].y = fmaf(p3, vv.y, o_reg[3][c].y);
                        o_reg[3][c].z = fmaf(p3, vv.z, o_reg[3][c].z);
                        o_reg[3][c].w = fmaf(p3, vv.w, o_reg[3][c].w);
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (r_base + i < q_len) {
            float* o_ptr = output + static_cast<size_t>(q_block_row + r_base + i) * d;
            const float inv_l = 1.0f / l[i];
            #pragma unroll
            for (int c = 0; c < kVecCols; ++c) {
                const int col = tx * 4 + c * 64;
                if (col < d) {
                    *reinterpret_cast<float4*>(&o_ptr[col]) = make_float4(
                        o_reg[i][c].x * inv_l, o_reg[i][c].y * inv_l,
                        o_reg[i][c].z * inv_l, o_reg[i][c].w * inv_l);
                }
            }
        }
    }
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void causal_self_attention_fallback_kernel(const float* __restrict__ Q,
                                                      const float* __restrict__ K,
                                                      const float* __restrict__ V,
                                                      float* __restrict__ output,
                                                      int M,
                                                      int d,
                                                      float scale) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int q_row = blockIdx.x * 4 + warp_id;
    if (q_row >= M) return;

    const float* q_ptr = Q + static_cast<size_t>(q_row) * d;
    float q_reg[4], o_reg[4];
    const int elems = (d + 31) >> 5;
    for (int i = 0; i < elems; ++i) {
        const int col = lane + i * 32;
        q_reg[i] = (col < d) ? q_ptr[col] * scale : 0.0f;
        o_reg[i] = 0.0f;
    }
    float m = -FLT_MAX, l = 0.0f;
    for (int k_row = 0; k_row <= q_row; ++k_row) {
        const float* k_ptr = K + static_cast<size_t>(k_row) * d;
        float dot = 0.0f;
        for (int i = 0; i < elems; ++i) {
            const int col = lane + i * 32;
            if (col < d) dot = fmaf(q_reg[i], k_ptr[col], dot);
        }
        float score = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0);
        float m_new = fmaxf(m, score);
        float alpha = __expf(m - m_new);
        float p = __expf(score - m_new);
        l = l * alpha + p;
        const float* v_ptr = V + static_cast<size_t>(k_row) * d;
        for (int i = 0; i < elems; ++i) {
            const int col = lane + i * 32;
            if (col < d) o_reg[i] = fmaf(p, v_ptr[col], o_reg[i] * alpha);
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

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M,
                      int d) {
    const float scale = rsqrtf(static_cast<float>(d));
    if ((d & 3) == 0) {
        dim3 block(kThreads);
        dim3 grid((M + kBr - 1) / kBr);
        const int d_stride = d + 4;
        const size_t smem_bytes =
            (static_cast<size_t>(kBr + kBc) * d_stride + kBr * 36ULL) * sizeof(float);
        if (d <= 64) {
            if (smem_bytes > 49152) {
                cudaFuncSetAttribute(causal_self_attention_br64_kernel<1>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(smem_bytes));
            }
            causal_self_attention_br64_kernel<1><<<grid, block, smem_bytes>>>(Q, K, V, output, M, d, scale);
        } else {
            if (smem_bytes > 49152) {
                cudaFuncSetAttribute(causal_self_attention_br64_kernel<2>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(smem_bytes));
            }
            causal_self_attention_br64_kernel<2><<<grid, block, smem_bytes>>>(Q, K, V, output, M, d, scale);
        }
        return;
    }
    dim3 block(128);
    dim3 grid((M + 3) / 4);
    causal_self_attention_fallback_kernel<<<grid, block>>>(Q, K, V, output, M, d, scale);
}
