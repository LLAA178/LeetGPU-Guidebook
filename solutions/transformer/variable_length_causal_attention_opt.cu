#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

namespace {

constexpr int kBr = 32;
constexpr int kBc = 32;
constexpr int kThreads = 128;  // 8x16: ty in [0..7] (4 Q rows/thread), tx in [0..15] (2 K cols / float4 O cols)

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

// Tiled float4-vectorized FlashAttention for packed variable-length sequences (Br=32, Bc=32):
// - Compact 1D grid (ceil(T / Br) + S): eliminates empty 2D-grid blocks with zero host sync.
// - Separate sK and sV buffers (30.0 KB for d=64 -> 2 blocks/SM with 64 KB smem carveout):
//   loads K & V together and cuts __syncthreads() from 4 to 2 per tile (+ __syncwarp for sP).
// - Reverse per-sequence tile scheduling + 4x Q-row register reuse + float4 sP loads.
template <int kVecCols>
__global__ void __launch_bounds__(128, 2)
varlen_causal_attention_br32_kernel(const float* __restrict__ Q,
                                    const float* __restrict__ K,
                                    const float* __restrict__ V,
                                    const int* __restrict__ cu_seqlens,
                                    float* __restrict__ output,
                                    int d,
                                    int S,
                                    float scale) {
    __shared__ int s_meta[3];  // [seq_start, seq_len, q_block_idx]
    const int tid = threadIdx.x;
    if (tid == 0) {
        const int bid = blockIdx.x;
        int acc = 0;
        int found_start = -1;
        int found_len = 0;
        int found_qblk = 0;
        int prev = cu_seqlens[0];
        for (int s = 0; s < S; ++s) {
            const int next = cu_seqlens[s + 1];
            const int len = next - prev;
            const int nb = (len + kBr - 1) / kBr;
            if (bid < acc + nb) {
                found_start = prev;
                found_len = len;
                found_qblk = nb - 1 - (bid - acc);
                break;
            }
            acc += nb;
            prev = next;
        }
        s_meta[0] = found_start;
        s_meta[1] = found_len;
        s_meta[2] = found_qblk;
    }
    __syncthreads();

    const int seq_start = s_meta[0];
    if (seq_start < 0) return;

    const int seq_len = s_meta[1];
    const int q_block_idx = s_meta[2];
    const int q_block_row = q_block_idx * kBr;
    const int q_len = min(kBr, seq_len - q_block_row);
    const int q_max_row = q_block_row + q_len - 1;

    extern __shared__ float smem[];
    const int d_stride = d + 4;
    float* sQ = smem;
    float* sK = sQ + kBr * d_stride;
    float* sV = sK + kBc * d_stride;
    float* sP = sV + kBc * d_stride;  // 32 * 36

    const int ty = tid >> 4;  // 0..7  -> 4 Q rows (ty*4 .. ty*4+3)
    const int tx = tid & 15;  // 0..15 -> 2 K cols / float4 O chunks
    const int vec_d = d >> 2;

    const float* Q_seq = Q + static_cast<size_t>(seq_start) * d;
    const float* K_seq = K + static_cast<size_t>(seq_start) * d;
    const float* V_seq = V + static_cast<size_t>(seq_start) * d;
    float* O_seq = output + static_cast<size_t>(seq_start) * d;

    for (int idx = tid; idx < q_len * vec_d; idx += kThreads) {
        const int r = idx / vec_d;
        const int c4 = idx % vec_d;
        float4 v = *reinterpret_cast<const float4*>(
            &Q_seq[static_cast<size_t>(q_block_row + r) * d + c4 * 4]);
        *reinterpret_cast<float4*>(&sQ[r * d_stride + c4 * 4]) =
            make_float4(v.x * scale, v.y * scale, v.z * scale, v.w * scale);
    }

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
        const int kv_len = min(kBc, seq_len - k_start);

        // Load K and V tiles together before the single pre-compute barrier.
        for (int idx = tid; idx < kv_len * vec_d; idx += kThreads) {
            const int r = idx / vec_d;
            const int c4 = idx % vec_d;
            const size_t off = static_cast<size_t>(k_start + r) * d + c4 * 4;
            *reinterpret_cast<float4*>(&sK[r * d_stride + c4 * 4]) =
                *reinterpret_cast<const float4*>(&K_seq[off]);
            *reinterpret_cast<float4*>(&sV[r * d_stride + c4 * 4]) =
                *reinterpret_cast<const float4*>(&V_seq[off]);
        }
        __syncthreads();

        const bool is_full_tile = (k_start + kBc - 1 <= q_block_row) && (q_len == kBr);

        float s0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float s1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        if (is_full_tile) {
            for (int k = 0; k < d; k += 4) {
                float4 kv0 = *reinterpret_cast<const float4*>(&sK[k0 * d_stride + k]);
                float4 kv1 = *reinterpret_cast<const float4*>(&sK[k1 * d_stride + k]);
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
                float4 kv0 = k0_in ? *reinterpret_cast<const float4*>(&sK[k0 * d_stride + k])
                                   : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 kv1 = k1_in ? *reinterpret_cast<const float4*>(&sK[k1 * d_stride + k])
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
                const int q_row_local = q_block_row + r_base + i;
                const bool valid_q = (r_base + i < q_len);
                const bool v0 = valid_q && k0_in && (k_start + k0 <= q_row_local);
                const bool v1 = valid_q && k1_in && (k_start + k1 <= q_row_local);

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
        // Only the 16-thread subwarp shares its own 4 rows of sP.
        __syncwarp();

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
                    const float* v_row = &sV[(j + jj) * d_stride];
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
                const float* v_row = &sV[j * d_stride];
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
            const float inv_l = 1.0f / l[i];
            float* out_row = &O_seq[static_cast<size_t>(q_block_row + r_base + i) * d];
            #pragma unroll
            for (int c = 0; c < kVecCols; ++c) {
                const int col = tx * 4 + c * 64;
                if (col < d) {
                    *reinterpret_cast<float4*>(&out_row[col]) = make_float4(
                        o_reg[i][c].x * inv_l,
                        o_reg[i][c].y * inv_l,
                        o_reg[i][c].z * inv_l,
                        o_reg[i][c].w * inv_l);
                }
            }
        }
    }
}

__global__ void varlen_causal_attention_fallback_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const int* __restrict__ cu_seqlens,
    float* __restrict__ output,
    int T,
    int d,
    int S,
    float scale) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (warp_id >= T) return;

    const int q_row = warp_id;
    int seq_start = 0;
    if (lane == 0) {
        seq_start = find_seq_start(cu_seqlens, S, q_row);
    }
    seq_start = __shfl_sync(0xffffffff, seq_start, 0);

    const int elems = (d + 31) / 32;
    float q_reg[8];
    float o_reg[8];
    const float* q_ptr = Q + static_cast<size_t>(q_row) * d;
    for (int i = 0; i < elems; ++i) {
        const int col = lane + i * 32;
        q_reg[i] = (col < d) ? q_ptr[col] * scale : 0.0f;
        o_reg[i] = 0.0f;
    }

    float m = -FLT_MAX;
    float l = 0.0f;
    for (int k_row = seq_start; k_row <= q_row; ++k_row) {
        const float* k_ptr = K + static_cast<size_t>(k_row) * d;
        float dot = 0.0f;
        for (int i = 0; i < elems; ++i) {
            const int col = lane + i * 32;
            if (col < d) dot = fmaf(q_reg[i], k_ptr[col], dot);
        }
        const float score = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0);
        const float m_new = fmaxf(m, score);
        const float alpha = __expf(m - m_new);
        const float p = __expf(score - m_new);
        l = l * alpha + p;

        const float* v_ptr = V + static_cast<size_t>(k_row) * d;
        for (int i = 0; i < elems; ++i) {
            const int col = lane + i * 32;
            if (col < d) o_reg[i] = fmaf(p, v_ptr[col], o_reg[i] * alpha);
        }
        m = m_new;
    }

    const float inv_l = 1.0f / l;
    float* out_ptr = output + static_cast<size_t>(q_row) * d;
    for (int i = 0; i < elems; ++i) {
        const int col = lane + i * 32;
        if (col < d) out_ptr[col] = o_reg[i] * inv_l;
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

    if ((d & 3) == 0 && d <= 128) {
        const int d_stride = d + 4;
        const size_t smem_bytes =
            (static_cast<size_t>(kBr + 2 * kBc) * d_stride + static_cast<size_t>(kBr) * 36) *
            sizeof(float);
        const int total_blocks = (T + kBr - 1) / kBr + S;
        if (d <= 64) {
            cudaFuncSetAttribute(varlen_causal_attention_br32_kernel<1>,
                                 cudaFuncAttributePreferredSharedMemoryCarveout,
                                 100);
            varlen_causal_attention_br32_kernel<1><<<total_blocks, kThreads, smem_bytes>>>(
                Q, K, V, cu_seqlens, output, d, S, scale);
        } else {
            if (smem_bytes > 48 * 1024) {
                cudaFuncSetAttribute(varlen_causal_attention_br32_kernel<2>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(smem_bytes));
            }
            varlen_causal_attention_br32_kernel<2><<<total_blocks, kThreads, smem_bytes>>>(
                Q, K, V, cu_seqlens, output, d, S, scale);
        }
    } else {
        const int blocks = (T + 3) / 4;
        varlen_causal_attention_fallback_kernel<<<blocks, 128>>>(
            Q, K, V, cu_seqlens, output, T, d, S, scale);
    }
}
