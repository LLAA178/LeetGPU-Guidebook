#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

namespace {

constexpr int kBr = 16;
constexpr int kBc = 32;
constexpr int kThreads = 64;  // 8x8: ty in [0..7] (2 Q rows), tx in [0..7] (4 K cols / kOCols)

__device__ __forceinline__ float subwarp8_max(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    return val;
}

__device__ __forceinline__ float subwarp8_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 1);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    return val;
}

// 2D Register-Tiled FlashAttention (Br=16, Bc=32, 64 threads = 8x8):
// Templated on kOCols so o_reg[2][kOCols] stays 100% in registers (0 stack frame).
template <int kOCols>
__global__ void __launch_bounds__(64, 2)
softmax_attention_tiled2d_kernel(const float* __restrict__ Q,
                                 const float* __restrict__ K,
                                 const float* __restrict__ V,
                                 float* __restrict__ output,
                                 int M,
                                 int N,
                                 int d,
                                 float scale) {
    extern __shared__ float smem[];
    const int d_stride = ((d + 3) & ~3) + 4;
    float* sQ = smem;                         // 16 * d_stride
    float* sKV = sQ + kBr * d_stride;         // 32 * d_stride (reused for K then V)
    float* sP = sKV + kBc * d_stride;         // 16 * 36

    const int tid = threadIdx.x;
    const int ty = tid >> 3;  // 0..7 -> 2 Q rows: ty*2, ty*2+1
    const int tx = tid & 7;   // 0..7 -> 4 K cols: tx*4 .. tx*4+3
    const int q_block_row = blockIdx.x * kBr;
    const int q_len = min(kBr, M - q_block_row);

    const bool use_vec4 = ((d & 3) == 0);
    const int vec_d = d >> 2;

    if (use_vec4) {
        for (int idx = tid; idx < q_len * vec_d; idx += kThreads) {
            const int r = idx / vec_d;
            const int c4 = idx % vec_d;
            float4 v = *reinterpret_cast<const float4*>(&Q[static_cast<size_t>(q_block_row + r) * d + c4 * 4]);
            *reinterpret_cast<float4*>(&sQ[r * d_stride + c4 * 4]) =
                make_float4(v.x * scale, v.y * scale, v.z * scale, v.w * scale);
        }
    } else {
        for (int idx = tid; idx < q_len * d; idx += kThreads) {
            const int r = idx / d;
            const int c = idx % d;
            sQ[r * d_stride + c] = Q[static_cast<size_t>(q_block_row + r) * d + c] * scale;
        }
    }
    __syncthreads();

    float o_reg[2][kOCols];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int c = 0; c < kOCols; ++c) {
            o_reg[i][c] = 0.0f;
        }
    }

    float m[2] = {-FLT_MAX, -FLT_MAX};
    float l[2] = {0.0f, 0.0f};

    for (int k_start = 0; k_start < N; k_start += kBc) {
        const int kv_len = min(kBc, N - k_start);

        if (use_vec4) {
            for (int idx = tid; idx < kv_len * vec_d; idx += kThreads) {
                const int r = idx / vec_d;
                const int c4 = idx % vec_d;
                *reinterpret_cast<float4*>(&sKV[r * d_stride + c4 * 4]) =
                    *reinterpret_cast<const float4*>(&K[static_cast<size_t>(k_start + r) * d + c4 * 4]);
            }
        } else {
            for (int idx = tid; idx < kv_len * d; idx += kThreads) {
                const int r = idx / d;
                const int c = idx % d;
                sKV[r * d_stride + c] = K[static_cast<size_t>(k_start + r) * d + c];
            }
        }
        __syncthreads();

        // 1. Pure FMA S = Q @ K^T (2x4 per thread, float4 vectorized when d % 4 == 0)
        float s_acc[2][4] = {{0.0f}};
        if (use_vec4) {
            for (int k = 0; k < d; k += 4) {
                float4 q0 = (ty * 2 + 0 < q_len)
                                ? *reinterpret_cast<const float4*>(&sQ[(ty * 2 + 0) * d_stride + k])
                                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 q1 = (ty * 2 + 1 < q_len)
                                ? *reinterpret_cast<const float4*>(&sQ[(ty * 2 + 1) * d_stride + k])
                                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (tx * 4 + j < kv_len) {
                        float4 kv = *reinterpret_cast<const float4*>(&sKV[(tx * 4 + j) * d_stride + k]);
                        s_acc[0][j] = fmaf(q0.x, kv.x, fmaf(q0.y, kv.y, fmaf(q0.z, kv.z, fmaf(q0.w, kv.w, s_acc[0][j]))));
                        s_acc[1][j] = fmaf(q1.x, kv.x, fmaf(q1.y, kv.y, fmaf(q1.z, kv.z, fmaf(q1.w, kv.w, s_acc[1][j]))));
                    }
                }
            }
        } else {
            for (int k = 0; k < d; ++k) {
                float q0 = (ty * 2 + 0 < q_len) ? sQ[(ty * 2 + 0) * d_stride + k] : 0.0f;
                float q1 = (ty * 2 + 1 < q_len) ? sQ[(ty * 2 + 1) * d_stride + k] : 0.0f;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float kv = (tx * 4 + j < kv_len) ? sKV[(tx * 4 + j) * d_stride + k] : 0.0f;
                    s_acc[0][j] = fmaf(q0, kv, s_acc[0][j]);
                    s_acc[1][j] = fmaf(q1, kv, s_acc[1][j]);
                }
            }
        }

        // 2. 8-thread sub-warp Online Softmax + write P to sP
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            float rmax = -FLT_MAX;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (tx * 4 + j < kv_len) {
                    rmax = fmaxf(rmax, s_acc[i][j]);
                }
            }
            float tile_max = subwarp8_max(rmax);
            float m_new = fmaxf(m[i], tile_max);
            float alpha = __expf(m[i] - m_new);

            float rsum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float p = (tx * 4 + j < kv_len) ? __expf(s_acc[i][j] - m_new) : 0.0f;
                sP[(ty * 2 + i) * 36 + (tx * 4 + j)] = p;
                rsum += p;
            }
            float tile_sum = subwarp8_sum(rsum);
            l[i] = l[i] * alpha + tile_sum;
            m[i] = m_new;

            #pragma unroll
            for (int c = 0; c < kOCols; ++c) {
                o_reg[i][c] *= alpha;
            }
        }
        __syncthreads();

        // 3. Load V tile into sKV
        if (use_vec4) {
            for (int idx = tid; idx < kv_len * vec_d; idx += kThreads) {
                const int r = idx / vec_d;
                const int c4 = idx % vec_d;
                *reinterpret_cast<float4*>(&sKV[r * d_stride + c4 * 4]) =
                    *reinterpret_cast<const float4*>(&V[static_cast<size_t>(k_start + r) * d + c4 * 4]);
            }
        } else {
            for (int idx = tid; idx < kv_len * d; idx += kThreads) {
                const int r = idx / d;
                const int c = idx % d;
                sKV[r * d_stride + c] = V[static_cast<size_t>(k_start + r) * d + c];
            }
        }
        __syncthreads();

        // 4. Pure FMA O += P @ V
        for (int j = 0; j < kv_len; ++j) {
            float p0 = sP[(ty * 2 + 0) * 36 + j];
            float p1 = sP[(ty * 2 + 1) * 36 + j];
            const float* v_row = &sKV[j * d_stride];
            #pragma unroll
            for (int c = 0; c < kOCols; ++c) {
                const int col = tx + c * 8;
                if (col < d) {
                    float vv = v_row[col];
                    o_reg[0][c] = fmaf(p0, vv, o_reg[0][c]);
                    o_reg[1][c] = fmaf(p1, vv, o_reg[1][c]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int qr = q_block_row + ty * 2 + i;
        if (qr < M) {
            float* o_ptr = output + static_cast<size_t>(qr) * d;
            const float inv_l = 1.0f / l[i];
            #pragma unroll
            for (int c = 0; c < kOCols; ++c) {
                const int col = tx + c * 8;
                if (col < d) {
                    o_ptr[col] = o_reg[i][c] * inv_l;
                }
            }
        }
    }
}

}  // namespace

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d) {
    dim3 block(kThreads);
    dim3 grid((M + kBr - 1) / kBr);
    const int d_stride = ((d + 3) & ~3) + 4;
    const size_t smem_bytes = (static_cast<size_t>(kBr + kBc) * d_stride + kBr * 36ULL) * sizeof(float);
    const float scale = rsqrtf(static_cast<float>(d));

    if (d <= 32) {
        softmax_attention_tiled2d_kernel<4><<<grid, block, smem_bytes>>>(Q, K, V, output, M, N, d, scale);
    } else if (d <= 64) {
        softmax_attention_tiled2d_kernel<8><<<grid, block, smem_bytes>>>(Q, K, V, output, M, N, d, scale);
    } else if (d <= 96) {
        softmax_attention_tiled2d_kernel<12><<<grid, block, smem_bytes>>>(Q, K, V, output, M, N, d, scale);
    } else {
        softmax_attention_tiled2d_kernel<16><<<grid, block, smem_bytes>>>(Q, K, V, output, M, N, d, scale);
    }
}
