#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kBr = 32;
constexpr int kBc = 32;
constexpr int kThreads = 128;  // For v3 FP32 fallback (d != 64)
constexpr int kTcBr = 64;      // 4 warps * 16 Q rows/warp for d=64 Tensor Core
constexpr int kTcThreads = 128;

__device__ __forceinline__ float subwarp4_max(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    return val;
}

__device__ __forceinline__ float subwarp4_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 1);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    return val;
}

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

__device__ __forceinline__ void split_f32_to_half2(float x, float y, uint32_t& hi, uint32_t& lo) {
    __half hx = __float2half(x);
    __half hy = __float2half(y);
    __half lx = __float2half(x - __half2float(hx));
    __half ly = __float2half(y - __half2float(hy));
    __half2 h2 = __halves2half2(hx, hy);
    __half2 l2 = __halves2half2(lx, ly);
    hi = *reinterpret_cast<const uint32_t*>(&h2);
    lo = *reinterpret_cast<const uint32_t*>(&l2);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
__device__ __forceinline__ void mma_m16n8k8_f32(float& d0,
                                                float& d1,
                                                float& d2,
                                                float& d3,
                                                uint32_t a0,
                                                uint32_t a1,
                                                uint32_t b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(b0));
}
#endif

// SM 7.5 3-MMA FP16-Split Tensor Core FlashAttention for d=64 (Br=64, Bc=32, 4 warps = 128 threads):
// - Splits FP32 x = x_hi + x_lo (FP16) and accumulates hi*hi + hi*lo + lo*hi -> FP32 (~2e-6 accuracy).
// - Q stays 100% in registers (16x64 per warp); pure-register Online Softmax (zero smem for P).
// - Only 18.0 KB shared memory (sK_hi, sK_lo, sV_hi, sV_lo); 2x less K/V gmem & split traffic vs Br=32.
// - Compact 1D grid (ceil(T / Br) + S) with reverse per-sequence tile scheduling.
__global__ void __launch_bounds__(128, 2)
varlen_causal_attention_tc_d64_kernel(const float* __restrict__ Q,
                                      const float* __restrict__ K,
                                      const float* __restrict__ V,
                                      const int* __restrict__ cu_seqlens,
                                      float* __restrict__ output,
                                      int S,
                                      float scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
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
            const int nb = (len + kTcBr - 1) / kTcBr;
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
    const int q_block_row = q_block_idx * kTcBr;
    const int q_len = min(kTcBr, seq_len - q_block_row);
    const int q_max_row = q_block_row + q_len - 1;

    // 18.0 KB total shared memory
    __shared__ __half sK_hi[32][72];
    __shared__ __half sK_lo[32][72];
    __shared__ __half sV_hi[64][36];
    __shared__ __half sV_lo[64][36];

    const int warp_id = tid >> 5;  // 0..1 (16 Q rows per warp)
    const int lane = tid & 31;
    const int g = lane >> 2;  // 0..7
    const int t = lane & 3;   // 0..3

    const float* Q_seq = Q + static_cast<size_t>(seq_start) * 64;
    const float* K_seq = K + static_cast<size_t>(seq_start) * 64;
    const float* V_seq = V + static_cast<size_t>(seq_start) * 64;
    float* O_seq = output + static_cast<size_t>(seq_start) * 64;

    const int q_row0 = q_block_row + warp_id * 16 + g;
    const int q_row1 = q_row0 + 8;
    const bool valid_q0 = (q_row0 < seq_len);
    const bool valid_q1 = (q_row1 < seq_len);

    // Load Q (16x64 per warp) directly into registers, pre-scaled & split into FP16 hi/lo.
    uint32_t q_hi[8][2];
    uint32_t q_lo[8][2];
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        const int col = ks * 8 + 2 * t;
        if (valid_q0) {
            float2 v0 = *reinterpret_cast<const float2*>(&Q_seq[static_cast<size_t>(q_row0) * 64 + col]);
            split_f32_to_half2(v0.x * scale, v0.y * scale, q_hi[ks][0], q_lo[ks][0]);
        } else {
            q_hi[ks][0] = 0;
            q_lo[ks][0] = 0;
        }
        if (valid_q1) {
            float2 v1 = *reinterpret_cast<const float2*>(&Q_seq[static_cast<size_t>(q_row1) * 64 + col]);
            split_f32_to_half2(v1.x * scale, v1.y * scale, q_hi[ks][1], q_lo[ks][1]);
        } else {
            q_hi[ks][1] = 0;
            q_lo[ks][1] = 0;
        }
    }

    float o_reg[8][4];
    #pragma unroll
    for (int n = 0; n < 8; ++n) {
        o_reg[n][0] = 0.0f;
        o_reg[n][1] = 0.0f;
        o_reg[n][2] = 0.0f;
        o_reg[n][3] = 0.0f;
    }
    float m0 = -FLT_MAX, m1 = -FLT_MAX;
    float l0 = 0.0f, l1 = 0.0f;

    for (int k_start = 0; k_start <= q_max_row; k_start += kBc) {
        const int kv_len = min(kBc, seq_len - k_start);

        // Cooperatively load K (32x64) and V (32x64 -> transposed 64x32) into sK/sV hi/lo.
        // 128 threads * 4 float4s = 512 float4s = 32 * 64 floats.
        #pragma unroll
        for (int iter = 0; iter < 4; ++iter) {
            const int idx = tid + iter * kTcThreads;  // 0..511
            const int r = idx >> 4;                   // 0..31
            const int c4 = (idx & 15) << 2;           // 0, 4, ..., 60
            if (r < kv_len) {
                const size_t off = static_cast<size_t>(k_start + r) * 64 + c4;
                float4 kv = *reinterpret_cast<const float4*>(&K_seq[off]);
                uint32_t kh0, kl0, kh1, kl1;
                split_f32_to_half2(kv.x, kv.y, kh0, kl0);
                split_f32_to_half2(kv.z, kv.w, kh1, kl1);
                *reinterpret_cast<uint32_t*>(&sK_hi[r][c4]) = kh0;
                *reinterpret_cast<uint32_t*>(&sK_lo[r][c4]) = kl0;
                *reinterpret_cast<uint32_t*>(&sK_hi[r][c4 + 2]) = kh1;
                *reinterpret_cast<uint32_t*>(&sK_lo[r][c4 + 2]) = kl1;

                float4 vv = *reinterpret_cast<const float4*>(&V_seq[off]);
                __half vh0 = __float2half(vv.x), vh1 = __float2half(vv.y);
                __half vh2 = __float2half(vv.z), vh3 = __float2half(vv.w);
                sV_hi[c4 + 0][r] = vh0;
                sV_lo[c4 + 0][r] = __float2half(vv.x - __half2float(vh0));
                sV_hi[c4 + 1][r] = vh1;
                sV_lo[c4 + 1][r] = __float2half(vv.y - __half2float(vh1));
                sV_hi[c4 + 2][r] = vh2;
                sV_lo[c4 + 2][r] = __float2half(vv.z - __half2float(vh2));
                sV_hi[c4 + 3][r] = vh3;
                sV_lo[c4 + 3][r] = __float2half(vv.w - __half2float(vh3));
            } else {
                *reinterpret_cast<uint32_t*>(&sK_hi[r][c4]) = 0;
                *reinterpret_cast<uint32_t*>(&sK_lo[r][c4]) = 0;
                *reinterpret_cast<uint32_t*>(&sK_hi[r][c4 + 2]) = 0;
                *reinterpret_cast<uint32_t*>(&sK_lo[r][c4 + 2]) = 0;
                sV_hi[c4 + 0][r] = __float2half(0.0f);
                sV_lo[c4 + 0][r] = __float2half(0.0f);
                sV_hi[c4 + 1][r] = __float2half(0.0f);
                sV_lo[c4 + 1][r] = __float2half(0.0f);
                sV_hi[c4 + 2][r] = __float2half(0.0f);
                sV_lo[c4 + 2][r] = __float2half(0.0f);
                sV_hi[c4 + 3][r] = __float2half(0.0f);
                sV_lo[c4 + 3][r] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // 3-MMA Q @ K^T: (16x64) @ (32x64)^T -> s_reg[4][4]
        float s_reg[4][4];
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            s_reg[n][0] = 0.0f;
            s_reg[n][1] = 0.0f;
            s_reg[n][2] = 0.0f;
            s_reg[n][3] = 0.0f;
        }
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const uint32_t qh0 = q_hi[ks][0], qh1 = q_hi[ks][1];
            const uint32_t ql0 = q_lo[ks][0], ql1 = q_lo[ks][1];
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                const uint32_t kh = *reinterpret_cast<const uint32_t*>(&sK_hi[n * 8 + g][ks * 8 + 2 * t]);
                const uint32_t kl = *reinterpret_cast<const uint32_t*>(&sK_lo[n * 8 + g][ks * 8 + 2 * t]);
                mma_m16n8k8_f32(s_reg[n][0], s_reg[n][1], s_reg[n][2], s_reg[n][3], qh0, qh1, kh);
                mma_m16n8k8_f32(s_reg[n][0], s_reg[n][1], s_reg[n][2], s_reg[n][3], qh0, qh1, kl);
                mma_m16n8k8_f32(s_reg[n][0], s_reg[n][1], s_reg[n][2], s_reg[n][3], ql0, ql1, kh);
            }
        }

        // Pure-register Online Softmax with causal masking
        const bool is_full_tile = (k_start + kBc - 1 <= q_block_row + warp_id * 16) && valid_q1;
        float rmax0 = -FLT_MAX;
        float rmax1 = -FLT_MAX;
        if (is_full_tile) {
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                rmax0 = fmaxf(rmax0, fmaxf(s_reg[n][0], s_reg[n][1]));
                rmax1 = fmaxf(rmax1, fmaxf(s_reg[n][2], s_reg[n][3]));
            }
        } else {
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                const int kcol0 = k_start + n * 8 + 2 * t;
                const int kcol1 = kcol0 + 1;
                if (!valid_q0 || kcol0 > q_row0 || kcol0 >= seq_len) s_reg[n][0] = -FLT_MAX;
                if (!valid_q0 || kcol1 > q_row0 || kcol1 >= seq_len) s_reg[n][1] = -FLT_MAX;
                if (!valid_q1 || kcol0 > q_row1 || kcol0 >= seq_len) s_reg[n][2] = -FLT_MAX;
                if (!valid_q1 || kcol1 > q_row1 || kcol1 >= seq_len) s_reg[n][3] = -FLT_MAX;
                rmax0 = fmaxf(rmax0, fmaxf(s_reg[n][0], s_reg[n][1]));
                rmax1 = fmaxf(rmax1, fmaxf(s_reg[n][2], s_reg[n][3]));
            }
        }

        const float tile_max0 = subwarp4_max(rmax0);
        const float tile_max1 = subwarp4_max(rmax1);
        const float m_new0 = fmaxf(m0, tile_max0);
        const float m_new1 = fmaxf(m1, tile_max1);
        const float alpha0 = (m_new0 > -FLT_MAX) ? __expf(m0 - m_new0) : 1.0f;
        const float alpha1 = (m_new1 > -FLT_MAX) ? __expf(m1 - m_new1) : 1.0f;

        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            o_reg[n][0] *= alpha0;
            o_reg[n][1] *= alpha0;
            o_reg[n][2] *= alpha1;
            o_reg[n][3] *= alpha1;
        }

        uint32_t p_hi[4][2];
        uint32_t p_lo[4][2];
        float rsum0 = 0.0f;
        float rsum1 = 0.0f;
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            const float p0 = (s_reg[n][0] > -FLT_MAX * 0.5f) ? __expf(s_reg[n][0] - m_new0) : 0.0f;
            const float p1 = (s_reg[n][1] > -FLT_MAX * 0.5f) ? __expf(s_reg[n][1] - m_new0) : 0.0f;
            const float p2 = (s_reg[n][2] > -FLT_MAX * 0.5f) ? __expf(s_reg[n][2] - m_new1) : 0.0f;
            const float p3 = (s_reg[n][3] > -FLT_MAX * 0.5f) ? __expf(s_reg[n][3] - m_new1) : 0.0f;
            rsum0 += p0 + p1;
            rsum1 += p2 + p3;
            split_f32_to_half2(p0, p1, p_hi[n][0], p_lo[n][0]);
            split_f32_to_half2(p2, p3, p_hi[n][1], p_lo[n][1]);
        }
        l0 = l0 * alpha0 + subwarp4_sum(rsum0);
        l1 = l1 * alpha1 + subwarp4_sum(rsum1);
        m0 = m_new0;
        m1 = m_new1;

        // 3-MMA P @ V: (16x32) @ (32x64) -> o_reg[8][4]
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const uint32_t ph0 = p_hi[ks][0], ph1 = p_hi[ks][1];
            const uint32_t pl0 = p_lo[ks][0], pl1 = p_lo[ks][1];
            #pragma unroll
            for (int n = 0; n < 8; ++n) {
                const uint32_t vh = *reinterpret_cast<const uint32_t*>(&sV_hi[n * 8 + g][ks * 8 + 2 * t]);
                const uint32_t vl = *reinterpret_cast<const uint32_t*>(&sV_lo[n * 8 + g][ks * 8 + 2 * t]);
                mma_m16n8k8_f32(o_reg[n][0], o_reg[n][1], o_reg[n][2], o_reg[n][3], ph0, ph1, vh);
                mma_m16n8k8_f32(o_reg[n][0], o_reg[n][1], o_reg[n][2], o_reg[n][3], ph0, ph1, vl);
                mma_m16n8k8_f32(o_reg[n][0], o_reg[n][1], o_reg[n][2], o_reg[n][3], pl0, pl1, vh);
            }
        }
        __syncthreads();
    }

    if (valid_q0) {
        const float inv_l0 = 1.0f / l0;
        float* out0 = &O_seq[static_cast<size_t>(q_row0) * 64];
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            *reinterpret_cast<float2*>(&out0[n * 8 + 2 * t]) =
                make_float2(o_reg[n][0] * inv_l0, o_reg[n][1] * inv_l0);
        }
    }
    if (valid_q1) {
        const float inv_l1 = 1.0f / l1;
        float* out1 = &O_seq[static_cast<size_t>(q_row1) * 64];
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            *reinterpret_cast<float2*>(&out1[n * 8 + 2 * t]) =
                make_float2(o_reg[n][2] * inv_l1, o_reg[n][3] * inv_l1);
        }
    }
#endif
}

// FP32 Tiled float4-vectorized FlashAttention fallback for d != 64 (d <= 128, d % 4 == 0)
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
    __shared__ int s_meta[3];
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
    float* sP = sV + kBc * d_stride;

    const int ty = tid >> 4;
    const int tx = tid & 15;
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

    if (d == 64) {
        const int total_blocks = (T + kTcBr - 1) / kTcBr + S;
        varlen_causal_attention_tc_d64_kernel<<<total_blocks, kTcThreads>>>(
            Q, K, V, cu_seqlens, output, S, scale);
    } else if ((d & 3) == 0 && d <= 128) {
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
