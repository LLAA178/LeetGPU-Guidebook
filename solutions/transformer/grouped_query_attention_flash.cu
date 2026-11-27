#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kWarpSize = 32;

// -----------------------------------------------------------------------------
// General FP32 FlashAttention-style tiled kernel (all shapes, all SM >= 7.0)
// -----------------------------------------------------------------------------
constexpr int kGenWarps = 8;
constexpr int kGenThreads = kGenWarps * kWarpSize;  // 256
constexpr int kGenBc = 16;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void gqa_flash_fp32_kernel(const float* __restrict__ Q,
                                      const float* __restrict__ K,
                                      const float* __restrict__ V,
                                      float* __restrict__ output,
                                      int num_q_heads,
                                      int num_kv_heads,
                                      int seq_len,
                                      int head_dim,
                                      float scale) {
    extern __shared__ float smem[];
    const int stride = head_dim + 4;
    float* sK = smem;
    float* sV = smem + kGenBc * stride;

    const int warp_id = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int q_row = blockIdx.x * kGenWarps + warp_id;
    const int q_head = blockIdx.y;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);

    const float* k_base = K + static_cast<size_t>(kv_head) * seq_len * head_dim;
    const float* v_base = V + static_cast<size_t>(kv_head) * seq_len * head_dim;

    float q_reg[8];
    float o_reg[8];
    const int elems = (head_dim + kWarpSize - 1) / kWarpSize;
    const bool valid_q = (q_row < seq_len);
    if (valid_q) {
        const float* q_ptr = Q + (static_cast<size_t>(q_head) * seq_len + q_row) * head_dim;
        for (int i = 0; i < elems; ++i) {
            const int d = lane + i * kWarpSize;
            q_reg[i] = (d < head_dim) ? q_ptr[d] * scale : 0.0f;
            o_reg[i] = 0.0f;
        }
    }

    float m = -FLT_MAX;
    float l = 0.0f;
    const int vec_d = head_dim / 4;

    for (int kv_start = 0; kv_start < seq_len; kv_start += kGenBc) {
        const int kv_len = min(kGenBc, seq_len - kv_start);
        // Cooperative vectorized load of K and V tiles into shared memory
        for (int idx = threadIdx.x; idx < kv_len * vec_d; idx += kGenThreads) {
            const int r = idx / vec_d;
            const int c4 = idx % vec_d;
            const size_t g_off = static_cast<size_t>(kv_start + r) * head_dim + c4 * 4;
            *reinterpret_cast<float4*>(&sK[r * stride + c4 * 4]) =
                *reinterpret_cast<const float4*>(&k_base[g_off]);
            *reinterpret_cast<float4*>(&sV[r * stride + c4 * 4]) =
                *reinterpret_cast<const float4*>(&v_base[g_off]);
        }
        __syncthreads();

        if (valid_q) {
            for (int r = 0; r < kv_len; ++r) {
                const float* k_row = &sK[r * stride];
                float dot = 0.0f;
                for (int i = 0; i < elems; ++i) {
                    const int d = lane + i * kWarpSize;
                    if (d < head_dim) {
                        dot = fmaf(q_reg[i], k_row[d], dot);
                    }
                }
                float score = warp_reduce_sum(dot);
                score = __shfl_sync(0xffffffff, score, 0);

                float m_new = fmaxf(m, score);
                float alpha = expf(m - m_new);
                float p = expf(score - m_new);
                l = l * alpha + p;

                const float* v_row = &sV[r * stride];
                for (int i = 0; i < elems; ++i) {
                    const int d = lane + i * kWarpSize;
                    if (d < head_dim) {
                        o_reg[i] = fmaf(p, v_row[d], o_reg[i] * alpha);
                    }
                }
                m = m_new;
            }
        }
        __syncthreads();
    }

    if (valid_q) {
        float* o_ptr = output + (static_cast<size_t>(q_head) * seq_len + q_row) * head_dim;
        const float inv_l = 1.0f / l;
        for (int i = 0; i < elems; ++i) {
            const int d = lane + i * kWarpSize;
            if (d < head_dim) {
                o_ptr[d] = o_reg[i] * inv_l;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// SM 8.0+ (A100) TF32 Tensor Core FlashAttention-2 kernel for head_dim == 128
// Br = 64 (4 warps x 16 rows), Bc = 64, D = 128
// -----------------------------------------------------------------------------
constexpr int kBr = 64;
constexpr int kBc = 64;
constexpr int kHeadDim = 128;
constexpr int kSmemStride = 132;  // 128 + 4 padding to avoid bank conflicts

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ uint32_t float_to_tf32(float x) {
    uint32_t ret;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(ret) : "f"(x));
    return ret;
}

__device__ __forceinline__ void split_tf32(float x, uint32_t& hi, uint32_t& lo) {
    hi = float_to_tf32(x);
    lo = float_to_tf32(x - __uint_as_float(hi));
}

__device__ __forceinline__ void mma_m16n8k8_tf32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* glob_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(glob_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}
#endif

__global__ void __launch_bounds__(128, 2)
gqa_flash_tf32_d128_kernel(const float* __restrict__ Q,
                           const float* __restrict__ K,
                           const float* __restrict__ V,
                           float* __restrict__ output,
                           int num_q_heads,
                           int num_kv_heads,
                           int seq_len,
                           float scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ float smem[];
    float* sK = smem;                         // 64 * 132 floats (~33.8 KB)
    float* sV = smem + kBc * kSmemStride;     // 64 * 132 floats (~33.8 KB)

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int group_id = lane >> 2;       // 0..7
    const int tid_in_group = lane & 3;    // 0..3

    const int q_block_row = blockIdx.x * kBr;
    const int q_warp_row = q_block_row + warp_id * 16;
    const int q_head = blockIdx.y;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);

    const float* q_base = Q + static_cast<size_t>(q_head) * seq_len * kHeadDim;
    const float* k_base = K + static_cast<size_t>(kv_head) * seq_len * kHeadDim;
    const float* v_base = V + static_cast<size_t>(kv_head) * seq_len * kHeadDim;
    float* o_base = output + static_cast<size_t>(q_head) * seq_len * kHeadDim;

    // Load Q (16 x 128 per warp) directly into FP32 registers, pre-scaled
    float q_reg[16][4];
    const int r0 = q_warp_row + group_id;
    const int r1 = q_warp_row + group_id + 8;
    const bool r0_valid = (r0 < seq_len);
    const bool r1_valid = (r1 < seq_len);

    #pragma unroll
    for (int k_step = 0; k_step < 16; ++k_step) {
        const int c0 = k_step * 8 + tid_in_group;
        const int c1 = c0 + 4;
        q_reg[k_step][0] = r0_valid ? q_base[static_cast<size_t>(r0) * kHeadDim + c0] * scale : 0.0f;
        q_reg[k_step][1] = r1_valid ? q_base[static_cast<size_t>(r1) * kHeadDim + c0] * scale : 0.0f;
        q_reg[k_step][2] = r0_valid ? q_base[static_cast<size_t>(r0) * kHeadDim + c1] * scale : 0.0f;
        q_reg[k_step][3] = r1_valid ? q_base[static_cast<size_t>(r1) * kHeadDim + c1] * scale : 0.0f;
    }

    // Output accumulator O (16 x 128 per warp) in FP32 registers
    float o_reg[16][4];
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        o_reg[j][0] = 0.0f;
        o_reg[j][1] = 0.0f;
        o_reg[j][2] = 0.0f;
        o_reg[j][3] = 0.0f;
    }

    float m0 = -FLT_MAX, m1 = -FLT_MAX;
    float l0 = 0.0f, l1 = 0.0f;

    const int src_lo = (lane & ~3) | (tid_in_group >> 1);
    const int src_hi = (lane & ~3) | ((tid_in_group >> 1) + 2);
    const bool take_odd = (tid_in_group & 1) != 0;

    for (int kv_start = 0; kv_start < seq_len; kv_start += kBc) {
        const int kv_len = min(kBc, seq_len - kv_start);

        // Async load K and V tiles (64 x 128 = 2048 float4s, 128 threads -> 16 float4s/thread)
        #pragma unroll
        for (int idx = threadIdx.x; idx < kBc * (kHeadDim / 4); idx += 128) {
            const int r = idx >> 5;           // 0..63
            const int c4 = idx & 31;          // 0..31
            float* sk_dst = &sK[r * kSmemStride + c4 * 4];
            float* sv_dst = &sV[r * kSmemStride + c4 * 4];
            if (r < kv_len) {
                const size_t g_off = static_cast<size_t>(kv_start + r) * kHeadDim + c4 * 4;
                cp_async_16(sk_dst, &k_base[g_off]);
                cp_async_16(sv_dst, &v_base[g_off]);
            } else {
                *reinterpret_cast<float4*>(sk_dst) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                *reinterpret_cast<float4*>(sv_dst) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // 1. S = Q @ K^T (16 x 64 per warp)
        float s_reg[8][4];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            s_reg[j][0] = 0.0f;
            s_reg[j][1] = 0.0f;
            s_reg[j][2] = 0.0f;
            s_reg[j][3] = 0.0f;
        }

        #pragma unroll
        for (int k_step = 0; k_step < 16; ++k_step) {
            uint32_t a0_hi, a0_lo, a1_hi, a1_lo, a2_hi, a2_lo, a3_hi, a3_lo;
            split_tf32(q_reg[k_step][0], a0_hi, a0_lo);
            split_tf32(q_reg[k_step][1], a1_hi, a1_lo);
            split_tf32(q_reg[k_step][2], a2_hi, a2_lo);
            split_tf32(q_reg[k_step][3], a3_hi, a3_lo);

            const int k_col0 = k_step * 8 + tid_in_group;
            const int k_col1 = k_col0 + 4;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int k_row = j * 8 + group_id;
                uint32_t b0_hi, b0_lo, b1_hi, b1_lo;
                split_tf32(sK[k_row * kSmemStride + k_col0], b0_hi, b0_lo);
                split_tf32(sK[k_row * kSmemStride + k_col1], b1_hi, b1_lo);
                mma_m16n8k8_tf32(
                    s_reg[j][0], s_reg[j][1], s_reg[j][2], s_reg[j][3],
                    a0_hi, a1_hi, a2_hi, a3_hi, b0_hi, b1_hi,
                    s_reg[j][0], s_reg[j][1], s_reg[j][2], s_reg[j][3]);
                mma_m16n8k8_tf32(
                    s_reg[j][0], s_reg[j][1], s_reg[j][2], s_reg[j][3],
                    a0_hi, a1_hi, a2_hi, a3_hi, b0_lo, b1_lo,
                    s_reg[j][0], s_reg[j][1], s_reg[j][2], s_reg[j][3]);
                mma_m16n8k8_tf32(
                    s_reg[j][0], s_reg[j][1], s_reg[j][2], s_reg[j][3],
                    a0_lo, a1_lo, a2_lo, a3_lo, b0_hi, b1_hi,
                    s_reg[j][0], s_reg[j][1], s_reg[j][2], s_reg[j][3]);
            }
        }

        // Mask out-of-bounds keys if partial tile
        if (kv_len < kBc) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int c0 = j * 8 + tid_in_group * 2;
                const int c1 = c0 + 1;
                if (c0 >= kv_len) {
                    s_reg[j][0] = -FLT_MAX;
                    s_reg[j][2] = -FLT_MAX;
                }
                if (c1 >= kv_len) {
                    s_reg[j][1] = -FLT_MAX;
                    s_reg[j][3] = -FLT_MAX;
                }
            }
        }

        // 2. Online Softmax in registers (4-thread sub-warp reductions)
        float row_max0 = -FLT_MAX;
        float row_max1 = -FLT_MAX;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            row_max0 = fmaxf(row_max0, fmaxf(s_reg[j][0], s_reg[j][1]));
            row_max1 = fmaxf(row_max1, fmaxf(s_reg[j][2], s_reg[j][3]));
        }
        row_max0 = fmaxf(row_max0, __shfl_xor_sync(0xffffffff, row_max0, 1));
        row_max0 = fmaxf(row_max0, __shfl_xor_sync(0xffffffff, row_max0, 2));
        row_max1 = fmaxf(row_max1, __shfl_xor_sync(0xffffffff, row_max1, 1));
        row_max1 = fmaxf(row_max1, __shfl_xor_sync(0xffffffff, row_max1, 2));

        float m0_new = fmaxf(m0, row_max0);
        float m1_new = fmaxf(m1, row_max1);
        float alpha0 = expf(m0 - m0_new);
        float alpha1 = expf(m1 - m1_new);

        float row_sum0 = 0.0f;
        float row_sum1 = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            s_reg[j][0] = expf(s_reg[j][0] - m0_new);
            s_reg[j][1] = expf(s_reg[j][1] - m0_new);
            s_reg[j][2] = expf(s_reg[j][2] - m1_new);
            s_reg[j][3] = expf(s_reg[j][3] - m1_new);
            row_sum0 += s_reg[j][0] + s_reg[j][1];
            row_sum1 += s_reg[j][2] + s_reg[j][3];
        }
        row_sum0 += __shfl_xor_sync(0xffffffff, row_sum0, 1);
        row_sum0 += __shfl_xor_sync(0xffffffff, row_sum0, 2);
        row_sum1 += __shfl_xor_sync(0xffffffff, row_sum1, 1);
        row_sum1 += __shfl_xor_sync(0xffffffff, row_sum1, 2);

        l0 = l0 * alpha0 + row_sum0;
        l1 = l1 * alpha1 + row_sum1;
        m0 = m0_new;
        m1 = m1_new;

        // Rescale previous O
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            o_reg[j][0] *= alpha0;
            o_reg[j][1] *= alpha0;
            o_reg[j][2] *= alpha1;
            o_reg[j][3] *= alpha1;
        }

        // 3. O += P @ V (16 x 128 per warp, 8 k-steps of 8)
        #pragma unroll
        for (int k_step = 0; k_step < 8; ++k_step) {
            float s0_lo = __shfl_sync(0xffffffff, s_reg[k_step][0], src_lo);
            float s1_lo = __shfl_sync(0xffffffff, s_reg[k_step][1], src_lo);
            float s2_lo = __shfl_sync(0xffffffff, s_reg[k_step][2], src_lo);
            float s3_lo = __shfl_sync(0xffffffff, s_reg[k_step][3], src_lo);
            float s0_hi = __shfl_sync(0xffffffff, s_reg[k_step][0], src_hi);
            float s1_hi = __shfl_sync(0xffffffff, s_reg[k_step][1], src_hi);
            float s2_hi = __shfl_sync(0xffffffff, s_reg[k_step][2], src_hi);
            float s3_hi = __shfl_sync(0xffffffff, s_reg[k_step][3], src_hi);

            float p_lo_0 = take_odd ? s1_lo : s0_lo;
            float p_lo_1 = take_odd ? s3_lo : s2_lo;
            float p_hi_0 = take_odd ? s1_hi : s0_hi;
            float p_hi_1 = take_odd ? s3_hi : s2_hi;

            uint32_t a0_hi, a0_lo, a1_hi, a1_lo, a2_hi, a2_lo, a3_hi, a3_lo;
            split_tf32(p_lo_0, a0_hi, a0_lo);
            split_tf32(p_lo_1, a1_hi, a1_lo);
            split_tf32(p_hi_0, a2_hi, a2_lo);
            split_tf32(p_hi_1, a3_hi, a3_lo);

            const int v_row0 = k_step * 8 + tid_in_group;
            const int v_row1 = v_row0 + 4;

            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int v_col = j * 8 + group_id;
                uint32_t b0_hi, b0_lo, b1_hi, b1_lo;
                split_tf32(sV[v_row0 * kSmemStride + v_col], b0_hi, b0_lo);
                split_tf32(sV[v_row1 * kSmemStride + v_col], b1_hi, b1_lo);
                mma_m16n8k8_tf32(
                    o_reg[j][0], o_reg[j][1], o_reg[j][2], o_reg[j][3],
                    a0_hi, a1_hi, a2_hi, a3_hi, b0_hi, b1_hi,
                    o_reg[j][0], o_reg[j][1], o_reg[j][2], o_reg[j][3]);
                mma_m16n8k8_tf32(
                    o_reg[j][0], o_reg[j][1], o_reg[j][2], o_reg[j][3],
                    a0_hi, a1_hi, a2_hi, a3_hi, b0_lo, b1_lo,
                    o_reg[j][0], o_reg[j][1], o_reg[j][2], o_reg[j][3]);
                mma_m16n8k8_tf32(
                    o_reg[j][0], o_reg[j][1], o_reg[j][2], o_reg[j][3],
                    a0_lo, a1_lo, a2_lo, a3_lo, b0_hi, b1_hi,
                    o_reg[j][0], o_reg[j][1], o_reg[j][2], o_reg[j][3]);
            }
        }
        __syncthreads();
    }

    // Write normalized O back to global memory
    const float inv_l0 = 1.0f / l0;
    const float inv_l1 = 1.0f / l1;
    if (r0_valid) {
        float* o_ptr0 = o_base + static_cast<size_t>(r0) * kHeadDim;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int c = j * 8 + tid_in_group * 2;
            *reinterpret_cast<float2*>(&o_ptr0[c]) =
                make_float2(o_reg[j][0] * inv_l0, o_reg[j][1] * inv_l0);
        }
    }
    if (r1_valid) {
        float* o_ptr1 = o_base + static_cast<size_t>(r1) * kHeadDim;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int c = j * 8 + tid_in_group * 2;
            *reinterpret_cast<float2*>(&o_ptr1[c]) =
                make_float2(o_reg[j][2] * inv_l1, o_reg[j][3] * inv_l1);
        }
    }
#endif  // __CUDA_ARCH__ >= 800
}

}  // namespace

extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int num_q_heads, int num_kv_heads, int seq_len, int head_dim) {
    const float scale = rsqrtf(static_cast<float>(head_dim));

    static int sm_major = -1;
    if (sm_major < 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, dev);
    }

    if (sm_major >= 8 && head_dim == kHeadDim) {
        const size_t smem_bytes = 2ULL * kBc * kSmemStride * sizeof(float);
        static bool attr_set = false;
        if (!attr_set) {
            cudaFuncSetAttribute(gqa_flash_tf32_d128_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(smem_bytes));
            attr_set = true;
        }
        dim3 block(128);
        dim3 grid((seq_len + kBr - 1) / kBr, num_q_heads);
        gqa_flash_tf32_d128_kernel<<<grid, block, smem_bytes>>>(
            Q, K, V, output, num_q_heads, num_kv_heads, seq_len, scale);
        return;
    }

    // General FP32 FlashAttention-style tiled kernel
    dim3 block(kGenThreads);
    dim3 grid((seq_len + kGenWarps - 1) / kGenWarps, num_q_heads);
    size_t smem_bytes = 2ULL * kGenBc * (head_dim + 4) * sizeof(float);
    gqa_flash_fp32_kernel<<<grid, block, smem_bytes>>>(
        Q, K, V, output, num_q_heads, num_kv_heads, seq_len, head_dim, scale);
}
