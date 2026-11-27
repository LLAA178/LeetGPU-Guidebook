#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

namespace {

constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Fast FlashAttention kernel specialized for D == 128 with 16-byte aligned float4 accesses.
// Tile: Br = 32 query rows, Bc = 32 key/value rows, 256 threads (8 warps).
// Each warp processes 4 query rows (4x Q-row register reuse of shared K/V tiles),
// and each lane owns 1 float4 (4 dims) across D = 128.
// Shared memory: K_smem[32][32] float4 (16 KB) + V_smem[32][32] float4 (16 KB) = 32 KB.
__global__ __launch_bounds__(256, 2) void cross_attention_flash_d128_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int M, int N, int H, float scale)
{
    constexpr int Br = 32;
    constexpr int Bc = 32;
    constexpr int D4 = 32;  // D / 4 = 128 / 4

    __shared__ float4 K_smem[Bc][D4];
    __shared__ float4 V_smem[Bc][D4];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int h = blockIdx.y;
    const int m_base = blockIdx.x * Br + warp_id * 4;

    // Load 4 Q rows into registers (each lane holds 1 float4 = 4 elements of D=128)
    float4 q_reg[4];
    float4 o_reg[4];
    float m_val[4];
    float l_val[4];

    const size_t stride_m = (size_t)H * D4;

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        int m = m_base + r;
        if (m < M) {
            const float4* q_ptr = reinterpret_cast<const float4*>(Q) + ((size_t)m * H + h) * D4;
            q_reg[r] = q_ptr[lane];
        } else {
            q_reg[r] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        o_reg[r] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        m_val[r] = -FLT_MAX;
        l_val[r] = 0.0f;
    }

    const float4* K4 = reinterpret_cast<const float4*>(K) + (size_t)h * D4;
    const float4* V4 = reinterpret_cast<const float4*>(V) + (size_t)h * D4;

    for (int n_start = 0; n_start < N; n_start += Bc) {
        const int tile_len = min(Bc, N - n_start);

        // Cooperatively load K and V tiles into shared memory:
        // 256 threads load 32 * 32 = 1024 float4s (4 float4s per thread).
        #pragma unroll
        for (int idx = tid; idx < Bc * D4; idx += 256) {
            int j = idx >> 5;      // 0 .. 31
            int d4 = idx & 31;     // 0 .. 31
            int n = n_start + j;
            if (n < N) {
                size_t off = (size_t)n * stride_m + d4;
                K_smem[j][d4] = K4[off];
                V_smem[j][d4] = V4[off];
            }
        }
        __syncthreads();

        // Process keys in groups of 4 to amortize O-register rescaling by 4x
        int j = 0;
        for (; j + 3 < tile_len; j += 4) {
            float s0[4], s1[4], s2[4], s3[4];

            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float4 kv = K_smem[j + k][lane];
                float d0 = q_reg[0].x * kv.x + q_reg[0].y * kv.y + q_reg[0].z * kv.z + q_reg[0].w * kv.w;
                float d1 = q_reg[1].x * kv.x + q_reg[1].y * kv.y + q_reg[1].z * kv.z + q_reg[1].w * kv.w;
                float d2 = q_reg[2].x * kv.x + q_reg[2].y * kv.y + q_reg[2].z * kv.z + q_reg[2].w * kv.w;
                float d3 = q_reg[3].x * kv.x + q_reg[3].y * kv.y + q_reg[3].z * kv.z + q_reg[3].w * kv.w;

                s0[k] = warp_reduce_sum(d0) * scale;
                s1[k] = warp_reduce_sum(d1) * scale;
                s2[k] = warp_reduce_sum(d2) * scale;
                s3[k] = warp_reduce_sum(d3) * scale;
            }

            float m_new0 = fmaxf(m_val[0], fmaxf(fmaxf(s0[0], s0[1]), fmaxf(s0[2], s0[3])));
            float m_new1 = fmaxf(m_val[1], fmaxf(fmaxf(s1[0], s1[1]), fmaxf(s1[2], s1[3])));
            float m_new2 = fmaxf(m_val[2], fmaxf(fmaxf(s2[0], s2[1]), fmaxf(s2[2], s2[3])));
            float m_new3 = fmaxf(m_val[3], fmaxf(fmaxf(s3[0], s3[1]), fmaxf(s3[2], s3[3])));

            float a0 = __expf(m_val[0] - m_new0);
            float a1 = __expf(m_val[1] - m_new1);
            float a2 = __expf(m_val[2] - m_new2);
            float a3 = __expf(m_val[3] - m_new3);

            o_reg[0].x *= a0; o_reg[0].y *= a0; o_reg[0].z *= a0; o_reg[0].w *= a0;
            o_reg[1].x *= a1; o_reg[1].y *= a1; o_reg[1].z *= a1; o_reg[1].w *= a1;
            o_reg[2].x *= a2; o_reg[2].y *= a2; o_reg[2].z *= a2; o_reg[2].w *= a2;
            o_reg[3].x *= a3; o_reg[3].y *= a3; o_reg[3].z *= a3; o_reg[3].w *= a3;

            l_val[0] *= a0;
            l_val[1] *= a1;
            l_val[2] *= a2;
            l_val[3] *= a3;

            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float p0 = __expf(s0[k] - m_new0);
                float p1 = __expf(s1[k] - m_new1);
                float p2 = __expf(s2[k] - m_new2);
                float p3 = __expf(s3[k] - m_new3);

                l_val[0] += p0;
                l_val[1] += p1;
                l_val[2] += p2;
                l_val[3] += p3;

                float4 vv = V_smem[j + k][lane];
                o_reg[0].x = fmaf(p0, vv.x, o_reg[0].x);
                o_reg[0].y = fmaf(p0, vv.y, o_reg[0].y);
                o_reg[0].z = fmaf(p0, vv.z, o_reg[0].z);
                o_reg[0].w = fmaf(p0, vv.w, o_reg[0].w);

                o_reg[1].x = fmaf(p1, vv.x, o_reg[1].x);
                o_reg[1].y = fmaf(p1, vv.y, o_reg[1].y);
                o_reg[1].z = fmaf(p1, vv.z, o_reg[1].z);
                o_reg[1].w = fmaf(p1, vv.w, o_reg[1].w);

                o_reg[2].x = fmaf(p2, vv.x, o_reg[2].x);
                o_reg[2].y = fmaf(p2, vv.y, o_reg[2].y);
                o_reg[2].z = fmaf(p2, vv.z, o_reg[2].z);
                o_reg[2].w = fmaf(p2, vv.w, o_reg[2].w);

                o_reg[3].x = fmaf(p3, vv.x, o_reg[3].x);
                o_reg[3].y = fmaf(p3, vv.y, o_reg[3].y);
                o_reg[3].z = fmaf(p3, vv.z, o_reg[3].z);
                o_reg[3].w = fmaf(p3, vv.w, o_reg[3].w);
            }

            m_val[0] = m_new0;
            m_val[1] = m_new1;
            m_val[2] = m_new2;
            m_val[3] = m_new3;
        }

        // Handle remaining keys in partial tile
        for (; j < tile_len; ++j) {
            float4 kv = K_smem[j][lane];
            float4 vv = V_smem[j][lane];

            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                float dot = q_reg[r].x * kv.x + q_reg[r].y * kv.y + q_reg[r].z * kv.z + q_reg[r].w * kv.w;
                float s = warp_reduce_sum(dot) * scale;
                float m_new = fmaxf(m_val[r], s);
                float alpha = __expf(m_val[r] - m_new);
                float p = __expf(s - m_new);
                l_val[r] = l_val[r] * alpha + p;
                o_reg[r].x = fmaf(p, vv.x, o_reg[r].x * alpha);
                o_reg[r].y = fmaf(p, vv.y, o_reg[r].y * alpha);
                o_reg[r].z = fmaf(p, vv.z, o_reg[r].z * alpha);
                o_reg[r].w = fmaf(p, vv.w, o_reg[r].w * alpha);
                m_val[r] = m_new;
            }
        }

        __syncthreads();
    }

    // Write normalized outputs
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        int m = m_base + r;
        if (m < M) {
            float inv_l = 1.0f / l_val[r];
            float4 out_val = make_float4(
                o_reg[r].x * inv_l,
                o_reg[r].y * inv_l,
                o_reg[r].z * inv_l,
                o_reg[r].w * inv_l
            );
            float4* out_ptr = reinterpret_cast<float4*>(output) + ((size_t)m * H + h) * D4;
            out_ptr[lane] = out_val;
        }
    }
}

// General fallback kernel for any D != 128 or unaligned pointers.
constexpr int FALLBACK_WARPS_PER_BLOCK = 4;

__global__ void cross_attention_fallback_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int M, int N, int H, int D, float scale)
{
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int m = blockIdx.x * FALLBACK_WARPS_PER_BLOCK + warp_id;
    int h = blockIdx.y;

    if (m >= M || h >= H) return;

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
        float alpha = __expf(m_val - m_new);
        float p = __expf(s - m_new);
        l_val = l_val * alpha + p;

        for (int i = 0; i < num_elems; ++i) {
            int d = lane + i * WARP_SIZE;
            if (d < D) {
                o_reg[i] = fmaf(p, V[kv_offset + d], o_reg[i] * alpha);
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
    bool aligned16 = (((uintptr_t)Q | (uintptr_t)K | (uintptr_t)V | (uintptr_t)output) & 15) == 0;

    if (D == 128 && aligned16) {
        dim3 block(256);
        dim3 grid((M + 31) / 32, H);
        cross_attention_flash_d128_kernel<<<grid, block>>>(Q, K, V, output, M, N, H, scale);
    } else {
        dim3 block(FALLBACK_WARPS_PER_BLOCK * WARP_SIZE);
        dim3 grid((M + FALLBACK_WARPS_PER_BLOCK - 1) / FALLBACK_WARPS_PER_BLOCK, H);
        cross_attention_fallback_kernel<<<grid, block>>>(Q, K, V, output, M, N, H, D, scale);
    }
}
