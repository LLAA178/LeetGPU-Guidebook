#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

#ifndef EMB_WARPS_PER_BLOCK
#define EMB_WARPS_PER_BLOCK 8
#endif

namespace {

constexpr int kWarp = 32;
constexpr int kWarpsPerBlock = EMB_WARPS_PER_BLOCK;
static_assert(kWarpsPerBlock >= 1 && kWarpsPerBlock <= 16 &&
                  (kWarpsPerBlock & (kWarpsPerBlock - 1)) == 0,
              "EMB_WARPS_PER_BLOCK must be a power of two in [1, 16]");
constexpr int kBlockThreads = kWarpsPerBlock * kWarp;
constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ float warp_allreduce_sum(float val) {
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(kFullMask, val, offset);
    }
    return val;
}

__device__ __forceinline__ void store_streaming_float4(float4* ptr, const float4& v) {
#if __CUDA_ARCH__ >= 700
    asm volatile("st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w));
#else
    *ptr = v;
#endif
}

// ============================================================================
// Exact Specialization for Official Perf Shape: D == 768 (192 float4s = 6 float4s/lane)!
// - Grid: dim3(T, b_chunks) where blockIdx.x = t!
// - Loads position_embeddings[position_ids[t], :], gamma[:], beta[:] ONCE into 9 KiB
//   shared memory per block and reuses them across all B items in the batch chunk!
// - Each warp keeps its entire 768-float summed vector in 6 float4 registers (24 floats),
//   computing exact 2-pass FP32 mean & shifted variance via warp shuffles (zero FP64,
//   zero second global read of token_embeddings)!
// ============================================================================
__global__ __launch_bounds__(kBlockThreads, 4) void token_embedding_d768_tmajor_kernel(
    const int* __restrict__ token_ids,
    const int* __restrict__ position_ids,
    const float* __restrict__ token_embeddings,
    const float* __restrict__ position_embeddings,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int B,
    int T,
    float eps) {
    __shared__ float4 s_pos[192];
    __shared__ float4 s_gamma[192];
    __shared__ float4 s_beta[192];

    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int pos_id = __ldg(&position_ids[t]);
    const float4* __restrict__ pos_row4 =
        reinterpret_cast<const float4*>(position_embeddings + static_cast<size_t>(pos_id) * 768ull);
    const float4* __restrict__ gamma4 = reinterpret_cast<const float4*>(gamma);
    const float4* __restrict__ beta4 = reinterpret_cast<const float4*>(beta);

    for (int i = tid; i < 192; i += kBlockThreads) {
        s_pos[i] = __ldg(&pos_row4[i]);
        s_gamma[i] = __ldg(&gamma4[i]);
        s_beta[i] = __ldg(&beta4[i]);
    }
    __syncthreads();

    const int b_chunks = gridDim.y;
    const int b_start = static_cast<int>(blockIdx.y) * kWarpsPerBlock + warp;
    const int b_stride = b_chunks * kWarpsPerBlock;
    constexpr float kInvD = 1.0f / 768.0f;

    for (int b = b_start; b < B; b += b_stride) {
        const size_t bt = static_cast<size_t>(b) * static_cast<size_t>(T) + static_cast<size_t>(t);
        const int tok_id = __ldg(&token_ids[bt]);
        const float4* __restrict__ tok_row4 =
            reinterpret_cast<const float4*>(token_embeddings + static_cast<size_t>(tok_id) * 768ull);
        float4* __restrict__ out_row4 =
            reinterpret_cast<float4*>(output + bt * 768ull);

        float4 s[6];
        const float4 tv0 = __ldg(&tok_row4[lane]);
        const float4 pv0 = s_pos[lane];
        s[0] = make_float4(tv0.x + pv0.x, tv0.y + pv0.y, tv0.z + pv0.z, tv0.w + pv0.w);
        const float ref = __shfl_sync(kFullMask, s[0].x, 0);
        float dsum = ((s[0].x - ref) + (s[0].y - ref)) + ((s[0].z - ref) + (s[0].w - ref));
#pragma unroll
        for (int k = 1; k < 6; ++k) {
            const int idx4 = k * 32 + lane;
            const float4 tv = __ldg(&tok_row4[idx4]);
            const float4 pv = s_pos[idx4];
            s[k].x = tv.x + pv.x;
            s[k].y = tv.y + pv.y;
            s[k].z = tv.z + pv.z;
            s[k].w = tv.w + pv.w;
            dsum += ((s[k].x - ref) + (s[k].y - ref)) + ((s[k].z - ref) + (s[k].w - ref));
        }

        const float mean = ref + static_cast<float>(static_cast<double>(warp_allreduce_sum(dsum)) * (1.0 / 768.0));

        float sq_sum = 0.0f;
#pragma unroll
        for (int k = 0; k < 6; ++k) {
            const float dx = s[k].x - mean;
            const float dy = s[k].y - mean;
            const float dz = s[k].z - mean;
            const float dw = s[k].w - mean;
            sq_sum += fmaf(dx, dx, dy * dy) + fmaf(dz, dz, dw * dw);
        }

        const float inv_std = rsqrtf(warp_allreduce_sum(sq_sum) * kInvD + eps);

#pragma unroll
        for (int k = 0; k < 6; ++k) {
            const int idx4 = k * 32 + lane;
            const float4 gv = s_gamma[idx4];
            const float4 bv = s_beta[idx4];
            float4 out_v;
            out_v.x = fmaf((s[k].x - mean) * inv_std, gv.x, bv.x);
            out_v.y = fmaf((s[k].y - mean) * inv_std, gv.y, bv.y);
            out_v.z = fmaf((s[k].z - mean) * inv_std, gv.z, bv.z);
            out_v.w = fmaf((s[k].w - mean) * inv_std, gv.w, bv.w);
            store_streaming_float4(&out_row4[idx4], out_v);
        }
    }
}

// ============================================================================
// Templated t-major Register Warp-Per-Token Kernel for Aligned D % 4 == 0 && D <= 1024!
// kMaxVec4 = 2 (D <= 256), 6 (D <= 768), 8 (D <= 1024).
// Shared memory: 3 * (kMaxVec4 * 32) * 16 B (<= 12 KiB at kMaxVec4 = 8 -> 4 blocks/SM!).
// ============================================================================
template <int kMaxVec4>
__global__ __launch_bounds__(kBlockThreads) void token_embedding_vec4_tmajor_kernel(
    const int* __restrict__ token_ids,
    const int* __restrict__ position_ids,
    const float* __restrict__ token_embeddings,
    const float* __restrict__ position_embeddings,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int B,
    int T,
    int d4,
    float inv_d,
    float eps) {
    constexpr int kSmemVecs = kMaxVec4 * 32;
    __shared__ float4 s_pos[kSmemVecs];
    __shared__ float4 s_gamma[kSmemVecs];
    __shared__ float4 s_beta[kSmemVecs];

    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const size_t stride_d4 = static_cast<size_t>(d4);

    const int pos_id = __ldg(&position_ids[t]);
    const float4* __restrict__ pos_row4 =
        reinterpret_cast<const float4*>(position_embeddings) + static_cast<size_t>(pos_id) * stride_d4;
    const float4* __restrict__ gamma4 = reinterpret_cast<const float4*>(gamma);
    const float4* __restrict__ beta4 = reinterpret_cast<const float4*>(beta);

    for (int i = tid; i < d4; i += kBlockThreads) {
        s_pos[i] = __ldg(&pos_row4[i]);
        s_gamma[i] = __ldg(&gamma4[i]);
        s_beta[i] = __ldg(&beta4[i]);
    }
    __syncthreads();

    const int b_chunks = gridDim.y;
    const int b_start = static_cast<int>(blockIdx.y) * kWarpsPerBlock + warp;
    const int b_stride = b_chunks * kWarpsPerBlock;

    for (int b = b_start; b < B; b += b_stride) {
        const size_t bt = static_cast<size_t>(b) * static_cast<size_t>(T) + static_cast<size_t>(t);
        const int tok_id = __ldg(&token_ids[bt]);
        const float4* __restrict__ tok_row4 =
            reinterpret_cast<const float4*>(token_embeddings) + static_cast<size_t>(tok_id) * stride_d4;
        float4* __restrict__ out_row4 =
            reinterpret_cast<float4*>(output) + bt * stride_d4;

        float4 s[kMaxVec4];
        const float4 tv0 = __ldg(&tok_row4[lane]);
        const float4 pv0 = s_pos[lane];
        s[0] = make_float4(tv0.x + pv0.x, tv0.y + pv0.y, tv0.z + pv0.z, tv0.w + pv0.w);
        const float ref = __shfl_sync(kFullMask, s[0].x, 0);
        float dsum = (lane < d4)
                         ? (((s[0].x - ref) + (s[0].y - ref)) + ((s[0].z - ref) + (s[0].w - ref)))
                         : 0.0f;
#pragma unroll
        for (int k = 1; k < kMaxVec4; ++k) {
            const int idx4 = k * 32 + lane;
            if (idx4 < d4) {
                const float4 tv = __ldg(&tok_row4[idx4]);
                const float4 pv = s_pos[idx4];
                s[k].x = tv.x + pv.x;
                s[k].y = tv.y + pv.y;
                s[k].z = tv.z + pv.z;
                s[k].w = tv.w + pv.w;
                dsum += ((s[k].x - ref) + (s[k].y - ref)) + ((s[k].z - ref) + (s[k].w - ref));
            } else {
                s[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }

        const float mean = static_cast<float>(
            static_cast<double>(ref) + static_cast<double>(warp_allreduce_sum(dsum)) * static_cast<double>(inv_d));

        float sq_sum = 0.0f;
#pragma unroll
        for (int k = 0; k < kMaxVec4; ++k) {
            const int idx4 = k * 32 + lane;
            if (idx4 < d4) {
                const float dx = s[k].x - mean;
                const float dy = s[k].y - mean;
                const float dz = s[k].z - mean;
                const float dw = s[k].w - mean;
                sq_sum += fmaf(dx, dx, dy * dy) + fmaf(dz, dz, dw * dw);
            }
        }

        const float inv_std = rsqrtf(warp_allreduce_sum(sq_sum) * inv_d + eps);

#pragma unroll
        for (int k = 0; k < kMaxVec4; ++k) {
            const int idx4 = k * 32 + lane;
            if (idx4 < d4) {
                const float4 gv = s_gamma[idx4];
                const float4 bv = s_beta[idx4];
                float4 out_v;
                out_v.x = fmaf((s[k].x - mean) * inv_std, gv.x, bv.x);
                out_v.y = fmaf((s[k].y - mean) * inv_std, gv.y, bv.y);
                out_v.z = fmaf((s[k].z - mean) * inv_std, gv.z, bv.z);
                out_v.w = fmaf((s[k].w - mean) * inv_std, gv.w, bv.w);
                out_row4[idx4] = out_v;
            }
        }
    }
}

// ============================================================================
// General Fallback: 1 block (256 threads) per (b, t) row for odd D, D > 1024, or
// misaligned pointers. Uses warp shuffle + 8-float smem reduction.
// ============================================================================
__global__ __launch_bounds__(256) void token_embedding_general_kernel(
    const int* __restrict__ token_ids,
    const int* __restrict__ position_ids,
    const float* __restrict__ token_embeddings,
    const float* __restrict__ position_embeddings,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int T,
    int D,
    float inv_d,
    float eps) {
    __shared__ float s_warp[8];
    __shared__ float s_scalar;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int t = row % T;

    const int tok_id = __ldg(&token_ids[row]);
    const int pos_id = __ldg(&position_ids[t]);

    const float* __restrict__ tok_row =
        token_embeddings + static_cast<size_t>(tok_id) * static_cast<size_t>(D);
    const float* __restrict__ pos_row =
        position_embeddings + static_cast<size_t>(pos_id) * static_cast<size_t>(D);
    float* __restrict__ out_row =
        output + static_cast<size_t>(row) * static_cast<size_t>(D);

    if (tid == 0) {
        s_scalar = __ldg(&tok_row[0]) + __ldg(&pos_row[0]);
    }
    __syncthreads();
    const float ref = s_scalar;

    float dsum = 0.0f;
    for (int d = tid; d < D; d += 256) {
        dsum += (__ldg(&tok_row[d]) + __ldg(&pos_row[d])) - ref;
    }
    dsum = warp_allreduce_sum(dsum);
    if (lane == 0) {
        s_warp[warp] = dsum;
    }
    __syncthreads();

    if (warp == 0) {
        float bsum = (lane < 8) ? s_warp[lane] : 0.0f;
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            bsum += __shfl_down_sync(kFullMask, bsum, offset);
        }
        if (lane == 0) {
            s_scalar = static_cast<float>(
                static_cast<double>(ref) + static_cast<double>(bsum) * static_cast<double>(inv_d));
        }
    }
    __syncthreads();
    const float mean = s_scalar;

    float sq_sum = 0.0f;
    for (int d = tid; d < D; d += 256) {
        const float diff = (__ldg(&tok_row[d]) + __ldg(&pos_row[d])) - mean;
        sq_sum = fmaf(diff, diff, sq_sum);
    }
    sq_sum = warp_allreduce_sum(sq_sum);
    if (lane == 0) {
        s_warp[warp] = sq_sum;
    }
    __syncthreads();

    if (warp == 0) {
        float bsq = (lane < 8) ? s_warp[lane] : 0.0f;
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            bsq += __shfl_down_sync(kFullMask, bsq, offset);
        }
        if (lane == 0) {
            s_scalar = rsqrtf(bsq * inv_d + eps);
        }
    }
    __syncthreads();
    const float inv_std = s_scalar;

    for (int d = tid; d < D; d += 256) {
        const float s = __ldg(&tok_row[d]) + __ldg(&pos_row[d]);
        out_row[d] = fmaf((s - mean) * inv_std, __ldg(&gamma[d]), __ldg(&beta[d]));
    }
}

}  // namespace

// token_ids, position_ids, token_embeddings, position_embeddings, gamma, beta, output are device pointers
extern "C" void solve(const int* token_ids,
                      const int* position_ids,
                      const float* token_embeddings,
                      const float* position_embeddings,
                      const float* gamma,
                      const float* beta,
                      float* output,
                      int B,
                      int T,
                      int V,
                      int P,
                      int D,
                      float eps) {
    (void)V;
    (void)P;
    if (B <= 0 || T <= 0 || D <= 0) {
        return;
    }

    const bool aligned_16 =
        (((reinterpret_cast<uintptr_t>(token_embeddings) |
           reinterpret_cast<uintptr_t>(position_embeddings) |
           reinterpret_cast<uintptr_t>(gamma) |
           reinterpret_cast<uintptr_t>(beta) |
           reinterpret_cast<uintptr_t>(output)) &
          15u) == 0u);

    const float inv_d = 1.0f / static_cast<float>(D);

    if (aligned_16 && (D & 3) == 0 && D <= 1024) {
        // Compute b_chunks so total blocks (T * b_chunks) >= 160 whenever enough (B, T) work exists,
        // while keeping b_chunks == 1 when T >= 160 (e.g. T = 512) to maximize pos/gamma/beta reuse!
        const int max_b_chunks = (B + kWarpsPerBlock - 1) / kWarpsPerBlock;
        int b_chunks = 1;
        if (T < 160 && max_b_chunks > 1) {
            b_chunks = (160 + T - 1) / T;
            if (b_chunks > max_b_chunks) {
                b_chunks = max_b_chunks;
            }
        }
        const dim3 grid(static_cast<unsigned>(T), static_cast<unsigned>(b_chunks), 1u);

        if (D == 768) {
            token_embedding_d768_tmajor_kernel<<<grid, kBlockThreads>>>(
                token_ids, position_ids, token_embeddings, position_embeddings,
                gamma, beta, output, B, T, eps);
            return;
        }

        const int d4 = D >> 2;
        if (d4 <= 64) {
            token_embedding_vec4_tmajor_kernel<2><<<grid, kBlockThreads>>>(
                token_ids, position_ids, token_embeddings, position_embeddings,
                gamma, beta, output, B, T, d4, inv_d, eps);
        } else if (d4 <= 192) {
            token_embedding_vec4_tmajor_kernel<6><<<grid, kBlockThreads>>>(
                token_ids, position_ids, token_embeddings, position_embeddings,
                gamma, beta, output, B, T, d4, inv_d, eps);
        } else {
            token_embedding_vec4_tmajor_kernel<8><<<grid, kBlockThreads>>>(
                token_ids, position_ids, token_embeddings, position_embeddings,
                gamma, beta, output, B, T, d4, inv_d, eps);
        }
        return;
    }

    const int total_tokens = B * T;
    token_embedding_general_kernel<<<total_tokens, 256>>>(
        token_ids, position_ids, token_embeddings, position_embeddings,
        gamma, beta, output, T, D, inv_d, eps);
}
