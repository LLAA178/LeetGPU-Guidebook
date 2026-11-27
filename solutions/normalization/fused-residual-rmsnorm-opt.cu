#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(kFullMask, val, offset);
    }
    return val;
}

template <int kThreads>
__device__ __forceinline__ float block_inv_rms(float local_sum_sq, float inv_c, float eps) {
    constexpr int kWarps = kThreads / kWarpSize;
    __shared__ float s_warp_sums[kWarps];
    __shared__ float s_inv_rms;

    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid >> 5;

    float sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane == 0) {
        s_warp_sums[warp] = sum_sq;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum = (lane < kWarps) ? s_warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            s_inv_rms = rsqrtf(block_sum * inv_c + eps);
        }
    }
    __syncthreads();

    return s_inv_rms;
}

// Dedicated zero-branch fast path for the official C = 4,096 benchmark shape (vec_cols = 1024).
// 256 threads x 4 float4s = 4,096 floats (16 KiB) cached entirely in registers, 4 blocks/SM on T4.
__global__ __launch_bounds__(256, 4) void fused_rmsnorm_exact_4096_kernel(
    const float4* __restrict__ x4,
    const float4* __restrict__ res4,
    const float4* __restrict__ weight4,
    float4* __restrict__ out4,
    float eps) {
    constexpr float kInv4096 = 1.0f / 4096.0f;
    const unsigned int tid = threadIdx.x;
    const unsigned int row_base = blockIdx.x << 10;  // blockIdx.x * 1024 float4s

    const float4 x0 = x4[row_base + tid];
    const float4 r0 = res4[row_base + tid];
    const float4 x1 = x4[row_base + tid + 256u];
    const float4 r1 = res4[row_base + tid + 256u];
    const float4 x2 = x4[row_base + tid + 512u];
    const float4 r2 = res4[row_base + tid + 512u];
    const float4 x3 = x4[row_base + tid + 768u];
    const float4 r3 = res4[row_base + tid + 768u];

    const float4 z0 = make_float4(x0.x + r0.x, x0.y + r0.y, x0.z + r0.z, x0.w + r0.w);
    const float4 z1 = make_float4(x1.x + r1.x, x1.y + r1.y, x1.z + r1.z, x1.w + r1.w);
    const float4 z2 = make_float4(x2.x + r2.x, x2.y + r2.y, x2.z + r2.z, x2.w + r2.w);
    const float4 z3 = make_float4(x3.x + r3.x, x3.y + r3.y, x3.z + r3.z, x3.w + r3.w);

    float sum_sq = fmaf(z0.x, z0.x, z0.y * z0.y) + fmaf(z0.z, z0.z, z0.w * z0.w);
    sum_sq += fmaf(z1.x, z1.x, z1.y * z1.y) + fmaf(z1.z, z1.z, z1.w * z1.w);
    sum_sq += fmaf(z2.x, z2.x, z2.y * z2.y) + fmaf(z2.z, z2.z, z2.w * z2.w);
    sum_sq += fmaf(z3.x, z3.x, z3.y * z3.y) + fmaf(z3.z, z3.z, z3.w * z3.w);

    const float inv_rms = block_inv_rms<256>(sum_sq, kInv4096, eps);

    const float4 w0 = __ldg(&weight4[tid]);
    const float4 w1 = __ldg(&weight4[tid + 256u]);
    const float4 w2 = __ldg(&weight4[tid + 512u]);
    const float4 w3 = __ldg(&weight4[tid + 768u]);

    out4[row_base + tid] =
        make_float4((z0.x * inv_rms) * w0.x, (z0.y * inv_rms) * w0.y, (z0.z * inv_rms) * w0.z, (z0.w * inv_rms) * w0.w);
    out4[row_base + tid + 256u] =
        make_float4((z1.x * inv_rms) * w1.x, (z1.y * inv_rms) * w1.y, (z1.z * inv_rms) * w1.z, (z1.w * inv_rms) * w1.w);
    out4[row_base + tid + 512u] =
        make_float4((z2.x * inv_rms) * w2.x, (z2.y * inv_rms) * w2.y, (z2.z * inv_rms) * w2.z, (z2.w * inv_rms) * w2.w);
    out4[row_base + tid + 768u] =
        make_float4((z3.x * inv_rms) * w3.x, (z3.y * inv_rms) * w3.y, (z3.z * inv_rms) * w3.z, (z3.w * inv_rms) * w3.w);
}

// Single-pass register-cached kernel for 16B-aligned rows with C <= kThreads * kVecs * 4.
// Each thread holds up to kVecs float4 vectors (z = x + residual) in registers across the
// block reduction, eliminating the second global memory read of x and residual completely.
template <int kThreads, int kMinBlocksPerSM, int kVecs, bool kExact>
__global__ __launch_bounds__(kThreads, kMinBlocksPerSM) void fused_rmsnorm_reg_vec4_kernel(
    const float4* __restrict__ x4,
    const float4* __restrict__ res4,
    const float4* __restrict__ weight4,
    float4* __restrict__ out4,
    int vec_cols,
    float inv_c,
    float eps) {
    const int tid = threadIdx.x;
    const unsigned int row_base = blockIdx.x * static_cast<unsigned int>(vec_cols);
    const float4* __restrict__ row_x4 = x4 + row_base;
    const float4* __restrict__ row_res4 = res4 + row_base;
    float4* __restrict__ row_out4 = out4 + row_base;

    float4 z_reg[kVecs];
    float sum_sq = 0.0f;

    if (kExact) {
        const float4 x0 = row_x4[0 * kThreads + tid];
        const float4 r0 = row_res4[0 * kThreads + tid];
        z_reg[0] = make_float4(x0.x + r0.x, x0.y + r0.y, x0.z + r0.z, x0.w + r0.w);
        sum_sq = fmaf(z_reg[0].x, z_reg[0].x, z_reg[0].y * z_reg[0].y) +
                 fmaf(z_reg[0].z, z_reg[0].z, z_reg[0].w * z_reg[0].w);

        #pragma unroll
        for (int v = 1; v < kVecs; ++v) {
            const int col4 = v * kThreads + tid;
            const float4 xv = row_x4[col4];
            const float4 rv = row_res4[col4];
            float4 zv;
            zv.x = xv.x + rv.x;
            zv.y = xv.y + rv.y;
            zv.z = xv.z + rv.z;
            zv.w = xv.w + rv.w;
            z_reg[v] = zv;

            sum_sq += fmaf(zv.x, zv.x, zv.y * zv.y) +
                      fmaf(zv.z, zv.z, zv.w * zv.w);
        }

        const float inv_rms = block_inv_rms<kThreads>(sum_sq, inv_c, eps);

        #pragma unroll
        for (int v = 0; v < kVecs; ++v) {
            const int col4 = v * kThreads + tid;
            const float4 wv = __ldg(&weight4[col4]);
            const float4 zv = z_reg[v];
            float4 ov;
            ov.x = (zv.x * inv_rms) * wv.x;
            ov.y = (zv.y * inv_rms) * wv.y;
            ov.z = (zv.z * inv_rms) * wv.z;
            ov.w = (zv.w * inv_rms) * wv.w;
            row_out4[col4] = ov;
        }
    } else {
        #pragma unroll
        for (int v = 0; v < kVecs; ++v) {
            const int col4 = v * kThreads + tid;
            float4 zv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (col4 < vec_cols) {
                const float4 xv = row_x4[col4];
                const float4 rv = row_res4[col4];
                zv.x = xv.x + rv.x;
                zv.y = xv.y + rv.y;
                zv.z = xv.z + rv.z;
                zv.w = xv.w + rv.w;
                sum_sq = fmaf(zv.x, zv.x, sum_sq);
                sum_sq = fmaf(zv.y, zv.y, sum_sq);
                sum_sq = fmaf(zv.z, zv.z, sum_sq);
                sum_sq = fmaf(zv.w, zv.w, sum_sq);
            }
            z_reg[v] = zv;
        }

        const float inv_rms = block_inv_rms<kThreads>(sum_sq, inv_c, eps);

        #pragma unroll
        for (int v = 0; v < kVecs; ++v) {
            const int col4 = v * kThreads + tid;
            if (col4 < vec_cols) {
                const float4 wv = __ldg(&weight4[col4]);
                const float4 zv = z_reg[v];
                float4 ov;
                ov.x = (zv.x * inv_rms) * wv.x;
                ov.y = (zv.y * inv_rms) * wv.y;
                ov.z = (zv.z * inv_rms) * wv.z;
                ov.w = (zv.w * inv_rms) * wv.w;
                row_out4[col4] = ov;
            }
        }
    }
}

// Single-pass dynamic shared-memory cached kernel for 8192 < C <= 12288 (<= 48 KiB smem)
// with 16B alignment. Keeps z = x + residual on-chip in shared memory to avoid re-reading
// global memory on the normalization writeback pass.
template <int kThreads>
__global__ __launch_bounds__(kThreads) void fused_rmsnorm_smem_vec4_kernel(
    const float4* __restrict__ x4,
    const float4* __restrict__ res4,
    const float4* __restrict__ weight4,
    float4* __restrict__ out4,
    int vec_cols,
    float inv_c,
    float eps) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    float4* s_z4 = reinterpret_cast<float4*>(smem_raw);

    const int tid = threadIdx.x;
    const unsigned int row_base = blockIdx.x * static_cast<unsigned int>(vec_cols);
    const float4* __restrict__ row_x4 = x4 + row_base;
    const float4* __restrict__ row_res4 = res4 + row_base;
    float4* __restrict__ row_out4 = out4 + row_base;

    float sum_sq = 0.0f;
    for (int col4 = tid; col4 < vec_cols; col4 += kThreads) {
        const float4 xv = row_x4[col4];
        const float4 rv = row_res4[col4];
        float4 zv;
        zv.x = xv.x + rv.x;
        zv.y = xv.y + rv.y;
        zv.z = xv.z + rv.z;
        zv.w = xv.w + rv.w;
        s_z4[col4] = zv;

        const float s0 = fmaf(zv.x, zv.x, zv.y * zv.y);
        const float s1 = fmaf(zv.z, zv.z, zv.w * zv.w);
        sum_sq += s0 + s1;
    }

    const float inv_rms = block_inv_rms<kThreads>(sum_sq, inv_c, eps);

    for (int col4 = tid; col4 < vec_cols; col4 += kThreads) {
        const float4 zv = s_z4[col4];
        const float4 wv = __ldg(&weight4[col4]);
        float4 ov;
        ov.x = (zv.x * inv_rms) * wv.x;
        ov.y = (zv.y * inv_rms) * wv.y;
        ov.z = (zv.z * inv_rms) * wv.z;
        ov.w = (zv.w * inv_rms) * wv.w;
        row_out4[col4] = ov;
    }
}

// Vectorized two-pass fallback for very wide aligned rows (C > 12288, C % 4 == 0).
template <int kThreads>
__global__ __launch_bounds__(kThreads) void fused_rmsnorm_twopass_vec4_kernel(
    const float4* __restrict__ x4,
    const float4* __restrict__ res4,
    const float4* __restrict__ weight4,
    float4* __restrict__ out4,
    int vec_cols,
    float inv_c,
    float eps) {
    const int tid = threadIdx.x;
    const unsigned int row_base = blockIdx.x * static_cast<unsigned int>(vec_cols);
    const float4* __restrict__ row_x4 = x4 + row_base;
    const float4* __restrict__ row_res4 = res4 + row_base;
    float4* __restrict__ row_out4 = out4 + row_base;

    float sum_sq = 0.0f;
    for (int col4 = tid; col4 < vec_cols; col4 += kThreads) {
        const float4 xv = row_x4[col4];
        const float4 rv = row_res4[col4];
        const float zx = xv.x + rv.x;
        const float zy = xv.y + rv.y;
        const float zz = xv.z + rv.z;
        const float zw = xv.w + rv.w;
        sum_sq += fmaf(zx, zx, zy * zy) + fmaf(zz, zz, zw * zw);
    }

    const float inv_rms = block_inv_rms<kThreads>(sum_sq, inv_c, eps);

    for (int col4 = tid; col4 < vec_cols; col4 += kThreads) {
        const float4 xv = row_x4[col4];
        const float4 rv = row_res4[col4];
        const float4 wv = __ldg(&weight4[col4]);
        float4 ov;
        ov.x = ((xv.x + rv.x) * inv_rms) * wv.x;
        ov.y = ((xv.y + rv.y) * inv_rms) * wv.y;
        ov.z = ((xv.z + rv.z) * inv_rms) * wv.z;
        ov.w = ((xv.w + rv.w) * inv_rms) * wv.w;
        row_out4[col4] = ov;
    }
}

// Single-pass shared-memory scalar kernel for unaligned pointers or C % 4 != 0 when C <= 12288.
template <int kThreads>
__global__ __launch_bounds__(kThreads) void fused_rmsnorm_smem_scalar_kernel(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int C,
    float inv_c,
    float eps) {
    extern __shared__ float s_z[];

    const int tid = threadIdx.x;
    const size_t row_base = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(C);
    const float* __restrict__ row_x = x + row_base;
    const float* __restrict__ row_res = residual + row_base;
    float* __restrict__ row_out = out + row_base;

    float sum_sq = 0.0f;
    for (int j = tid; j < C; j += kThreads) {
        const float z = row_x[j] + row_res[j];
        s_z[j] = z;
        sum_sq = fmaf(z, z, sum_sq);
    }

    const float inv_rms = block_inv_rms<kThreads>(sum_sq, inv_c, eps);

    for (int j = tid; j < C; j += kThreads) {
        row_out[j] = (s_z[j] * inv_rms) * __ldg(&weight[j]);
    }
}

// Universal two-pass scalar fallback for C > 12288 when pointers are unaligned or C % 4 != 0.
template <int kThreads>
__global__ __launch_bounds__(kThreads) void fused_rmsnorm_twopass_scalar_kernel(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int C,
    float inv_c,
    float eps) {
    const int tid = threadIdx.x;
    const size_t row_base = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(C);
    const float* __restrict__ row_x = x + row_base;
    const float* __restrict__ row_res = residual + row_base;
    float* __restrict__ row_out = out + row_base;

    float sum_sq = 0.0f;
    for (int j = tid; j < C; j += kThreads) {
        const float z = row_x[j] + row_res[j];
        sum_sq = fmaf(z, z, sum_sq);
    }

    const float inv_rms = block_inv_rms<kThreads>(sum_sq, inv_c, eps);

    for (int j = tid; j < C; j += kThreads) {
        const float z = row_x[j] + row_res[j];
        row_out[j] = (z * inv_rms) * __ldg(&weight[j]);
    }
}

inline bool is_aligned_16b(const void* p0, const void* p1, const void* p2, const void* p3) {
    const uintptr_t mask = reinterpret_cast<uintptr_t>(p0) |
                           reinterpret_cast<uintptr_t>(p1) |
                           reinterpret_cast<uintptr_t>(p2) |
                           reinterpret_cast<uintptr_t>(p3);
    return (mask & 15u) == 0u;
}

}  // namespace

// x, residual, weight, out are device pointers
extern "C" void solve(const float* x,
                      const float* residual,
                      const float* weight,
                      float* out,
                      int N,
                      int C,
                      float eps) {
    if (N <= 0 || C <= 0) {
        return;
    }

    const float inv_c = 1.0f / static_cast<float>(C);

    if ((C & 3) == 0 && is_aligned_16b(x, residual, weight, out)) {
        const int vec_cols = C >> 2;
        const float4* x4 = reinterpret_cast<const float4*>(x);
        const float4* res4 = reinterpret_cast<const float4*>(residual);
        const float4* weight4 = reinterpret_cast<const float4*>(weight);
        float4* out4 = reinterpret_cast<float4*>(out);

        if (C == 4096) {
            fused_rmsnorm_exact_4096_kernel<<<N, 256>>>(x4, res4, weight4, out4, eps);
        } else if (C <= 1024) {
            if (C == 1024) {
                fused_rmsnorm_reg_vec4_kernel<256, 4, 1, true><<<N, 256>>>(
                    x4, res4, weight4, out4, vec_cols, inv_c, eps);
            } else {
                fused_rmsnorm_reg_vec4_kernel<256, 4, 1, false><<<N, 256>>>(
                    x4, res4, weight4, out4, vec_cols, inv_c, eps);
            }
        } else if (C <= 2048) {
            if (C == 2048) {
                fused_rmsnorm_reg_vec4_kernel<256, 4, 2, true><<<N, 256>>>(
                    x4, res4, weight4, out4, vec_cols, inv_c, eps);
            } else {
                fused_rmsnorm_reg_vec4_kernel<256, 4, 2, false><<<N, 256>>>(
                    x4, res4, weight4, out4, vec_cols, inv_c, eps);
            }
        } else if (C < 4096) {
            fused_rmsnorm_reg_vec4_kernel<256, 4, 4, false><<<N, 256>>>(
                x4, res4, weight4, out4, vec_cols, inv_c, eps);
        } else if (C <= 8192) {
            if (C == 8192) {
                fused_rmsnorm_reg_vec4_kernel<512, 2, 4, true><<<N, 512>>>(
                    x4, res4, weight4, out4, vec_cols, inv_c, eps);
            } else {
                fused_rmsnorm_reg_vec4_kernel<512, 2, 4, false><<<N, 512>>>(
                    x4, res4, weight4, out4, vec_cols, inv_c, eps);
            }
        } else if (C <= 12224) {
            const size_t smem_bytes = static_cast<size_t>(C) * sizeof(float);
            fused_rmsnorm_smem_vec4_kernel<512><<<N, 512, smem_bytes>>>(
                x4, res4, weight4, out4, vec_cols, inv_c, eps);
        } else {
            fused_rmsnorm_twopass_vec4_kernel<512><<<N, 512>>>(
                x4, res4, weight4, out4, vec_cols, inv_c, eps);
        }
        return;
    }

    if (C <= 12224) {
        const size_t smem_bytes = static_cast<size_t>(C) * sizeof(float);
        if (C <= 256) {
            fused_rmsnorm_smem_scalar_kernel<128><<<N, 128, smem_bytes>>>(
                x, residual, weight, out, C, inv_c, eps);
        } else if (C <= 2048) {
            fused_rmsnorm_smem_scalar_kernel<256><<<N, 256, smem_bytes>>>(
                x, residual, weight, out, C, inv_c, eps);
        } else {
            fused_rmsnorm_smem_scalar_kernel<512><<<N, 512, smem_bytes>>>(
                x, residual, weight, out, C, inv_c, eps);
        }
    } else {
        fused_rmsnorm_twopass_scalar_kernel<512><<<N, 512>>>(
            x, residual, weight, out, C, inv_c, eps);
    }
}
