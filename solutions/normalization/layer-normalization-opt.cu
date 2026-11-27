#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kWarp = 32;
constexpr unsigned kFull = 0xffffffffu;

__device__ __forceinline__ float warp_allreduce_sum(float v) {
    #pragma unroll
    for (int mask = kWarp / 2; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(kFull, v, mask);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = kWarp / 2; off > 0; off >>= 1) {
        v += __shfl_down_sync(kFull, v, off);
    }
    return v;
}

template <int kThreads>
__device__ __forceinline__ float block_allreduce_sum(float v) {
    constexpr int kWarps = kThreads / kWarp;
    __shared__ float s_wsums[kWarps];
    __shared__ float s_total;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) s_wsums[warp] = v;
    __syncthreads();

    if (warp == 0) {
        float bsum = lane < kWarps ? s_wsums[lane] : 0.0f;
        bsum = warp_reduce_sum(bsum);
        if (lane == 0) s_total = bsum;
    }
    __syncthreads();
    return s_total;
}

// ============================================================================
// Dedicated zero-branch fast path for official benchmark: N = 65,536, C = 512.
// 1 Warp (32 threads) = 1 Row! 4 Warps (128 threads) per Block = 4 Rows/Block!
// Zero __syncthreads(), zero shared memory, <= 44 registers, 100% register-cached single-pass!
// ============================================================================
__global__ __launch_bounds__(128, 8) void layernorm_warp_exact_512_kernel(
    const float4* __restrict__ in4,
    const float4* __restrict__ w4,
    const float4* __restrict__ b4,
    float4* __restrict__ out4,
    int N,
    float eps) {
    constexpr float kInv512 = 1.0f / 512.0f;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = (blockIdx.x << 2) + warp;
    if (row >= N) return;

    const unsigned row_base = static_cast<unsigned>(row) << 7;  // row * 128 float4s
    float4 x0 = __ldg(&in4[row_base + lane]);
    float4 x1 = __ldg(&in4[row_base + lane + 32]);
    float4 x2 = __ldg(&in4[row_base + lane + 64]);
    float4 x3 = __ldg(&in4[row_base + lane + 96]);

    float sum = (x0.x + x0.y) + (x0.z + x0.w) +
                (x1.x + x1.y) + (x1.z + x1.w) +
                (x2.x + x2.y) + (x2.z + x2.w) +
                (x3.x + x3.y) + (x3.z + x3.w);
    const float mean = warp_allreduce_sum(sum) * kInv512;

    x0.x -= mean; x0.y -= mean; x0.z -= mean; x0.w -= mean;
    x1.x -= mean; x1.y -= mean; x1.z -= mean; x1.w -= mean;
    x2.x -= mean; x2.y -= mean; x2.z -= mean; x2.w -= mean;
    x3.x -= mean; x3.y -= mean; x3.z -= mean; x3.w -= mean;

    float var_sum = fmaf(x0.x, x0.x, x0.y * x0.y) + fmaf(x0.z, x0.z, x0.w * x0.w) +
                    fmaf(x1.x, x1.x, x1.y * x1.y) + fmaf(x1.z, x1.z, x1.w * x1.w) +
                    fmaf(x2.x, x2.x, x2.y * x2.y) + fmaf(x2.z, x2.z, x2.w * x2.w) +
                    fmaf(x3.x, x3.x, x3.y * x3.y) + fmaf(x3.z, x3.z, x3.w * x3.w);
    const float inv_std = rsqrtf(warp_allreduce_sum(var_sum) * kInv512 + eps);

    {
        const float4 g0 = __ldg(&w4[lane]), h0 = __ldg(&b4[lane]);
        out4[row_base + lane] = make_float4(
            fmaf(x0.x * inv_std, g0.x, h0.x), fmaf(x0.y * inv_std, g0.y, h0.y),
            fmaf(x0.z * inv_std, g0.z, h0.z), fmaf(x0.w * inv_std, g0.w, h0.w));
    }
    {
        const float4 g1 = __ldg(&w4[lane + 32]), h1 = __ldg(&b4[lane + 32]);
        out4[row_base + lane + 32] = make_float4(
            fmaf(x1.x * inv_std, g1.x, h1.x), fmaf(x1.y * inv_std, g1.y, h1.y),
            fmaf(x1.z * inv_std, g1.z, h1.z), fmaf(x1.w * inv_std, g1.w, h1.w));
    }
    {
        const float4 g2 = __ldg(&w4[lane + 64]), h2 = __ldg(&b4[lane + 64]);
        out4[row_base + lane + 64] = make_float4(
            fmaf(x2.x * inv_std, g2.x, h2.x), fmaf(x2.y * inv_std, g2.y, h2.y),
            fmaf(x2.z * inv_std, g2.z, h2.z), fmaf(x2.w * inv_std, g2.w, h2.w));
    }
    {
        const float4 g3 = __ldg(&w4[lane + 96]), h3 = __ldg(&b4[lane + 96]);
        out4[row_base + lane + 96] = make_float4(
            fmaf(x3.x * inv_std, g3.x, h3.x), fmaf(x3.y * inv_std, g3.y, h3.y),
            fmaf(x3.z * inv_std, g3.z, h3.z), fmaf(x3.w * inv_std, g3.w, h3.w));
    }
}

// Warp-per-row general aligned float4 kernel for C <= 512 (kVecs = 1, 2, 4):
template <int kVecs>
__global__ __launch_bounds__(256, 4) void layernorm_warp_vec4_kernel(
    const float4* __restrict__ in4,
    const float4* __restrict__ w4,
    const float4* __restrict__ b4,
    float4* __restrict__ out4,
    int N,
    int vec_cols,
    float inv_c,
    float eps) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = (blockIdx.x << 3) + warp;
    if (row >= N) return;

    const unsigned row_base = static_cast<unsigned>(row) * static_cast<unsigned>(vec_cols);
    float4 x_reg[kVecs];
    float sum = 0.0f;

    #pragma unroll
    for (int v = 0; v < kVecs; ++v) {
        const int c4 = v * 32 + lane;
        float4 xv = make_float4(0.f, 0.f, 0.f, 0.f);
        if (c4 < vec_cols) {
            xv = __ldg(&in4[row_base + c4]);
            sum += (xv.x + xv.y) + (xv.z + xv.w);
        }
        x_reg[v] = xv;
    }
    const float mean = warp_allreduce_sum(sum) * inv_c;

    float var_sum = 0.0f;
    #pragma unroll
    for (int v = 0; v < kVecs; ++v) {
        const int c4 = v * 32 + lane;
        if (c4 < vec_cols) {
            const float dx = x_reg[v].x - mean;
            const float dy = x_reg[v].y - mean;
            const float dz = x_reg[v].z - mean;
            const float dw = x_reg[v].w - mean;
            var_sum += fmaf(dx, dx, dy * dy) + fmaf(dz, dz, dw * dw);
        }
    }
    const float inv_std = rsqrtf(warp_allreduce_sum(var_sum) * inv_c + eps);

    #pragma unroll
    for (int v = 0; v < kVecs; ++v) {
        const int c4 = v * 32 + lane;
        if (c4 < vec_cols) {
            const float4 gv = __ldg(&w4[c4]);
            const float4 hv = __ldg(&b4[c4]);
            out4[row_base + c4] = make_float4(
                fmaf((x_reg[v].x - mean) * inv_std, gv.x, hv.x),
                fmaf((x_reg[v].y - mean) * inv_std, gv.y, hv.y),
                fmaf((x_reg[v].z - mean) * inv_std, gv.z, hv.z),
                fmaf((x_reg[v].w - mean) * inv_std, gv.w, hv.w));
        }
    }
}

// Block-per-row register-cached float4 kernel for 512 < C <= 4096 (256 threads, kVecs = 1, 2, 4):
template <int kVecs>
__global__ __launch_bounds__(256, 4) void layernorm_block_vec4_kernel(
    const float4* __restrict__ in4,
    const float4* __restrict__ w4,
    const float4* __restrict__ b4,
    float4* __restrict__ out4,
    int vec_cols,
    float inv_c,
    float eps) {
    const int tid = threadIdx.x;
    const unsigned row_base = static_cast<unsigned>(blockIdx.x) * static_cast<unsigned>(vec_cols);
    float4 x_reg[kVecs];
    float sum = 0.0f;

    #pragma unroll
    for (int v = 0; v < kVecs; ++v) {
        const int c4 = v * 256 + tid;
        float4 xv = make_float4(0.f, 0.f, 0.f, 0.f);
        if (c4 < vec_cols) {
            xv = __ldg(&in4[row_base + c4]);
            sum += (xv.x + xv.y) + (xv.z + xv.w);
        }
        x_reg[v] = xv;
    }
    const float mean = block_allreduce_sum<256>(sum) * inv_c;

    float var_sum = 0.0f;
    #pragma unroll
    for (int v = 0; v < kVecs; ++v) {
        const int c4 = v * 256 + tid;
        if (c4 < vec_cols) {
            const float dx = x_reg[v].x - mean;
            const float dy = x_reg[v].y - mean;
            const float dz = x_reg[v].z - mean;
            const float dw = x_reg[v].w - mean;
            var_sum += fmaf(dx, dx, dy * dy) + fmaf(dz, dz, dw * dw);
        }
    }
    const float inv_std = rsqrtf(block_allreduce_sum<256>(var_sum) * inv_c + eps);

    #pragma unroll
    for (int v = 0; v < kVecs; ++v) {
        const int c4 = v * 256 + tid;
        if (c4 < vec_cols) {
            const float4 gv = __ldg(&w4[c4]);
            const float4 hv = __ldg(&b4[c4]);
            out4[row_base + c4] = make_float4(
                fmaf((x_reg[v].x - mean) * inv_std, gv.x, hv.x),
                fmaf((x_reg[v].y - mean) * inv_std, gv.y, hv.y),
                fmaf((x_reg[v].z - mean) * inv_std, gv.z, hv.z),
                fmaf((x_reg[v].w - mean) * inv_std, gv.w, hv.w));
        }
    }
}

// Single-pass shared-memory scalar kernel for unaligned pointers or C % 4 != 0 (C <= 4096 = 16 KiB):
template <int kThreads>
__global__ __launch_bounds__(kThreads) void layernorm_smem_scalar_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int C,
    float inv_c,
    float eps) {
    extern __shared__ float s_x[];
    const int tid = threadIdx.x;
    const size_t row_base = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(C);
    const float* row_in = input + row_base;
    float* row_out = output + row_base;

    float sum = 0.0f;
    for (int j = tid; j < C; j += kThreads) {
        const float x = __ldg(&row_in[j]);
        s_x[j] = x;
        sum += x;
    }
    const float mean = block_allreduce_sum<kThreads>(sum) * inv_c;

    float var_sum = 0.0f;
    for (int j = tid; j < C; j += kThreads) {
        const float d = s_x[j] - mean;
        var_sum = fmaf(d, d, var_sum);
    }
    const float inv_std = rsqrtf(block_allreduce_sum<kThreads>(var_sum) * inv_c + eps);

    for (int j = tid; j < C; j += kThreads) {
        row_out[j] = fmaf((s_x[j] - mean) * inv_std, __ldg(&weight[j]), __ldg(&bias[j]));
    }
}

// Universal fallback if C > 4096 and unaligned:
__global__ __launch_bounds__(256) void layernorm_twopass_scalar_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int C,
    float inv_c,
    float eps) {
    const int tid = threadIdx.x;
    const size_t row_base = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(C);
    const float* row_in = input + row_base;
    float* row_out = output + row_base;

    float sum = 0.0f;
    for (int j = tid; j < C; j += 256) sum += __ldg(&row_in[j]);
    const float mean = block_allreduce_sum<256>(sum) * inv_c;

    float var_sum = 0.0f;
    for (int j = tid; j < C; j += 256) {
        const float d = __ldg(&row_in[j]) - mean;
        var_sum = fmaf(d, d, var_sum);
    }
    const float inv_std = rsqrtf(block_allreduce_sum<256>(var_sum) * inv_c + eps);

    for (int j = tid; j < C; j += 256) {
        row_out[j] = fmaf((__ldg(&row_in[j]) - mean) * inv_std, __ldg(&weight[j]), __ldg(&bias[j]));
    }
}

}  // namespace

extern "C" void solve(const float* input, const float* weight, const float* bias,
                      float* output, int N, int C, float eps) {
    if (N <= 0 || C <= 0) return;

    const float inv_c = 1.0f / static_cast<float>(C);
    const uintptr_t mask = reinterpret_cast<uintptr_t>(input) |
                           reinterpret_cast<uintptr_t>(weight) |
                           reinterpret_cast<uintptr_t>(bias) |
                           reinterpret_cast<uintptr_t>(output);

    if ((C & 3) == 0 && (mask & 15u) == 0u && C <= 4096) {
        const int vec_cols = C >> 2;
        const float4* in4 = reinterpret_cast<const float4*>(input);
        const float4* w4 = reinterpret_cast<const float4*>(weight);
        const float4* b4 = reinterpret_cast<const float4*>(bias);
        float4* out4 = reinterpret_cast<float4*>(output);

        if (C == 512) {
            layernorm_warp_exact_512_kernel<<<(N + 3) / 4, 128>>>(in4, w4, b4, out4, N, eps);
        } else if (C <= 128) {
            layernorm_warp_vec4_kernel<1><<<(N + 7) / 8, 256>>>(in4, w4, b4, out4, N, vec_cols, inv_c, eps);
        } else if (C <= 256) {
            layernorm_warp_vec4_kernel<2><<<(N + 7) / 8, 256>>>(in4, w4, b4, out4, N, vec_cols, inv_c, eps);
        } else if (C < 512) {
            layernorm_warp_vec4_kernel<4><<<(N + 7) / 8, 256>>>(in4, w4, b4, out4, N, vec_cols, inv_c, eps);
        } else if (C <= 1024) {
            layernorm_block_vec4_kernel<1><<<N, 256>>>(in4, w4, b4, out4, vec_cols, inv_c, eps);
        } else if (C <= 2048) {
            layernorm_block_vec4_kernel<2><<<N, 256>>>(in4, w4, b4, out4, vec_cols, inv_c, eps);
        } else {
            layernorm_block_vec4_kernel<4><<<N, 256>>>(in4, w4, b4, out4, vec_cols, inv_c, eps);
        }
        return;
    }

    if (C <= 4096) {
        const size_t smem = static_cast<size_t>(C) * sizeof(float);
        if (C <= 128) {
            layernorm_smem_scalar_kernel<64><<<N, 64, smem>>>(input, weight, bias, output, C, inv_c, eps);
        } else if (C <= 512) {
            layernorm_smem_scalar_kernel<128><<<N, 128, smem>>>(input, weight, bias, output, C, inv_c, eps);
        } else {
            layernorm_smem_scalar_kernel<256><<<N, 256, smem>>>(input, weight, bias, output, C, inv_c, eps);
        }
    } else {
        layernorm_twopass_scalar_kernel<<<N, 256>>>(input, weight, bias, output, C, inv_c, eps);
    }
}
