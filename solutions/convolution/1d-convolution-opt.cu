#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kBlockThreads = 256;
constexpr int kMaxSmemKernel = 2048;

__device__ __forceinline__ void store_cs_v4(float* ptr, float4 val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    asm volatile("st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(ptr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w)
                 : "memory");
#else
    *reinterpret_cast<float4*>(ptr) = val;
#endif
}

// ============================================================================
// Primary Fast Path (`1 <= kernel_size <= 2,048` — covers official `1,500,000, K = 2,047`!):
//   - Zero `cudaMemcpyToSymbol` host/driver overhead: each block cooperatively
//     loads `kernel` into 16B-aligned shared memory `s_k4[512]` (`8 KiB`) and pads
//     up to `k_padded = ceil4(kernel_size)` with `0.0f` so the inner loop runs
//     pure `float4` (`LDS.128`) steps with zero scalar remainder branches.
//   - Warp-coalesced `kGroups` x `float4` outputs per thread (`kGroups = 4` -> `16`
//     outputs/thread, `TILE_OUT = 4,096`, `32 KiB` total smem, `2` blocks/SM = `512`
//     threads/SM on T4; `kGroups = 1` -> `4` outputs/thread, `TILE_OUT = 1,024` for
//     smaller `output_size` to maximize SM parallelism).
//   - Every `s_in4[g * 256 + tid + kv_idx + 1]` read is stride-1 `float4` across
//     consecutive threads (`0..7`), guaranteeing 100% bank-conflict-free `LDS.128`,
//     and outputs write directly via coalesced `st.global.cs.v4.f32` with zero
//     output-stage `__syncthreads()`.
// ============================================================================
template <int kGroups>
__global__ __launch_bounds__(kBlockThreads) void conv1d_smem_vec4_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_size,
    int kernel_size,
    int output_size) {
    constexpr int kTileOut = kGroups * kBlockThreads * 4;
    constexpr int kMaxInVecs = (kTileOut + kMaxSmemKernel) / 4;

    __shared__ float4 s_k4[kMaxSmemKernel / 4];
    __shared__ float4 s_in4[kMaxInVecs];

    float* s_k = reinterpret_cast<float*>(s_k4);
    float* s_in = reinterpret_cast<float*>(s_in4);

    const int tid = static_cast<int>(threadIdx.x);
    const int block_out_base = static_cast<int>(blockIdx.x) * kTileOut;
    const int k_padded = (kernel_size + 3) & ~3;
    const int k_vecs = k_padded >> 2;

    // 1. Load `kernel` into shared memory, zero-padding up to `k_padded`
    if (((reinterpret_cast<uintptr_t>(kernel) & 15u) == 0u)) {
        const int full_k_vecs = kernel_size >> 2;
        for (int v = tid; v < k_vecs; v += kBlockThreads) {
            if (v < full_k_vecs) {
                s_k4[v] = __ldg(reinterpret_cast<const float4*>(kernel) + v);
            } else {
                const int base = v << 2;
                s_k4[v] = make_float4(
                    (base + 0 < kernel_size) ? __ldg(&kernel[base + 0]) : 0.0f,
                    (base + 1 < kernel_size) ? __ldg(&kernel[base + 1]) : 0.0f,
                    (base + 2 < kernel_size) ? __ldg(&kernel[base + 2]) : 0.0f,
                    (base + 3 < kernel_size) ? __ldg(&kernel[base + 3]) : 0.0f);
            }
        }
    } else {
        for (int idx = tid; idx < k_padded; idx += kBlockThreads) {
            s_k[idx] = (idx < kernel_size) ? __ldg(&kernel[idx]) : 0.0f;
        }
    }

    // 2. Load `input[block_out_base .. block_out_base + kTileOut + k_padded - 1]` + halo
    const int in_elems = kTileOut + k_padded;
    const int in_vecs = in_elems >> 2;
    if (((reinterpret_cast<uintptr_t>(input) & 15u) == 0u)) {
        // `block_out_base` is a multiple of `1,024` floats (`4,096` bytes), so `input + block_out_base` is 16B-aligned!
        for (int v = tid; v < in_vecs; v += kBlockThreads) {
            const int g_idx = block_out_base + (v << 2);
            if (g_idx + 4 <= input_size) {
                s_in4[v] = __ldg(reinterpret_cast<const float4*>(input + g_idx));
            } else {
                s_in4[v] = make_float4(
                    (g_idx + 0 < input_size) ? __ldg(&input[g_idx + 0]) : 0.0f,
                    (g_idx + 1 < input_size) ? __ldg(&input[g_idx + 1]) : 0.0f,
                    (g_idx + 2 < input_size) ? __ldg(&input[g_idx + 2]) : 0.0f,
                    (g_idx + 3 < input_size) ? __ldg(&input[g_idx + 3]) : 0.0f);
            }
        }
    } else {
        for (int idx = tid; idx < in_elems; idx += kBlockThreads) {
            const int g_idx = block_out_base + idx;
            s_in[idx] = (g_idx < input_size) ? __ldg(&input[g_idx]) : 0.0f;
        }
    }
    __syncthreads();

    // 3. Conflict-free `LDS.128` register sliding window across `k_vecs` (`4` taps/step)
    float acc[kGroups][4];
    float4 v0[kGroups];
    #pragma unroll
    for (int g = 0; g < kGroups; ++g) {
        acc[g][0] = 0.0f;
        acc[g][1] = 0.0f;
        acc[g][2] = 0.0f;
        acc[g][3] = 0.0f;
        v0[g] = s_in4[g * kBlockThreads + tid];
    }

    #pragma unroll 4
    for (int kv_idx = 0; kv_idx < k_vecs; ++kv_idx) {
        const float4 kv = s_k4[kv_idx];
        #pragma unroll
        for (int g = 0; g < kGroups; ++g) {
            const float4 v1 = s_in4[g * kBlockThreads + tid + kv_idx + 1];

            acc[g][0] = fmaf(v0[g].x, kv.x, acc[g][0]);
            acc[g][0] = fmaf(v0[g].y, kv.y, acc[g][0]);
            acc[g][0] = fmaf(v0[g].z, kv.z, acc[g][0]);
            acc[g][0] = fmaf(v0[g].w, kv.w, acc[g][0]);

            acc[g][1] = fmaf(v0[g].y, kv.x, acc[g][1]);
            acc[g][1] = fmaf(v0[g].z, kv.y, acc[g][1]);
            acc[g][1] = fmaf(v0[g].w, kv.z, acc[g][1]);
            acc[g][1] = fmaf(v1.x,    kv.w, acc[g][1]);

            acc[g][2] = fmaf(v0[g].z, kv.x, acc[g][2]);
            acc[g][2] = fmaf(v0[g].w, kv.y, acc[g][2]);
            acc[g][2] = fmaf(v1.x,    kv.z, acc[g][2]);
            acc[g][2] = fmaf(v1.y,    kv.w, acc[g][2]);

            acc[g][3] = fmaf(v0[g].w, kv.x, acc[g][3]);
            acc[g][3] = fmaf(v1.x,    kv.y, acc[g][3]);
            acc[g][3] = fmaf(v1.y,    kv.z, acc[g][3]);
            acc[g][3] = fmaf(v1.z,    kv.w, acc[g][3]);

            v0[g] = v1;
        }
    }

    // 4. Coalesced cache-streaming `st.global.cs.v4.f32` global writeback
    const bool out_aligned = ((reinterpret_cast<uintptr_t>(output) & 15u) == 0u);
    #pragma unroll
    for (int g = 0; g < kGroups; ++g) {
        const int out_idx = block_out_base + g * (kBlockThreads * 4) + tid * 4;
        if (out_aligned && (out_idx + 4 <= output_size)) {
            store_cs_v4(output + out_idx,
                        make_float4(acc[g][0], acc[g][1], acc[g][2], acc[g][3]));
        } else {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                if (out_idx + c < output_size) {
                    output[out_idx + c] = acc[g][c];
                }
            }
        }
    }
}

// ============================================================================
// General Tiled Fallback (`kernel_size > 2,048`):
//   Tiles `kernel` in chunks of `2,048` taps using shared memory `s_k4` + `s_in4`.
// ============================================================================
__global__ __launch_bounds__(kBlockThreads) void conv1d_large_k_tiled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_size,
    int kernel_size,
    int output_size) {
    constexpr int kTileOut = kBlockThreads * 4;  // 1,024 outputs/block
    constexpr int kMaxInVecs = (kTileOut + kMaxSmemKernel) / 4;

    __shared__ float4 s_k4[kMaxSmemKernel / 4];
    __shared__ float4 s_in4[kMaxInVecs];

    float* s_k = reinterpret_cast<float*>(s_k4);
    float* s_in = reinterpret_cast<float*>(s_in4);

    const int tid = static_cast<int>(threadIdx.x);
    const int block_out_base = static_cast<int>(blockIdx.x) * kTileOut;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (int k_off = 0; k_off < kernel_size; k_off += kMaxSmemKernel) {
        const int chunk_k = ((kernel_size - k_off) < kMaxSmemKernel)
                                ? (kernel_size - k_off)
                                : kMaxSmemKernel;
        const int chunk_padded = (chunk_k + 3) & ~3;
        const int chunk_vecs = chunk_padded >> 2;

        for (int idx = tid; idx < chunk_padded; idx += kBlockThreads) {
            s_k[idx] = (idx < chunk_k) ? __ldg(&kernel[k_off + idx]) : 0.0f;
        }

        const int in_elems = kTileOut + chunk_padded;
        const int in_base = block_out_base + k_off;
        for (int idx = tid; idx < in_elems; idx += kBlockThreads) {
            const int g_idx = in_base + idx;
            s_in[idx] = (g_idx < input_size) ? __ldg(&input[g_idx]) : 0.0f;
        }
        __syncthreads();

        float4 v0 = s_in4[tid];
        #pragma unroll 4
        for (int kv_idx = 0; kv_idx < chunk_vecs; ++kv_idx) {
            const float4 kv = s_k4[kv_idx];
            const float4 v1 = s_in4[tid + kv_idx + 1];

            acc0 = fmaf(v0.x, kv.x, acc0);
            acc0 = fmaf(v0.y, kv.y, acc0);
            acc0 = fmaf(v0.z, kv.z, acc0);
            acc0 = fmaf(v0.w, kv.w, acc0);

            acc1 = fmaf(v0.y, kv.x, acc1);
            acc1 = fmaf(v0.z, kv.y, acc1);
            acc1 = fmaf(v0.w, kv.z, acc1);
            acc1 = fmaf(v1.x, kv.w, acc1);

            acc2 = fmaf(v0.z, kv.x, acc2);
            acc2 = fmaf(v0.w, kv.y, acc2);
            acc2 = fmaf(v1.x, kv.z, acc2);
            acc2 = fmaf(v1.y, kv.w, acc2);

            acc3 = fmaf(v0.w, kv.x, acc3);
            acc3 = fmaf(v1.x, kv.y, acc3);
            acc3 = fmaf(v1.y, kv.z, acc3);
            acc3 = fmaf(v1.z, kv.w, acc3);

            v0 = v1;
        }
        __syncthreads();
    }

    const int out_idx = block_out_base + tid * 4;
    if (((reinterpret_cast<uintptr_t>(output) & 15u) == 0u) && (out_idx + 4 <= output_size)) {
        store_cs_v4(output + out_idx, make_float4(acc0, acc1, acc2, acc3));
    } else {
        if (out_idx + 0 < output_size) output[out_idx + 0] = acc0;
        if (out_idx + 1 < output_size) output[out_idx + 1] = acc1;
        if (out_idx + 2 < output_size) output[out_idx + 2] = acc2;
        if (out_idx + 3 < output_size) output[out_idx + 3] = acc3;
    }
}

}  // namespace

extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_size, int kernel_size) {
    if (input_size <= 0 || kernel_size <= 0 || kernel_size > input_size) {
        return;
    }

    const int output_size = input_size - kernel_size + 1;

    if (kernel_size <= kMaxSmemKernel) {
        if (output_size >= 163840) {
            constexpr int kTileOut = 4 * kBlockThreads * 4;  // 4,096 outputs/block
            const unsigned int blocks =
                static_cast<unsigned int>((output_size + kTileOut - 1) / kTileOut);
            conv1d_smem_vec4_kernel<4><<<blocks, kBlockThreads>>>(
                input, kernel, output, input_size, kernel_size, output_size);
        } else {
            constexpr int kTileOut = 1 * kBlockThreads * 4;  // 1,024 outputs/block
            const unsigned int blocks =
                static_cast<unsigned int>((output_size + kTileOut - 1) / kTileOut);
            conv1d_smem_vec4_kernel<1><<<blocks, kBlockThreads>>>(
                input, kernel, output, input_size, kernel_size, output_size);
        }
    } else {
        constexpr int kTileOut = kBlockThreads * 4;
        const unsigned int blocks =
            static_cast<unsigned int>((output_size + kTileOut - 1) / kTileOut);
        conv1d_large_k_tiled_kernel<<<blocks, kBlockThreads>>>(
            input, kernel, output, input_size, kernel_size, output_size);
    }
}
