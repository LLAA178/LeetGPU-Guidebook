#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

// Block geometry for register-tiled shared-halo kernels:
// 128 threads = 16 x 8 (tx in 0..15, ty in 0..7).
// Each thread computes 2 rows x 4 cols (float4) -> block output tile is 16 rows x 64 cols!
// On 512 x 512, launches (512/64) x (512/16) = 8 x 32 = 256 blocks!
constexpr int kBlockX = 16;
constexpr int kBlockY = 8;
constexpr int kThreads = kBlockX * kBlockY;  // 128 threads
constexpr int kOutH = kBlockY * 2;           // 16 rows per block
constexpr int kOutW = kBlockX * 4;           // 64 cols per block

__device__ __forceinline__ void store_streaming_float4(float4* addr, float4 v) {
    asm volatile("st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(addr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
                 : "memory");
}

// ============================================================================
// Compile-Time Unrolled Shared-Halo + Shared-Kernel 2x4 Register-Tiled Kernel
// Covers symmetric odd K in {1, 3, 5, 7, 9, 11, 13, 15} (including official 7x7!).
// - ZERO host D2D / cudaMemcpyToSymbolAsync overhead: loads kernel[] into s_k[]
//   in the SAME single pre-barrier phase as the input halo tile!
// - Interior blocks take a fast branchless float4 store path; boundary blocks
//   handle arbitrary input_rows, input_cols, and misaligned pointers cleanly!
// ============================================================================
template <int KH, int KW>
__global__ __launch_bounds__(kThreads, 4) void gaussian_blur_tiled_fixed_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_rows,
    int input_cols,
    bool out_vec4_aligned) {
    constexpr int kPadH = KH / 2;
    constexpr int kPadW = KW / 2;
    constexpr int kSmemH = kOutH + KH - 1;
    constexpr int kSmemW = kOutW + KW - 1;
    // Pad stride to odd multiple of 4 + 1 (or >= kSmemW) for zero shared-memory bank conflicts:
    constexpr int kSmemStride = ((kSmemW + 3) & ~3) + 1;
    constexpr int kKSize = KH * KW;

    __shared__ float s_in[kSmemH][kSmemStride];
    __shared__ float s_k[kKSize];

    const int tid = threadIdx.y * kBlockX + threadIdx.x;
    const int base_r = blockIdx.y * kOutH;
    const int base_c = blockIdx.x * kOutW;

    // 1a. Cooperatively load filter weights into shared memory
    for (int idx = tid; idx < kKSize; idx += kThreads) {
        s_k[idx] = __ldg(&kernel[idx]);
    }

    // 1b. Cooperatively load (kSmemH x kSmemW) input halo tile with zero-padding
    const int start_r = base_r - kPadH;
    const int start_c = base_c - kPadW;
    constexpr int kTotalHalo = kSmemH * kSmemW;

    const bool interior_halo = (start_r >= 0) &&
                               (start_r + kSmemH <= input_rows) &&
                               (start_c >= 0) &&
                               (start_c + kSmemW <= input_cols);

    if (interior_halo) {
        #pragma unroll
        for (int idx = tid; idx < kTotalHalo; idx += kThreads) {
            const int sr = idx / kSmemW;
            const int sc = idx % kSmemW;
            const size_t g_idx = static_cast<size_t>(start_r + sr) * static_cast<size_t>(input_cols) +
                                 static_cast<size_t>(start_c + sc);
            s_in[sr][sc] = __ldg(&input[g_idx]);
        }
    } else {
        #pragma unroll
        for (int idx = tid; idx < kTotalHalo; idx += kThreads) {
            const int sr = idx / kSmemW;
            const int sc = idx % kSmemW;
            const int gr = start_r + sr;
            const int gc = start_c + sc;
            float val = 0.0f;
            if (gr >= 0 && gr < input_rows && gc >= 0 && gc < input_cols) {
                val = __ldg(&input[static_cast<size_t>(gr) * static_cast<size_t>(input_cols) + gc]);
            }
            s_in[sr][sc] = val;
        }
    }

    // ONLY ONE __syncthreads() in the entire kernel!
    __syncthreads();

    // 2. Each thread computes a 2 x 4 output sub-tile in registers:
    const int sr0 = threadIdx.y * 2;
    const int sc0 = threadIdx.x * 4;
    const int out_r0 = base_r + sr0;
    const int out_c0 = base_c + sc0;

    if (out_r0 >= input_rows || out_c0 >= input_cols) {
        return;
    }

    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;

    #pragma unroll
    for (int kr = 0; kr < KH; ++kr) {
        // Preload row0 and row1 first KW - 1 elements, sliding across kc = 0..KW-1:
        #pragma unroll
        for (int kc = 0; kc < KW; ++kc) {
            const float kv = s_k[kr * KW + kc];
            const float i00 = s_in[sr0 + kr][sc0 + kc + 0];
            const float i01 = s_in[sr0 + kr][sc0 + kc + 1];
            const float i02 = s_in[sr0 + kr][sc0 + kc + 2];
            const float i03 = s_in[sr0 + kr][sc0 + kc + 3];

            const float i10 = s_in[sr0 + 1 + kr][sc0 + kc + 0];
            const float i11 = s_in[sr0 + 1 + kr][sc0 + kc + 1];
            const float i12 = s_in[sr0 + 1 + kr][sc0 + kc + 2];
            const float i13 = s_in[sr0 + 1 + kr][sc0 + kc + 3];

            acc00 = fmaf(i00, kv, acc00);
            acc01 = fmaf(i01, kv, acc01);
            acc02 = fmaf(i02, kv, acc02);
            acc03 = fmaf(i03, kv, acc03);

            acc10 = fmaf(i10, kv, acc10);
            acc11 = fmaf(i11, kv, acc11);
            acc12 = fmaf(i12, kv, acc12);
            acc13 = fmaf(i13, kv, acc13);
        }
    }

    // 3. Coalesced float4 streaming store when aligned and full 4-pixel quad is in bounds
    if (out_vec4_aligned && (out_c0 + 4 <= input_cols)) {
        float4* out_row0_v4 = reinterpret_cast<float4*>(
            output + static_cast<size_t>(out_r0) * static_cast<size_t>(input_cols) + out_c0);
        store_streaming_float4(out_row0_v4, make_float4(acc00, acc01, acc02, acc03));

        if (out_r0 + 1 < input_rows) {
            float4* out_row1_v4 = reinterpret_cast<float4*>(
                output + static_cast<size_t>(out_r0 + 1) * static_cast<size_t>(input_cols) + out_c0);
            store_streaming_float4(out_row1_v4, make_float4(acc10, acc11, acc12, acc13));
        }
    } else {
        const size_t row0_off = static_cast<size_t>(out_r0) * static_cast<size_t>(input_cols) + out_c0;
        output[row0_off + 0] = acc00;
        if (out_c0 + 1 < input_cols) output[row0_off + 1] = acc01;
        if (out_c0 + 2 < input_cols) output[row0_off + 2] = acc02;
        if (out_c0 + 3 < input_cols) output[row0_off + 3] = acc03;

        if (out_r0 + 1 < input_rows) {
            const size_t row1_off = row0_off + static_cast<size_t>(input_cols);
            output[row1_off + 0] = acc10;
            if (out_c0 + 1 < input_cols) output[row1_off + 1] = acc11;
            if (out_c0 + 2 < input_cols) output[row1_off + 2] = acc12;
            if (out_c0 + 3 < input_cols) output[row1_off + 3] = acc13;
        }
    }
}

// ============================================================================
// General Shared-Halo + Shared-Kernel Tiled Kernel for Any Odd/Asymmetric KH, KW <= 25
// Handles asymmetric kernels (e.g. 3x7, 5x3) and large kernels up to 21x21 (<= 16 KiB smem)!
// ============================================================================
__global__ __launch_bounds__(kThreads, 3) void gaussian_blur_tiled_general_smem_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_rows,
    int input_cols,
    int kernel_rows,
    int kernel_cols,
    int out_rows,
    int out_cols,
    bool out_vec4_aligned) {
    // Supports KH <= 25, KW <= 25: max smemH = 16 + 24 = 40, smemStride = 89 -> 14.2 KiB + 2.5 KiB = 16.7 KiB!
    constexpr int kMaxSmemH = kOutH + 24;
    constexpr int kMaxSmemStride = 93;
    __shared__ float s_in[kMaxSmemH][kMaxSmemStride];
    __shared__ float s_k[625];

    const int tid = threadIdx.y * kBlockX + threadIdx.x;
    const int base_r = blockIdx.y * kOutH;
    const int base_c = blockIdx.x * kOutW;

    const int ksize = kernel_rows * kernel_cols;
    for (int idx = tid; idx < ksize; idx += kThreads) {
        s_k[idx] = __ldg(&kernel[idx]);
    }

    const int pad_h = kernel_rows / 2;
    const int pad_w = kernel_cols / 2;
    const int smem_h = kOutH + kernel_rows - 1;
    const int smem_w = kOutW + kernel_cols - 1;
    const int total_halo = smem_h * smem_w;
    const int start_r = base_r - pad_h;
    const int start_c = base_c - pad_w;

    for (int idx = tid; idx < total_halo; idx += kThreads) {
        const int sr = idx / smem_w;
        const int sc = idx % smem_w;
        const int gr = start_r + sr;
        const int gc = start_c + sc;
        float val = 0.0f;
        if (gr >= 0 && gr < input_rows && gc >= 0 && gc < input_cols) {
            val = __ldg(&input[static_cast<size_t>(gr) * static_cast<size_t>(input_cols) + gc]);
        }
        s_in[sr][sc] = val;
    }
    __syncthreads();

    const int sr0 = threadIdx.y * 2;
    const int sc0 = threadIdx.x * 4;
    const int out_r0 = base_r + sr0;
    const int out_c0 = base_c + sc0;
    if (out_r0 >= out_rows || out_c0 >= out_cols) {
        return;
    }

    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;

    for (int kr = 0; kr < kernel_rows; ++kr) {
        const int k_base = kr * kernel_cols;
        for (int kc = 0; kc < kernel_cols; ++kc) {
            const float kv = s_k[k_base + kc];
            acc00 = fmaf(s_in[sr0 + kr][sc0 + kc + 0], kv, acc00);
            acc01 = fmaf(s_in[sr0 + kr][sc0 + kc + 1], kv, acc01);
            acc02 = fmaf(s_in[sr0 + kr][sc0 + kc + 2], kv, acc02);
            acc03 = fmaf(s_in[sr0 + kr][sc0 + kc + 3], kv, acc03);

            acc10 = fmaf(s_in[sr0 + 1 + kr][sc0 + kc + 0], kv, acc10);
            acc11 = fmaf(s_in[sr0 + 1 + kr][sc0 + kc + 1], kv, acc11);
            acc12 = fmaf(s_in[sr0 + 1 + kr][sc0 + kc + 2], kv, acc12);
            acc13 = fmaf(s_in[sr0 + 1 + kr][sc0 + kc + 3], kv, acc13);
        }
    }

    if (out_vec4_aligned && (out_c0 + 4 <= out_cols)) {
        float4* r0_v4 = reinterpret_cast<float4*>(
            output + static_cast<size_t>(out_r0) * static_cast<size_t>(out_cols) + out_c0);
        store_streaming_float4(r0_v4, make_float4(acc00, acc01, acc02, acc03));
        if (out_r0 + 1 < out_rows) {
            float4* r1_v4 = reinterpret_cast<float4*>(
                output + static_cast<size_t>(out_r0 + 1) * static_cast<size_t>(out_cols) + out_c0);
            store_streaming_float4(r1_v4, make_float4(acc10, acc11, acc12, acc13));
        }
    } else {
        const size_t row0_off = static_cast<size_t>(out_r0) * static_cast<size_t>(out_cols) + out_c0;
        output[row0_off + 0] = acc00;
        if (out_c0 + 1 < out_cols) output[row0_off + 1] = acc01;
        if (out_c0 + 2 < out_cols) output[row0_off + 2] = acc02;
        if (out_c0 + 3 < out_cols) output[row0_off + 3] = acc03;

        if (out_r0 + 1 < out_rows) {
            const size_t row1_off = row0_off + static_cast<size_t>(out_cols);
            output[row1_off + 0] = acc10;
            if (out_c0 + 1 < out_cols) output[row1_off + 1] = acc11;
            if (out_c0 + 2 < out_cols) output[row1_off + 2] = acc12;
            if (out_c0 + 3 < out_cols) output[row1_off + 3] = acc13;
        }
    }
}

// Universal fallback if KH > 25 or KW > 25
__global__ void gaussian_blur_large_k_fallback_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_rows,
    int input_cols,
    int kernel_rows,
    int kernel_cols,
    int out_rows,
    int out_cols) {
    const int col = blockIdx.x * 16 + threadIdx.x;
    const int row = blockIdx.y * 16 + threadIdx.y;
    if (row >= out_rows || col >= out_cols) {
        return;
    }

    const int pad_h = kernel_rows / 2;
    const int pad_w = kernel_cols / 2;
    float sum = 0.0f;
    for (int kr = 0; kr < kernel_rows; ++kr) {
        const int r = row + kr - pad_h;
        if (r >= 0 && r < input_rows) {
            const size_t in_base = static_cast<size_t>(r) * input_cols;
            const int k_base = kr * kernel_cols;
            for (int kc = 0; kc < kernel_cols; ++kc) {
                const int c = col + kc - pad_w;
                if (c >= 0 && c < input_cols) {
                    sum = fmaf(__ldg(&input[in_base + c]), __ldg(&kernel[k_base + kc]), sum);
                }
            }
        }
    }
    output[static_cast<size_t>(row) * out_cols + col] = sum;
}

inline bool is_aligned_16(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 15u) == 0u;
}

}  // namespace

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
    if (input_rows <= 0 || input_cols <= 0 || kernel_rows <= 0 || kernel_cols <= 0) {
        return;
    }

    // Match PyTorch conv2d(..., padding=(kernel_rows // 2, kernel_cols // 2)) output dimensions:
    const int pad_h = kernel_rows / 2;
    const int pad_w = kernel_cols / 2;
    const int out_rows = input_rows + 2 * pad_h - kernel_rows + 1;
    const int out_cols = input_cols + 2 * pad_w - kernel_cols + 1;
    if (out_rows <= 0 || out_cols <= 0) {
        return;
    }

    const dim3 block(kBlockX, kBlockY);
    const dim3 grid(static_cast<unsigned>((out_cols + kOutW - 1) / kOutW),
                    static_cast<unsigned>((out_rows + kOutH - 1) / kOutH));
    const bool out_vec4_aligned = ((out_cols & 3) == 0) && is_aligned_16(output);

    if (kernel_rows == kernel_cols && (kernel_rows & 1) == 1) {
        switch (kernel_rows) {
            case 1:
                gaussian_blur_tiled_fixed_kernel<1, 1><<<grid, block>>>(
                    input, kernel, output, input_rows, input_cols, out_vec4_aligned);
                return;
            case 3:
                gaussian_blur_tiled_fixed_kernel<3, 3><<<grid, block>>>(
                    input, kernel, output, input_rows, input_cols, out_vec4_aligned);
                return;
            case 5:
                gaussian_blur_tiled_fixed_kernel<5, 5><<<grid, block>>>(
                    input, kernel, output, input_rows, input_cols, out_vec4_aligned);
                return;
            case 7:
                gaussian_blur_tiled_fixed_kernel<7, 7><<<grid, block>>>(
                    input, kernel, output, input_rows, input_cols, out_vec4_aligned);
                return;
            case 9:
                gaussian_blur_tiled_fixed_kernel<9, 9><<<grid, block>>>(
                    input, kernel, output, input_rows, input_cols, out_vec4_aligned);
                return;
            case 11:
                gaussian_blur_tiled_fixed_kernel<11, 11><<<grid, block>>>(
                    input, kernel, output, input_rows, input_cols, out_vec4_aligned);
                return;
            case 13:
                gaussian_blur_tiled_fixed_kernel<13, 13><<<grid, block>>>(
                    input, kernel, output, input_rows, input_cols, out_vec4_aligned);
                return;
            case 15:
                gaussian_blur_tiled_fixed_kernel<15, 15><<<grid, block>>>(
                    input, kernel, output, input_rows, input_cols, out_vec4_aligned);
                return;
            default:
                break;
        }
    }

    if (kernel_rows <= 25 && kernel_cols <= 25) {
        gaussian_blur_tiled_general_smem_kernel<<<grid, block>>>(
            input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols,
            out_rows, out_cols, out_vec4_aligned);
        return;
    }

    const dim3 fblock(16, 16);
    const dim3 fgrid(static_cast<unsigned>((out_cols + 15) / 16),
                     static_cast<unsigned>((out_rows + 15) / 16));
    gaussian_blur_large_k_fallback_kernel<<<fgrid, fblock>>>(
        input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols,
        out_rows, out_cols);
}
