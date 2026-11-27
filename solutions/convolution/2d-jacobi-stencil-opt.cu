#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ void store_streaming_float4(float4* ptr, const float4& v) {
#if __CUDA_ARCH__ >= 700
    asm volatile("st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w));
#else
    *ptr = v;
#endif
}

// ============================================================================
// Cache-Line-Friendly 2-Row x float4 (2 x 1024-Column) Stencil Kernel!
// - 256 threads/block: each thread computes 2 adjacent rows (r0, r1) x 1 float4
//   (4 columns) = 8 outputs per thread (2 x 1,024 tile per block)!
// - Reads ONLY 4 coalesced float4s (rows r0 - 1, r0, r1, r1 + 1) per 2 output rows
//   (32 KB contiguous L2/DRAM bursts per row -> hits 4 rows in L2 across adjacent
//   Y-blocks!), exchanges horizontal neighbors via shared memory `s_edge` +
//   warp shuffles (ZERO scalar global loads!), and writes outputs via
//   `st.global.cs.v4.f32` (ZERO L2 write-allocate read!).
// ============================================================================
__global__ __launch_bounds__(kThreads, 4) void jacobi_2d_2row_vec4_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols,
    int w4) {
    __shared__ float s_left_w[2][kThreads + 2];   // stores .w shifted right by +1
    __shared__ float s_right_x[2][kThreads + 2];  // stores .x shifted left by -1

    const int tid = threadIdx.x;
    const int c4 = static_cast<int>(blockIdx.x) * kThreads + tid;
    const int r0 = static_cast<int>(blockIdx.y) * 2;
    const int r1 = r0 + 1;

    const float4* __restrict__ in4 = reinterpret_cast<const float4*>(input);
    float4* __restrict__ out4 = reinterpret_cast<float4*>(output);
    const size_t sw4 = static_cast<size_t>(w4);
    const size_t scols = static_cast<size_t>(cols);
    const bool valid_c = (c4 < w4);

    float4 v_m1 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 v_0 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 v_1 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 v_2 = make_float4(0.f, 0.f, 0.f, 0.f);

    if (valid_c) {
        if (r0 > 0 && r0 < rows) {
            v_m1 = __ldg(&in4[static_cast<size_t>(r0 - 1) * sw4 + static_cast<size_t>(c4)]);
        }
        if (r0 < rows) {
            v_0 = __ldg(&in4[static_cast<size_t>(r0) * sw4 + static_cast<size_t>(c4)]);
        }
        if (r1 < rows) {
            v_1 = __ldg(&in4[static_cast<size_t>(r1) * sw4 + static_cast<size_t>(c4)]);
        }
        if (r1 + 1 < rows) {
            v_2 = __ldg(&in4[static_cast<size_t>(r1 + 1) * sw4 + static_cast<size_t>(c4)]);
        }
    }

    s_left_w[0][tid + 1] = v_0.w;
    s_right_x[0][tid] = v_0.x;
    s_left_w[1][tid + 1] = v_1.w;
    s_right_x[1][tid] = v_1.x;

    if (tid == 0) {
        const int left_col = (c4 << 2) - 1;
        s_left_w[0][0] = (c4 > 0 && r0 < rows)
                             ? __ldg(&input[static_cast<size_t>(r0) * scols + left_col])
                             : 0.0f;
        s_left_w[1][0] = (c4 > 0 && r1 < rows)
                             ? __ldg(&input[static_cast<size_t>(r1) * scols + left_col])
                             : 0.0f;
    }
    if (tid == kThreads - 1 || c4 == w4 - 1) {
        const int right_col = (c4 << 2) + 4;
        if (valid_c) {
            s_right_x[0][tid + 1] = (right_col < cols && r0 < rows)
                                        ? __ldg(&input[static_cast<size_t>(r0) * scols + right_col])
                                        : 0.0f;
            s_right_x[1][tid + 1] = (right_col < cols && r1 < rows)
                                        ? __ldg(&input[static_cast<size_t>(r1) * scols + right_col])
                                        : 0.0f;
        }
    }
    __syncthreads();

    if (!valid_c) return;

    // Row r0
    if (r0 < rows) {
        float4 out0;
        if (r0 == 0 || r0 == rows - 1) {
            out0 = v_0;
        } else {
            const float l0 = s_left_w[0][tid];
            const float rr0 = s_right_x[0][tid + 1];
            out0.x = (c4 == 0) ? v_0.x : 0.25f * ((v_m1.x + v_1.x) + (l0 + v_0.y));
            out0.y = 0.25f * ((v_m1.y + v_1.y) + (v_0.x + v_0.z));
            out0.z = 0.25f * ((v_m1.z + v_1.z) + (v_0.y + v_0.w));
            out0.w = (c4 == w4 - 1) ? v_0.w : 0.25f * ((v_m1.w + v_1.w) + (v_0.z + rr0));
        }
        store_streaming_float4(&out4[static_cast<size_t>(r0) * sw4 + static_cast<size_t>(c4)], out0);
    }

    // Row r1
    if (r1 < rows) {
        float4 out1;
        if (r1 == rows - 1) {
            out1 = v_1;
        } else {
            const float l1 = s_left_w[1][tid];
            const float rr1 = s_right_x[1][tid + 1];
            out1.x = (c4 == 0) ? v_1.x : 0.25f * ((v_0.x + v_2.x) + (l1 + v_1.y));
            out1.y = 0.25f * ((v_0.y + v_2.y) + (v_1.x + v_1.z));
            out1.z = 0.25f * ((v_0.z + v_2.z) + (v_1.y + v_1.w));
            out1.w = (c4 == w4 - 1) ? v_1.w : 0.25f * ((v_0.w + v_2.w) + (v_1.z + rr1));
        }
        store_streaming_float4(&out4[static_cast<size_t>(r1) * sw4 + static_cast<size_t>(c4)], out1);
    }
}

__global__ __launch_bounds__(256) void jacobi_2d_general_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols) {
    __shared__ float s_tile[10][36];

    const int tx = threadIdx.x & 31;
    const int ty = threadIdx.x >> 5;
    const int j = static_cast<int>(blockIdx.x) * 32 + tx;
    const int i = static_cast<int>(blockIdx.y) * 8 + ty;

    const size_t stride_cols = static_cast<size_t>(cols);

    if (i < rows && j < cols) {
        s_tile[ty + 1][tx + 1] =
            __ldg(&input[static_cast<size_t>(i) * stride_cols + static_cast<size_t>(j)]);
    }
    if (ty == 0 && i > 0 && i <= rows && j < cols) {
        s_tile[0][tx + 1] =
            __ldg(&input[static_cast<size_t>(i - 1) * stride_cols + static_cast<size_t>(j)]);
    }
    if (ty == 7 && (i + 1) < rows && j < cols) {
        s_tile[9][tx + 1] =
            __ldg(&input[static_cast<size_t>(i + 1) * stride_cols + static_cast<size_t>(j)]);
    }
    if (tx == 0 && j > 0 && i < rows) {
        s_tile[ty + 1][0] =
            __ldg(&input[static_cast<size_t>(i) * stride_cols + static_cast<size_t>(j - 1)]);
    }
    if (tx == 31 && (j + 1) < cols && i < rows) {
        s_tile[ty + 1][33] =
            __ldg(&input[static_cast<size_t>(i) * stride_cols + static_cast<size_t>(j + 1)]);
    }
    __syncthreads();

    if (i >= rows || j >= cols) {
        return;
    }

    const size_t idx = static_cast<size_t>(i) * stride_cols + static_cast<size_t>(j);
    if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
        output[idx] = s_tile[ty + 1][tx + 1];
        return;
    }

    const float top = s_tile[ty][tx + 1];
    const float bot = s_tile[ty + 2][tx + 1];
    const float left = s_tile[ty + 1][tx];
    const float right = s_tile[ty + 1][tx + 2];
    output[idx] = 0.25f * ((top + bot) + (left + right));
}

}  // namespace

extern "C" void solve(const float* input, float* output, int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        return;
    }

    const bool aligned_16 =
        (((reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(output)) & 15u) == 0u);

    if (aligned_16 && (cols & 3) == 0 && rows >= 3 && cols >= 8) {
        const int w4 = cols >> 2;
        const dim3 grid(static_cast<unsigned>((w4 + kThreads - 1) / kThreads),
                        static_cast<unsigned>((rows + 1) / 2));
        jacobi_2d_2row_vec4_kernel<<<grid, kThreads>>>(input, output, rows, cols, w4);
        return;
    }

    const dim3 grid(static_cast<unsigned>((cols + 31) / 32),
                    static_cast<unsigned>((rows + 7) / 8));
    jacobi_2d_general_kernel<<<grid, 256>>>(input, output, rows, cols);
}
