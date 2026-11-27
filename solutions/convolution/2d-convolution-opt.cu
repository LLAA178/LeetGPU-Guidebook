#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

#ifndef CONV2D_BM
#define CONV2D_BM 64
#endif

static_assert(CONV2D_BM == 32 || CONV2D_BM == 64, "CONV2D_BM must be 32 or 64");

constexpr int kBN = 64;
constexpr int kBlockX = 16;                  // 16 * 4 = 64 output cols per block
constexpr int kBlockY = CONV2D_BM / 4;       // 8 (128 thr) for BM=32, 16 (256 thr) for BM=64
constexpr int kThreads = kBlockX * kBlockY;

// ============================================================================
// Compute-Bound 4x4 Register-Tiled + LDS.128 Shared-Memory Kernel for K in {3..15}
// On 3072 x 3072 with K = 15 (output 3058 x 3058 = 2.104G FMAs = 4.21 GFLOP):
// - Each thread computes a 4 x 4 output tile (16 accumulators in registers).
// - Uses 16B-aligned `float4` shared-memory loads (`LDS.128`) along each row `r + kr`
//   with odd-float4 stride (`kStrideVec4 * 4`, e.g. 21 * 4 = 84 floats) -> ZERO bank conflicts
//   and 12+ FFMAs per LDS.128 instruction!
// - Reuses `s_in` at the end of the kernel to stage the `BM x 64` output tile so global writes
//   to `output` (even for `out_cols = 3,058` which is 2 mod 4!) are 100% stride-1 coalesced!
// ============================================================================
template <int KH, int KW>
__global__ __launch_bounds__(kThreads, (CONV2D_BM == 32 ? 3 : 2)) void conv2d_valid_reg4x4_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_rows,
    int input_cols,
    int out_rows,
    int out_cols) {
    constexpr int kSmemH = CONV2D_BM + KH - 1;
    constexpr int kSmemW = kBN + KW - 1;
    // Number of float4s needed by tx=15 (base col4 = 15) reading kNumVecLoads float4s:
    // Per row, 4 outputs across KW taps need (KW + 3) floats = ((KW + 6) / 4) float4s!
    constexpr int kVecLoads = (KW + 6) / 4;
    constexpr int kMinVecCols = 15 + kVecLoads;
    // Odd count of float4s >= max((kSmemW + 3)/4, kMinVecCols) prevents cross-row bank conflicts
    // while keeping every row 16-byte aligned for LDS.128!
    constexpr int kRawVecCols = ((kSmemW + 3) / 4 > kMinVecCols) ? ((kSmemW + 3) / 4) : kMinVecCols;
    constexpr int kStrideVec4 = (kRawVecCols & 1) ? kRawVecCols : (kRawVecCols + 1);
    constexpr int kSmemStride = kStrideVec4 * 4;
    constexpr int kKSize = KH * KW;

    __shared__ __align__(16) float s_in[kSmemH][kSmemStride];
    __shared__ float s_k[kKSize];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kBlockX + tx;

    const int base_r = blockIdx.y * CONV2D_BM;
    const int base_c = blockIdx.x * kBN;

    // 1a. Load kernel weights into shared memory
    for (int idx = tid; idx < kKSize; idx += kThreads) {
        s_k[idx] = __ldg(&kernel[idx]);
    }

    // 1b. Warp-row coalesced load of (kSmemH x kSmemW) input tile (zero integer division/modulo!)
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    constexpr int kNumWarps = kThreads >> 5;
    const bool interior = (base_r + kSmemH <= input_rows) && (base_c + kSmemW <= input_cols);

    if (interior) {
        for (int sr = warp_id; sr < kSmemH; sr += kNumWarps) {
            const float* in_row =
                input + static_cast<size_t>(base_r + sr) * static_cast<size_t>(input_cols) + base_c;
            #pragma unroll
            for (int sc = lane; sc < kSmemW; sc += 32) {
                s_in[sr][sc] = __ldg(&in_row[sc]);
            }
        }
    } else {
        for (int sr = warp_id; sr < kSmemH; sr += kNumWarps) {
            const int gr = base_r + sr;
            const bool row_ok = (gr < input_rows);
            const float* in_row =
                row_ok ? (input + static_cast<size_t>(gr) * static_cast<size_t>(input_cols)) : nullptr;
            #pragma unroll
            for (int sc = lane; sc < kSmemStride; sc += 32) {
                const int gc = base_c + sc;
                s_in[sr][sc] = (row_ok && sc < kSmemW && gc < input_cols) ? __ldg(&in_row[gc]) : 0.0f;
            }
        }
    }
    __syncthreads();

    // 2. Compute 4x4 output tile per thread using LDS.128 + 2-float4 sliding register window
    const int sr0 = ty * 4;
    float acc[4][4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            acc[r][c] = 0.0f;
        }
    }

    constexpr int kKcChunks = (KW + 3) / 4;

    #pragma unroll
    for (int kr = 0; kr < KH; ++kr) {
        float4 v_cur[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float4* row_v4 = reinterpret_cast<const float4*>(&s_in[sr0 + r + kr][0]) + tx;
            v_cur[r] = row_v4[0];
        }

        #pragma unroll
        for (int q = 0; q < kKcChunks; ++q) {
            constexpr int kMaxChunk = 4;
            const int kc_base = q * 4;
            const int rem = KW - kc_base;

            float4 v_nxt[4];
            if (rem > 1) {
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const float4* row_v4 = reinterpret_cast<const float4*>(&s_in[sr0 + r + kr][0]) + tx;
                    v_nxt[r] = row_v4[q + 1];
                }
            } else {
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    v_nxt[r] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }

            float win[4][7];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                win[r][0] = v_cur[r].x;
                win[r][1] = v_cur[r].y;
                win[r][2] = v_cur[r].z;
                win[r][3] = v_cur[r].w;
                win[r][4] = v_nxt[r].x;
                win[r][5] = v_nxt[r].y;
                win[r][6] = v_nxt[r].z;
            }

            #pragma unroll
            for (int j = 0; j < kMaxChunk; ++j) {
                if (kc_base + j < KW) {
                    const float kv = s_k[kr * KW + kc_base + j];
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        acc[r][0] = fmaf(win[r][j + 0], kv, acc[r][0]);
                        acc[r][1] = fmaf(win[r][j + 1], kv, acc[r][1]);
                        acc[r][2] = fmaf(win[r][j + 2], kv, acc[r][2]);
                        acc[r][3] = fmaf(win[r][j + 3], kv, acc[r][3]);
                    }
                }
            }

            if (q + 1 < kKcChunks) {
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    v_cur[r] = v_nxt[r];
                }
            }
        }
    }
    __syncthreads();

    // 3. Stage the (CONV2D_BM x 64) output tile into s_in so global stores are 100% stride-1 coalesced!
    const int sc0 = tx * 4;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        float4* dst_v4 = reinterpret_cast<float4*>(&s_in[sr0 + r][sc0]);
        *dst_v4 = make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
    }
    __syncthreads();

    // Stride-1 coalesced write: 64 contiguous threads (2 warps) write each row of 64 floats!
    const int write_col = tid & 63;
    const int write_row_init = tid >> 6;
    constexpr int kRowsPerStep = kThreads / 64;  // 2 for 128 thr, 4 for 256 thr
    #pragma unroll
    for (int sr = write_row_init; sr < CONV2D_BM; sr += kRowsPerStep) {
        const int out_r = base_r + sr;
        const int out_c = base_c + write_col;
        if (out_r < out_rows && out_c < out_cols) {
            output[static_cast<size_t>(out_r) * static_cast<size_t>(out_cols) + out_c] =
                s_in[sr][write_col];
        }
    }
}

// ============================================================================
// General Shared-Memory 2x4 Register-Tiled Kernel for Arbitrary KH, KW in [1, 31]
// Handles asymmetric/even/odd kernels (1x1, 1x2, 1x3, 2x2, 31x31, etc.)!
// ============================================================================
__global__ __launch_bounds__(128, 2) void conv2d_valid_general_smem_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_rows,
    int input_cols,
    int kernel_rows,
    int kernel_cols,
    int out_rows,
    int out_cols) {
    // 16 x 64 output tile + max 31 x 31 kernel -> max halo = 46 x 94 -> 46 * 97 * 4B = 17.8 KiB + 3.8 KiB = 21.6 KiB!
    constexpr int kGenOutH = 16;
    constexpr int kGenOutW = 64;
    __shared__ float s_in[kGenOutH + 30][97];
    __shared__ float s_k[31 * 31];

    const int tx = threadIdx.x;  // 0..15
    const int ty = threadIdx.y;  // 0..7
    const int tid = ty * 16 + tx;

    const int base_r = blockIdx.y * kGenOutH;
    const int base_c = blockIdx.x * kGenOutW;

    const int ksize = kernel_rows * kernel_cols;
    for (int idx = tid; idx < ksize; idx += 128) {
        s_k[idx] = __ldg(&kernel[idx]);
    }

    const int smem_h = kGenOutH + kernel_rows - 1;
    const int smem_w = kGenOutW + kernel_cols - 1;
    const int warp_id = tid >> 5;  // 0..3
    const int lane = tid & 31;
    for (int sr = warp_id; sr < smem_h; sr += 4) {
        const int gr = base_r + sr;
        const bool row_ok = (gr < input_rows);
        const float* in_row =
            row_ok ? (input + static_cast<size_t>(gr) * static_cast<size_t>(input_cols)) : nullptr;
        for (int sc = lane; sc < smem_w; sc += 32) {
            const int gc = base_c + sc;
            s_in[sr][sc] = (row_ok && gc < input_cols) ? __ldg(&in_row[gc]) : 0.0f;
        }
    }
    __syncthreads();

    const int sr0 = ty * 2;
    const int sc0 = tx * 4;
    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;

    if (base_r + sr0 < out_rows && base_c + sc0 < out_cols) {
        for (int kr = 0; kr < kernel_rows; ++kr) {
            const int k_base = kr * kernel_cols;
            float r00 = 0.0f, r01 = 0.0f, r02 = 0.0f, r03 = 0.0f;
            float r10 = 0.0f, r11 = 0.0f, r12 = 0.0f, r13 = 0.0f;
            for (int kc = 0; kc < kernel_cols; ++kc) {
                const float kv = s_k[k_base + kc];
                r00 = fmaf(s_in[sr0 + kr][sc0 + kc + 0], kv, r00);
                r01 = fmaf(s_in[sr0 + kr][sc0 + kc + 1], kv, r01);
                r02 = fmaf(s_in[sr0 + kr][sc0 + kc + 2], kv, r02);
                r03 = fmaf(s_in[sr0 + kr][sc0 + kc + 3], kv, r03);

                r10 = fmaf(s_in[sr0 + 1 + kr][sc0 + kc + 0], kv, r10);
                r11 = fmaf(s_in[sr0 + 1 + kr][sc0 + kc + 1], kv, r11);
                r12 = fmaf(s_in[sr0 + 1 + kr][sc0 + kc + 2], kv, r12);
                r13 = fmaf(s_in[sr0 + 1 + kr][sc0 + kc + 3], kv, r13);
            }
            acc00 += r00;
            acc01 += r01;
            acc02 += r02;
            acc03 += r03;
            acc10 += r10;
            acc11 += r11;
            acc12 += r12;
            acc13 += r13;
        }
    }
    __syncthreads();

    s_in[sr0][sc0 + 0] = acc00;
    s_in[sr0][sc0 + 1] = acc01;
    s_in[sr0][sc0 + 2] = acc02;
    s_in[sr0][sc0 + 3] = acc03;
    s_in[sr0 + 1][sc0 + 0] = acc10;
    s_in[sr0 + 1][sc0 + 1] = acc11;
    s_in[sr0 + 1][sc0 + 2] = acc12;
    s_in[sr0 + 1][sc0 + 3] = acc13;
    __syncthreads();

    const int write_col = tid & 63;
    const int write_row_init = tid >> 6;  // 0..1
    #pragma unroll
    for (int sr = write_row_init; sr < kGenOutH; sr += 2) {
        const int out_r = base_r + sr;
        const int out_c = base_c + write_col;
        if (out_r < out_rows && out_c < out_cols) {
            output[static_cast<size_t>(out_r) * static_cast<size_t>(out_cols) + out_c] =
                s_in[sr][write_col];
        }
    }
}

// Fallback if kernel_rows > 31 or kernel_cols > 31
__global__ void conv2d_valid_huge_k_fallback_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
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

    float sum = 0.0f;
    for (int kr = 0; kr < kernel_rows; ++kr) {
        const size_t in_base = static_cast<size_t>(row + kr) * input_cols + col;
        const int k_base = kr * kernel_cols;
        float row_sum = 0.0f;
        for (int kc = 0; kc < kernel_cols; ++kc) {
            row_sum = fmaf(__ldg(&input[in_base + kc]), __ldg(&kernel[k_base + kc]), row_sum);
        }
        sum += row_sum;
    }
    output[static_cast<size_t>(row) * out_cols + col] = sum;
}

}  // namespace

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
    const int out_rows = input_rows - kernel_rows + 1;
    const int out_cols = input_cols - kernel_cols + 1;
    if (out_rows <= 0 || out_cols <= 0 || kernel_rows <= 0 || kernel_cols <= 0) {
        return;
    }

    if (kernel_rows == kernel_cols && out_rows >= 16 && out_cols >= 16) {
        const dim3 block(kBlockX, kBlockY);
        const dim3 grid(static_cast<unsigned>((out_cols + kBN - 1) / kBN),
                        static_cast<unsigned>((out_rows + CONV2D_BM - 1) / CONV2D_BM));
        switch (kernel_rows) {
            case 3:
                conv2d_valid_reg4x4_kernel<3, 3><<<grid, block>>>(
                    input, kernel, output, input_rows, input_cols, out_rows, out_cols);
                return;
            case 5:
                conv2d_valid_reg4x4_kernel<5, 5><<<grid, block>>>(
                    input, kernel, output, input_rows, input_cols, out_rows, out_cols);
                return;
            case 7:
                conv2d_valid_reg4x4_kernel<7, 7><<<grid, block>>>(
                    input, kernel, output, input_rows, input_cols, out_rows, out_cols);
                return;
            case 9:
                conv2d_valid_reg4x4_kernel<9, 9><<<grid, block>>>(
                    input, kernel, output, input_rows, input_cols, out_rows, out_cols);
                return;
            case 11:
                conv2d_valid_reg4x4_kernel<11, 11><<<grid, block>>>(
                    input, kernel, output, input_rows, input_cols, out_rows, out_cols);
                return;
            case 13:
                conv2d_valid_reg4x4_kernel<13, 13><<<grid, block>>>(
                    input, kernel, output, input_rows, input_cols, out_rows, out_cols);
                return;
            case 15:
                conv2d_valid_reg4x4_kernel<15, 15><<<grid, block>>>(
                    input, kernel, output, input_rows, input_cols, out_rows, out_cols);
                return;
            default:
                break;
        }
    }

    if (kernel_rows <= 31 && kernel_cols <= 31) {
        const dim3 block(16, 8);
        const dim3 grid(static_cast<unsigned>((out_cols + 63) / 64),
                        static_cast<unsigned>((out_rows + 15) / 16));
        conv2d_valid_general_smem_kernel<<<grid, block>>>(
            input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols,
            out_rows, out_cols);
        return;
    }

    const dim3 fblock(16, 16);
    const dim3 fgrid(static_cast<unsigned>((out_cols + 15) / 16),
                     static_cast<unsigned>((out_rows + 15) / 16));
    conv2d_valid_huge_k_fallback_kernel<<<fgrid, fblock>>>(
        input, kernel, output, input_cols, kernel_rows, kernel_cols, out_rows, out_cols);
}
