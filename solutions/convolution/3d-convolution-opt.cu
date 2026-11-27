#include <cuda_runtime.h>
#include <cstdint>

#ifndef CONV3D_BZ
#define CONV3D_BZ 4
#endif

namespace {

constexpr int kBZ = CONV3D_BZ;
static_assert(kBZ >= 1 && kBZ <= 12, "CONV3D_BZ must be in [1, 12]");

__device__ __forceinline__ void store_streaming_float4(float4* ptr, const float4& v) {
#if __CUDA_ARCH__ >= 700
    asm volatile("st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w));
#else
    *ptr = v;
#endif
}

// ============================================================================
// Templated Z-Slab Register-Tiled + Single-2D-Plane Shared-Memory LDS.128 Kernel!
// Specializes compile-time (KD, KH, KW) (e.g. 5x5x5 and 3x3x3):
// - Block: 128 threads (tx = 0..15, ty = 0..7) -> covers 16 x 64 (orow, ocol) 2D tile
//   (2 rows x 4 cols per thread!) across `kBZ` consecutive `od` slices!
// - Shared memory holds ONLY ONE 2D input plane `s_plane[16 + KH - 1][72]` (<= 5.8 KiB!)
//   + `s_k[KD * KH * KW]` (<= 500 B), enabling up to 6-8 blocks/SM capacity!
// - For each streamed plane `iz_rel`:
//   * Coalesced float4 loads when aligned & input_cols % 4 == 0!
//   * Each thread loads 2 aligned float4s (`p[0..7]`) per local row via `LDS.128` and
//     reuses those 8 registers across BOTH `r in {0, 1}` AND ALL valid `(bz, kd)` pairs!
// ============================================================================
template <int KD, int KH, int KW, int BZ, bool kAlignedIn4>
__global__ __launch_bounds__(128) void conv3d_zslab_specialized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_rows,
    int input_cols,
    int out_d,
    int out_r,
    int out_c,
    bool aligned_out4) {
    constexpr int kTileR = 16;
    constexpr int kTileC = 64;
    constexpr int kPlaneRows = kTileR + KH - 1;
    constexpr int kPlaneCols = kTileC + KW - 1;
    constexpr int kPlaneStride = 72;  // Multiple of 4 floats (16-byte aligned for LDS.128!)
    constexpr int kKSize = KD * KH * KW;

    __shared__ float4 s_plane4[kPlaneRows][kPlaneStride >> 2];
    __shared__ float s_k[kKSize];
    float (*s_plane)[kPlaneStride] = reinterpret_cast<float (*)[kPlaneStride]>(s_plane4);

    const int tid = threadIdx.x;
    const int tx = tid & 15;   // 0..15 -> ocol = oc0 + 4*tx
    const int ty = tid >> 4;   // 0..7  -> orow = or0 + 2*ty

    const int oc0 = static_cast<int>(blockIdx.x) * kTileC;
    const int or0 = static_cast<int>(blockIdx.y) * kTileR;
    const int od0 = static_cast<int>(blockIdx.z) * BZ;

    for (int i = tid; i < kKSize; i += 128) {
        s_k[i] = __ldg(&kernel[i]);
    }

    float acc[BZ][2][4];
#pragma unroll
    for (int bz = 0; bz < BZ; ++bz) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                acc[bz][r][c] = 0.0f;
            }
        }
    }

    const int bz_valid = (od0 + BZ <= out_d) ? BZ : (out_d - od0);
    const int total_planes = bz_valid + KD - 1;
    const size_t in_plane_elems = static_cast<size_t>(input_rows) * static_cast<size_t>(input_cols);

    for (int iz_rel = 0; iz_rel < total_planes; ++iz_rel) {
        const int iz = od0 + iz_rel;
        const float* __restrict__ in_plane =
            input + static_cast<size_t>(iz) * in_plane_elems;

        if (kAlignedIn4) {
            constexpr int kVecCols = (kPlaneCols + 3) >> 2;  // 17 float4s for KW=5 (68 cols)
            constexpr int kTotalVecs = kPlaneRows * kVecCols;
            for (int idx = tid; idx < kTotalVecs; idx += 128) {
                const int pr = idx / kVecCols;
                const int pc4 = idx - pr * kVecCols;
                const int ir = or0 + pr;
                const int ic = oc0 + (pc4 << 2);
                float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (ir < input_rows && ic < input_cols) {
                    v = __ldg(reinterpret_cast<const float4*>(
                        in_plane + static_cast<size_t>(ir) * static_cast<size_t>(input_cols) +
                        static_cast<size_t>(ic)));
                }
                *reinterpret_cast<float4*>(&s_plane[pr][pc4 << 2]) = v;
            }
        } else {
            constexpr int kTotalScalars = kPlaneRows * kPlaneCols;
            for (int idx = tid; idx < kTotalScalars; idx += 128) {
                const int pr = idx / kPlaneCols;
                const int pc = idx - pr * kPlaneCols;
                const int ir = or0 + pr;
                const int ic = oc0 + pc;
                float val = 0.0f;
                if (ir < input_rows && ic < input_cols) {
                    val = __ldg(&in_plane[static_cast<size_t>(ir) * static_cast<size_t>(input_cols) +
                                          static_cast<size_t>(ic)]);
                }
                s_plane[pr][pc] = val;
            }
        }
        __syncthreads();

        // Stream through the (2 + KH - 1) local rows needed by this thread's 2 output rows:
        // Each local row requires ONLY 2 LDS.128 loads (`p[0..7]`), reused across r in {0,1} and all bz!
        constexpr int kThreadRows = 2 + KH - 1;
#pragma unroll
        for (int lr = 0; lr < kThreadRows; ++lr) {
            const float4* smem_row4 =
                reinterpret_cast<const float4*>(&s_plane[(ty << 1) + lr][tx << 2]);
            const float4 p0 = smem_row4[0];
            const float4 p1 = smem_row4[1];
            const float p[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};

#pragma unroll
            for (int bz = 0; bz < BZ; ++bz) {
                const int kd = iz_rel - bz;
                if (kd >= 0 && kd < KD) {
                    if (lr < KH) {
                        const int kr0 = lr;
                        const float* kw0 = s_k + (kd * KH + kr0) * KW;
#pragma unroll
                        for (int kc = 0; kc < KW; ++kc) {
                            const float kv = kw0[kc];
                            acc[bz][0][0] = fmaf(p[kc + 0], kv, acc[bz][0][0]);
                            acc[bz][0][1] = fmaf(p[kc + 1], kv, acc[bz][0][1]);
                            acc[bz][0][2] = fmaf(p[kc + 2], kv, acc[bz][0][2]);
                            acc[bz][0][3] = fmaf(p[kc + 3], kv, acc[bz][0][3]);
                        }
                    }
                    if (lr >= 1 && (lr - 1) < KH) {
                        const int kr1 = lr - 1;
                        const float* kw1 = s_k + (kd * KH + kr1) * KW;
#pragma unroll
                        for (int kc = 0; kc < KW; ++kc) {
                            const float kv = kw1[kc];
                            acc[bz][1][0] = fmaf(p[kc + 0], kv, acc[bz][1][0]);
                            acc[bz][1][1] = fmaf(p[kc + 1], kv, acc[bz][1][1]);
                            acc[bz][1][2] = fmaf(p[kc + 2], kv, acc[bz][1][2]);
                            acc[bz][1][3] = fmaf(p[kc + 3], kv, acc[bz][1][3]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    const int out_col = oc0 + (tx << 2);
    if (out_col >= out_c) {
        return;
    }
    const size_t out_plane_elems = static_cast<size_t>(out_r) * static_cast<size_t>(out_c);

#pragma unroll
    for (int bz = 0; bz < BZ; ++bz) {
        const int od = od0 + bz;
        if (od < out_d) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int orow = or0 + (ty << 1) + r;
                if (orow < out_r) {
                    float* out_ptr =
                        output + static_cast<size_t>(od) * out_plane_elems +
                        static_cast<size_t>(orow) * static_cast<size_t>(out_c) +
                        static_cast<size_t>(out_col);
                    if (aligned_out4 && out_col + 3 < out_c) {
                        const float4 res = make_float4(
                            acc[bz][r][0], acc[bz][r][1], acc[bz][r][2], acc[bz][r][3]);
                        store_streaming_float4(reinterpret_cast<float4*>(out_ptr), res);
                    } else {
                        out_ptr[0] = acc[bz][r][0];
                        if (out_col + 1 < out_c) out_ptr[1] = acc[bz][r][1];
                        if (out_col + 2 < out_c) out_ptr[2] = acc[bz][r][2];
                        if (out_col + 3 < out_c) out_ptr[3] = acc[bz][r][3];
                    }
                }
            }
        }
    }
}

// ============================================================================
// General Z-Slab Shared-Memory Streaming Kernel for Arbitrary 1 <= KD, KH, KW <= 5!
// Covers asymmetric kernels like 2x3x3, 3x4x5, 1x2x3, 2x2x2, 1x1x1!
// ============================================================================
__global__ __launch_bounds__(128) void conv3d_zslab_general_k5_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_rows,
    int input_cols,
    int kd_dim,
    int kh_dim,
    int kw_dim,
    int out_d,
    int out_r,
    int out_c) {
    constexpr int kTileR = 16;
    constexpr int kTileC = 64;
    constexpr int kBZGen = 4;

    __shared__ float4 s_plane4[20][18];
    __shared__ float s_k[125];
    float (*s_plane)[72] = reinterpret_cast<float (*)[72]>(s_plane4);

    const int tid = threadIdx.x;
    const int tx = tid & 15;
    const int ty = tid >> 4;

    const int oc0 = static_cast<int>(blockIdx.x) * kTileC;
    const int or0 = static_cast<int>(blockIdx.y) * kTileR;
    const int od0 = static_cast<int>(blockIdx.z) * kBZGen;

    const int k_size = kd_dim * kh_dim * kw_dim;
    for (int i = tid; i < k_size; i += 128) {
        s_k[i] = __ldg(&kernel[i]);
    }

    float acc[kBZGen][2][4];
#pragma unroll
    for (int bz = 0; bz < kBZGen; ++bz) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                acc[bz][r][c] = 0.0f;
            }
        }
    }

    const int bz_valid = (od0 + kBZGen <= out_d) ? kBZGen : (out_d - od0);
    const int total_planes = bz_valid + kd_dim - 1;
    const int plane_rows = kTileR + kh_dim - 1;
    const int plane_cols = kTileC + kw_dim - 1;
    const int total_scalars = plane_rows * plane_cols;
    const size_t in_plane_elems = static_cast<size_t>(input_rows) * static_cast<size_t>(input_cols);

    for (int iz_rel = 0; iz_rel < total_planes; ++iz_rel) {
        const int iz = od0 + iz_rel;
        const float* __restrict__ in_plane =
            input + static_cast<size_t>(iz) * in_plane_elems;

        for (int idx = tid; idx < total_scalars; idx += 128) {
            const int pr = idx / plane_cols;
            const int pc = idx - pr * plane_cols;
            const int ir = or0 + pr;
            const int ic = oc0 + pc;
            float val = 0.0f;
            if (ir < input_rows && ic < input_cols) {
                val = __ldg(&in_plane[static_cast<size_t>(ir) * static_cast<size_t>(input_cols) +
                                      static_cast<size_t>(ic)]);
            }
            s_plane[pr][pc] = val;
        }
        __syncthreads();

        const int thread_rows = 2 + kh_dim - 1;
        for (int lr = 0; lr < thread_rows; ++lr) {
            const float4* smem_row4 =
                reinterpret_cast<const float4*>(&s_plane[(ty << 1) + lr][tx << 2]);
            const float4 p0 = smem_row4[0];
            const float4 p1 = smem_row4[1];
            const float p[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};

#pragma unroll
            for (int bz = 0; bz < kBZGen; ++bz) {
                const int kd = iz_rel - bz;
                if (kd >= 0 && kd < kd_dim) {
                    if (lr < kh_dim) {
                        const float* kw0 = s_k + (kd * kh_dim + lr) * kw_dim;
                        for (int kc = 0; kc < kw_dim; ++kc) {
                            const float kv = kw0[kc];
                            acc[bz][0][0] = fmaf(p[kc + 0], kv, acc[bz][0][0]);
                            acc[bz][0][1] = fmaf(p[kc + 1], kv, acc[bz][0][1]);
                            acc[bz][0][2] = fmaf(p[kc + 2], kv, acc[bz][0][2]);
                            acc[bz][0][3] = fmaf(p[kc + 3], kv, acc[bz][0][3]);
                        }
                    }
                    if (lr >= 1 && (lr - 1) < kh_dim) {
                        const float* kw1 = s_k + (kd * kh_dim + (lr - 1)) * kw_dim;
                        for (int kc = 0; kc < kw_dim; ++kc) {
                            const float kv = kw1[kc];
                            acc[bz][1][0] = fmaf(p[kc + 0], kv, acc[bz][1][0]);
                            acc[bz][1][1] = fmaf(p[kc + 1], kv, acc[bz][1][1]);
                            acc[bz][1][2] = fmaf(p[kc + 2], kv, acc[bz][1][2]);
                            acc[bz][1][3] = fmaf(p[kc + 3], kv, acc[bz][1][3]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    const int out_col = oc0 + (tx << 2);
    if (out_col >= out_c) {
        return;
    }
    const size_t out_plane_elems = static_cast<size_t>(out_r) * static_cast<size_t>(out_c);

#pragma unroll
    for (int bz = 0; bz < kBZGen; ++bz) {
        const int od = od0 + bz;
        if (od < out_d) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int orow = or0 + (ty << 1) + r;
                if (orow < out_r) {
                    float* out_ptr =
                        output + static_cast<size_t>(od) * out_plane_elems +
                        static_cast<size_t>(orow) * static_cast<size_t>(out_c) +
                        static_cast<size_t>(out_col);
                    out_ptr[0] = acc[bz][r][0];
                    if (out_col + 1 < out_c) out_ptr[1] = acc[bz][r][1];
                    if (out_col + 2 < out_c) out_ptr[2] = acc[bz][r][2];
                    if (out_col + 3 < out_c) out_ptr[3] = acc[bz][r][3];
                }
            }
        }
    }
}

// ============================================================================
// Beyond-Spec Large Kernel Fallback (if any kernel dim > 5)
// ============================================================================
__global__ __launch_bounds__(256) void conv3d_large_fallback_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_rows,
    int input_cols,
    int kd_dim,
    int kh_dim,
    int kw_dim,
    int out_d,
    int out_r,
    int out_c) {
    const int oc = blockIdx.x * 16 + (threadIdx.x & 15);
    const int orow = blockIdx.y * 16 + (threadIdx.x >> 4);
    const int od = blockIdx.z;
    if (oc >= out_c || orow >= out_r || od >= out_d) {
        return;
    }

    float sum = 0.0f;
    for (int kd = 0; kd < kd_dim; ++kd) {
        const size_t in_d_off = static_cast<size_t>(od + kd) * static_cast<size_t>(input_rows);
        const size_t k_d_off = static_cast<size_t>(kd) * static_cast<size_t>(kh_dim);
        for (int kr = 0; kr < kh_dim; ++kr) {
            const float* in_row =
                input + (in_d_off + static_cast<size_t>(orow + kr)) * static_cast<size_t>(input_cols) +
                static_cast<size_t>(oc);
            const float* k_row =
                kernel + (k_d_off + static_cast<size_t>(kr)) * static_cast<size_t>(kw_dim);
            for (int kc = 0; kc < kw_dim; ++kc) {
                sum = fmaf(__ldg(&in_row[kc]), __ldg(&k_row[kc]), sum);
            }
        }
    }

    output[(static_cast<size_t>(od) * static_cast<size_t>(out_r) + static_cast<size_t>(orow)) *
               static_cast<size_t>(out_c) +
           static_cast<size_t>(oc)] = sum;
}

}  // namespace

// input, kernel, output are device pointers
extern "C" void solve(const float* input,
                      const float* kernel,
                      float* output,
                      int input_depth,
                      int input_rows,
                      int input_cols,
                      int kernel_depth,
                      int kernel_rows,
                      int kernel_cols) {
    const int out_d = input_depth - kernel_depth + 1;
    const int out_r = input_rows - kernel_rows + 1;
    const int out_c = input_cols - kernel_cols + 1;
    if (out_d <= 0 || out_r <= 0 || out_c <= 0) {
        return;
    }

    const bool aligned_in4 =
        ((reinterpret_cast<uintptr_t>(input) & 15u) == 0u) && ((input_cols & 3) == 0);
    const bool aligned_out4 =
        ((reinterpret_cast<uintptr_t>(output) & 15u) == 0u) && ((out_c & 3) == 0);

    // 1. Exact compile-time 5x5x5 specialization (covers official perf 256x128x128 * 5x5x5!)
    if (kernel_depth == 5 && kernel_rows == 5 && kernel_cols == 5) {
        const dim3 grid(static_cast<unsigned>((out_c + 63) / 64),
                        static_cast<unsigned>((out_r + 15) / 16),
                        static_cast<unsigned>((out_d + kBZ - 1) / kBZ));
        if (aligned_in4) {
            conv3d_zslab_specialized_kernel<5, 5, 5, kBZ, true><<<grid, 128>>>(
                input, kernel, output, input_rows, input_cols, out_d, out_r, out_c, aligned_out4);
        } else {
            conv3d_zslab_specialized_kernel<5, 5, 5, kBZ, false><<<grid, 128>>>(
                input, kernel, output, input_rows, input_cols, out_d, out_r, out_c, aligned_out4);
        }
        return;
    }

    // 2. Exact compile-time 3x3x3 specialization
    if (kernel_depth == 3 && kernel_rows == 3 && kernel_cols == 3) {
        const dim3 grid(static_cast<unsigned>((out_c + 63) / 64),
                        static_cast<unsigned>((out_r + 15) / 16),
                        static_cast<unsigned>((out_d + kBZ - 1) / kBZ));
        if (aligned_in4) {
            conv3d_zslab_specialized_kernel<3, 3, 3, kBZ, true><<<grid, 128>>>(
                input, kernel, output, input_rows, input_cols, out_d, out_r, out_c, aligned_out4);
        } else {
            conv3d_zslab_specialized_kernel<3, 3, 3, kBZ, false><<<grid, 128>>>(
                input, kernel, output, input_rows, input_cols, out_d, out_r, out_c, aligned_out4);
        }
        return;
    }

    // 3. General Z-slab plane-streaming kernel for all 1 <= KD, KH, KW <= 5
    if (kernel_depth <= 5 && kernel_rows <= 5 && kernel_cols <= 5) {
        const dim3 grid(static_cast<unsigned>((out_c + 63) / 64),
                        static_cast<unsigned>((out_r + 15) / 16),
                        static_cast<unsigned>((out_d + 3) / 4));
        conv3d_zslab_general_k5_kernel<<<grid, 128>>>(
            input, kernel, output, input_rows, input_cols,
            kernel_depth, kernel_rows, kernel_cols, out_d, out_r, out_c);
        return;
    }

    // 4. Fallback for beyond-spec kernel sizes (> 5)
    const dim3 grid(static_cast<unsigned>((out_c + 15) / 16),
                    static_cast<unsigned>((out_r + 15) / 16),
                    static_cast<unsigned>(out_d));
    conv3d_large_fallback_kernel<<<grid, 256>>>(
        input, kernel, output, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols, out_d, out_r, out_c);
}
