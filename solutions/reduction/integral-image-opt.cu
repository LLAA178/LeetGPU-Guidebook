#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarp;  // 8 warps per block
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kFastMaxDim = 8192;
constexpr int kMaxColTiles = kFastMaxDim / 128;       // 64 column tiles of 128 cols (32 float4s)
constexpr int kMaxLookbackStrips = kFastMaxDim / 64;  // 128 vertical strips of 64 rows

// Compact 16.03 MiB static .bss (safe for LeetGPU dlopen sandbox)
__device__ double g_lb_agg[kMaxLookbackStrips * kFastMaxDim];
__device__ double g_lb_pref[kMaxLookbackStrips * kFastMaxDim];
__device__ int g_lb_flag[kMaxLookbackStrips * kMaxColTiles];
__device__ int g_col_tile_ctr[kMaxColTiles];
__device__ int g_call_epoch = 0;

__device__ __forceinline__ void store_streaming_float4(float4* addr, float4 v) {
#if __CUDA_ARCH__ >= 700
    asm volatile("st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(addr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
                 : "memory");
#else
    *addr = v;
#endif
}

__device__ __forceinline__ double load_global_cg_f64(const double* ptr) {
    double val;
#if __CUDA_ARCH__ >= 700
    asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(val) : "l"(ptr) : "memory");
#else
    val = *reinterpret_cast<const volatile double*>(ptr);
#endif
    return val;
}

__device__ __forceinline__ void store_global_cg_f64(double* ptr, double val) {
#if __CUDA_ARCH__ >= 700
    asm volatile("st.global.cg.f64 [%0], %1;" : : "l"(ptr), "d"(val) : "memory");
#else
    *reinterpret_cast<volatile double*>(ptr) = val;
#endif
}

__device__ __forceinline__ int load_acquire_flag(const int* ptr) {
    int flag;
#if __CUDA_ARCH__ >= 700
    asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(flag) : "l"(ptr) : "memory");
#else
    flag = *reinterpret_cast<const volatile int*>(ptr);
#endif
    __threadfence();
    return flag;
}

__device__ __forceinline__ void store_release_flag(int* ptr, int val) {
    __threadfence();
#if __CUDA_ARCH__ >= 700
    asm volatile("st.global.cg.s32 [%0], %1;" : : "l"(ptr), "r"(val) : "memory");
#else
    *reinterpret_cast<volatile int*>(ptr) = val;
#endif
}

__device__ __forceinline__ float warp_inclusive_scan_f32(float val, int lane) {
    #pragma unroll
    for (int offset = 1; offset < kWarp; offset <<= 1) {
        const float prev = __shfl_up_sync(kFullMask, val, offset);
        if (lane >= offset) {
            val += prev;
        }
    }
    return val;
}

// ============================================================================
// Pass 1: Zero-Barrier Warp-Per-Row Vectorized `float4` Inclusive Scan
// - 8 warps per 256-thread block independently scan 8 rows in parallel with
//   ZERO `__syncthreads()` and ZERO shared memory!
// - 2x-unrolled `64` float4s (`256` floats = `1 KiB`) per step, pure FP32
//   in-register shuffle scan + Kahan compensated carry `(carry_hi, carry_err)`
//   (eliminating 1/32-rate Turing FP64 stalls while keeping ~1-ULP accuracy!).
// - Fuses `g_call_epoch` & `g_col_tile_ctr` initialization on `blockIdx.x == 0`
//   so Pass 2 starts immediately with zero helper-kernel launch overhead!
// ============================================================================
__global__ __launch_bounds__(kThreads) void row_scan_warp_vec4_kernel(
    const float4* __restrict__ in4,
    float4* __restrict__ out4,
    int H,
    int W4,
    int init_col_tiles) {
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    if (blockIdx.x == 0 && init_col_tiles > 0) {
        if (tid == 0) {
            int e = g_call_epoch + 2;
            if (e > 1000000000) e = 2;
            g_call_epoch = e;
        }
        for (int i = tid; i < init_col_tiles; i += kThreads) {
            g_col_tile_ctr[i] = 0;
        }
    }

    const int row_stride = static_cast<int>(gridDim.x) * kWarps;
    for (int r = static_cast<int>(blockIdx.x) * kWarps + warp; r < H; r += row_stride) {
        const size_t row_off = static_cast<size_t>(r) * static_cast<size_t>(W4);
        float carry_hi = 0.0f;
        float carry_err = 0.0f;

        int c4_base = 0;
        for (; c4_base + 64 <= W4; c4_base += 64) {
            const float4 v0 = __ldg(&in4[row_off + c4_base + lane]);
            const float4 v1 = __ldg(&in4[row_off + c4_base + 32 + lane]);

            {
                const float p0 = v0.x;
                const float p1 = p0 + v0.y;
                const float p2 = p1 + v0.z;
                const float p3 = p2 + v0.w;
                const float w_inc = warp_inclusive_scan_f32(p3, lane);
                const float w_prev = __shfl_up_sync(kFullMask, w_inc, 1);
                const float w_excl = (lane > 0) ? w_prev : 0.0f;
                const float adj = w_excl + carry_err;

                float4 out0;
                out0.x = carry_hi + (adj + p0);
                out0.y = carry_hi + (adj + p1);
                out0.z = carry_hi + (adj + p2);
                out0.w = carry_hi + (adj + p3);
                store_streaming_float4(&out4[row_off + c4_base + lane], out0);

                const float step_sum = __shfl_sync(kFullMask, w_inc, 31);
                const float y = step_sum + carry_err;
                const float t = carry_hi + y;
                carry_err = y - (t - carry_hi);
                carry_hi = t;
            }

            {
                const float p0 = v1.x;
                const float p1 = p0 + v1.y;
                const float p2 = p1 + v1.z;
                const float p3 = p2 + v1.w;
                const float w_inc = warp_inclusive_scan_f32(p3, lane);
                const float w_prev = __shfl_up_sync(kFullMask, w_inc, 1);
                const float w_excl = (lane > 0) ? w_prev : 0.0f;
                const float adj = w_excl + carry_err;

                float4 out1;
                out1.x = carry_hi + (adj + p0);
                out1.y = carry_hi + (adj + p1);
                out1.z = carry_hi + (adj + p2);
                out1.w = carry_hi + (adj + p3);
                store_streaming_float4(&out4[row_off + c4_base + 32 + lane], out1);

                const float step_sum = __shfl_sync(kFullMask, w_inc, 31);
                const float y = step_sum + carry_err;
                const float t = carry_hi + y;
                carry_err = y - (t - carry_hi);
                carry_hi = t;
            }
        }

        for (; c4_base < W4; c4_base += 32) {
            const int c4 = c4_base + lane;
            const float4 v =
                (c4 < W4) ? __ldg(&in4[row_off + c4]) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const float p0 = v.x;
            const float p1 = p0 + v.y;
            const float p2 = p1 + v.z;
            const float p3 = p2 + v.w;
            const float w_inc = warp_inclusive_scan_f32(p3, lane);
            const float w_prev = __shfl_up_sync(kFullMask, w_inc, 1);
            const float w_excl = (lane > 0) ? w_prev : 0.0f;

            if (c4 < W4) {
                const float adj = w_excl + carry_err;
                float4 out_v;
                out_v.x = carry_hi + (adj + p0);
                out_v.y = carry_hi + (adj + p1);
                out_v.z = carry_hi + (adj + p2);
                out_v.w = carry_hi + (adj + p3);
                store_streaming_float4(&out4[row_off + c4], out_v);
            }

            const float step_sum = __shfl_sync(kFullMask, w_inc, 31);
            const float y = step_sum + carry_err;
            const float t = carry_hi + y;
            carry_err = y - (t - carry_hi);
            carry_hi = t;
        }
    }
}

// ============================================================================
// Pass 2: Register-Tiled Decoupled-Lookback Vertical `float4` Scan (`1R + 1W`)
// - 256 threads (8 warps) per block own a `64 rows x 32 float4s` (`128` cols)
//   tile: each warp holds 8 rows (`float4 v[8]`) directly in registers!
// - Shared memory is only `8.2 KiB` per block (down from `33.8 KiB`), boosting
//   Turing SM residency from 4 warps/SM (12.5%) up to 24-32 warps/SM (75-100%)!
// - Exact `double` lookback across strips + double-float `(hi, lo)` prefix
//   splitting cuts per-pixel FP64 ops to zero while preserving full precision!
// ============================================================================
__global__ __launch_bounds__(kThreads) void col_scan_onesweep_reg_vec4_kernel(
    float4* __restrict__ out4,
    int H,
    int W,
    int W4,
    int num_col_tiles) {
    __shared__ float4 s_warp_hi4[kWarps][32];  // 4 KiB: holds 8-row warp totals, then pref_hi
    __shared__ float4 s_warp_lo4[kWarps][32];  // 4 KiB: holds pref_lo
    __shared__ int s_sy;
    __shared__ int s_pred_state;

    const int bx = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);  // 0..255
    const int warp = tid >> 5;                      // 0..7
    const int lane = tid & 31;                      // 0..31
    const int c4 = bx * 32 + lane;
    const int col_scalar = bx * 128 + tid;          // valid for tid < 128
    const int epoch = g_call_epoch;

    if (tid == 0) {
        s_sy = atomicAdd(&g_col_tile_ctr[bx], 1);
    }
    __syncthreads();

    const int sy = s_sy;
    const int r_base = sy * 64;
    if (r_base >= H) {
        return;
    }

    // 1. Coalesced 8-row register load (`1R`) + in-register 8-row inclusive scan per warp
    const int warp_r0 = r_base + warp * 8;
    float4 v[8];
    if (c4 < W4 && warp_r0 + 7 < H) {
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            v[k] = __ldg(&out4[static_cast<size_t>(warp_r0 + k) * W4 + c4]);
        }
    } else {
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int r = warp_r0 + k;
            v[k] = (r < H && c4 < W4)
                       ? __ldg(&out4[static_cast<size_t>(r) * W4 + c4])
                       : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    #pragma unroll
    for (int k = 1; k < 8; ++k) {
        v[k].x += v[k - 1].x;
        v[k].y += v[k - 1].y;
        v[k].z += v[k - 1].z;
        v[k].w += v[k - 1].w;
    }

    s_warp_hi4[warp][lane] = v[7];
    __syncthreads();

    // 2. First 128 threads (`tid = 0..127`) sum the 8 warp totals in `double` (zero bank conflict)
    float* s_hi_f = reinterpret_cast<float*>(&s_warp_hi4[0][0]);  // [8][128]
    float* s_lo_f = reinterpret_cast<float*>(&s_warp_lo4[0][0]);  // [8][128]

    double local_acc = 0.0;
    if (tid < 128) {
        #pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            local_acc += static_cast<double>(s_hi_f[w * 128 + tid]);
        }
    }

    // 3. Decoupled lookback across vertical strips (all `__syncthreads()` strictly uniform!)
    const int slot = sy * num_col_tiles + bx;
    double strip_excl = 0.0;

    if (sy == 0) {
        if (tid < 128 && col_scalar < W) {
            store_global_cg_f64(&g_lb_pref[col_scalar], local_acc);
        }
        __syncthreads();
        if (tid == 0) {
            store_release_flag(&g_lb_flag[slot], epoch + 1);
        }
    } else {
        if (tid < 128 && col_scalar < W) {
            store_global_cg_f64(&g_lb_agg[static_cast<size_t>(sy) * W + col_scalar], local_acc);
        }
        __syncthreads();
        if (tid == 0) {
            store_release_flag(&g_lb_flag[slot], epoch);
        }

        int prev = sy - 1;
        while (prev >= 0) {
            if (tid == 0) {
                const int* pflag = &g_lb_flag[prev * num_col_tiles + bx];
                int st;
                do {
                    st = load_acquire_flag(pflag);
                } while (st != epoch && st != epoch + 1);
                s_pred_state = st;
            }
            __syncthreads();

            const int st = s_pred_state;
            if (st == epoch + 1) {
                if (tid < 128 && col_scalar < W) {
                    strip_excl +=
                        load_global_cg_f64(&g_lb_pref[static_cast<size_t>(prev) * W + col_scalar]);
                }
                break;
            } else {
                if (tid < 128 && col_scalar < W) {
                    strip_excl +=
                        load_global_cg_f64(&g_lb_agg[static_cast<size_t>(prev) * W + col_scalar]);
                }
                --prev;
            }
            __syncthreads();
        }

        if (tid < 128 && col_scalar < W) {
            store_global_cg_f64(&g_lb_pref[static_cast<size_t>(sy) * W + col_scalar],
                                strip_excl + local_acc);
        }
        __syncthreads();
        if (tid == 0) {
            store_release_flag(&g_lb_flag[slot], epoch + 1);
        }
    }

    // 4. Compute per-warp exclusive prefix and split into `(hi, lo)` float pair
    if (tid < 128) {
        double run = strip_excl;
        #pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            const double wt = static_cast<double>(s_hi_f[w * 128 + tid]);
            const float hi = static_cast<float>(run);
            const float lo = static_cast<float>(run - static_cast<double>(hi));
            s_hi_f[w * 128 + tid] = hi;
            s_lo_f[w * 128 + tid] = lo;
            run += wt;
        }
    }
    __syncthreads();

    // 5. Pure FP32 compensated prefix addition in registers + 128-bit streaming store (`1W`)
    if (c4 < W4) {
        const float4 p_hi = s_warp_hi4[warp][lane];
        const float4 p_lo = s_warp_lo4[warp][lane];
        if (warp_r0 + 7 < H) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float4 out_v;
                out_v.x = p_hi.x + (v[k].x + p_lo.x);
                out_v.y = p_hi.y + (v[k].y + p_lo.y);
                out_v.z = p_hi.z + (v[k].z + p_lo.z);
                out_v.w = p_hi.w + (v[k].w + p_lo.w);
                store_streaming_float4(
                    &out4[static_cast<size_t>(warp_r0 + k) * W4 + c4], out_v);
            }
        } else {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int r = warp_r0 + k;
                if (r < H) {
                    float4 out_v;
                    out_v.x = p_hi.x + (v[k].x + p_lo.x);
                    out_v.y = p_hi.y + (v[k].y + p_lo.y);
                    out_v.z = p_hi.z + (v[k].z + p_lo.z);
                    out_v.w = p_hi.w + (v[k].w + p_lo.w);
                    store_streaming_float4(
                        &out4[static_cast<size_t>(r) * W4 + c4], out_v);
                }
            }
        }
    }
}

// ============================================================================
// General Coalesced Fallback (Odd W, Small W < 128, H/W > 8192, or Unaligned)
// ============================================================================
__global__ __launch_bounds__(kThreads) void row_scan_warp_scalar_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int H,
    int W) {
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row_stride = static_cast<int>(gridDim.x) * kWarps;

    for (int r = static_cast<int>(blockIdx.x) * kWarps + warp; r < H; r += row_stride) {
        const size_t row_base = static_cast<size_t>(r) * static_cast<size_t>(W);
        float carry_hi = 0.0f;
        float carry_err = 0.0f;

        for (int c_base = 0; c_base < W; c_base += kWarp) {
            const int c = c_base + lane;
            const float v = (c < W) ? __ldg(&input[row_base + c]) : 0.0f;
            const float w_inc = warp_inclusive_scan_f32(v, lane);
            if (c < W) {
                output[row_base + c] = carry_hi + (w_inc + carry_err);
            }
            const float step_sum = __shfl_sync(kFullMask, w_inc, 31);
            const float y = step_sum + carry_err;
            const float t = carry_hi + y;
            carry_err = y - (t - carry_hi);
            carry_hi = t;
        }
    }
}

__global__ __launch_bounds__(kThreads) void col_scan_tiled32_kernel(
    float* __restrict__ output,
    int H,
    int W) {
    __shared__ float s_tile[32][33];

    const int col_base = static_cast<int>(blockIdx.x) * 32;
    const int tx = static_cast<int>(threadIdx.x) & 31;
    const int ty = static_cast<int>(threadIdx.x) >> 5;
    const int col = col_base + tx;

    double col_carry = 0.0;

    for (int r_base = 0; r_base < H; r_base += 32) {
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int r = r_base + m * 8 + ty;
            s_tile[m * 8 + ty][tx] =
                (r < H && col < W) ? output[static_cast<size_t>(r) * W + col] : 0.0f;
        }
        __syncthreads();

        if (ty == 0 && col < W) {
            #pragma unroll
            for (int k = 0; k < 32; ++k) {
                if (r_base + k < H) {
                    col_carry += static_cast<double>(s_tile[k][tx]);
                    s_tile[k][tx] = static_cast<float>(col_carry);
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int r = r_base + m * 8 + ty;
            if (r < H && col < W) {
                output[static_cast<size_t>(r) * W + col] = s_tile[m * 8 + ty][tx];
            }
        }
        __syncthreads();
    }
}

inline bool is_aligned_16b(const void* p0, const void* p1) {
    return ((reinterpret_cast<uintptr_t>(p0) | reinterpret_cast<uintptr_t>(p1)) & 15u) == 0u;
}

}  // namespace

extern "C" void solve(const float* input, float* output, int H, int W) {
    if (H <= 0 || W <= 0) {
        return;
    }

    const bool vec4_ok = ((W & 3) == 0) && is_aligned_16b(input, output);

    if (vec4_ok && W >= 128 && H <= kFastMaxDim && W <= kFastMaxDim) {
        const int W4 = W >> 2;
        const int num_col_tiles = (H > 1) ? ((W + 127) / 128) : 0;

        int row_blocks = (H + kWarps - 1) / kWarps;
        if (row_blocks < 1) row_blocks = 1;
        if (row_blocks > 1024) row_blocks = 1024;

        row_scan_warp_vec4_kernel<<<static_cast<unsigned>(row_blocks), kThreads>>>(
            reinterpret_cast<const float4*>(input),
            reinterpret_cast<float4*>(output),
            H,
            W4,
            num_col_tiles);

        if (H > 1) {
            const int num_y_strips = (H + 63) / 64;
            const dim3 grid(static_cast<unsigned>(num_col_tiles),
                            static_cast<unsigned>(num_y_strips));
            col_scan_onesweep_reg_vec4_kernel<<<grid, kThreads>>>(
                reinterpret_cast<float4*>(output),
                H,
                W,
                W4,
                num_col_tiles);
        }
        return;
    }

    int row_blocks = (H + kWarps - 1) / kWarps;
    if (row_blocks < 1) row_blocks = 1;
    if (row_blocks > 1024) row_blocks = 1024;
    row_scan_warp_scalar_kernel<<<static_cast<unsigned>(row_blocks), kThreads>>>(
        input, output, H, W);

    if (H > 1) {
        const int col_blocks = (W + 31) / 32;
        col_scan_tiled32_kernel<<<static_cast<unsigned>(col_blocks), kThreads>>>(output, H, W);
    }
}
