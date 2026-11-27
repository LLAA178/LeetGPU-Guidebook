#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

#ifndef POOL_Y_SPLITS
#define POOL_Y_SPLITS 4
#endif

namespace {

constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ void store_streaming_float4(float4* ptr, const float4& v) {
#if __CUDA_ARCH__ >= 700
    asm volatile("st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w));
#else
    *ptr = v;
#endif
}

__device__ __forceinline__ float4 max_float4_3(const float4& a, const float4& b, const float4& c) {
    float4 r;
    r.x = fmaxf(fmaxf(a.x, b.x), c.x);
    r.y = fmaxf(fmaxf(a.y, b.y), c.y);
    r.z = fmaxf(fmaxf(a.z, b.z), c.z);
    r.w = fmaxf(fmaxf(a.w, b.w), c.w);
    return r;
}

// ============================================================================
// Fast Path 1A: Exact Warp-Row Vertical-Strip Kernel for W == 256, K == 3, S == 2, P == 1!
// (Covers official benchmark N=4, C=64, H=256, W=256, K=3, S=2, P=1!)
// - Zero shared memory, zero __syncthreads(), zero horizontal halo loads!
// - 1 warp (32 lanes) spans the entire width W = 256:
//   * Pass 1: 32 lanes read 128B-coalesced float4 [0..127] (in_row4[lane])
//   * Pass 2: 32 lanes read 128B-coalesced float4 [128..255] (in_row4[32 + lane])
//   * Rearrange adjacent pairs via 2 warp shuffles so lane `l` holds [8*l .. 8*l+7],
//     and obtain `left = in_row[8*l - 1]` via 1 __shfl_up_sync(b.w, 1)!
//   * Vertical reuse: row `2*oh + 1`'s horizontal 3-max float4 `r1` becomes `prev` for `oh + 1`,
//     so only the single boundary row between Y-strips is loaded as halo!
// ============================================================================
__device__ __forceinline__ float4 load_and_horiz_max_w256(const float4* __restrict__ in_row4,
                                                          int lane) {
    const float4 lo = __ldg(&in_row4[lane]);       // [4*lane .. 4*lane+3]
    const float4 hi = __ldg(&in_row4[32 + lane]);  // [128 + 4*lane .. 128 + 4*lane+3]

    // Lane l < 16 wants (lo[2*l], lo[2*l+1]); lane l >= 16 wants (hi[2*(l-16)], hi[2*(l-16)+1]).
    const int src_even = (lane & 15) << 1;
    const int src_odd = src_even | 1;
    const float4 sel_src_a = (lane < 16) ? lo : hi;
    // Wait: __shfl_sync reads from `src_lane` whatever value `src_lane` passes in!
    // Notice `src_even` (0..30) needs to provide `lo` to lanes 0..15 AND `hi` to lanes 16..31!
    // So just shuffle `lo` for the first half and `hi` for the second half — 100% uniform!
    float4 lo_even, lo_odd, hi_even, hi_odd;
    lo_even.x = __shfl_sync(kFullMask, lo.x, src_even);
    lo_even.y = __shfl_sync(kFullMask, lo.y, src_even);
    lo_even.z = __shfl_sync(kFullMask, lo.z, src_even);
    lo_even.w = __shfl_sync(kFullMask, lo.w, src_even);

    lo_odd.x = __shfl_sync(kFullMask, lo.x, src_odd);
    lo_odd.y = __shfl_sync(kFullMask, lo.y, src_odd);
    lo_odd.z = __shfl_sync(kFullMask, lo.z, src_odd);
    lo_odd.w = __shfl_sync(kFullMask, lo.w, src_odd);

    hi_even.x = __shfl_sync(kFullMask, hi.x, src_even);
    hi_even.y = __shfl_sync(kFullMask, hi.y, src_even);
    hi_even.z = __shfl_sync(kFullMask, hi.z, src_even);
    hi_even.w = __shfl_sync(kFullMask, hi.w, src_even);

    hi_odd.x = __shfl_sync(kFullMask, hi.x, src_odd);
    hi_odd.y = __shfl_sync(kFullMask, hi.y, src_odd);
    hi_odd.z = __shfl_sync(kFullMask, hi.z, src_odd);
    hi_odd.w = __shfl_sync(kFullMask, hi.w, src_odd);
    (void)sel_src_a;

    const float4 a = (lane < 16) ? lo_even : hi_even;
    const float4 b = (lane < 16) ? lo_odd : hi_odd;

    // Element 8*lane - 1 is precisely `b.w` of `lane - 1`!
    const float prev_bw = __shfl_up_sync(kFullMask, b.w, 1);
    const float left = (lane == 0) ? (-INFINITY) : prev_bw;

    float4 h;
    h.x = fmaxf(fmaxf(left, a.x), a.y);
    h.y = fmaxf(fmaxf(a.y, a.z), a.w);
    h.z = fmaxf(fmaxf(a.w, b.x), b.y);
    h.w = fmaxf(fmaxf(b.y, b.z), b.w);
    return h;
}

__global__ __launch_bounds__(128, 8) void max_pool2d_w256_k3s2p1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int nc_total,
    int H,
    int H_out) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;  // 0..3 (4 warps per block)
    const int strip = blockIdx.x;
    const int num_strips = gridDim.x;
    const int nc = static_cast<int>(blockIdx.y) * 4 + warp;

    if (nc >= nc_total) {
        return;
    }

    const int oh_start = (H_out * strip) / num_strips;
    const int oh_end = (H_out * (strip + 1)) / num_strips;
    if (oh_start >= oh_end) {
        return;
    }

    // W == 256 -> 64 float4s per input row; W_out == 128 -> 32 float4s per output row!
    const float4* __restrict__ plane_in4 =
        reinterpret_cast<const float4*>(input) + static_cast<size_t>(nc) * static_cast<size_t>(H) * 64ull;
    float4* __restrict__ plane_out4 =
        reinterpret_cast<float4*>(output) + static_cast<size_t>(nc) * static_cast<size_t>(H_out) * 32ull;

    float4 prev;
    if (oh_start == 0) {
        prev = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
    } else {
        const int top_ih = 2 * oh_start - 1;
        prev = load_and_horiz_max_w256(plane_in4 + static_cast<size_t>(top_ih) * 64ull, lane);
    }

    for (int oh = oh_start; oh < oh_end; ++oh) {
        const int ih0 = oh << 1;
        const int ih1 = ih0 + 1;

        const float4 r0 = load_and_horiz_max_w256(plane_in4 + static_cast<size_t>(ih0) * 64ull, lane);
        float4 r1;
        if (ih1 < H) {
            r1 = load_and_horiz_max_w256(plane_in4 + static_cast<size_t>(ih1) * 64ull, lane);
        } else {
            r1 = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
        }

        const float4 out_v = max_float4_3(prev, r0, r1);
        store_streaming_float4(&plane_out4[static_cast<size_t>(oh) * 32ull + static_cast<size_t>(lane)],
                               out_v);
        prev = r1;
    }
}

// ============================================================================
// Fast Path 1B: General Aligned K=3, S=2, P=1 Shared-Memory Tile Kernel for any W % 8 == 0!
// Block (32 x 4 = 128 threads) computes 8 output rows x 128 output cols (256 input cols)
// using padded shared memory s_in[18][260] (18.3 KiB -> 2 blocks/SM) and float4 stores!
// ============================================================================
__global__ __launch_bounds__(128) void max_pool2d_tiled_k3s2p1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int H,
    int W,
    int H_out,
    int W_out) {
    __shared__ float s_in[18][260];

    const int tid = threadIdx.x;
    const int tx = tid & 31;   // 0..31 -> 4 output cols each = 128 output cols
    const int ty = tid >> 5;   // 0..3  -> 2 output rows each = 8 output rows
    const int bx_out = static_cast<int>(blockIdx.x) * 128;
    const int by_out = static_cast<int>(blockIdx.y) * 8;
    const int nc = blockIdx.z;

    const int in_y0 = by_out * 2 - 1;
    const int in_x0 = bx_out * 2;  // 16-byte aligned since bx_out % 128 == 0 and W % 8 == 0!

    const float* __restrict__ plane_in =
        input + static_cast<size_t>(nc) * static_cast<size_t>(H) * static_cast<size_t>(W);
    float* __restrict__ plane_out =
        output + static_cast<size_t>(nc) * static_cast<size_t>(H_out) * static_cast<size_t>(W_out);

    // Load 18 rows x 258 cols: s_in[r][0] is left halo (in_x0 - 1),
    // s_in[r][1..256] are 64 aligned float4s [in_x0 .. in_x0 + 255],
    // s_in[r][257] is right halo (in_x0 + 256).
    for (int idx = tid; idx < 18 * 64; idx += 128) {
        const int r = idx >> 6;
        const int c4 = idx & 63;
        const int iy = in_y0 + r;
        const int ix = in_x0 + (c4 << 2);
        float4 v = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
        if (iy >= 0 && iy < H && ix < W) {
            if (ix + 3 < W) {
                v = __ldg(reinterpret_cast<const float4*>(
                    plane_in + static_cast<size_t>(iy) * static_cast<size_t>(W) + static_cast<size_t>(ix)));
            } else {
                v.x = __ldg(&plane_in[static_cast<size_t>(iy) * static_cast<size_t>(W) + ix]);
                if (ix + 1 < W) v.y = __ldg(&plane_in[static_cast<size_t>(iy) * static_cast<size_t>(W) + ix + 1]);
                if (ix + 2 < W) v.z = __ldg(&plane_in[static_cast<size_t>(iy) * static_cast<size_t>(W) + ix + 2]);
            }
        }
        const int sc = 1 + (c4 << 2);
        s_in[r][sc + 0] = v.x;
        s_in[r][sc + 1] = v.y;
        s_in[r][sc + 2] = v.z;
        s_in[r][sc + 3] = v.w;
    }

    if (tid < 36) {
        const int r = tid >> 1;
        const bool is_right = (tid & 1) != 0;
        const int iy = in_y0 + r;
        const int ix = is_right ? (in_x0 + 256) : (in_x0 - 1);
        float val = -INFINITY;
        if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
            val = __ldg(&plane_in[static_cast<size_t>(iy) * static_cast<size_t>(W) + static_cast<size_t>(ix)]);
        }
        s_in[r][is_right ? 257 : 0] = val;
    }
    __syncthreads();

    const int ow0 = bx_out + (tx << 2);
    if (ow0 < W_out) {
        const int sc0 = tx << 3;  // s_in[r][sc0] corresponds to in_x0 - 1 + 8*tx!
        const int r_base = ty << 2;

        float h_rows[5][4];
#pragma unroll
        for (int dr = 0; dr < 5; ++dr) {
            const float* sr = s_in[r_base + dr] + sc0;
            const float a0 = sr[1], a1 = sr[2], a2 = sr[3], a3 = sr[4];
            const float b0 = sr[5], b1 = sr[6], b2 = sr[7], b3 = sr[8];
            h_rows[dr][0] = fmaxf(fmaxf(sr[0], a0), a1);
            h_rows[dr][1] = fmaxf(fmaxf(a1, a2), a3);
            h_rows[dr][2] = fmaxf(fmaxf(a3, b0), b1);
            h_rows[dr][3] = fmaxf(fmaxf(b1, b2), b3);
        }

#pragma unroll
        for (int sub_y = 0; sub_y < 2; ++sub_y) {
            const int oh = by_out + (ty << 1) + sub_y;
            if (oh < H_out) {
                const int d0 = sub_y << 1;
                float4 out_v;
                out_v.x = fmaxf(fmaxf(h_rows[d0][0], h_rows[d0 + 1][0]), h_rows[d0 + 2][0]);
                out_v.y = fmaxf(fmaxf(h_rows[d0][1], h_rows[d0 + 1][1]), h_rows[d0 + 2][1]);
                out_v.z = fmaxf(fmaxf(h_rows[d0][2], h_rows[d0 + 1][2]), h_rows[d0 + 2][2]);
                out_v.w = fmaxf(fmaxf(h_rows[d0][3], h_rows[d0 + 1][3]), h_rows[d0 + 2][3]);

                float* out_ptr =
                    plane_out + static_cast<size_t>(oh) * static_cast<size_t>(W_out) + static_cast<size_t>(ow0);
                if (ow0 + 3 < W_out && ((reinterpret_cast<uintptr_t>(out_ptr) & 15u) == 0u)) {
                    store_streaming_float4(reinterpret_cast<float4*>(out_ptr), out_v);
                } else {
                    out_ptr[0] = out_v.x;
                    if (ow0 + 1 < W_out) out_ptr[1] = out_v.y;
                    if (ow0 + 2 < W_out) out_ptr[2] = out_v.z;
                    if (ow0 + 3 < W_out) out_ptr[3] = out_v.w;
                }
            }
        }
    }
}

// ============================================================================
// Fast Path 2: Pure-Register Zero-Halo float4 Kernel for K=2, S=2, P=0 (W % 8 == 0, H % 2 == 0)!
// ============================================================================
__global__ __launch_bounds__(256) void max_pool2d_k2s2p0_vec4_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int H,
    int W,
    int H_out,
    int W_out4,
    size_t total_vec4_outputs) {
    const size_t global_tid = static_cast<size_t>(blockIdx.x) * 256ull + static_cast<size_t>(threadIdx.x);
    const size_t stride = static_cast<size_t>(gridDim.x) * 256ull;
    const size_t plane_out4 = static_cast<size_t>(H_out) * static_cast<size_t>(W_out4);
    const size_t W4 = static_cast<size_t>(W >> 2);

    const float4* __restrict__ in4 = reinterpret_cast<const float4*>(input);
    float4* __restrict__ out4 = reinterpret_cast<float4*>(output);

    for (size_t idx4 = global_tid; idx4 < total_vec4_outputs; idx4 += stride) {
        const size_t nc = idx4 / plane_out4;
        const size_t rem = idx4 - nc * plane_out4;
        const size_t oh = rem / static_cast<size_t>(W_out4);
        const size_t ow4 = rem - oh * static_cast<size_t>(W_out4);

        const size_t ih0 = (nc * static_cast<size_t>(H) + (oh << 1)) * W4 + (ow4 << 1);
        const size_t ih1 = ih0 + W4;

        const float4 r0a = __ldg(&in4[ih0]);
        const float4 r0b = __ldg(&in4[ih0 + 1]);
        const float4 r1a = __ldg(&in4[ih1]);
        const float4 r1b = __ldg(&in4[ih1 + 1]);

        float4 res;
        res.x = fmaxf(fmaxf(r0a.x, r0a.y), fmaxf(r1a.x, r1a.y));
        res.y = fmaxf(fmaxf(r0a.z, r0a.w), fmaxf(r1a.z, r1a.w));
        res.z = fmaxf(fmaxf(r0b.x, r0b.y), fmaxf(r1b.x, r1b.y));
        res.w = fmaxf(fmaxf(r0b.z, r0b.w), fmaxf(r1b.z, r1b.w));
        store_streaming_float4(&out4[idx4], res);
    }
}

// ============================================================================
// Fast Path 3: Shared-Memory Halo-Tiled Kernel for General K <= 7, stride <= 4!
// Block (32 x 8 = 256 threads) computes an 8 x 32 output tile; max smem tile is
// (7*4 + 7) x (31*4 + 7) = 35 x 131 floats = 18.3 KiB!
// ============================================================================
__global__ __launch_bounds__(256) void max_pool2d_smem_general_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int H,
    int W,
    int H_out,
    int W_out,
    int kernel_size,
    int stride,
    int padding) {
    __shared__ float s_tile[36 * 136];

    const int tx = threadIdx.x & 31;
    const int ty = threadIdx.x >> 5;  // 0..7
    const int tid = threadIdx.x;

    const int ow_base = static_cast<int>(blockIdx.x) * 32;
    const int oh_base = static_cast<int>(blockIdx.y) * 8;
    const int nc = blockIdx.z;

    const int tile_h = 7 * stride + kernel_size;
    const int tile_w = 31 * stride + kernel_size;
    const int in_h0 = oh_base * stride - padding;
    const int in_w0 = ow_base * stride - padding;

    const float* __restrict__ plane_in =
        input + static_cast<size_t>(nc) * static_cast<size_t>(H) * static_cast<size_t>(W);
    float* __restrict__ plane_out =
        output + static_cast<size_t>(nc) * static_cast<size_t>(H_out) * static_cast<size_t>(W_out);

    const int total_tile = tile_h * tile_w;
    for (int i = tid; i < total_tile; i += 256) {
        const int r = i / tile_w;
        const int c = i - r * tile_w;
        const int ih = in_h0 + r;
        const int iw = in_w0 + c;
        float v = -INFINITY;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            v = __ldg(&plane_in[static_cast<size_t>(ih) * static_cast<size_t>(W) + static_cast<size_t>(iw)]);
        }
        s_tile[r * 136 + c] = v;
    }
    __syncthreads();

    const int ow = ow_base + tx;
    const int oh = oh_base + ty;
    if (ow < W_out && oh < H_out) {
        const int sr0 = ty * stride;
        const int sc0 = tx * stride;
        float max_val = -INFINITY;
        for (int kh = 0; kh < kernel_size; ++kh) {
            const float* row_ptr = s_tile + (sr0 + kh) * 136 + sc0;
            for (int kw = 0; kw < kernel_size; ++kw) {
                max_val = fmaxf(max_val, row_ptr[kw]);
            }
        }
        plane_out[static_cast<size_t>(oh) * static_cast<size_t>(W_out) + static_cast<size_t>(ow)] =
            max_val;
    }
}

// ============================================================================
// General Fallback: Arbitrary (N, C, H, W, K <= 16+, stride <= 16+, padding <= 16+)
// ============================================================================
__global__ __launch_bounds__(256) void max_pool2d_fallback_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int H,
    int W,
    int H_out,
    int W_out,
    int kernel_size,
    int stride,
    int padding) {
    const int ow = static_cast<int>(blockIdx.x) * 32 + (threadIdx.x & 31);
    const int oh = static_cast<int>(blockIdx.y) * 8 + (threadIdx.x >> 5);
    const int nc = blockIdx.z;

    if (ow >= W_out || oh >= H_out) {
        return;
    }

    const float* __restrict__ plane_in =
        input + static_cast<size_t>(nc) * static_cast<size_t>(H) * static_cast<size_t>(W);
    float* __restrict__ plane_out =
        output + static_cast<size_t>(nc) * static_cast<size_t>(H_out) * static_cast<size_t>(W_out);

    const int h0 = oh * stride - padding;
    const int w0 = ow * stride - padding;
    const int kh_start = (h0 < 0) ? (-h0) : 0;
    const int kh_end = (h0 + kernel_size > H) ? (H - h0) : kernel_size;
    const int kw_start = (w0 < 0) ? (-w0) : 0;
    const int kw_end = (w0 + kernel_size > W) ? (W - w0) : kernel_size;

    float max_val = -INFINITY;
    for (int kh = kh_start; kh < kh_end; ++kh) {
        const float* __restrict__ in_row =
            plane_in + static_cast<size_t>(h0 + kh) * static_cast<size_t>(W);
        for (int kw = kw_start; kw < kw_end; ++kw) {
            max_val = fmaxf(max_val, __ldg(&in_row[w0 + kw]));
        }
    }

    plane_out[static_cast<size_t>(oh) * static_cast<size_t>(W_out) + static_cast<size_t>(ow)] =
        max_val;
}

}  // namespace

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input,
                      float* output,
                      int N,
                      int C,
                      int H,
                      int W,
                      int kernel_size,
                      int stride,
                      int padding) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || kernel_size <= 0 || stride <= 0) {
        return;
    }

    const int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    const int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    if (H_out <= 0 || W_out <= 0) {
        return;
    }

    const int nc_total = N * C;
    const bool aligned_16 =
        (((reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(output)) & 15u) == 0u);

    // 1A. Exact W == 256, K == 3, S == 2, P == 1 pure-register warp-row strip kernel!
    if (aligned_16 && W == 256 && kernel_size == 3 && stride == 2 && padding == 1 && H >= 2) {
        int y_splits = POOL_Y_SPLITS;
        if (y_splits > H_out) y_splits = H_out;
        if (y_splits < 1) y_splits = 1;
        const int nc_blocks = (nc_total + 3) / 4;
        for (int b_base = 0; b_base < nc_blocks; b_base += 65535) {
            const int b_chunk = (nc_blocks - b_base < 65535) ? (nc_blocks - b_base) : 65535;
            const int plane_base = b_base * 4;
            const int rem_planes = nc_total - plane_base;
            const size_t in_off = static_cast<size_t>(plane_base) * static_cast<size_t>(H) * 256ull;
            const size_t out_off = static_cast<size_t>(plane_base) * static_cast<size_t>(H_out) * 128ull;
            const dim3 grid(static_cast<unsigned>(y_splits), static_cast<unsigned>(b_chunk), 1u);
            max_pool2d_w256_k3s2p1_kernel<<<grid, 128>>>(
                input + in_off, output + out_off, rem_planes, H, H_out);
        }
        return;
    }

    // 1B. General aligned K == 3, S == 2, P == 1 shared-memory tiled float4 kernel for W % 8 == 0!
    if (aligned_16 && (W & 7) == 0 && (W_out & 3) == 0 &&
        kernel_size == 3 && stride == 2 && padding == 1) {
        for (int z_base = 0; z_base < nc_total; z_base += 65535) {
            const int z_chunk = (nc_total - z_base < 65535) ? (nc_total - z_base) : 65535;
            const dim3 grid(static_cast<unsigned>((W_out + 127) / 128),
                            static_cast<unsigned>((H_out + 7) / 8),
                            static_cast<unsigned>(z_chunk));
            const size_t in_off =
                static_cast<size_t>(z_base) * static_cast<size_t>(H) * static_cast<size_t>(W);
            const size_t out_off =
                static_cast<size_t>(z_base) * static_cast<size_t>(H_out) * static_cast<size_t>(W_out);
            max_pool2d_tiled_k3s2p1_kernel<<<grid, 128>>>(
                input + in_off, output + out_off, H, W, H_out, W_out);
        }
        return;
    }

    // 2. Pure-register zero-halo float4 fast path for K == 2, S == 2, P == 0!
    if (aligned_16 && kernel_size == 2 && stride == 2 && padding == 0 &&
        (W & 7) == 0 && (H & 1) == 0) {
        const int W_out4 = W_out >> 2;
        const size_t total_vec4 =
            static_cast<size_t>(nc_total) * static_cast<size_t>(H_out) * static_cast<size_t>(W_out4);
        size_t blocks = (total_vec4 + 255ull) / 256ull;
        if (blocks > 4096ull) blocks = 4096ull;
        if (blocks < 1ull) blocks = 1ull;
        const unsigned grid_blocks = static_cast<unsigned>(blocks);
        max_pool2d_k2s2p0_vec4_kernel<<<grid_blocks, 256>>>(
            input, output, H, W, H_out, W_out4, total_vec4);
        return;
    }

    // 3. Shared-memory halo-tiled kernel for general K in [2, 7] and stride <= 4!
    if (kernel_size >= 2 && kernel_size <= 7 && stride <= 4) {
        for (int z_base = 0; z_base < nc_total; z_base += 65535) {
            const int z_chunk = (nc_total - z_base < 65535) ? (nc_total - z_base) : 65535;
            const dim3 grid(static_cast<unsigned>((W_out + 31) / 32),
                            static_cast<unsigned>((H_out + 7) / 8),
                            static_cast<unsigned>(z_chunk));
            const size_t in_off =
                static_cast<size_t>(z_base) * static_cast<size_t>(H) * static_cast<size_t>(W);
            const size_t out_off =
                static_cast<size_t>(z_base) * static_cast<size_t>(H_out) * static_cast<size_t>(W_out);
            max_pool2d_smem_general_kernel<<<grid, 256>>>(
                input + in_off, output + out_off, H, W, H_out, W_out, kernel_size, stride, padding);
        }
        return;
    }

    // 4. Universal fallback (K == 1, K > 7, stride > 4, etc.)
    for (int z_base = 0; z_base < nc_total; z_base += 65535) {
        const int z_chunk = (nc_total - z_base < 65535) ? (nc_total - z_base) : 65535;
        const dim3 grid(static_cast<unsigned>((W_out + 31) / 32),
                        static_cast<unsigned>((H_out + 7) / 8),
                        static_cast<unsigned>(z_chunk));
        const size_t in_off =
            static_cast<size_t>(z_base) * static_cast<size_t>(H) * static_cast<size_t>(W);
        const size_t out_off =
            static_cast<size_t>(z_base) * static_cast<size_t>(H_out) * static_cast<size_t>(W_out);
        max_pool2d_fallback_kernel<<<grid, 256>>>(
            input + in_off, output + out_off, H, W, H_out, W_out, kernel_size, stride, padding);
    }
}
