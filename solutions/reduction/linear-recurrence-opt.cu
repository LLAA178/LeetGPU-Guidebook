#include <cuda_runtime.h>
#include <cstdint>

#ifndef LR_THREADS
#define LR_THREADS 512
#endif

namespace {

constexpr int kWarp = 32;
constexpr int kThreads = LR_THREADS;
static_assert(kThreads >= kWarp && kThreads <= 1024 && (kThreads & (kThreads - 1)) == 0,
              "LR_THREADS must be a power of two in [32, 1024]");
constexpr int kWarps = kThreads / kWarp;
constexpr int kTile = kThreads * 4;
constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ void warp_inclusive_scan_affine(float& m, float& v, int lane) {
#pragma unroll
    for (int step = 1; step < kWarp; step <<= 1) {
        const float prev_m = __shfl_up_sync(kFullMask, m, step);
        const float prev_v = __shfl_up_sync(kFullMask, v, step);
        if (lane >= step) {
            v = fmaf(m, prev_v, v);
            m *= prev_m;
        }
    }
}

__device__ __forceinline__ void scan_warp_totals(const float* s_warp_m,
                                                 const float* s_warp_v,
                                                 float* s_warp_h_in,
                                                 float* s_carry,
                                                 int lane) {
    float wm = (lane < kWarps) ? s_warp_m[lane] : 1.0f;
    float wv = (lane < kWarps) ? s_warp_v[lane] : 0.0f;
    const float carry = *s_carry;
    if (lane == 0) {
        s_warp_h_in[0] = carry;
        wv = fmaf(wm, carry, wv);
        wm = 0.0f;
    }
#pragma unroll
    for (int step = 1; step < kWarps; step <<= 1) {
        const float pm = __shfl_up_sync(kFullMask, wm, step);
        const float pv = __shfl_up_sync(kFullMask, wv, step);
        if (lane >= step) {
            wv = fmaf(wm, pv, wv);
            wm *= pm;
        }
    }
    const float prev_wv = __shfl_up_sync(kFullMask, wv, 1);
    if (lane > 0 && lane < kWarps) {
        s_warp_h_in[lane] = prev_wv;
    }
    if (lane == kWarps - 1) {
        *s_carry = wv;
    }
}

// Fast path: 16-byte aligned base pointers and L divisible by kTile (e.g. B=64, L=16384).
// Every global read/write is a 128-bit coalesced float4 transaction with zero tail branches.
__global__ __launch_bounds__(kThreads) void linear_recurrence_aligned_full_kernel(
    const float* __restrict__ a,
    const float* __restrict__ x,
    float* __restrict__ h,
    int L) {
    __shared__ float s_warp_m[kWarps];
    __shared__ float s_warp_v[kWarps];
    __shared__ float s_warp_h_in[kWarps];
    __shared__ float s_carry;

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & (kWarp - 1);
    const int warp = tid >> 5;

    const size_t row_offset = static_cast<size_t>(b) * static_cast<size_t>(L);
    const float4* __restrict__ row_a4 = reinterpret_cast<const float4*>(a + row_offset);
    const float4* __restrict__ row_x4 = reinterpret_cast<const float4*>(x + row_offset);
    float4* __restrict__ row_h4 = reinterpret_cast<float4*>(h + row_offset);

    if (tid == 0) {
        s_carry = 0.0f;
    }

    const int num_tiles = L / kTile;
    for (int tile = 0; tile < num_tiles; ++tile) {
        const int vec_idx = tile * kThreads + tid;
        float4 qa = __ldg(&row_a4[vec_idx]);
        const float4 qx = __ldg(&row_x4[vec_idx]);

        if (vec_idx == 0) {
            qa.x = 0.0f;
        }

        float m = qa.x;
        float v = qx.x;
        m = qa.y * m;
        v = fmaf(qa.y, v, qx.y);
        m = qa.z * m;
        v = fmaf(qa.z, v, qx.z);
        m = qa.w * m;
        v = fmaf(qa.w, v, qx.w);

        warp_inclusive_scan_affine(m, v, lane);

        if (lane == kWarp - 1) {
            s_warp_m[warp] = m;
            s_warp_v[warp] = v;
        }
        __syncthreads();

        if (warp == 0) {
            scan_warp_totals(s_warp_m, s_warp_v, s_warp_h_in, &s_carry, lane);
        }
        __syncthreads();

        const float prev_m = __shfl_up_sync(kFullMask, m, 1);
        const float prev_v = __shfl_up_sync(kFullMask, v, 1);
        const float h_in_w = s_warp_h_in[warp];
        const float h_prev = (lane == 0) ? h_in_w : fmaf(prev_m, h_in_w, prev_v);

        float4 qh;
        qh.x = fmaf(qa.x, h_prev, qx.x);
        qh.y = fmaf(qa.y, qh.x, qx.y);
        qh.z = fmaf(qa.z, qh.y, qx.z);
        qh.w = fmaf(qa.w, qh.z, qx.w);
        row_h4[vec_idx] = qh;

        if (tile + 1 < num_tiles) {
            __syncthreads();
        }
    }
}

// General path: handles arbitrary B >= 1, L >= 1, non-multiple-of-4 L, and misaligned pointers.
__global__ __launch_bounds__(kThreads) void linear_recurrence_general_kernel(
    const float* __restrict__ a,
    const float* __restrict__ x,
    float* __restrict__ h,
    int L) {
    __shared__ float s_warp_m[kWarps];
    __shared__ float s_warp_v[kWarps];
    __shared__ float s_warp_h_in[kWarps];
    __shared__ float s_carry;

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & (kWarp - 1);
    const int warp = tid >> 5;

    const size_t row_offset = static_cast<size_t>(b) * static_cast<size_t>(L);
    const float* __restrict__ row_a = a + row_offset;
    const float* __restrict__ row_x = x + row_offset;
    float* __restrict__ row_h = h + row_offset;

    const bool row_aligned =
        (((reinterpret_cast<uintptr_t>(row_a) |
           reinterpret_cast<uintptr_t>(row_x) |
           reinterpret_cast<uintptr_t>(row_h)) &
          15u) == 0u);
    const int l4 = L >> 2;

    if (tid == 0) {
        s_carry = 0.0f;
    }

    const int num_tiles = (L + kTile - 1) / kTile;
    for (int tile = 0; tile < num_tiles; ++tile) {
        const int vec_idx = tile * kThreads + tid;
        const int base_t = vec_idx << 2;

        float4 qa = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        float4 qx = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (row_aligned && vec_idx < l4) {
            qa = __ldg(&reinterpret_cast<const float4*>(row_a)[vec_idx]);
            qx = __ldg(&reinterpret_cast<const float4*>(row_x)[vec_idx]);
        } else if (base_t < L) {
            qa.x = __ldg(&row_a[base_t]);
            qx.x = __ldg(&row_x[base_t]);
            if (base_t + 1 < L) {
                qa.y = __ldg(&row_a[base_t + 1]);
                qx.y = __ldg(&row_x[base_t + 1]);
            }
            if (base_t + 2 < L) {
                qa.z = __ldg(&row_a[base_t + 2]);
                qx.z = __ldg(&row_x[base_t + 2]);
            }
            if (base_t + 3 < L) {
                qa.w = __ldg(&row_a[base_t + 3]);
                qx.w = __ldg(&row_x[base_t + 3]);
            }
        }

        if (base_t == 0) {
            qa.x = 0.0f;
        }

        float m = qa.x;
        float v = qx.x;
        m = qa.y * m;
        v = fmaf(qa.y, v, qx.y);
        m = qa.z * m;
        v = fmaf(qa.z, v, qx.z);
        m = qa.w * m;
        v = fmaf(qa.w, v, qx.w);

        warp_inclusive_scan_affine(m, v, lane);

        if (lane == kWarp - 1) {
            s_warp_m[warp] = m;
            s_warp_v[warp] = v;
        }
        __syncthreads();

        if (warp == 0) {
            scan_warp_totals(s_warp_m, s_warp_v, s_warp_h_in, &s_carry, lane);
        }
        __syncthreads();

        const float prev_m = __shfl_up_sync(kFullMask, m, 1);
        const float prev_v = __shfl_up_sync(kFullMask, v, 1);
        const float h_in_w = s_warp_h_in[warp];
        const float h_prev = (lane == 0) ? h_in_w : fmaf(prev_m, h_in_w, prev_v);

        float4 qh;
        qh.x = fmaf(qa.x, h_prev, qx.x);
        qh.y = fmaf(qa.y, qh.x, qx.y);
        qh.z = fmaf(qa.z, qh.y, qx.z);
        qh.w = fmaf(qa.w, qh.z, qx.w);

        if (row_aligned && vec_idx < l4) {
            reinterpret_cast<float4*>(row_h)[vec_idx] = qh;
        } else if (base_t < L) {
            row_h[base_t] = qh.x;
            if (base_t + 1 < L) row_h[base_t + 1] = qh.y;
            if (base_t + 2 < L) row_h[base_t + 2] = qh.z;
            if (base_t + 3 < L) row_h[base_t + 3] = qh.w;
        }

        if (tile + 1 < num_tiles) {
            __syncthreads();
        }
    }
}

}  // namespace

// a, x, h are device pointers
extern "C" void solve(const float* a, const float* x, float* h, int B, int L) {
    if (B <= 0 || L <= 0) {
        return;
    }

    const bool base_aligned =
        (((reinterpret_cast<uintptr_t>(a) |
           reinterpret_cast<uintptr_t>(x) |
           reinterpret_cast<uintptr_t>(h)) &
          15u) == 0u);

    if (base_aligned && (L % kTile == 0)) {
        linear_recurrence_aligned_full_kernel<<<B, kThreads>>>(a, x, h, L);
    } else {
        linear_recurrence_general_kernel<<<B, kThreads>>>(a, x, h, L);
    }
}
