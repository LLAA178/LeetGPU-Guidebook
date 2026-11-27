#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

#ifndef CONV1D_THREADS
#define CONV1D_THREADS 128
#endif

#ifndef CONV1D_LTILE
#define CONV1D_LTILE 128
#endif

static_assert(CONV1D_THREADS >= 32 && (CONV1D_THREADS % 32 == 0) && CONV1D_THREADS <= 1024,
              "CONV1D_THREADS must be a multiple of 32 in [32, 1024]");
static_assert(CONV1D_LTILE >= 4 && CONV1D_LTILE <= 512,
              "CONV1D_LTILE must be in [4, 512]");

__device__ __forceinline__ float4 zero_float4() {
    return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

__device__ __forceinline__ float4 fma_float4(const float4& w, const float4& x, const float4& acc) {
    return make_float4(
        fmaf(w.x, x.x, acc.x),
        fmaf(w.y, x.y, acc.y),
        fmaf(w.z, x.z, acc.z),
        fmaf(w.w, x.w, acc.w));
}

// Compile-time unrolled register sliding-window float4 kernel for K in {1..8}.
// Each thread handles 1 float4 (4 contiguous channels) across kLTile consecutive timesteps l,
// reading weight/bias once per block and streaming 1 coalesced float4 load + 1 coalesced float4 store per step.
template <int K, int kThreads, int kLTile>
__global__ __launch_bounds__(kThreads) void causal_conv1d_vec4_fixed_k_kernel(
    const float4* __restrict__ x4,
    const float* __restrict__ weight,
    const float4* __restrict__ bias4,
    float4* __restrict__ out4,
    int L,
    int D4) {
    const int d4 = blockIdx.x * kThreads + threadIdx.x;
    if (d4 >= D4) {
        return;
    }

    const int l_start = blockIdx.y * kLTile;
    const int b = blockIdx.z;

    const float4 b_vec = __ldg(&bias4[d4]);
    const int d0 = d4 << 2;

    float4 w_tap[K];
    if (K == 4 && ((reinterpret_cast<uintptr_t>(weight) & 15u) == 0u)) {
        const float4* w4 = reinterpret_cast<const float4*>(weight + d0 * 4);
        const float4 c0 = __ldg(&w4[0]);
        const float4 c1 = __ldg(&w4[1]);
        const float4 c2 = __ldg(&w4[2]);
        const float4 c3 = __ldg(&w4[3]);
        w_tap[0] = make_float4(c0.x, c1.x, c2.x, c3.x);
        w_tap[1] = make_float4(c0.y, c1.y, c2.y, c3.y);
        w_tap[2] = make_float4(c0.z, c1.z, c2.z, c3.z);
        w_tap[3] = make_float4(c0.w, c1.w, c2.w, c3.w);
    } else {
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            w_tap[k] = make_float4(
                __ldg(&weight[(d0 + 0) * K + k]),
                __ldg(&weight[(d0 + 1) * K + k]),
                __ldg(&weight[(d0 + 2) * K + k]),
                __ldg(&weight[(d0 + 3) * K + k]));
        }
    }

    const size_t batch_offset4 = static_cast<size_t>(b) * static_cast<size_t>(L) * static_cast<size_t>(D4)
                                 + static_cast<size_t>(d4);
    const size_t stride4 = static_cast<size_t>(D4);

    // hist[0] holds x[l - 1], hist[1] holds x[l - 2], ..., hist[K - 2] holds x[l - (K - 1)].
    constexpr int kHistLen = (K > 1) ? (K - 1) : 1;
    float4 hist[kHistLen];
    if (K > 1) {
        #pragma unroll
        for (int h = 0; h < K - 1; ++h) {
            const int hl = l_start - 1 - h;
            hist[h] = (hl >= 0)
                          ? __ldg(&x4[batch_offset4 + static_cast<size_t>(hl) * stride4])
                          : zero_float4();
        }
    }

    const int rem_l = L - l_start;
    if (rem_l >= kLTile) {
        size_t ptr_idx = batch_offset4 + static_cast<size_t>(l_start) * stride4;
        #pragma unroll 8
        for (int s = 0; s < kLTile; ++s) {
            const float4 x_cur = __ldg(&x4[ptr_idx]);
            float4 acc = fma_float4(w_tap[0], x_cur, b_vec);
            if (K > 1) {
                #pragma unroll
                for (int k = 1; k < K; ++k) {
                    acc = fma_float4(w_tap[k], hist[k - 1], acc);
                }
                #pragma unroll
                for (int h = K - 2; h > 0; --h) {
                    hist[h] = hist[h - 1];
                }
                hist[0] = x_cur;
            }
            out4[ptr_idx] = acc;
            ptr_idx += stride4;
        }
    } else {
        size_t ptr_idx = batch_offset4 + static_cast<size_t>(l_start) * stride4;
        for (int s = 0; s < rem_l; ++s) {
            const float4 x_cur = __ldg(&x4[ptr_idx]);
            float4 acc = fma_float4(w_tap[0], x_cur, b_vec);
            if (K > 1) {
                #pragma unroll
                for (int k = 1; k < K; ++k) {
                    acc = fma_float4(w_tap[k], hist[k - 1], acc);
                }
                #pragma unroll
                for (int h = K - 2; h > 0; --h) {
                    hist[h] = hist[h - 1];
                }
                hist[0] = x_cur;
            }
            out4[ptr_idx] = acc;
            ptr_idx += stride4;
        }
    }
}

// Scalar register sliding-window kernel for K in {1..8} when D % 4 != 0 or pointers are misaligned.
template <int K, int kThreads, int kLTile>
__global__ __launch_bounds__(kThreads) void causal_conv1d_scalar_fixed_k_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int L,
    int D) {
    const int d = blockIdx.x * kThreads + threadIdx.x;
    if (d >= D) {
        return;
    }

    const int l_start = blockIdx.y * kLTile;
    const int b = blockIdx.z;

    const float b_val = __ldg(&bias[d]);
    float w_tap[K];
    #pragma unroll
    for (int k = 0; k < K; ++k) {
        w_tap[k] = __ldg(&weight[d * K + k]);
    }

    const size_t batch_offset = static_cast<size_t>(b) * static_cast<size_t>(L) * static_cast<size_t>(D)
                                + static_cast<size_t>(d);
    const size_t stride = static_cast<size_t>(D);

    constexpr int kHistLen = (K > 1) ? (K - 1) : 1;
    float hist[kHistLen];
    if (K > 1) {
        #pragma unroll
        for (int h = 0; h < K - 1; ++h) {
            const int hl = l_start - 1 - h;
            hist[h] = (hl >= 0) ? __ldg(&x[batch_offset + static_cast<size_t>(hl) * stride]) : 0.0f;
        }
    }

    const int s_end = (L - l_start < kLTile) ? (L - l_start) : kLTile;
    size_t ptr_idx = batch_offset + static_cast<size_t>(l_start) * stride;
    for (int s = 0; s < s_end; ++s) {
        const float x_cur = __ldg(&x[ptr_idx]);
        float acc = fmaf(w_tap[0], x_cur, b_val);
        if (K > 1) {
            #pragma unroll
            for (int k = 1; k < K; ++k) {
                acc = fmaf(w_tap[k], hist[k - 1], acc);
            }
            #pragma unroll
            for (int h = K - 2; h > 0; --h) {
                hist[h] = hist[h - 1];
            }
            hist[0] = x_cur;
        }
        output[ptr_idx] = acc;
        ptr_idx += stride;
    }
}

// General fallback kernel supporting any K >= 1 (including K > 8 or K > L), any D >= 1, and any alignment.
template <int kThreads, int kLTile>
__global__ __launch_bounds__(kThreads) void causal_conv1d_general_k_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int L,
    int D,
    int K) {
    const int d = blockIdx.x * kThreads + threadIdx.x;
    if (d >= D) {
        return;
    }

    const int l_start = blockIdx.y * kLTile;
    const int b = blockIdx.z;
    const int s_end = (L - l_start < kLTile) ? (L - l_start) : kLTile;

    const float b_val = __ldg(&bias[d]);
    const float* __restrict__ w_row = weight + static_cast<size_t>(d) * static_cast<size_t>(K);
    const size_t batch_offset = static_cast<size_t>(b) * static_cast<size_t>(L) * static_cast<size_t>(D)
                                + static_cast<size_t>(d);
    const size_t stride = static_cast<size_t>(D);

    for (int s = 0; s < s_end; ++s) {
        const int l = l_start + s;
        const int k_max = (K - 1 < l) ? (K - 1) : l;
        float acc = b_val;
        for (int k = 0; k <= k_max; ++k) {
            const float x_val = __ldg(&x[batch_offset + static_cast<size_t>(l - k) * stride]);
            acc = fmaf(__ldg(&w_row[k]), x_val, acc);
        }
        output[batch_offset + static_cast<size_t>(l) * stride] = acc;
    }
}

inline bool is_aligned_16(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 15u) == 0u;
}

template <int K>
inline void launch_fixed_k(const float* x, const float* weight, const float* bias, float* output,
                           int B, int L, int D) {
    const bool use_vec4 = ((D & 3) == 0) &&
                          is_aligned_16(x) &&
                          is_aligned_16(bias) &&
                          is_aligned_16(output);
    if (use_vec4) {
        const int D4 = D >> 2;
        const dim3 grid(
            static_cast<unsigned>((D4 + CONV1D_THREADS - 1) / CONV1D_THREADS),
            static_cast<unsigned>((L + CONV1D_LTILE - 1) / CONV1D_LTILE),
            static_cast<unsigned>(B));
        causal_conv1d_vec4_fixed_k_kernel<K, CONV1D_THREADS, CONV1D_LTILE><<<grid, CONV1D_THREADS>>>(
            reinterpret_cast<const float4*>(x),
            weight,
            reinterpret_cast<const float4*>(bias),
            reinterpret_cast<float4*>(output),
            L,
            D4);
    } else {
        const dim3 grid(
            static_cast<unsigned>((D + CONV1D_THREADS - 1) / CONV1D_THREADS),
            static_cast<unsigned>((L + CONV1D_LTILE - 1) / CONV1D_LTILE),
            static_cast<unsigned>(B));
        causal_conv1d_scalar_fixed_k_kernel<K, CONV1D_THREADS, CONV1D_LTILE>
            <<<grid, CONV1D_THREADS>>>(x, weight, bias, output, L, D);
    }
}

}  // namespace

// x, weight, bias, output are device pointers
extern "C" void solve(const float* x, const float* weight, const float* bias, float* output, int B,
                      int L, int D, int K) {
    if (B <= 0 || L <= 0 || D <= 0 || K <= 0) {
        return;
    }

    switch (K) {
        case 1:
            launch_fixed_k<1>(x, weight, bias, output, B, L, D);
            return;
        case 2:
            launch_fixed_k<2>(x, weight, bias, output, B, L, D);
            return;
        case 3:
            launch_fixed_k<3>(x, weight, bias, output, B, L, D);
            return;
        case 4:
            launch_fixed_k<4>(x, weight, bias, output, B, L, D);
            return;
        case 5:
            launch_fixed_k<5>(x, weight, bias, output, B, L, D);
            return;
        case 6:
            launch_fixed_k<6>(x, weight, bias, output, B, L, D);
            return;
        case 7:
            launch_fixed_k<7>(x, weight, bias, output, B, L, D);
            return;
        case 8:
            launch_fixed_k<8>(x, weight, bias, output, B, L, D);
            return;
        default: {
            const dim3 grid(
                static_cast<unsigned>((D + CONV1D_THREADS - 1) / CONV1D_THREADS),
                static_cast<unsigned>((L + 16 - 1) / 16),
                static_cast<unsigned>(B));
            causal_conv1d_general_k_kernel<CONV1D_THREADS, 16>
                <<<grid, CONV1D_THREADS>>>(x, weight, bias, output, L, D, K);
            return;
        }
    }
}
