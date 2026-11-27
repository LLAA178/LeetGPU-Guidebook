#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarp;
constexpr int kSmallBins = 256;
constexpr int kMaxBins = 1024;
constexpr int kMaxBlocks = 160;  // 40 SMs * 4 blocks/SM (100% warp occupancy on T4 SM 7.5)
constexpr unsigned kFullMask = 0xffffffffu;

template <bool kSingleBlock>
__global__ __launch_bounds__(kThreads, 4) void histogram_small_vec4_kernel(
    const int* __restrict__ input,
    int* __restrict__ histogram,
    int n,
    int num_bins) {
    __shared__ int s_hist[kWarps][kSmallBins + 1];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const unsigned u_bins = static_cast<unsigned>(num_bins);

    #pragma unroll
    for (int b = lane; b < kSmallBins; b += kWarp) {
        s_hist[warp][b] = 0;
    }
    __syncthreads();

    const int n4 = n >> 2;
    const int n4_warp_full = n4 & ~31;
    const int4* __restrict__ input4 = reinterpret_cast<const int4*>(input);
    const int global_tid = blockIdx.x * kThreads + tid;
    const int stride = gridDim.x * kThreads;
    const int stride4 = stride << 2;

    int idx = global_tid;
    for (; idx + 3 * stride < n4_warp_full; idx += stride4) {
        const int4 v0 = __ldg(&input4[idx]);
        const int4 v1 = __ldg(&input4[idx + stride]);
        const int4 v2 = __ldg(&input4[idx + 2 * stride]);
        const int4 v3 = __ldg(&input4[idx + 3 * stride]);

        #define ACCUM_VEC4(vec)                                                        \
            do {                                                                       \
                const int same4 =                                                      \
                    ((vec).x == (vec).y) & ((vec).z == (vec).w) & ((vec).x == (vec).z); \
                const int ref = __shfl_sync(kFullMask, (vec).x, 0);                    \
                if (__all_sync(kFullMask, same4 & ((vec).x == ref))) {                 \
                    if (lane == 0 && static_cast<unsigned>(ref) < u_bins) {            \
                        s_hist[warp][ref] += 128;                                      \
                    }                                                                  \
                } else {                                                               \
                    if (static_cast<unsigned>((vec).x) < u_bins)                       \
                        atomicAdd(&s_hist[warp][(vec).x], 1);                          \
                    if (static_cast<unsigned>((vec).y) < u_bins)                       \
                        atomicAdd(&s_hist[warp][(vec).y], 1);                          \
                    if (static_cast<unsigned>((vec).z) < u_bins)                       \
                        atomicAdd(&s_hist[warp][(vec).z], 1);                          \
                    if (static_cast<unsigned>((vec).w) < u_bins)                       \
                        atomicAdd(&s_hist[warp][(vec).w], 1);                          \
                }                                                                      \
            } while (0)

        ACCUM_VEC4(v0);
        ACCUM_VEC4(v1);
        ACCUM_VEC4(v2);
        ACCUM_VEC4(v3);

        #undef ACCUM_VEC4
    }

    for (; idx < n4_warp_full; idx += stride) {
        const int4 v = __ldg(&input4[idx]);
        const int same4 = (v.x == v.y) & (v.z == v.w) & (v.x == v.z);
        const int ref = __shfl_sync(kFullMask, v.x, 0);
        if (__all_sync(kFullMask, same4 & (v.x == ref))) {
            if (lane == 0 && static_cast<unsigned>(ref) < u_bins) {
                s_hist[warp][ref] += 128;
            }
        } else {
            if (static_cast<unsigned>(v.x) < u_bins) atomicAdd(&s_hist[warp][v.x], 1);
            if (static_cast<unsigned>(v.y) < u_bins) atomicAdd(&s_hist[warp][v.y], 1);
            if (static_cast<unsigned>(v.z) < u_bins) atomicAdd(&s_hist[warp][v.z], 1);
            if (static_cast<unsigned>(v.w) < u_bins) atomicAdd(&s_hist[warp][v.w], 1);
        }
    }

    for (; idx < n4; idx += stride) {
        const int4 v = __ldg(&input4[idx]);
        if (v.x == v.y && v.z == v.w && v.x == v.z) {
            if (static_cast<unsigned>(v.x) < u_bins) {
                atomicAdd(&s_hist[warp][v.x], 4);
            }
        } else {
            if (static_cast<unsigned>(v.x) < u_bins) atomicAdd(&s_hist[warp][v.x], 1);
            if (static_cast<unsigned>(v.y) < u_bins) atomicAdd(&s_hist[warp][v.y], 1);
            if (static_cast<unsigned>(v.z) < u_bins) atomicAdd(&s_hist[warp][v.z], 1);
            if (static_cast<unsigned>(v.w) < u_bins) atomicAdd(&s_hist[warp][v.w], 1);
        }
    }

    if (blockIdx.x == 0) {
        const int tail_start = n4 << 2;
        if (tid < n - tail_start) {
            const int v = __ldg(&input[tail_start + tid]);
            if (static_cast<unsigned>(v) < u_bins) {
                atomicAdd(&s_hist[warp][v], 1);
            }
        }
    }
    __syncthreads();

    if (tid < num_bins) {
        int sum = s_hist[0][tid] + s_hist[1][tid] + s_hist[2][tid] + s_hist[3][tid] +
                  s_hist[4][tid] + s_hist[5][tid] + s_hist[6][tid] + s_hist[7][tid];
        if (kSingleBlock) {
            histogram[tid] = sum;
        } else if (sum != 0) {
            atomicAdd(&histogram[tid], sum);
        }
    }
}

template <bool kAligned, bool kSingleBlock>
__global__ __launch_bounds__(kThreads, 3) void histogram_large_bins_kernel(
    const int* __restrict__ input,
    int* __restrict__ histogram,
    int n,
    int num_bins) {
    constexpr int kCopies = 4;
    __shared__ int s_hist[kCopies][kMaxBins + 1];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int copy = (tid >> 5) & (kCopies - 1);
    const unsigned u_bins = static_cast<unsigned>(num_bins);

    for (int b = tid; b < num_bins; b += kThreads) {
        #pragma unroll
        for (int c = 0; c < kCopies; ++c) {
            s_hist[c][b] = 0;
        }
    }
    __syncthreads();

    const int global_tid = blockIdx.x * kThreads + tid;
    const int stride = gridDim.x * kThreads;

    if (kAligned) {
        const int n4 = n >> 2;
        const int n4_warp_full = n4 & ~31;
        const int4* __restrict__ input4 = reinterpret_cast<const int4*>(input);
        const int stride4 = stride << 2;

        int idx = global_tid;
        for (; idx + 3 * stride < n4_warp_full; idx += stride4) {
            const int4 v0 = __ldg(&input4[idx]);
            const int4 v1 = __ldg(&input4[idx + stride]);
            const int4 v2 = __ldg(&input4[idx + 2 * stride]);
            const int4 v3 = __ldg(&input4[idx + 3 * stride]);

            #define ACCUM_LARGE_VEC4(vec)                                                  \
                do {                                                                       \
                    const int same4 =                                                      \
                        ((vec).x == (vec).y) & ((vec).z == (vec).w) & ((vec).x == (vec).z); \
                    const int ref = __shfl_sync(kFullMask, (vec).x, 0);                    \
                    if (__all_sync(kFullMask, same4 & ((vec).x == ref))) {                 \
                        if (lane == 0 && static_cast<unsigned>(ref) < u_bins) {            \
                            atomicAdd(&s_hist[copy][ref], 128);                            \
                        }                                                                  \
                    } else {                                                               \
                        if (static_cast<unsigned>((vec).x) < u_bins)                       \
                            atomicAdd(&s_hist[copy][(vec).x], 1);                          \
                        if (static_cast<unsigned>((vec).y) < u_bins)                       \
                            atomicAdd(&s_hist[copy][(vec).y], 1);                          \
                        if (static_cast<unsigned>((vec).z) < u_bins)                       \
                            atomicAdd(&s_hist[copy][(vec).z], 1);                          \
                        if (static_cast<unsigned>((vec).w) < u_bins)                       \
                            atomicAdd(&s_hist[copy][(vec).w], 1);                          \
                    }                                                                      \
                } while (0)

            ACCUM_LARGE_VEC4(v0);
            ACCUM_LARGE_VEC4(v1);
            ACCUM_LARGE_VEC4(v2);
            ACCUM_LARGE_VEC4(v3);

            #undef ACCUM_LARGE_VEC4
        }

        for (; idx < n4_warp_full; idx += stride) {
            const int4 v = __ldg(&input4[idx]);
            const int same4 = (v.x == v.y) & (v.z == v.w) & (v.x == v.z);
            const int ref = __shfl_sync(kFullMask, v.x, 0);
            if (__all_sync(kFullMask, same4 & (v.x == ref))) {
                if (lane == 0 && static_cast<unsigned>(ref) < u_bins) {
                    atomicAdd(&s_hist[copy][ref], 128);
                }
            } else {
                if (static_cast<unsigned>(v.x) < u_bins) atomicAdd(&s_hist[copy][v.x], 1);
                if (static_cast<unsigned>(v.y) < u_bins) atomicAdd(&s_hist[copy][v.y], 1);
                if (static_cast<unsigned>(v.z) < u_bins) atomicAdd(&s_hist[copy][v.z], 1);
                if (static_cast<unsigned>(v.w) < u_bins) atomicAdd(&s_hist[copy][v.w], 1);
            }
        }

        for (; idx < n4; idx += stride) {
            const int4 v = __ldg(&input4[idx]);
            if (v.x == v.y && v.z == v.w && v.x == v.z) {
                if (static_cast<unsigned>(v.x) < u_bins) {
                    atomicAdd(&s_hist[copy][v.x], 4);
                }
            } else {
                if (static_cast<unsigned>(v.x) < u_bins) atomicAdd(&s_hist[copy][v.x], 1);
                if (static_cast<unsigned>(v.y) < u_bins) atomicAdd(&s_hist[copy][v.y], 1);
                if (static_cast<unsigned>(v.z) < u_bins) atomicAdd(&s_hist[copy][v.z], 1);
                if (static_cast<unsigned>(v.w) < u_bins) atomicAdd(&s_hist[copy][v.w], 1);
            }
        }

        if (blockIdx.x == 0) {
            const int tail_start = n4 << 2;
            if (tid < n - tail_start) {
                const int v = __ldg(&input[tail_start + tid]);
                if (static_cast<unsigned>(v) < u_bins) {
                    atomicAdd(&s_hist[copy][v], 1);
                }
            }
        }
    } else {
        const int n_warp_full = n & ~31;
        int i = global_tid;
        for (; i < n_warp_full; i += stride) {
            const int v = __ldg(&input[i]);
            const int ref = __shfl_sync(kFullMask, v, 0);
            if (__all_sync(kFullMask, v == ref)) {
                if (lane == 0 && static_cast<unsigned>(ref) < u_bins) {
                    atomicAdd(&s_hist[copy][ref], 32);
                }
            } else if (static_cast<unsigned>(v) < u_bins) {
                atomicAdd(&s_hist[copy][v], 1);
            }
        }
        if (i < n) {
            const int v = __ldg(&input[i]);
            if (static_cast<unsigned>(v) < u_bins) {
                atomicAdd(&s_hist[copy][v], 1);
            }
        }
    }
    __syncthreads();

    for (int b = tid; b < num_bins; b += kThreads) {
        const int sum = s_hist[0][b] + s_hist[1][b] + s_hist[2][b] + s_hist[3][b];
        if (kSingleBlock) {
            histogram[b] = sum;
        } else if (sum != 0) {
            atomicAdd(&histogram[b], sum);
        }
    }
}

template <bool kSingleBlock>
__global__ __launch_bounds__(kThreads, 4) void histogram_small_scalar_kernel(
    const int* __restrict__ input,
    int* __restrict__ histogram,
    int n,
    int num_bins) {
    __shared__ int s_hist[kWarps][kSmallBins + 1];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const unsigned u_bins = static_cast<unsigned>(num_bins);

    #pragma unroll
    for (int b = lane; b < kSmallBins; b += kWarp) {
        s_hist[warp][b] = 0;
    }
    __syncthreads();

    const int global_tid = blockIdx.x * kThreads + tid;
    const int stride = gridDim.x * kThreads;

    const int n_warp_full = n & ~31;
    int i = global_tid;
    for (; i < n_warp_full; i += stride) {
        const int v = __ldg(&input[i]);
        const int ref = __shfl_sync(kFullMask, v, 0);
        if (__all_sync(kFullMask, v == ref)) {
            if (lane == 0 && static_cast<unsigned>(ref) < u_bins) {
                s_hist[warp][ref] += 32;
            }
        } else if (static_cast<unsigned>(v) < u_bins) {
            atomicAdd(&s_hist[warp][v], 1);
        }
    }
    if (i < n) {
        const int v = __ldg(&input[i]);
        if (static_cast<unsigned>(v) < u_bins) {
            atomicAdd(&s_hist[warp][v], 1);
        }
    }
    __syncthreads();

    if (tid < num_bins) {
        const int sum = s_hist[0][tid] + s_hist[1][tid] + s_hist[2][tid] + s_hist[3][tid] +
                        s_hist[4][tid] + s_hist[5][tid] + s_hist[6][tid] + s_hist[7][tid];
        if (kSingleBlock) {
            histogram[tid] = sum;
        } else if (sum != 0) {
            atomicAdd(&histogram[tid], sum);
        }
    }
}

}  // namespace

extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    if (num_bins <= 0) {
        return;
    }
    if (N <= 0) {
        cudaMemsetAsync(histogram, 0, static_cast<size_t>(num_bins) * sizeof(int), 0);
        return;
    }

    const bool aligned = (reinterpret_cast<uintptr_t>(input) & 15u) == 0u;
    if (N <= 2048) {
        if (num_bins <= kSmallBins) {
            if (aligned) {
                histogram_small_vec4_kernel<true><<<1, kThreads>>>(input, histogram, N, num_bins);
            } else {
                histogram_small_scalar_kernel<true><<<1, kThreads>>>(input, histogram, N, num_bins);
            }
        } else {
            if (aligned) {
                histogram_large_bins_kernel<true, true><<<1, kThreads>>>(input, histogram, N, num_bins);
            } else {
                histogram_large_bins_kernel<false, true><<<1, kThreads>>>(input, histogram, N, num_bins);
            }
        }
        return;
    }

    cudaMemsetAsync(histogram, 0, static_cast<size_t>(num_bins) * sizeof(int), 0);

    if (num_bins <= kSmallBins) {
        if (aligned) {
            const int n4 = N >> 2;
            int blocks = (n4 + kThreads - 1) / kThreads;
            if (blocks < 1) blocks = 1;
            if (blocks > kMaxBlocks) blocks = kMaxBlocks;
            histogram_small_vec4_kernel<false><<<blocks, kThreads>>>(input, histogram, N, num_bins);
        } else {
            int blocks = (N + kThreads - 1) / kThreads;
            if (blocks > kMaxBlocks) blocks = kMaxBlocks;
            histogram_small_scalar_kernel<false><<<blocks, kThreads>>>(input, histogram, N, num_bins);
        }
    } else {
        if (aligned) {
            const int n4 = N >> 2;
            int blocks = (n4 + kThreads - 1) / kThreads;
            if (blocks < 1) blocks = 1;
            if (blocks > 120) blocks = 120;  // 40 SMs * 3 blocks/SM (16.4 KB smem/block)
            histogram_large_bins_kernel<true, false><<<blocks, kThreads>>>(input, histogram, N, num_bins);
        } else {
            int blocks = (N + kThreads - 1) / kThreads;
            if (blocks > 120) blocks = 120;
            histogram_large_bins_kernel<false, false><<<blocks, kThreads>>>(input, histogram, N, num_bins);
        }
    }
}
