#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarp;
constexpr int kVecsPerThread = 4;
constexpr int kTile = kThreads * kVecsPerThread * 4;  // 4,096 floats per tile
constexpr int kMaxStaticTiles = (100000000 + kTile - 1) / kTile + 64;
constexpr unsigned kFull = 0xffffffffu;

constexpr unsigned kStatusAggregate = 1u;
constexpr unsigned kStatusPrefix = 2u;

struct __align__(8) TaggedState {
    unsigned tag;
    float value;
};

__device__ unsigned long long g_dispatch = (1ull << 32);
__device__ TaggedState g_tagged_states[kMaxStaticTiles];

__device__ __forceinline__ TaggedState load_state(const TaggedState* ptr) {
    unsigned long long raw;
#if __CUDA_ARCH__ >= 700
    asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(raw) : "l"(ptr) : "memory");
#else
    raw = *reinterpret_cast<const volatile unsigned long long*>(ptr);
#endif
    TaggedState s;
    s.tag = static_cast<unsigned>(raw & 0xffffffffull);
    s.value = __uint_as_float(static_cast<unsigned>(raw >> 32));
    return s;
}

__device__ __forceinline__ void store_state(TaggedState* ptr, unsigned tag, float val) {
    const unsigned long long raw =
        static_cast<unsigned long long>(tag) |
        (static_cast<unsigned long long>(__float_as_uint(val)) << 32);
#if __CUDA_ARCH__ >= 700
    asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" :: "l"(ptr), "l"(raw) : "memory");
#else
    *reinterpret_cast<volatile unsigned long long*>(ptr) = raw;
#endif
}

__device__ __forceinline__ float warp_inclusive_scan(float val) {
    const int lane = threadIdx.x & (kWarp - 1);
    #pragma unroll
    for (int step = 1; step < kWarp; step <<= 1) {
        const float prev = __shfl_up_sync(kFull, val, step);
        if (lane >= step) val += prev;
    }
    return val;
}

template <bool kAligned>
__global__ __launch_bounds__(kThreads) void prefix_sum_lookback_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n4,
    int n,
    int num_tiles) {
    __shared__ int s_tile;
    __shared__ unsigned s_epoch;
    __shared__ float s_warp_totals[kWarps];
    __shared__ float s_exclusive_tile_prefix;

    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x / kWarp;

    if (num_tiles == 1) {
        if (threadIdx.x == 0) {
            s_tile = 0;
            s_epoch = 0u;
        }
    } else if (threadIdx.x == 0) {
        const unsigned long long old = atomicAdd(&g_dispatch, 1ull);
        const int tile = static_cast<int>(old & 0xffffffffull);
        const unsigned epoch = static_cast<unsigned>(old >> 32);
        s_tile = tile;
        s_epoch = epoch;
        if (tile == num_tiles - 1) {
            atomicExch(&g_dispatch, static_cast<unsigned long long>(epoch + 1u) << 32);
        }
    }
    __syncthreads();

    const int tile = s_tile;
    const int tile_vec_base = tile * (kThreads * kVecsPerThread);
    const int warp_vec_base = tile_vec_base + warp * (kWarp * kVecsPerThread);

    const float4* __restrict__ input4 = reinterpret_cast<const float4*>(input);
    float4* __restrict__ output4 = reinterpret_cast<float4*>(output);

    float4 vals[kVecsPerThread];
    float warp_running = 0.0f;

    #pragma unroll
    for (int r = 0; r < kVecsPerThread; ++r) {
        const int vec_idx = warp_vec_base + r * kWarp + lane;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (kAligned) {
            if (vec_idx < n4) {
                v = __ldg(&input4[vec_idx]);
            } else {
                const int base = vec_idx << 2;
                if (base < n) v.x = __ldg(&input[base]);
                if (base + 1 < n) v.y = __ldg(&input[base + 1]);
                if (base + 2 < n) v.z = __ldg(&input[base + 2]);
            }
        } else {
            const int base = vec_idx << 2;
            if (base < n) v.x = __ldg(&input[base]);
            if (base + 1 < n) v.y = __ldg(&input[base + 1]);
            if (base + 2 < n) v.z = __ldg(&input[base + 2]);
            if (base + 3 < n) v.w = __ldg(&input[base + 3]);
        }

        v.y += v.x;
        v.z += v.y;
        v.w += v.z;

        const float t_sum = v.w;
        const float w_inc = warp_inclusive_scan(t_sum);
        const float t_exc = warp_running + (w_inc - t_sum);
        v.x += t_exc;
        v.y += t_exc;
        v.z += t_exc;
        v.w += t_exc;
        vals[r] = v;
        warp_running += __shfl_sync(kFull, w_inc, kWarp - 1);
    }

    if (lane == 0) s_warp_totals[warp] = warp_running;
    __syncthreads();

    if (warp == 0) {
        const float wval = lane < kWarps ? s_warp_totals[lane] : 0.0f;
        const float wend = warp_inclusive_scan(wval);
        if (lane < kWarps) s_warp_totals[lane] = wend - wval;
        const float tile_agg = __shfl_sync(kFull, wend, kWarps - 1);

        if (num_tiles == 1) {
            if (lane == 0) s_exclusive_tile_prefix = 0.0f;
        } else {
            const unsigned epoch_agg = (s_epoch << 2) | kStatusAggregate;
            const unsigned epoch_pfx = (s_epoch << 2) | kStatusPrefix;

            if (tile == 0) {
                if (lane == 0) {
                    store_state(&g_tagged_states[0], epoch_pfx, tile_agg);
                    s_exclusive_tile_prefix = 0.0f;
                }
            } else {
                if (lane == 0) {
                    store_state(&g_tagged_states[tile], epoch_agg, tile_agg);
                }
                float excl = 0.0f;
                for (int prior = tile - 1; prior >= 0; prior -= kWarp) {
                    const int idx = prior - lane;
                    TaggedState st;
                    unsigned pfx_mask;
                    unsigned needed;
                    do {
                        if (idx >= 0) {
                            st = load_state(&g_tagged_states[idx]);
                        } else {
                            st.tag = epoch_pfx;
                            st.value = 0.0f;
                        }
                        const bool is_pfx = (st.tag == epoch_pfx);
                        const bool is_ready = is_pfx || (st.tag == epoch_agg);
                        pfx_mask = __ballot_sync(kFull, is_pfx);
                        const int stop = __ffs(pfx_mask);
                        needed = (stop > 0 && stop < 32) ? ((1u << stop) - 1u) : kFull;
                        const unsigned not_ready = __ballot_sync(kFull, !is_ready);
                        if ((not_ready & needed) == 0u) break;
                    } while (true);

                    if (needed & (1u << lane)) {
                        excl += st.value;
                    }
                    if (pfx_mask != 0u) break;
                }
                #pragma unroll
                for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
                    excl += __shfl_down_sync(kFull, excl, offset);
                }
                if (lane == 0) {
                    store_state(&g_tagged_states[tile], epoch_pfx, excl + tile_agg);
                    s_exclusive_tile_prefix = excl;
                }
            }
        }
    }
    __syncthreads();

    const float base_offset = s_exclusive_tile_prefix + s_warp_totals[warp];
    #pragma unroll
    for (int r = 0; r < kVecsPerThread; ++r) {
        const int vec_idx = warp_vec_base + r * kWarp + lane;
        if (kAligned) {
            if (vec_idx < n4) {
                output4[vec_idx] = make_float4(
                    vals[r].x + base_offset,
                    vals[r].y + base_offset,
                    vals[r].z + base_offset,
                    vals[r].w + base_offset);
            } else {
                const int base = vec_idx << 2;
                if (base < n) output[base] = vals[r].x + base_offset;
                if (base + 1 < n) output[base + 1] = vals[r].y + base_offset;
                if (base + 2 < n) output[base + 2] = vals[r].z + base_offset;
            }
        } else {
            const int base = vec_idx << 2;
            if (base < n) output[base] = vals[r].x + base_offset;
            if (base + 1 < n) output[base + 1] = vals[r].y + base_offset;
            if (base + 2 < n) output[base + 2] = vals[r].z + base_offset;
            if (base + 3 < n) output[base + 3] = vals[r].w + base_offset;
        }
    }
}

}  // namespace

extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;

    const int n4 = N >> 2;
    const int num_tiles = (N + kTile - 1) / kTile;
    const bool aligned = ((reinterpret_cast<uintptr_t>(input) |
                           reinterpret_cast<uintptr_t>(output)) & 15u) == 0u;
    if (aligned) {
        prefix_sum_lookback_kernel<true><<<num_tiles, kThreads>>>(
            input, output, n4, N, num_tiles);
    } else {
        prefix_sum_lookback_kernel<false><<<num_tiles, kThreads>>>(
            input, output, n4, N, num_tiles);
    }
}
