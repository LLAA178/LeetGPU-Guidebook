#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

namespace {

constexpr int kWarpSize = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarpSize;
constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(kFullMask, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_inclusive_scan(float val) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    #pragma unroll
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const float up = __shfl_up_sync(kFullMask, val, offset);
        if (lane >= offset) {
            val += up;
        }
    }
    return val;
}

__device__ __forceinline__ int warp_reduce_min(int val) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        const int other = __shfl_down_sync(kFullMask, val, offset);
        if (other < val) {
            val = other;
        }
    }
    return val;
}

// Baseline: parallelizes in an unfused style across all T positions (computing
// resampled tokens for every (b, i) plus the bonus token at T - 1), then selects
// the prefix up to first_reject — mirroring the JAX vectorized reference.
__global__ __launch_bounds__(kThreads) void spec_decode_baseline_kernel(
    const int* __restrict__ draft_tokens,
    const float* __restrict__ draft_probs,
    const float* __restrict__ target_probs,
    const float* __restrict__ uniform_samples,
    int* __restrict__ output_tokens,
    int T,
    int V) {
    __shared__ float s_warp_f[kWarps];
    __shared__ int s_warp_i[kWarps];
    __shared__ float s_total;
    __shared__ int s_resampled[16];
    __shared__ int s_bonus;

    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid >> 5;
    const int b = blockIdx.x;

    const float r_sample = uniform_samples[b * (T + 1) + T];

    // Evaluate resampled token for every position i = 0 .. T - 1 across full V.
    for (int i = 0; i < T; ++i) {
        const size_t base = (static_cast<size_t>(b) * static_cast<size_t>(T) +
                             static_cast<size_t>(i)) * static_cast<size_t>(V);
        const float* __restrict__ p_row = draft_probs + base;
        const float* __restrict__ q_row = target_probs + base;

        float lsum = 0.0f;
        for (int v = tid; v < V; v += kThreads) {
            lsum += fmaxf(q_row[v] - p_row[v], 0.0f);
        }
        lsum = warp_reduce_sum(lsum);
        if (lane == 0) s_warp_f[warp] = lsum;
        __syncthreads();
        if (warp == 0) {
            float bsum = (lane < kWarps) ? s_warp_f[lane] : 0.0f;
            bsum = warp_reduce_sum(bsum);
            if (lane == 0) s_total = bsum;
        }
        __syncthreads();

        const float total = s_total;
        const float inv_total = (total > 0.0f) ? (1.0f / total) : 0.0f;
        const float unif_p = 1.0f / static_cast<float>(V);

        float running = 0.0f;
        int hit = V - 1;
        bool found = false;
        for (int tile = 0; tile < V; tile += kThreads) {
            const int v = tile + tid;
            float prob = 0.0f;
            if (v < V) {
                prob = (total > 0.0f) ? (fmaxf(q_row[v] - p_row[v], 0.0f) * inv_total) : unif_p;
            }
            const float inc = warp_inclusive_scan(prob);
            if (lane == kWarpSize - 1) s_warp_f[warp] = inc;
            __syncthreads();
            if (warp == 0) {
                float w = (lane < kWarps) ? s_warp_f[lane] : 0.0f;
                w = warp_inclusive_scan(w);
                if (lane < kWarps) s_warp_f[lane] = w;
            }
            __syncthreads();

            const float cdf = running + ((warp > 0) ? s_warp_f[warp - 1] : 0.0f) + inc;
            const float tile_sum = s_warp_f[kWarps - 1];
            const bool is_last_tile = (tile + kThreads >= V);
            if (!found && (running + tile_sum >= r_sample || is_last_tile)) {
                int local_v = (v < V && cdf >= r_sample) ? v : (V - 1);
                local_v = warp_reduce_min(local_v);
                if (lane == 0) s_warp_i[warp] = local_v;
                __syncthreads();
                if (warp == 0) {
                    int mv = (lane < kWarps) ? s_warp_i[lane] : (V - 1);
                    mv = warp_reduce_min(mv);
                    if (lane == 0) hit = mv;
                }
                found = true;
            }
            __syncthreads();
            running += tile_sum;
        }
        if (tid == 0) {
            s_resampled[i] = hit;
        }
        __syncthreads();
    }

    // Bonus token from target_probs[b, T - 1].
    {
        const size_t base = (static_cast<size_t>(b) * static_cast<size_t>(T) +
                             static_cast<size_t>(T - 1)) * static_cast<size_t>(V);
        const float* __restrict__ q_last = target_probs + base;
        float running = 0.0f;
        int hit = V - 1;
        bool found = false;
        for (int tile = 0; tile < V; tile += kThreads) {
            const int v = tile + tid;
            const float prob = (v < V) ? q_last[v] : 0.0f;
            const float inc = warp_inclusive_scan(prob);
            if (lane == kWarpSize - 1) s_warp_f[warp] = inc;
            __syncthreads();
            if (warp == 0) {
                float w = (lane < kWarps) ? s_warp_f[lane] : 0.0f;
                w = warp_inclusive_scan(w);
                if (lane < kWarps) s_warp_f[lane] = w;
            }
            __syncthreads();

            const float cdf = running + ((warp > 0) ? s_warp_f[warp - 1] : 0.0f) + inc;
            const float tile_sum = s_warp_f[kWarps - 1];
            const bool is_last_tile = (tile + kThreads >= V);
            if (!found && (running + tile_sum >= r_sample || is_last_tile)) {
                int local_v = (v < V && cdf >= r_sample) ? v : (V - 1);
                local_v = warp_reduce_min(local_v);
                if (lane == 0) s_warp_i[warp] = local_v;
                __syncthreads();
                if (warp == 0) {
                    int mv = (lane < kWarps) ? s_warp_i[lane] : (V - 1);
                    mv = warp_reduce_min(mv);
                    if (lane == 0) hit = mv;
                }
                found = true;
            }
            __syncthreads();
            running += tile_sum;
        }
        if (tid == 0) {
            s_bonus = hit;
        }
        __syncthreads();
    }

    // Thread 0 resolves acceptance prefix and writes output_tokens[b, 0..T].
    if (tid == 0) {
        int* __restrict__ out_row = output_tokens + b * (T + 1);
        for (int k = 0; k <= T; ++k) {
            out_row[k] = 0;
        }
        bool rejected = false;
        for (int i = 0; i < T; ++i) {
            const int tok = draft_tokens[b * T + i];
            const size_t base = (static_cast<size_t>(b) * static_cast<size_t>(T) +
                                 static_cast<size_t>(i)) * static_cast<size_t>(V);
            const float p = draft_probs[base + static_cast<size_t>(tok)];
            const float q = target_probs[base + static_cast<size_t>(tok)];
            const float alpha = fminf(1.0f, q / p);
            if (uniform_samples[b * (T + 1) + i] < alpha) {
                out_row[i] = tok;
            } else {
                out_row[i] = s_resampled[i];
                rejected = true;
                break;
            }
        }
        if (!rejected) {
            out_row[T] = s_bonus;
        }
    }
}

}  // namespace

extern "C" void solve(const int* draft_tokens, const float* draft_probs, const float* target_probs,
                      const float* uniform_samples, int* output_tokens, int B, int T, int V) {
    if (B <= 0 || T <= 0 || V <= 0) {
        return;
    }
    spec_decode_baseline_kernel<<<B, kThreads>>>(
        draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens, T, V);
}
