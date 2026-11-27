#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarpSize;
constexpr int kMaxBlocks = 256;
constexpr int kMaxBssPow2 = 1048576;  // next power of 2 up to 1M vocab
constexpr unsigned kFullMask = 0xffffffffu;

__device__ uint64_t g_base_keys[kMaxBssPow2];
__device__ float g_base_probs[kMaxBssPow2];
__device__ float g_base_cumsum[kMaxBssPow2];
__device__ float g_base_tile_sums[kMaxBssPow2 / 1024];
__device__ float g_base_block_m[kMaxBlocks];
__device__ float g_base_block_d[kMaxBlocks];
__device__ float g_base_max_logit;
__device__ float g_base_inv_sum_exp;
__device__ int g_base_cutoff_idx;
__device__ float g_base_inv_nuc_sum;

__device__ __forceinline__ uint32_t pcg_hash_u32(uint32_t x) {
    const uint32_t state = x * 747796405u + 2891336453u;
    const uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ __forceinline__ uint32_t float_to_ordered_u32(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? (~u) : (u ^ 0x80000000u);
}

__device__ __forceinline__ float ordered_u32_to_float(uint32_t o) {
    const uint32_t u = (o & 0x80000000u) ? (o ^ 0x80000000u) : (~o);
    return __uint_as_float(u);
}

__device__ __forceinline__ uint64_t pack_desc_prob_asc_idx(float prob, int idx) {
    return (static_cast<uint64_t>(float_to_ordered_u32(prob)) << 32) |
           static_cast<uint64_t>(~static_cast<uint32_t>(idx));
}

__device__ __forceinline__ void unpack_key(uint64_t key, float* prob, int* idx) {
    *prob = ordered_u32_to_float(static_cast<uint32_t>(key >> 32));
    *idx = static_cast<int>(~static_cast<uint32_t>(key & 0xffffffffull));
}

__device__ __forceinline__ void online_merge(float m_b, float d_b, float* m_a, float* d_a) {
    if (d_b == 0.0f) return;
    if (*d_a == 0.0f) {
        *m_a = m_b;
        *d_a = d_b;
        return;
    }
    if (m_b > *m_a) {
        *d_a = (*d_a) * expf(*m_a - m_b) + d_b;
        *m_a = m_b;
    } else {
        *d_a += d_b * expf(m_b - *m_a);
    }
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        v += __shfl_down_sync(kFullMask, v, off);
    }
    return v;
}

__device__ __forceinline__ float warp_inclusive_scan(float v) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    #pragma unroll
    for (int off = 1; off < kWarpSize; off <<= 1) {
        const float up = __shfl_up_sync(kFullMask, v, off);
        if (lane >= off) v += up;
    }
    return v;
}

__global__ __launch_bounds__(kThreads) void base_softmax_stats_kernel(
    const float* __restrict__ logits,
    int vocab_size) {
    __shared__ float s_m[kWarps];
    __shared__ float s_d[kWarps];

    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid >> 5;

    float m = -INFINITY;
    float d = 0.0f;
    for (int i = blockIdx.x * kThreads + tid; i < vocab_size; i += gridDim.x * kThreads) {
        const float x = logits[i];
        online_merge(x, 1.0f, &m, &d);
    }

    #pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        const float om = __shfl_down_sync(kFullMask, m, off);
        const float od = __shfl_down_sync(kFullMask, d, off);
        online_merge(om, od, &m, &d);
    }
    if (lane == 0) {
        s_m[warp] = m;
        s_d[warp] = d;
    }
    __syncthreads();

    if (warp == 0) {
        float bm = (lane < kWarps) ? s_m[lane] : -INFINITY;
        float bd = (lane < kWarps) ? s_d[lane] : 0.0f;
        #pragma unroll
        for (int off = kWarpSize / 2; off > 0; off >>= 1) {
            const float om = __shfl_down_sync(kFullMask, bm, off);
            const float od = __shfl_down_sync(kFullMask, bd, off);
            online_merge(om, od, &bm, &bd);
        }
        if (lane == 0) {
            g_base_block_m[blockIdx.x] = bm;
            g_base_block_d[blockIdx.x] = bd;
        }
    }
}

__global__ __launch_bounds__(kThreads) void base_softmax_finalize_kernel(int num_blocks) {
    const int tid = threadIdx.x;
    float m = (tid < num_blocks) ? g_base_block_m[tid] : -INFINITY;
    float d = (tid < num_blocks) ? g_base_block_d[tid] : 0.0f;

    __shared__ float s_m[kWarps];
    __shared__ float s_d[kWarps];
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid >> 5;

    #pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        const float om = __shfl_down_sync(kFullMask, m, off);
        const float od = __shfl_down_sync(kFullMask, d, off);
        online_merge(om, od, &m, &d);
    }
    if (lane == 0) {
        s_m[warp] = m;
        s_d[warp] = d;
    }
    __syncthreads();

    if (warp == 0) {
        float bm = (lane < kWarps) ? s_m[lane] : -INFINITY;
        float bd = (lane < kWarps) ? s_d[lane] : 0.0f;
        #pragma unroll
        for (int off = kWarpSize / 2; off > 0; off >>= 1) {
            const float om = __shfl_down_sync(kFullMask, bm, off);
            const float od = __shfl_down_sync(kFullMask, bd, off);
            online_merge(om, od, &bm, &bd);
        }
        if (lane == 0) {
            g_base_max_logit = bm;
            g_base_inv_sum_exp = 1.0f / bd;
        }
    }
}

__global__ __launch_bounds__(kThreads) void base_init_sort_keys_kernel(
    const float* __restrict__ logits,
    int vocab_size,
    int padded_n) {
    const float max_l = g_base_max_logit;
    const float inv_z = g_base_inv_sum_exp;
    for (int i = blockIdx.x * kThreads + threadIdx.x; i < padded_n; i += gridDim.x * kThreads) {
        if (i < vocab_size) {
            const float prob = expf(logits[i] - max_l) * inv_z;
            g_base_keys[i] = pack_desc_prob_asc_idx(prob, i);
        } else {
            g_base_keys[i] = 0ull;  // minimal key, sinks to the tail of descending sort
        }
    }
}

__global__ __launch_bounds__(kThreads) void base_bitonic_step_kernel(int j, int k, int padded_n) {
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= padded_n) return;
    const int ixj = i ^ j;
    if (ixj > i) {
        const uint64_t ki = g_base_keys[i];
        const uint64_t kj = g_base_keys[ixj];
        const bool descending = ((i & k) == 0);
        if ((descending && ki < kj) || (!descending && ki > kj)) {
            g_base_keys[i] = kj;
            g_base_keys[ixj] = ki;
        }
    }
}

__global__ __launch_bounds__(kThreads) void base_unpack_and_tile_scan_kernel(int vocab_size) {
    __shared__ float s_warp[kWarps];
    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid >> 5;
    const int base = blockIdx.x * 1024 + (tid << 2);

    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
    int dummy_idx = 0;
    if (base < vocab_size) unpack_key(g_base_keys[base], &p0, &dummy_idx);
    if (base + 1 < vocab_size) unpack_key(g_base_keys[base + 1], &p1, &dummy_idx);
    if (base + 2 < vocab_size) unpack_key(g_base_keys[base + 2], &p2, &dummy_idx);
    if (base + 3 < vocab_size) unpack_key(g_base_keys[base + 3], &p3, &dummy_idx);

    const float tsum = (p0 + p1) + (p2 + p3);
    const float inc = warp_inclusive_scan(tsum);
    if (lane == kWarpSize - 1) s_warp[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        float w = (lane < kWarps) ? s_warp[lane] : 0.0f;
        w = warp_inclusive_scan(w);
        if (lane < kWarps) s_warp[lane] = w;
    }
    __syncthreads();

    const float excl = ((warp > 0) ? s_warp[warp - 1] : 0.0f) + inc - tsum;
    if (base < vocab_size) { g_base_probs[base] = p0; g_base_cumsum[base] = excl + p0; }
    if (base + 1 < vocab_size) { g_base_probs[base + 1] = p1; g_base_cumsum[base + 1] = excl + p0 + p1; }
    if (base + 2 < vocab_size) { g_base_probs[base + 2] = p2; g_base_cumsum[base + 2] = excl + p0 + p1 + p2; }
    if (base + 3 < vocab_size) { g_base_probs[base + 3] = p3; g_base_cumsum[base + 3] = excl + tsum; }

    if (tid == 0) {
        g_base_tile_sums[blockIdx.x] = s_warp[kWarps - 1];
    }
}

__global__ __launch_bounds__(kThreads) void base_find_cutoff_and_renorm_kernel(
    const float* __restrict__ p,
    int vocab_size,
    int num_tiles) {
    __shared__ float s_tile_excl[1024];
    __shared__ float s_warp[kWarps];
    __shared__ int s_min_idx[kWarps];

    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid >> 5;

    // Scan tile sums into shared memory.
    float running = 0.0f;
    for (int base = 0; base < num_tiles; base += kThreads) {
        const int t = base + tid;
        const float v = (t < num_tiles) ? g_base_tile_sums[t] : 0.0f;
        const float inc = warp_inclusive_scan(v);
        if (lane == kWarpSize - 1) s_warp[warp] = inc;
        __syncthreads();
        if (warp == 0) {
            float w = (lane < kWarps) ? s_warp[lane] : 0.0f;
            w = warp_inclusive_scan(w);
            if (lane < kWarps) s_warp[lane] = w;
        }
        __syncthreads();
        if (t < num_tiles) {
            s_tile_excl[t] = running + ((warp > 0) ? s_warp[warp - 1] : 0.0f) + inc - v;
        }
        const float chunk_total = s_warp[kWarps - 1];
        __syncthreads();
        running += chunk_total;
    }

    const float p_val = p[0];
    int first_hit = vocab_size;
    for (int i = tid; i < vocab_size; i += kThreads) {
        const float csum = g_base_cumsum[i] + s_tile_excl[i >> 10];
        if (csum >= p_val && i < first_hit) {
            first_hit = i;
        }
    }

    #pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        const int o = __shfl_down_sync(kFullMask, first_hit, off);
        if (o < first_hit) first_hit = o;
    }
    if (lane == 0) s_min_idx[warp] = first_hit;
    __syncthreads();

    if (warp == 0) {
        int m = (lane < kWarps) ? s_min_idx[lane] : vocab_size;
        #pragma unroll
        for (int off = kWarpSize / 2; off > 0; off >>= 1) {
            const int o = __shfl_down_sync(kFullMask, m, off);
            if (o < m) m = o;
        }
        if (lane == 0) {
            int cutoff = m + 1;
            if (cutoff > vocab_size) cutoff = vocab_size;
            if (cutoff < 1) cutoff = 1;
            g_base_cutoff_idx = cutoff;
        }
    }
    __syncthreads();

    const int cutoff = g_base_cutoff_idx;
    float lsum = 0.0f;
    for (int i = tid; i < cutoff; i += kThreads) {
        lsum += g_base_probs[i];
    }
    lsum = warp_reduce_sum(lsum);
    if (lane == 0) s_warp[warp] = lsum;
    __syncthreads();
    if (warp == 0) {
        float bsum = (lane < kWarps) ? s_warp[lane] : 0.0f;
        bsum = warp_reduce_sum(bsum);
        if (lane == 0) {
            g_base_inv_nuc_sum = 1.0f / bsum;
        }
    }
}

__global__ __launch_bounds__(kThreads) void base_exponential_race_kernel(
    const int* __restrict__ seed,
    int* __restrict__ sampled_token) {
    __shared__ double s_score[kWarps];
    __shared__ int s_pos[kWarps];

    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid >> 5;

    const int cutoff = g_base_cutoff_idx;
    const float inv_nuc = g_base_inv_nuc_sum;
    const uint32_t seed_hash = pcg_hash_u32(static_cast<uint32_t>(seed[0]));

    double best_score = -INFINITY;
    int best_pos = cutoff;

    for (int pos = tid; pos < cutoff; pos += kThreads) {
        float prob = 0.0f;
        int tok_idx = 0;
        unpack_key(g_base_keys[pos], &prob, &tok_idx);
        const float q = prob * inv_nuc;
        const uint32_t bits = pcg_hash_u32(seed_hash + static_cast<uint32_t>(tok_idx));
        const double u = (static_cast<double>(bits) + 0.5) * (1.0 / 4294967296.0);
        const double score = static_cast<double>(q) / (-log(u));
        if (score > best_score || (score == best_score && pos < best_pos)) {
            best_score = score;
            best_pos = pos;
        }
    }

    #pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        const double os = __shfl_down_sync(kFullMask, best_score, off);
        const int op = __shfl_down_sync(kFullMask, best_pos, off);
        if (os > best_score || (os == best_score && op < best_pos)) {
            best_score = os;
            best_pos = op;
        }
    }
    if (lane == 0) {
        s_score[warp] = best_score;
        s_pos[warp] = best_pos;
    }
    __syncthreads();

    if (warp == 0) {
        double bs = (lane < kWarps) ? s_score[lane] : -INFINITY;
        int bp = (lane < kWarps) ? s_pos[lane] : cutoff;
        #pragma unroll
        for (int off = kWarpSize / 2; off > 0; off >>= 1) {
            const double os = __shfl_down_sync(kFullMask, bs, off);
            const int op = __shfl_down_sync(kFullMask, bp, off);
            if (os > bs || (os == bs && op < bp)) {
                bs = os;
                bp = op;
            }
        }
        if (lane == 0) {
            float dummy_p = 0.0f;
            int win_tok = 0;
            unpack_key(g_base_keys[bp], &dummy_p, &win_tok);
            sampled_token[0] = win_tok;
        }
    }
}

}  // namespace

extern "C" void solve(const float* logits, const float* p, const int* seed, int* sampled_token,
                      int vocab_size) {
    if (vocab_size <= 0) {
        return;
    }

    int padded_n = 1;
    while (padded_n < vocab_size) {
        padded_n <<= 1;
    }

    int stats_blocks = (vocab_size + kThreads - 1) / kThreads;
    if (stats_blocks > kMaxBlocks) stats_blocks = kMaxBlocks;

    base_softmax_stats_kernel<<<stats_blocks, kThreads>>>(logits, vocab_size);
    base_softmax_finalize_kernel<<<1, kThreads>>>(stats_blocks);

    const int init_blocks = (padded_n + kThreads - 1) / kThreads;
    base_init_sort_keys_kernel<<<(init_blocks < kMaxBlocks ? init_blocks : kMaxBlocks), kThreads>>>(
        logits, vocab_size, padded_n);

    for (int k = 2; k <= padded_n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            base_bitonic_step_kernel<<<init_blocks, kThreads>>>(j, k, padded_n);
        }
    }

    const int num_tiles = (vocab_size + 1023) / 1024;
    base_unpack_and_tile_scan_kernel<<<num_tiles, kThreads>>>(vocab_size);
    base_find_cutoff_and_renorm_kernel<<<1, kThreads>>>(p, vocab_size, num_tiles);
    base_exponential_race_kernel<<<1, kThreads>>>(seed, sampled_token);
}
