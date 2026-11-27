#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kMaxElemsPerLane = 4;  // Supports d <= 128
constexpr float kLog2E = 1.44269504088896340736f;

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffffu, val, offset);
  }
  return val;
}

__device__ __forceinline__ float dot4(float4 a, float4 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __forceinline__ float4 fma4(float4 o, float alpha, float p, float4 v) {
  return make_float4(o.x * alpha + p * v.x,
                     o.y * alpha + p * v.y,
                     o.z * alpha + p * v.z,
                     o.w * alpha + p * v.w);
}

// Optimized tiled FlashAttention for d <= 128, d % 4 == 0:
// - Br = 32 query rows per block (8 warps = 256 threads, 4 Q-rows per warp).
// - Bc = 32 KV rows per shared-memory tile (sK + sV = 32 KB, float4 aligned).
// - Each thread holds 1 float4 per Q-row (4 Q-rows -> 4x register reuse of sK/sV).
// - Pre-scales Q by rsqrtf(d) * log2(e) so softmax uses single-instruction exp2f.
// - Fast unmasked inner loop for interior tiles (all 32 Q-rows attend to all tile keys).
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * kWarpSize;  // 256
constexpr int kRowsPerWarp = 4;
constexpr int kBr = kWarps * kRowsPerWarp;    // 32
constexpr int kBc = 32;

__device__ __forceinline__ void load_kv_tile(
    float4 (&sK)[kBc][32],
    float4 (&sV)[kBc][32],
    const float4* __restrict__ K4,
    const float4* __restrict__ V4,
    int tile_start,
    int tile_len,
    int d4,
    int tid) {
  int total_vecs = tile_len * d4;
  if (d4 == 32) {
    for (int idx = tid; idx < total_vecs; idx += kThreads) {
      int r = idx >> 5;
      int c = idx & 31;
      int g_row = tile_start + r;
      sK[r][c] = K4[static_cast<size_t>(g_row) * 32 + c];
      sV[r][c] = V4[static_cast<size_t>(g_row) * 32 + c];
    }
  } else {
    for (int idx = tid; idx < total_vecs; idx += kThreads) {
      int r = idx / d4;
      int c = idx - r * d4;
      int g_row = tile_start + r;
      sK[r][c] = K4[static_cast<size_t>(g_row) * d4 + c];
      sV[r][c] = V4[static_cast<size_t>(g_row) * d4 + c];
    }
  }
}

__device__ __forceinline__ void process_tile_unmasked(
    const float4 (&sK)[kBc][32],
    const float4 (&sV)[kBc][32],
    int tile_len,
    int lane,
    bool active_lane,
    const float4 (&q)[kRowsPerWarp],
    float4 (&o)[kRowsPerWarp],
    float (&m)[kRowsPerWarp],
    float (&l)[kRowsPerWarp]) {
  for (int j = 0; j < tile_len; ++j) {
    float4 k_vec = active_lane ? sK[j][lane] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 v_vec = active_lane ? sV[j][lane] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float s0 = dot4(q[0], k_vec);
    float s1 = dot4(q[1], k_vec);
    float s2 = dot4(q[2], k_vec);
    float s3 = dot4(q[3], k_vec);

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      s0 += __shfl_xor_sync(0xffffffffu, s0, offset);
      s1 += __shfl_xor_sync(0xffffffffu, s1, offset);
      s2 += __shfl_xor_sync(0xffffffffu, s2, offset);
      s3 += __shfl_xor_sync(0xffffffffu, s3, offset);
    }

    float scores[kRowsPerWarp] = {s0, s1, s2, s3};
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      float m_new = fmaxf(m[r], scores[r]);
      float alpha = exp2f(m[r] - m_new);
      float p = exp2f(scores[r] - m_new);
      l[r] = l[r] * alpha + p;
      m[r] = m_new;
      o[r] = fma4(o[r], alpha, p, v_vec);
    }
  }
}

__device__ __forceinline__ void process_tile_masked(
    const float4 (&sK)[kBc][32],
    const float4 (&sV)[kBc][32],
    int tile_start,
    int tile_len,
    int lane,
    bool active_lane,
    int num_sinks,
    int window_size,
    const int (&qr)[kRowsPerWarp],
    const float4 (&q)[kRowsPerWarp],
    float4 (&o)[kRowsPerWarp],
    float (&m)[kRowsPerWarp],
    float (&l)[kRowsPerWarp]) {
  for (int j = 0; j < tile_len; ++j) {
    int g_col = tile_start + j;
    float4 k_vec = active_lane ? sK[j][lane] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 v_vec = active_lane ? sV[j][lane] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      int row = qr[r];
      bool allowed = (g_col <= row) &&
                     (g_col < num_sinks || g_col >= row - window_size + 1);
      float dot = dot4(q[r], k_vec);
      float s = warp_reduce_sum(dot);
      if (allowed) {
        float m_new = fmaxf(m[r], s);
        float alpha = exp2f(m[r] - m_new);
        float p = exp2f(s - m_new);
        l[r] = l[r] * alpha + p;
        m[r] = m_new;
        o[r] = fma4(o[r], alpha, p, v_vec);
      }
    }
  }
}

__global__ void __launch_bounds__(kThreads, 2)
attention_with_sinks_opt_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int M,
    int d,
    int num_sinks,
    int window_size) {
  __shared__ float4 sK[kBc][32];
  __shared__ float4 sV[kBc][32];

  int tid = threadIdx.x;
  int warp_id = tid >> 5;
  int lane = tid & 31;
  int d4 = d >> 2;
  bool active_lane = (lane < d4);

  int block_q_start = blockIdx.x * kBr;
  int block_q_end = block_q_start + kBr - 1;
  if (block_q_end >= M) block_q_end = M - 1;

  const float4* Q4 = reinterpret_cast<const float4*>(Q);
  const float4* K4 = reinterpret_cast<const float4*>(K);
  const float4* V4 = reinterpret_cast<const float4*>(V);
  float4* O4 = reinterpret_cast<float4*>(output);

  float scale_log2e = rsqrtf(static_cast<float>(d)) * kLog2E;

  int qr[kRowsPerWarp];
  float4 q[kRowsPerWarp];
  float4 o[kRowsPerWarp];
  float m[kRowsPerWarp];
  float l[kRowsPerWarp];

#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
    int row = block_q_start + warp_id * kRowsPerWarp + r;
    qr[r] = row;
    if (row < M && active_lane) {
      float4 v = Q4[static_cast<size_t>(row) * d4 + lane];
      q[r] = make_float4(v.x * scale_log2e, v.y * scale_log2e,
                         v.z * scale_log2e, v.w * scale_log2e);
    } else {
      q[r] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    o[r] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    m[r] = -INFINITY;
    l[r] = 0.0f;
  }

  int block_win_start = block_q_start - window_size + 1;
  if (block_win_start < 0) block_win_start = 0;

  if (block_win_start <= num_sinks) {
    // Union of sinks and window is a single contiguous range [0, block_q_end].
    for (int t_start = 0; t_start <= block_q_end; t_start += kBc) {
      int tile_len = block_q_end - t_start + 1;
      if (tile_len > kBc) tile_len = kBc;
      int t_end = t_start + tile_len - 1;

      load_kv_tile(sK, sV, K4, V4, t_start, tile_len, d4, tid);
      __syncthreads();

      bool all_in_causal = (t_end <= block_q_start);
      bool all_in_window = (t_start >= block_q_end - window_size + 1);
      bool all_in_sinks = (t_end < num_sinks);

      if (all_in_causal && (all_in_window || all_in_sinks)) {
        process_tile_unmasked(sK, sV, tile_len, lane, active_lane, q, o, m, l);
      } else {
        process_tile_masked(sK, sV, t_start, tile_len, lane, active_lane,
                            num_sinks, window_size, qr, q, o, m, l);
      }
      __syncthreads();
    }
  } else {
    // Disjoint ranges:
    // 1) Sinks [0, num_sinks - 1] (num_sinks <= 16 <= kBc; all block queries attend to all sinks)
    load_kv_tile(sK, sV, K4, V4, 0, num_sinks, d4, tid);
    __syncthreads();
    process_tile_unmasked(sK, sV, num_sinks, lane, active_lane, q, o, m, l);
    __syncthreads();

    // 2) Sliding window [block_win_start, block_q_end]
    for (int t_start = block_win_start; t_start <= block_q_end; t_start += kBc) {
      int tile_len = block_q_end - t_start + 1;
      if (tile_len > kBc) tile_len = kBc;
      int t_end = t_start + tile_len - 1;

      load_kv_tile(sK, sV, K4, V4, t_start, tile_len, d4, tid);
      __syncthreads();

      if (t_end <= block_q_start && t_start >= block_q_end - window_size + 1) {
        process_tile_unmasked(sK, sV, tile_len, lane, active_lane, q, o, m, l);
      } else {
        process_tile_masked(sK, sV, t_start, tile_len, lane, active_lane,
                            num_sinks, window_size, qr, q, o, m, l);
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
    int row = qr[r];
    if (row < M && active_lane) {
      float inv_l = 1.0f / l[r];
      O4[static_cast<size_t>(row) * d4 + lane] = make_float4(
          o[r].x * inv_l, o[r].y * inv_l, o[r].z * inv_l, o[r].w * inv_l);
    }
  }
}

// General fallback for arbitrary d (e.g. d % 4 != 0 or unaligned pointers).
constexpr int kFallbackWarps = 4;

__device__ __forceinline__ void fallback_step(
    const float* __restrict__ K_row,
    const float* __restrict__ V_row,
    const float (&q_reg)[kMaxElemsPerLane],
    float (&o_reg)[kMaxElemsPerLane],
    float& m,
    float& l,
    int d,
    int lane) {
  float dot = 0.0f;
#pragma unroll
  for (int c = 0; c < kMaxElemsPerLane; ++c) {
    int col = lane + c * kWarpSize;
    if (col < d) dot += q_reg[c] * K_row[col];
  }
  float s = warp_reduce_sum(dot);
  float m_new = fmaxf(m, s);
  float alpha = exp2f(m - m_new);
  float p = exp2f(s - m_new);
  l = l * alpha + p;
  m = m_new;
#pragma unroll
  for (int c = 0; c < kMaxElemsPerLane; ++c) {
    int col = lane + c * kWarpSize;
    if (col < d) o_reg[c] = o_reg[c] * alpha + p * V_row[col];
  }
}

__global__ void attention_with_sinks_fallback_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int M,
    int d,
    int num_sinks,
    int window_size) {
  int warp_id = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int row = blockIdx.x * kFallbackWarps + warp_id;
  if (row >= M) return;

  float scale_log2e = rsqrtf(static_cast<float>(d)) * kLog2E;
  const float* q_ptr = Q + static_cast<size_t>(row) * d;

  float q_reg[kMaxElemsPerLane];
  float o_reg[kMaxElemsPerLane];
#pragma unroll
  for (int c = 0; c < kMaxElemsPerLane; ++c) {
    int col = lane + c * kWarpSize;
    q_reg[c] = (col < d) ? q_ptr[col] * scale_log2e : 0.0f;
    o_reg[c] = 0.0f;
  }

  float m = -INFINITY;
  float l = 0.0f;
  int win_start = row - window_size + 1;
  if (win_start < 0) win_start = 0;

  if (win_start <= num_sinks) {
    for (int j = 0; j <= row; ++j) {
      fallback_step(K + static_cast<size_t>(j) * d,
                    V + static_cast<size_t>(j) * d,
                    q_reg, o_reg, m, l, d, lane);
    }
  } else {
    for (int j = 0; j < num_sinks; ++j) {
      fallback_step(K + static_cast<size_t>(j) * d,
                    V + static_cast<size_t>(j) * d,
                    q_reg, o_reg, m, l, d, lane);
    }
    for (int j = win_start; j <= row; ++j) {
      fallback_step(K + static_cast<size_t>(j) * d,
                    V + static_cast<size_t>(j) * d,
                    q_reg, o_reg, m, l, d, lane);
    }
  }

  float inv_l = 1.0f / l;
  float* out_ptr = output + static_cast<size_t>(row) * d;
#pragma unroll
  for (int c = 0; c < kMaxElemsPerLane; ++c) {
    int col = lane + c * kWarpSize;
    if (col < d) out_ptr[col] = o_reg[c] * inv_l;
  }
}

}  // namespace

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V,
                      float* output, int M, int d, int num_sinks,
                      int window_size) {
  if (M <= 0 || d <= 0) return;

  bool aligned16 = ((reinterpret_cast<uintptr_t>(Q) & 15u) == 0) &&
                   ((reinterpret_cast<uintptr_t>(K) & 15u) == 0) &&
                   ((reinterpret_cast<uintptr_t>(V) & 15u) == 0) &&
                   ((reinterpret_cast<uintptr_t>(output) & 15u) == 0);

  if (aligned16 && d <= 128 && (d % 4) == 0 && num_sinks <= kBc) {
    int blocks = (M + kBr - 1) / kBr;
    attention_with_sinks_opt_kernel<<<blocks, kThreads>>>(
        Q, K, V, output, M, d, num_sinks, window_size);
  } else {
    int blocks = (M + kFallbackWarps - 1) / kFallbackWarps;
    attention_with_sinks_fallback_kernel<<<blocks, kFallbackWarps * kWarpSize>>>(
        Q, K, V, output, M, d, num_sinks, window_size);
  }
}
