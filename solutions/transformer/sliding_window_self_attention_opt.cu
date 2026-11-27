#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kFallbackWarps = 4;
constexpr int kMaxElemsPerLane = 4;  // Supports d <= 128
constexpr float kLog2E = 1.44269504088896340736f;

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffffu, val, offset);
  }
  return val;
}

__device__ __forceinline__ float halfwarp_reduce_sum(unsigned mask, float val) {
#pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(mask, val, offset);
  }
  return val;
}

__device__ __forceinline__ float subwarp8_reduce_sum(float val) {
#pragma unroll
  for (int offset = 4; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffffu, val, offset);
  }
  return val;
}

// Single-load zero-loop shared-window fast path for d <= 64, d % 4 == 0, W <= 16:
// - Br = 32 query rows per block (256 threads = 32 sub-warps of 8 threads).
// - Entire block's KV window (<= 32 + 2*16 = 64 rows) is loaded ONCE into
//   shared memory (32 KB total) with float4, followed by ONE __syncthreads().
// - Each 8-thread sub-warp owns 1 query row (2 float4s per thread = 8 floats).
// - Pass 1 computes all <= 33 scores in registers using 3-shuffle subwarp8
//   reduction and tracks max_s.
// - Pass 2 computes exp2f(score - max_s) once per key and accumulates P @ V
//   directly with ZERO output rescaling.
constexpr int kFastBr = 32;
constexpr int kFastMaxKV = 64;
constexpr int kFastMaxWin = 33;  // 2 * 16 + 1
constexpr int kFastThreads = 256;

__global__ void __launch_bounds__(kFastThreads, 2)
sliding_window_attn_fast_d64_w16_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int M,
    int d,
    int window_size) {
  __shared__ float4 sK[kFastMaxKV][16];
  __shared__ float4 sV[kFastMaxKV][16];

  int tid = threadIdx.x;
  int d4 = d >> 2;
  int block_q_start = blockIdx.x * kFastBr;
  int block_q_end = block_q_start + kFastBr - 1;
  if (block_q_end >= M) block_q_end = M - 1;

  int kv_start = block_q_start - window_size;
  if (kv_start < 0) kv_start = 0;
  int kv_end = block_q_end + window_size;
  if (kv_end >= M) kv_end = M - 1;
  int kv_len = kv_end - kv_start + 1;

  const float4* Q4 = reinterpret_cast<const float4*>(Q);
  const float4* K4 = reinterpret_cast<const float4*>(K);
  const float4* V4 = reinterpret_cast<const float4*>(V);
  float4* O4 = reinterpret_cast<float4*>(output);

  // Single cooperative float4 load of the entire block's K and V window.
  int total_vecs = kv_len * d4;
  if (d4 == 16) {
    for (int idx = tid; idx < total_vecs; idx += kFastThreads) {
      int r = idx >> 4;
      int c = idx & 15;
      int g_row = kv_start + r;
      sK[r][c] = K4[g_row * 16 + c];
      sV[r][c] = V4[g_row * 16 + c];
    }
  } else {
    for (int idx = tid; idx < total_vecs; idx += kFastThreads) {
      int r = idx / d4;
      int c = idx - r * d4;
      int g_row = kv_start + r;
      sK[r][c] = K4[g_row * d4 + c];
      sV[r][c] = V4[g_row * d4 + c];
    }
  }
  __syncthreads();

  int row_in_block = tid >> 3;  // 0..31
  int tx = tid & 7;             // 0..7
  int qr = block_q_start + row_in_block;
  bool valid_row = (qr < M);
  int safe_qr = valid_row ? qr : (M - 1);

  float scale_log2e = rsqrtf(static_cast<float>(d)) * kLog2E;

  float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float4 q1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  if (tx < d4) {
    float4 v0 = Q4[safe_qr * d4 + tx];
    q0 = make_float4(v0.x * scale_log2e, v0.y * scale_log2e,
                     v0.z * scale_log2e, v0.w * scale_log2e);
  }
  if (tx + 8 < d4) {
    float4 v1 = Q4[safe_qr * d4 + (tx + 8)];
    q1 = make_float4(v1.x * scale_log2e, v1.y * scale_log2e,
                     v1.z * scale_log2e, v1.w * scale_log2e);
  }

  int j_lo = safe_qr - window_size - kv_start;
  if (j_lo < 0) j_lo = 0;
  int j_hi = safe_qr + window_size - kv_start;
  if (j_hi >= kv_len) j_hi = kv_len - 1;
  int win_len = j_hi - j_lo + 1;

  float scores[kFastMaxWin];
  float max_s = -INFINITY;
  float sum_p = 0.0f;
  float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float4 o1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  if (d4 == 16 && win_len == 33) {
    // Hot path for M=5000, d=64, W=16 interior rows (99.4% of rows):
    // compile-time unroll keeps scores[33] in registers with 0 stack frame.
#pragma unroll
    for (int k = 0; k < 33; ++k) {
      int j = j_lo + k;
      float4 k0 = sK[j][tx];
      float4 k1 = sK[j][tx + 8];
      float dot = fmaf(q0.x, k0.x, fmaf(q0.y, k0.y, fmaf(q0.z, k0.z, q0.w * k0.w)));
      dot = fmaf(q1.x, k1.x, fmaf(q1.y, k1.y, fmaf(q1.z, k1.z, fmaf(q1.w, k1.w, dot))));
      float s = subwarp8_reduce_sum(dot);
      scores[k] = s;
      max_s = fmaxf(max_s, s);
    }
#pragma unroll
    for (int k = 0; k < 33; ++k) {
      int j = j_lo + k;
      float p = exp2f(scores[k] - max_s);
      sum_p += p;
      float4 v0 = sV[j][tx];
      float4 v1 = sV[j][tx + 8];
      o0.x = fmaf(p, v0.x, o0.x);
      o0.y = fmaf(p, v0.y, o0.y);
      o0.z = fmaf(p, v0.z, o0.z);
      o0.w = fmaf(p, v0.w, o0.w);
      o1.x = fmaf(p, v1.x, o1.x);
      o1.y = fmaf(p, v1.y, o1.y);
      o1.z = fmaf(p, v1.z, o1.z);
      o1.w = fmaf(p, v1.w, o1.w);
    }
  } else {
    // Boundary rows or smaller d/W: online softmax (0 stack frame).
    for (int k = 0; k < win_len; ++k) {
      int j = j_lo + k;
      float dot = 0.0f;
      if (tx < d4) {
        float4 k0 = sK[j][tx];
        dot = fmaf(q0.x, k0.x, fmaf(q0.y, k0.y, fmaf(q0.z, k0.z, q0.w * k0.w)));
      }
      if (tx + 8 < d4) {
        float4 k1 = sK[j][tx + 8];
        dot = fmaf(q1.x, k1.x, fmaf(q1.y, k1.y, fmaf(q1.z, k1.z, fmaf(q1.w, k1.w, dot))));
      }
      float s = subwarp8_reduce_sum(dot);
      float m_new = fmaxf(max_s, s);
      float alpha = exp2f(max_s - m_new);
      float beta = exp2f(s - m_new);
      sum_p = fmaf(sum_p, alpha, beta);
      max_s = m_new;
      if (tx < d4) {
        float4 v0 = sV[j][tx];
        o0.x = fmaf(o0.x, alpha, beta * v0.x);
        o0.y = fmaf(o0.y, alpha, beta * v0.y);
        o0.z = fmaf(o0.z, alpha, beta * v0.z);
        o0.w = fmaf(o0.w, alpha, beta * v0.w);
      }
      if (tx + 8 < d4) {
        float4 v1 = sV[j][tx + 8];
        o1.x = fmaf(o1.x, alpha, beta * v1.x);
        o1.y = fmaf(o1.y, alpha, beta * v1.y);
        o1.z = fmaf(o1.z, alpha, beta * v1.z);
        o1.w = fmaf(o1.w, alpha, beta * v1.w);
      }
    }
  }

  if (valid_row) {
    float inv_sum = 1.0f / sum_p;
    if (tx < d4) {
      O4[qr * d4 + tx] = make_float4(
          o0.x * inv_sum, o0.y * inv_sum, o0.z * inv_sum, o0.w * inv_sum);
    }
    if (tx + 8 < d4) {
      O4[qr * d4 + (tx + 8)] = make_float4(
          o1.x * inv_sum, o1.y * inv_sum, o1.z * inv_sum, o1.w * inv_sum);
    }
  }
}

// General fallback for d % 4 != 0 or unaligned pointers.
__global__ void sliding_window_attn_fallback_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int M,
    int d,
    int window_size) {
  int warp_id = threadIdx.x / kWarpSize;
  int lane = threadIdx.x % kWarpSize;
  int row = blockIdx.x * kFallbackWarps + warp_id;
  if (row >= M) return;

  float scale_log2e = rsqrtf(static_cast<float>(d)) * kLog2E;
  float q_reg[kMaxElemsPerLane];
  float o_reg[kMaxElemsPerLane];
#pragma unroll
  for (int e = 0; e < kMaxElemsPerLane; ++e) {
    int col = lane + e * kWarpSize;
    q_reg[e] = (col < d) ? Q[row * d + col] * scale_log2e : 0.0f;
    o_reg[e] = 0.0f;
  }

  int j_start = row - window_size;
  if (j_start < 0) j_start = 0;
  int j_end = row + window_size;
  if (j_end >= M) j_end = M - 1;

  float m = -INFINITY;
  float l = 0.0f;

  for (int j = j_start; j <= j_end; ++j) {
    const float* k_row = K + j * d;
    float dot = 0.0f;
#pragma unroll
    for (int e = 0; e < kMaxElemsPerLane; ++e) {
      int col = lane + e * kWarpSize;
      if (col < d) {
        dot = fmaf(q_reg[e], k_row[col], dot);
      }
    }
    float score = warp_reduce_sum(dot);

    float m_new = fmaxf(m, score);
    float alpha = exp2f(m - m_new);
    float beta = exp2f(score - m_new);
    l = fmaf(l, alpha, beta);
    m = m_new;

    const float* v_row = V + j * d;
#pragma unroll
    for (int e = 0; e < kMaxElemsPerLane; ++e) {
      int col = lane + e * kWarpSize;
      if (col < d) {
        o_reg[e] = fmaf(o_reg[e], alpha, beta * v_row[col]);
      }
    }
  }

  float inv_l = 1.0f / l;
  float* out_row = output + row * d;
#pragma unroll
  for (int e = 0; e < kMaxElemsPerLane; ++e) {
    int col = lane + e * kWarpSize;
    if (col < d) {
      out_row[col] = o_reg[e] * inv_l;
    }
  }
}

// Optimized FlashAttention-style sliding window kernel for d <= 64, d % 4 == 0:
// - Br = 32 query rows per block (128 threads = 8 half-warps; each half-warp
//   owns 4 consecutive query rows and covers up to 64 columns via float4).
// - Bc = 32 KV rows staged in shared memory per tile (16 KB total, zero bank
//   conflicts for half-warp float4 row reads).
// - Pre-scales Q by (1 / sqrt(d)) * log2(e) so Online Softmax uses single-
//   instruction hardware exp2f (MUFU.EX2).
// - Block only iterates KV tiles intersecting [q_start - W, q_end + W], and
//   each half-warp skips KV rows outside its 4-row window union.
// - 4x register reuse of loaded K/V float4 values across the 4 query rows.
constexpr int kBr64 = 32;
constexpr int kBc = 32;
constexpr int kRowsPerGroup = 4;
constexpr int kThreads64 = 128;

__global__ void __launch_bounds__(kThreads64, 4)
sliding_window_attn_opt_d64_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int M,
    int d,
    int window_size) {
  __shared__ float4 sK[kBc][16];
  __shared__ float4 sV[kBc][16];

  int tid = threadIdx.x;
  int group_id = tid >> 4;  // 0..7 (8 half-warps)
  int col4 = tid & 15;      // 0..15 (each owns 4 floats)
  int d4 = d >> 2;
  unsigned mask = ((tid & 31) < 16) ? 0x0000ffffu : 0xffff0000u;

  int block_q_start = blockIdx.x * kBr64;
  int row0 = block_q_start + group_id * kRowsPerGroup;

  const float4* Q4 = reinterpret_cast<const float4*>(Q);
  const float4* K4 = reinterpret_cast<const float4*>(K);
  const float4* V4 = reinterpret_cast<const float4*>(V);
  float4* O4 = reinterpret_cast<float4*>(output);

  float scale_log2e = rsqrtf(static_cast<float>(d)) * kLog2E;

  float4 q_reg[kRowsPerGroup];
  float4 o_reg[kRowsPerGroup];
  float m_reg[kRowsPerGroup];
  float l_reg[kRowsPerGroup];

#pragma unroll
  for (int r = 0; r < kRowsPerGroup; ++r) {
    int qr = row0 + r;
    if (qr < M && col4 < d4) {
      float4 qv = Q4[qr * d4 + col4];
      q_reg[r] = make_float4(qv.x * scale_log2e, qv.y * scale_log2e,
                             qv.z * scale_log2e, qv.w * scale_log2e);
    } else {
      q_reg[r] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    o_reg[r] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    m_reg[r] = -INFINITY;
    l_reg[r] = 0.0f;
  }

  int block_q_end = block_q_start + kBr64 - 1;
  if (block_q_end >= M) block_q_end = M - 1;
  int kv_start = block_q_start - window_size;
  if (kv_start < 0) kv_start = 0;
  int kv_end = block_q_end + window_size;
  if (kv_end >= M) kv_end = M - 1;

  int group_kv_min = row0 - window_size;
  int group_kv_max = row0 + (kRowsPerGroup - 1) + window_size;

  for (int tile_start = kv_start; tile_start <= kv_end; tile_start += kBc) {
    int tile_len = kv_end - tile_start + 1;
    if (tile_len > kBc) tile_len = kBc;

    int total_vecs = tile_len * d4;
    for (int idx = tid; idx < total_vecs; idx += kThreads64) {
      int r = idx / d4;
      int c = idx - r * d4;
      int global_row = tile_start + r;
      sK[r][c] = K4[global_row * d4 + c];
      sV[r][c] = V4[global_row * d4 + c];
    }
    __syncthreads();

    int j_lo = group_kv_min - tile_start;
    if (j_lo < 0) j_lo = 0;
    int j_hi = group_kv_max - tile_start;
    if (j_hi >= tile_len) j_hi = tile_len - 1;

    if (row0 < M && j_lo <= j_hi) {
      for (int j = j_lo; j <= j_hi; ++j) {
        int kv_row = tile_start + j;
        float4 k_vec = (col4 < d4) ? sK[j][col4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 v_vec = (col4 < d4) ? sV[j][col4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

#pragma unroll
        for (int r = 0; r < kRowsPerGroup; ++r) {
          int qr = row0 + r;
          if (qr < M && abs(qr - kv_row) <= window_size) {
            float dot = fmaf(q_reg[r].x, k_vec.x,
                        fmaf(q_reg[r].y, k_vec.y,
                        fmaf(q_reg[r].z, k_vec.z, q_reg[r].w * k_vec.w)));
            float score = halfwarp_reduce_sum(mask, dot);

            float m_new = fmaxf(m_reg[r], score);
            float alpha = exp2f(m_reg[r] - m_new);
            float beta = exp2f(score - m_new);
            l_reg[r] = fmaf(l_reg[r], alpha, beta);
            m_reg[r] = m_new;

            o_reg[r].x = fmaf(o_reg[r].x, alpha, beta * v_vec.x);
            o_reg[r].y = fmaf(o_reg[r].y, alpha, beta * v_vec.y);
            o_reg[r].z = fmaf(o_reg[r].z, alpha, beta * v_vec.z);
            o_reg[r].w = fmaf(o_reg[r].w, alpha, beta * v_vec.w);
          }
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < kRowsPerGroup; ++r) {
    int qr = row0 + r;
    if (qr < M && col4 < d4) {
      float inv_l = 1.0f / l_reg[r];
      O4[qr * d4 + col4] = make_float4(
          o_reg[r].x * inv_l, o_reg[r].y * inv_l,
          o_reg[r].z * inv_l, o_reg[r].w * inv_l);
    }
  }
}

// Optimized kernel for 64 < d <= 128, d % 4 == 0:
// - Br = 32 query rows per block (256 threads = 8 warps; each warp owns
//   4 consecutive query rows and covers up to 128 columns via float4).
// - Bc = 32 KV rows staged in shared memory per tile.
constexpr int kBr128 = 32;
constexpr int kThreads128 = 256;

__global__ void __launch_bounds__(kThreads128, 2)
sliding_window_attn_opt_d128_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int M,
    int d,
    int window_size) {
  __shared__ float4 sK[kBc][32];
  __shared__ float4 sV[kBc][32];

  int tid = threadIdx.x;
  int warp_id = tid >> 5;  // 0..7
  int col4 = tid & 31;     // 0..31
  int d4 = d >> 2;

  int block_q_start = blockIdx.x * kBr128;
  int row0 = block_q_start + warp_id * kRowsPerGroup;

  const float4* Q4 = reinterpret_cast<const float4*>(Q);
  const float4* K4 = reinterpret_cast<const float4*>(K);
  const float4* V4 = reinterpret_cast<const float4*>(V);
  float4* O4 = reinterpret_cast<float4*>(output);

  float scale_log2e = rsqrtf(static_cast<float>(d)) * kLog2E;

  float4 q_reg[kRowsPerGroup];
  float4 o_reg[kRowsPerGroup];
  float m_reg[kRowsPerGroup];
  float l_reg[kRowsPerGroup];

#pragma unroll
  for (int r = 0; r < kRowsPerGroup; ++r) {
    int qr = row0 + r;
    if (qr < M && col4 < d4) {
      float4 qv = Q4[qr * d4 + col4];
      q_reg[r] = make_float4(qv.x * scale_log2e, qv.y * scale_log2e,
                             qv.z * scale_log2e, qv.w * scale_log2e);
    } else {
      q_reg[r] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    o_reg[r] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    m_reg[r] = -INFINITY;
    l_reg[r] = 0.0f;
  }

  int block_q_end = block_q_start + kBr128 - 1;
  if (block_q_end >= M) block_q_end = M - 1;
  int kv_start = block_q_start - window_size;
  if (kv_start < 0) kv_start = 0;
  int kv_end = block_q_end + window_size;
  if (kv_end >= M) kv_end = M - 1;

  int group_kv_min = row0 - window_size;
  int group_kv_max = row0 + (kRowsPerGroup - 1) + window_size;

  for (int tile_start = kv_start; tile_start <= kv_end; tile_start += kBc) {
    int tile_len = kv_end - tile_start + 1;
    if (tile_len > kBc) tile_len = kBc;

    int total_vecs = tile_len * d4;
    for (int idx = tid; idx < total_vecs; idx += kThreads128) {
      int r = idx / d4;
      int c = idx - r * d4;
      int global_row = tile_start + r;
      sK[r][c] = K4[global_row * d4 + c];
      sV[r][c] = V4[global_row * d4 + c];
    }
    __syncthreads();

    int j_lo = group_kv_min - tile_start;
    if (j_lo < 0) j_lo = 0;
    int j_hi = group_kv_max - tile_start;
    if (j_hi >= tile_len) j_hi = tile_len - 1;

    if (row0 < M && j_lo <= j_hi) {
      for (int j = j_lo; j <= j_hi; ++j) {
        int kv_row = tile_start + j;
        float4 k_vec = (col4 < d4) ? sK[j][col4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 v_vec = (col4 < d4) ? sV[j][col4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

#pragma unroll
        for (int r = 0; r < kRowsPerGroup; ++r) {
          int qr = row0 + r;
          if (qr < M && abs(qr - kv_row) <= window_size) {
            float dot = fmaf(q_reg[r].x, k_vec.x,
                        fmaf(q_reg[r].y, k_vec.y,
                        fmaf(q_reg[r].z, k_vec.z, q_reg[r].w * k_vec.w)));
            float score = warp_reduce_sum(dot);

            float m_new = fmaxf(m_reg[r], score);
            float alpha = exp2f(m_reg[r] - m_new);
            float beta = exp2f(score - m_new);
            l_reg[r] = fmaf(l_reg[r], alpha, beta);
            m_reg[r] = m_new;

            o_reg[r].x = fmaf(o_reg[r].x, alpha, beta * v_vec.x);
            o_reg[r].y = fmaf(o_reg[r].y, alpha, beta * v_vec.y);
            o_reg[r].z = fmaf(o_reg[r].z, alpha, beta * v_vec.z);
            o_reg[r].w = fmaf(o_reg[r].w, alpha, beta * v_vec.w);
          }
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < kRowsPerGroup; ++r) {
    int qr = row0 + r;
    if (qr < M && col4 < d4) {
      float inv_l = 1.0f / l_reg[r];
      O4[qr * d4 + col4] = make_float4(
          o_reg[r].x * inv_l, o_reg[r].y * inv_l,
          o_reg[r].z * inv_l, o_reg[r].w * inv_l);
    }
  }
}

}  // namespace

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V,
                      float* output, int M, int d, int window_size) {
  if (M <= 0 || d <= 0) return;

  bool aligned16 =
      ((reinterpret_cast<uintptr_t>(Q) & 15) == 0) &&
      ((reinterpret_cast<uintptr_t>(K) & 15) == 0) &&
      ((reinterpret_cast<uintptr_t>(V) & 15) == 0) &&
      ((reinterpret_cast<uintptr_t>(output) & 15) == 0);

  if ((d & 3) == 0 && aligned16) {
    if (d <= 64 && window_size <= 16) {
      int grid = (M + kFastBr - 1) / kFastBr;
      sliding_window_attn_fast_d64_w16_kernel<<<grid, kFastThreads>>>(
          Q, K, V, output, M, d, window_size);
      return;
    } else if (d <= 64) {
      int grid = (M + kBr64 - 1) / kBr64;
      sliding_window_attn_opt_d64_kernel<<<grid, kThreads64>>>(
          Q, K, V, output, M, d, window_size);
      return;
    } else if (d <= 128) {
      int grid = (M + kBr128 - 1) / kBr128;
      sliding_window_attn_opt_d128_kernel<<<grid, kThreads128>>>(
          Q, K, V, output, M, d, window_size);
      return;
    }
  }

  int grid = (M + kFallbackWarps - 1) / kFallbackWarps;
  sliding_window_attn_fallback_kernel<<<grid, kWarpSize * kFallbackWarps>>>(
      Q, K, V, output, M, d, window_size);
}
