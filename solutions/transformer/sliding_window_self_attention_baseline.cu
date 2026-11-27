#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr int kMaxElemsPerLane = 4;  // Supports d <= 128

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffffu, val, offset);
  }
  return val;
}

__global__ void sliding_window_attn_baseline_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int M,
    int d,
    int window_size) {
  int warp_id = threadIdx.x / kWarpSize;
  int lane = threadIdx.x % kWarpSize;
  int row = blockIdx.x * kWarpsPerBlock + warp_id;
  if (row >= M) return;

  float q_reg[kMaxElemsPerLane];
  float o_reg[kMaxElemsPerLane];
#pragma unroll
  for (int e = 0; e < kMaxElemsPerLane; ++e) {
    int col = lane + e * kWarpSize;
    q_reg[e] = (col < d) ? Q[row * d + col] : 0.0f;
    o_reg[e] = 0.0f;
  }

  float scale = rsqrtf(static_cast<float>(d));
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
    float score = warp_reduce_sum(dot) * scale;

    float m_new = fmaxf(m, score);
    float alpha = expf(m - m_new);
    float beta = expf(score - m_new);
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

}  // namespace

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V,
                      float* output, int M, int d, int window_size) {
  if (M <= 0 || d <= 0) return;
  int grid = (M + kWarpsPerBlock - 1) / kWarpsPerBlock;
  sliding_window_attn_baseline_kernel<<<grid, kThreadsPerBlock>>>(
      Q, K, V, output, M, d, window_size);
}
