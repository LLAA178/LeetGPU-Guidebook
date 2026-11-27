#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kMaxElemsPerLane = 4;  // Supports d <= 128

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffffu, val, offset);
  }
  return val;
}

__device__ __forceinline__ void online_softmax_step(
    const float* __restrict__ K_row,
    const float* __restrict__ V_row,
    const float (&q_reg)[kMaxElemsPerLane],
    float (&o_reg)[kMaxElemsPerLane],
    float& m,
    float& l,
    int d,
    int lane,
    float scale) {
  float dot = 0.0f;
#pragma unroll
  for (int c = 0; c < kMaxElemsPerLane; ++c) {
    int col = lane + c * kWarpSize;
    if (col < d) {
      dot += q_reg[c] * K_row[col];
    }
  }
  float score = warp_reduce_sum(dot) * scale;

  float m_new = fmaxf(m, score);
  float alpha = expf(m - m_new);
  float p = expf(score - m_new);
  l = l * alpha + p;
  m = m_new;

#pragma unroll
  for (int c = 0; c < kMaxElemsPerLane; ++c) {
    int col = lane + c * kWarpSize;
    if (col < d) {
      o_reg[c] = o_reg[c] * alpha + p * V_row[col];
    }
  }
}

__global__ void attention_with_sinks_baseline_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int M,
    int d,
    int num_sinks,
    int window_size) {
  int warp_id = threadIdx.x / kWarpSize;
  int lane = threadIdx.x % kWarpSize;
  int row = blockIdx.x * kWarpsPerBlock + warp_id;
  if (row >= M) return;

  float scale = rsqrtf(static_cast<float>(d));
  const float* q_ptr = Q + static_cast<size_t>(row) * d;

  float q_reg[kMaxElemsPerLane];
  float o_reg[kMaxElemsPerLane];
#pragma unroll
  for (int c = 0; c < kMaxElemsPerLane; ++c) {
    int col = lane + c * kWarpSize;
    q_reg[c] = (col < d) ? q_ptr[col] : 0.0f;
    o_reg[c] = 0.0f;
  }

  float m = -INFINITY;
  float l = 0.0f;

  int win_start = row - window_size + 1;
  if (win_start < 0) win_start = 0;

  if (win_start <= num_sinks) {
    for (int j = 0; j <= row; ++j) {
      online_softmax_step(K + static_cast<size_t>(j) * d,
                          V + static_cast<size_t>(j) * d,
                          q_reg, o_reg, m, l, d, lane, scale);
    }
  } else {
    for (int j = 0; j < num_sinks; ++j) {
      online_softmax_step(K + static_cast<size_t>(j) * d,
                          V + static_cast<size_t>(j) * d,
                          q_reg, o_reg, m, l, d, lane, scale);
    }
    for (int j = win_start; j <= row; ++j) {
      online_softmax_step(K + static_cast<size_t>(j) * d,
                          V + static_cast<size_t>(j) * d,
                          q_reg, o_reg, m, l, d, lane, scale);
    }
  }

  float inv_l = 1.0f / l;
  float* out_ptr = output + static_cast<size_t>(row) * d;
#pragma unroll
  for (int c = 0; c < kMaxElemsPerLane; ++c) {
    int col = lane + c * kWarpSize;
    if (col < d) {
      out_ptr[col] = o_reg[c] * inv_l;
    }
  }
}

}  // namespace

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V,
                      float* output, int M, int d, int num_sinks,
                      int window_size) {
  if (M <= 0 || d <= 0) return;
  int blocks = (M + kWarpsPerBlock - 1) / kWarpsPerBlock;
  attention_with_sinks_baseline_kernel<<<blocks, kWarpsPerBlock * kWarpSize>>>(
      Q, K, V, output, M, d, num_sinks, window_size);
}
