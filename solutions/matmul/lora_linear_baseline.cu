#include <cuda_runtime.h>

namespace {

constexpr int kTile = 16;

__global__ void lora_down_baseline_kernel(const float* __restrict__ x,
                                          const float* __restrict__ A,
                                          float* __restrict__ hidden,
                                          int batch, int d_in, int rank,
                                          float lora_scale) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= batch || col >= rank) return;

  float acc = 0.0f;
  const float* x_row = x + row * d_in;
  const float* a_row = A + col * d_in;
  for (int k = 0; k < d_in; ++k) {
    acc += x_row[k] * a_row[k];
  }
  hidden[row * rank + col] = lora_scale * acc;
}

__global__ void lora_linear_baseline_kernel(const float* __restrict__ x,
                                            const float* __restrict__ W,
                                            const float* __restrict__ hidden,
                                            const float* __restrict__ B,
                                            float* __restrict__ output,
                                            int batch, int d_in, int d_out,
                                            int rank) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= batch || col >= d_out) return;

  float acc = 0.0f;
  const float* x_row = x + row * d_in;
  const float* w_row = W + col * d_in;
  for (int k = 0; k < d_in; ++k) {
    acc += x_row[k] * w_row[k];
  }

  const float* h_row = hidden + row * rank;
  const float* b_row = B + col * rank;
  for (int r = 0; r < rank; ++r) {
    acc += h_row[r] * b_row[r];
  }

  output[row * d_out + col] = acc;
}

}  // namespace

// x, W, A, B, output are device pointers
extern "C" void solve(const float* x, const float* W, const float* A, const float* B,
                      float* output, int batch, int d_in, int d_out, int rank,
                      float lora_scale) {
  float* hidden = nullptr;
  cudaMalloc(&hidden, static_cast<size_t>(batch) * rank * sizeof(float));

  dim3 block(kTile, kTile);
  dim3 grid_down((rank + kTile - 1) / kTile, (batch + kTile - 1) / kTile);
  lora_down_baseline_kernel<<<grid_down, block>>>(x, A, hidden, batch, d_in, rank, lora_scale);

  dim3 grid_out((d_out + kTile - 1) / kTile, (batch + kTile - 1) / kTile);
  lora_linear_baseline_kernel<<<grid_out, block>>>(x, W, hidden, B, output, batch, d_in, d_out, rank);

  cudaFree(hidden);
}
