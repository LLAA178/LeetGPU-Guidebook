#include <cuda_runtime.h>
#include <math.h>

namespace {

__device__ __forceinline__ float phi_elu_plus_1(float x) {
  return x > 0.0f ? (x + 1.0f) : expf(x);
}

__global__ void compute_kv_z_baseline_kernel(
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ S,
    float* __restrict__ z,
    int M,
    int d) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= d || col >= d) return;

  float s_acc = 0.0f;
  float z_acc = 0.0f;
  for (int m = 0; m < M; ++m) {
    float pk = phi_elu_plus_1(K[m * d + row]);
    s_acc = fmaf(pk, V[m * d + col], s_acc);
    if (col == 0) {
      z_acc += pk;
    }
  }
  S[row * d + col] = s_acc;
  if (col == 0) {
    z[row] = z_acc;
  }
}

__global__ void compute_output_baseline_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ S,
    const float* __restrict__ z,
    float* __restrict__ output,
    int M,
    int d) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= d) return;

  float num = 0.0f;
  float denom = 0.0f;
  const float* q_row = Q + row * d;
  for (int k = 0; k < d; ++k) {
    float pq = phi_elu_plus_1(q_row[k]);
    num = fmaf(pq, S[k * d + col], num);
    denom = fmaf(pq, z[k], denom);
  }
  output[row * d + col] = num / denom;
}

float* get_workspace() {
  static float* d_ws = nullptr;
  if (d_ws == nullptr) {
    // Max d = 128: S (128 * 128) + z (128)
    cudaMalloc(&d_ws, (128 * 128 + 128) * sizeof(float));
  }
  return d_ws;
}

}  // namespace

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {
  if (M <= 0 || d <= 0) return;

  float* d_ws = get_workspace();
  float* d_S = d_ws;
  float* d_z = d_ws + d * d;

  dim3 block(16, 16);
  dim3 grid_kv((d + 15) / 16, (d + 15) / 16);
  compute_kv_z_baseline_kernel<<<grid_kv, block>>>(K, V, d_S, d_z, M, d);

  dim3 grid_out((d + 15) / 16, (M + 15) / 16);
  compute_output_baseline_kernel<<<grid_out, block>>>(Q, d_S, d_z, output, M, d);
}
