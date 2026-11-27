#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr int kTileR = 64;
constexpr int kTileC = 64;
constexpr int kTileM = 16;
constexpr int kTileK = 16;

__device__ __forceinline__ float phi_elu_plus_1(float x) {
  return x > 0.0f ? (x + 1.0f) : __expf(x);
}

// Kernel 1: S = phi(K)^T @ V (d x d) and z = sum_m phi(K[m]) (d), split along M.
// Each block computes a 64x64 tile of S over a slice [m_start, m_end) of M.
__global__ void __launch_bounds__(256)
kv_z_splitm_opt_kernel(
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ S,
    float* __restrict__ z,
    int M,
    int d,
    int split_m) {
  const int c_base = blockIdx.x * kTileC;
  const int r_base = blockIdx.y * kTileR;
  const int split_id = blockIdx.z;

  const int chunk = (M + split_m - 1) / split_m;
  const int m_start = split_id * chunk;
  const int m_end = min(m_start + chunk, M);
  if (m_start >= m_end) return;

  const int tid = threadIdx.x;
  const int ty = tid >> 4;   // 0..15
  const int tx = tid & 15;   // 0..15

  __shared__ float sK[kTileM][kTileR];
  __shared__ float sV[kTileM][kTileC];

  float acc[4][4] = {{0.0f}};
  const bool compute_z = (blockIdx.x == 0);
  float z_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  const int load_m = tid >> 4;          // 0..15
  const int load_c4 = tid & 15;         // 0..15 (16 float4s = 64 floats)
  const int load_col = load_c4 * 4;
  const bool vec_k = ((d & 3) == 0) && (r_base + load_col + 3 < d);
  const bool vec_v = ((d & 3) == 0) && (c_base + load_col + 3 < d);

  for (int m_base = m_start; m_base < m_end; m_base += kTileM) {
    const int gm = m_base + load_m;
    if (gm < m_end) {
      if (vec_k) {
        float4 k4 = *reinterpret_cast<const float4*>(K + gm * d + r_base + load_col);
        k4.x = phi_elu_plus_1(k4.x);
        k4.y = phi_elu_plus_1(k4.y);
        k4.z = phi_elu_plus_1(k4.z);
        k4.w = phi_elu_plus_1(k4.w);
        *reinterpret_cast<float4*>(&sK[load_m][load_col]) = k4;
        if (compute_z) {
          z_acc[0] += k4.x;
          z_acc[1] += k4.y;
          z_acc[2] += k4.z;
          z_acc[3] += k4.w;
        }
      } else {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          int col = r_base + load_col + i;
          float val = (col < d) ? phi_elu_plus_1(K[gm * d + col]) : 0.0f;
          sK[load_m][load_col + i] = val;
          if (compute_z) z_acc[i] += val;
        }
      }

      if (vec_v) {
        float4 v4 = *reinterpret_cast<const float4*>(V + gm * d + c_base + load_col);
        *reinterpret_cast<float4*>(&sV[load_m][load_col]) = v4;
      } else {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          int col = c_base + load_col + i;
          sV[load_m][load_col + i] = (col < d) ? V[gm * d + col] : 0.0f;
        }
      }
    } else {
      *reinterpret_cast<float4*>(&sK[load_m][load_col]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      *reinterpret_cast<float4*>(&sV[load_m][load_col]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    __syncthreads();

    #pragma unroll
    for (int m = 0; m < kTileM; ++m) {
      float k_reg[4];
      float v_reg[4];
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        k_reg[i] = sK[m][ty + i * 16];
        v_reg[i] = sV[m][tx + i * 16];
      }
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          acc[i][j] = fmaf(k_reg[i], v_reg[j], acc[i][j]);
        }
      }
    }

    __syncthreads();
  }

  // Write S tile
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    int gr = r_base + ty + i * 16;
    if (gr < d) {
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        int gc = c_base + tx + j * 16;
        if (gc < d) {
          if (split_m == 1) {
            S[gr * d + gc] = acc[i][j];
          } else {
            atomicAdd(&S[gr * d + gc], acc[i][j]);
          }
        }
      }
    }
  }

  // Reduce z_acc across load_m = 0..15 and write z
  if (compute_z) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      sK[load_m][load_col + i] = z_acc[i];
    }
    __syncthreads();
    if (tid < kTileR) {
      int gr = r_base + tid;
      if (gr < d) {
        float sum = 0.0f;
        #pragma unroll
        for (int m = 0; m < kTileM; ++m) {
          sum += sK[m][tid];
        }
        if (split_m == 1) {
          z[gr] = sum;
        } else {
          atomicAdd(&z[gr], sum);
        }
      }
    }
  }
}

// Kernel 2: output = (phi(Q) @ S) / (phi(Q) @ z), fused in a single pass over Q.
// Each block computes a 64x64 tile of output.
__global__ void __launch_bounds__(256)
q_s_z_opt_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ S,
    const float* __restrict__ z,
    float* __restrict__ output,
    int M,
    int d) {
  const int n_base = blockIdx.x * kTileC;
  const int m_base = blockIdx.y * kTileR;
  const int tid = threadIdx.x;
  const int ty = tid >> 4;   // 0..15
  const int tx = tid & 15;   // 0..15

  __shared__ float sQ[kTileK][kTileR + 1];
  __shared__ float sS[kTileK][kTileC];
  __shared__ float sZ[128];
  __shared__ float sDenom[kTileR];

  if (tid < d) {
    sZ[tid] = z[tid];
  }
  if (tid < kTileR) {
    sDenom[tid] = 0.0f;
  }
  __syncthreads();

  float acc[4][4] = {{0.0f}};

  // Q loader: 64 rows x 16 cols = 256 float4s -> 1 float4 per thread
  const int q_load_m = tid >> 2;         // 0..63
  const int q_load_k4 = tid & 3;         // 0..3
  const int q_load_k = q_load_k4 * 4;
  const int gm_load = m_base + q_load_m;

  // S loader: 16 rows x 64 cols = 256 float4s -> 1 float4 per thread
  const int s_load_k = tid >> 4;         // 0..15
  const int s_load_n4 = tid & 15;        // 0..15
  const int s_load_n = s_load_n4 * 4;
  const bool vec_s = ((d & 3) == 0) && (n_base + s_load_n + 3 < d);

  for (int k_base = 0; k_base < d; k_base += kTileK) {
    // Load Q tile (transposed into sQ[k][m]) and accumulate denom = phi(Q) @ z
    float d_part = 0.0f;
    if (gm_load < M) {
      if (((d & 3) == 0) && (k_base + q_load_k + 3 < d)) {
        float4 q4 = *reinterpret_cast<const float4*>(Q + gm_load * d + k_base + q_load_k);
        q4.x = phi_elu_plus_1(q4.x);
        q4.y = phi_elu_plus_1(q4.y);
        q4.z = phi_elu_plus_1(q4.z);
        q4.w = phi_elu_plus_1(q4.w);
        sQ[q_load_k + 0][q_load_m] = q4.x;
        sQ[q_load_k + 1][q_load_m] = q4.y;
        sQ[q_load_k + 2][q_load_m] = q4.z;
        sQ[q_load_k + 3][q_load_m] = q4.w;
        d_part = fmaf(q4.x, sZ[k_base + q_load_k + 0], d_part);
        d_part = fmaf(q4.y, sZ[k_base + q_load_k + 1], d_part);
        d_part = fmaf(q4.z, sZ[k_base + q_load_k + 2], d_part);
        d_part = fmaf(q4.w, sZ[k_base + q_load_k + 3], d_part);
      } else {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          int gk = k_base + q_load_k + i;
          float val = (gk < d) ? phi_elu_plus_1(Q[gm_load * d + gk]) : 0.0f;
          sQ[q_load_k + i][q_load_m] = val;
          if (gk < d) {
            d_part = fmaf(val, sZ[gk], d_part);
          }
        }
      }
    } else {
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        sQ[q_load_k + i][q_load_m] = 0.0f;
      }
    }

    // Reduce d_part across the 4 threads sharing q_load_m
    d_part += __shfl_xor_sync(0xffffffff, d_part, 1);
    d_part += __shfl_xor_sync(0xffffffff, d_part, 2);
    if (q_load_k4 == 0) {
      sDenom[q_load_m] += d_part;
    }

    // Load S tile
    const int gk_s = k_base + s_load_k;
    if (gk_s < d) {
      if (vec_s) {
        float4 s4 = *reinterpret_cast<const float4*>(S + gk_s * d + n_base + s_load_n);
        *reinterpret_cast<float4*>(&sS[s_load_k][s_load_n]) = s4;
      } else {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          int gn = n_base + s_load_n + i;
          sS[s_load_k][s_load_n + i] = (gn < d) ? S[gk_s * d + gn] : 0.0f;
        }
      }
    } else {
      *reinterpret_cast<float4*>(&sS[s_load_k][s_load_n]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < kTileK; ++k) {
      float q_reg[4];
      float s_reg[4];
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        q_reg[i] = sQ[k][ty + i * 16];
        s_reg[i] = sS[k][tx + i * 16];
      }
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          acc[i][j] = fmaf(q_reg[i], s_reg[j], acc[i][j]);
        }
      }
    }

    __syncthreads();
  }

  // Write normalized output
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    int gm = m_base + ty + i * 16;
    if (gm < M) {
      float inv_denom = 1.0f / sDenom[ty + i * 16];
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        int gn = n_base + tx + j * 16;
        if (gn < d) {
          output[gm * d + gn] = acc[i][j] * inv_denom;
        }
      }
    }
  }
}

float* get_workspace() {
  static float* d_ws = nullptr;
  if (d_ws == nullptr) {
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

  const int tiles_c = (d + kTileC - 1) / kTileC;
  const int tiles_r = (d + kTileR - 1) / kTileR;
  int split_m = 1;
  if (M >= 512) {
    int target_blocks = 80;
    int base_blocks = tiles_c * tiles_r;
    split_m = (target_blocks + base_blocks - 1) / base_blocks;
    int max_splits = (M + 127) / 128;
    if (split_m > max_splits) split_m = max_splits;
    if (split_m < 1) split_m = 1;
  }

  if (split_m > 1) {
    cudaMemsetAsync(d_ws, 0, (static_cast<size_t>(d) * d + d) * sizeof(float));
  }

  dim3 grid_kv(tiles_c, tiles_r, split_m);
  kv_z_splitm_opt_kernel<<<grid_kv, 256>>>(K, V, d_S, d_z, M, d, split_m);

  dim3 grid_out(tiles_c, (M + kTileR - 1) / kTileR);
  q_s_z_opt_kernel<<<grid_out, 256>>>(Q, d_S, d_z, output, M, d);
}
