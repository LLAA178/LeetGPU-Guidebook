#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>

namespace {

// Static reusable device buffer for H = lora_scale * (x @ A^T) of shape [batch, rank]
float* g_hidden_buf = nullptr;
size_t g_hidden_cap = 0;

float* ensure_hidden_buffer(size_t elems) {
  if (elems > g_hidden_cap) {
    if (g_hidden_buf != nullptr) {
      cudaFree(g_hidden_buf);
      g_hidden_buf = nullptr;
    }
    cudaMalloc(&g_hidden_buf, elems * sizeof(float));
    g_hidden_cap = elems;
  }
  return g_hidden_buf;
}

// Kernel 1A (Fast Split-K=8 path): H[batch, rank] = lora_scale * (x[batch, d_in] @ A[rank, d_in]^T)
// Tile: 32x32 outputs per block (256 threads = 16x16, 2x2 per thread), BK = 32, Split-K = 8.
__global__ void lora_down_splitk8_kernel(const float* __restrict__ x,
                                         const float* __restrict__ A,
                                         float* __restrict__ hidden,
                                         int batch, int d_in, int rank,
                                         float lora_scale) {
  constexpr int BM = 32, BN = 32, BK = 32;
  __shared__ float sx[BM][BK + 1];
  __shared__ float sa[BN][BK + 1];

  int tx = threadIdx.x;  // 0..15 -> 2 cols
  int ty = threadIdx.y;  // 0..15 -> 2 rows
  int tid = ty * 16 + tx;
  int split = blockIdx.z;
  int k_per_split = d_in >> 3;
  int k_start = split * k_per_split;
  int k_end = k_start + k_per_split;

  float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;
  int r = tid >> 3;         // 0..31
  int c4 = (tid & 7) << 2;  // 0, 4, ..., 28

  for (int k0 = k_start; k0 < k_end; k0 += BK) {
    float4 vx = *reinterpret_cast<const float4*>(x + (blockIdx.y * BM + r) * d_in + k0 + c4);
    float4 va = *reinterpret_cast<const float4*>(A + (blockIdx.x * BN + r) * d_in + k0 + c4);
    sx[r][c4 + 0] = vx.x; sx[r][c4 + 1] = vx.y; sx[r][c4 + 2] = vx.z; sx[r][c4 + 3] = vx.w;
    sa[r][c4 + 0] = va.x; sa[r][c4 + 1] = va.y; sa[r][c4 + 2] = va.z; sa[r][c4 + 3] = va.w;
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      float x0 = sx[ty * 2 + 0][k];
      float x1 = sx[ty * 2 + 1][k];
      float a0 = sa[tx * 2 + 0][k];
      float a1 = sa[tx * 2 + 1][k];
      acc00 = fmaf(x0, a0, acc00);
      acc01 = fmaf(x0, a1, acc01);
      acc10 = fmaf(x1, a0, acc10);
      acc11 = fmaf(x1, a1, acc11);
    }
    __syncthreads();
  }

  int r0 = blockIdx.y * BM + ty * 2;
  int c0 = blockIdx.x * BN + tx * 2;
  atomicAdd(&hidden[(r0 + 0) * rank + (c0 + 0)], lora_scale * acc00);
  atomicAdd(&hidden[(r0 + 0) * rank + (c0 + 1)], lora_scale * acc01);
  atomicAdd(&hidden[(r0 + 1) * rank + (c0 + 0)], lora_scale * acc10);
  atomicAdd(&hidden[(r0 + 1) * rank + (c0 + 1)], lora_scale * acc11);
}

// Kernel 1B (General fallback): H[batch, rank] = lora_scale * (x[batch, d_in] @ A[rank, d_in]^T)
constexpr int kDownBM = 16;
constexpr int kDownBN = 16;
constexpr int kDownBK = 32;

__global__ void lora_down_opt_kernel(const float* __restrict__ x,
                                     const float* __restrict__ A,
                                     float* __restrict__ hidden,
                                     int batch, int d_in, int rank,
                                     float lora_scale) {
  __shared__ float smem_x[kDownBM][kDownBK + 1];
  __shared__ float smem_a[kDownBN][kDownBK + 1];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = ty * 16 + tx;
  int row = blockIdx.y * kDownBM + ty;
  int col = blockIdx.x * kDownBN + tx;

  float acc = 0.0f;
  const bool vec4_ok = ((d_in & 3) == 0);

  for (int k0 = 0; k0 < d_in; k0 += kDownBK) {
    if (vec4_ok && k0 + kDownBK <= d_in) {
      if (tid < 128) {
        int r = tid >> 3;
        int c4 = (tid & 7) << 2;
        int gr = blockIdx.y * kDownBM + r;
        float4 v = (gr < batch) ? *reinterpret_cast<const float4*>(x + gr * d_in + k0 + c4)
                                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        smem_x[r][c4 + 0] = v.x;
        smem_x[r][c4 + 1] = v.y;
        smem_x[r][c4 + 2] = v.z;
        smem_x[r][c4 + 3] = v.w;
      } else {
        int t = tid - 128;
        int r = t >> 3;
        int c4 = (t & 7) << 2;
        int gc = blockIdx.x * kDownBN + r;
        float4 v = (gc < rank) ? *reinterpret_cast<const float4*>(A + gc * d_in + k0 + c4)
                               : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        smem_a[r][c4 + 0] = v.x;
        smem_a[r][c4 + 1] = v.y;
        smem_a[r][c4 + 2] = v.z;
        smem_a[r][c4 + 3] = v.w;
      }
    } else {
      for (int i = tid; i < kDownBM * kDownBK; i += 256) {
        int r = i / kDownBK;
        int c = i % kDownBK;
        int gr = blockIdx.y * kDownBM + r;
        int gk = k0 + c;
        smem_x[r][c] = (gr < batch && gk < d_in) ? x[gr * d_in + gk] : 0.0f;
      }
      for (int i = tid; i < kDownBN * kDownBK; i += 256) {
        int r = i / kDownBK;
        int c = i % kDownBK;
        int gc = blockIdx.x * kDownBN + r;
        int gk = k0 + c;
        smem_a[r][c] = (gc < rank && gk < d_in) ? A[gc * d_in + gk] : 0.0f;
      }
    }

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < kDownBK; ++k) {
      acc = fmaf(smem_x[ty][k], smem_a[tx][k], acc);
    }

    __syncthreads();
  }

  if (row < batch && col < rank) {
    hidden[row * rank + col] = lora_scale * acc;
  }
}

#if __CUDA_ARCH__ >= 750 || !defined(__CUDA_ARCH__)
// --- SM 7.5+ 3-MMA FP16-split Tensor Core helpers ---
__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1,
                                            uint32_t& r2, uint32_t& r3,
                                            const void* smem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k8_f32_f16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t b0,
    float c0, float c1, float c2, float c3) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(b0), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}
__device__ __forceinline__ void store_split_f4(float4 v, half* hi_dst, half* lo_dst) {
  half h0 = __float2half_rn(v.x), h1 = __float2half_rn(v.y);
  half h2 = __float2half_rn(v.z), h3 = __float2half_rn(v.w);
  half l0 = __float2half_rn(v.x - __half2float(h0));
  half l1 = __float2half_rn(v.y - __half2float(h1));
  half l2 = __float2half_rn(v.z - __half2float(h2));
  half l3 = __float2half_rn(v.w - __half2float(h3));
  reinterpret_cast<half2*>(hi_dst)[0] = __halves2half2(h0, h1);
  reinterpret_cast<half2*>(hi_dst)[1] = __halves2half2(h2, h3);
  reinterpret_cast<half2*>(lo_dst)[0] = __halves2half2(l0, l1);
  reinterpret_cast<half2*>(lo_dst)[1] = __halves2half2(l2, l3);
}
#endif

// Kernel 2A (SM 7.5+ 3-MMA FP16-split Tensor Core fast path):
// BM=128, BN=64, BK=32, 256 threads (8 warps = 4x2, each warp computes 32x32 = 2x4 m16n8 tiles).
// Cuts W global memory traffic in half (2x reads = 128 MB for batch=256) with 30.7 KB smem (2 blocks/SM).
constexpr int kTcBM = 128;
constexpr int kTcBN = 64;
constexpr int kTcBK = 32;
constexpr int kTcStride = 40;  // 32 + 8 padding for bank-conflict-free ldmatrix

__global__ void __launch_bounds__(256, 2)
lora_linear_fused_tc_3mma_kernel(const float* __restrict__ x,
                                 const float* __restrict__ W,
                                 const float* __restrict__ hidden,
                                 const float* __restrict__ B,
                                 float* __restrict__ output,
                                 int batch, int d_in, int d_out, int rank) {
#if __CUDA_ARCH__ >= 750
  __shared__ half sA_hi[kTcBM][kTcStride];
  __shared__ half sA_lo[kTcBM][kTcStride];
  __shared__ half sB_hi[kTcBN][kTcStride];
  __shared__ half sB_lo[kTcBN][kTcStride];

  int tid = threadIdx.x;    // 0..255
  int warp_id = tid >> 5;   // 0..7
  int lane_id = tid & 31;   // 0..31
  int warp_m = warp_id >> 1;  // 0..3 (each 32 rows)
  int warp_n = warp_id & 1;   // 0..1 (each 32 cols)

  int block_row = blockIdx.y * kTcBM;
  int block_col = blockIdx.x * kTcBN;

  // 2x4 tiles of m16n8 = 32x32 outputs per warp, 4 FP32 regs per tile = 32 acc regs
  float acc[2][4][4];
  #pragma unroll
  for (int im = 0; im < 2; ++im)
    #pragma unroll
    for (int in = 0; in < 4; ++in)
      #pragma unroll
      for (int e = 0; e < 4; ++e)
        acc[im][in][e] = 0.0f;

  // Coalesced load coordinates: 256 threads load 128x32 X (4 passes) and 64x32 W (2 passes)
  int load_r = tid >> 3;         // 0..31
  int load_c = (tid & 7) << 2;   // 0, 4, ..., 28

  auto run_gemm_3mma = [&](const float* matA, const float* matB, int K) {
    for (int k0 = 0; k0 < K; k0 += kTcBK) {
      #pragma unroll
      for (int p = 0; p < 4; ++p) {
        int r = load_r + p * 32;
        float4 vx = *reinterpret_cast<const float4*>(matA + (block_row + r) * K + k0 + load_c);
        store_split_f4(vx, &sA_hi[r][load_c], &sA_lo[r][load_c]);
      }
      #pragma unroll
      for (int p = 0; p < 2; ++p) {
        int r = load_r + p * 32;
        float4 vw = *reinterpret_cast<const float4*>(matB + (block_col + r) * K + k0 + load_c);
        store_split_f4(vw, &sB_hi[r][load_c], &sB_lo[r][load_c]);
      }
      __syncthreads();

      #pragma unroll
      for (int kk = 0; kk < kTcBK; kk += 8) {
        uint32_t a_hi[4], a_lo[4], b_hi[4], b_lo[4];
        ldmatrix_x4(a_hi[0], a_hi[1], a_hi[2], a_hi[3], &sA_hi[warp_m * 32 + lane_id][kk]);
        ldmatrix_x4(a_lo[0], a_lo[1], a_lo[2], a_lo[3], &sA_lo[warp_m * 32 + lane_id][kk]);
        ldmatrix_x4(b_hi[0], b_hi[1], b_hi[2], b_hi[3], &sB_hi[warp_n * 32 + lane_id][kk]);
        ldmatrix_x4(b_lo[0], b_lo[1], b_lo[2], b_lo[3], &sB_lo[warp_n * 32 + lane_id][kk]);

        #pragma unroll
        for (int im = 0; im < 2; ++im) {
          uint32_t ah0 = a_hi[im * 2 + 0], ah1 = a_hi[im * 2 + 1];
          uint32_t al0 = a_lo[im * 2 + 0], al1 = a_lo[im * 2 + 1];
          #pragma unroll
          for (int in = 0; in < 4; ++in) {
            float* c = acc[im][in];
            mma_m16n8k8_f32_f16(c[0], c[1], c[2], c[3], ah0, ah1, b_hi[in], c[0], c[1], c[2], c[3]);
            mma_m16n8k8_f32_f16(c[0], c[1], c[2], c[3], ah0, ah1, b_lo[in], c[0], c[1], c[2], c[3]);
            mma_m16n8k8_f32_f16(c[0], c[1], c[2], c[3], al0, al1, b_hi[in], c[0], c[1], c[2], c[3]);
          }
        }
      }
      __syncthreads();
    }
  };

  // Phase 1: Base linear x @ W^T
  run_gemm_3mma(x, W, d_in);
  // Phase 2: Fused LoRA up-projection H @ B^T
  run_gemm_3mma(hidden, B, rank);

  // Epilogue: Store m16n8 outputs
  int group_id = lane_id >> 2;
  int tid_in_group = lane_id & 3;
  #pragma unroll
  for (int im = 0; im < 2; ++im) {
    int r0 = block_row + warp_m * 32 + im * 16 + group_id;
    int r1 = r0 + 8;
    #pragma unroll
    for (int in = 0; in < 4; ++in) {
      int c0 = block_col + warp_n * 32 + in * 8 + tid_in_group * 2;
      *reinterpret_cast<float2*>(output + r0 * d_out + c0) = make_float2(acc[im][in][0], acc[im][in][1]);
      *reinterpret_cast<float2*>(output + r1 * d_out + c0) = make_float2(acc[im][in][2], acc[im][in][3]);
    }
  }
#endif
}

// Kernel 2B (General fallback for arbitrary/unaligned dimensions): 128x128 FP32 SGEMM
constexpr int kBM = 128;
constexpr int kBN = 128;
constexpr int kBK = 8;

__device__ __forceinline__ float4 fetch_tile_128x8(
    const float* __restrict__ mat, int rows, int cols,
    int row_base, int k_base, int tid, bool vec4_ok) {
  int r = tid >> 1;
  int c = (tid & 1) << 2;
  int gr = row_base + r;
  if (vec4_ok && gr < rows && k_base + kBK <= cols) {
    return *reinterpret_cast<const float4*>(mat + gr * cols + k_base + c);
  }
  float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  if (gr < rows) {
    if (k_base + c + 0 < cols) v.x = mat[gr * cols + k_base + c + 0];
    if (k_base + c + 1 < cols) v.y = mat[gr * cols + k_base + c + 1];
    if (k_base + c + 2 < cols) v.z = mat[gr * cols + k_base + c + 2];
    if (k_base + c + 3 < cols) v.w = mat[gr * cols + k_base + c + 3];
  }
  return v;
}

__device__ __forceinline__ void store_tile_128x8(
    float4 v, float smem[kBK][kBM + 4], int tid) {
  int r = tid >> 1;
  int c = (tid & 1) << 2;
  smem[c + 0][r] = v.x;
  smem[c + 1][r] = v.y;
  smem[c + 2][r] = v.z;
  smem[c + 3][r] = v.w;
}

__global__ void __launch_bounds__(256, 2)
lora_linear_fused_128x128_generic_kernel(const float* __restrict__ x,
                                         const float* __restrict__ W,
                                         const float* __restrict__ hidden,
                                         const float* __restrict__ B,
                                         float* __restrict__ output,
                                         int batch, int d_in, int d_out, int rank) {
  __shared__ float As[2][kBK][kBM + 4];
  __shared__ float Bs[2][kBK][kBN + 4];

  int tid = threadIdx.y * 16 + threadIdx.x;
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int block_row = blockIdx.y * kBM;
  int block_col = blockIdx.x * kBN;

  float acc[8][8];
  #pragma unroll
  for (int i = 0; i < 8; ++i)
    #pragma unroll
    for (int j = 0; j < 8; ++j)
      acc[i][j] = 0.0f;

  float ra[8], rb[8];

  const bool vec4_in = ((d_in & 3) == 0);
  int num_k_tiles = (d_in + kBK - 1) / kBK;
  if (num_k_tiles > 0) {
    float4 ldg_a = fetch_tile_128x8(x, batch, d_in, block_row, 0, tid, vec4_in);
    float4 ldg_b = fetch_tile_128x8(W, d_out, d_in, block_col, 0, tid, vec4_in);
    store_tile_128x8(ldg_a, As[0], tid);
    store_tile_128x8(ldg_b, Bs[0], tid);
    __syncthreads();

    for (int kt = 0; kt < num_k_tiles; ++kt) {
      int cur = kt & 1;
      int nxt = cur ^ 1;
      if (kt + 1 < num_k_tiles) {
        int next_k = (kt + 1) * kBK;
        ldg_a = fetch_tile_128x8(x, batch, d_in, block_row, next_k, tid, vec4_in);
        ldg_b = fetch_tile_128x8(W, d_out, d_in, block_col, next_k, tid, vec4_in);
      }
      #pragma unroll
      for (int k = 0; k < kBK; ++k) {
        *reinterpret_cast<float4*>(&ra[0]) = *reinterpret_cast<const float4*>(&As[cur][k][ty * 4]);
        *reinterpret_cast<float4*>(&ra[4]) = *reinterpret_cast<const float4*>(&As[cur][k][ty * 4 + 64]);
        *reinterpret_cast<float4*>(&rb[0]) = *reinterpret_cast<const float4*>(&Bs[cur][k][tx * 4]);
        *reinterpret_cast<float4*>(&rb[4]) = *reinterpret_cast<const float4*>(&Bs[cur][k][tx * 4 + 64]);
        #pragma unroll
        for (int i = 0; i < 8; ++i)
          #pragma unroll
          for (int j = 0; j < 8; ++j)
            acc[i][j] = fmaf(ra[i], rb[j], acc[i][j]);
      }
      if (kt + 1 < num_k_tiles) {
        store_tile_128x8(ldg_a, As[nxt], tid);
        store_tile_128x8(ldg_b, Bs[nxt], tid);
      }
      __syncthreads();
    }
  }

  const bool vec4_rank = ((rank & 3) == 0);
  int num_r_tiles = (rank + kBK - 1) / kBK;
  if (num_r_tiles > 0) {
    float4 ldg_a = fetch_tile_128x8(hidden, batch, rank, block_row, 0, tid, vec4_rank);
    float4 ldg_b = fetch_tile_128x8(B, d_out, rank, block_col, 0, tid, vec4_rank);
    store_tile_128x8(ldg_a, As[0], tid);
    store_tile_128x8(ldg_b, Bs[0], tid);
    __syncthreads();

    for (int rt = 0; rt < num_r_tiles; ++rt) {
      int cur = rt & 1;
      int nxt = cur ^ 1;
      if (rt + 1 < num_r_tiles) {
        int next_r = (rt + 1) * kBK;
        ldg_a = fetch_tile_128x8(hidden, batch, rank, block_row, next_r, tid, vec4_rank);
        ldg_b = fetch_tile_128x8(B, d_out, rank, block_col, next_r, tid, vec4_rank);
      }
      #pragma unroll
      for (int k = 0; k < kBK; ++k) {
        *reinterpret_cast<float4*>(&ra[0]) = *reinterpret_cast<const float4*>(&As[cur][k][ty * 4]);
        *reinterpret_cast<float4*>(&ra[4]) = *reinterpret_cast<const float4*>(&As[cur][k][ty * 4 + 64]);
        *reinterpret_cast<float4*>(&rb[0]) = *reinterpret_cast<const float4*>(&Bs[cur][k][tx * 4]);
        *reinterpret_cast<float4*>(&rb[4]) = *reinterpret_cast<const float4*>(&Bs[cur][k][tx * 4 + 64]);
        #pragma unroll
        for (int i = 0; i < 8; ++i)
          #pragma unroll
          for (int j = 0; j < 8; ++j)
            acc[i][j] = fmaf(ra[i], rb[j], acc[i][j]);
      }
      if (rt + 1 < num_r_tiles) {
        store_tile_128x8(ldg_a, As[nxt], tid);
        store_tile_128x8(ldg_b, Bs[nxt], tid);
      }
      __syncthreads();
    }
  }

  const bool vec4_out = ((d_out & 3) == 0);
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    int r0 = block_row + ty * 4 + i;
    int r1 = block_row + ty * 4 + 64 + i;
    int c0 = block_col + tx * 4;
    int c1 = block_col + tx * 4 + 64;
    if (r0 < batch) {
      if (vec4_out && c0 + 4 <= d_out) {
        *reinterpret_cast<float4*>(output + r0 * d_out + c0) =
            make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
      } else {
        for (int j = 0; j < 4; ++j)
          if (c0 + j < d_out) output[r0 * d_out + c0 + j] = acc[i][j];
      }
      if (vec4_out && c1 + 4 <= d_out) {
        *reinterpret_cast<float4*>(output + r0 * d_out + c1) =
            make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
      } else {
        for (int j = 0; j < 4; ++j)
          if (c1 + j < d_out) output[r0 * d_out + c1 + j] = acc[i][4 + j];
      }
    }
    if (r1 < batch) {
      if (vec4_out && c0 + 4 <= d_out) {
        *reinterpret_cast<float4*>(output + r1 * d_out + c0) =
            make_float4(acc[i + 4][0], acc[i + 4][1], acc[i + 4][2], acc[i + 4][3]);
      } else {
        for (int j = 0; j < 4; ++j)
          if (c0 + j < d_out) output[r1 * d_out + c0 + j] = acc[i + 4][j];
      }
      if (vec4_out && c1 + 4 <= d_out) {
        *reinterpret_cast<float4*>(output + r1 * d_out + c1) =
            make_float4(acc[i + 4][4], acc[i + 4][5], acc[i + 4][6], acc[i + 4][7]);
      } else {
        for (int j = 0; j < 4; ++j)
          if (c1 + j < d_out) output[r1 * d_out + c1 + j] = acc[i + 4][4 + j];
      }
    }
  }
}

}  // namespace

// x, W, A, B, output are device pointers
extern "C" void solve(const float* x, const float* W, const float* A, const float* B,
                      float* output, int batch, int d_in, int d_out, int rank,
                      float lora_scale) {
  float* hidden = ensure_hidden_buffer(static_cast<size_t>(batch) * rank);

  if ((batch % 32 == 0) && (rank % 32 == 0) && (d_in % 256 == 0)) {
    cudaMemsetAsync(hidden, 0, static_cast<size_t>(batch) * rank * sizeof(float));
    dim3 down_block(16, 16);
    dim3 down_grid(rank / 32, batch / 32, 8);
    lora_down_splitk8_kernel<<<down_grid, down_block>>>(x, A, hidden, batch, d_in, rank, lora_scale);
  } else {
    dim3 down_block(16, 16);
    dim3 down_grid((rank + kDownBN - 1) / kDownBN, (batch + kDownBM - 1) / kDownBM);
    lora_down_opt_kernel<<<down_grid, down_block>>>(x, A, hidden, batch, d_in, rank, lora_scale);
  }

  if ((batch % kTcBM == 0) && (d_out % kTcBN == 0) &&
      (d_in % kTcBK == 0) && (rank % kTcBK == 0)) {
    dim3 tc_grid(d_out / kTcBN, batch / kTcBM);
    lora_linear_fused_tc_3mma_kernel<<<tc_grid, 256>>>(
        x, W, hidden, B, output, batch, d_in, d_out, rank);
  } else {
    dim3 fused_block(16, 16);
    dim3 fused_grid((d_out + kBN - 1) / kBN, (batch + kBM - 1) / kBM);
    lora_linear_fused_128x128_generic_kernel<<<fused_grid, fused_block>>>(
        x, W, hidden, B, output, batch, d_in, d_out, rank);
  }
}
