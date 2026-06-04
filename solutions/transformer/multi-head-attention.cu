#include <cuda_runtime.h>

#include <math.h>

#ifndef ONLINE_JUDGE
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#endif

extern "C" void solve_base(const float* Q,
                           const float* K,
                           const float* V,
                           float* output,
                           int N,
                           int d_model,
                           int h);

namespace {

// 基础 kernel 使用 256 线程做一整行 attention。
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpCount = kThreads / kWarpSize;
constexpr float kNegInf = -INFINITY;

// FlashAttention 风格小头专门化路径的参数：
// - kFlashTile 固定为 32，对齐一个 warp 的 lane 数
// - 窄头（<=64）时一个 block 放 8 个 warp，尽量提高并发
// - 宽头（128）时一个 block 放 4 个 warp，给每个 warp 更多共享内存/寄存器余量
constexpr int kFlashThreads = 256;
constexpr int kFlashTile = 32;
constexpr int kFlashWarpsPerBlockNarrow = 8;
constexpr int kFlashWarpsPerBlockWide = 4;

// 小头专门化路径最多支持 4 个 32-wide tile，也就是 head_size <= 128。
constexpr int kFlashMaxDkTiles = 4;
constexpr int kFlashSmallMaxHeadSize = kFlashTile * kFlashMaxDkTiles;

// warp all-reduce：要求 warp 内所有 lane 都拿到同一个和。
// 这里用 xor-shuffle，而不是 down-shuffle。原因是 online softmax 的每一步都要求每个 lane
// 都拿到同样的 max/sum，不能只让 lane0 正确。
__device__ __forceinline__ float warp_allreduce_sum(float value) {
  #pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

// warp 内最大值 all-reduce，保证每个 lane 都能拿到同一个最大值。
__device__ __forceinline__ float warp_allreduce_max(float value) {
  #pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
  }
  return value;
}

// 下面两组 down-reduce 只保证 lane0 最终拿到正确结果。
// 这类 reduce 适合 block 级二次归约，不适合直接做 online softmax。
__device__ __forceinline__ float warp_reduce_max(float value) {
  #pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
  #pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

// block 级 max 归约：先每个 warp 局部归约，再让 warp0 做第二次归约。
__device__ __forceinline__ float block_reduce_max(float value) {
  __shared__ float warp_values[kWarpCount];

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;

  value = warp_reduce_max(value);
  if (lane == 0) {
    warp_values[warp_id] = value;
  }
  __syncthreads();

  float block_value = kNegInf;
  if (warp_id == 0) {
    block_value = lane < kWarpCount ? warp_values[lane] : kNegInf;
    block_value = warp_reduce_max(block_value);
    if (lane == 0) {
      warp_values[0] = block_value;
    }
  }
  __syncthreads();
  return warp_values[0];
}

// block 级 sum 归约，结构和 block_reduce_max 对称。
__device__ __forceinline__ float block_reduce_sum(float value) {
  __shared__ float warp_values[kWarpCount];

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;

  value = warp_reduce_sum(value);
  if (lane == 0) {
    warp_values[warp_id] = value;
  }
  __syncthreads();

  float block_value = 0.0f;
  if (warp_id == 0) {
    block_value = lane < kWarpCount ? warp_values[lane] : 0.0f;
    block_value = warp_reduce_sum(block_value);
    if (lane == 0) {
      warp_values[0] = block_value;
    }
  }
  __syncthreads();
  return warp_values[0];
}

template <int kWarpRows>
__global__ void flash_attn_small_kernel(const float* __restrict__ Q,
                                        const float* __restrict__ K,
                                        const float* __restrict__ V,
                                        float* __restrict__ output,
                                        int N,
                                        int d_model,
                                        int head_size,
                                        float scale) {
  // 一个 warp 处理一行 Q；tx 是 warp 内 lane，ty 是 block 内第几个 warp。
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // blockIdx.x 负责 Q 的行块，blockIdx.y 负责 head。
  const int q_row = blockIdx.x * kWarpRows + ty;
  const int head_offset = blockIdx.y * head_size;

  // d_k/head_size 按 32 列切 tile，N 按 32 个 key/value 切 tile。
  const int d_k_tiles = (head_size + kFlashTile - 1) / kFlashTile;
  const int num_kv_blocks = (N + kFlashTile - 1) / kFlashTile;

  // sQ: 每个 warp 自己的 Q tile
  // sK: 当前 K/V tile，所有 warp 复用
  // sP: 当前 32 个 key 的 softmax 权重（每个 warp 各一行）
  __shared__ float sQ[kWarpRows][kFlashTile + 1];
  __shared__ float sK[kFlashTile][kFlashTile + 1];
  __shared__ float sP[kWarpRows][kFlashTile + 1];

  // output_local[tile] 保存当前线程负责的输出列，在所有 kv block 上做在线累加。
  float output_local[kFlashMaxDkTiles];
  #pragma unroll
  for (int tile = 0; tile < kFlashMaxDkTiles; ++tile) {
    output_local[tile] = 0.0f;
  }

  // online softmax 的两个核心状态：
  // - running_max: 到目前为止所有 score 的最大值
  // - running_sum: 在 running_max 基准下累计的 exp 和
  float running_max = -INFINITY;
  float running_sum = 0.0f;

  // 外层循环逐块扫描 K/V。每次处理 32 个 key/value。
  for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
    const int kv_base = kv_block * kFlashTile;

    // 当前 lane 负责当前 kv tile 里的一个 key 列（kv_base + tx）的 score。
    float score = 0.0f;

    // 先在 d_k 维度上做 tile 化的 Q·K。
    #pragma unroll
    for (int tile = 0; tile < kFlashMaxDkTiles; ++tile) {
      if (tile >= d_k_tiles) {
        break;
      }

      const int dk_col = tile * kFlashTile + tx;

      // 每个 warp 把自己对应 Q 行的一个 32-wide 子块搬到 shared memory。
      sQ[ty][tx] = (q_row < N && dk_col < head_size)
          ? Q[q_row * d_model + head_offset + dk_col]
          : 0.0f;

      // K tile 由 block 内多个 warp 协作装载。
      #pragma unroll
      for (int row = ty; row < kFlashTile; row += kWarpRows) {
        const int kv_row = kv_base + row;
        sK[row][tx] = (kv_row < N && dk_col < head_size)
            ? K[kv_row * d_model + head_offset + dk_col]
            : 0.0f;
      }
      __syncthreads();

      // 此时：
      // - sQ[ty][k] 是当前 Q 行在第 k 列的元素
      // - sK[tx][k] 是当前 lane 对应 key 行在第 k 列的元素
      // 所以这 32 次乘加正好得到一个 score 的一个 d_k 子块贡献。
      #pragma unroll
      for (int k = 0; k < kFlashTile; ++k) {
        score += sQ[ty][k] * sK[tx][k];
      }
      __syncthreads();
    }

    // 越界的 key 直接视为 -inf，保证 softmax 后权重为 0。
    const int kv_col = kv_base + tx;
    score = (q_row < N && kv_col < N) ? score * scale : -INFINITY;

    // online softmax：
    // 1. warp 内先求出当前 32 个 score 的最大值和 exp 和
    // 2. 再把历史 running_max/running_sum 合并进来
    const float block_max = warp_allreduce_max(score);
    const float next_max = fmaxf(running_max, block_max);
    const float alpha = expf(running_max - next_max);
    const float prob = expf(score - next_max);
    const float block_sum = warp_allreduce_sum(prob);

    // 由于 softmax 的归一化基准从 running_max 变成了 next_max，
    // 已累计的输出也要统一乘上 alpha 做重标定。
    #pragma unroll
    for (int tile = 0; tile < kFlashMaxDkTiles; ++tile) {
      output_local[tile] *= alpha;
    }
    running_sum = alpha * running_sum + block_sum;
    running_max = next_max;

    // 把当前 32 个 key 的权重缓存下来，后面直接乘 V tile。
    sP[ty][tx] = prob;
    __syncthreads();

    // 用当前 tile 的 softmax 权重做 P @ V。
    #pragma unroll
    for (int tile = 0; tile < kFlashMaxDkTiles; ++tile) {
      if (tile >= d_k_tiles) {
        break;
      }

      const int dk_col = tile * kFlashTile + tx;

      // 复用 sK 这块 shared memory，当作 V tile 缓冲区。
      #pragma unroll
      for (int row = ty; row < kFlashTile; row += kWarpRows) {
        const int kv_row = kv_base + row;
        sK[row][tx] = (kv_row < N && dk_col < head_size)
            ? V[kv_row * d_model + head_offset + dk_col]
            : 0.0f;
      }
      __syncthreads();

      // 当前线程负责输出列 dk_col，把 32 个 key 的贡献累加起来。
      float acc = 0.0f;
      #pragma unroll
      for (int jj = 0; jj < kFlashTile; ++jj) {
        acc += sP[ty][jj] * sK[jj][tx];
      }
      output_local[tile] += acc;
      __syncthreads();
    }
  }

  // 所有 kv block 扫完后，最后用 running_sum 做一次归一化再写回。
  if (q_row < N) {
    const float inv_running_sum = 1.0f / running_sum;
    #pragma unroll
    for (int tile = 0; tile < kFlashMaxDkTiles; ++tile) {
      if (tile >= d_k_tiles) {
        break;
      }
      const int dk_col = tile * kFlashTile + tx;
      if (dk_col < head_size) {
        output[q_row * d_model + head_offset + dk_col] = output_local[tile] * inv_running_sum;
      }
    }
  }
}

__global__ void flash_attn_generic_kernel(const float* __restrict__ Q,
                                          const float* __restrict__ K,
                                          const float* __restrict__ V,
                                          float* __restrict__ output,
                                          int N,
                                          int d_model,
                                          int head_size) {
  // generic 路径按“一个 block 处理一个 (q_row, head)”组织，
  // 优先保证覆盖任意 head_size，而不是追求极致性能。
  const int q_row = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int tid = threadIdx.x;

  if (q_row >= N) {
    return;
  }

  extern __shared__ float shared_mem[];
  float* scores = shared_mem;
  float* q_vec = scores + N;
  __shared__ float score_sum_shared;

  const int head_offset = head_idx * head_size;
  const int q_base = q_row * d_model + head_offset;

  for (int dim = tid; dim < head_size; dim += blockDim.x) {
    q_vec[dim] = Q[q_base + dim];
  }
  __syncthreads();

  // 第一步：并行计算当前 query 对所有 key 的 score。
  const float scale = rsqrtf(static_cast<float>(head_size));
  for (int k_row = tid; k_row < N; k_row += blockDim.x) {
    const float* k_ptr = K + k_row * d_model + head_offset;
    float dot = 0.0f;
    for (int dim = 0; dim < head_size; ++dim) {
      dot += q_vec[dim] * k_ptr[dim];
    }
    scores[k_row] = dot * scale;
  }
  __syncthreads();

  // 第二步：用 tid==0 串行做 softmax 的 max 和 sum。
  // 这个版本的目的主要是覆盖大 head_size 场景，所以实现上刻意保守。
  if (tid == 0) {
    float score_max = kNegInf;
    for (int k_row = 0; k_row < N; ++k_row) {
      score_max = fmaxf(score_max, scores[k_row]);
    }

    float score_sum = 0.0f;
    for (int k_row = 0; k_row < N; ++k_row) {
      scores[k_row] = expf(scores[k_row] - score_max);
      score_sum += scores[k_row];
    }

    score_sum_shared = score_sum;
  }
  __syncthreads();

  // 第三步：各线程并行负责一部分输出维度，做权重和 V 的乘加。
  const float score_sum = score_sum_shared;
  for (int dim = tid; dim < head_size; dim += blockDim.x) {
    float result = 0.0f;
    for (int k_row = 0; k_row < N; ++k_row) {
      const float weight = scores[k_row] / score_sum;
      result += weight * V[k_row * d_model + head_offset + dim];
    }
    output[q_row * d_model + head_offset + dim] = result;
  }
}
__global__ void mha_kernel(const float* __restrict__ Q,
                           const float* __restrict__ K,
                           const float* __restrict__ V,
                           float* __restrict__ output,
                           int N,
                           int d_model,
                           int h) {
  const int head_size = d_model / h;
  const int q_row = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int tid = threadIdx.x;

  if (q_row >= N || head_idx >= h) {
    return;
  }

  extern __shared__ float shared_mem[];
  float* scores = shared_mem;
  float* q_vec = scores + N;

  const int head_offset = head_idx * head_size;
  const int q_base = q_row * d_model + head_offset;

  for (int dim = tid; dim < head_size; dim += blockDim.x) {
    q_vec[dim] = Q[q_base + dim];
  }
  __syncthreads();

  const float scale = rsqrtf(static_cast<float>(head_size));
  for (int k_row = tid; k_row < N; k_row += blockDim.x) {
    const float* k_ptr = K + k_row * d_model + head_offset;
    float dot = 0.0f;
    for (int dim = 0; dim < head_size; ++dim) {
      dot += q_vec[dim] * k_ptr[dim];
    }
    scores[k_row] = dot * scale;
  }
  __syncthreads();

  float local_max = kNegInf;
  for (int k_row = tid; k_row < N; k_row += blockDim.x) {
    local_max = fmaxf(local_max, scores[k_row]);
  }
  const float score_max = block_reduce_max(local_max);

  float local_sum = 0.0f;
  for (int k_row = tid; k_row < N; k_row += blockDim.x) {
    const float exp_score = expf(scores[k_row] - score_max);
    scores[k_row] = exp_score;
    local_sum += exp_score;
  }
  const float score_sum = block_reduce_sum(local_sum);

  for (int dim = tid; dim < head_size; dim += blockDim.x) {
    float result = 0.0f;
    for (int k_row = 0; k_row < N; ++k_row) {
      const float weight = scores[k_row] / score_sum;
      result += weight * V[k_row * d_model + head_offset + dim];
    }
    output[q_row * d_model + head_offset + dim] = result;
  }
}

}  // namespace
extern "C" void solve_flash_attn(const float* Q,
                                 const float* K,
                                 const float* V,
                                 float* output,
                                 int N,
                                 int d_model,
                                 int h) {
  if (Q == nullptr || K == nullptr || V == nullptr || output == nullptr) {
    return;
  }
  if (N <= 0 || d_model <= 0 || h <= 0 || (d_model % h) != 0) {
    return;
  }
  const int head_size = d_model / h;

  // 调度策略：
  // - head_size <= 128: 走 warp-reduce + online softmax 的专门化 kernel
  // - head_size  > 128: 走 generic kernel，保证题目全范围都能覆盖
  if (head_size <= kFlashSmallMaxHeadSize) {
    const float scale = rsqrtf(static_cast<float>(head_size));

    // 128 宽头比 64/32 宽头更吃共享内存与寄存器，所以 block 内 warp 数减半。
    if (head_size > 64) {
      dim3 block(kFlashTile, kFlashWarpsPerBlockWide);
      dim3 grid((N + kFlashWarpsPerBlockWide - 1) / kFlashWarpsPerBlockWide, h);
      flash_attn_small_kernel<kFlashWarpsPerBlockWide>
          <<<grid, block>>>(Q, K, V, output, N, d_model, head_size, scale);
    } else {
      dim3 block(kFlashTile, kFlashWarpsPerBlockNarrow);
      dim3 grid((N + kFlashWarpsPerBlockNarrow - 1) / kFlashWarpsPerBlockNarrow, h);
      flash_attn_small_kernel<kFlashWarpsPerBlockNarrow>
          <<<grid, block>>>(Q, K, V, output, N, d_model, head_size, scale);
    }
    return;
  }

  const size_t shared_mem_size = static_cast<size_t>(N + head_size) * sizeof(float);
  dim3 grid(N, h);
  flash_attn_generic_kernel<<<grid, kFlashThreads, shared_mem_size>>>(Q, K, V, output, N, d_model, head_size);
}

extern "C" void solve_base(const float* Q,
                           const float* K,
                           const float* V,
                           float* output,
                           int N,
                           int d_model,
                           int h) {
  if (Q == nullptr || K == nullptr || V == nullptr || output == nullptr) {
    return;
  }
  if (N <= 0 || d_model <= 0 || h <= 0 || (d_model % h) != 0) {
    return;
  }

  const int head_size = d_model / h;
  const size_t shared_mem_size = static_cast<size_t>(N + head_size) * sizeof(float);
  dim3 grid(N, h);
  mha_kernel<<<grid, kThreads, shared_mem_size>>>(Q, K, V, output, N, d_model, h);
}

extern "C" void solve(const float* Q,
                      const float* K,
                      const float* V,
                      float* output,
                      int N,
                      int d_model,
                      int h) {
  // 当前对外默认统一走 FA 路径；solve_base 仅保留作对拍和基线性能比较。
  solve_flash_attn(Q, K, V, output, N, d_model, h);
}

#ifndef ONLINE_JUDGE
namespace {

constexpr float kTolerance = 2e-4f;

std::vector<float> mha_reference(const std::vector<float>& q,
                                 const std::vector<float>& k,
                                 const std::vector<float>& v,
                                 int N,
                                 int d_model,
                                 int h) {
  const int head_size = d_model / h;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));

  std::vector<float> output(static_cast<size_t>(N) * d_model, 0.0f);
  std::vector<float> scores(N, 0.0f);

  for (int q_row = 0; q_row < N; ++q_row) {
    for (int head = 0; head < h; ++head) {
      const int head_offset = head * head_size;

      float max_score = -INFINITY;
      for (int k_row = 0; k_row < N; ++k_row) {
        float dot = 0.0f;
        for (int dim = 0; dim < head_size; ++dim) {
          dot += q[q_row * d_model + head_offset + dim] *
                 k[k_row * d_model + head_offset + dim];
        }
        scores[k_row] = dot * scale;
        max_score = std::max(max_score, scores[k_row]);
      }

      float denom = 0.0f;
      for (int k_row = 0; k_row < N; ++k_row) {
        scores[k_row] = std::exp(scores[k_row] - max_score);
        denom += scores[k_row];
      }

      for (int dim = 0; dim < head_size; ++dim) {
        float acc = 0.0f;
        for (int k_row = 0; k_row < N; ++k_row) {
          const float weight = scores[k_row] / denom;
          acc += weight * v[k_row * d_model + head_offset + dim];
        }
        output[q_row * d_model + head_offset + dim] = acc;
      }
    }
  }

  return output;
}

bool check_cuda(cudaError_t status, const std::string& step, const std::string& context) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << "[FAIL] " << context << ": " << step
            << " failed: " << cudaGetErrorString(status) << '\n';
  return false;
}

std::vector<float> make_random_vector(size_t count, int seed, float low, float high) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(low, high);
  std::vector<float> values(count);
  for (float& value : values) {
    value = dist(rng);
  }
  return values;
}

bool run_solver_on_gpu(const std::string& case_name,
                       bool use_base_solver,
                       const std::vector<float>& q,
                       const std::vector<float>& k,
                       const std::vector<float>& v,
                       int N,
                       int d_model,
                       int h,
                       std::vector<float>* output_host) {
  const char* solver_name = use_base_solver ? "solve_base" : "solve";
  if (output_host == nullptr) {
    std::cerr << "[FAIL] " << case_name << ": null output buffer for " << solver_name << '\n';
    return false;
  }

  const size_t count = static_cast<size_t>(N) * d_model;
  const size_t bytes = count * sizeof(float);
  output_host->assign(count, 0.0f);

  float* d_q = nullptr;
  float* d_k = nullptr;
  float* d_v = nullptr;
  float* d_output = nullptr;

  auto cleanup = [&]() {
    if (d_q != nullptr) cudaFree(d_q);
    if (d_k != nullptr) cudaFree(d_k);
    if (d_v != nullptr) cudaFree(d_v);
    if (d_output != nullptr) cudaFree(d_output);
  };

  if (!check_cuda(cudaMalloc(&d_q, bytes), "cudaMalloc(Q)", case_name) ||
      !check_cuda(cudaMalloc(&d_k, bytes), "cudaMalloc(K)", case_name) ||
      !check_cuda(cudaMalloc(&d_v, bytes), "cudaMalloc(V)", case_name) ||
      !check_cuda(cudaMalloc(&d_output, bytes), "cudaMalloc(output)", case_name)) {
    cleanup();
    return false;
  }

  if (!check_cuda(cudaMemcpy(d_q, q.data(), bytes, cudaMemcpyHostToDevice), "H2D Q", case_name) ||
      !check_cuda(cudaMemcpy(d_k, k.data(), bytes, cudaMemcpyHostToDevice), "H2D K", case_name) ||
      !check_cuda(cudaMemcpy(d_v, v.data(), bytes, cudaMemcpyHostToDevice), "H2D V", case_name)) {
    cleanup();
    return false;
  }

  if (use_base_solver) {
    solve_base(d_q, d_k, d_v, d_output, N, d_model, h);
  } else {
    solve(d_q, d_k, d_v, d_output, N, d_model, h);
  }
  if (!check_cuda(cudaGetLastError(), std::string(solver_name) + " launch", case_name) ||
      !check_cuda(cudaDeviceSynchronize(), std::string(solver_name) + " execution", case_name) ||
      !check_cuda(cudaMemcpy(output_host->data(), d_output, bytes, cudaMemcpyDeviceToHost),
                  "D2H output",
                  case_name)) {
    cleanup();
    return false;
  }

  cleanup();
  return true;
}

bool run_ut_case(const std::string& case_name,
                 const std::vector<float>& q,
                 const std::vector<float>& k,
                 const std::vector<float>& v,
                 int N,
                 int d_model,
                 int h) {
  const size_t count = static_cast<size_t>(N) * d_model;
  const std::vector<float> expected = mha_reference(q, k, v, N, d_model, h);
  std::vector<float> actual(count, 0.0f);

  if (!run_solver_on_gpu(case_name, false, q, k, v, N, d_model, h, &actual)) {
    return false;
  }

  float max_abs_diff = 0.0f;
  int max_index = -1;
  for (size_t i = 0; i < count; ++i) {
    const float diff = std::fabs(actual[i] - expected[i]);
    if (diff > max_abs_diff) {
      max_abs_diff = diff;
      max_index = static_cast<int>(i);
    }
  }

  if (max_abs_diff > kTolerance) {
    std::cerr << "[FAIL] " << case_name << ": max_abs_diff=" << max_abs_diff
              << " at flat index " << max_index
              << ", expected=" << expected[max_index]
              << ", actual=" << actual[max_index] << '\n';
    return false;
  }

  std::cout << "[PASS] " << case_name << " (max_abs_diff=" << max_abs_diff << ")\n";
  return true;
}

bool run_solver_parity_case(const std::string& case_name,
                            const std::vector<float>& q,
                            const std::vector<float>& k,
                            const std::vector<float>& v,
                            int N,
                            int d_model,
                            int h) {
  const size_t count = static_cast<size_t>(N) * d_model;
  std::vector<float> expected(count, 0.0f);
  std::vector<float> actual(count, 0.0f);

  if (!run_solver_on_gpu(case_name, true, q, k, v, N, d_model, h, &expected) ||
      !run_solver_on_gpu(case_name, false, q, k, v, N, d_model, h, &actual)) {
    return false;
  }

  float max_abs_diff = 0.0f;
  int max_index = -1;
  for (size_t i = 0; i < count; ++i) {
    const float diff = std::fabs(actual[i] - expected[i]);
    if (diff > max_abs_diff) {
      max_abs_diff = diff;
      max_index = static_cast<int>(i);
    }
  }

  if (max_abs_diff > kTolerance) {
    std::cerr << "[FAIL] " << case_name << ": max_abs_diff=" << max_abs_diff
              << " at flat index " << max_index
              << ", expected=" << expected[max_index]
              << ", actual=" << actual[max_index] << '\n';
    return false;
  }

  std::cout << "[PASS] " << case_name << " (max_abs_diff=" << max_abs_diff << ")\n";
  return true;
}

bool run_benchmark_case(const std::string& name,
                        int N,
                        int d_model,
                        int h,
                        int warmup,
                        int iterations) {
  const size_t count = static_cast<size_t>(N) * d_model;
  const size_t bytes = count * sizeof(float);

  const std::vector<float> q = make_random_vector(count, 101, -1.0f, 1.0f);
  const std::vector<float> k = make_random_vector(count, 202, -1.0f, 1.0f);
  const std::vector<float> v = make_random_vector(count, 303, -1.0f, 1.0f);

  float* d_q = nullptr;
  float* d_k = nullptr;
  float* d_v = nullptr;
  float* d_output = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  auto cleanup = [&]() {
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
    if (d_q != nullptr) cudaFree(d_q);
    if (d_k != nullptr) cudaFree(d_k);
    if (d_v != nullptr) cudaFree(d_v);
    if (d_output != nullptr) cudaFree(d_output);
  };

  if (!check_cuda(cudaMalloc(&d_q, bytes), "cudaMalloc(Q)", name) ||
      !check_cuda(cudaMalloc(&d_k, bytes), "cudaMalloc(K)", name) ||
      !check_cuda(cudaMalloc(&d_v, bytes), "cudaMalloc(V)", name) ||
      !check_cuda(cudaMalloc(&d_output, bytes), "cudaMalloc(output)", name) ||
      !check_cuda(cudaMemcpy(d_q, q.data(), bytes, cudaMemcpyHostToDevice), "H2D Q", name) ||
      !check_cuda(cudaMemcpy(d_k, k.data(), bytes, cudaMemcpyHostToDevice), "H2D K", name) ||
      !check_cuda(cudaMemcpy(d_v, v.data(), bytes, cudaMemcpyHostToDevice), "H2D V", name) ||
      !check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)", name) ||
      !check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)", name)) {
    cleanup();
    return false;
  }

  for (int i = 0; i < warmup; ++i) {
    solve(d_q, d_k, d_v, d_output, N, d_model, h);
  }
  if (!check_cuda(cudaDeviceSynchronize(), "warmup sync", name)) {
    cleanup();
    return false;
  }

  if (!check_cuda(cudaEventRecord(start), "cudaEventRecord(start)", name)) {
    cleanup();
    return false;
  }
  for (int i = 0; i < iterations; ++i) {
    solve(d_q, d_k, d_v, d_output, N, d_model, h);
  }
  if (!check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)", name) ||
      !check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)", name) ||
      !check_cuda(cudaGetLastError(), "post-benchmark kernel status", name)) {
    cleanup();
    return false;
  }

  float elapsed_ms = 0.0f;
  if (!check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime", name)) {
    cleanup();
    return false;
  }

  const float avg_ms = elapsed_ms / static_cast<float>(iterations);
  const double flops = 4.0 * static_cast<double>(N) * static_cast<double>(N) *
                       static_cast<double>(d_model);
  const double tflops = flops / (avg_ms * 1.0e-3) / 1.0e12;

  std::cout << std::left << std::setw(18) << name
            << " N=" << std::setw(5) << N
            << " d_model=" << std::setw(5) << d_model
            << " h=" << std::setw(3) << h
            << " avg_ms=" << std::fixed << std::setprecision(3) << std::setw(8) << avg_ms
            << " approx_TFLOPS=" << std::setprecision(4) << tflops << '\n';

  cleanup();
  return true;
}

}  // namespace

int main() {
  int passed = 0;
  int total = 0;

  {
    ++total;
    const int N = 2;
    const int d_model = 4;
    const int h = 2;
    const std::vector<float> q = {
        1.0f, 2.0f, 3.0f, 4.0f,
        0.5f, -1.0f, 2.0f, 1.5f,
    };
    const std::vector<float> k = {
        1.0f, 0.0f, 0.5f, 1.0f,
        0.0f, 1.0f, 1.5f, -0.5f,
    };
    const std::vector<float> v = {
        1.0f, 10.0f, 2.0f, 20.0f,
        3.0f, 30.0f, 4.0f, 40.0f,
    };
    passed += run_ut_case("manual_small_case", q, k, v, N, d_model, h) ? 1 : 0;
  }

  {
    ++total;
    const int N = 7;
    const int d_model = 16;
    const int h = 4;
    const std::vector<float> q = make_random_vector(static_cast<size_t>(N) * d_model, 1, -1.0f, 1.0f);
    const std::vector<float> k = make_random_vector(static_cast<size_t>(N) * d_model, 2, -1.0f, 1.0f);
    const std::vector<float> v = make_random_vector(static_cast<size_t>(N) * d_model, 3, -1.0f, 1.0f);
    passed += run_ut_case("random_7x16_h4", q, k, v, N, d_model, h) ? 1 : 0;
  }

  {
    ++total;
    const int N = 33;
    const int d_model = 96;
    const int h = 3;
    const std::vector<float> q = make_random_vector(static_cast<size_t>(N) * d_model, 4, -0.5f, 0.5f);
    const std::vector<float> k = make_random_vector(static_cast<size_t>(N) * d_model, 5, -0.5f, 0.5f);
    const std::vector<float> v = make_random_vector(static_cast<size_t>(N) * d_model, 6, -0.5f, 0.5f);
    passed += run_ut_case("random_33x96_h3", q, k, v, N, d_model, h) ? 1 : 0;
  }

  {
    ++total;
    const int N = 128;
    const int d_model = 64;
    const int h = 1;
    const std::vector<float> q = make_random_vector(static_cast<size_t>(N) * d_model, 7, -0.25f, 0.25f);
    const std::vector<float> k = make_random_vector(static_cast<size_t>(N) * d_model, 8, -0.25f, 0.25f);
    const std::vector<float> v = make_random_vector(static_cast<size_t>(N) * d_model, 9, -0.25f, 0.25f);
    passed += run_ut_case("random_128x64_h1", q, k, v, N, d_model, h) ? 1 : 0;
  }

  {
    ++total;
    const int N = 1024;
    const int d_model = 1024;
    const int h = 16;
    const std::vector<float> q = make_random_vector(static_cast<size_t>(N) * d_model, 1, -10.0f, 10.0f);
    const std::vector<float> k = make_random_vector(static_cast<size_t>(N) * d_model, 2, -10.0f, 10.0f);
    const std::vector<float> v = make_random_vector(static_cast<size_t>(N) * d_model, 3, -10.0f, 10.0f);
    passed += run_solver_parity_case("flash_regression_1024x1024_h16_seed123", q, k, v, N, d_model, h) ? 1 : 0;
  }

  {
    ++total;
    const int N = 1024;
    const int d_model = 1024;
    const int h = 32;
    const std::vector<float> q = make_random_vector(static_cast<size_t>(N) * d_model, 10, -10.0f, 10.0f);
    const std::vector<float> k = make_random_vector(static_cast<size_t>(N) * d_model, 11, -10.0f, 10.0f);
    const std::vector<float> v = make_random_vector(static_cast<size_t>(N) * d_model, 12, -10.0f, 10.0f);
    passed += run_solver_parity_case("flash_regression_1024x1024_h32_seed101112", q, k, v, N, d_model, h) ? 1 : 0;
  }

  {
    ++total;
    const int N = 1024;
    const int d_model = 1024;
    const int h = 8;
    const std::vector<float> q = make_random_vector(static_cast<size_t>(N) * d_model, 13, -10.0f, 10.0f);
    const std::vector<float> k = make_random_vector(static_cast<size_t>(N) * d_model, 14, -10.0f, 10.0f);
    const std::vector<float> v = make_random_vector(static_cast<size_t>(N) * d_model, 15, -10.0f, 10.0f);
    passed += run_solver_parity_case("flash_regression_1024x1024_h8_seed131415", q, k, v, N, d_model, h) ? 1 : 0;
  }

  std::cout << "Passed " << passed << " / " << total << " cases\n";
  if (passed != total) {
    return 1;
  }

  int failed_benchmarks = 0;
  failed_benchmarks += run_benchmark_case("gpt2_like_128", 128, 768, 12, 10, 50) ? 0 : 1;
  failed_benchmarks += run_benchmark_case("gpt2_like_256", 256, 768, 12, 10, 20) ? 0 : 1;
  failed_benchmarks += run_benchmark_case("gpt2_like_512", 512, 768, 12, 5, 10) ? 0 : 1;
  failed_benchmarks += run_benchmark_case("single_head_1k", 1024, 64, 1, 5, 10) ? 0 : 1;
  return failed_benchmarks == 0 ? 0 : 1;
}
#endif
