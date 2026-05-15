#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

#define CHECK_CUDA(call)                                                      \
  do {                                                                        \
    cudaError_t status__ = (call);                                            \
    if (status__ != cudaSuccess) {                                            \
      std::cerr << "CUDA error: " << cudaGetErrorString(status__)             \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;       \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (false)

constexpr int kNumElems = 1 << 20;
constexpr int kIters = 256;
constexpr int kRepeats = 5;
constexpr int kBlockSize = 256;
constexpr std::array<int, 4> kExtraSmemBytes = {0, 8 << 10, 16 << 10, 32 << 10};
constexpr std::array<int, 4> kSampleIndices = {0, 17, kNumElems / 2, kNumElems - 1};

float g_dep0_ms = 0.0f;

template <int kChains>
__global__ void occupancy_ilp_kernel(
    float* out,
    const float* in,
    int n,
    int iters,
    int smem_floats) {
  static_assert(kChains == 1 || kChains == 2 || kChains == 4, "unsupported chain count");
  constexpr int kFmasPerChain = 4 / kChains;

  extern __shared__ float smem[];

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane = threadIdx.x;

  float x = 0.0f;
  if (tid < n) {
    x = in[tid];
  }

  if (lane < smem_floats) {
    smem[lane] = (tid < n) ? x : 0.0f;
  }
  __syncthreads();

  if (tid >= n) {
    return;
  }

  float chains[kChains];
#pragma unroll
  for (int chain = 0; chain < kChains; ++chain) {
    chains[chain] = x + 0.03125f * static_cast<float>(chain + 1) + smem[lane];
  }

#pragma unroll 1
  for (int iter = 0; iter < iters; ++iter) {
    const float neighbor = smem[(lane + iter) % blockDim.x];
#pragma unroll
    for (int chain = 0; chain < kChains; ++chain) {
#pragma unroll
      for (int step = 0; step < kFmasPerChain; ++step) {
        chains[chain] = fmaf(
            chains[chain],
            1.00001f + 0.000001f * static_cast<float>(chain + step),
            neighbor + 0.001f * static_cast<float>(step + 1));
      }
    }
  }

  float sum = 0.0f;
#pragma unroll
  for (int chain = 0; chain < kChains; ++chain) {
    sum += chains[chain];
  }
  out[tid] = sum;
}

template <int kChains>
float host_reference(
    const std::vector<float>& host_in,
    int idx,
    int n,
    int iters,
    int block_size) {
  static_assert(kChains == 1 || kChains == 2 || kChains == 4, "unsupported chain count");
  constexpr int kFmasPerChain = 4 / kChains;

  const int lane = idx % block_size;
  const int block_base = idx - lane;
  const float x = host_in[static_cast<size_t>(idx)];

  std::array<float, kChains> chains{};
#pragma unroll
  for (int chain = 0; chain < kChains; ++chain) {
    chains[chain] = x + 0.03125f * static_cast<float>(chain + 1) + x;
  }

#pragma unroll 1
  for (int iter = 0; iter < iters; ++iter) {
    const int neighbor_lane = (lane + iter) % block_size;
    const int neighbor_idx = block_base + neighbor_lane;
    const float neighbor = neighbor_idx < n ? host_in[static_cast<size_t>(neighbor_idx)] : 0.0f;
#pragma unroll
    for (int chain = 0; chain < kChains; ++chain) {
#pragma unroll
      for (int step = 0; step < kFmasPerChain; ++step) {
        chains[chain] = std::fma(
            chains[chain],
            1.00001f + 0.000001f * static_cast<float>(chain + step),
            neighbor + 0.001f * static_cast<float>(step + 1));
      }
    }
  }

  float sum = 0.0f;
#pragma unroll
  for (float value : chains) {
    sum += value;
  }
  return sum;
}

void fill_input(std::vector<float>* host_in) {
  for (int i = 0; i < kNumElems; ++i) {
    (*host_in)[static_cast<size_t>(i)] = static_cast<float>(i % 251) * 0.25f;
  }
}

template <int kChains>
const void* kernel_ptr() {
  return reinterpret_cast<const void*>(occupancy_ilp_kernel<kChains>);
}

template <int kChains>
void launch_kernel(
    float* out,
    const float* in,
    int n,
    int iters,
    int smem_floats,
    size_t total_smem_bytes,
    cudaStream_t stream) {
  const int grid_size = (n + kBlockSize - 1) / kBlockSize;
  occupancy_ilp_kernel<kChains><<<grid_size, kBlockSize, total_smem_bytes, stream>>>(
      out, in, n, iters, smem_floats);
}

template <int kChains>
float validate_samples(
    const float* device_out,
    const std::vector<float>& host_in,
    int block_size) {
  float max_abs_error = 0.0f;
  for (int idx : kSampleIndices) {
    float actual = 0.0f;
    CHECK_CUDA(cudaMemcpy(&actual, device_out + idx, sizeof(float), cudaMemcpyDeviceToHost));
    const float expected = host_reference<kChains>(host_in, idx, kNumElems, kIters, block_size);
    max_abs_error = std::max(max_abs_error, std::fabs(actual - expected));
  }
  return max_abs_error;
}

template <int kChains>
void run_variant(
    std::string_view variant_name,
    const float* device_in,
    float* device_out,
    const std::vector<float>& host_in,
    const cudaDeviceProp& prop,
    cudaStream_t stream,
    cudaEvent_t start,
    cudaEvent_t stop) {
  cudaFuncAttributes attr{};
  CHECK_CUDA(cudaFuncGetAttributes(&attr, kernel_ptr<kChains>()));
  CHECK_CUDA(cudaFuncSetAttribute(
      kernel_ptr<kChains>(),
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      prop.sharedMemPerBlockOptin));

  for (int extra_smem_bytes : kExtraSmemBytes) {
    const size_t total_smem_bytes = static_cast<size_t>(kBlockSize * sizeof(float) + extra_smem_bytes);
    const int smem_floats = static_cast<int>(total_smem_bytes / sizeof(float));

    int active_blocks_per_sm = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm,
        kernel_ptr<kChains>(),
        kBlockSize,
        total_smem_bytes));

    launch_kernel<kChains>(
        device_out,
        device_in,
        kNumElems,
        kIters,
        smem_floats,
        total_smem_bytes,
        stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int repeat = 0; repeat < kRepeats; ++repeat) {
      launch_kernel<kChains>(
          device_out,
          device_in,
          kNumElems,
          kIters,
          smem_floats,
          total_smem_bytes,
          stream);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const float avg_ms = elapsed_ms / static_cast<float>(kRepeats);
    if (variant_name == "dep_chain" && extra_smem_bytes == 0) {
      g_dep0_ms = avg_ms;
    }

    const float active_warps =
        static_cast<float>(active_blocks_per_sm * kBlockSize) / static_cast<float>(prop.warpSize);
    const float max_warps =
        static_cast<float>(prop.maxThreadsPerMultiProcessor) / static_cast<float>(prop.warpSize);
    const float theoretical_occupancy = active_warps / max_warps;

    const float max_abs_error = validate_samples<kChains>(device_out, host_in, kBlockSize);
    const bool ok = max_abs_error < 1e-3f;

    std::cout << std::fixed << std::setprecision(3)
              << variant_name << ' '
              << extra_smem_bytes << ' '
              << attr.numRegs << ' '
              << attr.localSizeBytes << ' '
              << active_blocks_per_sm << ' '
              << active_warps << ' '
              << max_warps << ' '
              << theoretical_occupancy << ' '
              << avg_ms << ' '
              << (g_dep0_ms / avg_ms) << ' '
              << max_abs_error << ' '
              << (ok ? "PASS" : "FAIL") << '\n';
  }
}

}  // namespace

int main() {
  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

  std::vector<float> host_in(static_cast<size_t>(kNumElems));
  fill_input(&host_in);

  float* device_in = nullptr;
  float* device_out = nullptr;
  CHECK_CUDA(cudaMalloc(&device_in, static_cast<size_t>(kNumElems) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&device_out, static_cast<size_t>(kNumElems) * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(
      device_in,
      host_in.data(),
      static_cast<size_t>(kNumElems) * sizeof(float),
      cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  cudaEvent_t start;
  cudaEvent_t stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "device: " << prop.name << '\n';
  std::cout << "num_elems: " << kNumElems << '\n';
  std::cout << "iters: " << kIters << '\n';
  std::cout << "repeats: " << kRepeats << '\n';
  std::cout << "block_size: " << kBlockSize << '\n';
  std::cout << "sm_count: " << prop.multiProcessorCount << '\n';
  std::cout << "max_threads_per_sm: " << prop.maxThreadsPerMultiProcessor << '\n';
  std::cout << "warp_size: " << prop.warpSize << '\n';
  std::cout << "variant extra_smem_bytes regs_per_thread local_bytes_per_thread"
            << " active_blocks_per_sm active_warps_per_sm max_warps_per_sm theoretical_occupancy"
            << " kernel_ms speedup_vs_dep_0KB max_abs_error check"
            << '\n';

  std::cout << "experiment: occupancy_ilp_matrix\n";

  run_variant<1>("dep_chain", device_in, device_out, host_in, prop, stream, start, stop);
  run_variant<2>("ilp2", device_in, device_out, host_in, prop, stream, start, stop);
  run_variant<4>("ilp4", device_in, device_out, host_in, prop, stream, start, stop);

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFree(device_in));
  CHECK_CUDA(cudaFree(device_out));
  return 0;
}
