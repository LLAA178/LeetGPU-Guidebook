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
constexpr int kIters = 128;
constexpr int kRepeats = 5;
constexpr std::array<int, 4> kBlockSizes = {64, 128, 256, 512};
constexpr std::array<int, 5> kExtraSmemBytes = {0, 4 << 10, 8 << 10, 16 << 10, 32 << 10};
constexpr std::array<int, 4> kSampleIndices = {0, 17, kNumElems / 2, kNumElems - 1};

template <int kRegs>
__global__ void occupancy_sweep_kernel(
    float* out,
    const float* in,
    int n,
    int iters,
    int smem_floats) {
  // The sweep keeps the instruction stream fixed and only changes residency.
  extern __shared__ float smem[];

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane = threadIdx.x;

  float x = 0.0f;
  if (tid < n) {
    x = in[tid];
  }

  if (lane < smem_floats) {
    smem[lane] = (lane < blockDim.x && tid < n) ? x : 0.0f;
  }
  __syncthreads();

  if (tid >= n) {
    return;
  }

  float regs[kRegs];
#pragma unroll
  for (int i = 0; i < kRegs; ++i) {
    regs[i] = x + 0.125f * static_cast<float>(i + 1);
  }

  float acc = x + smem[lane];
#pragma unroll 1
  for (int iter = 0; iter < iters; ++iter) {
    const float neighbor = smem[(lane + iter) % blockDim.x];
#pragma unroll
    for (int i = 0; i < kRegs; ++i) {
      regs[i] = fmaf(regs[i], 1.00001f + 0.000001f * static_cast<float>(i), acc + neighbor);
      acc = fmaf(acc, 0.99991f, regs[i] * 0.0001f);
    }
  }

  float sum = acc;
#pragma unroll
  for (int i = 0; i < kRegs; ++i) {
    sum += regs[i];
  }
  out[tid] = sum;
}

template <int kRegs>
float host_reference(const std::vector<float>& host_in, int idx, int n, int block_size, int iters) {
  const int lane = idx % block_size;
  const int block_base = idx - lane;
  const float x = host_in[static_cast<size_t>(idx)];
  std::vector<float> regs(static_cast<size_t>(kRegs));
  for (int i = 0; i < kRegs; ++i) {
    regs[static_cast<size_t>(i)] = x + 0.125f * static_cast<float>(i + 1);
  }

  float acc = x + x;
  for (int iter = 0; iter < iters; ++iter) {
    const int neighbor_lane = (lane + iter) % block_size;
    const int neighbor_idx = block_base + neighbor_lane;
    const float neighbor = neighbor_idx < n ? host_in[static_cast<size_t>(neighbor_idx)] : 0.0f;
    for (int i = 0; i < kRegs; ++i) {
      regs[static_cast<size_t>(i)] =
          std::fma(regs[static_cast<size_t>(i)], 1.00001f + 0.000001f * static_cast<float>(i), acc + neighbor);
      acc = std::fma(acc, 0.99991f, regs[static_cast<size_t>(i)] * 0.0001f);
    }
  }

  float sum = acc;
  for (float value : regs) {
    sum += value;
  }
  return sum;
}

void fill_input(std::vector<float>* host_in) {
  for (int i = 0; i < kNumElems; ++i) {
    (*host_in)[static_cast<size_t>(i)] = static_cast<float>(i % 251) * 0.25f;
  }
}

template <int kRegs>
const void* kernel_ptr() {
  return reinterpret_cast<const void*>(occupancy_sweep_kernel<kRegs>);
}

template <int kRegs>
void launch_kernel(
    float* out,
    const float* in,
    int n,
    int iters,
    int smem_floats,
    int block_size,
    size_t total_smem_bytes,
    cudaStream_t stream) {
  const int grid_size = (n + block_size - 1) / block_size;
  occupancy_sweep_kernel<kRegs><<<grid_size, block_size, total_smem_bytes, stream>>>(
      out, in, n, iters, smem_floats);
}

template <int kRegs>
float validate_samples(
    const float* device_out,
    const std::vector<float>& host_in,
    int block_size) {
  float max_abs_error = 0.0f;
  for (int idx : kSampleIndices) {
    float actual = 0.0f;
    CHECK_CUDA(cudaMemcpy(&actual, device_out + idx, sizeof(float), cudaMemcpyDeviceToHost));
    const float expected = host_reference<kRegs>(host_in, idx, kNumElems, block_size, kIters);
    max_abs_error = std::max(max_abs_error, std::fabs(actual - expected));
  }
  return max_abs_error;
}

template <int kRegs>
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
  CHECK_CUDA(cudaFuncGetAttributes(&attr, occupancy_sweep_kernel<kRegs>));

  for (int block_size : kBlockSizes) {
    const int base_smem_bytes = block_size * static_cast<int>(sizeof(float));
    for (int extra_smem_bytes : kExtraSmemBytes) {
      const size_t total_smem_bytes = static_cast<size_t>(base_smem_bytes + extra_smem_bytes);
      const int smem_floats = static_cast<int>(total_smem_bytes / sizeof(float));

      int active_blocks_per_sm = 0;
      CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks_per_sm,
          occupancy_sweep_kernel<kRegs>,
          block_size,
          total_smem_bytes));

      launch_kernel<kRegs>(
          device_out,
          device_in,
          kNumElems,
          kIters,
          smem_floats,
          block_size,
          total_smem_bytes,
          stream);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaStreamSynchronize(stream));

      CHECK_CUDA(cudaEventRecord(start, stream));
      for (int repeat = 0; repeat < kRepeats; ++repeat) {
        launch_kernel<kRegs>(
            device_out,
            device_in,
            kNumElems,
            kIters,
            smem_floats,
            block_size,
            total_smem_bytes,
            stream);
      }
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaEventRecord(stop, stream));
      CHECK_CUDA(cudaEventSynchronize(stop));

      float elapsed_ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
      const float avg_ms = elapsed_ms / static_cast<float>(kRepeats);

      const float active_warps =
          static_cast<float>(active_blocks_per_sm * block_size) / static_cast<float>(prop.warpSize);
      const float max_warps =
          static_cast<float>(prop.maxThreadsPerMultiProcessor) / static_cast<float>(prop.warpSize);
      const float theoretical_occupancy = active_warps / max_warps;

      const float max_abs_error = validate_samples<kRegs>(device_out, host_in, block_size);
      const bool ok = max_abs_error < 1e-3f;

      std::cout << variant_name << ' '
                << block_size << ' '
                << attr.numRegs << ' '
                << attr.localSizeBytes << ' '
                << extra_smem_bytes << ' '
                << total_smem_bytes << ' '
                << active_blocks_per_sm << ' '
                << active_warps << ' '
                << max_warps << ' '
                << theoretical_occupancy << ' '
                << avg_ms << ' '
                << max_abs_error << ' '
                << (ok ? "PASS" : "FAIL") << '\n';
    }
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
  std::cout << "sm_count: " << prop.multiProcessorCount << '\n';
  std::cout << "max_threads_per_sm: " << prop.maxThreadsPerMultiProcessor << '\n';
  std::cout << "warp_size: " << prop.warpSize << '\n';
  std::cout << "variant block_size regs_per_thread local_bytes_per_thread extra_smem_bytes total_smem_bytes"
            << " active_blocks_per_sm active_warps_per_sm max_warps_per_sm theoretical_occupancy"
            << " kernel_ms max_abs_error check"
            << '\n';

  run_variant<4>("low_regs", device_in, device_out, host_in, prop, stream, start, stop);
  run_variant<32>("mid_regs", device_in, device_out, host_in, prop, stream, start, stop);

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFree(device_in));
  CHECK_CUDA(cudaFree(device_out));
  return 0;
}
