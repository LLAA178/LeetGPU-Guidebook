#include <cuda_runtime.h>

namespace {

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarp;
constexpr int kTile = kThreads;
constexpr int kScanThreads = 512;
constexpr unsigned kFull = 0xffffffffu;

__device__ __forceinline__ float warp_inclusive_scan(float val) {
    const int lane = threadIdx.x & (kWarp - 1);
    #pragma unroll
    for (int step = 1; step < kWarp; step <<= 1) {
        const float prev = __shfl_up_sync(kFull, val, step);
        if (lane >= step) val += prev;
    }
    return val;
}

__global__ __launch_bounds__(kThreads) void scan_tiles_scalar(
    const float* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ tile_sums,
    int n) {
    __shared__ float s_warp_offsets[kWarps];

    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x / kWarp;
    const int idx = blockIdx.x * kTile + threadIdx.x;

    const float val = idx < n ? input[idx] : 0.0f;
    const float warp_prefix = warp_inclusive_scan(val);

    if (lane == kWarp - 1) s_warp_offsets[warp] = warp_prefix;
    __syncthreads();

    if (warp == 0) {
        const float wval = lane < kWarps ? s_warp_offsets[lane] : 0.0f;
        const float wend = warp_inclusive_scan(wval);
        if (lane < kWarps) s_warp_offsets[lane] = wend - wval;
        if (tile_sums && lane == kWarps - 1) tile_sums[blockIdx.x] = wend;
    }
    __syncthreads();

    if (idx < n) {
        output[idx] = s_warp_offsets[warp] + warp_prefix;
    }
}

__global__ __launch_bounds__(kScanThreads) void scan_offsets_exclusive(
    const float* __restrict__ in_sums,
    float* __restrict__ out_offsets,
    float* __restrict__ group_sums,
    int n) {
    constexpr int kGroupWarps = kScanThreads / kWarp;
    __shared__ float s_warp_offsets[kGroupWarps];

    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x / kWarp;
    const int idx = blockIdx.x * kScanThreads + threadIdx.x;

    const float val = idx < n ? in_sums[idx] : 0.0f;
    const float warp_prefix = warp_inclusive_scan(val);

    if (lane == kWarp - 1) s_warp_offsets[warp] = warp_prefix;
    __syncthreads();

    if (warp == 0) {
        const float wval = lane < kGroupWarps ? s_warp_offsets[lane] : 0.0f;
        const float wend = warp_inclusive_scan(wval);
        if (lane < kGroupWarps) s_warp_offsets[lane] = wend - wval;
        if (group_sums && lane == kGroupWarps - 1) group_sums[blockIdx.x] = wend;
    }
    __syncthreads();

    if (idx < n) {
        out_offsets[idx] = s_warp_offsets[warp] + warp_prefix - val;
    }
}

__global__ __launch_bounds__(kThreads) void add_tile_offsets_scalar(
    float* __restrict__ output,
    const float* __restrict__ tile_offsets,
    int n) {
    if (blockIdx.x == 0) return;
    const int idx = blockIdx.x * kTile + threadIdx.x;
    if (idx < n) {
        output[idx] += tile_offsets[blockIdx.x];
    }
}

__global__ __launch_bounds__(kScanThreads) void add_group_offsets(
    float* __restrict__ data,
    const float* __restrict__ group_offsets,
    int n) {
    if (blockIdx.x == 0) return;
    const int idx = blockIdx.x * kScanThreads + threadIdx.x;
    if (idx < n) {
        data[idx] += group_offsets[blockIdx.x];
    }
}

void scan_block_sums(float* sums, int tiles) {
    if (tiles <= kScanThreads) {
        scan_offsets_exclusive<<<1, kScanThreads>>>(sums, sums, nullptr, tiles);
        return;
    }
    const int groups = (tiles + kScanThreads - 1) / kScanThreads;
    float* group_sums = nullptr;
    cudaMallocAsync(&group_sums, static_cast<size_t>(groups) * sizeof(float), 0);
    scan_offsets_exclusive<<<groups, kScanThreads>>>(sums, sums, group_sums, tiles);
    scan_block_sums(group_sums, groups);
    add_group_offsets<<<groups, kScanThreads>>>(sums, group_sums, tiles);
    cudaFreeAsync(group_sums, 0);
}

}  // namespace

extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;

    const int tiles = (N + kTile - 1) / kTile;
    if (tiles == 1) {
        scan_tiles_scalar<<<1, kThreads>>>(input, output, nullptr, N);
        return;
    }

    float* tile_sums = nullptr;
    cudaMallocAsync(&tile_sums, static_cast<size_t>(tiles) * sizeof(float), 0);
    scan_tiles_scalar<<<tiles, kThreads>>>(input, output, tile_sums, N);
    scan_block_sums(tile_sums, tiles);
    add_tile_offsets_scalar<<<tiles, kThreads>>>(output, tile_sums, N);
    cudaFreeAsync(tile_sums, 0);
}
