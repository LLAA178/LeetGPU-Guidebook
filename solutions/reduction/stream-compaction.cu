#include <cuda_runtime.h>

namespace {

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr int kRows = 4;
constexpr int kTile = kThreads * kRows;
constexpr int kGroup = 512;
constexpr unsigned kFull = 0xffffffffu;
static_assert((100000000 + kTile * kGroup - 1) / (kTile * kGroup) <= kGroup,
              "Group totals must fit in a single scan block");

__device__ int warp_sum_prefix(int value) {
    for (int step = 1; step < kWarp; step *= 2) {
        const int earlier = __shfl_up_sync(kFull, value, step);
        if ((threadIdx.x % kWarp) >= step) value += earlier;
    }
    return value;
}

// Return the number of selected items in all preceding warps of this block.
template<int Threads>
__device__ int preceding_warps(int warp_total, int* shared) {
    const int warp = threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    if (lane == kWarp - 1) shared[warp] = warp_total;
    __syncthreads();
    if (warp == 0) {
        const int value = lane < Threads / kWarp ? shared[lane] : 0;
        const int end = warp_sum_prefix(value);
        if (lane < Threads / kWarp) shared[lane] = end - value;
    }
    __syncthreads();
    return shared[warp];
}

__global__ void scan_groups(int* counts, int n, int* totals) {
    __shared__ int warp_offsets[kGroup / kWarp];
    const int i = blockIdx.x * kGroup + threadIdx.x;
    const int value = i < n ? counts[i] : 0;
    const int prefix = warp_sum_prefix(value);
    const int end = preceding_warps<kGroup>(prefix, warp_offsets) + prefix;
    if (i < n) counts[i] = end - value;
    if (totals && threadIdx.x == kGroup - 1) totals[blockIdx.x] = end;
}

template<bool Scatter>
__global__ void process_tiles(const float* input, int n, float* output,
                              int* tile_offsets, const int* group_offsets) {
    __shared__ int warp_offsets[kThreads / kWarp];
    const int warp = threadIdx.x / kWarp;
    const int lane = threadIdx.x % kWarp;
    const int start = blockIdx.x * kTile + warp * (kWarp * kRows) + lane;
    float values[kRows];
    unsigned masks[kRows];
    int total = 0;
    #pragma unroll
    for (int row = 0; row < kRows; ++row) {
        const int i = start + row * kWarp;
        values[row] = i < n ? input[i] : 0.0f;
        masks[row] = __ballot_sync(kFull, values[row] > 0.0f);
        total += __popc(masks[row]);
    }

    const int warp_begin = preceding_warps<kThreads>(total, warp_offsets);
    if (!Scatter) {
        if (threadIdx.x == kThreads - 1) tile_offsets[blockIdx.x] = warp_begin + total;
        return;
    }
    int base = warp_begin;
    if (tile_offsets) base += tile_offsets[blockIdx.x] + group_offsets[blockIdx.x / kGroup];
    const unsigned lower_lanes = (1u << lane) - 1u;
    #pragma unroll
    for (int row = 0; row < kRows; ++row) {
        if (values[row] > 0.0f) {
            output[base + __popc(masks[row] & lower_lanes)] = values[row];
        }
        base += __popc(masks[row]);
    }
}

}  // namespace

extern "C" void solve(const float* A, int N, float* out) {
    const int tiles = (N + kTile - 1) / kTile;
    if (tiles == 1) {
        process_tiles<true><<<1, kThreads>>>(A, N, out, nullptr, nullptr);
        return;
    }

    const int groups = (tiles + kGroup - 1) / kGroup;
    int* offsets;
    cudaMallocAsync(&offsets, (tiles + groups) * sizeof(int), 0);
    int* group_offsets = offsets + tiles;
    process_tiles<false><<<tiles, kThreads>>>(A, N, out, offsets, nullptr);
    scan_groups<<<groups, kGroup>>>(offsets, tiles, group_offsets);
    scan_groups<<<1, kGroup>>>(group_offsets, groups, nullptr);
    process_tiles<true><<<tiles, kThreads>>>(A, N, out, offsets, group_offsets);
    cudaFreeAsync(offsets, 0);
}
