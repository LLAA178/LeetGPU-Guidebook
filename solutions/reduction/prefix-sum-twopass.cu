#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarp;
constexpr int kTile = kThreads * 4;  // 1024 floats per tile
constexpr int kGroup = 512;
constexpr int kMaxFastTiles = 1024;  // Covers up to 1,048,576 floats with zero alloc
constexpr unsigned kFull = 0xffffffffu;

__device__ float g_tile_sums[kMaxFastTiles];
__device__ float g_group_sums[(kMaxFastTiles + kGroup - 1) / kGroup];

__device__ __forceinline__ float warp_inclusive_scan(float val) {
    const int lane = threadIdx.x & (kWarp - 1);
    #pragma unroll
    for (int step = 1; step < kWarp; step <<= 1) {
        const float prev = __shfl_up_sync(kFull, val, step);
        if (lane >= step) val += prev;
    }
    return val;
}

// Fast 2-pass Level-1: only compute per-tile sum (no write to output).
__global__ __launch_bounds__(kThreads) void reduce_tiles_vec4(
    const float4* __restrict__ input4,
    const float* __restrict__ input,
    float* __restrict__ tile_sums,
    int n4,
    int n) {
    __shared__ float s_warp_sums[kWarps];

    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x / kWarp;
    const int vec_idx = blockIdx.x * kThreads + threadIdx.x;

    float sum = 0.0f;
    if (vec_idx < n4) {
        const float4 v = __ldg(&input4[vec_idx]);
        sum = (v.x + v.y) + (v.z + v.w);
    } else {
        const int base = vec_idx << 2;
        if (base < n) sum += __ldg(&input[base]);
        if (base + 1 < n) sum += __ldg(&input[base + 1]);
        if (base + 2 < n) sum += __ldg(&input[base + 2]);
    }

    // Reduce across warp
    #pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(kFull, sum, offset);
    }
    if (lane == 0) s_warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        float bsum = lane < kWarps ? s_warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = kWarps / 2; offset > 0; offset >>= 1) {
            bsum += __shfl_down_sync(kFull, bsum, offset);
        }
        if (lane == 0) tile_sums[blockIdx.x] = bsum;
    }
}

__global__ __launch_bounds__(kGroup) void scan_groups_exclusive(
    float* __restrict__ sums,
    float* __restrict__ group_totals,
    int count) {
    constexpr int kGroupWarps = kGroup / kWarp;
    __shared__ float s_warp_offsets[kGroupWarps];

    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x / kWarp;
    const int idx = blockIdx.x * kGroup + threadIdx.x;

    const float val = idx < count ? sums[idx] : 0.0f;
    const float wprefix = warp_inclusive_scan(val);

    if (lane == kWarp - 1) s_warp_offsets[warp] = wprefix;
    __syncthreads();

    if (warp == 0) {
        const float wval = lane < kGroupWarps ? s_warp_offsets[lane] : 0.0f;
        const float wend = warp_inclusive_scan(wval);
        if (lane < kGroupWarps) s_warp_offsets[lane] = wend - wval;
        if (group_totals && lane == kGroupWarps - 1) group_totals[blockIdx.x] = wend;
    }
    __syncthreads();

    if (idx < count) {
        sums[idx] = s_warp_offsets[warp] + wprefix - val;
    }
}

__global__ __launch_bounds__(kGroup) void add_group_base(
    float* __restrict__ sums,
    const float* __restrict__ group_offsets,
    int count) {
    if (blockIdx.x == 0) return;
    const int idx = blockIdx.x * kGroup + threadIdx.x;
    if (idx < count) {
        sums[idx] += group_offsets[blockIdx.x];
    }
}

// Fast 2-pass Level-3 (or single-tile): read float4, scan within tile, add tile_offset, write float4.
__global__ __launch_bounds__(kThreads) void scan_write_tiles_vec4(
    const float4* __restrict__ input4,
    const float* __restrict__ input,
    float4* __restrict__ output4,
    float* __restrict__ output,
    const float* __restrict__ tile_offsets,
    const float* __restrict__ group_offsets,
    int n4,
    int n) {
    __shared__ float s_warp_offsets[kWarps];

    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x / kWarp;
    const int vec_idx = blockIdx.x * kThreads + threadIdx.x;

    float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (vec_idx < n4) {
        v = __ldg(&input4[vec_idx]);
    } else {
        const int base = vec_idx << 2;
        if (base < n) v.x = __ldg(&input[base]);
        if (base + 1 < n) v.y = __ldg(&input[base + 1]);
        if (base + 2 < n) v.z = __ldg(&input[base + 2]);
    }

    // Local inclusive prefix inside float4
    v.y += v.x;
    v.z += v.y;
    v.w += v.z;

    const float thread_total = v.w;
    const float warp_inc = warp_inclusive_scan(thread_total);
    if (lane == kWarp - 1) s_warp_offsets[warp] = warp_inc;
    __syncthreads();

    if (warp == 0) {
        const float wval = lane < kWarps ? s_warp_offsets[lane] : 0.0f;
        const float wend = warp_inclusive_scan(wval);
        if (lane < kWarps) s_warp_offsets[lane] = wend - wval;
    }
    __syncthreads();

    float base_prefix = s_warp_offsets[warp] + (warp_inc - thread_total);
    if (tile_offsets && blockIdx.x > 0) {
        base_prefix += tile_offsets[blockIdx.x];
        if (group_offsets) {
            base_prefix += group_offsets[blockIdx.x / kGroup];
        }
    }

    if (vec_idx < n4) {
        output4[vec_idx] = make_float4(
            v.x + base_prefix,
            v.y + base_prefix,
            v.z + base_prefix,
            v.w + base_prefix);
    } else {
        const int base = vec_idx << 2;
        if (base < n) output[base] = v.x + base_prefix;
        if (base + 1 < n) output[base + 1] = v.y + base_prefix;
        if (base + 2 < n) output[base + 2] = v.z + base_prefix;
    }
}

// Unaligned scalar fallback per tile (4 elements per thread)
__global__ __launch_bounds__(kThreads) void reduce_tiles_unaligned(
    const float* __restrict__ input,
    float* __restrict__ tile_sums,
    int n) {
    __shared__ float s_warp_sums[kWarps];
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x / kWarp;
    const int base = blockIdx.x * kTile + (threadIdx.x << 2);

    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        if (base + k < n) sum += __ldg(&input[base + k]);
    }
    #pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(kFull, sum, offset);
    }
    if (lane == 0) s_warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        float bsum = lane < kWarps ? s_warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = kWarps / 2; offset > 0; offset >>= 1) {
            bsum += __shfl_down_sync(kFull, bsum, offset);
        }
        if (lane == 0) tile_sums[blockIdx.x] = bsum;
    }
}

__global__ __launch_bounds__(kThreads) void scan_write_tiles_unaligned(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ tile_offsets,
    const float* __restrict__ group_offsets,
    int n) {
    __shared__ float s_warp_offsets[kWarps];
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x / kWarp;
    const int base = blockIdx.x * kTile + (threadIdx.x << 2);

    float vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        if (base + k < n) vals[k] = __ldg(&input[base + k]);
    }
    vals[1] += vals[0];
    vals[2] += vals[1];
    vals[3] += vals[2];

    const float thread_total = vals[3];
    const float warp_inc = warp_inclusive_scan(thread_total);
    if (lane == kWarp - 1) s_warp_offsets[warp] = warp_inc;
    __syncthreads();

    if (warp == 0) {
        const float wval = lane < kWarps ? s_warp_offsets[lane] : 0.0f;
        const float wend = warp_inclusive_scan(wval);
        if (lane < kWarps) s_warp_offsets[lane] = wend - wval;
    }
    __syncthreads();

    float base_prefix = s_warp_offsets[warp] + (warp_inc - thread_total);
    if (tile_offsets && blockIdx.x > 0) {
        base_prefix += tile_offsets[blockIdx.x];
        if (group_offsets) {
            base_prefix += group_offsets[blockIdx.x / kGroup];
        }
    }

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        if (base + k < n) output[base + k] = vals[k] + base_prefix;
    }
}

void scan_hierarchy_large(float* sums, int count) {
    if (count <= kGroup) {
        scan_groups_exclusive<<<1, kGroup>>>(sums, nullptr, count);
        return;
    }
    const int groups = (count + kGroup - 1) / kGroup;
    float* group_sums = nullptr;
    cudaMallocAsync(&group_sums, static_cast<size_t>(groups) * sizeof(float), 0);
    scan_groups_exclusive<<<groups, kGroup>>>(sums, group_sums, count);
    scan_hierarchy_large(group_sums, groups);
    add_group_base<<<groups, kGroup>>>(sums, group_sums, count);
    cudaFreeAsync(group_sums, 0);
}

}  // namespace

extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;

    const bool aligned = ((reinterpret_cast<uintptr_t>(input) |
                           reinterpret_cast<uintptr_t>(output)) & 15u) == 0u;
    const int n4 = N >> 2;
    const int tiles = (N + kTile - 1) / kTile;

    if (tiles == 1) {
        if (aligned) {
            scan_write_tiles_vec4<<<1, kThreads>>>(
                reinterpret_cast<const float4*>(input), input,
                reinterpret_cast<float4*>(output), output,
                nullptr, nullptr, n4, N);
        } else {
            scan_write_tiles_unaligned<<<1, kThreads>>>(
                input, output, nullptr, nullptr, N);
        }
        return;
    }

    if (tiles <= kMaxFastTiles) {
        float* tile_sums = nullptr;
        float* group_sums = nullptr;
        cudaGetSymbolAddress(reinterpret_cast<void**>(&tile_sums), g_tile_sums);
        const int groups = (tiles + kGroup - 1) / kGroup;

        if (aligned) {
            reduce_tiles_vec4<<<tiles, kThreads>>>(
                reinterpret_cast<const float4*>(input), input, tile_sums, n4, N);
        } else {
            reduce_tiles_unaligned<<<tiles, kThreads>>>(input, tile_sums, N);
        }

        if (groups == 1) {
            scan_groups_exclusive<<<1, kGroup>>>(tile_sums, nullptr, tiles);
        } else {
            cudaGetSymbolAddress(reinterpret_cast<void**>(&group_sums), g_group_sums);
            scan_groups_exclusive<<<groups, kGroup>>>(tile_sums, group_sums, tiles);
            scan_groups_exclusive<<<1, kGroup>>>(group_sums, nullptr, groups);
        }

        if (aligned) {
            scan_write_tiles_vec4<<<tiles, kThreads>>>(
                reinterpret_cast<const float4*>(input), input,
                reinterpret_cast<float4*>(output), output,
                tile_sums, group_sums, n4, N);
        } else {
            scan_write_tiles_unaligned<<<tiles, kThreads>>>(
                input, output, tile_sums, group_sums, N);
        }
        return;
    }

    float* tile_sums = nullptr;
    cudaMallocAsync(&tile_sums, static_cast<size_t>(tiles) * sizeof(float), 0);
    if (aligned) {
        reduce_tiles_vec4<<<tiles, kThreads>>>(
            reinterpret_cast<const float4*>(input), input, tile_sums, n4, N);
        scan_hierarchy_large(tile_sums, tiles);
        scan_write_tiles_vec4<<<tiles, kThreads>>>(
            reinterpret_cast<const float4*>(input), input,
            reinterpret_cast<float4*>(output), output,
            tile_sums, nullptr, n4, N);
    } else {
        reduce_tiles_unaligned<<<tiles, kThreads>>>(input, tile_sums, N);
        scan_hierarchy_large(tile_sums, tiles);
        scan_write_tiles_unaligned<<<tiles, kThreads>>>(
            input, output, tile_sums, nullptr, N);
    }
    cudaFreeAsync(tile_sums, 0);
}
