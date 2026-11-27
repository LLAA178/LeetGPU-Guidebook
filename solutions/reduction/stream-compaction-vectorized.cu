#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarp;
constexpr int kVecsPerThread = 4;
constexpr int kTile = kThreads * kVecsPerThread * 4;  // 4096 floats per tile
constexpr int kGroup = 512;
constexpr int kMaxFastTiles = 32768;  // Covers up to 134,217,728 floats
constexpr int kMaxFastGroups = (kMaxFastTiles + kGroup - 1) / kGroup;
constexpr unsigned kFull = 0xffffffffu;

static_assert(kWarps <= kWarp, "Block warps must fit in a single warp");
static_assert(kGroup / kWarp <= kWarp, "Group warps must fit in a single warp");
static_assert((100000000 + kTile - 1) / kTile <= kMaxFastTiles,
              "Static scratch buffer must cover 100M floats");
static_assert(kMaxFastGroups <= kGroup,
              "Group totals must fit in a single scan block");

__device__ int g_tile_offsets[kMaxFastTiles];
__device__ int g_group_offsets[kMaxFastGroups];

__device__ __forceinline__ int warp_reduce_sum(int value) {
#if __CUDA_ARCH__ >= 800
    return __reduce_add_sync(kFull, value);
#else
    #pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFull, value, offset);
    }
    return value;
#endif
}

__device__ __forceinline__ int warp_inclusive_scan(int value) {
    const int lane = threadIdx.x & (kWarp - 1);
    #pragma unroll
    for (int step = 1; step < kWarp; step <<= 1) {
        const int earlier = __shfl_up_sync(kFull, value, step);
        if (lane >= step) value += earlier;
    }
    return value;
}

__global__ __launch_bounds__(kThreads) void count_tiles(
    const float* __restrict__ raw_input,
    const float* __restrict__ aligned_input,
    int head,
    int n_body,
    int* __restrict__ tile_offsets) {
    __shared__ int s_warp_sums[kWarps];

    const int warp = threadIdx.x / kWarp;
    const int lane = threadIdx.x & (kWarp - 1);
    const int tile_start = blockIdx.x * kTile;
    const int warp_start = tile_start + warp * (kWarp * kVecsPerThread * 4);

    int count = 0;
    if (tile_start + kTile <= n_body) {
        const float4* __restrict__ vec_in =
            reinterpret_cast<const float4*>(aligned_input + warp_start);
        #pragma unroll
        for (int r = 0; r < kVecsPerThread; ++r) {
            const float4 v = __ldg(&vec_in[r * kWarp + lane]);
            count += (v.x > 0.0f) + (v.y > 0.0f) + (v.z > 0.0f) + (v.w > 0.0f);
        }
    } else {
        #pragma unroll
        for (int r = 0; r < kVecsPerThread; ++r) {
            const int elem_idx = warp_start + (r * kWarp + lane) * 4;
            if (elem_idx + 4 <= n_body) {
                const float4 v =
                    __ldg(reinterpret_cast<const float4*>(aligned_input + elem_idx));
                count += (v.x > 0.0f) + (v.y > 0.0f) + (v.z > 0.0f) + (v.w > 0.0f);
            } else {
                if (elem_idx < n_body) count += (aligned_input[elem_idx] > 0.0f);
                if (elem_idx + 1 < n_body) count += (aligned_input[elem_idx + 1] > 0.0f);
                if (elem_idx + 2 < n_body) count += (aligned_input[elem_idx + 2] > 0.0f);
            }
        }
    }

    const int warp_sum = warp_reduce_sum(count);
    if (lane == 0) s_warp_sums[warp] = warp_sum;
    __syncthreads();

    if (warp == 0) {
        const int val = lane < kWarps ? s_warp_sums[lane] : 0;
        int block_sum = warp_reduce_sum(val);
        if (lane == 0) {
            if (blockIdx.x == 0 && head > 0) {
                for (int i = 0; i < head; ++i) {
                    block_sum += (raw_input[i] > 0.0f);
                }
            }
            tile_offsets[blockIdx.x] = block_sum;
        }
    }
}

__global__ __launch_bounds__(kGroup) void scan_groups(
    int* __restrict__ counts,
    int n,
    int* __restrict__ totals) {
    constexpr int kGroupWarps = kGroup / kWarp;
    __shared__ int s_warp_offsets[kGroupWarps];

    const int warp = threadIdx.x / kWarp;
    const int lane = threadIdx.x & (kWarp - 1);
    const int idx = blockIdx.x * kGroup + threadIdx.x;

    const int value = idx < n ? counts[idx] : 0;
    const int warp_prefix = warp_inclusive_scan(value);

    if (lane == kWarp - 1) s_warp_offsets[warp] = warp_prefix;
    __syncthreads();

    if (warp == 0) {
        const int wval = lane < kGroupWarps ? s_warp_offsets[lane] : 0;
        const int wend = warp_inclusive_scan(wval);
        if (lane < kGroupWarps) s_warp_offsets[lane] = wend - wval;
    }
    __syncthreads();

    const int block_prefix = s_warp_offsets[warp] + warp_prefix;
    if (idx < n) counts[idx] = block_prefix - value;
    if (totals && threadIdx.x == kGroup - 1) totals[blockIdx.x] = block_prefix;
}

__global__ __launch_bounds__(kThreads) void scatter_tiles(
    const float* __restrict__ raw_input,
    const float* __restrict__ aligned_input,
    int head,
    int n_body,
    float* __restrict__ output,
    const int* __restrict__ tile_offsets,
    const int* __restrict__ group_offsets) {
    __shared__ float s_out[kTile + 4];
    __shared__ int s_warp_offsets[kWarps];
    __shared__ int s_head_count;
    __shared__ int s_tile_total;

    const int warp = threadIdx.x / kWarp;
    const int lane = threadIdx.x & (kWarp - 1);
    const int tile_start = blockIdx.x * kTile;
    const int warp_start = tile_start + warp * (kWarp * kVecsPerThread * 4);

    if (blockIdx.x == 0 && head > 0 && threadIdx.x == 0) {
        int hc = 0;
        for (int i = 0; i < head; ++i) {
            const float v = raw_input[i];
            if (v > 0.0f) s_out[hc++] = v;
        }
        s_head_count = hc;
    }

    float4 vals[kVecsPerThread];
    if (tile_start + kTile <= n_body) {
        const float4* __restrict__ vec_in =
            reinterpret_cast<const float4*>(aligned_input + warp_start);
        #pragma unroll
        for (int r = 0; r < kVecsPerThread; ++r) {
            vals[r] = __ldg(&vec_in[r * kWarp + lane]);
        }
    } else {
        #pragma unroll
        for (int r = 0; r < kVecsPerThread; ++r) {
            const int elem_idx = warp_start + (r * kWarp + lane) * 4;
            if (elem_idx + 4 <= n_body) {
                vals[r] =
                    __ldg(reinterpret_cast<const float4*>(aligned_input + elem_idx));
            } else {
                float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (elem_idx < n_body) v.x = aligned_input[elem_idx];
                if (elem_idx + 1 < n_body) v.y = aligned_input[elem_idx + 1];
                if (elem_idx + 2 < n_body) v.z = aligned_input[elem_idx + 2];
                vals[r] = v;
            }
        }
    }

    int counts[kVecsPerThread];
    int thread_sum = 0;
    #pragma unroll
    for (int r = 0; r < kVecsPerThread; ++r) {
        const int c = (vals[r].x > 0.0f) + (vals[r].y > 0.0f) +
                      (vals[r].z > 0.0f) + (vals[r].w > 0.0f);
        counts[r] = c;
        thread_sum += c;
    }

    const int warp_total = warp_reduce_sum(thread_sum);
    if (lane == 0) s_warp_offsets[warp] = warp_total;
    __syncthreads();

    if (warp == 0) {
        const int wval = lane < kWarps ? s_warp_offsets[lane] : 0;
        const int wend = warp_inclusive_scan(wval);
        if (lane < kWarps) s_warp_offsets[lane] = wend - wval;
        if (lane == kWarps - 1) s_tile_total = wend;
    }
    __syncthreads();

    const int head_off = (blockIdx.x == 0 && head > 0) ? s_head_count : 0;
    const int tile_total = head_off + s_tile_total;
    if (tile_total == 0) return;

    int warp_cursor = head_off + s_warp_offsets[warp];
    #pragma unroll
    for (int r = 0; r < kVecsPerThread; ++r) {
        const int lane_end = warp_inclusive_scan(counts[r]);
        int pos = warp_cursor + (lane_end - counts[r]);
        if (counts[r] > 0) {
            const float4 v = vals[r];
            if (v.x > 0.0f) s_out[pos++] = v.x;
            if (v.y > 0.0f) s_out[pos++] = v.y;
            if (v.z > 0.0f) s_out[pos++] = v.z;
            if (v.w > 0.0f) s_out[pos++] = v.w;
        }
        warp_cursor += __shfl_sync(kFull, lane_end, kWarp - 1);
    }
    __syncthreads();

    int global_base = 0;
    if (tile_offsets) {
        global_base = tile_offsets[blockIdx.x];
        if (group_offsets) {
            global_base += group_offsets[blockIdx.x / kGroup];
        }
    }
    float* __restrict__ dst = output + global_base;

    const int align_rem =
        static_cast<int>((16u - (reinterpret_cast<uintptr_t>(dst) & 15u)) & 15u) >> 2;
    const int prefix = align_rem < tile_total ? align_rem : tile_total;

    if (threadIdx.x < prefix) {
        dst[threadIdx.x] = s_out[threadIdx.x];
    }

    const int remaining = tile_total - prefix;
    const int vec_count = remaining >> 2;
    float4* __restrict__ dst_vec = reinterpret_cast<float4*>(dst + prefix);
    for (int i = threadIdx.x; i < vec_count; i += kThreads) {
        const int s_idx = prefix + (i << 2);
        dst_vec[i] = make_float4(s_out[s_idx], s_out[s_idx + 1],
                                 s_out[s_idx + 2], s_out[s_idx + 3]);
    }

    const int tail_start = prefix + (vec_count << 2);
    const int tail_idx = tail_start + threadIdx.x;
    if (tail_idx < tile_total) {
        dst[tail_idx] = s_out[tail_idx];
    }
}

}  // namespace

extern "C" void solve(const float* A, int N, float* out) {
    if (N <= 0) return;

    const int misalign =
        static_cast<int>((16u - (reinterpret_cast<uintptr_t>(A) & 15u)) & 15u) >> 2;
    const int head = misalign < N ? misalign : N;
    const float* aligned_A = A + head;
    const int n_body = N - head;
    const int tiles = n_body > 0 ? (n_body + kTile - 1) / kTile : 1;

    if (tiles == 1) {
        scatter_tiles<<<1, kThreads>>>(A, aligned_A, head, n_body, out, nullptr, nullptr);
        return;
    }

    const int groups = (tiles + kGroup - 1) / kGroup;
    int* offsets = nullptr;
    int* group_offsets = nullptr;
    bool dynamic_alloc = false;

    if (tiles <= kMaxFastTiles) {
        cudaGetSymbolAddress(reinterpret_cast<void**>(&offsets), g_tile_offsets);
        if (groups > 1) {
            cudaGetSymbolAddress(reinterpret_cast<void**>(&group_offsets), g_group_offsets);
        }
    } else {
        dynamic_alloc = true;
        cudaMallocAsync(&offsets, static_cast<size_t>(tiles + groups) * sizeof(int), 0);
        group_offsets = groups > 1 ? (offsets + tiles) : nullptr;
    }

    count_tiles<<<tiles, kThreads>>>(A, aligned_A, head, n_body, offsets);
    if (groups == 1) {
        scan_groups<<<1, kGroup>>>(offsets, tiles, nullptr);
    } else {
        scan_groups<<<groups, kGroup>>>(offsets, tiles, group_offsets);
        scan_groups<<<1, kGroup>>>(group_offsets, groups, nullptr);
    }
    scatter_tiles<<<tiles, kThreads>>>(A, aligned_A, head, n_body, out, offsets, group_offsets);

    if (dynamic_alloc) {
        cudaFreeAsync(offsets, 0);
    }
}
