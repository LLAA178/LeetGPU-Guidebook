#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarp;
constexpr int kItemsPerThread = 4;
constexpr int kTileSize = kThreads * kItemsPerThread;  // 1,024 keys per tile
constexpr int kBins = 256;
constexpr int kMaxN = 100000000;
constexpr int kMaxTiles = (kMaxN + kTileSize - 1) / kTileSize + 8;
constexpr unsigned kFullMask = 0xffffffffu;

__device__ __align__(16) uint32_t g_base_temp[kMaxN];
__device__ uint32_t g_tile_hist[kBins * kMaxTiles];
__device__ uint32_t g_bin_totals[kBins];
__device__ uint32_t g_bin_offsets[kBins];

// Pass Kernel 2: Exclusive scan across tiles for each bin b = blockIdx.x (256 blocks)
__global__ __launch_bounds__(kThreads) void baseline_scan_tile_bins_kernel(int num_tiles) {
    __shared__ uint32_t s_warp_sums[kWarps];
    __shared__ uint32_t s_carry;

    const int bin = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & (kWarp - 1);
    const int warp = tid / kWarp;

    if (tid == 0) {
        s_carry = 0u;
    }
    __syncthreads();

    uint32_t* row = g_tile_hist + static_cast<size_t>(bin) * static_cast<size_t>(num_tiles);
    for (int base = 0; base < num_tiles; base += kThreads) {
        const int t = base + tid;
        const uint32_t c = (t < num_tiles) ? row[t] : 0u;

        uint32_t incl = c;
        #pragma unroll
        for (int step = 1; step < kWarp; step <<= 1) {
            const uint32_t prev = __shfl_up_sync(kFullMask, incl, step);
            if (lane >= step) incl += prev;
        }
        if (lane == kWarp - 1) {
            s_warp_sums[warp] = incl;
        }
        __syncthreads();

        uint32_t warp_excl = 0u;
        for (int w = 0; w < warp; ++w) {
            warp_excl += s_warp_sums[w];
        }
        const uint32_t carry = s_carry;
        if (t < num_tiles) {
            row[t] = carry + warp_excl + (incl - c);
        }
        __syncthreads();

        if (tid == kThreads - 1) {
            s_carry = carry + warp_excl + incl;
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_bin_totals[bin] = s_carry;
    }
}

// Pass Kernel 3: Single-block exclusive scan of the 256 global bin totals
__global__ __launch_bounds__(kThreads) void baseline_scan_global_bins_kernel() {
    __shared__ uint32_t s_warp_sums[kWarps];

    const int tid = threadIdx.x;
    const int lane = tid & (kWarp - 1);
    const int warp = tid / kWarp;

    const uint32_t c = g_bin_totals[tid];
    uint32_t incl = c;
    #pragma unroll
    for (int step = 1; step < kWarp; step <<= 1) {
        const uint32_t prev = __shfl_up_sync(kFullMask, incl, step);
        if (lane >= step) incl += prev;
    }
    if (lane == kWarp - 1) {
        s_warp_sums[warp] = incl;
    }
    __syncthreads();

    uint32_t warp_excl = 0u;
    for (int w = 0; w < warp; ++w) {
        warp_excl += s_warp_sums[w];
    }
    g_bin_offsets[tid] = warp_excl + (incl - c);
}

// Pass Kernel 1 & 4 templated on pass index so ping-pong buffers are resolved on-device
template <int kPass>
__global__ __launch_bounds__(kThreads) void baseline_pass_hist_entry(
    const uint32_t* __restrict__ input,
    const uint32_t* __restrict__ output,
    int n,
    int num_tiles) {
    __shared__ uint32_t s_hist[kBins];
    const uint32_t* src = (kPass == 0) ? input : ((kPass & 1) ? g_base_temp : output);
    constexpr int kShift = kPass * 8;

    const int tid = threadIdx.x;
    const int tile = blockIdx.x;
    s_hist[tid] = 0u;
    __syncthreads();

    const int tile_base = tile * kTileSize;
    #pragma unroll
    for (int r = 0; r < kItemsPerThread; ++r) {
        const int idx = tile_base + r * kThreads + tid;
        if (idx < n) {
            const unsigned digit = (src[idx] >> kShift) & 0xffu;
            atomicAdd(&s_hist[digit], 1u);
        }
    }
    __syncthreads();

    g_tile_hist[static_cast<size_t>(tid) * static_cast<size_t>(num_tiles) + tile] = s_hist[tid];
}

template <int kPass>
__global__ __launch_bounds__(kThreads) void baseline_pass_scatter_entry(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    int n,
    int num_tiles) {
    const uint32_t* src = (kPass == 0) ? input : ((kPass & 1) ? g_base_temp : output);
    uint32_t* dst = (kPass & 1) ? output : g_base_temp;
    constexpr int kShift = kPass * 8;

    __shared__ uint32_t s_warp_hist[kWarps][kBins];
    __shared__ uint32_t s_bin_base[kBins];

    const int tid = threadIdx.x;
    const int lane = tid & (kWarp - 1);
    const int warp = tid / kWarp;
    const int tile = blockIdx.x;
    const int tile_base = tile * kTileSize;

    #pragma unroll
    for (int w = 0; w < kWarps; ++w) {
        s_warp_hist[w][tid] = 0u;
    }
    s_bin_base[tid] = g_bin_offsets[tid] +
                      g_tile_hist[static_cast<size_t>(tid) * static_cast<size_t>(num_tiles) + tile];
    __syncthreads();

    uint32_t keys[kItemsPerThread];
    uint16_t ranks[kItemsPerThread];
    bool valid[kItemsPerThread];

    const int warp_base = tile_base + warp * (kWarp * kItemsPerThread);
    #pragma unroll
    for (int r = 0; r < kItemsPerThread; ++r) {
        const int idx = warp_base + r * kWarp + lane;
        const bool ok = (idx < n);
        valid[r] = ok;
        const uint32_t key = ok ? src[idx] : 0u;
        keys[r] = key;
        const unsigned digit = ok ? ((key >> kShift) & 0xffu) : (256u + static_cast<unsigned>(lane));

        unsigned peer_mask = kFullMask;
        #pragma unroll
        for (int bit = 0; bit < 9; ++bit) {
            const unsigned bit_val = (digit >> bit) & 1u;
            const unsigned b_mask = __ballot_sync(kFullMask, bit_val);
            peer_mask &= bit_val ? b_mask : ~b_mask;
        }

        const unsigned peer_rank = static_cast<unsigned>(__popc(peer_mask & ((1u << lane) - 1u)));
        const int leader = __ffs(peer_mask) - 1;
        unsigned warp_base_cnt = 0u;
        if (ok && lane == leader) {
            warp_base_cnt = s_warp_hist[warp][digit];
            s_warp_hist[warp][digit] = warp_base_cnt + static_cast<unsigned>(__popc(peer_mask));
        }
        warp_base_cnt = __shfl_sync(kFullMask, warp_base_cnt, leader);
        ranks[r] = static_cast<uint16_t>(warp_base_cnt + peer_rank);
    }
    __syncthreads();

    uint32_t running = s_bin_base[tid];
    #pragma unroll
    for (int w = 0; w < kWarps; ++w) {
        const uint32_t c = s_warp_hist[w][tid];
        s_warp_hist[w][tid] = running;
        running += c;
    }
    __syncthreads();

    #pragma unroll
    for (int r = 0; r < kItemsPerThread; ++r) {
        if (valid[r]) {
            const unsigned digit = (keys[r] >> kShift) & 0xffu;
            dst[s_warp_hist[warp][digit] + ranks[r]] = keys[r];
        }
    }
}

}  // namespace

// input, output are device pointers
extern "C" void solve(const unsigned int* input, unsigned int* output, int N) {
    if (N <= 0) {
        return;
    }

    const int num_tiles = (N + kTileSize - 1) / kTileSize;
    const unsigned grid_tiles = static_cast<unsigned>(num_tiles);

    baseline_pass_hist_entry<0><<<grid_tiles, kThreads>>>(input, output, N, num_tiles);
    baseline_scan_tile_bins_kernel<<<kBins, kThreads>>>(num_tiles);
    baseline_scan_global_bins_kernel<<<1, kThreads>>>();
    baseline_pass_scatter_entry<0><<<grid_tiles, kThreads>>>(input, output, N, num_tiles);

    baseline_pass_hist_entry<1><<<grid_tiles, kThreads>>>(input, output, N, num_tiles);
    baseline_scan_tile_bins_kernel<<<kBins, kThreads>>>(num_tiles);
    baseline_scan_global_bins_kernel<<<1, kThreads>>>();
    baseline_pass_scatter_entry<1><<<grid_tiles, kThreads>>>(input, output, N, num_tiles);

    baseline_pass_hist_entry<2><<<grid_tiles, kThreads>>>(input, output, N, num_tiles);
    baseline_scan_tile_bins_kernel<<<kBins, kThreads>>>(num_tiles);
    baseline_scan_global_bins_kernel<<<1, kThreads>>>();
    baseline_pass_scatter_entry<2><<<grid_tiles, kThreads>>>(input, output, N, num_tiles);

    baseline_pass_hist_entry<3><<<grid_tiles, kThreads>>>(input, output, N, num_tiles);
    baseline_scan_tile_bins_kernel<<<kBins, kThreads>>>(num_tiles);
    baseline_scan_global_bins_kernel<<<1, kThreads>>>();
    baseline_pass_scatter_entry<3><<<grid_tiles, kThreads>>>(input, output, N, num_tiles);
}
