#include <cuda_runtime.h>

namespace {
constexpr int kThreads = 256;
constexpr int kRows = 16;
constexpr int kWarps = kThreads / 32;
constexpr int kTile = kThreads * kRows;
constexpr unsigned kFull = 0xffffffffu;
constexpr unsigned kPartial = 1u << 30;
constexpr unsigned kComplete = 1u << 31;
constexpr unsigned kPayload = kPartial - 1;

__device__ unsigned read_state(const unsigned* address) {
    unsigned value;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
    return value;
}

__device__ void write_state(unsigned* address, unsigned value) {
    asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(address), "r"(value) : "memory");
}

// Each state carries its count and status in one atomic word.
__device__ int tile_base(const unsigned* states, int tile) {
    const int lane = threadIdx.x % 32;
    int sum = 0;
    for (int prior = tile - 1; prior >= 0; prior -= 32) {
        const int index = prior - lane;
        unsigned state;
        unsigned complete;
        unsigned needed;
        do {
            state = index >= 0 ? read_state(states + index) : kComplete;
            complete = __ballot_sync(kFull, state & kComplete);
            const int stop = __ffs(complete);
            // Ignore older states once a complete prefix is available.
            needed = stop && stop < 32 ? (1u << stop) - 1u : kFull;
        } while (__ballot_sync(kFull, state == 0u) & needed);
        if (needed & (1u << lane)) sum += state & kPayload;
        if (complete) break;
    }
    for (int step = 16; step; step /= 2) sum += __shfl_down_sync(kFull, sum, step);
    return sum;
}

__global__ void compact(const float* input, int n, float* output,
                        unsigned* counter, unsigned* states) {
    __shared__ int tile;
    __shared__ int warp_offsets[kWarps];
    __shared__ int global_base;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    // Issue logical tiles to running blocks so earlier tiles can make progress.
    if (threadIdx.x == 0) tile = counter ? atomicAdd(counter, 1u) : 0;
    __syncthreads();

    const int start = tile * kTile + warp * (32 * kRows) + lane;
    float values[kRows];
    unsigned masks[kRows];
    int count = 0;
    #pragma unroll
    for (int row = 0; row < kRows; ++row) {
        const int i = start + row * 32;
        values[row] = i < n ? input[i] : 0.0f;
        masks[row] = __ballot_sync(kFull, values[row] > 0.0f);
        count += __popc(masks[row]);
    }
    if (lane == 0) warp_offsets[warp] = count;
    __syncthreads();

    if (warp == 0) {
        const int value = lane < kWarps ? warp_offsets[lane] : 0;
        int prefix = value;
        for (int step = 1; step < kWarps; step *= 2) {
            const int prev = __shfl_up_sync(kFull, prefix, step);
            if (lane >= step) prefix += prev;
        }
        if (lane < kWarps) warp_offsets[lane] = prefix - value;
        const int tile_count = __shfl_sync(kFull, prefix, kWarps - 1);
        if (states && lane == 0 && tile) write_state(states + tile, kPartial | tile_count);
        const int base = tile_base(states, tile);
        if (lane == 0) {
            global_base = base;
            if (states) write_state(states + tile, kComplete | (base + tile_count));
        }
    }
    __syncthreads();

    int dest = global_base + warp_offsets[warp];
    const unsigned earlier = (1u << lane) - 1;
    #pragma unroll
    for (int row = 0; row < kRows; ++row) {
        if (values[row] > 0.0f) output[dest + __popc(masks[row] & earlier)] = values[row];
        dest += __popc(masks[row]);
    }
}
}  // namespace

extern "C" void solve(const float* A, int N, float* out) {
    const int tiles = (N + kTile - 1) / kTile;
    if (tiles == 1) {
        compact<<<1, kThreads>>>(A, N, out, nullptr, nullptr);
        return;
    }
    unsigned* scratch;
    const size_t bytes = (tiles + 1) * sizeof(unsigned);
    cudaMallocAsync(&scratch, bytes, 0);
    cudaMemsetAsync(scratch, 0, bytes);
    compact<<<tiles, kThreads>>>(A, N, out, scratch, scratch + 1);
    cudaFreeAsync(scratch, 0);
}
