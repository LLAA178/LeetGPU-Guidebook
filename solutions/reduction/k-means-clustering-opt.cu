#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

// Compile-time mode override for ablation benchmarking:
//   0 = Auto (Cooperative Multi-SM Register Kernel -> Single-Block Register -> Multi-Pass Shared)
//   1 = Force Single-Block <<<1, 1024>>> Register Kernel (for N <= 10,240, k <= 32)
//   2 = Force Multi-Pass Shared-Histogram Kernel
#ifndef KMEANS_FORCE_MODE
#define KMEANS_FORCE_MODE 0
#endif

namespace {

// Official constraints: 1 <= sample_size <= 1,000,000; 1 <= k <= 1,000.
// Total static .bss = ~53 KiB (<< 2 MiB sandbox limit!).
// All kernels strictly use 1-arg `__launch_bounds__(...)` for PTX JIT compatibility!
constexpr int kMaxK = 1024;
#if KMEANS_FORCE_MODE == 0 || KMEANS_FORCE_MODE == 1
constexpr int kSmallK = 32;
#endif
#if KMEANS_FORCE_MODE == 0

constexpr int kMaxCoopBlocks = 40;
constexpr int kCoopThreads = 256;
constexpr int kCoopPtsPerThread = 2;
constexpr int kCoopPtsPerBlock = kCoopThreads * kCoopPtsPerThread;  // 512 points/block -> 20 blocks @ N=10,000!

// Module-scope static device workspace (referenced ONLY inside device kernels!)
__device__ double g_coop_sx[kMaxCoopBlocks * kSmallK];
__device__ double g_coop_sy[kMaxCoopBlocks * kSmallK];
__device__ int g_coop_cnt[kMaxCoopBlocks * kSmallK];
__device__ unsigned int g_bar_arrived;
__device__ int g_bar_sense;
#endif

__device__ double g_large_sx[kMaxK];
__device__ double g_large_sy[kMaxK];
__device__ int g_large_cnt[kMaxK];
__device__ int g_converged;

__device__ __forceinline__ double atomic_add_f64(double* addr, double val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    return atomicAdd(addr, val);
#else
    unsigned long long* addr_ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *addr_ull;
    unsigned long long assumed;
    do {
        assumed = old;
        const double updated = __longlong_as_double(static_cast<long long>(assumed)) + val;
        old = atomicCAS(addr_ull, assumed,
                        static_cast<unsigned long long>(__double_as_longlong(updated)));
    } while (assumed != old);
    return __longlong_as_double(static_cast<long long>(old));
#endif
}

#if KMEANS_FORCE_MODE == 0 || KMEANS_FORCE_MODE == 1
__device__ __forceinline__ double warp_sum_f64(double val) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const int hi = __double2hiint(val);
        const int lo = __double2loint(val);
        const int hi_o = __shfl_down_sync(0xffffffffu, hi, off);
        const int lo_o = __shfl_down_sync(0xffffffffu, lo, off);
        val += __hiloint2double(hi_o, lo_o);
    }
    return val;
}
#endif

__device__ __forceinline__ float exact_dist2d_sq(float x, float y, float cx, float cy) {
    const float dx = __fsub_rn(x, cx);
    const float dy = __fsub_rn(y, cy);
    return __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
}

#if KMEANS_FORCE_MODE == 0
// ============================================================================
// Sense-reversing L2-coherent barrier across `num_blocks` co-resident blocks
// launched via `cudaLaunchCooperativeKernel` (zero <cooperative_groups.h> / rdc needed!).
// ============================================================================
__device__ __forceinline__ void coop_grid_sync(unsigned int num_blocks, int& local_sense) {
    __syncthreads();
    if (num_blocks > 1u) {
        if (threadIdx.x == 0) {
            __threadfence();
            const int next_sense = local_sense ^ 1;
            volatile int* v_sense = &g_bar_sense;
            const unsigned int arrived = atomicAdd(&g_bar_arrived, 1u);
            if (arrived + 1u == num_blocks) {
                g_bar_arrived = 0u;
                __threadfence();
                *v_sense = next_sense;
            } else {
                while (*v_sense != next_sense) {
                }
                __threadfence();
            }
            local_sense = next_sense;
        }
        __syncthreads();
    }
}

// ============================================================================
// Tier 1A: Multi-SM Register-Resident Cooperative Persistent Kernel
//   - Covers `sample_size <= max_coop_blocks * 1,024` (up to `40,960` on T4,
//     including Official Perf `sample_size = 10,000, k = 5, max_iterations = 30`!).
//   - Loads `data_x, data_y` into registers ONCE (zero global point reads across
//     all 30 iterations!), executes ONLY 1 grid sync per iteration by redundantly
//     summing the `<= 40` block partials from coherent L2 in every block, exits
//     the instant centroids converge bitwise, and writes `labels` ONCE at the end!
// ============================================================================
__global__ __launch_bounds__(kCoopThreads) void kmeans_coop_reg_kernel(
    const float* __restrict__ data_x,
    const float* __restrict__ data_y,
    int* __restrict__ labels,
    const float* __restrict__ init_cx,
    const float* __restrict__ init_cy,
    float* __restrict__ final_cx,
    float* __restrict__ final_cy,
    int sample_size,
    int k,
    int max_iterations) {
    __shared__ float s_cx[kSmallK];
    __shared__ float s_cy[kSmallK];
    __shared__ double s_warp_sx[8][kSmallK];  // 2 KiB
    __shared__ double s_warp_sy[8][kSmallK];  // 2 KiB
    __shared__ int s_warp_cnt[8][kSmallK];    // 1 KiB (~5.3 KiB total smem!)
    __shared__ int s_any_changed;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int blk = static_cast<int>(blockIdx.x);
    const unsigned int num_blocks = gridDim.x;
    const int block_base = blk * kCoopPtsPerBlock;

    // Read initial barrier sense BEFORE the first grid sync so all blocks agree
    int local_sense = *const_cast<volatile int*>(&g_bar_sense);

    if (tid < k) {
        s_cx[tid] = __ldg(&init_cx[tid]);
        s_cy[tid] = __ldg(&init_cy[tid]);
    }
    __syncthreads();

    if (max_iterations <= 0) {
        if (blk == 0 && tid < k) {
            final_cx[tid] = s_cx[tid];
            final_cy[tid] = s_cy[tid];
        }
        return;
    }

    // Load up to 4 points per thread into registers ONCE!
    float px[kCoopPtsPerThread];
    float py[kCoopPtsPerThread];
    int lbl[kCoopPtsPerThread];
    #pragma unroll
    for (int r = 0; r < kCoopPtsPerThread; ++r) {
        const int idx = block_base + r * kCoopThreads + tid;
        if (idx < sample_size) {
            px[r] = __ldg(&data_x[idx]);
            py[r] = __ldg(&data_y[idx]);
        } else {
            px[r] = 0.0f;
            py[r] = 0.0f;
        }
        lbl[r] = 0;
    }

    volatile double* v_coop_sx = g_coop_sx;
    volatile double* v_coop_sy = g_coop_sy;
    volatile int* v_coop_cnt = g_coop_cnt;

    for (int it = 0; it < max_iterations; ++it) {
        if (k <= 8) {
            double tx[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            double ty[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            int tc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

            #pragma unroll
            for (int r = 0; r < kCoopPtsPerThread; ++r) {
                const int idx = block_base + r * kCoopThreads + tid;
                if (idx < sample_size) {
                    const float x = px[r];
                    const float y = py[r];
                    float best_d2 = exact_dist2d_sq(x, y, s_cx[0], s_cy[0]);
                    int best_c = 0;
                    for (int c = 1; c < k; ++c) {
                        const float d2 = exact_dist2d_sq(x, y, s_cx[c], s_cy[c]);
                        if (d2 < best_d2) {
                            best_d2 = d2;
                            best_c = c;
                        }
                    }
                    lbl[r] = best_c;
                    switch (best_c) {
                        case 0: tx[0] += static_cast<double>(x); ty[0] += static_cast<double>(y); tc[0] += 1; break;
                        case 1: tx[1] += static_cast<double>(x); ty[1] += static_cast<double>(y); tc[1] += 1; break;
                        case 2: tx[2] += static_cast<double>(x); ty[2] += static_cast<double>(y); tc[2] += 1; break;
                        case 3: tx[3] += static_cast<double>(x); ty[3] += static_cast<double>(y); tc[3] += 1; break;
                        case 4: tx[4] += static_cast<double>(x); ty[4] += static_cast<double>(y); tc[4] += 1; break;
                        case 5: tx[5] += static_cast<double>(x); ty[5] += static_cast<double>(y); tc[5] += 1; break;
                        case 6: tx[6] += static_cast<double>(x); ty[6] += static_cast<double>(y); tc[6] += 1; break;
                        default: tx[7] += static_cast<double>(x); ty[7] += static_cast<double>(y); tc[7] += 1; break;
                    }
                }
            }

            // Uniform warp reduction for each active cluster `0 .. k - 1`
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                if (c < k) {
                    const double wsx = warp_sum_f64(tx[c]);
                    const double wsy = warp_sum_f64(ty[c]);
                    int wcnt = tc[c];
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        wcnt += __shfl_down_sync(0xffffffffu, wcnt, off);
                    }
                    if (lane == 0) {
                        s_warp_sx[warp_id][c] = wsx;
                        s_warp_sy[warp_id][c] = wsy;
                        s_warp_cnt[warp_id][c] = wcnt;
                    }
                }
            }
            __syncthreads();

            if (tid < k) {
                double bsx = 0.0;
                double bsy = 0.0;
                int bcnt = 0;
                #pragma unroll
                for (int w = 0; w < (kCoopThreads / 32); ++w) {
                    bsx += s_warp_sx[w][tid];
                    bsy += s_warp_sy[w][tid];
                    bcnt += s_warp_cnt[w][tid];
                }
                v_coop_sx[blk * kSmallK + tid] = bsx;
                v_coop_sy[blk * kSmallK + tid] = bsy;
                v_coop_cnt[blk * kSmallK + tid] = bcnt;
            }
        } else {
            // `9 <= k <= 32`: per-warp private histogram row in shared memory (zero inter-warp contention)
            for (int idx = tid; idx < (kCoopThreads / 32) * k; idx += kCoopThreads) {
                const int w = idx / k;
                const int c = idx - w * k;
                s_warp_sx[w][c] = 0.0;
                s_warp_sy[w][c] = 0.0;
                s_warp_cnt[w][c] = 0;
            }
            __syncthreads();

            #pragma unroll
            for (int r = 0; r < kCoopPtsPerThread; ++r) {
                const int idx = block_base + r * kCoopThreads + tid;
                if (idx < sample_size) {
                    const float x = px[r];
                    const float y = py[r];
                    float best_d2 = exact_dist2d_sq(x, y, s_cx[0], s_cy[0]);
                    int best_c = 0;
                    for (int c = 1; c < k; ++c) {
                        const float d2 = exact_dist2d_sq(x, y, s_cx[c], s_cy[c]);
                        if (d2 < best_d2) {
                            best_d2 = d2;
                            best_c = c;
                        }
                    }
                    lbl[r] = best_c;
                    atomic_add_f64(&s_warp_sx[warp_id][best_c], static_cast<double>(x));
                    atomic_add_f64(&s_warp_sy[warp_id][best_c], static_cast<double>(y));
                    atomicAdd(&s_warp_cnt[warp_id][best_c], 1);
                }
            }
            __syncthreads();

            if (tid < k) {
                double bsx = 0.0;
                double bsy = 0.0;
                int bcnt = 0;
                #pragma unroll
                for (int w = 0; w < (kCoopThreads / 32); ++w) {
                    bsx += s_warp_sx[w][tid];
                    bsy += s_warp_sy[w][tid];
                    bcnt += s_warp_cnt[w][tid];
                }
                v_coop_sx[blk * kSmallK + tid] = bsx;
                v_coop_sy[blk * kSmallK + tid] = bsy;
                v_coop_cnt[blk * kSmallK + tid] = bcnt;
            }
        }

        // Wait for all `num_blocks` blocks to publish their partial sums to L2
        coop_grid_sync(num_blocks, local_sense);

        // Every block redundantly combines the `num_blocks <= 40` partials from coherent L2!
        if (tid == 0) {
            s_any_changed = 0;
        }
        __syncthreads();

        if (tid < k) {
            double gsx = 0.0;
            double gsy = 0.0;
            int gcnt = 0;
            for (unsigned int b = 0; b < num_blocks; ++b) {
                gsx += v_coop_sx[b * kSmallK + tid];
                gsy += v_coop_sy[b * kSmallK + tid];
                gcnt += v_coop_cnt[b * kSmallK + tid];
            }
            const float old_x = s_cx[tid];
            const float old_y = s_cy[tid];
            const float new_x = (gcnt > 0) ? static_cast<float>(gsx / static_cast<double>(gcnt)) : old_x;
            const float new_y = (gcnt > 0) ? static_cast<float>(gsy / static_cast<double>(gcnt)) : old_y;
            s_cx[tid] = new_x;
            s_cy[tid] = new_y;
            if (__float_as_uint(new_x) != __float_as_uint(old_x) ||
                __float_as_uint(new_y) != __float_as_uint(old_y)) {
                atomicOr(&s_any_changed, 1);
            }
        }
        __syncthreads();

        if (s_any_changed == 0 || it + 1 == max_iterations) {
            break;
        }

        // Ensure all blocks finished reading `v_coop_*` before next iteration overwrites it
        coop_grid_sync(num_blocks, local_sense);
    }

    // Write final labels ONCE from registers and final centroids from block 0!
    #pragma unroll
    for (int r = 0; r < kCoopPtsPerThread; ++r) {
        const int idx = block_base + r * kCoopThreads + tid;
        if (idx < sample_size) {
            labels[idx] = lbl[r];
        }
    }
    if (blk == 0 && tid < k) {
        final_cx[tid] = s_cx[tid];
        final_cy[tid] = s_cy[tid];
    }
}
#endif  // KMEANS_FORCE_MODE == 0

#if KMEANS_FORCE_MODE == 0 || KMEANS_FORCE_MODE == 1
// ============================================================================
// Tier 1B: Portable Single-Block `<<<1, 1024>>>` Register-Resident Kernel
//   - Covers `sample_size <= 10,240 && k <= 32` (`10` points/thread in registers),
//     using ONLY `__syncthreads()` (zero cooperative launch requirement!).
// ============================================================================
constexpr int kSingleThreads = 1024;
constexpr int kSinglePtsPerThread = 10;

__global__ __launch_bounds__(kSingleThreads) void kmeans_single_block_reg_kernel(
    const float* __restrict__ data_x,
    const float* __restrict__ data_y,
    int* __restrict__ labels,
    const float* __restrict__ init_cx,
    const float* __restrict__ init_cy,
    float* __restrict__ final_cx,
    float* __restrict__ final_cy,
    int sample_size,
    int k,
    int max_iterations) {
    __shared__ float s_cx[kSmallK];
    __shared__ float s_cy[kSmallK];
    __shared__ double s_warp_sx[32][kSmallK];  // 8 KiB
    __shared__ double s_warp_sy[32][kSmallK];  // 8 KiB
    __shared__ int s_warp_cnt[32][kSmallK];    // 4 KiB (20.25 KiB total smem!)
    __shared__ int s_any_changed;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    if (tid < k) {
        s_cx[tid] = __ldg(&init_cx[tid]);
        s_cy[tid] = __ldg(&init_cy[tid]);
    }
    __syncthreads();

    if (max_iterations <= 0) {
        if (tid < k) {
            final_cx[tid] = s_cx[tid];
            final_cy[tid] = s_cy[tid];
        }
        return;
    }

    float px[kSinglePtsPerThread];
    float py[kSinglePtsPerThread];
    int lbl[kSinglePtsPerThread];
    #pragma unroll
    for (int r = 0; r < kSinglePtsPerThread; ++r) {
        const int idx = r * kSingleThreads + tid;
        if (idx < sample_size) {
            px[r] = __ldg(&data_x[idx]);
            py[r] = __ldg(&data_y[idx]);
        } else {
            px[r] = 0.0f;
            py[r] = 0.0f;
        }
        lbl[r] = 0;
    }

    for (int it = 0; it < max_iterations; ++it) {
        for (int idx = tid; idx < 32 * k; idx += kSingleThreads) {
            const int w = idx / k;
            const int c = idx - w * k;
            s_warp_sx[w][c] = 0.0;
            s_warp_sy[w][c] = 0.0;
            s_warp_cnt[w][c] = 0;
        }
        if (tid == 0) {
            s_any_changed = 0;
        }
        __syncthreads();

        if (k <= 8) {
            double tx[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            double ty[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            int tc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

            #pragma unroll
            for (int r = 0; r < kSinglePtsPerThread; ++r) {
                const int idx = r * kSingleThreads + tid;
                if (idx < sample_size) {
                    const float x = px[r];
                    const float y = py[r];
                    float best_d2 = exact_dist2d_sq(x, y, s_cx[0], s_cy[0]);
                    int best_c = 0;
                    for (int c = 1; c < k; ++c) {
                        const float d2 = exact_dist2d_sq(x, y, s_cx[c], s_cy[c]);
                        if (d2 < best_d2) {
                            best_d2 = d2;
                            best_c = c;
                        }
                    }
                    lbl[r] = best_c;
                    switch (best_c) {
                        case 0: tx[0] += static_cast<double>(x); ty[0] += static_cast<double>(y); tc[0] += 1; break;
                        case 1: tx[1] += static_cast<double>(x); ty[1] += static_cast<double>(y); tc[1] += 1; break;
                        case 2: tx[2] += static_cast<double>(x); ty[2] += static_cast<double>(y); tc[2] += 1; break;
                        case 3: tx[3] += static_cast<double>(x); ty[3] += static_cast<double>(y); tc[3] += 1; break;
                        case 4: tx[4] += static_cast<double>(x); ty[4] += static_cast<double>(y); tc[4] += 1; break;
                        case 5: tx[5] += static_cast<double>(x); ty[5] += static_cast<double>(y); tc[5] += 1; break;
                        case 6: tx[6] += static_cast<double>(x); ty[6] += static_cast<double>(y); tc[6] += 1; break;
                        default: tx[7] += static_cast<double>(x); ty[7] += static_cast<double>(y); tc[7] += 1; break;
                    }
                }
            }

            for (int c = 0; c < k; ++c) {
                const double wsx = warp_sum_f64(tx[c]);
                const double wsy = warp_sum_f64(ty[c]);
                int wcnt = tc[c];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    wcnt += __shfl_down_sync(0xffffffffu, wcnt, off);
                }
                if (lane == 0) {
                    s_warp_sx[warp_id][c] = wsx;
                    s_warp_sy[warp_id][c] = wsy;
                    s_warp_cnt[warp_id][c] = wcnt;
                }
            }
        } else {
            #pragma unroll
            for (int r = 0; r < kSinglePtsPerThread; ++r) {
                const int idx = r * kSingleThreads + tid;
                if (idx < sample_size) {
                    const float x = px[r];
                    const float y = py[r];
                    float best_d2 = exact_dist2d_sq(x, y, s_cx[0], s_cy[0]);
                    int best_c = 0;
                    for (int c = 1; c < k; ++c) {
                        const float d2 = exact_dist2d_sq(x, y, s_cx[c], s_cy[c]);
                        if (d2 < best_d2) {
                            best_d2 = d2;
                            best_c = c;
                        }
                    }
                    lbl[r] = best_c;
                    atomic_add_f64(&s_warp_sx[warp_id][best_c], static_cast<double>(x));
                    atomic_add_f64(&s_warp_sy[warp_id][best_c], static_cast<double>(y));
                    atomicAdd(&s_warp_cnt[warp_id][best_c], 1);
                }
            }
        }
        __syncthreads();

        if (tid < k) {
            double gsx = 0.0;
            double gsy = 0.0;
            int gcnt = 0;
            #pragma unroll
            for (int w = 0; w < 32; ++w) {
                gsx += s_warp_sx[w][tid];
                gsy += s_warp_sy[w][tid];
                gcnt += s_warp_cnt[w][tid];
            }
            const float old_x = s_cx[tid];
            const float old_y = s_cy[tid];
            const float new_x = (gcnt > 0) ? static_cast<float>(gsx / static_cast<double>(gcnt)) : old_x;
            const float new_y = (gcnt > 0) ? static_cast<float>(gsy / static_cast<double>(gcnt)) : old_y;
            s_cx[tid] = new_x;
            s_cy[tid] = new_y;
            if (__float_as_uint(new_x) != __float_as_uint(old_x) ||
                __float_as_uint(new_y) != __float_as_uint(old_y)) {
                atomicOr(&s_any_changed, 1);
            }
        }
        __syncthreads();

        if (s_any_changed == 0) {
            break;
        }
    }

    #pragma unroll
    for (int r = 0; r < kSinglePtsPerThread; ++r) {
        const int idx = r * kSingleThreads + tid;
        if (idx < sample_size) {
            labels[idx] = lbl[r];
        }
    }
    if (tid < k) {
        final_cx[tid] = s_cx[tid];
        final_cy[tid] = s_cy[tid];
    }
}
#endif  // KMEANS_FORCE_MODE == 0 || KMEANS_FORCE_MODE == 1

// ============================================================================
// Tier 2: General Multi-Pass Shared-Memory Tiled & Histogrammed Kernels
//   - Covers ALL `1 <= sample_size <= 1,000,000` and `1 <= k <= 1,000`!
//   - Device-side `g_converged` flag turns post-convergence iterations into
//     instant no-ops with zero `cudaDeviceSynchronize()` host stalls!
// ============================================================================
__global__ __launch_bounds__(256) void kmeans_large_init_kernel(
    const float* __restrict__ init_cx,
    const float* __restrict__ init_cy,
    float* __restrict__ final_cx,
    float* __restrict__ final_cy,
    int k) {
    const int tid = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
    if (tid == 0) {
        g_converged = 0;
    }
    if (tid < k) {
        final_cx[tid] = __ldg(&init_cx[tid]);
        final_cy[tid] = __ldg(&init_cy[tid]);
        g_large_sx[tid] = 0.0;
        g_large_sy[tid] = 0.0;
        g_large_cnt[tid] = 0;
    }
}

__global__ __launch_bounds__(256) void kmeans_large_assign_kernel(
    const float* __restrict__ data_x,
    const float* __restrict__ data_y,
    const float* __restrict__ cx,
    const float* __restrict__ cy,
    int* __restrict__ labels,
    int sample_size,
    int k) {
    if (g_converged != 0) {
        return;
    }

    __shared__ float s_cx[256];
    __shared__ float s_cy[256];
    __shared__ double s_sx[256];
    __shared__ double s_sy[256];
    __shared__ int s_cnt[256];

    const int tid = static_cast<int>(threadIdx.x);
    const bool use_smem_hist = (k <= 256);

    if (use_smem_hist) {
        if (tid < k) {
            s_cx[tid] = __ldg(&cx[tid]);
            s_cy[tid] = __ldg(&cy[tid]);
            s_sx[tid] = 0.0;
            s_sy[tid] = 0.0;
            s_cnt[tid] = 0;
        }
        __syncthreads();

        const int stride = static_cast<int>(gridDim.x) * 256;
        for (int i = static_cast<int>(blockIdx.x) * 256 + tid; i < sample_size; i += stride) {
            const float px = __ldg(&data_x[i]);
            const float py = __ldg(&data_y[i]);
            float best_d2 = exact_dist2d_sq(px, py, s_cx[0], s_cy[0]);
            int best_c = 0;
            for (int c = 1; c < k; ++c) {
                const float d2 = exact_dist2d_sq(px, py, s_cx[c], s_cy[c]);
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_c = c;
                }
            }
            labels[i] = best_c;
            atomic_add_f64(&s_sx[best_c], static_cast<double>(px));
            atomic_add_f64(&s_sy[best_c], static_cast<double>(py));
            atomicAdd(&s_cnt[best_c], 1);
        }
        __syncthreads();

        if (tid < k && s_cnt[tid] > 0) {
            atomic_add_f64(&g_large_sx[tid], s_sx[tid]);
            atomic_add_f64(&g_large_sy[tid], s_sy[tid]);
            atomicAdd(&g_large_cnt[tid], s_cnt[tid]);
        }
    } else {
        // `256 < k <= 1,000`: tile centroids through shared memory in chunks of 256!
        const int i = static_cast<int>(blockIdx.x) * 256 + tid;
        float px = 0.0f, py = 0.0f;
        if (i < sample_size) {
            px = __ldg(&data_x[i]);
            py = __ldg(&data_y[i]);
        }

        float best_d2 = INFINITY;
        int best_c = 0;

        for (int c_base = 0; c_base < k; c_base += 256) {
            const int chunk = ((k - c_base) < 256) ? (k - c_base) : 256;
            if (tid < chunk) {
                s_cx[tid] = __ldg(&cx[c_base + tid]);
                s_cy[tid] = __ldg(&cy[c_base + tid]);
            }
            __syncthreads();

            if (i < sample_size) {
                for (int c = 0; c < chunk; ++c) {
                    const float d2 = exact_dist2d_sq(px, py, s_cx[c], s_cy[c]);
                    if (d2 < best_d2) {
                        best_d2 = d2;
                        best_c = c_base + c;
                    }
                }
            }
            __syncthreads();
        }

        if (i < sample_size) {
            labels[i] = best_c;
            atomic_add_f64(&g_large_sx[best_c], static_cast<double>(px));
            atomic_add_f64(&g_large_sy[best_c], static_cast<double>(py));
            atomicAdd(&g_large_cnt[best_c], 1);
        }
    }
}

// Single-block centroid update (`k <= 1,000`) + exact bitwise convergence check!
__global__ __launch_bounds__(256) void kmeans_large_update_kernel(
    float* __restrict__ final_cx,
    float* __restrict__ final_cy,
    int k) {
    if (g_converged != 0) {
        return;
    }

    __shared__ int s_changed;
    const int tid = static_cast<int>(threadIdx.x);
    if (tid == 0) {
        s_changed = 0;
    }
    __syncthreads();

    int local_changed = 0;
    for (int c = tid; c < k; c += 256) {
        const int cnt = g_large_cnt[c];
        if (cnt > 0) {
            const float old_x = final_cx[c];
            const float old_y = final_cy[c];
            const float new_x = static_cast<float>(g_large_sx[c] / static_cast<double>(cnt));
            const float new_y = static_cast<float>(g_large_sy[c] / static_cast<double>(cnt));
            final_cx[c] = new_x;
            final_cy[c] = new_y;
            if (__float_as_uint(new_x) != __float_as_uint(old_x) ||
                __float_as_uint(new_y) != __float_as_uint(old_y)) {
                local_changed = 1;
            }
        }
        g_large_sx[c] = 0.0;
        g_large_sy[c] = 0.0;
        g_large_cnt[c] = 0;
    }

    if (local_changed != 0) {
        atomicOr(&s_changed, 1);
    }
    __syncthreads();

    if (tid == 0 && s_changed == 0) {
        g_converged = 1;
    }
}

#if KMEANS_FORCE_MODE == 0
int get_max_coop_blocks() {
    static int cached = -1;
    if (cached >= 0) {
        return cached;
    }
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        cached = 0;
        return 0;
    }
    int coop_supp = 0;
    if (cudaDeviceGetAttribute(&coop_supp, cudaDevAttrCooperativeLaunch, dev) != cudaSuccess ||
        coop_supp == 0) {
        cached = 0;
        return 0;
    }
    int num_sms = 0;
    if (cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
        num_sms <= 0) {
        cached = 0;
        return 0;
    }
    int blocks_per_sm = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, kmeans_coop_reg_kernel, kCoopThreads, 0) != cudaSuccess ||
        blocks_per_sm <= 0) {
        cached = 0;
        return 0;
    }
    int max_b = num_sms * blocks_per_sm;
    if (max_b > kMaxCoopBlocks) {
        max_b = kMaxCoopBlocks;
    }
    cached = max_b;
    return cached;
}
#endif

}  // namespace

extern "C" void solve(const float* data_x, const float* data_y, int* labels,
                      float* initial_centroid_x, float* initial_centroid_y,
                      float* final_centroid_x, float* final_centroid_y,
                      int sample_size, int k, int max_iterations) {
    if (sample_size <= 0 || k <= 0) {
        return;
    }

#if KMEANS_FORCE_MODE == 0
    if (k <= kSmallK) {
        const int max_coop_blocks = get_max_coop_blocks();
        const int needed_blocks = (sample_size + kCoopPtsPerBlock - 1) / kCoopPtsPerBlock;
        if (max_coop_blocks > 0 && needed_blocks <= max_coop_blocks) {
            const float* p_dx = data_x;
            const float* p_dy = data_y;
            int* p_lbl = labels;
            const float* p_icx = initial_centroid_x;
            const float* p_icy = initial_centroid_y;
            float* p_fcx = final_centroid_x;
            float* p_fcy = final_centroid_y;
            int s_n = sample_size;
            int s_k = k;
            int s_it = max_iterations;
            void* args[] = {
                &p_dx, &p_dy, &p_lbl, &p_icx, &p_icy, &p_fcx, &p_fcy, &s_n, &s_k, &s_it
            };
            const dim3 grid(static_cast<unsigned int>(needed_blocks));
            const dim3 block(static_cast<unsigned int>(kCoopThreads));
            if (cudaLaunchCooperativeKernel(
                    reinterpret_cast<void*>(kmeans_coop_reg_kernel), grid, block, args, 0, nullptr) ==
                cudaSuccess) {
                return;
            }
        }
    }
#endif

#if KMEANS_FORCE_MODE == 0 || KMEANS_FORCE_MODE == 1
    if (k <= kSmallK && sample_size <= kSingleThreads * kSinglePtsPerThread) {
        kmeans_single_block_reg_kernel<<<1, kSingleThreads>>>(
            data_x, data_y, labels,
            initial_centroid_x, initial_centroid_y,
            final_centroid_x, final_centroid_y,
            sample_size, k, max_iterations);
        return;
    }
#endif

    // General Multi-Pass Shared-Memory Path (`1 <= sample_size <= 1,000,000`, `1 <= k <= 1,000`)
    const unsigned int init_blocks = static_cast<unsigned int>((k + 255) / 256);
    kmeans_large_init_kernel<<<init_blocks, 256>>>(
        initial_centroid_x, initial_centroid_y, final_centroid_x, final_centroid_y, k);

    if (max_iterations <= 0) {
        return;
    }

    unsigned int assign_blocks = static_cast<unsigned int>((sample_size + 255) / 256);
    if (k <= 256 && assign_blocks > 512u) {
        assign_blocks = 512u;  // Grid-stride amortizes shared-memory histogram flushes
    }

    for (int it = 0; it < max_iterations; ++it) {
        kmeans_large_assign_kernel<<<assign_blocks, 256>>>(
            data_x, data_y, final_centroid_x, final_centroid_y, labels, sample_size, k);
        kmeans_large_update_kernel<<<1, 256>>>(final_centroid_x, final_centroid_y, k);
    }
}
