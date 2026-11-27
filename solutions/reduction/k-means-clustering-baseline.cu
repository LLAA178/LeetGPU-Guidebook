#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kMaxK = 1024;
constexpr int kThreads = 256;

__device__ double g_base_sum_x[kMaxK];
__device__ double g_base_sum_y[kMaxK];
__device__ int g_base_count[kMaxK];

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

__device__ __forceinline__ float exact_dist2d_sq(float x, float y, float cx, float cy) {
    const float dx = __fsub_rn(x, cx);
    const float dy = __fsub_rn(y, cy);
    return __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
}

__global__ __launch_bounds__(kThreads) void baseline_init_centroids_kernel(
    const float* __restrict__ init_cx,
    const float* __restrict__ init_cy,
    float* __restrict__ final_cx,
    float* __restrict__ final_cy,
    int k) {
    const int c = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    if (c < k) {
        final_cx[c] = init_cx[c];
        final_cy[c] = init_cy[c];
        g_base_sum_x[c] = 0.0;
        g_base_sum_y[c] = 0.0;
        g_base_count[c] = 0;
    }
}

__global__ __launch_bounds__(kThreads) void baseline_assign_kernel(
    const float* __restrict__ data_x,
    const float* __restrict__ data_y,
    const float* __restrict__ cx,
    const float* __restrict__ cy,
    int* __restrict__ labels,
    int sample_size,
    int k) {
    const int i = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    if (i >= sample_size) {
        return;
    }

    const float px = data_x[i];
    const float py = data_y[i];

    float best_d2 = INFINITY;
    int best_c = 0;
    for (int c = 0; c < k; ++c) {
        const float d2 = exact_dist2d_sq(px, py, cx[c], cy[c]);
        if (d2 < best_d2) {
            best_d2 = d2;
            best_c = c;
        }
    }

    labels[i] = best_c;
    atomic_add_f64(&g_base_sum_x[best_c], static_cast<double>(px));
    atomic_add_f64(&g_base_sum_y[best_c], static_cast<double>(py));
    atomicAdd(&g_base_count[best_c], 1);
}

__global__ __launch_bounds__(kThreads) void baseline_update_kernel(
    float* __restrict__ final_cx,
    float* __restrict__ final_cy,
    int k) {
    const int c = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    if (c < k) {
        const int cnt = g_base_count[c];
        if (cnt > 0) {
            final_cx[c] = static_cast<float>(g_base_sum_x[c] / static_cast<double>(cnt));
            final_cy[c] = static_cast<float>(g_base_sum_y[c] / static_cast<double>(cnt));
        }
        g_base_sum_x[c] = 0.0;
        g_base_sum_y[c] = 0.0;
        g_base_count[c] = 0;
    }
}

}  // namespace

extern "C" void solve(const float* data_x, const float* data_y, int* labels,
                      float* initial_centroid_x, float* initial_centroid_y,
                      float* final_centroid_x, float* final_centroid_y,
                      int sample_size, int k, int max_iterations) {
    if (sample_size <= 0 || k <= 0) {
        return;
    }

    const unsigned int k_blocks = static_cast<unsigned int>((k + kThreads - 1) / kThreads);
    baseline_init_centroids_kernel<<<k_blocks, kThreads>>>(
        initial_centroid_x, initial_centroid_y, final_centroid_x, final_centroid_y, k);

    if (max_iterations <= 0) {
        return;
    }

    const unsigned int n_blocks = static_cast<unsigned int>((sample_size + kThreads - 1) / kThreads);
    for (int it = 0; it < max_iterations; ++it) {
        baseline_assign_kernel<<<n_blocks, kThreads>>>(
            data_x, data_y, final_centroid_x, final_centroid_y, labels, sample_size, k);
        baseline_update_kernel<<<k_blocks, kThreads>>>(final_centroid_x, final_centroid_y, k);
    }
}
