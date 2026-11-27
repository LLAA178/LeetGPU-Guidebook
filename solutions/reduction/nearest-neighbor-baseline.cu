#include <cuda_runtime.h>

#include <climits>
#include <cmath>

namespace {

constexpr int kThreads = 256;

// Exact non-FMA IEEE-754 round-to-nearest squared Euclidean distance matching
// PyTorch `diff = p_i - p_j; dist_sq = sum(diff * diff, dim=-1)` bit-for-bit.
__device__ __forceinline__ float exact_dist_sq(float x0, float y0, float z0,
                                               float x1, float y1, float z1) {
    const float dx = __fsub_rn(x0, x1);
    const float dy = __fsub_rn(y0, y1);
    const float dz = __fsub_rn(z0, z1);
    const float dx2 = __fmul_rn(dx, dx);
    const float dy2 = __fmul_rn(dy, dy);
    const float dz2 = __fmul_rn(dz, dz);
    return __fadd_rn(__fadd_rn(dx2, dy2), dz2);
}

__global__ __launch_bounds__(kThreads) void nn_baseline_kernel(
    const float* __restrict__ points,
    int* __restrict__ indices,
    int N) {
    const int i = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    if (i >= N) {
        return;
    }
    if (N == 1) {
        indices[0] = 0;
        return;
    }

    const float xi = points[i * 3 + 0];
    const float yi = points[i * 3 + 1];
    const float zi = points[i * 3 + 2];

    float best_d2 = INFINITY;
    int best_j = INT_MAX;

    for (int j = 0; j < N; ++j) {
        if (j == i) {
            continue;
        }
        const float d2 = exact_dist_sq(
            xi, yi, zi,
            points[j * 3 + 0], points[j * 3 + 1], points[j * 3 + 2]);
        if (d2 < best_d2) {
            best_d2 = d2;
            best_j = j;
        }
    }

    indices[i] = best_j;
}

}  // namespace

extern "C" void solve(const float* points, int* indices, int N) {
    if (N <= 0) {
        return;
    }
    const unsigned int blocks = static_cast<unsigned int>((N + kThreads - 1) / kThreads);
    nn_baseline_kernel<<<blocks, kThreads>>>(points, indices, N);
}
