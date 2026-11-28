#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;
constexpr float kRadiusSq = 25.0f;
constexpr float kAlpha = 0.05f;

// Exact IEEE-754 non-FMA squared distance so boundary `dist_sq == 25.0f` (e.g., (3, 4))
// matches PyTorch `(diff**2).sum(dim=2) < 25.0` bit-for-bit.
__device__ __forceinline__ float exact_dist2d_sq(float x0, float y0, float x1, float y1) {
    const float dx = __fsub_rn(x0, x1);
    const float dy = __fsub_rn(y0, y1);
    return __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
}

__global__ __launch_bounds__(kThreads) void boids_baseline_kernel(
    const float* __restrict__ agents,
    float* __restrict__ agents_next,
    int N) {
    const int i = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    if (i >= N) {
        return;
    }

    const float px = agents[i * 4 + 0];
    const float py = agents[i * 4 + 1];
    const float vx = agents[i * 4 + 2];
    const float vy = agents[i * 4 + 3];

    double sum_vx = 0.0;
    double sum_vy = 0.0;
    int cnt = 0;

    for (int j = 0; j < N; ++j) {
        if (j == i) {
            continue;
        }
        const float qx = agents[j * 4 + 0];
        const float qy = agents[j * 4 + 1];
        if (exact_dist2d_sq(px, py, qx, qy) < kRadiusSq) {
            sum_vx += static_cast<double>(agents[j * 4 + 2]);
            sum_vy += static_cast<double>(agents[j * 4 + 3]);
            ++cnt;
        }
    }

    float new_vx = vx;
    float new_vy = vy;
    if (cnt > 0) {
        const double inv_cnt = 1.0 / static_cast<double>(cnt);
        const float avg_vx = static_cast<float>(sum_vx * inv_cnt);
        const float avg_vy = static_cast<float>(sum_vy * inv_cnt);
        new_vx = vx + kAlpha * (avg_vx - vx);
        new_vy = vy + kAlpha * (avg_vy - vy);
    }

    agents_next[i * 4 + 0] = px + new_vx;
    agents_next[i * 4 + 1] = py + new_vy;
    agents_next[i * 4 + 2] = new_vx;
    agents_next[i * 4 + 3] = new_vy;
}

}  // namespace

extern "C" void solve(const float* agents, float* agents_next, int N) {
    if (N <= 0) {
        return;
    }
    const unsigned int blocks = static_cast<unsigned int>((N + kThreads - 1) / kThreads);
    boids_baseline_kernel<<<blocks, kThreads>>>(agents, agents_next, N);
}
