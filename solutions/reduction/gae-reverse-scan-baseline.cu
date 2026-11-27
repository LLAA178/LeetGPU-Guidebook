#include <cuda_runtime.h>

namespace {

constexpr int kBaselineThreads = 64;

__global__ void gae_reverse_scan_baseline_kernel(
    const float* __restrict__ rewards,
    const float* __restrict__ values,
    float* __restrict__ advantages,
    float gamma,
    float decay,
    int B,
    int S) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) {
        return;
    }

    const int row_offset = b * S;
    const float* __restrict__ row_r = rewards + row_offset;
    const float* __restrict__ row_v = values + row_offset;
    float* __restrict__ row_adv = advantages + row_offset;

    float next_v = 0.0f;
    float last_gae = 0.0f;
    for (int t = S - 1; t >= 0; --t) {
        const float v_t = row_v[t];
        const float delta = fmaf(gamma, next_v, row_r[t] - v_t);
        last_gae = fmaf(decay, last_gae, delta);
        row_adv[t] = last_gae;
        next_v = v_t;
    }
}

}  // namespace

// rewards, values, advantages are device pointers
extern "C" void solve(const float* rewards, const float* values, float* advantages, float gamma,
                      float lam, int B, int S) {
    if (B <= 0 || S <= 0) {
        return;
    }

    const float decay = static_cast<float>(static_cast<double>(gamma) * static_cast<double>(lam));
    const int blocks = (B + kBaselineThreads - 1) / kBaselineThreads;
    gae_reverse_scan_baseline_kernel<<<blocks, kBaselineThreads>>>(
        rewards, values, advantages, gamma, decay, B, S);
}
