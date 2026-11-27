#include <cuda_runtime.h>
#include <cmath>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxState = 64;

__global__ __launch_bounds__(kThreads) void ssm_selective_scan_baseline_kernel(
    const float* __restrict__ u,
    const float* __restrict__ delta,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    const float* __restrict__ skip,
    float* __restrict__ y,
    int batch,
    int seq_len,
    int d_model,
    int d_state) {
    const int bd = blockIdx.x * kThreads + threadIdx.x;
    if (bd >= batch * d_model) {
        return;
    }

    const int b = bd / d_model;
    const int d = bd - b * d_model;

    float h[kMaxState];
    for (int n = 0; n < d_state; ++n) {
        h[n] = 0.0f;
    }

    const float skip_d = skip[d];
    const float* __restrict__ a_row = A + static_cast<size_t>(d) * d_state;

    for (int t = 0; t < seq_len; ++t) {
        const size_t bt = static_cast<size_t>(b) * seq_len + t;
        const size_t btd = bt * d_model + d;
        const float* __restrict__ b_row = B + bt * d_state;
        const float* __restrict__ c_row = C + bt * d_state;

        const float u_val = u[btd];
        const float dt = delta[btd];

        float y_acc = 0.0f;
        for (int n = 0; n < d_state; ++n) {
            const float a_bar = expf(dt * a_row[n]);
            const float b_bar = dt * b_row[n];
            h[n] = a_bar * h[n] + b_bar * u_val;
            y_acc += c_row[n] * h[n];
        }

        y[btd] = y_acc + skip_d * u_val;
    }
}

}  // namespace

extern "C" void solve(const float* u,
                      const float* delta,
                      const float* A,
                      const float* B,
                      const float* C,
                      const float* skip,
                      float* y,
                      int batch,
                      int seq_len,
                      int d_model,
                      int d_state) {
    if (batch <= 0 || seq_len <= 0 || d_model <= 0 || d_state <= 0) {
        return;
    }

    const int total_channels = batch * d_model;
    const int blocks = (total_channels + kThreads - 1) / kThreads;
    ssm_selective_scan_baseline_kernel<<<blocks, kThreads>>>(
        u, delta, A, B, C, skip, y, batch, seq_len, d_model, d_state);
}
