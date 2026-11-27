#include <cuda_runtime.h>
#include <cmath>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxFeatures = 1000;

__device__ double g_base_xtx[kMaxFeatures * kMaxFeatures];
__device__ double g_base_xty[kMaxFeatures];

__global__ __launch_bounds__(kThreads) void ols_base_xtx_kernel(const float* __restrict__ X,
                                                                int n_samples,
                                                                int F) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= F * F) {
        return;
    }
    const int i = idx / F;
    const int j = idx - i * F;
    if (j > i) {
        return;
    }

    double sum = 0.0;
    for (int s = 0; s < n_samples; ++s) {
        const float* row = X + static_cast<size_t>(s) * F;
        sum += static_cast<double>(row[i]) * static_cast<double>(row[j]);
    }
    g_base_xtx[i * F + j] = sum;
    g_base_xtx[j * F + i] = sum;
}

__global__ __launch_bounds__(kThreads) void ols_base_xty_kernel(const float* __restrict__ X,
                                                                const float* __restrict__ y,
                                                                int n_samples,
                                                                int F) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= F) {
        return;
    }

    double sum = 0.0;
    for (int s = 0; s < n_samples; ++s) {
        sum += static_cast<double>(X[static_cast<size_t>(s) * F + i]) * static_cast<double>(y[s]);
    }
    g_base_xty[i] = sum;
}

__global__ __launch_bounds__(kThreads) void ols_base_cholesky_solve_kernel(
    float* __restrict__ beta,
    int F) {
    __shared__ double s_inv_diag;
    __shared__ double s_val;

    const int tid = threadIdx.x;

    for (int k = 0; k < F; ++k) {
        if (tid == 0) {
            double diag = g_base_xtx[k * F + k];
            if (diag < 1.0e-30) diag = 1.0e-30;
            const double l_kk = sqrt(diag);
            const double inv = 1.0 / l_kk;
            g_base_xtx[k * F + k] = l_kk;
            s_inv_diag = inv;
            const double zk = g_base_xty[k] * inv;
            g_base_xty[k] = zk;
            s_val = zk;
        }
        __syncthreads();

        const double inv = s_inv_diag;
        const double zk = s_val;
        for (int i = k + 1 + tid; i < F; i += kThreads) {
            const double l_ik = g_base_xtx[i * F + k] * inv;
            g_base_xtx[i * F + k] = l_ik;
            g_base_xty[i] -= l_ik * zk;
        }
        __syncthreads();

        const int rem = F - (k + 1);
        const int total_pairs = rem * rem;
        for (int p = tid; p < total_pairs; p += kThreads) {
            const int r = p / rem;
            const int c = p - r * rem;
            if (c <= r) {
                const int i = k + 1 + r;
                const int j = k + 1 + c;
                g_base_xtx[i * F + j] -= g_base_xtx[i * F + k] * g_base_xtx[j * F + k];
            }
        }
        __syncthreads();
    }

    for (int k = F - 1; k >= 0; --k) {
        if (tid == 0) {
            const double bk = g_base_xty[k] / g_base_xtx[k * F + k];
            s_val = bk;
            beta[k] = static_cast<float>(bk);
        }
        __syncthreads();

        const double bk = s_val;
        for (int i = tid; i < k; i += kThreads) {
            g_base_xty[i] -= g_base_xtx[k * F + i] * bk;
        }
        __syncthreads();
    }
}

}  // namespace

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    if (n_samples <= 0 || n_features <= 0 || n_features > kMaxFeatures) {
        return;
    }

    const int xtx_blocks = (n_features * n_features + kThreads - 1) / kThreads;
    const int xty_blocks = (n_features + kThreads - 1) / kThreads;

    ols_base_xtx_kernel<<<xtx_blocks, kThreads>>>(X, n_samples, n_features);
    ols_base_xty_kernel<<<xty_blocks, kThreads>>>(X, y, n_samples, n_features);
    ols_base_cholesky_solve_kernel<<<1, kThreads>>>(beta, n_features);
}
