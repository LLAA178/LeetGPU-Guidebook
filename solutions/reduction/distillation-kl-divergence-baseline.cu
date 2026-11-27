#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstddef>

namespace {

constexpr int kBaselineThreads = 256;

__global__ void distillation_kl_3pass_baseline_kernel(
    const float* __restrict__ teacher_logits,
    const float* __restrict__ student_logits,
    float* __restrict__ kl,
    int V,
    float inv_temp) {
    __shared__ float s_a[kBaselineThreads];
    __shared__ float s_b[kBaselineThreads];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const size_t base = static_cast<size_t>(row) * static_cast<size_t>(V);
    const float* __restrict__ t_row = teacher_logits + base;
    const float* __restrict__ s_row = student_logits + base;

    // Pass 1: Find row-wise maxima m_a and m_b separately
    float max_a = -FLT_MAX;
    float max_b = -FLT_MAX;
    for (int j = tid; j < V; j += kBaselineThreads) {
        max_a = fmaxf(max_a, t_row[j] * inv_temp);
        max_b = fmaxf(max_b, s_row[j] * inv_temp);
    }
    s_a[tid] = max_a;
    s_b[tid] = max_b;
    __syncthreads();

    for (int offset = kBaselineThreads / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_a[tid] = fmaxf(s_a[tid], s_a[tid + offset]);
            s_b[tid] = fmaxf(s_b[tid], s_b[tid + offset]);
        }
        __syncthreads();
    }
    const float m_a = s_a[0];
    const float m_b = s_b[0];

    // Pass 2: Sum shifted exponentials d_a and d_b
    float sum_a = 0.0f;
    float sum_b = 0.0f;
    for (int j = tid; j < V; j += kBaselineThreads) {
        sum_a += expf(t_row[j] * inv_temp - m_a);
        sum_b += expf(s_row[j] * inv_temp - m_b);
    }
    s_a[tid] = sum_a;
    s_b[tid] = sum_b;
    __syncthreads();

    for (int offset = kBaselineThreads / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_a[tid] += s_a[tid + offset];
            s_b[tid] += s_b[tid + offset];
        }
        __syncthreads();
    }
    const float log_d_a = logf(s_a[0]);
    const float log_d_b = logf(s_b[0]);
    const float inv_d_a = 1.0f / s_a[0];

    // Pass 3: Accumulate sum_j p_j * (log p_j - log q_j)
    float kl_sum = 0.0f;
    for (int j = tid; j < V; j += kBaselineThreads) {
        const float u = t_row[j] * inv_temp - m_a;
        const float v = s_row[j] * inv_temp - m_b;
        const float log_p = u - log_d_a;
        const float log_q = v - log_d_b;
        kl_sum += (expf(u) * inv_d_a) * (log_p - log_q);
    }
    s_a[tid] = kl_sum;
    __syncthreads();

    for (int offset = kBaselineThreads / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_a[tid] += s_a[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        kl[row] = fmaxf(s_a[0], 0.0f);
    }
}

}  // namespace

// teacher_logits, student_logits, kl are device pointers
extern "C" void solve(const float* teacher_logits, const float* student_logits, float* kl, int N,
                      int V, float temperature) {
    if (N <= 0 || V <= 0) {
        return;
    }

    const float inv_temp = 1.0f / temperature;
    distillation_kl_3pass_baseline_kernel<<<N, kBaselineThreads>>>(
        teacher_logits, student_logits, kl, V, inv_temp);
}
