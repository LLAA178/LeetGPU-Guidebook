#include <cuda_runtime.h>
#include <cmath>

namespace {

constexpr int kThreads = 256;

__global__ void token_embedding_baseline_kernel(const int* __restrict__ token_ids,
                                                const int* __restrict__ position_ids,
                                                const float* __restrict__ token_embeddings,
                                                const float* __restrict__ position_embeddings,
                                                const float* __restrict__ gamma,
                                                const float* __restrict__ beta,
                                                float* __restrict__ output,
                                                int T,
                                                int D,
                                                float eps) {
    __shared__ float s_reduce[kThreads];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int t = row % T;

    const int tok_id = token_ids[row];
    const int pos_id = position_ids[t];

    const float* tok_row = token_embeddings + static_cast<size_t>(tok_id) * static_cast<size_t>(D);
    const float* pos_row = position_embeddings + static_cast<size_t>(pos_id) * static_cast<size_t>(D);
    float* out_row = output + static_cast<size_t>(row) * static_cast<size_t>(D);

    const float ref = tok_row[0] + pos_row[0];
    float local_sum = 0.0f;
    for (int d = tid; d < D; d += kThreads) {
        local_sum += (tok_row[d] + pos_row[d]) - ref;
    }
    s_reduce[tid] = local_sum;
    __syncthreads();

    for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_reduce[tid] += s_reduce[tid + offset];
        }
        __syncthreads();
    }
    const float mean = static_cast<float>(
        static_cast<double>(ref) + static_cast<double>(s_reduce[0]) / static_cast<double>(D));
    __syncthreads();

    float local_sq = 0.0f;
    for (int d = tid; d < D; d += kThreads) {
        const float diff = (tok_row[d] + pos_row[d]) - mean;
        local_sq += diff * diff;
    }
    s_reduce[tid] = local_sq;
    __syncthreads();

    for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_reduce[tid] += s_reduce[tid + offset];
        }
        __syncthreads();
    }
    const float inv_std = rsqrtf(s_reduce[0] / static_cast<float>(D) + eps);

    for (int d = tid; d < D; d += kThreads) {
        const float s = tok_row[d] + pos_row[d];
        out_row[d] = ((s - mean) * inv_std) * gamma[d] + beta[d];
    }
}

}  // namespace

// token_ids, position_ids, token_embeddings, position_embeddings, gamma, beta, output are device pointers
extern "C" void solve(const int* token_ids,
                      const int* position_ids,
                      const float* token_embeddings,
                      const float* position_embeddings,
                      const float* gamma,
                      const float* beta,
                      float* output,
                      int B,
                      int T,
                      int V,
                      int P,
                      int D,
                      float eps) {
    (void)V;
    (void)P;
    if (B <= 0 || T <= 0 || D <= 0) {
        return;
    }

    const int total_tokens = B * T;
    token_embedding_baseline_kernel<<<total_tokens, kThreads>>>(
        token_ids, position_ids, token_embeddings, position_embeddings,
        gamma, beta, output, T, D, eps);
}
