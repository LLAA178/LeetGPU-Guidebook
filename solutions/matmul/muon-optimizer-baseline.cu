#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

namespace {

constexpr int kMaxDim = 1024;
constexpr int kMaxElems = kMaxDim * kMaxDim;
constexpr int kTile = 16;

constexpr float kCoefA = 3.4445f;
constexpr float kCoefB = -4.7750f;
constexpr float kCoefC = 2.0315f;

__device__ float g_base_V[kMaxElems];
__device__ float g_base_X0[kMaxElems];
__device__ float g_base_X1[kMaxElems];
__device__ float g_base_A[kMaxElems];
__device__ float g_base_B[kMaxElems];
__device__ double g_base_norm_partials[256];
__device__ float g_base_inv_norm;

__global__ __launch_bounds__(256) void baseline_combine_partial_norm_kernel(
    const float* __restrict__ G,
    const float* __restrict__ Buf,
    int total,
    float momentum) {
    __shared__ double s_sum[256];
    const int tid = threadIdx.x;
    const int gid = static_cast<int>(blockIdx.x) * 256 + tid;
    const int stride = static_cast<int>(gridDim.x) * 256;

    double acc = 0.0;
    for (int i = gid; i < total; i += stride) {
        const float v = fmaf(momentum, Buf[i], G[i]);
        g_base_V[i] = v;
        acc += static_cast<double>(v) * static_cast<double>(v);
    }
    s_sum[tid] = acc;
    __syncthreads();

    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) {
            s_sum[tid] += s_sum[tid + off];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_base_norm_partials[blockIdx.x] = s_sum[0];
    }
}

__global__ __launch_bounds__(256) void baseline_finalize_norm_kernel(int num_blocks) {
    __shared__ double s_sum[256];
    const int tid = threadIdx.x;
    double acc = 0.0;
    for (int i = tid; i < num_blocks; i += 256) {
        acc += g_base_norm_partials[i];
    }
    s_sum[tid] = acc;
    __syncthreads();

    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) {
            s_sum[tid] += s_sum[tid + off];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const float norm_f = static_cast<float>(sqrt(s_sum[0]));
        g_base_inv_norm = 1.0f / (norm_f + 1e-7f);
    }
}

__global__ __launch_bounds__(256) void baseline_normalize_init_kernel(
    int M,
    int N,
    bool transposed) {
    const int idx = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
    const int total = M * N;
    if (idx >= total) {
        return;
    }
    const float inv_n = g_base_inv_norm;
    const float val = g_base_V[idx] * inv_n;
    if (!transposed) {
        g_base_X0[idx] = val;
    } else {
        const int r = idx / N;
        const int c = idx - r * N;
        g_base_X0[c * M + r] = val;
    }
}

// Full R x R SGEMM: A = X @ X^T (`X` has shape [R, C])
__global__ __launch_bounds__(256) void baseline_gram_kernel(int R, int C, int src_slot) {
    __shared__ float sL[kTile][kTile];
    __shared__ float sR[kTile][kTile];

    const float* X = (src_slot == 0) ? g_base_X0 : g_base_X1;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = static_cast<int>(blockIdx.y) * kTile + ty;
    const int col = static_cast<int>(blockIdx.x) * kTile + tx;

    float sum = 0.0f;
    for (int kt = 0; kt < C; kt += kTile) {
        sL[ty][tx] = (row < R && kt + tx < C) ? X[row * C + (kt + tx)] : 0.0f;
        const int r_col = static_cast<int>(blockIdx.x) * kTile + ty;
        sR[ty][tx] = (r_col < R && kt + tx < C) ? X[r_col * C + (kt + tx)] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < kTile; ++k) {
            sum = fmaf(sL[ty][k], sR[tx][k], sum);
        }
        __syncthreads();
    }

    if (row < R && col < R) {
        g_base_A[row * R + col] = sum;
    }
}

// Full R x R SGEMM: B = b * A + c * (A @ A)
__global__ __launch_bounds__(256) void baseline_polyB_kernel(int R) {
    __shared__ float sL[kTile][kTile];
    __shared__ float sR[kTile][kTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = static_cast<int>(blockIdx.y) * kTile + ty;
    const int col = static_cast<int>(blockIdx.x) * kTile + tx;

    float sum = 0.0f;
    for (int kt = 0; kt < R; kt += kTile) {
        sL[ty][tx] = (row < R && kt + tx < R) ? g_base_A[row * R + (kt + tx)] : 0.0f;
        sR[ty][tx] = (kt + ty < R && col < R) ? g_base_A[(kt + ty) * R + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < kTile; ++k) {
            sum = fmaf(sL[ty][k], sR[k][tx], sum);
        }
        __syncthreads();
    }

    if (row < R && col < R) {
        g_base_B[row * R + col] = fmaf(kCoefB, g_base_A[row * R + col], kCoefC * sum);
    }
}

// Full R x C SGEMM: X_dst = a * X_src + B @ X_src
__global__ __launch_bounds__(256) void baseline_updateX_kernel(int R, int C, int src_slot) {
    __shared__ float sB[kTile][kTile];
    __shared__ float sX[kTile][kTile];

    const float* X_src = (src_slot == 0) ? g_base_X0 : g_base_X1;
    float* X_dst = (src_slot == 0) ? g_base_X1 : g_base_X0;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = static_cast<int>(blockIdx.y) * kTile + ty;
    const int col = static_cast<int>(blockIdx.x) * kTile + tx;

    float sum = 0.0f;
    for (int kt = 0; kt < R; kt += kTile) {
        sB[ty][tx] = (row < R && kt + tx < R) ? g_base_B[row * R + (kt + tx)] : 0.0f;
        sX[ty][tx] = (kt + ty < R && col < C) ? X_src[(kt + ty) * C + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < kTile; ++k) {
            sum = fmaf(sB[ty][k], sX[k][tx], sum);
        }
        __syncthreads();
    }

    if (row < R && col < C) {
        X_dst[row * C + col] = fmaf(kCoefA, X_src[row * C + col], sum);
    }
}

__global__ __launch_bounds__(256) void baseline_scale_output_kernel(
    float* __restrict__ output,
    int M,
    int N,
    bool transposed,
    float scale) {
    const int idx = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
    const int total = M * N;
    if (idx >= total) {
        return;
    }
    // After 5 steps (0,1,2,3,4), final state sits in g_base_X1 (`src_slot` flipped 5 times: 0->1)
    if (!transposed) {
        output[idx] = g_base_X1[idx] * scale;
    } else {
        const int r = idx / N;
        const int c = idx - r * N;
        output[idx] = g_base_X1[c * M + r] * scale;
    }
}

}  // namespace

extern "C" void solve(const float* G, const float* Buf, float* output,
                      int M, int N, float momentum) {
    if (M <= 0 || N <= 0) {
        return;
    }

    const int total = M * N;
    int norm_blocks = (total + 255) / 256;
    if (norm_blocks > 256) {
        norm_blocks = 256;
    }

    baseline_combine_partial_norm_kernel<<<norm_blocks, 256>>>(G, Buf, total, momentum);
    baseline_finalize_norm_kernel<<<1, 256>>>(norm_blocks);

    const bool transposed = (M > N);
    const int R = transposed ? N : M;
    const int C = transposed ? M : N;

    const int elem_blocks = (total + 255) / 256;
    baseline_normalize_init_kernel<<<elem_blocks, 256>>>(M, N, transposed);

    const dim3 block2d(kTile, kTile);
    const dim3 grid_rr(static_cast<unsigned int>((R + kTile - 1) / kTile),
                       static_cast<unsigned int>((R + kTile - 1) / kTile));
    const dim3 grid_rc(static_cast<unsigned int>((C + kTile - 1) / kTile),
                       static_cast<unsigned int>((R + kTile - 1) / kTile));

    for (int step = 0; step < 5; ++step) {
        const int src_slot = step & 1;
        baseline_gram_kernel<<<grid_rr, block2d>>>(R, C, src_slot);
        baseline_polyB_kernel<<<grid_rr, block2d>>>(R);
        baseline_updateX_kernel<<<grid_rc, block2d>>>(R, C, src_slot);
    }

    const float scale = std::sqrt(static_cast<float>((M > N) ? M : N));
    baseline_scale_output_kernel<<<elem_blocks, 256>>>(output, M, N, transposed, scale);
}
