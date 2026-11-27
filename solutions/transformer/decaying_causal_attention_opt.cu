#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>

namespace {

constexpr int kChunkSize = 64;
constexpr int kHeadDim64 = 64;
constexpr int kPadStride = 68;  // 64 + 4 to avoid shared-memory bank conflicts
constexpr int kTileK = 16;

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kFallbackThreads = kWarpSize * kWarpsPerBlock;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// General fallback for arbitrary d_model (1..256) or small seq_len (< 128).
__global__ void decaying_causal_attention_fallback_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int seq_len,
    int d_model,
    float gamma,
    float inv_sqrt_d) {
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int n = blockIdx.x * kWarpsPerBlock + warp_id;
    if (n >= seq_len) {
        return;
    }

    const int elems = (d_model + kWarpSize - 1) / kWarpSize;
    float q_reg[8];
    float o_reg[8];

    const float* q_ptr = Q + static_cast<size_t>(n) * d_model;
    for (int i = 0; i < elems; ++i) {
        const int col = lane + i * kWarpSize;
        q_reg[i] = (col < d_model) ? q_ptr[col] * inv_sqrt_d : 0.0f;
        o_reg[i] = 0.0f;
    }

    float decay = 1.0f;
    for (int m = n; m >= 0; --m) {
        const float* k_ptr = K + static_cast<size_t>(m) * d_model;
        const float* v_ptr = V + static_cast<size_t>(m) * d_model;

        float dot = 0.0f;
        for (int i = 0; i < elems; ++i) {
            const int col = lane + i * kWarpSize;
            if (col < d_model) {
                dot = fmaf(q_reg[i], k_ptr[col], dot);
            }
        }
        float score = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0);
        float weight = score * decay;

        for (int i = 0; i < elems; ++i) {
            const int col = lane + i * kWarpSize;
            if (col < d_model) {
                o_reg[i] = fmaf(weight, v_ptr[col], o_reg[i]);
            }
        }
        decay *= gamma;
    }

    float* out_ptr = output + static_cast<size_t>(n) * d_model;
    for (int i = 0; i < elems; ++i) {
        const int col = lane + i * kWarpSize;
        if (col < d_model) {
            out_ptr[col] = o_reg[i];
        }
    }
}

// Stage 1: Compute per-chunk KV state R_c[64, 64] = sum_{i=0..63} gamma^(63-i) * K[c*64+i]^T * V[c*64+i]
// Only chunks c in [0, num_chunks - 2] are needed for inter-chunk propagation.
__global__ void chunk_kv_state_kernel(
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ state,
    float gamma) {
    const int c = blockIdx.x;
    const int tid = threadIdx.x;  // 0..255
    const int ty = tid >> 4;      // 0..15 -> rows of R_c (k_dim)
    const int tx = tid & 15;      // 0..15 -> cols of R_c (v_dim)

    __shared__ float sK[kTileK][kPadStride];
    __shared__ float sV[kTileK][kPadStride];

    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    const int base_row = c * kChunkSize;

    for (int t = 0; t < kChunkSize; t += kTileK) {
        // 256 threads load 16x64 floats = 1024 floats = 256 float4s
        const int row_in_tile = tid >> 4;   // 0..15
        const int col_vec = tid & 15;       // 0..15 -> 4 floats each
        const int global_row = base_row + t + row_in_tile;
        const float decay = powf(gamma, static_cast<float>(kChunkSize - 1 - (t + row_in_tile)));

        float4 kv = reinterpret_cast<const float4*>(K + static_cast<size_t>(global_row) * kHeadDim64)[col_vec];
        float4 vv = reinterpret_cast<const float4*>(V + static_cast<size_t>(global_row) * kHeadDim64)[col_vec];

        sK[row_in_tile][col_vec * 4 + 0] = kv.x * decay;
        sK[row_in_tile][col_vec * 4 + 1] = kv.y * decay;
        sK[row_in_tile][col_vec * 4 + 2] = kv.z * decay;
        sK[row_in_tile][col_vec * 4 + 3] = kv.w * decay;

        sV[row_in_tile][col_vec * 4 + 0] = vv.x;
        sV[row_in_tile][col_vec * 4 + 1] = vv.y;
        sV[row_in_tile][col_vec * 4 + 2] = vv.z;
        sV[row_in_tile][col_vec * 4 + 3] = vv.w;

        __syncthreads();

#pragma unroll
        for (int i = 0; i < kTileK; ++i) {
            const float k0 = sK[i][ty * 4 + 0];
            const float k1 = sK[i][ty * 4 + 1];
            const float k2 = sK[i][ty * 4 + 2];
            const float k3 = sK[i][ty * 4 + 3];

            const float v0 = sV[i][tx * 4 + 0];
            const float v1 = sV[i][tx * 4 + 1];
            const float v2 = sV[i][tx * 4 + 2];
            const float v3 = sV[i][tx * 4 + 3];

            acc[0][0] = fmaf(k0, v0, acc[0][0]);
            acc[0][1] = fmaf(k0, v1, acc[0][1]);
            acc[0][2] = fmaf(k0, v2, acc[0][2]);
            acc[0][3] = fmaf(k0, v3, acc[0][3]);

            acc[1][0] = fmaf(k1, v0, acc[1][0]);
            acc[1][1] = fmaf(k1, v1, acc[1][1]);
            acc[1][2] = fmaf(k1, v2, acc[1][2]);
            acc[1][3] = fmaf(k1, v3, acc[1][3]);

            acc[2][0] = fmaf(k2, v0, acc[2][0]);
            acc[2][1] = fmaf(k2, v1, acc[2][1]);
            acc[2][2] = fmaf(k2, v2, acc[2][2]);
            acc[2][3] = fmaf(k2, v3, acc[2][3]);

            acc[3][0] = fmaf(k3, v0, acc[3][0]);
            acc[3][1] = fmaf(k3, v1, acc[3][1]);
            acc[3][2] = fmaf(k3, v2, acc[3][2]);
            acc[3][3] = fmaf(k3, v3, acc[3][3]);
        }

        __syncthreads();
    }

    float* out_state = state + static_cast<size_t>(c) * (kHeadDim64 * kHeadDim64);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float4 res = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        reinterpret_cast<float4*>(out_state + (ty * 4 + i) * kHeadDim64)[tx] = res;
    }
}

// Stage 2: Exclusive decaying prefix scan over chunk states:
// H_0 = 0, H_c = gamma^64 * H_{c-1} + R_{c-1}
__global__ void chunk_state_scan_kernel(
    float* __restrict__ state,
    int num_chunks,
    float gamma_chunk) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 0..4095
    if (idx >= kHeadDim64 * kHeadDim64) {
        return;
    }

    float h = 0.0f;
    for (int c = 0; c < num_chunks; ++c) {
        const size_t offset = static_cast<size_t>(c) * (kHeadDim64 * kHeadDim64) + idx;
        const float r = (c < num_chunks - 1) ? state[offset] : 0.0f;
        state[offset] = h;
        h = fmaf(gamma_chunk, h, r);
    }
}

// Stage 3: Combine inter-chunk (Q_c * H_c) and intra-chunk ((Q_c K_c^T * D_causal) * V_c).
__global__ void chunk_output_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ state,
    float* __restrict__ output,
    int seq_len,
    float gamma,
    float inv_sqrt_d) {
    const int c = blockIdx.x;
    const int tid = threadIdx.x;  // 0..255
    const int ty = tid >> 4;      // 0..15 -> 4 rows of chunk
    const int tx = tid & 15;      // 0..15 -> 4 cols

    const int base_row = c * kChunkSize;
    const int valid_rows = min(kChunkSize, seq_len - base_row);

    // 3 shared buffers totaling 42.5 KB (< 48 KB static limit), padded to avoid bank conflicts
    __shared__ float sQ[kChunkSize][kPadStride];
    __shared__ float sS[kChunkSize][kPadStride];
    __shared__ float sBuf[kTileK][kPadStride];

    // Load Q chunk (64x64 = 4096 floats = 1024 float4s; 256 threads x 4 float4s each)
#pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        const int r = pass * 16 + ty;
        if (r < valid_rows) {
            float4 qv = reinterpret_cast<const float4*>(Q + static_cast<size_t>(base_row + r) * kHeadDim64)[tx];
            sQ[r][tx * 4 + 0] = qv.x * inv_sqrt_d;
            sQ[r][tx * 4 + 1] = qv.y * inv_sqrt_d;
            sQ[r][tx * 4 + 2] = qv.z * inv_sqrt_d;
            sQ[r][tx * 4 + 3] = qv.w * inv_sqrt_d;
        } else {
            sQ[r][tx * 4 + 0] = 0.0f;
            sQ[r][tx * 4 + 1] = 0.0f;
            sQ[r][tx * 4 + 2] = 0.0f;
            sQ[r][tx * 4 + 3] = 0.0f;
        }
    }
    __syncthreads();

    float out_acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            out_acc[i][j] = 0.0f;
        }
    }

    // 1. Inter-chunk contribution: gamma^(r+1) * (Q_scaled[r] @ H_c)
    if (c > 0) {
        const float* H_c = state + static_cast<size_t>(c) * (kHeadDim64 * kHeadDim64);
        for (int k0 = 0; k0 < kHeadDim64; k0 += kTileK) {
            // Load 16x64 slab of H_c into sBuf
            float4 hv = reinterpret_cast<const float4*>(H_c + (k0 + ty) * kHeadDim64)[tx];
            sBuf[ty][tx * 4 + 0] = hv.x;
            sBuf[ty][tx * 4 + 1] = hv.y;
            sBuf[ty][tx * 4 + 2] = hv.z;
            sBuf[ty][tx * 4 + 3] = hv.w;
            __syncthreads();

#pragma unroll
            for (int kk = 0; kk < kTileK; ++kk) {
                const float q0 = sQ[ty * 4 + 0][k0 + kk];
                const float q1 = sQ[ty * 4 + 1][k0 + kk];
                const float q2 = sQ[ty * 4 + 2][k0 + kk];
                const float q3 = sQ[ty * 4 + 3][k0 + kk];

                const float h0 = sBuf[kk][tx * 4 + 0];
                const float h1 = sBuf[kk][tx * 4 + 1];
                const float h2 = sBuf[kk][tx * 4 + 2];
                const float h3 = sBuf[kk][tx * 4 + 3];

                out_acc[0][0] = fmaf(q0, h0, out_acc[0][0]);
                out_acc[0][1] = fmaf(q0, h1, out_acc[0][1]);
                out_acc[0][2] = fmaf(q0, h2, out_acc[0][2]);
                out_acc[0][3] = fmaf(q0, h3, out_acc[0][3]);

                out_acc[1][0] = fmaf(q1, h0, out_acc[1][0]);
                out_acc[1][1] = fmaf(q1, h1, out_acc[1][1]);
                out_acc[1][2] = fmaf(q1, h2, out_acc[1][2]);
                out_acc[1][3] = fmaf(q1, h3, out_acc[1][3]);

                out_acc[2][0] = fmaf(q2, h0, out_acc[2][0]);
                out_acc[2][1] = fmaf(q2, h1, out_acc[2][1]);
                out_acc[2][2] = fmaf(q2, h2, out_acc[2][2]);
                out_acc[2][3] = fmaf(q2, h3, out_acc[2][3]);

                out_acc[3][0] = fmaf(q3, h0, out_acc[3][0]);
                out_acc[3][1] = fmaf(q3, h1, out_acc[3][1]);
                out_acc[3][2] = fmaf(q3, h2, out_acc[3][2]);
                out_acc[3][3] = fmaf(q3, h3, out_acc[3][3]);
            }
            __syncthreads();
        }

        float decay = powf(gamma, static_cast<float>(ty * 4 + 1));
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                out_acc[i][j] *= decay;
            }
            decay *= gamma;
        }
    }

    // 2. Intra-chunk attention scores S[64, 64] = (Q_scaled @ K^T) * D_causal
    // Compute 4x4 tile of S per thread: rows ty*4..ty*4+3, cols tx*4..tx*4+3
    float s_acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            s_acc[i][j] = 0.0f;
        }
    }

    // Tile K in 16-row slabs along the key-row dimension? No: for S = Q @ K^T,
    // K is indexed by [j_row, k_dim]. Since tx selects j_row = tx*4..tx*4+3,
    // we tile along k_dim in slabs of 16, or load full K[64, 64] into sS? No, sS is output.
    // Tiling along j_row doesn't work for fixed tx, so we load K[64, 16] slabs along k_dim!
    // Wait: 64 rows x 16 cols = 1024 floats = 256 float4s, fits in sBuf[16][68] transposed!
    for (int k0 = 0; k0 < kHeadDim64; k0 += kTileK) {
        // Load K[0..63, k0..k0+15] into sBuf[kk][j_row] (16 x 64)
        // tid = 0..255: j_row = tid >> 2 (0..63), kk_vec = tid & 3 (0..3 -> 4 floats)
        const int j_row = tid >> 2;
        const int kk_vec = tid & 3;
        if (j_row < valid_rows) {
            float4 kv = reinterpret_cast<const float4*>(
                K + static_cast<size_t>(base_row + j_row) * kHeadDim64 + k0)[kk_vec];
            sBuf[kk_vec * 4 + 0][j_row] = kv.x;
            sBuf[kk_vec * 4 + 1][j_row] = kv.y;
            sBuf[kk_vec * 4 + 2][j_row] = kv.z;
            sBuf[kk_vec * 4 + 3][j_row] = kv.w;
        } else {
            sBuf[kk_vec * 4 + 0][j_row] = 0.0f;
            sBuf[kk_vec * 4 + 1][j_row] = 0.0f;
            sBuf[kk_vec * 4 + 2][j_row] = 0.0f;
            sBuf[kk_vec * 4 + 3][j_row] = 0.0f;
        }
        __syncthreads();

        if (tx <= ty) {
#pragma unroll
            for (int kk = 0; kk < kTileK; ++kk) {
                const float q0 = sQ[ty * 4 + 0][k0 + kk];
                const float q1 = sQ[ty * 4 + 1][k0 + kk];
                const float q2 = sQ[ty * 4 + 2][k0 + kk];
                const float q3 = sQ[ty * 4 + 3][k0 + kk];

                const float k0_val = sBuf[kk][tx * 4 + 0];
                const float k1_val = sBuf[kk][tx * 4 + 1];
                const float k2_val = sBuf[kk][tx * 4 + 2];
                const float k3_val = sBuf[kk][tx * 4 + 3];

                s_acc[0][0] = fmaf(q0, k0_val, s_acc[0][0]);
                s_acc[0][1] = fmaf(q0, k1_val, s_acc[0][1]);
                s_acc[0][2] = fmaf(q0, k2_val, s_acc[0][2]);
                s_acc[0][3] = fmaf(q0, k3_val, s_acc[0][3]);

                s_acc[1][0] = fmaf(q1, k0_val, s_acc[1][0]);
                s_acc[1][1] = fmaf(q1, k1_val, s_acc[1][1]);
                s_acc[1][2] = fmaf(q1, k2_val, s_acc[1][2]);
                s_acc[1][3] = fmaf(q1, k3_val, s_acc[1][3]);

                s_acc[2][0] = fmaf(q2, k0_val, s_acc[2][0]);
                s_acc[2][1] = fmaf(q2, k1_val, s_acc[2][1]);
                s_acc[2][2] = fmaf(q2, k2_val, s_acc[2][2]);
                s_acc[2][3] = fmaf(q2, k3_val, s_acc[2][3]);

                s_acc[3][0] = fmaf(q3, k0_val, s_acc[3][0]);
                s_acc[3][1] = fmaf(q3, k1_val, s_acc[3][1]);
                s_acc[3][2] = fmaf(q3, k2_val, s_acc[3][2]);
                s_acc[3][3] = fmaf(q3, k3_val, s_acc[3][3]);
            }
        }
        __syncthreads();
    }

    // Apply causal decay mask and store S into sS[64][68]
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = ty * 4 + i;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = tx * 4 + j;
            float val = 0.0f;
            if (col <= r && r < valid_rows) {
                val = s_acc[i][j] * powf(gamma, static_cast<float>(r - col));
            }
            sS[r][col] = val;
        }
    }
    __syncthreads();

    // 3. Intra-chunk value aggregation: out_acc += sS @ V_c
    for (int j0 = 0; j0 < kChunkSize; j0 += kTileK) {
        if (j0 + ty < valid_rows) {
            float4 vv = reinterpret_cast<const float4*>(
                V + static_cast<size_t>(base_row + j0 + ty) * kHeadDim64)[tx];
            sBuf[ty][tx * 4 + 0] = vv.x;
            sBuf[ty][tx * 4 + 1] = vv.y;
            sBuf[ty][tx * 4 + 2] = vv.z;
            sBuf[ty][tx * 4 + 3] = vv.w;
        } else {
            sBuf[ty][tx * 4 + 0] = 0.0f;
            sBuf[ty][tx * 4 + 1] = 0.0f;
            sBuf[ty][tx * 4 + 2] = 0.0f;
            sBuf[ty][tx * 4 + 3] = 0.0f;
        }
        __syncthreads();

        // Only rows with r_max >= j0 have non-zero entries in sS[:, j0..j0+15]
        if (ty * 4 + 3 >= j0) {
#pragma unroll
            for (int jj = 0; jj < kTileK; ++jj) {
                const float s0 = sS[ty * 4 + 0][j0 + jj];
                const float s1 = sS[ty * 4 + 1][j0 + jj];
                const float s2 = sS[ty * 4 + 2][j0 + jj];
                const float s3 = sS[ty * 4 + 3][j0 + jj];

                const float v0 = sBuf[jj][tx * 4 + 0];
                const float v1 = sBuf[jj][tx * 4 + 1];
                const float v2 = sBuf[jj][tx * 4 + 2];
                const float v3 = sBuf[jj][tx * 4 + 3];

                out_acc[0][0] = fmaf(s0, v0, out_acc[0][0]);
                out_acc[0][1] = fmaf(s0, v1, out_acc[0][1]);
                out_acc[0][2] = fmaf(s0, v2, out_acc[0][2]);
                out_acc[0][3] = fmaf(s0, v3, out_acc[0][3]);

                out_acc[1][0] = fmaf(s1, v0, out_acc[1][0]);
                out_acc[1][1] = fmaf(s1, v1, out_acc[1][1]);
                out_acc[1][2] = fmaf(s1, v2, out_acc[1][2]);
                out_acc[1][3] = fmaf(s1, v3, out_acc[1][3]);

                out_acc[2][0] = fmaf(s2, v0, out_acc[2][0]);
                out_acc[2][1] = fmaf(s2, v1, out_acc[2][1]);
                out_acc[2][2] = fmaf(s2, v2, out_acc[2][2]);
                out_acc[2][3] = fmaf(s2, v3, out_acc[2][3]);

                out_acc[3][0] = fmaf(s3, v0, out_acc[3][0]);
                out_acc[3][1] = fmaf(s3, v1, out_acc[3][1]);
                out_acc[3][2] = fmaf(s3, v2, out_acc[3][2]);
                out_acc[3][3] = fmaf(s3, v3, out_acc[3][3]);
            }
        }
        __syncthreads();
    }

    // Store output chunk using float4
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = ty * 4 + i;
        if (r < valid_rows) {
            float4 res = make_float4(out_acc[i][0], out_acc[i][1], out_acc[i][2], out_acc[i][3]);
            reinterpret_cast<float4*>(output + static_cast<size_t>(base_row + r) * kHeadDim64)[tx] = res;
        }
    }
}

float* get_chunk_state_buffer(size_t required_floats) {
    static float* d_buf = nullptr;
    static size_t cap_floats = 0;
    if (required_floats > cap_floats) {
        if (d_buf != nullptr) {
            cudaFree(d_buf);
            d_buf = nullptr;
            cap_floats = 0;
        }
        if (cudaMalloc(&d_buf, required_floats * sizeof(float)) == cudaSuccess) {
            cap_floats = required_floats;
        } else {
            d_buf = nullptr;
            cap_floats = 0;
        }
    }
    return d_buf;
}

}  // namespace

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int seq_len,
                      int d_model, float gamma) {
    if (seq_len <= 0 || d_model <= 0) {
        return;
    }

    const float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(d_model));

    if (d_model == kHeadDim64 && seq_len >= 128) {
        const int num_chunks = (seq_len + kChunkSize - 1) / kChunkSize;
        const size_t state_floats = static_cast<size_t>(num_chunks) * kHeadDim64 * kHeadDim64;
        float* d_state = get_chunk_state_buffer(state_floats);
        if (d_state != nullptr) {
            // Stage 1: per-chunk KV state R_c for c = 0 .. num_chunks - 2
            if (num_chunks > 1) {
                chunk_kv_state_kernel<<<num_chunks - 1, 256>>>(K, V, d_state, gamma);
                // Stage 2: exclusive decaying prefix scan -> H_c
                const float gamma_chunk = powf(gamma, static_cast<float>(kChunkSize));
                chunk_state_scan_kernel<<<16, 256>>>(d_state, num_chunks, gamma_chunk);
            }
            // Stage 3: inter-chunk + intra-chunk output
            chunk_output_kernel<<<num_chunks, 256>>>(
                Q, K, V, d_state, output, seq_len, gamma, inv_sqrt_d);
            return;
        }
    }

    const int blocks = (seq_len + kWarpsPerBlock - 1) / kWarpsPerBlock;
    decaying_causal_attention_fallback_kernel<<<blocks, kFallbackThreads>>>(
        Q, K, V, output, seq_len, d_model, gamma, inv_sqrt_d);
}
