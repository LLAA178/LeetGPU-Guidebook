#include <cuda_runtime.h>

namespace {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;

// Double-buffered 128x128 register-tiled (8x8 per thread) FP32 Grouped GEMM.
// Fast path requires K % 4 == 0 and N % 4 == 0 for float4 vectorized global loads/stores.
__global__ __launch_bounds__(256, 2)
void grouped_gemm_opt_vec4_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const int* __restrict__ group_offsets,
    float* __restrict__ C,
    int G,
    int K,
    int N
) {
    // Map blockIdx.y to (group_id, local_tile_m) without host-device sync.
    const int tile_id = blockIdx.y;
    int acc_tiles = 0;
    int group_id = -1;
    int m_start = 0;
    int m_end = 0;
    int local_tile_m = 0;

    for (int g = 0; g < G; ++g) {
        int s = group_offsets[g];
        int e = group_offsets[g + 1];
        int rows = e - s;
        if (rows <= 0) continue;
        int tiles = (rows + BM - 1) / BM;
        if (tile_id < acc_tiles + tiles) {
            group_id = g;
            m_start = s;
            m_end = e;
            local_tile_m = tile_id - acc_tiles;
            break;
        }
        acc_tiles += tiles;
    }
    if (group_id < 0) return;

    const int block_m = m_start + local_tile_m * BM;
    const int block_n = blockIdx.x * BN;
    const int tid = threadIdx.x;  // 0..255
    const int ty = tid >> 4;      // 0..15
    const int tx = tid & 15;      // 0..15

    // Shared memory double buffers:
    // As[stage][k][m] transposed so m is contiguous for float4 loads.
    // Bs[stage][k][n] row-major so n is contiguous for float4 loads.
    __shared__ float As[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    const float* B_group = B + (size_t)group_id * K * N;

    // Loader indices for A: 128 rows x (BK/4 = 2) float4s = 256 float4s
    const int a_row_local = tid >> 1;        // 0..127
    const int a_col_local = (tid & 1) << 2;  // 0 or 4
    const int a_row_global = block_m + a_row_local;
    const bool a_row_valid = (a_row_global < m_end);

    // Loader indices for B: BK=8 rows x (128/4 = 32) float4s = 256 float4s
    const int b_row_local = tid >> 5;        // 0..7
    const int b_col_local = (tid & 31) << 2; // 0..124 step 4
    const int b_col_global = block_n + b_col_local;
    const bool b_col_valid = (b_col_global < N);

    // 8x8 accumulators per thread, organized as 2x2 grid of 4x4 sub-tiles
    // spanning rows {ty*4..ty*4+3, ty*4+64..ty*4+67} and cols {tx*4..tx*4+3, tx*4+64..tx*4+67}
    float c00[4][4] = {};
    float c01[4][4] = {};
    float c10[4][4] = {};
    float c11[4][4] = {};

    // Prologue: load first BK tile into stage 0
    {
        float4 a4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (a_row_valid) {
            a4 = *reinterpret_cast<const float4*>(A + (size_t)a_row_global * K + a_col_local);
        }
        As[0][a_col_local + 0][a_row_local] = a4.x;
        As[0][a_col_local + 1][a_row_local] = a4.y;
        As[0][a_col_local + 2][a_row_local] = a4.z;
        As[0][a_col_local + 3][a_row_local] = a4.w;

        float4 b4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (b_col_valid) {
            b4 = *reinterpret_cast<const float4*>(B_group + (size_t)b_row_local * N + b_col_global);
        }
        *reinterpret_cast<float4*>(&Bs[0][b_row_local][b_col_local]) = b4;
    }
    __syncthreads();

    int stage = 0;
    const int num_k_tiles = K / BK;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int next_stage = stage ^ 1;
        const int next_k = (kt + 1) * BK;

        // Prefetch next global tile into registers while computing current stage
        float4 next_a4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 next_b4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (kt + 1 < num_k_tiles) {
            if (a_row_valid) {
                next_a4 = *reinterpret_cast<const float4*>(
                    A + (size_t)a_row_global * K + next_k + a_col_local
                );
            }
            if (b_col_valid) {
                next_b4 = *reinterpret_cast<const float4*>(
                    B_group + (size_t)(next_k + b_row_local) * N + b_col_global
                );
            }
        }

        // Compute 8x8 outer product over BK=8
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float4 a0 = *reinterpret_cast<const float4*>(&As[stage][k][ty * 4]);
            float4 a1 = *reinterpret_cast<const float4*>(&As[stage][k][ty * 4 + 64]);
            float4 b0 = *reinterpret_cast<const float4*>(&Bs[stage][k][tx * 4]);
            float4 b1 = *reinterpret_cast<const float4*>(&Bs[stage][k][tx * 4 + 64]);

            float av0[4] = {a0.x, a0.y, a0.z, a0.w};
            float av1[4] = {a1.x, a1.y, a1.z, a1.w};
            float bv0[4] = {b0.x, b0.y, b0.z, b0.w};
            float bv1[4] = {b1.x, b1.y, b1.z, b1.w};

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    c00[i][j] = fmaf(av0[i], bv0[j], c00[i][j]);
                    c01[i][j] = fmaf(av0[i], bv1[j], c01[i][j]);
                    c10[i][j] = fmaf(av1[i], bv0[j], c10[i][j]);
                    c11[i][j] = fmaf(av1[i], bv1[j], c11[i][j]);
                }
            }
        }

        if (kt + 1 < num_k_tiles) {
            As[next_stage][a_col_local + 0][a_row_local] = next_a4.x;
            As[next_stage][a_col_local + 1][a_row_local] = next_a4.y;
            As[next_stage][a_col_local + 2][a_row_local] = next_a4.z;
            As[next_stage][a_col_local + 3][a_row_local] = next_a4.w;
            *reinterpret_cast<float4*>(&Bs[next_stage][b_row_local][b_col_local]) = next_b4;
            __syncthreads();
            stage = next_stage;
        }
    }

    // Store 8x8 output with float4 vectorized writes
    const int c_col0 = block_n + tx * 4;
    const int c_col1 = block_n + tx * 4 + 64;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int r0 = block_m + ty * 4 + i;
        if (r0 < m_end) {
            if (c_col0 < N) {
                *reinterpret_cast<float4*>(C + (size_t)r0 * N + c_col0) =
                    make_float4(c00[i][0], c00[i][1], c00[i][2], c00[i][3]);
            }
            if (c_col1 < N) {
                *reinterpret_cast<float4*>(C + (size_t)r0 * N + c_col1) =
                    make_float4(c01[i][0], c01[i][1], c01[i][2], c01[i][3]);
            }
        }

        int r1 = block_m + ty * 4 + 64 + i;
        if (r1 < m_end) {
            if (c_col0 < N) {
                *reinterpret_cast<float4*>(C + (size_t)r1 * N + c_col0) =
                    make_float4(c10[i][0], c10[i][1], c10[i][2], c10[i][3]);
            }
            if (c_col1 < N) {
                *reinterpret_cast<float4*>(C + (size_t)r1 * N + c_col1) =
                    make_float4(c11[i][0], c11[i][1], c11[i][2], c11[i][3]);
            }
        }
    }
}

// General 16x16 shared-memory tiled fallback for arbitrary unaligned K or N.
__global__ void grouped_gemm_opt_general_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const int* __restrict__ group_offsets,
    float* __restrict__ C,
    int G,
    int K,
    int N
) {
    constexpr int TILE = 16;
    const int tile_id = blockIdx.y;
    int acc_tiles = 0;
    int group_id = -1;
    int m_start = 0;
    int m_end = 0;
    int local_tile_m = 0;

    for (int g = 0; g < G; ++g) {
        int s = group_offsets[g];
        int e = group_offsets[g + 1];
        int rows = e - s;
        if (rows <= 0) continue;
        int tiles = (rows + TILE - 1) / TILE;
        if (tile_id < acc_tiles + tiles) {
            group_id = g;
            m_start = s;
            m_end = e;
            local_tile_m = tile_id - acc_tiles;
            break;
        }
        acc_tiles += tiles;
    }
    if (group_id < 0) return;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = m_start + local_tile_m * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    const float* B_group = B + (size_t)group_id * K * N;

    float sum = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int ak = k0 + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < m_end && ak < K) ? A[(size_t)row * K + ak] : 0.0f;

        int bk = k0 + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B_group[(size_t)bk * N + col] : 0.0f;

        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum = fmaf(As[threadIdx.y][k], Bs[k][threadIdx.x], sum);
        }
        __syncthreads();
    }

    if (row < m_end && col < N) {
        C[(size_t)row * N + col] = sum;
    }
}

}  // namespace

extern "C" void solve(const float* A, const float* B, const int* group_offsets, float* C, int G,
                      int M_total, int K, int N) {
    if (G <= 0 || M_total <= 0 || K <= 0 || N <= 0) return;

    if ((K % BK == 0) && (N % 4 == 0)) {
        dim3 block(256);
        int grid_x = (N + BN - 1) / BN;
        int grid_y = (M_total + BM - 1) / BM + G;
        dim3 grid(grid_x, grid_y);
        grouped_gemm_opt_vec4_kernel<<<grid, block>>>(A, B, group_offsets, C, G, K, N);
    } else {
        dim3 block(16, 16);
        int grid_x = (N + 15) / 16;
        int grid_y = (M_total + 15) / 16 + G;
        dim3 grid(grid_x, grid_y);
        grouped_gemm_opt_general_kernel<<<grid, block>>>(A, B, group_offsets, C, G, K, N);
    }
}
