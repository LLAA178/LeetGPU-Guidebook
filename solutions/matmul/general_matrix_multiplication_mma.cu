#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kFallbackTile = 16;

__global__ void gemm_fallback_kernel(const half* __restrict__ A,
                                     const half* __restrict__ B,
                                     half* __restrict__ C,
                                     int M,
                                     int N,
                                     int K,
                                     float alpha,
                                     float beta) {
    const int row = blockIdx.y * kFallbackTile + threadIdx.y;
    const int col = blockIdx.x * kFallbackTile + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }

    float acc = 0.0f;
    const half* a_row = A + static_cast<size_t>(row) * K;
    for (int k = 0; k < K; ++k) {
        acc = fmaf(__half2float(a_row[k]), __half2float(B[static_cast<size_t>(k) * N + col]), acc);
    }

    const size_t c_idx = static_cast<size_t>(row) * N + col;
    float out = alpha * acc;
    if (beta != 0.0f) {
        out = fmaf(beta, __half2float(C[c_idx]), out);
    }
    C[c_idx] = __float2half(out);
}

// -----------------------------------------------------------------------------
// SM 7.5+ (T4 / A100) Double-Buffered ldmatrix + mma.sync.m16n8k8 FP16 GEMM
// BM = 64, BN = 64, BK = 32, 4 warps (2x2 warp grid, each warp 32x32)
// -----------------------------------------------------------------------------
constexpr int kBM = 64;
constexpr int kBN = 64;
constexpr int kBK = 32;
constexpr int kAStride = kBK + 8;  // 40 halfs (80 bytes, 16-byte aligned, bank-conflict free)
constexpr int kBStride = kBN + 8;  // 72 halfs (144 bytes, 16-byte aligned, bank-conflict free)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0,
                                            uint32_t& r1,
                                            uint32_t& r2,
                                            uint32_t& r3,
                                            const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0,
                                                  uint32_t& r1,
                                                  uint32_t& r2,
                                                  uint32_t& r3,
                                                  const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k8_f16(float& d0,
                                                float& d1,
                                                float& d2,
                                                float& d3,
                                                uint32_t a0,
                                                uint32_t a1,
                                                uint32_t b0,
                                                float c0,
                                                float c1,
                                                float c2,
                                                float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

#endif

__global__ void __launch_bounds__(128, 3)
gemm_mma_64x64_kernel(const half* __restrict__ A,
                      const half* __restrict__ B,
                      half* __restrict__ C,
                      int M,
                      int N,
                      int K,
                      float alpha,
                      float beta) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    __shared__ alignas(16) half sA[2][kBM][kAStride];
    __shared__ alignas(16) half sB[2][kBK][kBStride];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = (warp_id >> 1) * 32;  // 0 or 32
    const int warp_n = (warp_id & 1) * 32;   // 0 or 32

    const int block_m = blockIdx.y * kBM;
    const int block_n = blockIdx.x * kBN;

    // 128 threads load 64x32 A (256 uint4s -> 2/thread) and 32x64 B (256 uint4s -> 2/thread)
    const int a_r0 = tid >> 2;         // 0..31
    const int a_c = (tid & 3) * 8;     // 0, 8, 16, 24
    const int a_r1 = a_r0 + 32;        // 32..63

    const int b_r0 = tid >> 3;         // 0..15
    const int b_c = (tid & 7) * 8;     // 0..56
    const int b_r1 = b_r0 + 16;        // 16..31

    // Accumulators: 2 (m16) x 4 (n8) = 8 tiles x 4 floats = 32 floats
    float acc[2][4][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j][0] = 0.0f;
            acc[i][j][1] = 0.0f;
            acc[i][j][2] = 0.0f;
            acc[i][j][3] = 0.0f;
        }
    }

    // Load tile 0 into stage 0
    uint4 reg_a0 = *reinterpret_cast<const uint4*>(&A[static_cast<size_t>(block_m + a_r0) * K + a_c]);
    uint4 reg_a1 = *reinterpret_cast<const uint4*>(&A[static_cast<size_t>(block_m + a_r1) * K + a_c]);
    uint4 reg_b0 = *reinterpret_cast<const uint4*>(&B[static_cast<size_t>(b_r0) * N + block_n + b_c]);
    uint4 reg_b1 = *reinterpret_cast<const uint4*>(&B[static_cast<size_t>(b_r1) * N + block_n + b_c]);

    *reinterpret_cast<uint4*>(&sA[0][a_r0][a_c]) = reg_a0;
    *reinterpret_cast<uint4*>(&sA[0][a_r1][a_c]) = reg_a1;
    *reinterpret_cast<uint4*>(&sB[0][b_r0][b_c]) = reg_b0;
    *reinterpret_cast<uint4*>(&sB[0][b_r1][b_c]) = reg_b1;
    __syncthreads();

    const int num_tiles = K / kBK;
    const int ld_a_row = lane;              // 0..31 covers both 16x8 A tiles
    const int ld_b_row = lane & 7;          // 0..7
    const int ld_b_col = (lane >> 3) * 8;   // 0, 8, 16, 24 covers all four 8x8 B tiles

    for (int t = 0; t < num_tiles; ++t) {
        const int cur = t & 1;
        const int nxt = cur ^ 1;

        // Prefetch next tile from global memory into registers while computing current tile
        if (t + 1 < num_tiles) {
            const int k_next = (t + 1) * kBK;
            reg_a0 = *reinterpret_cast<const uint4*>(
                &A[static_cast<size_t>(block_m + a_r0) * K + k_next + a_c]);
            reg_a1 = *reinterpret_cast<const uint4*>(
                &A[static_cast<size_t>(block_m + a_r1) * K + k_next + a_c]);
            reg_b0 = *reinterpret_cast<const uint4*>(
                &B[static_cast<size_t>(k_next + b_r0) * N + block_n + b_c]);
            reg_b1 = *reinterpret_cast<const uint4*>(
                &B[static_cast<size_t>(k_next + b_r1) * N + block_n + b_c]);
        }

        // Compute 4 k-steps of 8 (total BK = 32)
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int k_off = ks * 8;
            uint32_t a_frag[2][2];
            uint32_t b_frag[4];

            ldmatrix_x4(a_frag[0][0], a_frag[0][1], a_frag[1][0], a_frag[1][1],
                        &sA[cur][warp_m + ld_a_row][k_off]);
            ldmatrix_x4_trans(b_frag[0], b_frag[1], b_frag[2], b_frag[3],
                              &sB[cur][k_off + ld_b_row][warp_n + ld_b_col]);
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    mma_m16n8k8_f16(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3],
                                    a_frag[i][0], a_frag[i][1], b_frag[j],
                                    acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]);
                }
            }
        }

        if (t + 1 < num_tiles) {
            *reinterpret_cast<uint4*>(&sA[nxt][a_r0][a_c]) = reg_a0;
            *reinterpret_cast<uint4*>(&sA[nxt][a_r1][a_c]) = reg_a1;
            *reinterpret_cast<uint4*>(&sB[nxt][b_r0][b_c]) = reg_b0;
            *reinterpret_cast<uint4*>(&sB[nxt][b_r1][b_c]) = reg_b1;
            __syncthreads();
        }
    }

    // Epilogue: write half2 directly to C
    const int group_id = lane >> 2;       // 0..7
    const int tid_in_group = lane & 3;    // 0..3

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int r0 = block_m + warp_m + i * 16 + group_id;
        const int r1 = r0 + 8;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = block_n + warp_n + j * 8 + tid_in_group * 2;
            half2* ptr0 = reinterpret_cast<half2*>(&C[static_cast<size_t>(r0) * N + c]);
            half2* ptr1 = reinterpret_cast<half2*>(&C[static_cast<size_t>(r1) * N + c]);

            float v0 = alpha * acc[i][j][0];
            float v1 = alpha * acc[i][j][1];
            float v2 = alpha * acc[i][j][2];
            float v3 = alpha * acc[i][j][3];

            if (beta != 0.0f) {
                float2 old0 = __half22float2(*ptr0);
                float2 old1 = __half22float2(*ptr1);
                v0 = fmaf(beta, old0.x, v0);
                v1 = fmaf(beta, old0.y, v1);
                v2 = fmaf(beta, old1.x, v2);
                v3 = fmaf(beta, old1.y, v3);
            }
            *ptr0 = __floats2half2_rn(v0, v1);
            *ptr1 = __floats2half2_rn(v2, v3);
        }
    }
#endif
}

}  // namespace

extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta) {
    const bool aligned = ((reinterpret_cast<uintptr_t>(A) & 15) == 0) &&
                         ((reinterpret_cast<uintptr_t>(B) & 15) == 0) &&
                         ((reinterpret_cast<uintptr_t>(C) & 3) == 0);
    if (aligned && (M % kBM == 0) && (N % kBN == 0) && (K % kBK == 0)) {
        dim3 block(128);
        dim3 grid(N / kBN, M / kBM);
        gemm_mma_64x64_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
        return;
    }

    dim3 block(kFallbackTile, kFallbackTile);
    dim3 grid((N + kFallbackTile - 1) / kFallbackTile, (M + kFallbackTile - 1) / kFallbackTile);
    gemm_fallback_kernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
}
