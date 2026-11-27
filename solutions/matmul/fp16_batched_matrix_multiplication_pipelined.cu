#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kSize = 256;
constexpr int kTile = 64;
constexpr int kDepth = 32;

struct alignas(16) Stage {
    half a[kTile * kDepth];
    half b[kDepth * kTile];
};

// Permute 16-byte segments within each row while preserving vector alignment.
template<int Width>
__device__ int shared_offset(int row, int col) {
    return row * Width + (col ^ ((row % (Width / 8)) * 8));
}

__device__ void copy_vector(half* dst, const half* src) {
#if __CUDA_ARCH__ >= 800
    unsigned addr = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(addr), "l"(src) : "memory");
#else
    *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(src);
#endif
}

__device__ void load_stage(Stage& dst, const half* A, const half* B, int k) {
    const int tid = threadIdx.x;
    #pragma unroll
    for (int x = tid; x < kTile * kDepth / 8; x += 128) {
        int row = x / (kDepth / 8);
        int col = x % (kDepth / 8) * 8;
        copy_vector(dst.a + shared_offset<kDepth>(row, col), A + row * kSize + k + col);
        row = x / (kTile / 8);
        col = x % (kTile / 8) * 8;
        copy_vector(dst.b + shared_offset<kTile>(row, col), B + (k + row) * kSize + col);
    }
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

__device__ void wait_stage() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
    __syncthreads();
}

__device__ void compute_stage(const Stage& s, float (&acc)[2][4][4]) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int row = warp / 2 * 32;
    const int col = warp % 2 * 32;
    #pragma unroll
    for (int k = 0; k < kDepth; k += 16) {
        unsigned a[2][4], b[4][2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            unsigned addr = __cvta_generic_to_shared(
                s.a + shared_offset<kDepth>(row + i * 16 + lane % 16, k + lane / 16 * 8));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(a[i][0]), "=r"(a[i][1]), "=r"(a[i][2]), "=r"(a[i][3]) : "r"(addr));
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned addr = __cvta_generic_to_shared(
                s.b + shared_offset<kTile>(k + lane % 16, col + j * 8));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                : "=r"(b[j][0]), "=r"(b[j][1]) : "r"(addr));
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float* d = acc[i][j];
#if __CUDA_ARCH__ >= 800
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                    : "r"(a[i][0]), "r"(a[i][1]), "r"(a[i][2]), "r"(a[i][3]),
                      "r"(b[j][0]), "r"(b[j][1]));
#else
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                        : "r"(a[i][2*h]), "r"(a[i][2*h+1]), "r"(b[j][h]));
                }
#endif
            }
        }
    }
}

__global__ void bmm256(const half* A, const half* B, half* C) {
    __shared__ Stage stage[2];
    int batch = blockIdx.z * kSize * kSize;
    int row = blockIdx.y * kTile;
    int col = blockIdx.x * kTile;
    A += batch + row * kSize;
    B += batch + col;
    C += batch + row * kSize + col;
    float acc[2][4][4] = {};
    load_stage(stage[0], A, B, 0);
    for (int t = 0; t < kSize / kDepth; ++t) {
        wait_stage();
        if (t + 1 < kSize / kDepth) load_stage(stage[(t + 1) % 2], A, B, (t + 1) * kDepth);
        compute_stage(stage[t % 2], acc);
    }
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    C += (warp / 2 * 32 + lane / 4) * kSize + warp % 2 * 32 + lane % 4 * 2;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float* d = acc[i][j];
            half* out = C + i * 16 * kSize + j * 8;
            *reinterpret_cast<half2*>(out) = __floats2half2_rn(d[0], d[1]);
            *reinterpret_cast<half2*>(out + 8 * kSize) = __floats2half2_rn(d[2], d[3]);
        }
    }
}

__global__ void scalar_bmm(const half* A, const half* B, half* C, int M, int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    const int batch = blockIdx.z;
    A += (batch * M + row) * K;
    B += batch * K * N + col;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum = fmaf(__half2float(A[k]), __half2float(B[k * N]), sum);
    }
    C[(batch * M + row) * N + col] = __float2half_rn(sum);
}

}  // namespace

extern "C" void solve(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
    const uintptr_t inputs = reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B);
    if (M == kSize && N == kSize && K == kSize && inputs % 16 == 0 &&
        reinterpret_cast<uintptr_t>(C) % alignof(half2) == 0) {
        bmm256<<<dim3(4, 4, BATCH), 128>>>(A, B, C);
    } else {
        scalar_bmm<<<dim3((N + 31) / 32, (M + 7) / 8, BATCH), dim3(32, 8)>>>(A, B, C, M, N, K);
    }
}
