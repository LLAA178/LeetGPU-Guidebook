# 算力修仙笔记：LeetGPU 三十题
网址：https://leetgpu.com/challenges
力扣GPU版，本文旨在提供详尽题解与思路。
## easy
### Vector Addition
环境: B200 + CUDA
#### 基础实现
```CUDA
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
        C[idx] = A[idx] + B[idx];
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```
0.09953 ms 11.4th percentile
这两个数据分别为运行耗时，百分比排名。耗时越少，百分比排名越大。
#### 向量化
```CUDA
#include <cuda_runtime.h>

__global__ void vector_add4(const float4* A, const float4* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N / 4)
        reinterpret_cast<float4*      >(C)[idx] = make_float4(A[idx].x + B[idx].x, A[idx].y + B[idx].y, A[idx].z + B[idx].z, A[idx].w + B[idx].w);
    else if (idx == N / 4)
        for(int i = N - N % 4; i < N; i++)
            C[i] = reinterpret_cast<const float*      >(A)[i] + reinterpret_cast<const float*      >(B)[i];
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N / 4 + threadsPerBlock - 1) / threadsPerBlock;
    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    vector_add4<<<blocksPerGrid, threadsPerBlock>>>(A4, B4, C, N);
    cudaDeviceSynchronize();
}
```
0.06307 ms 61.4th percentile
运用了GPU向量化访存的特性，float4占32*4=128位，运行时调用向量寄存器，访存大幅加快。

#### 编译加速
```CUDA
#include <cuda_runtime.h>
#include <cstdint>

__global__ void vecadd4_kernel(const float4* __restrict__ A4,
                               const float4* __restrict__ B4,
                               float4* __restrict__ C4,
                               int N4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N4) {
        float4 a = A4[idx];
        float4 b = B4[idx];
        C4[idx] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
}

__global__ void tail_kernel(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int start, int N) {
    int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int N4   = N >> 2;        // N / 4
    int tail = N & 3;         // N % 4

    if (N4 > 0) {
        const float4* A4 = reinterpret_cast<const float4*>(A);
        const float4* B4 = reinterpret_cast<const float4*>(B);
        float4*       C4 = reinterpret_cast<float4*>(C);

        int threads = 256;
        int blocks  = (N4 + threads - 1) / threads;
        vecadd4_kernel<<<blocks, threads>>>(A4, B4, C4, N4);
    }

    if (tail) {
        int start   = N & ~3; // 4 对齐的起点
        // 1 个 block、32 线程足够处理 <=3 的尾巴
        tail_kernel<<<1, 32>>>(A, B, C, start, N);
    }
}
```
0.05337 ms 80.0th percentile
加上__restrict__编译选项，并且把尾部另外起了一个kernel避免在主kernel中串行，提高效率。
