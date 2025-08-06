# 算力修仙笔记：LeetGPU 三十题
网址：https://leetgpu.com/challenges

力扣GPU版，本文旨在提供详尽题解与思路。

## 简单题
### [向量相加](https://leetgpu.com/challenges/vector-addition)
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

运用了GPU向量化访存的特性，float4占32*4=128位，运行时调用向量寄存器，访存大幅加快。但是需要额外处理数据不足4的情况。

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

### [矩阵转置](https://leetgpu.com/challenges/matrix-transpose)
#### 基础实现
```cuda
#include <cuda_runtime.h>
#define BLOCK_SIZE 32
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(r < rows && c < cols)
        output[c * rows + r] = input[r * cols + c];
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
```
0.27386 ms 18.9th percentile

中规中矩，将输入的行列互换赋值到输出。
#### 共享内存基础版
```cuda
#define BLOCK_SIZE 16
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];
    if(r < rows && c < cols){
        tile[threadIdx.x][threadIdx.y] = input[r * cols + c];
        output[c * rows + r] = tile[threadIdx.x][threadIdx.y];
    }
}
```
0.18147 ms 35.9th percentile

仅仅访存时经过共享内存，就有大幅提升。原因是input数组的访存在warp内(threadIdx.x增长方向，即c）是连续的，走共享内存时可以合并。
#### 共享内存进阶版
```cuda
#define BLOCK_SIZE 16
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int transR = blockIdx.x * blockDim.x + threadIdx.y;
    int transC = blockIdx.y * blockDim.y + threadIdx.x;
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];
    if(r < rows && c < cols){
        tile[threadIdx.y][threadIdx.x] = input[r * cols + c];
    }
    __syncthreads();
    if(transR < cols && transC < rows){
        output[transR * rows + transC] = tile[threadIdx.x][threadIdx.y];
    }
}
```
0.178 ms 37.7th percentile

通过将共享内存的坐标转置，使得output访存也变得连续。但是线程之间存在交叉访问，需要做同步。

为什么共享内存可以跨行访问，但是输入输出数组不能呢？
因为共享内存访存效率非常高，和L1缓存相当，约几十个cycles，但input/output属于global显存，每次访问需要上百个cycles，不合并的话效率非常低下。

#### 终极版
```cuda
// solution.cu
#include <cuda_runtime.h>
#define TILE_DIM    32
#define BLOCK_ROWS   2

__global__ void transpose_cp_async_kernel(
    const float* __restrict__ input,
          float* __restrict__ output,
    int width, int height)  // width=cols, height=rows
{
    // 消除 bank conflict，多一列
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    const float* srcPtr = input + yIndex * width + xIndex;


    // 普通拷贝
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        int y = yIndex + i;
        if (y < height && xIndex < width)
            tile[threadIdx.y + i][threadIdx.x] =
                input[y * width + xIndex];
    }

    __syncthreads();

    // 写回全局内存，注意转置坐标
    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        int y = yIndex + i;
        if (y < width && xIndex < height) {
            output[y * height + xIndex] =
                tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

extern "C" void solve(
    const float* input, float* output,
    int rows, int cols)
{
    dim3 threads(TILE_DIM, BLOCK_ROWS);
    dim3 grid((cols + TILE_DIM - 1) / TILE_DIM,
              (rows + TILE_DIM - 1) / TILE_DIM);

    transpose_cp_async_kernel
        <<<grid, threads>>>(input, output, cols, rows);
    //cudaDeviceSynchronize();
}
```
0.06781 ms 98.1th percentile

<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/fb5d3574-3dd2-4d7c-8382-7ddcb9078ef3" />

主要有这几个优化点：
- 共享内存列号+1,让每行物理跨过bank，消除bank conflict
- tile为32正好是warp大小，row为2实现指令并行
- restrict限定符，设定内存不重叠方便编译优化
- 去除了显式的device同步




