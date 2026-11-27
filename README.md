# 算力修仙笔记：LeetGPU 题解
网址：https://leetgpu.com/challenges
主页：https://leetgpu.com/EpicSamurai464

按题目记录 CUDA 解法、优化思路和部分完整实现。

## CUDA Best Practices

- [章节笔记](docs/cuda-best-practices/README.md)
- [实验文档](docs/cuda-best-practices/experiments/README.md)

## 索引

| 简单题 | 中等题 | 困难题 |
| --- | --- | --- |
| [vector-addition](#vector-addition) | [reduction](#reduction) | [multi-head-attention](#multi-head-attention) |
| [matrix-multiplication](#matrix-multiplication) | [softmax](#softmax) | [gpt-2-transformer-block](#gpt-2-transformer-block) |
| [matrix-transpose](#matrix-transpose) | [dot-product](#dot-product) |  |
| [color-inversion](#color-inversion) | [parallel-merge](#parallel-merge) |  |
| [1d-convolution](#1d-convolution) | [int8-quantized-matmul](#int8-quantized-matmul) |  |
| [reverse-array](#reverse-array) | [batch-normalization](#batch-normalization) |  |
| [relu](#relu) | [rotary-positional-embedding](#rotary-positional-embedding) |  |
| [leaky-relu](#leaky-relu) | [rms-normalization](#rms-normalization) |  |
| [rainbow-table](#rainbow-table) | [weight-dequantization](#weight-dequantization) |  |
| [matrix-copy](#matrix-copy) | [general-matrix-multiplication](#general-matrix-multiplication) |  |
| [count-array-element](#count-array-element) | [softmax-attention](#softmax-attention) |  |
| [count-2d-array-element](#count-2d-array-element) |  |  |
| [value-clipping](#value-clipping) |  |  |

## 简单题
### [vector-addition](https://leetgpu.com/challenges/vector-addition)

> 基础实现
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
0.09953 ms 11.4th percentile (B200)

这两个数据分别为运行耗时，百分比排名。耗时越少，百分比排名越大。

> 向量化
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
    int blocksPerGrid = ((N+3) / 4 + threadsPerBlock - 1) / threadsPerBlock;
    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    vector_add4<<<blocksPerGrid, threadsPerBlock>>>(A4, B4, C, N);
    cudaDeviceSynchronize();
}
```
0.06307 ms 61.4th percentile (B200)

运用了GPU向量化访存的特性，float4占32*4=128位，运行时调用向量寄存器，访存大幅加快。但是需要额外处理数据不足4的情况。

> 编译加速
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
0.05337 ms 80.0th percentile (B200)

加上__restrict__编译选项，并且把尾部另外起了一个kernel避免在主kernel中串行，提高效率。

### [matrix-multiplication](https://leetgpu.com/challenges/matrix-multiplication)
> 基础款
```cuda
#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < M && col < K){
        float sum = 0.0f;
        #pragma unroll
        for(int r = 0; r < N; r++){
            sum += A[row * N + r] * B[r * K + col];
        }
        C[row * K + col] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```
71.06379 ms 27.0th percentile (B200)

冷知识：GEMM是AI使用最多的算子。

### [matrix-transpose](https://leetgpu.com/challenges/matrix-transpose)

> 基础实现
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
0.27386 ms 18.9th percentile (B200)

中规中矩，将输入的行列互换赋值到输出。

> 共享内存基础版
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
0.18147 ms 35.9th percentile (B200)

仅仅访存时经过共享内存，就有大幅提升。原因是input数组的访存在warp内(threadIdx.x增长方向，即c）是连续的，走共享内存时可以合并。

> 共享内存进阶版
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
0.178 ms 37.7th percentile (B200)

通过将共享内存的坐标转置，使得output访存也变得连续。但是线程之间存在交叉访问，需要做同步。

为什么共享内存可以跨行访问，但是输入输出数组不能呢？
因为共享内存访存效率非常高，和L1缓存相当，约几十个cycles，但input/output属于global显存，每次访问需要上百个cycles，不合并的话效率非常低下。

> 终极版
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
0.06781 ms 98.1th percentile (B200)

<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/fb5d3574-3dd2-4d7c-8382-7ddcb9078ef3" />

主要有这几个优化点：
- 共享内存列号+1,让每行物理跨过bank，消除bank conflict
- tile为32正好是warp大小，row为2实现指令并行
- restrict限定符，设定内存不重叠方便编译优化
- 去除了显式的device同步

进一步了解：https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

### [color-inversion](https://leetgpu.com/challenges/color-inversion)

> 基础版
```cuda
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= width * height) return;
    image[idx * 4]      = 255 - image[idx * 4];
    image[idx * 4 + 1]  = 255 - image[idx * 4 + 1];
    image[idx * 4 + 2]  = 255 - image[idx * 4 + 2];
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    //cudaDeviceSynchronize();
}
```
0.06884 ms 21.2th percentile (B200)

每个线程获取32位，反转前24位rgb即可。

> 位运算
```cuda
__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= width * height) return;
    auto flip24 = [](int32_t x) -> int32_t {
    // XOR 后低 24 位翻转，高 8 位不变
        return x ^ 0xFFFFFF;
    };
    int temp = flip24(reinterpret_cast<int32_t*>(image)[idx]);
    reinterpret_cast<int32_t*>(image)[idx] = temp;
}
```
0.06728 ms 40.2th percentile (B200)

使用位运算函数直接反转32位，替换了逐个元素反转。

> 向量化
```cuda
#define BLOCK_SIZE 256
__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx > width * height / 4) return;
    auto flip24 = [](int32_t x) -> int32_t {
    // XOR 后低 24 位翻转，高 8 位不变
        return x ^ 0xFFFFFF;
    };
    if(idx == width * height / 4){
        if(width * height % 4 == 0)
            return;
        #pragma unroll
        for(int i = idx * 4; i < width * height; i++){
            int temp = flip24(reinterpret_cast<int32_t*>(image)[i]);
            reinterpret_cast<int32_t*>(image)[i] = temp;
        }
        return;
    }
    auto temp = reinterpret_cast<int4*>(image)[idx];
    //__syncthreads();
    reinterpret_cast<int4*>(image)[idx] = make_int4(flip24(temp.x), flip24(temp.y), 
    flip24(temp.z), flip24(temp.w));
}
```
0.03529 ms 78.4th percentile (B200)

经典向量化加快访存。

> 流水线
```cuda
#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

// 低 24 位取反（RGB），A 不变
__device__ __forceinline__ uint32_t flip24(uint32_t x) { return x ^ 0x00FFFFFFu; }

// 128-bit streaming load（L1 no-allocate + L2 128B），失败则退化为普通加载
__device__ __forceinline__ uint4 ld_stream128(const uint4* ptr) {
#if __CUDA_ARCH__ >= 900  // Hopper/Blackwell
    uint4 v;
    asm volatile(
        "ld.global.nc.L1::no_allocate.L2::128B.v4.u32 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(ptr));
    return v;
#else
    return *ptr;
#endif
}

// 128-bit streaming store
__device__ __forceinline__ void st_stream128(uint4* ptr, const uint4& v) {
#if __CUDA_ARCH__ >= 900
    asm volatile(
        "st.global.wb.v4.u32 [%0], {%1,%2,%3,%4};\n" ::
        "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
#else
    *ptr = v;
#endif
}

__global__ void invert_rgba_prefetch(uint32_t* __restrict__ data, size_t num_pixels)
{
    // 以 4 像素（16B）为一组
    size_t n4 = num_pixels >> 2;
    uint4* __restrict__ p4 = reinterpret_cast<uint4*>(data);

    size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    // 软件流水线：先预取一组
    if (tid < n4) {
        size_t j  = tid;
        uint4 r0  = ld_stream128(p4 + j);
        j += stride;

        // 主体：每轮预取下一组 r1，同时处理并写回 r0
        for (; j < n4; j += stride) {
            uint4 r1 = ld_stream128(p4 + j);

            r0.x = flip24(r0.x);
            r0.y = flip24(r0.y);
            r0.z = flip24(r0.z);
            r0.w = flip24(r0.w);
            st_stream128(p4 + (j - stride), r0);

            r0 = r1; // 滑动窗口
        }

        // 收尾：把最后一组 r0 写回
        r0.x = flip24(r0.x);
        r0.y = flip24(r0.y);
        r0.z = flip24(r0.z);
        r0.w = flip24(r0.w);
        st_stream128(p4 + (j - stride), r0);
    }

    // 处理余数（不足 4 像素）——单线程避免竞争
    if ((num_pixels & 3) && tid == 0) {
        for (size_t k = (n4 << 2); k < num_pixels; ++k) {
            data[k] = flip24(data[k]);
        }
    }
}

extern "C" void solve(unsigned char* image, int width, int height) {
    const size_t N = (size_t)width * (size_t)height;
    uint32_t* data = reinterpret_cast<uint32_t*>(image);

    int sm = 148; //cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    const int threads = BLOCK_SIZE;
    size_t n4 = N >> 2;

    int blocks;
    if (n4 == 0) blocks = 1;
    else {
        // 轻度超额订阅（8–12×SM），并受需求上限约束
        int target = sm * 22;
        size_t need = (n4 + threads - 1) / threads;
        blocks = std::max(1, (int)std::min((size_t)target, need));
    }

    invert_rgba_prefetch<<<blocks, threads>>>(data, N);
}
```
0.03195 ms 99.0th percentile (B200)

这里运用了一些高级特性
- 流水线：每轮预取下一次的输入，增加吞吐
- ptx：直接指定底层访存指令
- 限制block数：148为B200 SM数，防止带宽阻塞

### [1d-convolution](https://leetgpu.com/challenges/1d-convolution)
> 基础版
```cuda
#include <cuda_runtime.h>
__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= input_size - kernel_size + 1) return;
    float temp = 0.0f;
    #pragma unroll
    for(int i = 0; i < kernel_size; i++){
        float tempI = input[idx + i];
        float tempK = kernel[i];
        temp += tempI * tempK;
    }
    output[idx] = temp;
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size);
    //cudaDeviceSynchronize();
}
```
0.82399 ms 62.5th percentile (B200)

1维卷积。理解很难，动手写却很简单。

> 向量化
```cuda
#include <cuda_runtime.h>
__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= input_size - kernel_size + 1) return;
    float temp = 0.0f;
    #pragma unroll
    for(int i = 0; i < kernel_size / 4; i++){
        int p = i * 4;
        float4 tempI = make_float4(input[idx + p], input[idx + p + 1], input[idx + p + 2], input[idx + p + 3]);
        float4 tempK = make_float4(kernel[p], kernel[p + 1], kernel[p + 2], kernel[p + 3]);
        temp += tempI.x * tempK.x + tempI.y * tempK.y + tempI.z * tempK.z + tempI.w * tempK.w ;
    }
    for(int i = kernel_size & ~3; i < kernel_size; i++ ){
        float tempI = input[idx + i];
        float tempK = kernel[i];
        temp += tempI * tempK;
    }
    output[idx] = temp;
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 1024;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size);
    //cudaDeviceSynchronize();
}
```
0.73152 ms 80.0th percentile (B200)

这个向量化的独特处是，在单个线程的循环内做向量化而不是用更少线程数。

> 共享内存
```cuda
#include <cuda_runtime.h>

__global__ void conv1d_shared_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_size, int kernel_size)
{
    const int out_size = input_size - kernel_size + 1;
    const int out_start = blockIdx.x * blockDim.x;                  // 本 block 负责的输出起点（全局）
    const int out_end   = min(out_start + blockDim.x, out_size);    // 本 block 负责的输出终点（开区间）

    // 共享内存布局：[ 输入切片 (blockDim.x + K - 1) | kernel (K) ]
    extern __shared__ float smem[];
    float* s_in = smem;
    float* s_k  = smem + (blockDim.x + kernel_size - 1);

    // 这个 tile 对应的输入起点
    const int tile_in_start = out_start; // valid 卷积时输入对齐输出
    const int tile_in_len   = min(blockDim.x + kernel_size - 1, input_size - tile_in_start);

    // 1) cooperative load: 输入切片 -> shared
    #pragma unroll
    for (int t = threadIdx.x; t < tile_in_len; t += blockDim.x) {
        s_in[t] = input[tile_in_start + t];
    }
    #pragma unroll
    // 2) cooperative load: kernel -> shared
    for (int t = threadIdx.x; t < kernel_size; t += blockDim.x) {
        s_k[t] = kernel[t];
    }
    __syncthreads();

    // 3) 计算本线程的输出（若在有效范围内）
    const int out_idx = out_start + threadIdx.x;
    if (out_idx < out_end) {
        // 在共享内存上做点积：s_in[threadIdx.x + j] * s_k[j]
        float acc = 0.f;

        // 手动 4-步展开（避免 float4 对齐问题，且适用于任意对齐的起点）
        int j = 0;
        #pragma unroll
        for (; j + 3 < kernel_size; j += 4) {
            float a0 = s_in[threadIdx.x + j + 0];
            float a1 = s_in[threadIdx.x + j + 1];
            float a2 = s_in[threadIdx.x + j + 2];
            float a3 = s_in[threadIdx.x + j + 3];
            float b0 = s_k[j + 0];
            float b1 = s_k[j + 1];
            float b2 = s_k[j + 2];
            float b3 = s_k[j + 3];
            acc += a0*b0 + a1*b1 + a2*b2 + a3*b3;
        }
        for (; j < kernel_size; ++j) {
            acc += s_in[threadIdx.x + j] * s_k[j];
        }

        output[out_idx] = acc;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_size, int kernel_size)
{
    const int out_size = input_size - kernel_size + 1;
    if (out_size <= 0) return;

    const int threadsPerBlock = 1024;
    const int blocksPerGrid   = (out_size + threadsPerBlock - 1) / threadsPerBlock;

    // 共享内存大小 = 输入切片 + kernel
    const size_t shmem_bytes =
        (threadsPerBlock + kernel_size - 1 + kernel_size) * sizeof(float);

    conv1d_shared_kernel<<<std::min(148*22,blocksPerGrid), threadsPerBlock, shmem_bytes>>>(
        input, kernel, output, input_size, kernel_size);

}
```
0.68818 ms 88.4th percentile (B200)

把当前block输入和kernel都加载进共享内存，减少全局内存访问。

> 多级访存
```cuda
#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_THREADS 256
#define THREAD_OUT    8
#define MAX_K         4096

__constant__ float ck[MAX_K];

template<int BYTES>
__device__ __forceinline__ void cp_async_smem_gmem(void* smem_dst, const void* gmem_src) {
    uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst)); // 32-bit shared addr
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;\n" ::
        "r"(saddr), "l"(gmem_src), "n"(BYTES)
    );
}

__global__ void conv1d_bk_cpasync_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int input_size, int kernel_size)
{
    const int out_size = input_size - kernel_size + 1;
    if (out_size <= 0) return;

    const int out_per_blk = BLOCK_THREADS * THREAD_OUT;
    const int out_blk_beg = blockIdx.x * out_per_blk;
    if (out_blk_beg >= out_size) return;

    // tile 输入覆盖 [out_blk_beg, out_blk_beg + out_per_blk + K - 1)
    const int tile_in_start = out_blk_beg;
    const int stage_elems   = out_per_blk + kernel_size - 1;
    const int tile_in_len   = max(0, min(stage_elems, input_size - tile_in_start));

    extern __shared__ float smem[];
    float* s_in = smem;

    // ===== 全覆盖装载：float4 协同 + 尾部标量 =====
    const int vec_elems  = (tile_in_len & ~3);     // 4 对齐部分
    const int vec_chunks = (vec_elems >> 2);       // float4 个数

    // 关键修复：所有线程覆盖所有 chunk（不再按 warp 残缺分段）
    for (int i = threadIdx.x; i < vec_chunks; i += BLOCK_THREADS) {
        const char* gptr = reinterpret_cast<const char*>(input + tile_in_start) + i * 16;
        char*       sptr = reinterpret_cast<char*>(s_in)                               + i * 16;
        cp_async_smem_gmem<16>(sptr, gptr);
    }
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);

    // 剩余 <4 元素用标量补齐
    for (int t = vec_elems + threadIdx.x; t < tile_in_len; t += BLOCK_THREADS) {
        s_in[t] = input[tile_in_start + t];
    }
    __syncthreads();

    // ===== 计算：每线程 8 输出 =====
    const int tlocal  = threadIdx.x * THREAD_OUT;
    const int out_g0  = out_blk_beg + tlocal;
    const int n_valid = max(0, min(THREAD_OUT, out_size - out_g0));
    if (n_valid <= 0) return;

    float acc[THREAD_OUT] = {0};

    int j = 0;
    for (; j + 3 < kernel_size; j += 4) {
        float k0 = ck[j+0], k1 = ck[j+1], k2 = ck[j+2], k3 = ck[j+3];
        #pragma unroll
        for (int o = 0; o < THREAD_OUT; ++o) {
            float a0 = s_in[tlocal + o + j + 0];
            float a1 = s_in[tlocal + o + j + 1];
            float a2 = s_in[tlocal + o + j + 2];
            float a3 = s_in[tlocal + o + j + 3];
            acc[o] += a0*k0 + a1*k1 + a2*k2 + a3*k3;  // 让编译器自由做 FMA/ILP
        }
    }
    for (; j < kernel_size; ++j) {
        float kj = ck[j];
        #pragma unroll
        for (int o = 0; o < THREAD_OUT; ++o) {
            acc[o] += s_in[tlocal + o + j] * kj;
        }
    }

    #pragma unroll
    for (int o = 0; o < THREAD_OUT; ++o) {
        if (o < n_valid) output[out_g0 + o] = acc[o];
    }
}

extern "C" void solve(const float* __restrict__ input,
                      const float* __restrict__ kernel,
                      float* __restrict__ output,
                      int input_size, int kernel_size)
{
    const int out_size = input_size - kernel_size + 1;
    if (out_size <= 0) return;

    // kernel 放常量内存（D2D）
    cudaMemcpyToSymbol(ck, kernel, kernel_size * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    const int out_per_blk = BLOCK_THREADS * THREAD_OUT;
    dim3 block(BLOCK_THREADS);
    dim3 grid((out_size + out_per_blk - 1) / out_per_blk);

    // 只用一份 tile 的 shared（无流水，简单稳）
    size_t shmem_bytes = (size_t)(out_per_blk + kernel_size - 1) * sizeof(float);

    conv1d_bk_cpasync_kernel<<<grid, block, shmem_bytes>>>(input, output, input_size, kernel_size);
}
```
0.27392 ms 94.4th percentile (B200)

- 常量显存，命中常量缓存并走广播时，延迟大致在 ~10–20 个周期（SM 级别）量级，且一次取数供一整个 warp，等效到“每线程”近似可忽略。
- cp.async.cg.shared.global 中的 .cg 表示 cache global，只走 L2，不经过 L1。适合大块数据搬运（避免 L1 污染）。
- 每个输入元素平均会被重复用 ~K 次。把它放到 shared 后，这些重复访问都变成了 SMEM 命中（单周期、超高带宽），极大降低了 DRAM/L2 读流量。

### [reverse-array](https://leetgpu.com/challenges/reverse-array)
> 基础版
```cuda
#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx > N / 2) return;
    int other_idx = N - 1 - idx;
    float ta = input[idx];
    float tb = input[other_idx];
    input[idx] = tb;
    input[other_idx] = ta;
}
```
0.1115 ms 23.1th percentile (B200)

经典双指针反转数组。

> 共享内存
```cuda
#include <cuda_runtime.h>
#define BLOCK_DIM 256
__global__ void reverse_array(float* input, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= N) return;
    __shared__ float copy_of_input[BLOCK_DIM];
    copy_of_input[threadIdx.x] = input[idx];
    __syncthreads();
    input[N - 1 - idx] = copy_of_input[threadIdx.x];
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = BLOCK_DIM;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    //cudaDeviceSynchronize();
}
```
0.08007 ms 69.2th percentile (B200)

过共享内存做中转。

### [relu](https://leetgpu.com/challenges/relu)
```cuda
__global__ void relu_kernel(const float* input, float* output, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N){
        const float tmp = input[idx];
        output[idx] = tmp > 0 ? tmp : 0;
    }
}
```
0.09597 ms 15.4th percentile (B200)

常用激活函数之一，消灭负数。

### [leaky-relu](https://leetgpu.com/challenges/leaky-relu)
```cuda
__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= N)    return;
    auto tmp = input[idx];
    output[idx] = tmp >= 0 ? tmp : tmp * 0.01;
}
```
0.18015 ms 6.7th percentile (B200)

仁慈版relu。

### [rainbow-table](https://leetgpu.com/challenges/rainbow-table)
```cuda
__global__ void fnv1a_hash_kernel(const int* input, unsigned int* output, int N, int R) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx>=N) return;
    auto temp = input[idx];
    while(R--){
        temp = fnv1a_hash(temp);
    }
    output[idx] = temp;
}
```
0.05202 ms 12.5th percentile (B200)

### [matrix-copy](https://leetgpu.com/challenges/matrix-copy)
```cuda
extern "C" void solve(const float* A, float* B, const int N) {
    cudaMemcpy(B, A, N*N*sizeof(float), cudaMemcpyDeviceToDevice);
}
```
0.0381 ms 52.0th percentile (B200)

### [count-array-element](https://leetgpu.com/challenges/count-array-element)
```cuda
__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    int warp_num = 0;
    warp_num += input[idx] == K;
    for(int stride = warpSize / 2; stride > 0; stride /= 2)
        warp_num += __shfl_down_sync(0xffffffff,warp_num,stride);
    if(threadIdx.x % warpSize == 0) atomicAdd(output,warp_num);
}
```
2.05445 ms 0.0th percentile (B200)

### [count-2d-array-element](https://leetgpu.com/challenges/count-2d-array-element)
```cuda
__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if(col == 0 && row < N){
        for(int i = 0; i < M; i++){
            bool equal = input[row * M + i] == K ;
            atomicAdd(output, equal);
        }
    }
}
```
34.01452 ms 0.0th percentile (B200)

### [value-clipping](https://leetgpu.com/challenges/value-clipping)
*环境: CUDA*

将每个浮点数限制在 `[lo, hi]` 内。两个版本都只启动一个 kernel，相邻线程连续读写。

> baseline

```CUDA
#include <cuda_runtime.h>

__global__ void clip_kernel(const float* input, float* output, float lo, float hi, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float value = input[i];
    output[i] = value > hi ? hi : (value < lo ? lo : value);
}

extern "C" void solve(const float* input, float* output, float lo, float hi, int n) {
    clip_kernel<<<(n + 255) / 256, 256>>>(input, output, lo, hi, n);
}
```
0.03 ms 74.1th percentile (A100)

每个线程处理一个元素，每个 block 使用 256 个线程。

> optimized

```CUDA
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int kThreads = 128;

__global__ void clip_pairs(const float2* input, float2* output, float lo, float hi, int pairs) {
    int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= pairs) return;
    float2 a = input[i];
    output[i] = make_float2(fminf(hi, fmaxf(lo, a.x)), fminf(hi, fmaxf(lo, a.y)));
}

__global__ void clip_scalar(const float* input, float* output, float lo, float hi, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = fminf(hi, fmaxf(lo, input[i]));
}
}  // namespace

extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    if (N <= 0) return;
    uintptr_t addresses = reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(output);
    if (N % 2 == 0 && addresses % alignof(float2) == 0) {
        int pairs = N / 2;
        clip_pairs<<<(pairs + kThreads - 1) / kThreads, kThreads>>>(
            reinterpret_cast<const float2*>(input), reinterpret_cast<float2*>(output), lo, hi, pairs);
    } else {
        clip_scalar<<<(N + kThreads - 1) / kThreads, kThreads>>>(input, output, lo, hi, N);
    }
}
```

0.02 ms 92.4th percentile (A100)

当长度为偶数且输入、输出地址均按 8 字节对齐时，每个线程用一次 `float2` 加载和一次存储处理两个元素。一个 warp 连续覆盖 64 个浮点数；相同数据量只需一半的线程执行索引计算和边界判断。每个 block 使用 128 个线程，仍覆盖 256 个元素。读写字节总量相同，实际收益取决于机型和启动开销。奇数长度或地址未对齐时走标量路径。

## 中等题
### [reduction](https://leetgpu.com/challenges/reduction)
> 分层归约
```CUDA
#include <cuda_runtime.h>
#define BLOCK_SIZE 1024
__global__ void reduction(const float* input, float* output, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    //1. copy to shared
    //2. shared reduction
    //3. warp reduction
    //4. block sum
    __shared__ float sharedMem[BLOCK_SIZE];
    //拷到共享内存（越界用0），不能提前return
    float v = (idx < N) ? input[idx] : 0.0f;
    sharedMem[threadIdx.x] = v;
    __syncthreads(); 

    for(int stride = blockDim.x / 2; stride >= warpSize; stride /= 2){
        if(threadIdx.x < stride){
            sharedMem[threadIdx.x] += sharedMem[threadIdx.x + stride];
        }
        __syncthreads();    
    }

    if(threadIdx.x < warpSize){
        float sum = sharedMem[threadIdx.x];          
        unsigned mask = 0xffffffff;
        sum += __shfl_down_sync(mask, sum, 16);
        sum += __shfl_down_sync(mask, sum, 8);
        sum += __shfl_down_sync(mask, sum, 4);
        sum += __shfl_down_sync(mask, sum, 2);
        sum += __shfl_down_sync(mask, sum, 1);
        if(threadIdx.x == 0){
            atomicAdd(output, sum);
        }
    }
}
// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {  
    reduction<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE ,BLOCK_SIZE>>>(input, output, N);
}
```
0.27248 ms 24.6th percentile (B200)

> sweet spot
```cuda
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#define FULL_MASK 0xffffffffu

__inline__ __device__ float warp_reduce_sum(float v) {
    v += __shfl_down_sync(FULL_MASK, v, 16);
    v += __shfl_down_sync(FULL_MASK, v, 8);
    v += __shfl_down_sync(FULL_MASK, v, 4);
    v += __shfl_down_sync(FULL_MASK, v, 2);
    v += __shfl_down_sync(FULL_MASK, v, 1);
    return v;
}

template<int BS>
__global__ void reduce_limit_blocks(const float* __restrict__ x,
                                    float* __restrict__ out,
                                    int n) {
    __shared__ float smem[BS];
    const int tid   = threadIdx.x;
    const int bid   = blockIdx.x;
    const int gsize = gridDim.x * (BS * 2);   // 2x 展开
    int i = bid * (BS * 2) + tid;

    // 线程本地累加（两元素/线程）
    float sum = 0.f;
    if (i < n)              sum += x[i];
    if (i + BS < n)         sum += x[i + BS];

    // grid-stride 回合也按 2x 展开
    for (i += gsize; i < n; i += gsize) {
        sum += x[i];
        if (i + BS < n) sum += x[i + BS];
    }

    smem[tid] = sum;
    __syncthreads();

    // block 内规约（到 32 停）
    #pragma unroll
    for (int s = BS / 2; s >= 32; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    // warp 规约
    if (tid < 32) {
        float v = smem[tid];
        v = warp_reduce_sum(v);
        if (tid == 0) atomicAdd(out, v);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    // 设备信息
    int dev = 0, sm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);

    // 自然网格（两元素/线程）
    int natural = (N + (BLOCK_SIZE * 2) - 1) / (BLOCK_SIZE * 2);

    // 关键：限制但别太小。经验：SM * 6~10 比较稳
    int grid = sm * 8;
    if (natural < grid) grid = natural;
    if (grid < 1) grid = 1;

    reduce_limit_blocks<BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(input, output, N);
}
```
0.05636 ms 60.0th percentile (B200)

限制block数 + 双元素读写 + 精心设计参数，吃到了带宽甜点位。
> 双层旋风
```cuda
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256   // 可试 128/192/320 做 AB
#endif

#define FULL_MASK 0xffffffffu

__inline__ __device__ float warp_reduce_sum(float v) {
    v += __shfl_down_sync(FULL_MASK, v, 16);
    v += __shfl_down_sync(FULL_MASK, v, 8);
    v += __shfl_down_sync(FULL_MASK, v, 4);
    v += __shfl_down_sync(FULL_MASK, v, 2);
    v += __shfl_down_sync(FULL_MASK, v, 1);
    return v;
}

template<int BS>
__global__ void reduce_fast(const float* __restrict__ x,
                            float* __restrict__ out,
                            int n) {
    const int tid      = threadIdx.x;
    const int lane     = tid & 31;
    const int warp_id  = tid >> 5;             // 0..(BS/32-1)
    const int warps    = BS / 32;

    float sum = 0.f;

    // -------- 向量化主循环：float4 + 2× 展开 => 每迭代处理 8 个元素/线程 --------
    const bool aligned16 = ((reinterpret_cast<uintptr_t>(x) & 15u) == 0);
    if (aligned16) {
        const int n4 = n >> 2;                 // 以 float4 计
        const float4* __restrict__ v = reinterpret_cast<const float4*>(x);

        int j = blockIdx.x * BS + tid;         // 每线程一个向量索引
        const int stride4 = gridDim.x * BS;

        // 每回合拿两组：j 与 j+stride4（手动 2× 展开）
        for (; j < n4; j += (stride4 << 1)) {
            float4 a = v[j];
            sum += a.x + a.y + a.z + a.w;

            int j2 = j + stride4;
            if (j2 < n4) {
                float4 b = v[j2];
                sum += b.x + b.y + b.z + b.w;
            }
        }

        // 处理 4 对齐剩余的尾巴（0~3 个），仍用 grid-stride
        int base = n4 << 2;
        for (int i = base + blockIdx.x * BS + tid; i < n; i += gridDim.x * BS) {
            sum += x[i];
        }
    } else {
        // 非 16B 对齐：回退到标量 2× 展开
        const int gsize = gridDim.x * (BS * 2);
        int i = blockIdx.x * (BS * 2) + tid;
        if (i < n)       sum += x[i];
        if (i + BS < n)  sum += x[i + BS];
        for (i += gsize; i < n; i += gsize) {
            sum += x[i];
            if (i + BS < n) sum += x[i + BS];
        }
    }

    // -------- 纯 warp 规约 + 仅一次同步 --------
    sum = warp_reduce_sum(sum);

    __shared__ float smem[BS / 32];            // 每个 warp 一个槽
    if (lane == 0) smem[warp_id] = sum;
    __syncthreads();                           // 仅此一次

    // 由 warp0 做最后规约
    float block_sum = 0.f;
    if (warp_id == 0) {
        block_sum = (tid < warps) ? smem[lane] : 0.f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) atomicAdd(out, block_sum);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int dev = 0, sm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);

    // grid 适度加大以饱和带宽。经验：SM * 10~12 往往比 *8 更满
    int natural = (N + (BLOCK_SIZE * 8) - 1) / (BLOCK_SIZE * 8); // 估算向量化吞吐
    int grid = sm * 12;
    if (natural > 0) grid = min(grid, natural);
    if (grid < 1) grid = 1;

    // 若调用方没清零，这里别清。确保测试基准一致
    reduce_fast<BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(input, output, N);
}
```
0.04402 ms 98.7th percentile (B200)

两级warp归约 + 一次同步，避免了shared频繁读写。向量化。

### [softmax](https://leetgpu.com/challenges/softmax)
> 天地同寿
```cuda
#include <cuda_runtime.h>
#define BLOCK_SIZE 256
#define UPPER_DIV(A, B) ((A + B - 1) / B)
__global__ void block_max_kernel(const float* input, float* output, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    //same as reduction, but get max instead of sum
    __shared__ float sharedMem[BLOCK_SIZE];
    float v = (idx < N) ? input[idx] : 0.0f;
    sharedMem[threadIdx.x] = v;
    __syncthreads(); 

    for(int stride = blockDim.x / 2; stride >= warpSize; stride /= 2){
        if(threadIdx.x < stride){
            sharedMem[threadIdx.x] = fmaxf(sharedMem[threadIdx.x], sharedMem[threadIdx.x + stride]);
        }
        __syncthreads();    
    }

    if(threadIdx.x < warpSize){
        float blockMax = sharedMem[threadIdx.x];          
        for(int stride = warpSize / 2; stride > 0; stride /= 2){
            blockMax = fmaxf(blockMax, __shfl_down_sync(0xffffffff, blockMax, stride));
        }
        if(threadIdx.x == 0){
            output[blockIdx.x] = blockMax;
        }
    }  
}
__global__ void softmax_kernel(const float* input, float* output, float* inputMax, double* expSum, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float sharedMem[BLOCK_SIZE];
    float v = (idx < N) ? __expf(input[idx] - inputMax[0]) : 0.0f;
    sharedMem[threadIdx.x] = v;
    __syncthreads(); 

    for(int stride = blockDim.x / 2; stride >= warpSize; stride /= 2){
        if(threadIdx.x < stride){
            sharedMem[threadIdx.x] += sharedMem[threadIdx.x + stride];
        }
        __syncthreads();    
    }

    if(threadIdx.x < warpSize){
        float blockSum = sharedMem[threadIdx.x];          
        for(int stride = warpSize / 2; stride > 0; stride /= 2){
            blockSum += __shfl_down_sync(0xffffffff, blockSum, stride);
        }
        if(threadIdx.x == 0){
            atomicAdd(expSum, blockSum);
        }
    }
}
__global__ void softmax_tail_kernel(const float* input, float* output, float* inputMax, double* expSum, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= N ) return;
    auto sum = *expSum;
    output[idx] = __expf(input[idx] - inputMax[0]) / sum;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    const int blocksPerGrid = UPPER_DIV(N, BLOCK_SIZE);
    double *expSum;
    float *blockMax;
    cudaStream_t asyncStream;
    cudaStreamCreate(&asyncStream);
    cudaMallocAsync(&expSum, sizeof(double), asyncStream);
    cudaMallocAsync(&blockMax, sizeof(float) * blocksPerGrid, asyncStream);
    block_max_kernel<<<blocksPerGrid, BLOCK_SIZE, 0, asyncStream>>>(input, blockMax, N); 
    int blocks = UPPER_DIV(blocksPerGrid, BLOCK_SIZE);
    for(; blocks > 1 ; blocks = UPPER_DIV(blocks, BLOCK_SIZE)){
        block_max_kernel<<<blocks, BLOCK_SIZE, 0, asyncStream>>>(blockMax, blockMax, N);    
    }
    softmax_kernel<<<blocksPerGrid, BLOCK_SIZE, 0, asyncStream>>>(input, output, blockMax, expSum, N);
    softmax_tail_kernel<<<blocksPerGrid, BLOCK_SIZE, 0, asyncStream>>>(input, output, blockMax, expSum, N);
    cudaFreeAsync(expSum, asyncStream);
    cudaFreeAsync(blockMax, asyncStream);
    cudaStreamDestroy(asyncStream);
}
```
0.42211 ms 20.0th percentile (B200)

乍一看和归约很相似，只是从求和变成求和+归一化。

但是坑点在于，求和过程需要先求幂，而直接求幂则必然溢出，简单举例：e^1000能用浮点数表示吗？

解决方法是：求出最大值，然后把所有input除以最大值再求幂。一来不影响结果，因为分子分母都除。二来不会溢出，因为e^-1000能用浮点数表示。

流程变为求最值+求和+归一化。注意求最值和求和流程基本一样，只是atomicMax(float)没有实现，所以这里做了用循环blockMax替代。

> 浮点原子化
```cuda
#include <cuda_runtime.h>
#define BLOCK_SIZE 1024
#define UPPER_DIV(A, B) ((A + B - 1) / B)
__device__ float atomic_max_float(float *address, float val)
{
    int *address_as_int = (int *)address;
    int old = *address_as_int, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void max_kernel(const float* input, float* output, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    //same as reduction, but get max instead of sum
    __shared__ float sharedMem[BLOCK_SIZE];
    float v = (idx < N) ? input[idx] : 0.0f;
    sharedMem[threadIdx.x] = v;
    __syncthreads(); 

    for(int stride = blockDim.x / 2; stride >= warpSize; stride /= 2){
        if(threadIdx.x < stride){
            sharedMem[threadIdx.x] = fmaxf(sharedMem[threadIdx.x], sharedMem[threadIdx.x + stride]);
        }
        __syncthreads();    
    }

    if(threadIdx.x < warpSize){
        float blockMax = sharedMem[threadIdx.x];          
        for(int stride = warpSize / 2; stride > 0; stride /= 2){
            blockMax = fmaxf(blockMax, __shfl_down_sync(0xffffffff, blockMax, stride));
        }
        if(threadIdx.x == 0){
            atomic_max_float(output, blockMax);
        }
    }  
}
__global__ void softmax_kernel(const float* input, float* output, float* inputMax, float* expSum, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float sharedMem[BLOCK_SIZE];
    float v = (idx < N) ? __expf(input[idx] - *inputMax) : 0.0f;
    sharedMem[threadIdx.x] = v;
    __syncthreads(); 

    for(int stride = blockDim.x / 2; stride >= warpSize; stride /= 2){
        if(threadIdx.x < stride){
            sharedMem[threadIdx.x] += sharedMem[threadIdx.x + stride];
        }
        __syncthreads();    
    }

    if(threadIdx.x < warpSize){
        float blockSum = sharedMem[threadIdx.x];          
        for(int stride = warpSize / 2; stride > 0; stride /= 2){
            blockSum += __shfl_down_sync(0xffffffff, blockSum, stride);
        }
        if(threadIdx.x == 0){
            atomicAdd(expSum, blockSum);
        }
    }
}
__global__ void softmax_tail_kernel(const float* input, float* output, float* inputMax, float* expSum, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= N ) return;
    auto sum = *expSum;
    output[idx] = __expf(input[idx] - *inputMax) / sum;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    const int blocksPerGrid = UPPER_DIV(N, BLOCK_SIZE);
    float *expSum;
    float *inputMax;
    cudaMallocAsync(&expSum, sizeof(float), cudaStreamDefault);
    cudaMallocAsync(&inputMax, sizeof(float), cudaStreamDefault);
    max_kernel<<<blocksPerGrid, BLOCK_SIZE, 0, cudaStreamDefault>>>(input, inputMax, N); 
    softmax_kernel<<<blocksPerGrid, BLOCK_SIZE, 0, cudaStreamDefault>>>(input, output, inputMax, expSum, N);
    softmax_tail_kernel<<<blocksPerGrid, BLOCK_SIZE, 0, cudaStreamDefault>>>(input, output, inputMax, expSum, N);
}
```
0.0305 ms 88.6th percentile (B200)

使用atomicCAS实现了atomicMax(float)，与归约保持一致。

### [dot-product](https://leetgpu.com/challenges/dot-product)

这题本质上就是一个 reduction，只不过先做逐元素乘法，再做求和。

最稳的做法是两层归约：

- 每个线程先沿着全局 stride 累加自己的局部和
- warp 内用 `shuffle` 归约
- block 内只让一个 warp 收尾
- 最后对全局结果做一次 `atomicAdd`

这样共享内存只用来暂存 warp 结果，不需要整块树形归约。

```cuda
__device__ __forceinline__ float warp_reduce_sum(float x) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, offset);
  }
  return x;
}

__global__ void dot_product_kernel(const float* A, const float* B, float* result, int N) {
  float thread_sum = 0.0f;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < N; i += stride) {
    thread_sum = fmaf(A[i], B[i], thread_sum);
  }

  __shared__ float warp_sums[32];
  int lane = threadIdx.x & (warpSize - 1);
  int warp = threadIdx.x / warpSize;
  thread_sum = warp_reduce_sum(thread_sum);

  if (lane == 0) warp_sums[warp] = thread_sum;
  __syncthreads();

  if (warp == 0) {
    float block_sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? warp_sums[lane] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) atomicAdd(result, block_sum);
  }
}
```
0.05 ms 13.8th (A100)

这里的关键不是“乘法”，而是 reduction 路径怎么短。

完整代码：[`solutions/reduction/dot-product.cu`](./solutions/reduction/dot-product.cu)

### [parallel-merge](https://leetgpu.com/challenges/parallel-merge)

这题的核心不是比较本身，而是怎么把两个有序数组切成彼此独立的小段。

> baseline

基础版可以先不做 tile merge，而是直接让每个元素自己去找最终位置。

做法很直接：

- `A[i]` 用二分找出它前面应该有多少个 `B` 元素
- `B[j]` 用二分找出它前面应该有多少个 `A` 元素
- 然后各自把自己写到 `C` 的唯一位置

真正难想的是“二分里到底比较什么”：

- `A` 这边要找 `lower_bound(B, A[i])`
- `B` 这边要找 `upper_bound(A, B[j])`

为什么要这样分：

- 对 `A[i]` 来说，要数的是严格小于它的 `B`。所以一旦看到 `B[mid] >= A[i]`，答案一定在左边，这就是 `lower_bound`
- 对 `B[j]` 来说，要把所有等于它的 `A` 也算进前缀里。所以一旦看到 `A[mid] > B[j]`，才说明右边太大，这就是 `upper_bound`

这两个判断一左一右，重复元素就不会撞到同一个位置，稳定顺序也自然成立。

```cuda
#define BLOCK_SIZE 256

__device__ inline int binary_search(const float* B, int N, float k, bool is_upper) {
  int l = 0, r = N;
  while (l < r) {
    int mid = (l + r) / 2;
    if ((!is_upper && B[mid] >= k) || (is_upper && B[mid] > k)) {
      r = mid;
    } else {
      l = mid + 1;
    }
  }
  return l;
}

__global__ void asc_merge_A(const float* A, const float* B, float* C, int M, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M) return;
  int countB = binary_search(B, N, A[idx], false);
  C[countB + idx] = A[idx];
}

__global__ void asc_merge_B(const float* A, const float* B, float* C, int M, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  int countA = binary_search(A, M, B[idx], true);
  C[countA + idx] = B[idx];
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
  asc_merge_A<<<(M + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(A, B, C, M, N);
  asc_merge_B<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(A, B, C, M, N);
}
```
2.05 ms 50.0th (A100)

这版最适合先把“相等元素时为什么一个用 lower_bound、一个用 upper_bound”这件事想透。

> optimized

再往上走，才是 `merge path`：

- 先在全局对角线上给每个 block 分一段输出区间
- 用二分找到这段区间对应的 `A/B` 分界点
- 把这块输入搬到 shared memory
- 再在 block 内给每个线程分一个更小的对角线段

这样每个 block 都是在合并自己的一小块局部问题，不需要线程之间反复抢位置。

```cuda
template <typename T>
__device__ __forceinline__ int merge_path_partition(const T* a, int m, const T* b, int n, int diag) {
  int low = max(0, diag - n);
  int high = min(diag, m);
  while (low < high) {
    int mid = (low + high) >> 1;
    int j = diag - mid;
    if (j > 0 && mid < m && b[j - 1] >= a[mid]) low = mid + 1;
    else high = mid;
  }
  return low;
}

__global__ void parallel_merge_kernel(const float* a, const float* b, float* c, int m, int n) {
  __shared__ float tile[kTileItems];
  int block_diag = blockIdx.x * kTileItems;
  int total = m + n;
  if (block_diag >= total) return;

  int tile_count = min(kTileItems, total - block_diag);
  int a0 = merge_path_partition(a, m, b, n, block_diag);
  int a1 = merge_path_partition(a, m, b, n, block_diag + tile_count);
  int b0 = block_diag - a0;
  int b1 = block_diag + tile_count - a1;

  // load tile_a / tile_b to shared memory
  // each thread merges its own diagonal slice
  // write back contiguous outputs
}
```
1.02 ms 83.3th (A100)

这题真正值钱的是“先切输出，再反推输入”的思路，和 GEMM 那类题很像。

完整代码：[`solutions/reduction/parallel-merge.cu`](./solutions/reduction/parallel-merge.cu)

### [int8-quantized-matmul](https://leetgpu.com/challenges/int8-quantized-matmul)

这题最顺的想法不是先看 `A / B`，而是先看输出 `C`。

先把 `C` 切成 `16 x 16` 的小块。一个 block 负责一个 `C tile`，一个线程负责一个输出 `C[row, col]`。这样再反推输入，就会发现这个 block 需要的只是：

- `A` 的一段 `16 x K`
- `B` 的一段 `K x 16`

从 `C` 往回看，访存图案就是一个十字架。这个视角定下来以后，分块、边界、量化收尾都会清楚很多。

真正的技术难点不在 GEMM 主体，而在这三个地方：

- `zero point` 什么时候减，放在内层还是外层
- 量化收尾的舍入方式，是否和判题一致
- 想加速时，怎样保留 `int32` 累加结果再统一量化

> 公共前置

先记住这两个式子，后面三版都围着它们转：

```cuda
int corrected = raw - zp_b * row_sum_a - zp_a * col_sum_b + K * zp_a * zp_b;
int q = nearbyintf(raw_or_corrected * scale_a * scale_b / scale_c) + zp_c;
```

第一版可以直接在内层减 `zero point`。后两版更适合先算 `raw int32 GEMM`，最后一次性修正并量化。

> scalar

`scalar` 的核心不是“朴素三重循环”，而是先把 `C` 定成线程映射。

- 一个 block 对应一个 `16 x 16` 输出块
- 一个线程只管一个输出元素
- 每个线程沿着 `K` 方向扫一遍
- 从 `C[row, col]` 反推，只需要读 `A[row, :]` 和 `B[:, col]`

```cuda
constexpr int kTile = 16;

__device__ __forceinline__ int8_t clamp_to_int8(int x) {
  return static_cast<int8_t>(max(-128, min(127, x)));
}

__global__ void scalar_kernel(const int8_t* A,
                              const int8_t* B,
                              int8_t* C,
                              int M,
                              int N,
                              int K,
                              float scale_A,
                              float scale_B,
                              float scale_C,
                              int zp_A,
                              int zp_B,
                              int zp_C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  int acc = 0;
  for (int k = 0; k < K; ++k) {
    int a = static_cast<int>(A[row * K + k]) - zp_A;
    int b = static_cast<int>(B[k * N + col]) - zp_B;
    acc += a * b;
  }

  float x = static_cast<float>(acc) * scale_A * scale_B / scale_C;
  C[row * N + col] = clamp_to_int8(static_cast<int>(nearbyintf(x)) + zp_C);
}

extern "C" void solve(...) {
  dim3 threads(kTile, kTile);
  dim3 blocks((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
  scalar_kernel<<<blocks, threads>>>(...);
}
```
49.04 ms 4.5th (A100)

这一版里，共享内存不是必须的。先把输出映射和量化细节做对，比先上 shared memory 更重要。

> tensor core

到 A100 就可以考虑 `wmma` 的 `int8` 路线了。

- 一个 warp 负责一个 `16 x 16` 输出块
- 沿 `K` 方向按 `16` 递进
- 先得到 `int32 accumulator`
- 再把块内结果量化写回

```cuda
#include <mma.h>
using namespace nvcuda;

__global__ void wmma_int8_kernel(const int8_t* A,
                                 const int8_t* B,
                                 int8_t* C,
                                 int M,
                                 int N,
                                 int K,
                                 float scale_A,
                                 float scale_B,
                                 float scale_C,
                                 int zp_C) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 720
  constexpr int WM = 16, WN = 16, WK = 16;
  int warp_id = threadIdx.x / 32;
  int row = blockIdx.y * 64 + (warp_id / 2) * WM;
  int col = blockIdx.x * 32 + (warp_id % 2) * WN;

  wmma::fragment<wmma::accumulator, WM, WN, WK, int> acc;
  wmma::fill_fragment(acc, 0);

  for (int kk = 0; kk < K; kk += WK) {
    wmma::fragment<wmma::matrix_a, WM, WN, WK, signed char, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, signed char, wmma::row_major> b;
    wmma::load_matrix_sync(a, reinterpret_cast<const signed char*>(A + row * K + kk), K);
    wmma::load_matrix_sync(b, reinterpret_cast<const signed char*>(B + kk * N + col), N);
    wmma::mma_sync(acc, a, b, acc);
  }

  // store acc -> int32 tile, then quantize to int8
#endif
}
```
5.48 ms 95.5th (A100)

如果 `zero_point_A == 0 && zero_point_B == 0`，这条路径最直接。  
如果 `zero point` 非零，思路还是一样：

- 先用 tensor core 算 `raw_C`
- 再用公共公式做修正和量化

完整代码：[`solutions/quantization/int8-quantized-matmul.cu`](./solutions/quantization/int8-quantized-matmul.cu)

### [batch-normalization](https://leetgpu.com/challenges/batch-normalization)

BatchNorm 的关键不是公式，而是统计量和归一化之间的拆分方式。

> baseline

最朴素的写法是每个 channel 单独起一个 block：

- 先扫一遍求 `mean`
- 再扫一遍求 `variance`
- 最后第三遍写回归一化结果

```cuda
__global__ void batch_norm_baseline_kernel(const float* input,
                                           const float* gamma,
                                           const float* beta,
                                           float* output,
                                           int N,
                                           int C,
                                           float eps) {
  __shared__ float mean;
  __shared__ float variance;

  int channel = blockIdx.x;
  int tid = threadIdx.x;

  float thread_sum = 0.0f;
  for (int row = tid; row < N; row += blockDim.x) {
    thread_sum += input[row * C + channel];
  }

  float sum = block_reduce_sum(thread_sum);
  if (tid == 0) mean = sum / float(N);
  __syncthreads();

  float thread_sq_sum = 0.0f;
  for (int row = tid; row < N; row += blockDim.x) {
    float diff = input[row * C + channel] - mean;
    thread_sq_sum += diff * diff;
  }

  float sq_sum = block_reduce_sum(thread_sq_sum);
  if (tid == 0) variance = sq_sum / float(N);
  __syncthreads();

  float inv_std = rsqrtf(variance + eps);
  for (int row = tid; row < N; row += blockDim.x) {
    int index = row * C + channel;
    output[index] = gamma[channel] * ((input[index] - mean) * inv_std) + beta[channel];
  }
}
```
0.19 ms 43.8th (A100)

这版简单直接，但一个 channel 要来回扫三次，`N` 大时带宽利用不算好。

> tiled

思路变成：

- 先按 `(row tile, channel tile)` 分块累加 `sum / sumsq`
- 再单独 kernel 生成 `mean / inv_std`
- 最后按二维 tile 把输出写回

```cuda
__global__ void accumulate_stats_tiled_kernel(const float* input,
                                              float* sum,
                                              float* sumsq,
                                              int N,
                                              int C) {
  __shared__ float shared_sum[kStatsThreadRows][kTileChannels];
  __shared__ float shared_sumsq[kStatsThreadRows][kTileChannels];

  int channel = blockIdx.x * blockDim.x + threadIdx.x;
  int row_begin = blockIdx.y * kStatsRowsPerBlock;
  int row_end = min(row_begin + kStatsRowsPerBlock, N);

  float local_sum = 0.0f;
  float local_sumsq = 0.0f;
  if (channel < C) {
    for (int row = row_begin + threadIdx.y; row < row_end; row += blockDim.y) {
      float v = input[row * C + channel];
      local_sum += v;
      local_sumsq += v * v;
    }
  }

  shared_sum[threadIdx.y][threadIdx.x] = local_sum;
  shared_sumsq[threadIdx.y][threadIdx.x] = local_sumsq;
  __syncthreads();

  if (threadIdx.y == 0 && channel < C) {
    float total_sum = 0.0f, total_sumsq = 0.0f;
    for (int i = 0; i < kStatsThreadRows; ++i) {
      total_sum += shared_sum[i][threadIdx.x];
      total_sumsq += shared_sumsq[i][threadIdx.x];
    }
    atomicAdd(sum + channel, total_sum);
    atomicAdd(sumsq + channel, total_sumsq);
  }
}
```
0.06 ms 93.8th (A100)

这版的重点是把统计和写回都改成二维分块，让 `channel` 方向始终连续，适合 `N` 和 `C` 都比较大的情况。

这题最值得学的是：统计量是按 channel 聚合的，所以分块时要优先保证 channel 方向的连续性。

完整代码：[`solutions/normalization/batch-normalization.cu`](./solutions/normalization/batch-normalization.cu)

### [rotary-positional-embedding](https://leetgpu.com/challenges/rotary-positional-embedding)

RoPE 的核心不是矩阵乘，而是成对变换。

如果直接按元素看，很容易绕进去。先把一行看成两半：

- 每一行拆成前半和后半
- 每个位置只和自己的配对位置交互
- 低半区写 `q_low * cos - q_high * sin`
- 高半区写 `q_high * cos + q_low * sin`

> baseline

最直观的版本是一个 block 处理一个 token。

- 先把这一整行 `Q` 搬进 shared memory
- 每个线程处理这一行的一个维度
- 低半区去读高半区配对值，高半区去读低半区配对值

这样想法最顺，因为“配对元素一定在同一行里”，所以先把整行缓存下来，后面的交叉访问就只是 shared memory 里的读写。

```cuda
__global__ void ropeToken(float* Q, float* cos, float* sin, float* output, int M, int D) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  extern __shared__ float token[];
  int offset = bid * D;

  token[tid] = Q[offset + tid];
  __syncthreads();

  output[offset + tid] = token[tid] * cos[offset + tid];
  if (tid >= D / 2) {
    output[offset + tid] += token[tid - D / 2] * sin[offset + tid];
  } else {
    output[offset + tid] -= token[tid + D / 2] * sin[offset + tid];
  }
}

extern "C" void solve(float* Q, float* cos, float* sin, float* output, int M, int D) {
  ropeToken<<<M, D, D * sizeof(float)>>>(Q, cos, sin, output, M, D);
  cudaDeviceSynchronize();
}
```
1.53 ms 9.5th (A100)

这版的关键不是公式，而是先确认“旋转只发生在同一个 token 内部”，所以分块应该先按行切，再在行内找配对维度。

> optimized

再往上走，最自然的并行单位就不是“一个元素”，而是“一个 pair”。这样每个线程直接同时处理一对位置，不需要先把整行搬进 shared memory。`D == 128` 时还可以继续做 `float4` 向量化，一次处理 4 对。

```cuda
__global__ void rope_pairs_kernel(const float* Q,
                                  const float* cos,
                                  const float* sin,
                                  float* output,
                                  int M,
                                  int D) {
  size_t half = D >> 1;
  size_t pair_idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total_pairs = size_t(M) * half;
  if (pair_idx >= total_pairs) return;

  size_t row = pair_idx / half;
  size_t col = pair_idx - row * half;
  size_t base = row * size_t(D) + col;
  size_t high = base + half;

  float q_low = Q[base];
  float q_high = Q[high];
  output[base] = fmaf(q_low, cos[base], -q_high * sin[base]);
  output[high] = fmaf(q_high, cos[high], q_low * sin[high]);
}
```
1.22 ms 97.6th (A100)

`D = 128` 这类固定热路径，可以专门做 `float4` 版本，把前半和后半各按向量读出来再一起写回。

完整代码：[`solutions/transformer/rotary-positional-embedding.cu`](./solutions/transformer/rotary-positional-embedding.cu)

### [rms-normalization](https://leetgpu.com/challenges/rms-normalization)

RMSNorm 可以拆成两个阶段：

- 先求 `sum(x^2)`，拿到全局 scale
- 再做 `y = x * scale * gamma + beta`

> baseline

基础版就是老老实实拆成三步：

- 第一个 kernel 先算每个 block 的平方和
- 第二个 kernel 再把这些 partial sums 归约成总和
- 把总和拷回 host 算出 `inv_rms`
- 第三个 kernel 做归一化写回

这个版本不追求极限性能，但结构最直观，也最容易验证对不对。

```cuda
namespace {

constexpr int kBlockSize = 256;

__global__ void rms_sum_squares(const float* input, double* partial_sums, int N) {
  __shared__ double shared[kBlockSize];

  int tid = threadIdx.x;
  int global_tid = blockIdx.x * blockDim.x + tid;
  int stride = blockDim.x * gridDim.x;

  double local_sum = 0.0;
  for (int idx = global_tid; idx < N; idx += stride) {
    double value = static_cast<double>(input[idx]);
    local_sum += value * value;
  }
  shared[tid] = local_sum;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) shared[tid] += shared[tid + offset];
    __syncthreads();
  }

  if (tid == 0) partial_sums[blockIdx.x] = shared[0];
}

__global__ void rms_reduce_partial_sums(const double* partial_sums, double* total_sum, int count) {
  __shared__ double shared[kBlockSize];

  int tid = threadIdx.x;
  double local_sum = 0.0;
  for (int idx = tid; idx < count; idx += blockDim.x) {
    local_sum += partial_sums[idx];
  }
  shared[tid] = local_sum;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) shared[tid] += shared[tid + offset];
    __syncthreads();
  }

  if (tid == 0) total_sum[0] = shared[0];
}

__global__ void rms_normalize(const float* input, float gamma, float beta, float* output, int N,
                              float inv_rms) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    output[idx] = input[idx] * inv_rms * gamma + beta;
  }
}

}  // namespace
```
0.23 ms 9.9th (A100)

> optimized

真正麻烦的是这两个阶段之间要共享一个标量，而且不想把整段流程拆成太多 kernel。这里比较顺的做法是：

- 优先尝试 cooperative launch
- 用整个 grid 先归约平方和
- `grid.sync()` 后由一个线程写出全局 scale
- 再 `grid.sync()` 一次，所有线程直接做归一化写回

这样只要 cooperative launch 成功，整题就是单 kernel 完成。再配一个普通 fallback 路径兜底。

```cuda
template <int kThreads>
__global__ void rms_cooperative_kernel(const float* input,
                                       float gamma,
                                       float beta,
                                       float* output,
                                       int N,
                                       float eps) {
  cg::grid_group grid = cg::this_grid();
  float local_sum = 0.0f;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    local_sum += input[i] * input[i];
  }

  float block_sum = block_reduce_sum<kThreads>(local_sum);
  if (threadIdx.x == 0) atomicAdd(&g_sum_sq, block_sum);
  grid.sync();

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    g_scale = rsqrtf(g_sum_sq / float(N) + eps) * gamma;
  }
  grid.sync();

  float scale = g_scale;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    output[i] = fmaf(input[i], scale, beta);
  }
}
```
0.04 ms 74.1th (A100)

这题最值钱的点不是公式，而是怎么把“先求 scale，再用 scale”压成一条高效执行路径。

完整代码：[`solutions/normalization/rms-normalization.cu`](./solutions/normalization/rms-normalization.cu)

### [weight-dequantization](https://leetgpu.com/challenges/weight-dequantization)

这题最容易写成纯标量版，但真正该利用的是一个 tile 里所有元素共享同一个 scale。

也就是说，输出矩阵虽然是逐元素写回，但 scale 不是逐元素查的，而是按块查的。顺着这个观察，优化方向就很自然：

- 如果 `N` 能按 4 对齐，优先用 `float4` 向量化
- 如果 `tile_size` 比较大，直接让一个 block 对应一个 scale tile
- 这样同一个 block 里的线程都复用同一个 scale，不必反复算索引

```cuda
template <int kTileSize>
__global__ void weight_dequant_tile_vec4_kernel(const float* X, const float* S, float* Y, int M, int N) {
  constexpr int kBlockRows = 8;
  constexpr int kVecWidth = 4;

  int tile_col = blockIdx.x;
  int row = blockIdx.y * kBlockRows + threadIdx.y;
  int col = tile_col * kTileSize + threadIdx.x * kVecWidth;
  if (row >= M || col >= N) return;

  int scale_cols = (N + kTileSize - 1) / kTileSize;
  float scale = S[(row / kTileSize) * scale_cols + tile_col];
  float4 v = reinterpret_cast<const float4*>(X + row * N + col)[0];
  reinterpret_cast<float4*>(Y + row * N + col)[0] =
      make_float4(v.x * scale, v.y * scale, v.z * scale, v.w * scale);
}
```
0.34 ms 75.0th (A100)

这题和普通逐元素 kernel 的差别在于：优化目标不是少做乘法，而是把“同一个 tile 共用同一个 scale”这件事吃干净。

完整代码：[`solutions/quantization/weight-dequantization.cu`](./solutions/quantization/weight-dequantization.cu)

### [general-matrix-multiplication](https://leetgpu.com/challenges/general-matrix-multiplication-gemm)
*环境: CUDA，Tensor Core 版本需要 SM 7.5 或更高架构*

这题要算半精度浮点（FP16）通用矩阵乘加 `C = alpha * (A × B) + beta * C`。输入和输出矩阵都是 FP16，中间点积用单精度浮点（FP32）累加保证精度，最后转回 FP16 写回。

举个小例子：假设 `M = N = K = 1024`，如果每个线程独立算 `C` 里的 1 个数，每算 1 个输出就要从显存读 2048 个半精度数，算力单元大部分时间都在等数据。

> baseline

每个线程负责 1 个输出元素，沿 `K` 累加 FP32 点积，最后乘以 `alpha` 并加上 `beta * C`（`beta == 0` 时跳过读取原 `C`）。

```CUDA
float acc = 0.0f;
for (int k = 0; k < K; ++k) {
    acc = fmaf(__half2float(a_row[k]), __half2float(B[static_cast<size_t>(k) * N + col]), acc);
}
float out = alpha * acc;
if (beta != 0.0f) out = fmaf(beta, __half2float(C[c_idx]), out);
C[c_idx] = __float2half(out);
```

> tensor core

为了把 T4 的矩阵计算单元（Tensor Core）跑满，优化版对整除且对齐的形状做了三步加速（其余边界形状自动走标量版）：
- **分块与双缓冲预取**：一个线程块（4 个线程束、128 线程）负责 `64×64` 的输出块，沿 `K` 每次推进 32。线程用 `uint4`（一次 16 字节、8 个 FP16）把 `A/B` 搬进共享内存（片上高速缓存），在计算当前块的同时把下一块提前读进寄存器，把访存延迟藏在计算后面。
- **`ldmatrix.x4` 批量加载**：共享内存每行多留 8 个 FP16 的空位（padding）避免存储体冲突（bank conflict）；每个 `k = 8` 子步只需 2 条 `ldmatrix.x4`（片上矩阵加载指令），就能把 2 块 `A` 和 4 块 `B` 一次喂给 `mma.sync` 矩阵乘加指令。
- **`half2` 直接写回**：按 `m16n8k8` 的输出布局，每个线程正好握着相邻两个元素，算完 `alpha/beta` 后直接用 `half2` 写回 `C`，省掉共享内存中转，`beta == 0` 时也不读原 `C`。

优化版在 LeetGPU 官方 T4 上通过全部测试，耗时 **0.29 ms**（第 **95.6** 百分位）；本地 T4 对照测试中，基线版 `1024×1024×1024` 耗时 `3.74 ms`，优化版降至 `0.105 ms`（快了 35 倍）。

完整代码：[标量基线版](solutions/matmul/general_matrix_multiplication_baseline.cu) · [双缓冲 MMA 优化版](solutions/matmul/general_matrix_multiplication_mma.cu)

### [softmax-attention](https://leetgpu.com/challenges/softmax-attention)
*环境: CUDA*

这题要算缩放点积注意力：给查询矩阵 `Q[M, d]`、键矩阵 `K[N, d]` 和值矩阵 `V[N, d]`（单精度浮点 FP32），按 `softmax(Q × K^T / sqrt(d)) × V` 算出输出矩阵 `output[M, d]`。

举个小例子：假设 `M = 512, N = 256, d = 128`。如果先把 `512 × 256` 的注意力分数表写回全局显存再做 softmax，会多出整整一轮显存读写；直接用 Online Softmax（在线归一化）一边扫 `K/V` 一边更新最大值和指数和，就能把中间分数表省掉。

> baseline

每个线程束（warp，32 个线程）负责 1 行 `Q`，把这一行 `Q` 和输出累加器留在寄存器里。遍历每一行 `K/V` 时，线程束内用洗牌归约算出点积分数，在线更新当前最大值 `m`、指数和 `l` 和加权输出，最后统一除以 `l`。

```CUDA
float m_new = fmaxf(m, score);
float alpha = expf(m - m_new);
float p = expf(score - m_new);
l = l * alpha + p;
for (int i = 0; i < elems; ++i) {
    o_reg[i] = fmaf(p, v_ptr[col], o_reg[i] * alpha);
}
m = m_new;
```

> 4-row warp (v1)

基线版里每个线程束只算 1 行 `Q`，每扫一行 `K/V` 只用一次。第一版优化让每个线程束把 4 行 `Q` 和 4 行累加器常驻寄存器，每读入 1 行 `K/V` 同时算 4 行，减少 4 倍 `K/V` 读取，并用 `__expf`（硬件快速指数指令）替代软件 `expf`。这一版在 LeetGPU 官方 T4 上通过全部测试，耗时 **0.36 ms**（第 **82.4** 百分位）。

> optimized (8×16 float4 FlashAttention)

v1 版在 `Q × K^T` 阶段每算一个点积都要做整束 32 线程洗牌归约，指令开销偏高。优化版改为二维共享内存分块（`Br = 8, Bc = 32`）加全路径 `float4`（一次读写 4 个 FP32）向量化：
- **铺满 SM 的分块规模**：`M = 512` 按 `Br = 8` 切出 64 个线程块，正好铺满 T4 的全部 40 个流多处理器（SM）；共享内存 `sKV` 先装 `K` 再复用装 `V`，总占用约 22 KB。
- **每行 16 线程并行 + 全 `float4` 向量化**：每块 `8 × 16 = 128` 线程，每行 `Q` 分到 16 个线程。算 `Q × K^T` 时每线程只算 2 列 `K`，算 `P × V` 和写回时每线程只管 2 个 `float4`（`d = 128`），彻底去掉 `Q × K^T` 的线程束归约，寄存器压到 85 个且零溢出。
- **16 线程子束在线归一化**：softmax 只需 4 次洗牌（`__shfl_xor_sync`）即可完成整行最大值与求和归约。其余 `d % 4 != 0` 的维度自动走通用回退路径。

这两步把 LeetGPU 官方 T4 实测耗时从 **0.36 ms** 压到 **0.27 ms**（缩短 25%，第 **94.2** 百分位）。

完整代码：[Online Softmax 基线版](solutions/transformer/softmax_attention_baseline.cu) · [4-Row 线程束复用版](solutions/transformer/softmax_attention_flash.cu) · [8×16 float4 优化版](solutions/transformer/softmax_attention_opt.cu)

## 困难题

### [multi-head-attention](https://leetgpu.com/challenges/multi-head-attention)

这题可以先从 **Multi-Head Attention** 的三个核心阶段入手：`Q @ K^T`、`softmax`、`P @ V`。把这三步拆清楚之后，再看哪些维度值得分块、哪些数据值得搬进 shared memory 复用，整体思路就会顺很多。

> baseline
```cuda
__global__ void mha_kernel(const float* Q,
                           const float* K,
                           const float* V,
                           float* output,
                           int N,
                           int d_model,
                           int h) {
  int head_size = d_model / h;
  int q_row = blockIdx.x;
  int head_idx = blockIdx.y;
  int tid = threadIdx.x;

  extern __shared__ float shared_mem[];
  float* scores = shared_mem;
  float* q_vec = scores + N;

  // load one query row for this head
  for (int dim = tid; dim < head_size; dim += blockDim.x) {
    q_vec[dim] = Q[q_row * d_model + head_idx * head_size + dim];
  }
  __syncthreads();

  // compute the whole score row
  for (int k_row = tid; k_row < N; k_row += blockDim.x) {
    float dot = 0.0f;
    for (int dim = 0; dim < head_size; ++dim) {
      dot += q_vec[dim] * K[k_row * d_model + head_idx * head_size + dim];
    }
    scores[k_row] = dot * rsqrtf(float(head_size));
  }
  __syncthreads();

  // softmax + weighted sum over V
  ...
}
```
10.94 ms 36.4th (A100)

这版映射很直接：

- 一个 block 处理一个 `(query row, head)`。
- 先把当前 query 的一整行 head 向量放进 shared memory，避免后面每个 key 都重复读 Q。
- 再把长度为 `N` 的整行 score 算出来，做 softmax，最后拿权重去乘 `V`。

它的好处是非常好讲，也非常容易对拍。缺点也很明显：

- 同一个 block 要完整扫一遍 `N` 个 key。
- score 行需要先全部落到 shared memory，再做 softmax。
- `K` 和 `V` 的 tile 复用还不够积极。

> optimized
```cuda
template <int kWarpRows>
__global__ void flash_attn_small_kernel(const float* Q,
                                        const float* K,
                                        const float* V,
                                        float* output,
                                        int N,
                                        int d_model,
                                        int head_size,
                                        float scale) {
  int tx = threadIdx.x;              // lane inside a warp
  int ty = threadIdx.y;              // warp index inside a block
  int q_row = blockIdx.x * kWarpRows + ty;
  int head_offset = blockIdx.y * head_size;

  __shared__ float sQ[kWarpRows][33];
  __shared__ float sK[32][33];
  __shared__ float sP[kWarpRows][33];

  float output_local[4] = {0.0f};
  float running_max = -INFINITY;
  float running_sum = 0.0f;

  for (int kv_block = 0; kv_block < (N + 31) / 32; ++kv_block) {
    // load tiled Q/K, compute one 32-key score block
    // warp_allreduce_max / warp_allreduce_sum for online softmax
    // reuse shared memory again as V tile and accumulate P @ V
  }

  // normalize with running_sum and write output
}
```
1.51 ms 95.5th (A100)

这一版的重点在于把分块逻辑真正落到 kernel 映射里，而不只是把实现换成一个 FlashAttention 风格的名字。

先看 block 怎么切：

- `blockIdx.y` 仍然对应 head。
- `blockIdx.x` 不再只对应一个 query row，而是对应一小组 query row。
- block 内每个 warp 负责一行 query，所以同一个 block 可以同时处理多行 Q。

这样切的目的，是让一个 block 读进来的 `K/V` tile 能被多个 query row 复用，让同一份 tile 服务多行 query。

再看两层 tile：

1. `N` 维按 32 个 key/value 切块。  
这表示每一轮只处理 `32` 个 key。当前这 `32` 个 key 的 `K` 和 `V` 会先搬进 shared memory，再被 block 内多个 warp 复用。

2. `head_size` 维也按 32 列切块。  
因为一个 warp 正好是 32 个 lane，最自然的做法就是让 lane 直接对应 tile 内的一列。比如 `head_size = 128` 时，会被拆成 4 个 `32`-wide 子块，逐块做乘加。

这两个维度一起切，kernel 里实际做的是：

- 外层循环扫 `kv_block`，每次处理 `32` 个 key。
- 内层循环扫 `d_k tile`，每次处理 `32` 个 head 维度。
- 同一个 warp 的 32 个 lane，一起把“当前 query 对当前 32 个 key 的 score”算出来。

online softmax 是这套分块能成立的关键。因为我们不想把整行 `N` 个 score 全存下来再做 softmax，而是希望处理完一个 `32-key` 块，就把这一块的信息合并进全局结果。所以每个 warp 持续维护两份状态：

- `running_max`：到当前为止见过的最大 score
- `running_sum`：在这个最大值基准下累计的 `exp` 和

当前 tile 的 score 算出来后：

- 先用 `warp_allreduce_max` 求出这 32 个 score 的最大值。
- 再用 `warp_allreduce_sum` 求出这 32 个 `exp(score - new_max)` 的和。
- 然后把历史累计结果按 `alpha = exp(old_max - new_max)` 统一重标定，再把新块加进去。

这里一定要用 **all-reduce**，不能用只让 `lane0` 正确的 down-reduce。因为 online softmax 里每个 lane 后续都还要继续参与自己的输出列累加，所以所有 lane 都必须拿到同一个 `max` 和 `sum`。

`P @ V` 这一步也复用了同一块 shared memory：

- 前半段 `sK` 存的是 `K` tile，用来算 score。
- score 和 softmax 做完后，`sK` 这块空间立刻改存 `V` tile。
- `sP` 里已经放好了当前 32 个 key 的概率权重，于是每个 lane 直接做一段长度为 32 的加权和，累加到 `output_local`。

当前实现的调度策略是：

- `head_size <= 128`：走专门化 small-kernel。
- `head_size > 128`：走通用 fallback kernel，优先保证题目全范围都能覆盖。
- `head_size <= 64` 时一个 block 放 8 个 warp；`head_size = 128` 时一个 block 放 4 个 warp，给更宽的头留出更多共享内存和寄存器余量。

所以这题更值得单独看的，是下面三件事怎样一起配合起来：

- `(query row, head)` 怎么映射到 warp / block
- `N` 和 `head_size` 两个维度为什么都要切 tile
- online softmax 怎么把“分块处理 score”变成合法的整体 softmax

完整代码：[`solutions/transformer/multi-head-attention.cu`](./solutions/transformer/multi-head-attention.cu)

### [gpt-2-transformer-block](https://leetgpu.com/challenges/gpt-2-transformer-block)

这题不是某一个 kernel 难，而是整条计算图怎么串起来。

如果直接看 packed weights 很容易乱，最稳的方式是先把 block 拆成五段：

1. `LN1`
2. `QKV -> attention -> output projection`
3. 第一次 residual
4. `LN2`
5. `FC -> GELU -> projection -> residual`

当前实现走的是很朴素但清晰的分段流水：

- LayerNorm 单独 kernel
- QKV projection 单独 kernel
- attention 按 `(token, head)` 展开
- 输出投影走通用线性层
- FFN 再走一遍 `up -> GELU -> proj`

```cuda
extern "C" void solve(const float* x, float* output, const float* weights, int seq_len) {
  // allocate ln1 / qkv / attn / residual / ln2 / ffn buffers

  layerNorm<<<seq_len, 1>>>(x, ln1, weights, seq_len);
  qkv<<<seq_len, 256>>>(ln1, qkv_buf, weights, seq_len);

  attn<<<dim3(seq_len, kNumHeads), kHeadDim,
         static_cast<size_t>(seq_len) * sizeof(float)>>>(qkv_buf, attn_concat, weights, seq_len);

  linear_bias<<<linear_grid, 256>>>(
      attn_concat, attn_proj, weights + kWAttnOffset, weights + kBAttnOffset,
      seq_len, kDModel, kDModel);

  add_residual<<<(token_count + 255) / 256, 256>>>(x, attn_proj, residual1, token_count);
  layerNorm2<<<seq_len, 1>>>(residual1, ln2, weights, seq_len);
  ffn<<<seq_len, 256>>>(ln2, ff2, weights, seq_len);
  add_residual<<<(token_count + 255) / 256, 256>>>(residual1, ff2, output, token_count);
}
```

这一版的价值主要在于把 GPT-2 block 的结构完整跑通，而不是把每一层都做到最优。

里面最值得单独看的是 attention 这步：实现按 `(row, head)` 展开，一个 block 处理一个 query token 的一个 head，先算整行 score，再做 softmax，再拿权重去乘 `V`。

```cuda
__global__ void attn(const float* qkv, float* output, const float* weights, int seq_len) {
  int row = blockIdx.x;
  int head = blockIdx.y;
  int lane = threadIdx.x;
  if (row >= seq_len || head >= kNumHeads || lane >= kHeadDim) return;

  extern __shared__ float scores[];

  if (lane == 0) {
    // compute full score row
    // softmax over seq_len
  }
  __syncthreads();

  float acc = 0.0f;
  for (int j = 0; j < seq_len; ++j) {
    acc += scores[j] * V(row=j, head, lane);
  }
  output[row * kDModel + head * kHeadDim + lane] = acc;
}
```

所以这题的重点不是某一个花哨技巧，而是先把 transformer block 的数据流理顺：

- packed weights 怎么拆
- 中间 buffer 怎么接
- residual 放在什么位置
- attention 的 `(token, head, dim)` 映射怎么落到 kernel

完整代码：[`solutions/transformer/gpt-2-transformer-block.cu`](./solutions/transformer/gpt-2-transformer-block.cu)
