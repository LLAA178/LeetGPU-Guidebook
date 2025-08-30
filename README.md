# 算力修仙笔记：LeetGPU 三十题
网址：https://leetgpu.com/challenges

力扣GPU版，本文旨在提供详尽题解与思路。

## 简单题
### [vector-addition](https://leetgpu.com/challenges/vector-addition)
*环境: B200 + CUDA*

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
0.09953 ms 11.4th percentile

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
    int blocksPerGrid = (N / 4 + threadsPerBlock - 1) / threadsPerBlock;
    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    vector_add4<<<blocksPerGrid, threadsPerBlock>>>(A4, B4, C, N);
    cudaDeviceSynchronize();
}
```
0.06307 ms 61.4th percentile

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
0.05337 ms 80.0th percentile

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
71.06379 ms 27.0th percentile

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
0.27386 ms 18.9th percentile

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
0.18147 ms 35.9th percentile

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
0.178 ms 37.7th percentile

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
0.06781 ms 98.1th percentile

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
0.06884 ms 21.2th percentile

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
0.06728 ms 40.2th percentile

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
0.03529 ms 78.4th percentile

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
0.03195 ms 99.0th percentile

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
0.82399 ms 62.5th percentile

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
0.73152 ms 80.0th percentile

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
0.68818 ms 88.4th percentile

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
0.27392 ms 94.4th percentile

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
0.1115 ms 23.1th percentile

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
0.08007 ms 69.2th percentile

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
0.09597 ms 15.4th percentile

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
0.18015 ms 6.7th percentile

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
0.05202 ms 12.5th percentile

### [matrix-copy](https://leetgpu.com/challenges/matrix-copy)
```cuda
extern "C" void solve(const float* A, float* B, const int N) {
    cudaMemcpy(B, A, N*N*sizeof(float), cudaMemcpyDeviceToDevice);
}
```
0.0381 ms 52.0th percentile

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
2.05445 ms 0.0th percentile

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
34.01452 ms 0.0th percentile

## 中等题
#### [reduction](https://leetgpu.com/challenges/reduction)
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
0.27248 ms 24.6th percentile
分层归约
