# 算力修仙笔记：LeetGPU 三十题
网址：https://leetgpu.com/challenges
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
#### 流水线
```CUDA
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// 1-stage 软件流水：每线程按 grid-stride 处理多段 float4
__global__ void vecadd_v4_pipeline_cg(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int N, int N_vec4 /* (N+3)/4 */)
{
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    const int last_vec = N_vec4 - 1;

    auto ld4 = [](const float* p, float &x, float &y, float &z, float &w) {
#if __CUDA_ARCH__ >= 800
        asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];\n\t"
            : "=f"(x), "=f"(y), "=f"(z), "=f"(w) : "l"(p));
#else
        x=p[0]; y=p[1]; z=p[2]; w=p[3];
#endif
    };
    auto st4 = [](float* p, float x, float y, float z, float w) {
#if __CUDA_ARCH__ >= 800
        asm volatile("st.global.wb.v4.f32 [%4], {%0,%1,%2,%3};\n\t"
            : : "f"(x), "f"(y), "f"(z), "f"(w), "l"(p));
#else
        p[0]=x; p[1]=y; p[2]=z; p[3]=w;
#endif
    };

    // 预取第一块
    int i4 = tid;
    if (i4 >= N_vec4) return;

    float ax,ay,az,aw, bx,by,bz,bw;
    float nx,ny,nz,nw, mx,my,mz,mw; // next buffers

    const float* Ap = A + ((size_t)i4 << 2);
    const float* Bp = B + ((size_t)i4 << 2);
    ld4(Ap, ax,ay,az,aw);
    ld4(Bp, bx,by,bz,bw);

    for (;;)
    {
        // 预取下一块到 next 寄存器
        int i4n = i4 + stride;
        bool has_next = (i4n < N_vec4);
        if (has_next) {
            const float* An = A + ((size_t)i4n << 2);
            const float* Bn = B + ((size_t)i4n << 2);
            ld4(An, nx,ny,nz,nw);
            ld4(Bn, mx,my,mz,mw);
        }

        // 计算当前
        float sx = ax + bx;
        float sy = ay + by;
        float sz = az + bz;
        float sw = aw + bw;

        const int base = i4 << 2;
        if (i4 != last_vec) {
            // 热路径：向量化写
            st4(C + base, sx,sy,sz,sw);
        } else {
            // 仅尾块掩码写
            if (base + 0 < N) C[base + 0] = sx;
            if (base + 1 < N) C[base + 1] = sy;
            if (base + 2 < N) C[base + 2] = sz;
            if (base + 3 < N) C[base + 3] = sw;
        }

        if (!has_next) break;

        // 滚动寄存器，进入下一轮
        i4 = i4n;
        ax=nx; ay=ny; az=nz; aw=nw;
        bx=mx; by=my; bz=mz; bw=mw;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N)
{
    if (N <= 0) return;
    const int N_vec4 = (N + 3) >> 2;

    // 适度超额并行
    int blocks = (N_vec4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks = min(blocks * 4, 8192);

    vecadd_v4_pipeline_cg<<<blocks, BLOCK_SIZE>>>(A, B, C, N, N_vec4);
    // cudaDeviceSynchronize();
}
```
0.05376 ms 78.0th percentile
