### [gaussian-blur](https://leetgpu.com/challenges/gaussian-blur)
*环境: CUDA*

给定按行连续存储的二维图像 `input`（尺寸 `input_rows x input_cols`）和二维滤波核 `kernel`（尺寸 `kernel_rows x kernel_cols`），在图像四周补零（zero-padding，上下补 `kernel_rows / 2` 行、左右补 `kernel_cols / 2` 列）后，计算每个像素邻域的二维加权互相关并写入同尺寸的 `output`。

拿官方样例 2（`3 x 3` 图像 `[[10, 20, 30], [40, 50, 60], [70, 80, 90]]`，`3 x 3` 卷积核中心权重为 `0.2`、周围 8 个位置权重均为 `0.1`）演示：
- 左上角 `(0, 0)` 上方和左侧越界位置按 `0` 处理，只有右下 4 个邻居非零：`10 * 0.2 + (20 + 40 + 50) * 0.1 = 13.0`；
- 中心点 `(1, 1)` 完整覆盖全部 9 个像素：`50 * 0.2 + (10 + 20 + 30 + 40 + 60 + 70 + 80 + 90) * 0.1 = 50.0`。

> baseline：二维网格逐像素直接卷积

基础版按 `16 x 16` 二维线程块给每个输出像素 `(row, col)` 分配一个线程，双重循环遍历 `kernel_rows x kernel_cols` 窗口，在边界内累加乘积：

```cuda
__global__ void gaussian_blur_baseline_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= input_rows || col >= input_cols) return;

    const int pad_h = kernel_rows / 2, pad_w = kernel_cols / 2;
    float sum = 0.0f;
    for (int kr = 0; kr < kernel_rows; ++kr) {
        const int r = row + kr - pad_h;
        if (r >= 0 && r < input_rows) {
            for (int kc = 0; kc < kernel_cols; ++kc) {
                const int c = col + kc - pad_w;
                if (c >= 0 && c < input_cols) {
                    sum = fmaf(input[size_t(r) * input_cols + c], kernel[kr * kernel_cols + kc], sum);
                }
            }
        }
    }
    output[size_t(row) * input_cols + col] = sum;
}
```

在官方评测规模 `512 x 512` 图像配 `7 x 7` 滤波核下，图像本身只有 `1.05 MB`（纯数据搬运仅几微秒），但每个像素如果独立去全局显存读 49 次输入和 49 次核权重，邻域重叠像素被反复加载，且内层循环里充满边界判断分支。

> optimized：单次启动共享内存光环分块 + `2 x 4` 寄存器滑窗与 `float4` 写回

因为 `512 x 512` 整题耗时在微秒级，为了避免引入主机端 `cudaMemcpyToSymbolAsync` 常量内存拷贝调度开销，优化版直接在单次 kernel 启动内把滤波核和输入光环块（halo tile）一起搬进 shared memory（块内高速共享内存）：

- **滤波核与光环块同阶段预取（整核仅一次同步）**：每个 `16 x 8`（128 线程）块负责 `16 x 64` 的输出分块。同步屏障前，线程协作把 `KH * KW` 个核权重读进 `s_k`，同时把带外圈零填充的 `(16 + KH - 1) x (64 + KW - 1)` 输入分块读进经过列宽对齐填充（消除存储体冲突 bank conflict）的 `s_in`；对完全落在图像内部的块直接走无边界分支加载路径，随后全核只需调用**一次** `__syncthreads()`。
- **`2 x 4` 寄存器分块复用与编译期展开**：对常用的奇数尺寸核（`3x3, 5x5, 7x7, ..., 15x15`）做模板特化展开。每个线程在寄存器里同时维护 2 行 4 列共 8 个累加器，每次从 `s_k` 取出一个权重就能同时喂给 8 个输出位置，且上下两行与左右相邻像素在片上高度复用；算完后通过带缓存流式提示（cache-streaming hint）的 16 字节向量存储指令 `st.global.cs.v4.f32` 将每行 4 个相邻结果打包写出。

```cuda
// 编译期全展开 KH x KW 窗口：单次取出的核权重 kv 同时复用在 2x4 共 8 个寄存器累加器上
#pragma unroll
for (int kr = 0; kr < KH; ++kr) {
    #pragma unroll
    for (int kc = 0; kc < KW; ++kc) {
        const float kv = s_k[kr * KW + kc];
        acc00 = fmaf(s_in[sr0 + kr][sc0 + kc + 0], kv, acc00);
        acc01 = fmaf(s_in[sr0 + kr][sc0 + kc + 1], kv, acc01);
        acc02 = fmaf(s_in[sr0 + kr][sc0 + kc + 2], kv, acc02);
        acc03 = fmaf(s_in[sr0 + kr][sc0 + kc + 3], kv, acc03);
        // 第二行 (acc10..acc13) 同步累加 ...
    }
}
```

这一步让块内线程共享已加载的邻域像素与滤波核权重（仅相邻线程块边界处的窄光环存在少量跨块重读），并消除了内层循环边界分支。实测 `512 x 512`（`7 x 7` 核）时，耗时从基线的 `0.0484 ms` 降至 **`0.0164 ms`**（提速 **`2.95x`**；在更大规模 `2,048 x 2,048` 下从 `0.688 ms` 降至 **`0.164 ms`**，提速 **`4.20x`**）；在 LeetGPU 官方 T4 评测中一次通过全部测试，耗时 **0.06 ms**（第 **94.4** 百分位）。

完整代码：[逐像素直接卷积基线](../../solutions/convolution/gaussian-blur-baseline.cu) · [单核共享光环加寄存器分块优化版](../../solutions/convolution/gaussian-blur-opt.cu)
