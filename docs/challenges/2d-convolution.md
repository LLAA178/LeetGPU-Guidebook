### [2d-convolution](https://leetgpu.com/challenges/2d-convolution)
*环境: CUDA*

这道题实现无填充的二维有效互相关（valid 2D cross-correlation）：将尺寸为 `kernel_rows × kernel_cols` 的卷积核在不翻转的情况下滑过 `input_rows × input_cols` 的输入矩阵，输出尺寸为 `(input_rows - kernel_rows + 1) × (input_cols - kernel_cols + 1)` 的结果矩阵。

以官方示例 1 为例，输入为 `3×3` 矩阵 `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]`，卷积核为 `2×2` 矩阵 `[[0, 1], [1, 0]]`，有效输出尺寸为 `(3 - 2 + 1) × (3 - 2 + 1) = 2×2`：左上角窗口 `[[1, 2], [4, 5]]` 与卷积核逐项相乘求和得到 `1×0 + 2×1 + 4×1 + 5×0 = 6`，右上角窗口得到 `2×0 + 3×1 + 5×1 + 6×0 = 8`，左下角与右下角窗口同理得到 `12` 和 `14`，最终输出 `[[6, 8], [12, 14]]`。

> baseline
```cuda
__global__ void conv2d_valid_baseline_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_cols,
    int kernel_rows,
    int kernel_cols,
    int out_rows,
    int out_cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= out_rows || col >= out_cols) return;

    float sum = 0.0f;
    for (int kr = 0; kr < kernel_rows; ++kr) {
        const size_t in_row_base = static_cast<size_t>(row + kr) * static_cast<size_t>(input_cols);
        const int k_row_base = kr * kernel_cols;
        float row_sum = 0.0f;
        for (int kc = 0; kc < kernel_cols; ++kc) {
            row_sum = fmaf(input[in_row_base + col + kc], kernel[k_row_base + kc], row_sum);
        }
        sum += row_sum;
    }
    output[static_cast<size_t>(row) * static_cast<size_t>(out_cols) + col] = sum;
}
```

在性能评测规模 `3072×3072`、卷积核 `15×15`（有效输出 `3058×3058`）下，全图共有约 `21.04 亿` 次乘加运算（约 `4.21 GFLOP`），属于典型的算力与片上访存密集型算子。基线版让每个线程独立负责一个输出像素，相邻像素的 `15×15` 感受野高度重叠，同一输入元素会被周边最多 `225` 个线程重复读取；即便把输入搬进共享内存（片上高速缓存）后仍保持“一线程算一像素”，每次乘加也得配一条标量共享内存读取指令，受限于片上访存指令发射吞吐。

> optimized

优化版针对常见奇数方核（`3×3` 至 `15×15`）采用编译期模板展开的二维寄存器分块核函数，并为其余任意 `1 ≤ kernel_rows, kernel_cols ≤ 31` 的非对称或偶数核提供通用共享内存分块路径：
- **按行合并搬运与零存储体冲突填充**：线程块（`16×16 = 256` 线程，负责 `64×64` 输出块）按线程束（warp，32 线程）整行协作，把带光环边界（halo）的输入分块与卷积核一次性加载进共享内存；共享内存行步长补齐为奇数个 `float4`（例如 `15×15` 核取 `21 × 4 = 84` 个 `float`），既保证每行首地址满足 16 字节对齐，又错开相邻行的存储体（bank）映射，消除跨行存储体冲突。
- **`4×4` 寄存器累加与 `LDS.128` 滑动窗口**：每个线程在寄存器中维护 `4×4 = 16` 个输出累加器。沿卷积核每一行推进时，线程用 16 字节向量类型 `float4`（对应底层 `LDS.128` 片上宽读取指令）按 4 列步长读取共享内存，并用双 `float4` 寄存器滑动窗口在 `4×4` 输出块之间复用同一批输入与权重，把片上读取指令数压缩至标量方案的约十二分之一，让执行流水线集中在单精度融合乘加指令（`fmaf`）上。
- **共享内存中转合并写回**：像 `3058 = 3072 - 15 + 1` 这样的输出列数无法被 4 整除，直接对全局显存做 `float4` 向量写出会引发非对齐访问异常。核函数在收尾阶段复用共享内存缓冲，先将各线程的 `4×4` 结果写回片上，再由连续 64 个线程按步长 1 逐行对齐写回全局显存，兼顾任意输出宽度的内存合并效率与边界安全。

在 `3072×3072`（`15×15` 卷积核）基准评测中，基线版耗时 `7.87 ms`，优化版降至 **`0.634 ms`**（提速 **`12.41x`**，实测吞吐达 **`6.63 TFLOPS`**）。

完整代码：[标量基线版](../../solutions/convolution/2d-convolution-baseline.cu) · [寄存器分块优化版](../../solutions/convolution/2d-convolution-opt.cu)
