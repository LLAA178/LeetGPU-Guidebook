### [matrix-transpose](https://leetgpu.com/challenges/matrix-transpose)
*环境: CUDA*

给定行主序存储的 `rows × cols` 单精度浮点矩阵 `input`，将其转置为 `cols × rows` 的行主序结果矩阵 `output`，满足任意位置 `output[c * rows + r] = input[r * cols + c]`。

用官方示例 1 走一遍坐标映射过程：输入为 `2 × 3` 矩阵 `[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]`，第 `0` 行的三个元素 `(0, 0)=1.0, (0, 1)=2.0, (0, 2)=3.0` 依次落到输出矩阵的第 `0` 列 `(0, 0), (1, 0), (2, 0)`；第 `1` 行的 `4.0, 5.0, 6.0` 落到输出矩阵的第 `1` 列 `(0, 1), (1, 1), (2, 1)`，最终输出 `3 × 2` 矩阵 `[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]`。

> baseline

基线版采用 `16 × 16` 二维线程块直接完成全局显存读写：同一线程束沿 `c` 维连续读取 `input[r * cols + c]`，但写出 `output[c * rows + r]` 时相邻线程地址跨度高达 `rows × 4` 字节：

```cuda
__global__ __launch_bounds__(256) void matrix_transpose_baseline_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols) {
    const int c = static_cast<int>(blockIdx.x) * 16 + static_cast<int>(threadIdx.x);
    const int r = static_cast<int>(blockIdx.y) * 16 + static_cast<int>(threadIdx.y);
    if (r < rows && c < cols) {
        output[static_cast<size_t>(c) * static_cast<size_t>(rows) + r] =
            input[static_cast<size_t>(r) * static_cast<size_t>(cols) + c];
    }
}
```

> optimized：`64 × 64` 向量化寄存器微转置 + 异或无冲突片上交换与二维分块交错调度

在官方性能评测规模 `rows = 7,000, cols = 6,000` 下，单次转置需要搬运读写共 `2 × 42,000,000 × 4 ≈ 336 MB` 显存数据，属于纯粹的显存带宽受限算子。基线版写回 `output` 时相邻线程间隔 `28,000` 字节，每个 32 字节缓存扇区只写 4 字节有效数据，显存写事务放大了 8 倍，同时栅格顺序启动的线程块同时横跨上百兆虚拟地址空间，引发严重的页表缓存（TLB）颠簸与显存分区拥塞。

优化版针对满足 16 字节对齐且维度为 4 倍数的矩阵（完整覆盖 `7,000 × 6,000` 与上限 `8,192 × 8,192`）构建全向量化零冲突流水线，并为任意奇数维度或非对齐指针提供 `32 × 33` 填充共享内存兜底路径：
- **`4 × 4` 寄存器微转置与异或无冲突 `STS.128 / LDS.128`**：每个 `256` 线程块负责 `64 × 64` 数据块（仅占 `16.0 KiB` 静态共享内存）。线程 `(tx, ty)` 先连发 4 条 16 字节合并读取指令 `__ldg`（`float4`）读入 `4 × 4` 子块，直接在寄存器内完成 `4 × 4` 转置，再按异或列索引 `phys_col = tx ^ ty` 写入 `__shared__ float4 s_v4[64][16]`；无需任何冗余填充字节，就让片上宽写（`STS.128`）与同步后的片上宽读（`LDS.128`）在每组 8 线程波前上均匀覆盖全部 32 个共享内存存储体（bank），实现双阶段零存储体冲突：

```cuda
// 寄存器内完成 4x4 转置，并按异或列索引 tx ^ ty 无冲突写入与读出 16 KiB 共享内存
const int s_row = tx * 4;
const int phys_col = tx ^ ty;
s_v4[s_row + 0][phys_col] = make_float4(v0.x, v1.x, v2.x, v3.x);
s_v4[s_row + 1][phys_col] = make_float4(v0.y, v1.y, v2.y, v3.y);
s_v4[s_row + 2][phys_col] = make_float4(v0.z, v1.z, v2.z, v3.z);
s_v4[s_row + 3][phys_col] = make_float4(v0.w, v1.w, v2.w, v3.w);
```

- **边界分块零退化与流式提示宽写出**：由于 `7,000` 与 `6,000` 均为 4 的倍数，即便落在矩阵右边缘与下边缘的残余 `64 × 64` 分块内，每个线程负责的 `4 × 4` 子块也天然整体落于有效边界内或外，全图 100% 走带缓存流式提示（cache-streaming hint）的 16 字节向量存储指令 `st.global.cs.v4.f32`。
- **二维线程块分组交错调度（CTA Swizzling）**：将连续启动的线程块按 `8` 个行分块为一组交错映射 `(bx, by)`，使并发驻留的线程块集中在局部矩形窗口内读写，大幅收敛并发活跃页表跨度并打散 `8,192` 等二次幂步长下的显存控制器热点。

在官方评测规模 `rows = 7,000, cols = 6,000` 下（每次计时前刷空 L2 缓存，取 15 轮中位数）：二维直写基线耗时 `1.52 ms`，异或无冲突向量化转置优化版降至 **`1.39 ms`**（提速 **`1.10x`**，等效读写带宽 **`242.2 GB/s`**）；在 LeetGPU 官方 Tesla T4 评测中取得 **`1.85 ms`（第 `96.0` 百分位）**。

完整代码：[二维直写基线](../../solutions/elementwise/matrix-transpose-baseline.cu) · [`64×64` 异或无冲突向量化转置优化](../../solutions/elementwise/matrix-transpose-opt.cu)
