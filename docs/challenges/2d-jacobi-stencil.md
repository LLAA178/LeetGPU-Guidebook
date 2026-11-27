### [2d-jacobi-stencil](https://leetgpu.com/challenges/2d-jacobi-stencil)
*环境: CUDA*

给定形状为 `[rows, cols]` 的二维浮点网格 `input`，执行一步二维五点雅可比（Jacobi）迭代：最外圈边界元素原样复制到 `output`，全部内部元素更新为上下左右四个相邻元素的算术平均值 `0.25f * (top + bottom + left + right)`。

用官方 `4 × 4` 样例（`1..16` 按行顺序排列）走一遍：

- 四周共 12 个边界位置（第 `0` 行、第 `3` 行、第 `0` 列、第 `3` 列）保持原值不变。
- 中心 4 个内部位置分别取四邻域均值：`(1, 1)` 为 `0.25 * (2 + 10 + 5 + 7) = 6.0`，`(1, 2)` 为 `0.25 * (3 + 11 + 6 + 8) = 7.0`，`(2, 1)` 为 `0.25 * (6 + 14 + 9 + 11) = 10.0`，`(2, 2)` 为 `0.25 * (7 + 15 + 10 + 12) = 11.0`。

> baseline

每个线程对应二维网格中的一个坐标 `(i, j)`：落在外圈边界时直接拷贝 `input[i, j]`，落在内部时从全局显存读取上下左右四个邻居求均值写入 `output[i, j]`。

```cuda
size_t idx = static_cast<size_t>(i) * cols + j;
if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
  output[idx] = input[idx];
  return;
}
output[idx] = 0.25f * (input[idx - cols] + input[idx + cols] + input[idx - 1] + input[idx + 1]);
```

> optimized：双行 `float4` 共享边界交换 + 流式免回读写出

在官方性能测试规模 `8,192 × 8,192` 下，输入和输出各占 `268.44 MB`（理论最小单读单写流量为 `536.87 MB`），每算一个点只有 3 次加法和 1 次乘法，完全受制于全局显存带宽与缓存行利用率。优化版让每个 `256` 线程块一次负责相邻 `2` 行 × `1,024` 列（每线程 `2` 行 × `1` 个 `float4` 共 8 个输出点）：

- **双行 `float4` 合并加载与中间行寄存器直用**：计算相邻两行 `r0` 和 `r1` 只需要读取 `r0 - 1`、`r0`、`r1`、`r1 + 1` 共 4 行输入，其中 `v_0` 和 `v_1` 在寄存器里同时充当自身行的左右源数据与相邻行的上下源数据，且 `32 KB` 宽的连续行块让相邻线程块在二级缓存（L2 cache）上自然命中边界重叠行。
- **轻量共享数组交换跨线程左右边界**：一个 `float4` 内部的相邻元素本身就在同一线程寄存器内；跨 `float4` 边界的左邻 `.w` 与右邻 `.x` 通过仅占 `4 KiB` 的片上共享数组 `s_left_w`、`s_right_x` 一次同步完成交换，整块 1,024 列每行仅在块首尾补读 2 个标量。
- **流式存储省掉 268 MB 写分配回读**：普通全局写指令在未命中二级缓存时会先从显存把目标缓存行读上来（write-allocate），白白多出 `268.44 MB` 显存读流量。改用流式存储指令 `st.global.cs.v4.f32` 直接按 16 字节整行穿透写出，省掉输出缓冲区的无效回读；任意非 4 倍数尺寸或非对齐指针自动走带光环（halo）的二维共享内存分块核兜底。

```cuda
// 一次读入 v_m1, v_0, v_1, v_2 四个 float4，同时算出上下相邻两行并流式穿透写出
out0.y = 0.25f * ((v_m1.y + v_1.y) + (v_0.x + v_0.z));
out0.z = 0.25f * ((v_m1.z + v_1.z) + (v_0.y + v_0.w));
out1.y = 0.25f * ((v_0.y + v_2.y) + (v_1.x + v_1.z));
out1.z = 0.25f * ((v_0.z + v_2.z) + (v_1.y + v_1.w));
store_streaming_float4(&out4[r0 * sw4 + c4], out0);
store_streaming_float4(&out4[r1 * sw4 + c4], out1);
```

在官方性能测试规模 `8,192 × 8,192` 下（每次计时前刷空二级缓存，取 15 轮中位数），双行 `float4` 流式版与基线均跑满显存带宽上限，实测中位耗时 **`1.89 ms`**（等效单读单写有效带宽达 **`283.6 GB/s`**）。

完整代码：[`2d-jacobi-stencil-baseline.cu`](../../solutions/convolution/2d-jacobi-stencil-baseline.cu) · [`2d-jacobi-stencil-opt.cu`](../../solutions/convolution/2d-jacobi-stencil-opt.cu)
