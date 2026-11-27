### [3d-convolution](https://leetgpu.com/challenges/3d-convolution)
*环境: CUDA*

给定形状为 `[input_depth, input_rows, input_cols]` 的三维输入体 `input` 和形状为 `[kernel_depth, kernel_rows, kernel_cols]` 的三维卷积核 `kernel`，执行步长为 `1`、无填充（valid）的三维互相关运算，将形状为 `[input_depth - kernel_depth + 1, input_rows - kernel_rows + 1, input_cols - kernel_cols + 1]` 的结果写入 `output`。

用官方样例 1（输入体 `3 × 3 × 3` 按 `1..27` 顺序编号，卷积核大小 `2 × 3 × 3`）走一遍：

- 输出形状为 `(3 - 2 + 1) × (3 - 3 + 1) × (3 - 3 + 1) = 2 × 1 × 1`。
- **算 `output[0, 0, 0]`**：将卷积核两层与输入第 `0`、`1` 层对应元素相乘相加，第 `0` 层选出 `1 + 4 + 5 + 6 = 16`，第 `1` 层选出 `10 + 11 + 13 + 14 + 18 = 66`，合计 `16 + 66 = 82`。
- **算 `output[1, 0, 0]`**：卷积核沿深度方向下移一层，覆盖输入第 `1`、`2` 层，第 `1` 层选出 `10 + 13 + 14 + 15 = 52`，第 `2` 层选出 `19 + 20 + 22 + 23 + 27 = 111`，合计 `52 + 111 = 163`，输出 `[82, 163]`。注意第 `1` 层输入同时参与了上下两个输出深度的计算。

> baseline

每个线程独立负责一个输出体素 `(od, orow, oc)`，三层嵌套循环遍历 `kernel_depth × kernel_rows × kernel_cols` 个权重与对应输入位置做乘加并写回。

```cuda
float sum = 0.0f;
for (int kd = 0; kd < kernel_depth; ++kd) {
  for (int kr = 0; kr < kernel_rows; ++kr) {
    for (int kc = 0; kc < kernel_cols; ++kc) {
      float iv = input[((od + kd) * input_rows + (orow + kr)) * input_cols + (oc + kc)];
      float kv = kernel[(kd * kernel_rows + kr) * kernel_cols + kc];
      sum = fmaf(iv, kv, sum);
    }
  }
}
output[(od * out_r + orow) * out_c + oc] = sum;
```

> optimized：深度寄存器分片（Z-Slab）+ 单平面共享内存流式复用

在官方评测规模 `256 × 128 × 128` 卷 `5 × 5 × 5` 下，输出共 `252 × 124 × 124 ≈ 387.5 万`个体素，每个体素要做 `125` 次乘加（总计近 `9.7 亿`次浮点运算），而输入和输出体积加起来只有 `32.3 MB`。如果只按二维平面做分块，每算 1 层输出就要从显存读 5 层输入，深度方向放大了 5 倍带宽；如果把多层三维块全塞进片上内存，又会严重挤压线程块并发度。优化版采用**单平面共享内存 + 深度维寄存器分片**同时吃透三维复用：

- **深度维 `BZ = 4` 寄存器累加**：每个线程块（128 个线程）在平面上覆盖 `16 × 64` 的输出块（每线程负责 `2 × 4` 个平面位置），同时在寄存器数组 `acc[4][2][4]` 中一次推进连续 `4` 层输出深度。算完 `4` 层输出只需要流式扫过 `4 + 5 - 1 = 8` 层输入平面，把深度方向的全局读取放大率从 `5.0` 倍压低到 `2.0` 倍，同时保持紧凑寄存器占用以提升并发线程块数。
- **单层 5.625 KiB 共享内存流式滑窗与 `LDS.128` 多路复用**：片上共享内存（shared memory，线程块内高速片上内存）每次只存放**当前一层** `20 × 72` 输入切片（`5,760` 字节，即 `5.625 KiB`，加上卷积核缓冲整块共享内存约 `6.1 KiB`）。当第 `iz` 层输入以 `float4` 向量化搬入共享内存后，每个线程逐行用两条 128 位片上加载指令 `LDS.128` 取出连续 8 个输入元素 `p[0..7]`，一口气复用于本线程的上下两行 `r = 0, 1` 以及所有满足 `bz + kd == iz` 的深度层组合，单次片上读取即可喂饱上百条融合乘加 `fmaf` 指令。
- **编译期核尺寸特化与流式写回**：对高频的 `5 × 5 × 5` 和 `3 × 3 × 3` 卷积核做模板常量展开，消除全部内层循环分支，输出对齐时通过流式存储指令 `st.global.cs.v4.f32` 按 `float4` 写回显存；任意 `1..5` 非对称核尺寸自动走通用流式分片核兜底。

```cuda
// 单层输入平面驻留共享内存期间，每行仅用 2 条 float4 片上加载，同时复用于 2 个输出行和全部有效深度层 bz
const float4* smem_row4 = reinterpret_cast<const float4*>(&s_plane[(ty << 1) + lr][tx << 2]);
float4 p0 = smem_row4[0], p1 = smem_row4[1];
float p[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};

#pragma unroll
for (int bz = 0; bz < BZ; ++bz) {
  int kd = iz_rel - bz;
  if (kd >= 0 && kd < KD && lr < KH) {
    const float* kw0 = s_k + (kd * KH + lr) * KW;
    #pragma unroll
    for (int kc = 0; kc < KW; ++kc) {
      acc[bz][0][0] = fmaf(p[kc + 0], kw0[kc], acc[bz][0][0]);
      acc[bz][0][1] = fmaf(p[kc + 1], kw0[kc], acc[bz][0][1]);
      acc[bz][0][2] = fmaf(p[kc + 2], kw0[kc], acc[bz][0][2]);
      acc[bz][0][3] = fmaf(p[kc + 3], kw0[kc], acc[bz][0][3]);
    }
  }
}
```

二维平面 `float4` 片上缓存消除了每个体素 25 倍的平面重复访存，深度维寄存器分片又大幅砍掉了跨层重复读取，让算术单元持续满载运行。在官方性能测试规模 `256 × 128 × 128` 卷 `5 × 5 × 5` 下，本地 Tesla T4 实测中位耗时从 `1.81 ms` 降至 **`0.244 ms`**（提速 **`7.43x`**）；在 LeetGPU 官方 T4 评测中一次通过全部测试，耗时 **0.66 ms**（第 **93.9** 百分位）。

完整代码：[`3d-convolution-baseline.cu`](../../solutions/convolution/3d-convolution-baseline.cu) · [`3d-convolution-opt.cu`](../../solutions/convolution/3d-convolution-opt.cu)
