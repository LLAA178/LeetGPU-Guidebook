### [integral-image](https://leetgpu.com/challenges/integral-image)
*环境: CUDA*

给定形状为 `[H, W]` 的二维单精度浮点图像 `input`，计算其积分图（Integral Image，也称二维求和面积表）：每个位置 `(r, c)` 输出左上角子矩形 `input[0..r, 0..c]` 内所有像素的累加和 `output[r, c]`。

用官方 `3 × 3` 样例 `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]` 走一遍：

- 先沿每一行从左往右求包含前缀和，得到中间行和矩阵 `[[1, 3, 6], [4, 9, 15], [7, 15, 24]]`；
- 再沿每一列从上往下求包含前缀和：第 0 行保持 `[1, 3, 6]`，第 1 行加上第 0 行变成 `[5, 12, 21]`，第 2 行再加上第 1 行变成 `[12, 27, 45]`。

> baseline

按二维前缀和的行列可分离性质拆成两次 kernel 启动：第一个 kernel 每行一个线程把行向包含前缀和写进 `output`，第二个 kernel 每列一个线程在 `output` 上原地往下累加：

```cuda
__global__ void row_inclusive_scan_baseline_kernel(
    const float* __restrict__ input, float* __restrict__ output, int H, int W) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= H) return;
    double acc = 0.0;
    for (int c = 0; c < W; ++c) {
        acc += static_cast<double>(input[size_t(r) * W + c]);
        output[size_t(r) * W + c] = static_cast<float>(acc);
    }
}
```

在官方评测规模 `H = 8,192, W = 8,192`（共 6,710 万像素，输入输出各占 `268.4 MB`）下，单线程扫整行和整列既缺乏线程块内并行度，又在横向扫描时发生跨行跨距的未合并显存访问；同时全图累加项多达数千万个，纯单精度逐项累加会产生明显舍入漂移。

> optimized：零同步束级 `float4` 行扫描 + 寄存器分块解耦前瞻列扫描

优化版将横向与纵向两趟扫描全部压缩为两次核函数启动（将全局批次号与列分片计数复位顺带融合进第一趟 `blockIdx.x == 0`），并在保持双精度级跨块精度的同时避开架构上低吞吐的逐像素双精度管线：

- **第一趟零同步单线程束行向 `float4` 扫描（Zero-Barrier Warp-Per-Row Scan）**：每个 `256` 线程块拆为 `8` 个独立线程束并行负责 `8` 行，全程零共享内存、零 `__syncthreads()` 同步。每个线程束每步两路展开连续读取 `64` 个 `float4`（`256` 列共 `1 KiB` 合并流量），寄存器内通过 `__shfl_up_sync` 完成单精度束内前缀和，并用 Kahan 误差补偿对 `(carry_hi, carry_err)` 传递跨步进位，最后经 `st.global.cs.v4.f32` 流式写出 `output`。
- **第二趟 `64 × 128` 寄存器分块与解耦前瞻（Decoupled Lookback）列扫描**：沿列方向按 `64` 行 × `128` 列（`32` 个 `float4`）切分任务块，每块 `256` 线程（`8` 个线程束），每个线程束直接在寄存器内驻留 `8` 行 `float4` 并完成局部纵向包含扫描。块内共享内存仅需 `8.2 KiB` 用于汇总 `8` 个线程束的块内列和及回传双浮点补偿前缀 `(pref_hi, pref_lo)`，使流式多处理器可同时驻留多个线程块以拉满访存并行度；前 `128` 个线程以 `double` 执行跨分片解耦前瞻，随后全体线程在寄存器内用单精度补偿加法合成终值并通过 `st.global.cs.v4.f32` 一次性写回。

```cuda
// 每个线程束在寄存器内完成 8 行 float4 局部扫描，解耦前瞻拿到双精度前缀后拆成 (p_hi, p_lo) 单精度合成写出
const float4 p_hi = s_warp_hi4[warp][lane];
const float4 p_lo = s_warp_lo4[warp][lane];
#pragma unroll
for (int k = 0; k < 8; ++k) {
    float4 out_v = make_float4(p_hi.x + (v[k].x + p_lo.x), p_hi.y + (v[k].y + p_lo.y),
                               p_hi.z + (v[k].z + p_lo.z), p_hi.w + (v[k].w + p_lo.w));
    store_streaming_float4(&out4[size_t(warp_r0 + k) * W4 + c4], out_v);
}
```

对于宽度未满 128、尺寸超过 8,192、非 4 倍数或地址未对齐的输入，代码自动切换至束级行扫描与 `32 × 33` 共享内存无冲突列分块通用路径。在官方性能测试规模 `H = 8,192, W = 8,192`（两趟共约 `1,073.7 MB` 显存读写流量）下（每次计时前刷空 L2 缓存，取 15 轮中位数）：标量双趟基线耗时 `11.41 ms`，零同步束级行扫描与寄存器分块前瞻优化版降至 **`3.89 ms`**（提速 **`2.94x`**，较上一版共享内存列扫描的 `7.30 ms` 再提速 **`1.88x`**，等效显存带宽达 **`276.4 GB/s`**）。

完整代码：[`integral-image-baseline.cu`](../../solutions/reduction/integral-image-baseline.cu) · [`integral-image-opt.cu`](../../solutions/reduction/integral-image-opt.cu)
