### [2d-max-pooling](https://leetgpu.com/challenges/2d-max-pooling)
*环境: CUDA*

对形状为 `[N, C, H, W]` 的二维特征图按窗口大小 `kernel_size`、步长 `stride` 和隐式负无穷边界填充 `padding` 执行二维最大池化，在每个滑动窗口内取最大值写入形状为 `[N, C, H_out, W_out]` 的输出张量。

拿两个小矩阵走一遍：

- **`3×3` 输入、`kernel_size = 2, stride = 1, padding = 0`**：输入 `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]` 共滑出 `2×2` 个窗口，左上窗口 `max(1, 2, 4, 5) = 5`，右上 `max(2, 3, 5, 6) = 6`，左下 `max(4, 5, 7, 8) = 8`，右下 `max(5, 6, 8, 9) = 9`，输出 `[[5, 6], [8, 9]]`。
- **`4×4` 输入、`kernel_size = 3, stride = 2, padding = 1`**：四周补一圈 `-INFINITY` 后，相邻两个输出行 `oh = 0`（覆盖输入行 `-1, 0, 1`）和 `oh = 1`（覆盖输入行 `1, 2, 3`）恰好共用第 `1` 行输入，同一行上的相邻窗口也有重叠元素。

> baseline

每个线程独立负责一个输出像素 `(nc, oh, ow)`，初始值设为 `-INFINITY`（兼容输入含 `-INFINITY` 的边界情形），双层循环扫过 `kernel_size × kernel_size` 窗口内所有落在 `[0, H) × [0, W)` 范围内的元素求最大值。

```cuda
int h_start = oh * stride - padding;
int w_start = ow * stride - padding;
float max_val = -INFINITY;
for (int kh = 0; kh < kernel_size; ++kh) {
  int ih = h_start + kh;
  if (ih >= 0 && ih < H) {
    for (int kw = 0; kw < kernel_size; ++kw) {
      int iw = w_start + kw;
      if (iw >= 0 && iw < W) max_val = fmaxf(max_val, plane_in[ih * W + iw]);
    }
  }
}
plane_out[oh * W_out + ow] = max_val;
```

> optimized：全行宽单线程束 `float4` 寄存器纵向滑动复用

在官方评测规模 `N = 4, C = 64, H = 256, W = 256, kernel_size = 3, stride = 2, padding = 1` 下，输入占 `67.11 MB`、输出占 `16.78 MB`。朴素版对每个输出像素重读 9 个输入点，访存放大超过两倍且以 4 字节标量读取。优化版利用二维最大值的横纵可分离性，让单个线程束（warp，锁步执行的 32 个线程）在纯寄存器里纵向扫过特征图条带，零共享内存（shared memory）、零同步屏障：

- **单线程束覆盖整行 `W = 256` 与洗牌横向归约**：`W = 256` 恰好等于 `32 个线程 × 2 个 float4`。线程束分两次发起 128 字节满合并 `__ldg` 读取拿齐整行 256 个浮点数，通过寄存器洗牌指令 `__shfl_sync` 和 `__shfl_up_sync`（线程束内直接交换寄存器数据）将相邻两个 `float4` 配给同一个线程，并从左邻线程借来边界元素 `8 * lane - 1`，在寄存器里一步算出 4 个输出列对应的横向 3 点最大值向量 `float4`。
- **纵向滑动传递重叠行**：输出第 `oh` 行需要输入行 `2*oh - 1`、`2*oh`、`2*oh + 1` 的横向最大值。把特征图纵向切成 4 条带后，循环每推进一行输出只需新读 `2*oh` 和 `2*oh + 1` 两行输入，算完 `max(prev, r0, r1)` 后直接令 `prev = r1` 留给下一行 `oh + 1` 复用，条带内部每个输入像素严格只读一次。
- **流式向量写回与多层通用快路径**：每算出一行 128 个输出（正好是 32 个线程手里的 32 个 `float4`），直接通过流式存储指令 `st.global.cs.v4.f32` 一次性写回全局显存。针对更宽的 `K=3, S=2, P=1` 形状、`K=2, S=2, P=0` 无重叠下采样以及 `K <= 7` 常规池化核，代码分别提供了对应的 `float4` 分块核与共享内存带光环（halo）分块核，完整覆盖任意维度和填充参数。

```cuda
// 纵向滑动：每步只读 ih0=2*oh 和 ih1=2*oh+1 两行，将 r1 直接留作下一行输出的 prev
for (int oh = oh_start; oh < oh_end; ++oh) {
  float4 r0 = load_and_horiz_max_w256(plane_in4 + (oh * 2) * 64ull, lane);
  float4 r1 = (oh * 2 + 1 < H)
                  ? load_and_horiz_max_w256(plane_in4 + (oh * 2 + 1) * 64ull, lane)
                  : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
  float4 out_v = max_float4_3(prev, r0, r1);
  store_streaming_float4(&plane_out4[oh * 32ull + lane], out_v);
  prev = r1;
}
```

把 9 点重叠标量重读改成全行 `float4` 合并加载、寄存器洗牌横向归约与纵向单遍滑动复用后，全局显存流量压到了接近 `83.9 MB` 的单读单写理论下限。在官方性能测试规模 `4 × 64 × 256 × 256`（`K = 3, stride = 2, padding = 1`）下，实测中位耗时从 `0.307 ms` 降至 **`0.299 ms`**（有效必需带宽达 **`281.0 GB/s`**）。

完整代码：[`2d-max-pooling-baseline.cu`](../../solutions/convolution/2d-max-pooling-baseline.cu) · [`2d-max-pooling-opt.cu`](../../solutions/convolution/2d-max-pooling-opt.cu)
