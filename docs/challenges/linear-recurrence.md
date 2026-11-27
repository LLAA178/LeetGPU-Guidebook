### [linear-recurrence](https://leetgpu.com/challenges/linear-recurrence)
*环境: CUDA*

给定两个形状均为 `[B, L]` 的浮点矩阵 `a` 和 `x`，按行独立计算一阶线性递推：`h[b, 0] = x[b, 0]`，当 `t >= 1` 时 `h[b, t] = a[b, t] * h[b, t - 1] + x[b, t]`（这是 Mamba、S4 等状态空间模型的核心扫描算子）。

先用 4 个元素的例子看清串行递推怎样变成并行前缀扫描：

- 取 `a = [0.5, 0.5, 0.5, 0.5]`、`x = [1, 0, 0, 0]`，逐步代入可得 `h = [1, 0.5, 0.25, 0.125]`（注意 `h[0] = x[0]`，所以 `t = 0` 处相当于把 `a[0]` 视作 `0`）。
- 把每一步写成对输入状态 `h_in` 的仿射变换 `f_t(h_in) = m_t * h_in + v_t`，也就是二元组 `[m_t, v_t]`。连续执行两步 `[a1, x1]` 与 `[a2, x2]` 时，把 `h1 = a1 * h_in + x1` 代入 `h2 = a2 * h1 + x2`，展开得到 `h2 = (a2 * a1) * h_in + (a2 * x1 + x2)`，即合并规则 `[a1, x1] o [a2, x2] = [a2 * a1, a2 * x1 + x2]`。
- 这个合并运算满足结合律。对上面的 4 个元素，先把相邻两项配对合并：`[0, 1] o [0.5, 0] = [0, 0.5]`，`[0.5, 0] o [0.5, 0] = [0.25, 0]`；再把前半段结果合到后半段：`[0, 0.5] o [0.5, 0] = [0, 0.25]`，`[0, 0.5] o [0.25, 0] = [0, 0.125]`，一步就能并行算出全部时刻的状态。

> baseline

每个线程负责一整行 `b`，从 `t = 0` 到 `L - 1` 顺序递推。写法最直接，但在官方评测规模 `B = 64, L = 16,384` 下只有 64 个活跃线程，且同一线程束（warp，GPU 上锁步执行的 32 个线程）内相邻线程访问的物理地址相隔 `16,384` 个浮点数，全局内存读写无法合并，每个线程还要串行等完 16,384 步乘加。

```cuda
__global__ void linear_recurrence_baseline_kernel(const float* __restrict__ a,
                                                  const float* __restrict__ x,
                                                  float* __restrict__ h,
                                                  int B,
                                                  int L) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) return;

  size_t row = static_cast<size_t>(b) * static_cast<size_t>(L);
  float prev = x[row];
  h[row] = prev;
  for (int t = 1; t < L; ++t) {
    prev = a[row + t] * prev + x[row + t];
    h[row + t] = prev;
  }
}
```

> optimized：`float4` 合并访存 + 仿射前缀扫描

让一个线程块（512 个线程）协作处理一行序列，每次推进一个 `2,048` 元素的分块（每线程 1 个 `float4`），把串行长链拆成寄存器与共享内存（shared memory，线程块内高速片上内存）上的两级并行扫描：

- **`float4` 128 位合并读写与寄存器内 4 步归约**：同一线程束的 32 个线程连续读取相邻的 `float4`，合并成单条 128 字节内存事务（`t = 0` 处直接将 `qa.x` 置 `0.0f`）。每个线程先在寄存器里用 3 次融合乘加 `fmaf` 把自己的 4 个元素压成单个二元组 `(m, v)`。
- **线程束洗牌与跨束扫描**：线程束内用寄存器洗牌指令 `__shfl_up_sync`（无需经过共享内存，直接读取前序线程寄存器）通过 5 步倍增算出 32 个二元组的包含前缀；随后由第 0 号线程束把上一分块传下来的标量进位 `s_carry` 并入第 0 束，对 16 个线程束总计做一次扫描，直接得出每个线程束的输入初值 `s_warp_h_in[warp]` 并更新下一分块的 `s_carry`。
- **寄存器回代与一次性写回**：每个线程拿到前一个线程的二元组前缀后，将本束初值代入算出本线程 4 个元素的输入状态 `h_prev`，在寄存器内顺序推进 4 步并按 `float4` 一次性写回全局内存。整张表 `a`、`x`、`h` 都只走一遍顺序合并访存。

```cuda
// 线程束内对仿射二元组 (m, v) 做 5 步倍增包含扫描
#pragma unroll
for (int step = 1; step < 32; step <<= 1) {
  float prev_m = __shfl_up_sync(0xffffffffu, m, step);
  float prev_v = __shfl_up_sync(0xffffffffu, v, step);
  if (lane >= step) {
    v = fmaf(m, prev_v, v);
    m *= prev_m;
  }
}

// 代入本束初值 h_in_w，寄存器内算出 4 个输出并以 float4 写回
float h_prev = (lane == 0) ? h_in_w : fmaf(prev_m, h_in_w, prev_v);
float4 qh;
qh.x = fmaf(qa.x, h_prev, qx.x);
qh.y = fmaf(qa.y, qh.x, qx.y);
qh.z = fmaf(qa.z, qh.y, qx.z);
qh.w = fmaf(qa.w, qh.z, qx.w);
row_h4[vec_idx] = qh;
```

0.08 ms 98.6th percentile (T4)

跨线程束传递标量状态省去了全局临时缓冲区和二次读写，`float4` 合并访存把每行 16,384 步跨行串行访存压成了 8 轮满带宽顺序流式吞吐。在官方评测规模 `B = 64, L = 16,384`（共 `12 MiB` 读写，每次计时前刷空 L2，21 轮中位数）下，本地 Tesla T4（CUDA 12.4）实测耗时从基线的 `2.4103 ms` 降至 **`0.0477 ms`**（提速 **50.5 倍**，有效显存带宽达 **`263.7 GB/s`**），两版均通过 47 组多尺寸、极端边界与非对齐地址测试（相对双精度参考最大绝对误差 `4.34e-6`）。LeetGPU 官方 Tesla T4 首次提交即通过全部测试，耗时 **`0.08 ms`（第 `98.6` 百分位）**。

完整代码：[`linear-recurrence-baseline.cu`](../../solutions/reduction/linear-recurrence-baseline.cu) · [`linear-recurrence-opt.cu`](../../solutions/reduction/linear-recurrence-opt.cu)
