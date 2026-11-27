### [fused-residual-add-and-rms-norm](https://leetgpu.com/challenges/fused-residual-add-and-rms-norm)
*环境: CUDA*

对形状为 `[N, C]` 的输入矩阵 `x` 与残差矩阵 `residual` 先逐元素相加得到 `z`，再按行独立计算均方根归一化（RMSNorm），最后乘上长度为 `C` 的通道缩放权重 `weight` 写回输出矩阵 `out`。

用官方单行 4 元素的例子看清三步计算：`x = [1.0, 0.0, -1.0, 2.0]`，`residual = [0.5, 1.5, 0.5, -0.5]`，`weight = [1.0, 2.0, 1.0, 0.5]`，`eps = 1e-5`：
1. **残差相加**：对应位置相加得到 `z = [1.5, 1.5, -0.5, 1.5]`。
2. **求行内缩放系数**：平方和为 `2.25 + 2.25 + 0.25 + 2.25 = 7.0`，均方值 `7.0 / 4 = 1.75`，均方根 `sqrt(1.75 + 1e-5) ≈ 1.32288`，取倒数得到 `inv_rms ≈ 0.75593`。
3. **归一化并乘权重**：每个 `z[j]` 乘上 `inv_rms * weight[j]`，输出 `[1.13389, 2.26778, -0.37796, 0.56695]`。

> baseline

每行计算互不依赖，直接让 1 个线程块（block，256 个线程）负责 1 行。第一遍循环读入 `x` 和 `residual` 累加平方和，先用线程束洗牌指令 `__shfl_down_sync` 在 32 线程组成的线程束（warp）内求和，再借片上共享内存（shared memory）汇总 8 个 warp 的部分和，由 0 号线程算出整行的 `inv_rms`；第二遍循环重新读取 `x` 和 `residual`，乘上 `inv_rms * weight[j]` 写回：

```cuda
float sum_sq = 0.0f;
for (int j = tid; j < C; j += kThreads) {
    float z = row_x[j] + row_res[j];
    sum_sq = fmaf(z, z, sum_sq);
}
// 块内 warp + shared memory 归约得到 s_inv_rms ...
float inv_rms = s_inv_rms;
for (int j = tid; j < C; j += kThreads) {
    float z = row_x[j] + row_res[j];
    row_out[j] = z * inv_rms * weight[j];
}
```

> optimized：寄存器常驻单遍访存

看官方计时形状 `N = 4,096, C = 4,096` 的显存账：`x`、`residual`、`out` 各占 `64 MiB`。双遍版第一遍读入 `x` 和 `residual` 共 `128 MiB`，远超 T4 的 `4 MiB` 二级缓存（L2 cache）容量，旧数据在第一遍结束前已被冲刷出缓存；第二遍写回前只能从全局显存再读一遍 `x` 和 `residual`（又多出 `128 MiB`），整道算子总显存流量高达 `320 MiB`。

单行的 `C = 4,096` 个浮点数只占 `16 KiB`。让一个块内的 `256` 个线程以 `float4`（单条 128 位指令一次读写 4 个连续 `float`）向量化加载，每个线程恰好分担 `4,096 / 256 = 16` 个浮点数，即 **4 个 `float4` 寄存器**。第一遍算出 `z = x + residual` 后直接留在各自的寄存器里，块内归约拿到 `inv_rms` 后立刻从寄存器取出 `z`，配合只读数据缓存指令 `__ldg` 读取仅占 `16 KiB` 的共享权重 `weight`，以 `float4` 连续合并写回 `out`：

```cuda
const float4 x0 = x4[row_base + tid],       r0 = res4[row_base + tid];
const float4 x1 = x4[row_base + tid + 256], r1 = res4[row_base + tid + 256];
const float4 x2 = x4[row_base + tid + 512], r2 = res4[row_base + tid + 512];
const float4 x3 = x4[row_base + tid + 768], r3 = res4[row_base + tid + 768];

// z0..z3 常驻寄存器，块内归约得到 inv_rms 后直接写回，省去第二遍全局重读
const float inv_rms = block_inv_rms<256>(sum_sq, 1.0f / 4096.0f, eps);

const float4 w0 = __ldg(&weight4[tid]);
out4[row_base + tid] = make_float4(
    (z0.x * inv_rms) * w0.x, (z0.y * inv_rms) * w0.y,
    (z0.z * inv_rms) * w0.z, (z0.w * inv_rms) * w0.w);
// 同理写回 +256, +512, +768 处的 z1, z2, z3 ...
```

0.85 ms 97.1th percentile (T4)

这样每行只读一次 `x` 和 `residual`、写一次 `out`，直接省掉 `128 MiB` 的重复显存读取，总流量从 `320 MiB` 降到理论下限 `192 MiB`。每个线程仅占用 48 个寄存器且零寄存器溢出（zero spill），单个流式多处理器（SM）可同时驻留 4 个线程块（共 1,024 个活跃线程）隐藏访存延迟。完整实现按行宽 `C` 分档调度：`C <= 8,192` 走寄存器常驻单遍模板，`8,192 < C <= 12,224` 走动态共享内存单遍缓存，超宽行或非 16 字节对齐输入自动回退至双遍路径，覆盖题目 `1 <= C <= 65,536` 全范围。

本地 Tesla T4（CUDA 12.4）每次计时前刷空 L2 缓存，在官方计时规模 `N = 4,096, C = 4,096` 下取 15 轮中位数：双遍基线耗时 `1.1145 ms`，寄存器常驻单遍版降至 **`0.6943 ms`**（提速 **1.61 倍**，单遍有效带宽达 `290.0 GB/s`）。LeetGPU 官方 Tesla T4 首次提交即通过全部测试，耗时 **`0.85 ms`（第 `97.1` 百分位）**。

完整代码：[双遍基线](../../solutions/normalization/fused-residual-rmsnorm-baseline.cu) · [寄存器常驻单遍优化](../../solutions/normalization/fused-residual-rmsnorm-opt.cu)
