### [ppo-clipped-surrogate-loss](https://leetgpu.com/challenges/ppo-clipped-surrogate-loss)
*环境: CUDA*

对形状同为 `[B, S]` 的优势值 `advantages`、新策略对数概率 `log_pi` 与旧策略对数概率 `log_pi_old`，逐元素算出概率比 `r = exp(log_pi - log_pi_old)` 及其在 `[1 - clip_eps, 1 + clip_eps]` 区间内的截断值 `r_hat`，取 `min(r * advantages, r_hat * advantages)` 的全量均值的相反数写入 `output[0]`。

用官方例子 `B = 1, S = 4, clip_eps = 0.2` 看清正负优势下的截断规则：设 `advantages = [1.0, -2.0, 3.0, -4.0]`，对数概率差还原出的概率比依次为 `r = [1.3, 0.7, 1.1, 0.8]`，截断区间为 `[0.8, 1.2]`，于是 `r_hat = [1.2, 0.8, 1.1, 0.8]`：
1. **逐项取悲观下界**：分别算 `r * A` 与 `r_hat * A` 并取较小者——第 0 项 `min(1.3, 1.2) = 1.2`（正优势超出上限时按 `1.2` 截断）；第 1 项 `min(-1.4, -1.6) = -1.6`（负优势乘积更小的 `-1.6` 被保留）；第 2 项 `min(3.3, 3.3) = 3.3`；第 3 项 `min(-3.2, -3.2) = -3.2`。
2. **求均值取反输出**：四项代理目标相加为 `1.2 - 1.6 + 3.3 - 3.2 = -0.3`，除以 `4` 再取相反数，得到 `output[0] = -(-0.3 / 4) = 0.075`。

> baseline

基线版把 `[B, S]` 展平为一维的 `N = B * S` 个元素，用两趟核函数完成全局归约：第一个核函数按标量网格跨步读取三路输入，在双精度 `double` 下累加每个线程块的部分和（避免百万级单精度累加漂移）；第二个核函数汇总所有块的部分和，乘上 `-1.0 / N` 写回 `output[0]`：

```cuda
for (size_t i = global_tid; i < total_elems; i += stride) {
    const float a = advantages[i];
    const float r = expf(log_pi[i] - log_pi_old[i]);
    const float r_hat = fminf(fmaxf(r, lo_clip), hi_clip);
    local_sum += static_cast<double>(fminf(r * a, r_hat * a));
}
// 块内用线程束洗牌指令 __shfl_down_sync + 共享内存归约后写入 g_base_block_sums[blockIdx.x]
```

> optimized：16 元成对累加向量化流式扫描 + 单核原子尾块退役

看官方计时规模 `B = 256, S = 16,384`（共 `4,194,304` 个元素）：三路输入共 `50.33 MB`（`48 MiB`），按 Tesla T4 约 `310 GB/s` 的显存带宽计算，纯数据搬运时间只有 **`0.16 ms`** 左右。在这种百微秒级带宽受限算子里，多启动一次归约核函数的调度延迟，以及图灵架构（Turing `sm_75`，双精度单元吞吐仅为单精度的 `1/32`）逐元素执行 `double` 加法的指令阻塞，都会明显拖慢耗时。

优化版用三招把执行路径压到紧贴硬件带宽上限：
- **`4 × float4` 突发加载配无分支截断**：每轮循环连续用 `__ldg` 预取 4 组 `float4`（一次吞吐 16 个元素），直接计算 `expf(lp - lpo)` 并用硬件最小/最大指令无分支算出 `fminf(r * a, r_hat * a)`。
- **16 元单精度成对归约后再转双精度**：先把这 16 个单精度代理目标按树形两两相加 `(s0 + s1) + (s2 + s3)` 聚合成 1 个 `float`，每处理 **16 个元素才做 1 次 `double` 累加**。双精度加法指令数直接砍掉 `15/16`，同时大数组归约误差依然稳定在 `1e-7` 量级（远优于 `1e-4` 评测阈值）。
- **单内核原子尾块退役（Last-Block Retirement）**：每个线程块算出双精度部分和写入静态设备端数组 `g_block_sums[blockIdx.x]`，执行 `__threadfence()` 后对 `g_retired_blocks` 做原子加一；最后一个完成的线程块顺手归约全部部分和并写出 `output[0]`，**整道算子仅需 1 次内核启动**、零显存分配：

```cuda
// 每轮展开 4 组 float4（16 个元素），先在 FP32 内两两成对求和，每 16 个元素只做 1 次 FP64 累加
const float sum01 = ppo_vec4_sum(a0, p0, q0, lo_clip, hi_clip) +
                    ppo_vec4_sum(a1, p1, q1, lo_clip, hi_clip);
const float sum23 = ppo_vec4_sum(a2, p2, q2, lo_clip, hi_clip) +
                    ppo_vec4_sum(a3, p3, q3, lo_clip, hi_clip);
acc_d += static_cast<double>(sum01 + sum23);
// 循环结束后通过 __threadfence() + atomicAdd(&g_retired_blocks, 1u) 由尾块单核写出 output[0]
```

线程块网格上限对齐 T4 的 40 个流式多处理器（SM）整波次（`320` 块 = 2 个整波次），尾部不足 4 的余数元素及非 16 字节对齐地址自动走标量分支无缝处理。

本地 Tesla T4 每次计时前刷空 L2 缓存，在官方计时规模 `B = 256, S = 16,384`（`clip_eps = 0.2`）下取 15 轮中位数：双核标量基线耗时 `0.1944 ms`，单核向量化成对归约版降至 **`0.1764 ms`**（提速 **`1.10x`**，有效带宽达 **`285.3 GB/s`**）。

完整代码：[双核标量基线](../../solutions/reduction/ppo-clipped-surrogate-loss-baseline.cu) · [单核向量化成对归约优化](../../solutions/reduction/ppo-clipped-surrogate-loss-opt.cu)
