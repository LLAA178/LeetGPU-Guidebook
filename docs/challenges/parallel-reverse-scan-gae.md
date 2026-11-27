### [parallel-reverse-scan-gae](https://leetgpu.com/challenges/parallel-reverse-scan-gae)
*环境: CUDA*

给定形状为 `[B, S]` 的奖励序列 `rewards` 和状态价值序列 `values`，对每条序列先算出单步时序差分误差 `delta[b, t] = rewards[b, t] + gamma * values[b, t + 1] - values[b, t]`（末尾 `values[b, S] = 0`），再按衰减系数 `decay = gamma * lam` 从右向左递推广义优势估计 `advantages[b, t] = delta[b, t] + decay * advantages[b, t + 1]`。

拿官方 4 步样例 `rewards = [1, 2, 3, 4]`、`values = [0.5, 1.0, 1.5, 2.0]`、`gamma = 0.9`、`lam = 0.5`（`decay = 0.45`）演示：先算出四步误差 `delta = [1.4, 2.35, 3.3, 2.0]`。串行倒推需要一步步算 `A[3] = 2.0`、`A[2] = 3.3 + 0.45 * 2.0 = 4.2`、`A[1] = 2.35 + 0.45 * 4.2 = 4.24`、`A[0] = 1.4 + 0.45 * 4.24 = 3.308`。并行化利用常数衰减线性递推的结合律：一段长度为 `k` 的区间可以先在内部独立算出局部首项汇总 `sum`（假设区间右侧进位为 0），区间整体对右侧进位的放大倍数恒为 `decay^k`；把相邻左区间 `(sum_L, decay^k_L)` 与右区间 `(sum_R, decay^k_R)` 合并时，新汇总等于 `sum_L + decay^k_L * sum_R`，新放大倍数等于 `decay^(k_L + k_R)`。按两步一组切分上面的 4 个位置，左半组 `[0, 1]` 独立算出 `g1 = 2.35`、`g0 = 1.4 + 0.45 * 2.35 = 2.4575`（跨 2 步倍数为 `0.45^2 = 0.2025`），右半组 `[2, 3]` 独立算出 `A[3] = 2.0`、`A[2] = 4.2`；随后把右半组首项 `4.2` 作为进位一次传给左半组，直接得到 `A[1] = 2.35 + 0.45 * 4.2 = 4.24`、`A[0] = 2.4575 + 0.2025 * 4.2 = 3.308`。

> baseline：单序列单线程逆向递推

基础版给每条序列 `b` 分配一个线程，从 `t = S - 1` 倒着循环到 `0`，边算 `delta` 边更新 `last_gae`：

```cuda
__global__ void gae_reverse_scan_baseline_kernel(
    const float* __restrict__ rewards,
    const float* __restrict__ values,
    float* __restrict__ advantages,
    float gamma,
    float decay,
    int B,
    int S) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const int row_offset = b * S;
    float next_v = 0.0f;
    float last_gae = 0.0f;
    for (int t = S - 1; t >= 0; --t) {
        const float v_t = values[row_offset + t];
        const float delta = fmaf(gamma, next_v, rewards[row_offset + t] - v_t);
        last_gae = fmaf(decay, last_gae, delta);
        advantages[row_offset + t] = last_gae;
        next_v = v_t;
    }
}
```

在官方评测规模 `B = 64, S = 4,096` 下，这种写法全场只启动 64 个线程（仅 2 个 warp，即 32 线程组成的硬件执行单元），每个线程串行依赖跑满 4,096 步，并且同一 warp 内相邻线程访问地址相距 `4,096 * 4 = 16 KB`，全局显存读写完全跨行失配。

> optimized：单块整行 `float4` 向量化 + 线程束逆向洗牌扫描

优化版让每条序列独占一个线程块（`gridDim.x = B`），在块内把 4,096 步长链压成单遍向量化并行扫描：

- **`float4` 合并访存与零重复加载边界传递**：512 个线程各用两次 `float4` 读取连续元素，把单行 4,096 个位置分摊到寄存器并实现满带宽 128 位合并访存。计算每个 `float4` 组第 4 个位置的 `delta` 需要下一个组的首元素 `v4.x`：warp 内部直接用寄存器洗牌指令 `__shfl_down_sync`（同一 warp 内直接读取后方线程寄存器）拿到，跨 warp 边界只用 `s_first_v[16]`（64 字节 shared memory，即块内高速共享内存）中转一次，省去所有重复全局读取。
- **寄存器局部逆扫 + 两级 `decay^k` 逆向扫描**：每个线程先在寄存器里完成本地 8 个元素的局部逆扫（跨度倍数为 `decay^8`），再在 warp 内用 5 步 `__shfl_down_sync` 完成 32 路逆向前缀合并，接着由第 0 个 warp 在 shared memory 上扫完 16 个 warp 的汇总，最后把右侧进位一次性乘上对应衰减幂次加回局部结果并以 `float4` 合并写回。对超出单块容量的长序列（如 `S = 65,536`）或非对齐输入，同一个块按从右到左的分片循环携带 `row_carry` 和边界 `v0` 扫完全行，无需额外全局工作区或多次启动 kernel。

```cuda
// 1. 寄存器内 4 元组局部逆向递推（假设右侧进位为 0）
const float g3 = d3;
const float g2 = fmaf(p.d1, g3, d2);
const float g1 = fmaf(p.d1, g2, d1);
const float g0 = fmaf(p.d1, g1, d0);

// 2. warp 内 5 步逆向洗牌合并，固定跨度为 4, 8, 16, 32, 64 步
float val = g0;
float nbr = __shfl_down_sync(0xffffffffu, val, 1);
if (lane <= 30) val = fmaf(p.d4, nbr, val);
// ... 依次合并跨度 2(d8), 4(d16), 8(d32), 16(d64) ...

// 3. 拿到右侧总进位 c_in 后，一次还原 4 个位置并向量化写回
float4 out4;
out4.w = fmaf(p.d1, c_in, g3);
out4.z = fmaf(p.d2, c_in, g2);
out4.y = fmaf(p.d3, c_in, g1);
out4.x = fmaf(p.d4, c_in, g0);
advantages4[vec_idx] = out4;
```

0.03 ms 93.3th percentile (T4)

这一步消除了单线程 4,096 步串行递推依赖和跨 16 KB 步长的分散访存，把整题的 `3 MiB` 数据读写全部变成单遍 128 位合并访存。在本地 Tesla T4（CUDA 12.4）上实测官方计时规模 `B = 64, S = 4,096`（每次刷空 L2，25 轮中位数）：单线程基线耗时 `0.6864 ms`，优化版（512 线程 × 每线程 2 个 `float4`）降至 **`0.0146 ms`**（提速 **47.0 倍**，有效带宽达 `216.1 GB/s`），两版均通过 45 组多尺寸、极端边界与非对齐地址测试（最大绝对误差 `2.46e-5`）。LeetGPU 官方 Tesla T4 首次提交即通过全部测试，耗时 **`0.03 ms`（第 `93.3` 百分位）**。

完整代码：[单线程逆向递推基线](../../solutions/reduction/gae-reverse-scan-baseline.cu) · [向量化线程束逆向扫描优化版](../../solutions/reduction/gae-reverse-scan-opt.cu)
