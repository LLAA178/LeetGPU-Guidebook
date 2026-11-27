### [decaying-causal-attention](https://leetgpu.com/challenges/decaying-causal-attention)
*环境: CUDA*

一句话：给定形状为 `seq_len × d_model` 的 `Q`、`K`、`V` 以及衰减系数 `gamma ∈ (0, 1]`，为每个查询位置 `n` 只对当前及之前的位置 `m <= n` 计算无 softmax（归一化指数函数）的缩放点积注意力，并按距离乘上指数衰减权重 `gamma^(n - m)`，再对 `V[m]` 加权求和；这对应 RetNet 中的 Retention（保留机制）并行形式。

举个小例子（`seq_len=2, d_model=4, gamma=0.5`）：若 `Q` 两行都是 `[1, 1, 0, 0]`，`K` 两行分别是 `[1, 0, 0, 0]` 和 `[0, 1, 0, 0]`，`V` 两行都是 `[4, 8, 12, 16]`，则缩放点积 `Q K^T / sqrt(4)` 全为 `0.5`。叠加因果衰减掩码 `[[1, 0], [0.5, 1]]` 后，权重矩阵为 `[[0.5, 0], [0.25, 0.5]]`，最终输出第 0 行为 `0.5 × V[0] = [2, 4, 6, 8]`，第 1 行为 `0.25 × V[0] + 0.5 × V[1] = [3, 6, 9, 12]`。

> baseline

每个 warp（线程束，GPU 上同步执行的 32 个线程）负责一行查询 `n`，将 `Q[n] / sqrt(d_model)` 缓存在寄存器中，随后从 `m = n` 倒序遍历到 `0`，每步用 `decay *= gamma` 递推 `gamma^(n - m)`，经 warp shuffle（线程束内寄存器洗牌归约）算出点积后累加到输出寄存器。

```CUDA
float decay = 1.0f;
for (int m = n; m >= 0; --m) {
    float dot = 0.0f;
    for (int i = 0; i < elems; ++i) {
        const int col = lane + i * kWarpSize;
        if (col < d_model) dot = fmaf(q_reg[i], k_ptr[col], dot);
    }
    float score = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0);
    float weight = score * decay;
    for (int i = 0; i < elems; ++i) {
        const int col = lane + i * kWarpSize;
        if (col < d_model) o_reg[i] = fmaf(weight, v_ptr[col], o_reg[i]);
    }
    decay *= gamma;
}
```

> optimized

由于公式中没有 softmax，注意力对 `Q`、`K`、`V` 满足线性结合律，可以把原本 $O(S^2 D)$ 的全局两两计算改写为 **Chunkwise Parallel Retention（分块并行保留机制）**，将时间复杂度降到 $O(S D^2)$：把长度为 `seq_len` 的序列按 `C = 64` 切块，在 `d_model = 64` 且 `seq_len >= 128` 的主路径上分三阶段执行（小序列或其余维度自动回退到通用 warp 内核）：
- **阶段 1（块内 KV 状态归约）**：每个 256 线程的 block（线程块）负责一个 64 行分块，用 `float4`（128 位向量访存指令）将 `K`、`V` 分批搬入带 4 列 padding（填充，用于避免共享内存 bank conflict 存储体冲突）的 shared memory（片上共享内存），每线程以 `4 × 4` 寄存器分块累加出该块的 `64 × 64` 衰减状态矩阵 $R_c = \sum_{i=0}^{63} \gamma^{63-i} K_{c,i}^T V_{c,i}$。
- **阶段 2（块间衰减前缀扫描）**：启动 4096 个线程并行处理状态矩阵的每个元素，按 $H_0 = 0,\; H_c = \gamma^{64} H_{c-1} + R_{c-1}$ 原地完成排他性衰减前缀和，得到每个分块之前的历史汇总状态 $H_c$。
- **阶段 3（块间 + 块内融合输出）**：每个 block 先计算块间贡献 $\gamma^{r+1} (Q_c / \sqrt{d}) H_c$，再利用同一份片上 `sQ` 计算块内因果下三角分数 `S = (Q_c K_c^T / sqrt(d)) * D_causal`，跳过严格上三角线程块，最后累加 `S @ V_c` 并以 `float4` 写回输出。

```CUDA
// 阶段 1：计算各分块的衰减 KV 状态 R_c (64x64)
chunk_kv_state_kernel<<<num_chunks - 1, 256>>>(K, V, d_state, gamma);
// 阶段 2：4096 线程并行做块间衰减前缀扫描 H_c = gamma^64 * H_{c-1} + R_{c-1}
chunk_state_scan_kernel<<<16, 256>>>(d_state, num_chunks, gamma_chunk);
// 阶段 3：融合块间输出 gamma^(r+1) * Q_c H_c 与块内因果注意力输出
chunk_output_kernel<<<num_chunks, 256>>>(Q, K, V, d_state, output, seq_len, gamma, inv_sqrt_d);
```

> 省了哪一步与实测效果
- **算法复杂度降维（$O(S^2 D) \to O(S D^2)$）**：在基准形状 `seq_len = 4096, d_model = 64` 下，直接计算需要约 2.15 GFLOPs，分块并行保留机制只需约 0.10 GFLOPs，计算量与历史 KV 重复读取同时减少一个数量级以上。
- **片上共享内存复用 + 零寄存器溢出**：三阶段内核均使用 `4 × 4` 寄存器分块、`float4` 向量化读写与静态缓存的状态缓冲区，编译结果显示主路径三个内核的栈帧与寄存器溢出（spill）均为 0 字节。
- **本地实测（T4 GPU0）**：在官方基准形状 `seq_len = 4096, d_model = 64, gamma = 0.9` 上，baseline 耗时 `9.9278 ms`（相对 FP64 参考最大绝对误差 `9.537e-06`），优化版降至 `0.0634 ms`（提速 `156.6x`，最大绝对误差 `3.815e-06`），14 组功能与边界用例全部通过。
- **LeetGPU 平台实测（T4）**：优化版实测 `0.27 ms`（第 96.5 百分位），全部用例通过。

完整代码：
- baseline：[`solutions/transformer/decaying_causal_attention_baseline.cu`](../../solutions/transformer/decaying_causal_attention_baseline.cu)
- optimized：[`solutions/transformer/decaying_causal_attention_opt.cu`](../../solutions/transformer/decaying_causal_attention_opt.cu)
