### [grouped-query-attention](https://leetgpu.com/challenges/grouped-query-attention)
*环境: CUDA，Tensor Core 路径需要 SM 8.0 或更高架构*

这题要算分组查询注意力（GQA）：把 `num_q_heads` 个查询头（Q）均分成 `num_kv_heads` 组，每组共享同一份键（K）和值（V），按 `softmax(Q × K^T / sqrt(head_dim)) × V` 算出输出，所有张量都是单精度浮点（FP32）。

举个小例子：假设有 4 个 Q 头、2 个 KV 头，那么 `Q[0], Q[1]` 都去查 `K[0], V[0]`，`Q[2], Q[3]` 都去查 `K[1], V[1]`。如果先把完整的 `seq_len × seq_len` 注意力分数表写回全局显存再做 softmax，读写量会非常大。

> baseline

每个线程束（warp，32 个线程）负责一个 Q 头里的一行输出，把这一行 `Q` 和输出累加器 `O` 留在寄存器里。遍历 `K/V` 的每一行时，用 Online Softmax（在线归一化）一边扫一边更新当前最大值 `m` 和指数和 `l`，最后统一除以 `l`，全程不需要把分数表写进显存。

```CUDA
float m_new = fmaxf(m, score);
float alpha = expf(m - m_new);
float p = expf(score - m_new);
l = l * alpha + p;
for (int i = 0; i < elems_per_lane; ++i) {
    o_reg[i] = fmaf(p, v_ptr[d], o_reg[i] * alpha);
}
m = m_new;
```

> optimized

基线版每次只算一行 Q，每行都要重新从显存读一遍完整的 `K/V`。优化版针对 `head_dim = 128` 做了三件事把速度提上去：
- **分块复用 `K/V`**：一个线程块（4 个线程束）一次处理 64 行 `Q`，沿序列长度每次把 64 行 `K/V` 用 `cp.async`（异步拷贝指令）搬进共享内存（片上高速缓存），64 行 `Q` 共享同一块 `K/V`，显存读取量大幅下降。
- **3-MMA TF32 拆分**：A100 的矩阵计算单元（Tensor Core）原生支持 TF32（10 位尾数），但单次 TF32 截断在 `head_dim = 128` 下会产生约 `3.5e-4` 误差。把每个 FP32 数拆成高位和低位两段 TF32（`x = x_hi + x_lo`），用 `hi·hi + hi·lo + lo·hi` 三条矩阵乘加指令累加到 FP32，三次乘加降低了舍入误差，结果通过平台精度检查，同时保留了 Tensor Core 的高吞吐。
- **纯寄存器做 Softmax**：矩阵乘加的输出布局里，同一行的元素恰好分布在 4 个同组线程上。行最大值和行和只需两次线程束内洗牌（`__shfl_xor_sync`）就能算完，完全不经过共享内存。SM 8.0 以下或其他维度自动回退到共享内存分块的 FP32 路径。

优化版在 LeetGPU 官方 A100-80GB 上通过全部测试，耗时 **1.23 ms**（第 **94.7** 百分位）。

完整代码：[Online Softmax 基线版](../../solutions/transformer/grouped_query_attention_baseline.cu) · [FlashAttention-2 + 3-MMA TF32 优化版](../../solutions/transformer/grouped_query_attention_flash.cu)
