### [causal-self-attention](https://leetgpu.com/challenges/causal-self-attention)
*环境: CUDA*

一句话：给定形状为 `M×d` 的 `Q`、`K`、`V`，为第 `i` 个查询位置只对“它自己和之前的键”（`j <= i`）做缩放点积和按行 softmax，再用得到的权重对 `V` 加权求和，相当于把自注意力矩阵的上三角全部遮掉。

举个小例子（`M=2, d=2`）：第 0 行查询只能看到第 0 个键，softmax 权重固定是 `[1.0]`，输出直接等于 `V[0]`；第 1 行查询可以看到第 0、1 两个键，先算出两个缩放点积再做 softmax，把 `V[0]` 和 `V[1]` 按权重混合。

> baseline

每个 warp（线程束，GPU 上同步执行的 32 个线程）负责一行 `Q`，顺序遍历 `0..q_row` 的每一行 `K` 和 `V`，用 Online Softmax（在线 softmax，一边扫一边维护当前最大值和指数和，省去把 `M×M` 注意力分数矩阵写回显存的开销）流式更新输出。

```CUDA
for (int k_row = 0; k_row <= q_row; ++k_row) {
    float dot = 0.0f;
    for (int i = 0; i < elems; ++i) {
        const int col = lane + i * 32;
        if (col < d) dot = fmaf(q_reg[i], k_ptr[col], dot);
    }
    float score = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0);
    float m_new = fmaxf(m, score);
    float alpha = __expf(m - m_new);
    float p = __expf(score - m_new);
    l = l * alpha + p;
    for (int i = 0; i < elems; ++i) {
        const int col = lane + i * 32;
        if (col < d) o_reg[i] = fmaf(p, v_ptr[col], o_reg[i] * alpha);
    }
    m = m_new;
}
```

> optimized

baseline 里每个 warp 每次只算一行 `Q`，`K` 和 `V` 都要反复从 global memory（全局显存）读；优化版改用 FlashAttention 分块思路：一个 256 线程的 block（线程块，组织成 `16×16`、共 8 个 warp）一次处理 `Br=64` 行 `Q`，每个线程同时维护 4 行 `Q` 的累加器，每次把 `Bc=32` 行 `K` 或 `V` 搬进 shared memory（片上共享内存，速度远快于全局显存）复用，并用 `float4`（一次读写 4 个 `float` 的 128 位向量指令）加速 `Q/K/V` 与中间权重表 `sP` 的访存。同时利用因果掩码做了三处优化：
- **外层分块提前终止**：当 `k_start > q_max_row` 时直接结束循环，整块跳过上三角的无效 `K/V` 搬运与计算。
- **内部完整分块无分支路径**：当 `k_start + Bc - 1 <= q_block_row` 时，整个分块全部位于下三角，去掉内层逐元素掩码判断。
- **逆序 block 调度**：从最大的查询行分块开始发射，让计算量最重的 block 先跑，消除尾部波次的负载不均衡。

```CUDA
const int q_block_idx = gridDim.x - 1 - blockIdx.x;  // 逆序调度，重负载 block 先跑
for (int k_start = 0; k_start <= q_max_row; k_start += kBc) {
    // 1. float4 协作加载 K 分块到 shared memory
    // 2. 内部完整分块走无分支路径，对角分块施加 j <= q_row 掩码；每份 K float4 复用 4 行 Q
    // 3. 16 线程子束内用 shuffle 归约完成 Online Softmax，写回 sP
    // 4. float4 协作加载 V 分块到 shared memory，float4 读取 sP 并把每份 V float4 复用 4 行 Q
}
```

> 省了哪一步与实测效果
- **64 行共享片上分块 + 4 倍寄存器复用**：64 行 `Q` 共享同一份片上 `K/V` 分块，每个线程同时算 4 行 `Q`，并按 4 个元素一组用 `float4` 读 `sP`，大幅降低全局显存与片上访存压力。
- **因果剪枝与负载均衡**：按块跳过上三角分块、内部块去分支、逆序发射重负载 block，减少无效计算与尾部等待。
- **本地实测**：在 `M=5000, d=64` 上，优化版从 `15.323 ms` 降到 `1.526 ms`（提速 `10.04x`）；在 `M=5000, d=128` 上，从 `22.766 ms` 降到 `2.693 ms`（提速 `8.45x`），数值误差均小于 `5e-7`。
- **LeetGPU 平台实测（T4）**：`v1`（`Br=8`）`18.72 ms`（第 73.8 百分位）→ `v2`（`Br=32`，4 倍寄存器复用 + 逆序调度）`9.05 ms`（第 89.4 百分位）→ `v3`（`Br=64`，64 KB 动态共享内存 + `float4` 读 `sP`）`7.31 ms`（第 90.7 百分位），全部用例通过。

完整代码：
- baseline：[`solutions/transformer/causal_self_attention_baseline.cu`](../../solutions/transformer/causal_self_attention_baseline.cu)
- optimized：[`solutions/transformer/causal_self_attention_opt.cu`](../../solutions/transformer/causal_self_attention_opt.cu)
