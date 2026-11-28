### [variable-length-causal-attention](https://leetgpu.com/challenges/variable-length-causal-attention)
*环境: CUDA*

一句话：把 `S` 条长度各不相同的序列首尾拼成形状为 `T×d` 的 `Q`、`K`、`V`，用 `cu_seqlens` 标记每条序列的起止偏移，每个 token 只对同一条序列内位于自己及之前的键做缩放点积和按行 softmax，再用得到的权重对 `V` 加权求和，相当于块对角线上的因果自注意力。

举个小例子（`T=3, d=4, S=2, cu_seqlens=[0, 2, 3]`）：第 0 条序列占行 `[0, 2)`，第 1 条序列占行 `[2, 3)`。第 0 行只看第 0 行，输出直接等于 `V[0]`；第 1 行看第 0、1 两行，算出缩放点积后按行 softmax 混合 `V[0]` 和 `V[1]`；第 2 行属于第 1 条序列，只能看自己，输出直接等于 `V[2]`，完全跨不到前一条序列。

> baseline

每个 warp（线程束，GPU 上同步执行的 32 个线程）负责一个打包后的查询行 `q_row`，由 lane 0 在 `cu_seqlens` 上做二分查找确定所属序列的起始行 `seq_start`，再顺序遍历 `seq_start..q_row` 的每一行 `K` 和 `V`，用 Online Softmax（在线 softmax，边扫描边维护当前最大值和指数和，省去把 `T×T` 注意力分数矩阵写回显存的开销）流式更新输出。

```CUDA
int seq_start = 0;
if (lane == 0) {
    seq_start = find_seq_start(cu_seqlens, S, q_row);
}
seq_start = __shfl_sync(0xffffffff, seq_start, 0);

for (int k_row = seq_start; k_row <= q_row; ++k_row) {
    float dot = 0.0f;
    for (int i = 0; i < elems; ++i) {
        const int col = lane + i * 32;
        if (col < d) dot = fmaf(q_reg[i], k_ptr[col], dot);
    }
    const float score = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0);
    const float m_new = fmaxf(m, score);
    const float alpha = __expf(m - m_new);
    const float p = __expf(score - m_new);
    l = l * alpha + p;
    for (int i = 0; i < elems; ++i) {
        const int col = lane + i * 32;
        if (col < d) o_reg[i] = fmaf(p, v_ptr[col], o_reg[i] * alpha);
    }
    m = m_new;
}
```

> optimized

baseline 里每个 warp 每次只算一行 `Q`，`K` 和 `V` 都要反复从 global memory（全局显存）读取。优化版按紧凑的一维网格 `ceil(T / Br) + S` 发射 block（线程块），避免了按 `(max_q_blocks, S)` 二维网格发射时产生大量空 block 的调度开销，也无需把 `cu_seqlens` 拷回主机端。每个 128 线程的 block（组织成 `8×16`、共 4 个 warp）由 0 号线程扫描 `cu_seqlens` 定位所属序列与分块偏移，只负责单条序列内的 `Br=32` 行 `Q`。片上复用与调度优化包括：
- **紧凑一维网格 + 序列内逆序调度**：在官方基准规模（`T=8192, S=32, Br=32`）下仅发射 `288` 个 block（较二维网格的 `8192` 个减少 `28` 倍）；每条序列内按 `q_block_idx = nb - 1 - (bid - acc)` 逆序映射，让计算量最重的尾部因果分块先跑。
- **shared memory 分块与双缓冲分离**：把 `sK` 与 `sV` 拆成独立片上缓存（`d=64` 时总计 `30.0 KB`，配合 64 KB 共享内存配额单 SM 可并发 2 个 block），每轮合并搬运 `Bc=32` 行 `K` 与 `V`，将每个分块的 `__syncthreads()` 从 4 次压到 2 次（子束内 `sP` 同步仅需 `__syncwarp()`）；每个线程同时持有 4 行 `Q` 的累加器，并用 `float4`（一次读写 4 个 `float` 的 128 位向量指令）加速 `Q/K/V`、中间权重表 `sP` 与输出访存。
- **因果提前终止与内部块去分支**：外层循环在 `k_start > q_max_row` 时直接结束，跳过上三角搬运；当整个分块位于下三角且满块时走无分支路径，仅对角分块做逐元素掩码判断。对于 `d > 128` 或 `d` 非 4 的倍数的情况，自动回退到通用 warp-per-row 路径。

```CUDA
if (tid == 0) {
    int acc = 0, prev = cu_seqlens[0];
    for (int s = 0; s < S; ++s) {
        const int next = cu_seqlens[s + 1];
        const int len = next - prev;
        const int nb = (len + kBr - 1) / kBr;
        if (bid < acc + nb) {
            s_meta[0] = prev;
            s_meta[1] = len;
            s_meta[2] = nb - 1 - (bid - acc);  // 序列内逆序调度
            break;
        }
        acc += nb;
        prev = next;
    }
}
__syncthreads();
for (int k_start = 0; k_start <= q_max_row; k_start += kBc) {
    // 1. float4 协作加载本序列的 K 分块到 shared memory
    // 2. 满块下三角走无分支路径，对角分块施加 k_start + k <= q_row_local 掩码
    // 3. 16 线程子束内用 shuffle 归约完成 Online Softmax，写入 sP
    // 4. float4 协作加载 V 分块并复用 4 行 Q 累加器
}
```

> tensor core

在 `d=64` 的基准形状下，FP32 CUDA Core 已接近算力上限。若直接把输入降成单份 FP16 送入 Tensor Core（张量核心，GPU 上专做小矩阵乘加的硬件单元），精度误差约 `1e-3`，无法满足本题 `atol=1e-4` 的要求。因此采用 **3-MMA FP16 拆分**方案：把每个 FP32 拆成高位与残差两份 FP16（`x = x_hi + x_lo`），用三条 `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` 指令累加 `hi·hi + hi·lo + lo·hi` 到 FP32，把误差压到 `2.2e-6` 量级。配合 `Br=32, Bc=32`（2 个 warp、64 线程）：
- **Q 常驻寄存器 + 纯寄存器 Online Softmax**：每个 warp 负责 16 行 `Q`，预缩放并拆分后的 `q_hi/q_lo` 全程留在寄存器；`QK^T` 得到的 `s_reg` 直接在 4 线程子束内做归约和 softmax，生成的 `p_reg` 恰好符合下一阶段 `P @ V` 的 MMA A-fragment 布局，彻底省去 `sP` 共享内存与额外 shuffle。
- **18.0 KB 共享内存实现高并发**：片上仅存放 `sK_hi/sK_lo` 与转置后的 `sV_hi/sV_lo`（共 `18.0 KB`），单 SM 可并发 3 个 block，核心 kernel 编译实现 0 字节寄存器溢出。

```CUDA
// 3-MMA 累加 hi*hi + hi*lo + lo*hi -> FP32
mma_m16n8k8_f32(s0, s1, s2, s3, qh0, qh1, kh);
mma_m16n8k8_f32(s0, s1, s2, s3, qh0, qh1, kl);
mma_m16n8k8_f32(s0, s1, s2, s3, ql0, ql1, kh);
```

> 省了哪一步与实测效果
- **消除 `T×T` 中间矩阵与空 block 调度**：紧凑一维网格把发射块数从 `8192` 压到 `288`，按 `cu_seqlens` 直接切到单条序列并做序列内逆序调度，让最重的因果分块先跑。
- **3-MMA Tensor Core + 纯寄存器 Softmax**：在 `d=64` 上用 3 条 FP16 MMA 保住 FP32 级别精度，`Q` 与 `P` 均不落共享内存，片上仅占 `18.0 KB`。
- **本地实测**：在官方基准规模（`T=8192, d=64, S=32`，序列长度分布于 `128..384`）下，平均耗时从 baseline 的 `1.6933 ms` 降到 FP32 优化版的 `0.1645 ms`（`10.30x`），再降到 3-MMA Tensor Core 版的 `0.1130 ms`（提速 `14.98x`），相对 FP64 CPU 参考实现的最大绝对误差为 `2.205e-06`。

完整代码：
- baseline：[`solutions/transformer/variable_length_causal_attention_baseline.cu`](../../solutions/transformer/variable_length_causal_attention_baseline.cu)
- optimized：[`solutions/transformer/variable_length_causal_attention_opt.cu`](../../solutions/transformer/variable_length_causal_attention_opt.cu)
