### [attention-with-linear-biases](https://leetgpu.com/challenges/attention-with-linear-biases)
*环境: CUDA*

一句话：本题计算带线性位置偏置（ALiBi）的注意力，对查询矩阵 `Q`（`M×d`）、键矩阵 `K`（`N×d`）和值矩阵 `V`（`N×d`）求 `softmax(Q @ K^T / sqrt(d) + alpha * (i - j)) @ V`，其中 `i`、`j` 分别是查询行号和键行号。

举个小例子：当 `M=1, N=2, d=2, alpha=0.8`，`Q=[[1, 2]]`，`K=[[1, 0], [0, 1]]`，`V=[[3, 4], [5, 6]]` 时，第 0 个查询和第 0、1 个键的点积除以 `sqrt(2)` 分别是 `0.707` 和 `1.414`，加上线性偏置 `0.8*(0-0)=0` 和 `0.8*(0-1)=-0.8` 后得到 `0.707` 和 `0.614`，做完行内 softmax 得到权重 `[0.523, 0.477]`，再对 `V` 加权求和就是 `[3.95, 4.95]`。

> baseline
每个 warp（32 个并行执行的线程）负责一行 `Q`，依次扫描每一行 `K` 和 `V`，并用 Online Softmax（在线归约最大值和指数和，省去把整张 `M×N` 分数矩阵写回显存）直接累加输出：

```CUDA
float score = __shfl_sync(0xffffffff, warp_reduce_sum(dot), 0);
score += alibi_alpha * static_cast<float>(q_row - k_row);

float m_new = fmaxf(m, score);
float rescale = __expf(m - m_new);
float p = __expf(score - m_new);
l = l * rescale + p;
```

> 优化点
- **大维度（`d > 256`）走三阶段 `128×128` 寄存器分块 SGEMM 流水线**：当 `d` 较大（例如 `d=1024`）时，把整行 `Q`/`K`/`V` 塞进片上共享内存会受限于容量。优化版在同一张 GPU 上复用一块静态临时分数缓冲区（`2048×2048` 的 FP32 缓冲区约占 `16 MiB`），依次执行：`128×128 / 8×8` 双缓冲矩阵乘 `Q @ K^T`（在写回阶段顺手加上缩放与 `alpha * (i - j)` 偏置）、行内 `float4` 向量化 Softmax、`128×128 / 8×8` 双缓冲矩阵乘 `P @ V`。
- **中小维度（`d <= 256`）走 `8×16` FlashAttention**：按 `Br=8, Bc=32` 分块，128 线程块把 8 行 `Q` 和 32 行 `K`/`V` 搬进 shared memory（片上共享内存）复用，配合 `float4`（128 位向量读写）和寄存器累加；其余不规则尺寸回退到通用 warp 版本。
- **实测成绩**：本地 T4 对比（`M=2048, N=2048`）中，`d=1024` 耗时相对 baseline 从 `185.617 ms` 降到 `3.913 ms`（本地 baseline 对比约 `47.4×` 加速），`d=64` 从 `5.031 ms` 降到 `1.268 ms`（约 `4.0×` 加速），与 FP64 CPU 参考结果最大误差低于 `4.6e-7`；LeetGPU 官方 T4 平台实测 **`6.36 ms`（第 99.2 百分位）**。

> optimized
```CUDA
if (k0 < kv_len) {
    s0 = fmaf(alibi_alpha, static_cast<float>(q_global_row - (k_start + k0)), s0);
}
if (k1 < kv_len) {
    s1 = fmaf(alibi_alpha, static_cast<float>(q_global_row - (k_start + k1)), s1);
}
float tile_max = subwarp16_max(rmax);
float m_new = fmaxf(m, tile_max);
float rescale = __expf(m - m_new);
```

完整代码：
- Baseline：[`solutions/transformer/attention_with_linear_biases_baseline.cu`](../../solutions/transformer/attention_with_linear_biases_baseline.cu)
- Optimized：[`solutions/transformer/attention_with_linear_biases_opt.cu`](../../solutions/transformer/attention_with_linear_biases_opt.cu)
