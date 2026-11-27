### [softmax-attention](https://leetgpu.com/challenges/softmax-attention)
*环境: CUDA*

这题要算缩放点积注意力：给查询矩阵 `Q[M, d]`、键矩阵 `K[N, d]` 和值矩阵 `V[N, d]`（单精度浮点 FP32），按 `softmax(Q × K^T / sqrt(d)) × V` 算出输出矩阵 `output[M, d]`。

举个小例子：假设 `M = 512, N = 256, d = 128`。如果先把 `512 × 256` 的注意力分数表写回全局显存再做 softmax，会多出整整一轮显存读写；直接用 Online Softmax（在线归一化）一边扫 `K/V` 一边更新最大值和指数和，就能把中间分数表省掉。

> baseline

每个线程束（warp，32 个线程）负责 1 行 `Q`，把这一行 `Q` 和输出累加器留在寄存器里。遍历每一行 `K/V` 时，线程束内用洗牌归约算出点积分数，在线更新当前最大值 `m`、指数和 `l` 和加权输出，最后统一除以 `l`。

```CUDA
float m_new = fmaxf(m, score);
float alpha = expf(m - m_new);
float p = expf(score - m_new);
l = l * alpha + p;
for (int i = 0; i < elems; ++i) {
    o_reg[i] = fmaf(p, v_ptr[col], o_reg[i] * alpha);
}
m = m_new;
```

> 4-row warp (v1)

基线版里每个线程束只算 1 行 `Q`，每扫一行 `K/V` 只用一次。第一版优化让每个线程束把 4 行 `Q` 和 4 行累加器常驻寄存器，每读入 1 行 `K/V` 同时算 4 行，减少 4 倍 `K/V` 读取，并用 `__expf`（硬件快速指数指令）替代软件 `expf`。这一版在 LeetGPU 官方 T4 上通过全部测试，耗时 **0.36 ms**（第 **82.4** 百分位）。

> optimized (8×16 float4 FlashAttention)

v1 版在 `Q × K^T` 阶段每算一个点积都要做整束 32 线程洗牌归约，指令开销偏高。优化版改为二维共享内存分块（`Br = 8, Bc = 32`）加全路径 `float4`（一次读写 4 个 FP32）向量化：
- **铺满 SM 的分块规模**：`M = 512` 按 `Br = 8` 切出 64 个线程块，正好铺满 T4 的全部 40 个流多处理器（SM）；共享内存 `sKV` 先装 `K` 再复用装 `V`，总占用约 22 KB。
- **每行 16 线程并行 + 全 `float4` 向量化**：每块 `8 × 16 = 128` 线程，每行 `Q` 分到 16 个线程。算 `Q × K^T` 时每线程只算 2 列 `K`，算 `P × V` 和写回时每线程只管 2 个 `float4`（`d = 128`），彻底去掉 `Q × K^T` 的线程束归约，寄存器压到 85 个且零溢出。
- **16 线程子束在线归一化**：softmax 只需 4 次洗牌（`__shfl_xor_sync`）即可完成整行最大值与求和归约。其余 `d % 4 != 0` 的维度自动走通用回退路径。

这两步把 LeetGPU 官方 T4 实测耗时从 **0.36 ms** 压到 **0.27 ms**（缩短 25%，第 **94.2** 百分位）。

完整代码：[Online Softmax 基线版](../../solutions/transformer/softmax_attention_baseline.cu) · [4-Row 线程束复用版](../../solutions/transformer/softmax_attention_flash.cu) · [8×16 float4 优化版](../../solutions/transformer/softmax_attention_opt.cu)
