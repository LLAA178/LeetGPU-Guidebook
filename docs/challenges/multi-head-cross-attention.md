### [multi-head-cross-attention](https://leetgpu.com/challenges/multi-head-cross-attention)

一句话说清这题要算什么：给解码器（Decoder）的 `M` 个查询向量 `Q(M, H, D)` 和编码器（Encoder）的 `N` 个键值向量 `K(N, H, D)`、`V(N, H, D)`，对每个注意力头 `h` 独立计算无因果掩码的缩放点积注意力 $\text{softmax}(Q_h K_h^\top / \sqrt{D}) V_h$，输出写回 `output(M, H, D)`。和自注意力不同，交叉注意力的查询长度 `M` 与键值长度 `N` 可以不等，且每个查询都能看到全部 `N` 个编码器位置。

用官方小例子感受一下（`M=2, N=3, H=2, D=2`）：
- 头 0 的第 0 个查询 $Q_0[0]=[1,0]$ 与三个键 $K_0=\{[1,0],[0,1],[1,1]\}$ 做点积再除以 $\sqrt{2}$，得到分数 $[0.7071, 0, 0.7071]$；
- 经 Softmax 归一化得到权重 $[0.4011, 0.1978, 0.4011]$，再对 $V_0=\{[1,2],[3,4],[5,6]\}$ 加权求和，结果恰好是 $[3.0000, 4.0000]$。

**省了哪一步？**

1. **基线版（Warp 级 Online Softmax）**：用 1 个 Warp（线程束，32 个线程）负责 1 行查询 `(m, h)`，通过 Online Softmax（在线归一化）边扫 `K/V` 边维护局部最大值 `m_val` 和分母 `l_val`，彻底省去把 `M × N` 注意力分数矩阵写回显存的开销。但每行查询都会把整份 `K/V` 从全局显存重复读一遍，访存带宽成为瓶颈。
2. **优化版（分块 FlashAttention + 4 行查询复用）**：针对官方基准规模 `M=1024, N=2048, H=16, D=128`，按 `Br=32` 查询行、`Bc=32` 键值行分块：
   - 256 个线程协作以 `float4`（128 位向量化访存）把 `K/V` 分块搬入 Shared Memory（共享内存，共 32 KB），整块 32 行查询共享同一份片上 `K/V`，全局显存读取量大幅下降；
   - 每个 Warp 同时在寄存器里常驻 4 行 `Q` 与 4 行累加输出 `O`（每线程刚好持有 `D=128` 中的 1 个 `float4`），每次从共享内存读入 1 个 `K/V` 向量就同时喂给 4 行查询，并按每 4 个键合并一次 `O` 的重缩放，减少 `expf` 与乘法次数，编译做到 0 寄存器溢出（0 spills）；
   - 当 `D != 128` 或指针未 16 字节对齐时，自动回退到通用 Warp 级 Online Softmax 内核，保证所有约束范围（`1 <= M, N <= 4096, 1 <= H <= 64, 1 <= D <= 256`）全部正确通过。

**实测结果（官方基准 `M=1024, N=2048, H=16, D=128`）**：
- 基线版（`multi_head_cross_attention_baseline.cu`）：本地中位耗时 **51.117 ms**（0.34 TFLOPS）
- 优化版（`multi_head_cross_attention_opt.cu`）：本地中位耗时 **11.394 ms**（1.51 TFLOPS），相比基线加速 **4.49×**，全部功能用例及基准抽检与 FP64 CPU 参考实现最大绝对误差 `2.16e-7`（远小于 `1e-4` 容差）；LeetGPU 平台 T4 实测 **13.50 ms，第 92.3 百分位**。

完整代码：
- 基线版：[`solutions/transformer/multi_head_cross_attention_baseline.cu`](../../solutions/transformer/multi_head_cross_attention_baseline.cu)
- 优化版：[`solutions/transformer/multi_head_cross_attention_opt.cu`](../../solutions/transformer/multi_head_cross_attention_opt.cu)
