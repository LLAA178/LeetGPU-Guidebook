### [token-embedding-layer](https://leetgpu.com/challenges/token-embedding-layer)
*环境: CUDA*

根据词元索引矩阵 `token_ids`（形状 `[B, T]`）从词元嵌入表 `token_embeddings`（`[V, D]`）查出每一行，再加上位置索引 `position_ids`（`[T]`）对应的位置嵌入 `position_embeddings`（`[P, D]`），沿嵌入维度 `D` 做不带贝塞尔校正的层归一化（Layer Normalization，除以 `D` 计算方差）并乘加缩放参数 `gamma` 和偏移参数 `beta`，写出形状为 `[B, T, D]` 的输出张量。

用官方样例（`B = 1, T = 2, V = 3, P = 2, D = 4, eps = 1e-5`，`token_ids = [[2, 0]]`，`position_ids = [0, 1]`，`gamma = [1, 1, 1, 1]`，`beta = [0, 0, 0, 0]`）走一遍：

- **`t = 0`**：取第 2 号词元向量 `[2, 0, -2, 0]` 与第 0 号位置向量 `[0, 0, 0, 0]` 相加，得到 `s = [2, 0, -2, 0]`；均值 `mean = 0`，方差 `var = (4 + 0 + 4 + 0) / 4 = 2`，除以 `sqrt(2 + 1e-5)` 后输出 `[1.4142, 0.0, -1.4142, 0.0]`。
- **`t = 1`**：取第 0 号词元向量 `[1, 2, 3, 4]` 与第 1 号位置向量 `[1, 0, -1, 0]` 相加，得到 `s = [2, 2, 2, 4]`；均值 `mean = 2.5`，差值为 `[-0.5, -0.5, -0.5, 1.5]`，方差 `var = 3 / 4 = 0.75`，除以 `sqrt(0.75 + 1e-5) ≈ 0.86603` 后输出 `[-0.5773, -0.5773, -0.5773, 1.7320]`。

> baseline

每个线程块负责一个 `(b, t)` 位置，分三趟扫过维度 `D`：第一趟求均值 `mean`，第二趟重新读取嵌入表求中心化方差 `var`，第三趟再次读取嵌入表与 `gamma`、`beta` 完成归一化写回。每次计算都会对 `92.16 MB`（`87.89 MiB`）的词元嵌入表发起三次随机行读取。

```cuda
float local_sum = 0.0f;
for (int d = tid; d < D; d += kThreads) local_sum += tok_row[d] + pos_row[d];
// 共享内存折半归约得到 mean ...

float local_sq = 0.0f;
for (int d = tid; d < D; d += kThreads) {
  float diff = (tok_row[d] + pos_row[d]) - mean;
  local_sq += diff * diff;
}
// 共享内存折半归约得到 inv_std，第三趟读表乘 gamma 加 beta 写回 out_row
```

> optimized：位置主序跨批共享 + 单线程束 `float4` 寄存器单遍归一化

在官方性能测试规模 `B = 32, T = 512, V = 30,000, P = 2,048, D = 768` 下，`token_embeddings` 占 `92.16 MB`（`87.89 MiB`，远超 L2 缓存），随机行查表延迟是主要瓶颈。优化版同时从跨批次数据复用和单线程束寄存器驻留两方面消除冗余访存：

- **按序列位置 `t` 划分线程块，跨 `B = 32` 共享位置向量与仿射参数**：同一个序列位置 `t` 在全部 `B = 32` 个样本上使用完全相同的 `position_embeddings[position_ids[t]]`、`gamma` 和 `beta`。令 `blockIdx.x = t`，每个线程块（8 个线程束、256 个线程）在启动时只用 `9.0 KiB` 静态共享内存（shared memory，线程块内高速片上内存）加载一次这三条长度为 `768` 的向量，供块内 8 个线程束循环处理全部 `32` 个批次样本，省掉 31/32 的位置表与参数全局读取，充分隐藏词元查表延迟。
- **单线程束 6 个 `float4` 寄存器驻留整行**：`D = 768` 恰好等于 `32 个线程 × 6 个 float4`（每线程 24 个浮点数）。一个线程束（warp，锁步执行的 32 个线程）用 6 条 128 位 `__ldg` 指令把整行词元向量读进寄存器并加上共享内存中的位置向量，后续完全驻留在寄存器内：先用 5 步蝴蝶洗牌指令 `__shfl_xor_sync`（线程束内直接交换寄存器数据）求得精确均值 `mean`，再直接在寄存器里算 `(s - mean)^2` 并做第二次 5 步洗牌求得 `inv_std`，既保持两遍中心化方差的数值稳定性，又把词元嵌入行的全局读取严格压到**单遍**。
- **流式提示 128 位合并写回**：输出张量 `output` 占 `50.33 MB`（`48.00 MiB`）且后续不再读取，通过带缓存流式提示（cache-streaming hint）的 16 字节向量存储指令 `st.global.cs.v4.f32` 按 `float4` 写回显存，降低输出写入对 L2 缓存的污染。其余维度或非对齐输入自动走模板化向量核或通用单块核兜底。

```cuda
// 单线程束（32 线程）用 6 个 float4 寄存器装下整行 D = 768，只读一次词元嵌入表
float4 s[6];
float sum = 0.0f;
#pragma unroll
for (int k = 0; k < 6; ++k) {
  int idx4 = k * 32 + lane;
  float4 tv = __ldg(&tok_row4[idx4]);
  float4 pv = s_pos[idx4];
  s[k] = make_float4(tv.x + pv.x, tv.y + pv.y, tv.z + pv.z, tv.w + pv.w);
  sum += (s[k].x + s[k].y) + (s[k].z + s[k].w);
}
float mean = warp_allreduce_sum(sum) * (1.0f / 768.0f);
// 寄存器内直接计算 (s[k] - mean)^2，二次洗牌求出 inv_std 后乘 s_gamma 加 s_beta 流式写回
```

把词元嵌入行的全局读取从三遍降为单遍 `float4` 寄存器驻留、并在线程块内让 32 个批次共享同一份位置嵌入与归一化参数后（内核其余显存流量主要为 `50.33 MB` / `48.00 MiB` 的输出写出及少量索引与共享参数读取），查表访存开销大幅收敛。在官方评测规模 `B = 32, T = 512, V = 30,000, P = 2,048, D = 768` 下（每次计时前刷空 L2 缓存，取 21 轮中位数），实测中位耗时从基线的 `0.7644 ms` 降至 **`0.3725 ms`**（提速 **`2.05x`**，等效带宽 **`274.6 GB/s`**）；在 LeetGPU 官方 T4 评测中一次通过全部测试，耗时 **0.47 ms**（第 **93.1** 百分位）。

完整代码：[`token-embedding-layer-baseline.cu`](../../solutions/normalization/token-embedding-layer-baseline.cu) · [`token-embedding-layer-opt.cu`](../../solutions/normalization/token-embedding-layer-opt.cu)
