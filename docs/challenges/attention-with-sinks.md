### [attention-with-sinks](https://leetgpu.com/challenges/attention-with-sinks)

一句话说明要算什么：给三张大小同为 `M × d` 的矩阵 `Q`、`K`、`V`，以及锚点数量 `num_sinks` 和窗口长度 `window_size`，每个查询位置 `i` 在满足因果约束（`j <= i`）的前提下，只关注开头的 `num_sinks` 个锚点（`j < num_sinks`）以及最近的 `window_size` 个位置（`j >= i - window_size + 1`），对其缩放点积做 Softmax（归一化指数权重）并加权求和 `V`。

先看一个小例子：假设 `M = 6`，`num_sinks = 1`，`window_size = 2`。
- 位置 `0` 看 `{0}`；
- 位置 `1` 看 `{0, 1}`；
- 位置 `2` 看 `{0, 1, 2}`（窗口 `[1, 2]` 与锚点 `{0}` 紧挨着，合成连续区间 `[0, 2]`）；
- 位置 `3` 看 `{0, 2, 3}`（锚点 `{0}` 加最近窗口 `[2, 3]`，跳过中间的 `1`）；
- 位置 `5` 看 `{0, 4, 5}`（锚点 `{0}` 加最近窗口 `[4, 5]`，跳过 `1, 2, 3`）。

每一行把允许位置上的点积除以 `sqrt(d)` 后做 Softmax，再按权重把对应行的 `V` 加起来，就得到这一行的输出。

**省了哪一步（核心优化思路）**

1. **基线版（Warp 级 Online Softmax）**：一个 Warp（线程束，32 个并行线程）负责一行 `Q_i`，直接按两段区间 `[0, num_sinks - 1]` 和 `[max(0, i - window_size + 1), i]` 遍历允许的键（若两段重叠则合并为 `[0, i]`），用 Online Softmax（在线归一化，一边扫描一边维护当前最大值 `m` 和分母 `l`，省去把完整注意力分数矩阵写回显存）配合 Warp Shuffle（线程束内寄存器洗牌规约）完成点积。
2. **优化版（分块共享内存 + 4× 行复用 + 无分支内区路径 + `exp2f`）**：相邻的 `Br = 32` 行查询共享绝大部分锚点与滑动窗口键。优化版将 `Bc = 32` 行 `K`、`V` 用 `float4`（128 位向量化访存）一次性搬入 Shared Memory（共享内存，块内高速片上缓存，共 `32 KB`），每个 Warp 同时持有 4 行 `Q` 并在寄存器内交错完成 4 路点积与洗牌规约，使每次共享内存读取的 `K`/`V` 被复用 4 次；同时预先把 `Q` 乘上 `rsqrtf(d) * log2(e)`，把指数运算换成单指令硬件 `exp2f`。对于完全落在锚点区或窗口内部的中间分块，直接走无掩码判断的快速内循环，仅在窗口边缘和因果对角线分块做掩码过滤；当 `d` 不能被 4 整除或指针未 16 字节对齐时，自动回退到通用 Warp 路径。

**实测结果（`M = 5000, d = 128, num_sinks = 4, window_size = 1024`，FP64 参考对拍）**

| 版本 | 最大误差 (`max_diff`) | 平均耗时 | 加速比 |
| --- | --- | --- | --- |
| Baseline（Warp Online Softmax） | `1.527e-07` | `3.5174 ms` | `1.00×` |
| Optimized（分块共享内存 + 4× 行复用 + `exp2f`） | `1.453e-07` | `1.7644 ms` | `1.99×` |

**LeetGPU 平台成绩（T4）**

- 优化版：`4.25 ms`，第 `98.8` 百分位

完整代码：
- 基线版：[`solutions/transformer/attention_with_sinks_baseline.cu`](../../solutions/transformer/attention_with_sinks_baseline.cu)
- 优化版：[`solutions/transformer/attention_with_sinks_opt.cu`](../../solutions/transformer/attention_with_sinks_opt.cu)
