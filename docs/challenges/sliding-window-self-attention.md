### [sliding-window-self-attention](https://leetgpu.com/challenges/sliding-window-self-attention)

一句话说明要算什么：给三张大小同为 `M × d` 的矩阵 `Q`、`K`、`V` 和窗口半径 `window_size`，每个位置 `i` 只跟左右距离不超过 `window_size` 的邻居 `j ∈ [max(0, i - window_size), min(M - 1, i + window_size)]` 计算缩放点积注意力并输出加权和。

先看一个小例子：假设 `M = 4`，`window_size = 1`。
- 位置 `0` 只看位置 `0, 1`；
- 位置 `1` 只看位置 `0, 1, 2`；
- 位置 `2` 只看位置 `1, 2, 3`；
- 位置 `3` 只看位置 `2, 3`。

每一行把窗口内的点积除以 `sqrt(d)` 后做 Softmax（归一化指数权重），再按权重把对应行的 `V` 加起来，就得到这一行的输出。

**省了哪一步（核心优化思路）**

1. **基线版（Warp 级 Online Softmax）**：一个 Warp（线程束，32 个并行线程）负责一行 `Q_i`，只遍历 `[max(0, i - W), min(M - 1, i + W)]`，用 Online Softmax（在线归一化，一边扫邻居一边维护当前最大值 `m` 和分母 `l`，省去把整行分数存回显存的中间矩阵），再配合 Warp Shuffle（线程束内寄存器洗牌规约）求点积。
2. **优化版（单次加载共享窗口 + 寄存器双遍 Softmax + `exp2f`）**：相邻的 `Br = 32` 行查询窗口高度重叠——以官方基准形状 `M = 5000, d = 64, window_size = 16` 为例，整个块 32 行查询的窗口并集总共只需要 `32 + 2 × 16 = 64` 行 `K`/`V`（共 `32 KB`）。优化版用 `float4`（128 位向量化访存）一次性把整块需要的 `K`/`V` 搬进 Shared Memory（共享内存，块内高速片上缓存），整个 kernel 只需一次 `__syncthreads()`、无外层分块循环；块内每 8 个线程负责 1 行 `Q`，第一遍在寄存器内算完 33 个窗口分数（3 次 Warp Shuffle）并求最大值，第二遍直接用预缩放 `Q` 配合单指令硬件 `exp2f` 累加 `P @ V`，省去 Online Softmax 每步重缩放输出累加器的开销。对于更大的 `d` 或 `window_size`，自动回退到分块共享内存路径或通用 Warp 路径。

**实测结果（`M = 5000, d = 64, window_size = 16`，FP64 参考对拍）**

| 版本 | 最大误差 (`max_diff`) | 平均耗时 | 加速比 |
| --- | --- | --- | --- |
| Baseline（Warp Online Softmax） | `1.788e-07` | `0.0943 ms` | `1.00×` |
| Optimized（单次加载共享窗口 + 寄存器双遍 Softmax） | `1.490e-07` | `0.0452 ms` | `2.09×` |

**LeetGPU 平台成绩（T4）**

- 第一版（分块共享内存循环 + 4× 行复用）：`0.25 ms`，第 `67.5` 百分位
- 第二版（单次加载共享窗口 + 零栈帧寄存器双遍 Softmax）：`0.13 ms`，第 `98.7` 百分位（耗时缩短 `48%`）

完整代码：
- 基线版：[`solutions/transformer/sliding_window_self_attention_baseline.cu`](../../solutions/transformer/sliding_window_self_attention_baseline.cu)
- 优化版：[`solutions/transformer/sliding_window_self_attention_opt.cu`](../../solutions/transformer/sliding_window_self_attention_opt.cu)
