### [speculative-decoding-verification](https://leetgpu.com/challenges/speculative-decoding-verification)
*环境: CUDA*

对批次大小 `B`、草稿长度 `T`、词表大小 `V` 的草稿词元 `draft_tokens` 及草稿分布 `draft_probs`、目标分布 `target_probs`，按位置依次校验草稿词元是否被目标分布接受：首次拒绝处从修正残差分布 `max(0, q - p)` 中逆累计分布（Inverse CDF）重采样 1 个词元并把后续位置填 `0`，若 `T` 个草稿全部通过则从 `q_{T-1}` 额外采样 1 个奖励词元写入末尾，结果存入形状 `[B, T + 1]` 的 `output_tokens`。

用官方例子 `B = 1, T = 3, V = 4`、`draft_tokens = [1, 2, 0]`、`uniform_samples = [0.50, 0.70, 0.30, 0.90]` 走一遍：
1. **位置 0（草稿词元 `1`）**：查出 `p_0(1) = 0.60, q_0(1) = 0.50`，接受率 `alpha_0 = min(1, 0.50 / 0.60) ≈ 0.833`。随机数 `u_0 = 0.50 < 0.833`，通过校验，写入 `output_tokens[0] = 1`。
2. **位置 1（草稿词元 `2`）**：查出 `p_1(2) = 0.50, q_1(2) = 0.20`，接受率 `alpha_1 = 0.20 / 0.50 = 0.40`。随机数 `u_1 = 0.70 >= 0.40`，在此处拒绝并停止后续草稿校验！
3. **仅在拒绝位置重采样并清零尾部**：用 `q_1 = [0.30, 0.20, 0.20, 0.30]` 减去 `p_1 = [0.10, 0.20, 0.50, 0.20]` 并截断负数得到残差 `[0.20, 0, 0, 0.10]`，归一化后累计分布（CDF）为 `[0.667, 0.667, 0.667, 1.0]`。拿重采样随机数 `r = uniform_samples[3] = 0.90` 查找首个 `CDF >= 0.90` 的下标，命中词元 `3`，位置 `2` 的整行概率完全跳过不读，最终输出 `[1, 3, 0, 0]`。

> baseline

张量化基线照搬向量化参考写法，1 个线程块负责 1 条序列，先把 `i = 0 .. T - 1` 所有位置的残差分布重采样结果和末尾的奖励词元全部算出来，最后由 0 号线程顺着接受条件挑出截断位置写入 `output_tokens`：

```cuda
// 先遍历全部 i = 0 .. T - 1，对每一行 V 个概率做 max(0, q - p) 求和与前缀扫描重采样
for (int i = 0; i < T; ++i) {
    // 扫描长度为 V 的 p_row 与 q_row，算出 s_resampled[i] ...
}
// 再扫一遍 q_last 算出 s_bonus，最后由 tid == 0 按 u_i < min(1, q/p) 挑出输出前缀并把尾部填 0
```

> optimized：首束标量定界 + 单切片 `float4` 提前终止扫描

看清官方计时规模 `B = 64, T = 8, V = 32,768` 的数据访问盲区：`draft_probs` 与 `target_probs` 两个大张量合起来占 `134.2 MB`，但决定每条序列在哪个位置停下来的校验公式 `u_i < min(1, q_i[tok] / p_i[tok])` 其实只需要读取 **`T = 8` 个标量**！一旦确定了首个拒绝位置 `first_reject`（或全部接受），整条序列自始至终**只需要扫描唯一一个长度为 `V` 的词表切片**，其余 `T - 1 = 7` 行词表数据连碰都不必碰。

优化版用单次内核启动（1 个 512 线程块负责 1 条序列）把无效显存流量砍掉 `8` 倍以上：
- **0 号线程束并行做完 `T` 步标量校验**：`T <= 16` 不超过一个线程束（warp，32 个线程）的宽度。0 号 warp 里前 `T` 个线程各自按草稿下标点查 `p` 和 `q`，一条投票指令 `__ballot_sync` 配找首置位指令 `__ffs` 就在二十几个周期内算出 `first_reject`，同时直接把通过的前缀词元与拒绝点右侧的 `0` 一次性写好。
- **只读唯一活跃词表行（流量立减 `7/8`）**：若全部通过（`first_reject == T`），整块线程只读 `target_probs[b, T - 1]` 一行（`draft_probs` 完全不读）；若在 `first_reject < T` 处拒绝，整块线程只读 `(b, first_reject)` 这一行的 `p` 和 `q` 计算残差总和（若残差全为 `0` 自动切到 `1 / V` 均匀分布）。仅此一步就把张量读取量从 `134.2 MB` 压到 `8.4–16.8 MB`。
- **`float4` 分块累计与跨阈值立刻 `break`**：逆 CDF 查找满足 `cumsum >= r` 的最小词元下标时，按每块 `2,048` 个元素做 `float4`（单条 128 位指令读取 4 个 `float`）加载与块内洗牌前缀和；一旦当前分块累加和跨过随机数 `r`，全块线程统一归约出最小命中下标后**立刻 `break` 跳出词表循环**，平均再省掉后半程 `50%` 的词表读取：

```cuda
// 0 号 warp 用 T 次标量点查 + __ballot_sync + __ffs 直接锁定唯一需要扫描的词表行 active_pos
const unsigned int rej_mask = __ballot_sync(kFullMask, rejected_here);
const int first_rej = (rej_mask != 0u) ? (__ffs(static_cast<int>(rej_mask)) - 1) : T;

// 随后全块 512 线程只扫描 active_pos 这一行词表；分块前缀和一旦跨过 r_sample 立即全块统一 break！
if (running + tile_sum >= r_sample) {
    // 块内归约最小命中列下标写入 s_chosen_tok ...
    break;
}
```

整套实现仅需 **1 次内核启动**、零额外显存分配，且所有洗牌与 `__syncthreads()` 均保持全束/全块一致执行，覆盖 `1 <= T <= 16`、`2 <= V <= 131,072`、奇数词表与非对齐地址等全部情形。

在官方计时规模 `B = 64, T = 8, V = 32,768` 下（每次计时前刷空二级缓存，取 15 轮中位数），全位置扫描基线耗时 `1.26 ms`，单切片提前终止优化版降至 **`0.102 ms`**（提速 **`12.32x`**）；在 LeetGPU 官方 T4 评测中一次通过全部测试，耗时 **0.15 ms**（第 **91.3** 百分位）。

完整代码：[全位置扫描基线](../../solutions/reduction/speculative-decoding-verification-baseline.cu) · [单切片提前终止优化](../../solutions/reduction/speculative-decoding-verification-opt.cu)
