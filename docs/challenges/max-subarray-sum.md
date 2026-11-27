### [max-subarray-sum](https://leetgpu.com/challenges/max-subarray-sum)
*环境: CUDA*

对长度为 `N` 的整数数组 `input` 和固定窗口长度 `window_size`（记为 `W`），在全部 `N - W + 1` 个长度为 `W` 的连续子数组里找出最大的元素和，写入 `output[0]`。

用官方例子 `input = [1, 2, 4, 2, 3]`、`W = 2` 看清怎样把滑动窗口拆成完全并行的两步：
1. **先算前缀和 `P`**：设 `P[0] = 0`，依次累加得到 `P = [0, 1, 3, 7, 9, 12]`（中间累加统一用 64 位整数 `long long` 保存，防止元素和中途越过 32 位 `INT_MAX` 溢出）。
2. **用前缀差并行求每个窗口和**：任意起点 `s` 的长度为 `2` 的窗口和等于 `P[s + 2] - P[s]`。四个窗口同时做一次减法即可算出 `[3 - 0, 7 - 1, 9 - 3, 12 - 7] = [3, 6, 6, 5]`，再做一次最大值归约得到 `6`。

> baseline

基线版直接让每个线程认领若干个窗口起点 `s`，在内层循环里顺着 `k = 0 .. W - 1` 把这 `W` 个数加起来，再用线程束洗牌指令 `__shfl_down_sync` 与片上共享内存（shared memory）归约出全局最大值：

```cuda
int local_max = INT_MIN;
for (int s = global_tid; s < num_windows; s += stride) {
    long long sum = 0;
    for (int k = 0; k < window_size; ++k) {
        sum += static_cast<long long>(input[s + k]);
    }
    local_max = max(local_max, static_cast<int>(sum));
}
// 块内与跨块归约 local_max 写回 output[0]
```

> optimized：64 位前缀和 + 共享内存块偏移融合差分归约

在官方计时规模 `N = 50,000, W = 25,000` 下，基线版对 `25,001` 个窗口各自累加 `25,000` 个数，总加法次数高达 `6.25 亿` 次；而整份输入数组其实只有 `200 KB`，完全装得进 Tesla T4 的 `4 MiB` 二级缓存（L2 cache）。只要换成前缀和差分，总计算量立刻从 `O(N × W)` 降到 `O(N)`；更进一步，在 `200 KB` 这种小数据规模下，连内核启动次数本身都是主要耗时项，因此还要把传统三趟前缀和压成两趟：

- **第一趟（`int4` 分块局部扫描）**：每个 256 线程块用 `int4`（单条 128 位指令一次读取 4 个连续 `int`）负责 `1,024` 个元素，`N = 50,000` 只需 `49` 个块。块内用寄存器洗牌指令 `__shfl_up_sync` 算出本块的 `long long` 局部前缀和写入静态设备端数组 `g_prefix`（驻留在 L2 缓存中），同时把本块总和写入 `g_tile_sums[tile]`，并顺手把 `output[0]` 预置为 `INT_MIN` 覆盖可能存在的脏初值。
- **第二趟（块总和片上扫描与窗口差分二合一）**：`49` 个分块总和还不到 `400` 字节。第二个核函数启动后，直接在片上共享内存里花几个时钟周期把 `g_tile_sums` 扫描成块偏移表 `s_tile_excl`，省去单独启动中间层块扫描核函数的开销；随后每个线程直接按 `P(k) = g_prefix[k] + s_tile_excl[(k - 1) >> 10]` 即时拼出窗口首尾的全局前缀和做减法，块内归约出最大值后调用一次 `atomicMax(output, block_max)` 完成汇总：

```cuda
// 片上共享内存算出每个 1024 元分块的全局偏移 s_tile_excl 后，直接拼出窗口首尾前缀差
int local_max = INT_MIN;
for (int win = global_tid; win < num_windows; win += stride) {
    const int end_idx = win + W;
    const long long p_end = g_prefix[end_idx] + s_tile_excl[(end_idx - 1) >> 10];
    const long long p_start = (win == 0) ? 0LL : (g_prefix[win] + s_tile_excl[(win - 1) >> 10]);
    const int diff = static_cast<int>(p_end - p_start);
    if (diff > local_max) local_max = diff;
}
```

0.04 ms 99.5th percentile (T4)

整套实现零 `cudaMalloc`、零 `cudaMemset`，并按数组长度分三档覆盖全范围：`N <= 4,096` 直接单块全片上一次内核搞定；`4,096 < N <= 2,097,152` 走上述两趟 L2 前缀差分管线；超大输入 `N > 2,097,152` 自动切换分块增量滑动扫描兜底，保持严格 `O(N)` 复杂度且不占额外显存。

本地 Tesla T4（CUDA 12.4）每次计时前刷空 L2 缓存，在官方计时规模 `N = 50,000, window_size = 25,000` 下取 15 轮中位数：逐窗口累加基线耗时 `1.0851 ms`，两趟前缀差分优化版降至 **`0.0122 ms`**（提速 **89.2 倍**），两版均通过 21 组全负数、首窗口最大、单窗口 `W = N`、32 位前缀和溢出与非对齐地址测试。LeetGPU 官方 Tesla T4 首次提交即通过全部测试，耗时 **`0.04 ms`（第 `99.5` 百分位）**。

完整代码：[逐窗口并行基线](../../solutions/reduction/max-subarray-sum-baseline.cu) · [64 位前缀和融合差分优化](../../solutions/reduction/max-subarray-sum-opt.cu)
