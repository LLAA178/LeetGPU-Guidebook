### [histogramming](https://leetgpu.com/challenges/histogramming)
*环境: CUDA*

统计长度为 `N` 的整数数组 `input` 中每个落在 `[0, num_bins)` 范围内的整数出现次数，把结果写入长度为 `num_bins` 的数组 `histogram`（越界整数安全忽略，且 `histogram` 传入时可能含有未清零的旧值）。

举个例子：`input = [0, 1, 2, 1, 0]`，`num_bins = 3`。如果所有线程直接往全局显存里的 `histogram[0..2]` 做原子加法（`atomicAdd`，保证多线程同时加同一个桶时不丢数的硬件读改写指令），成千上万个线程会在同一个显存地址前排队。更省时间的做法是拆成两级：
1. **块内私有分桶**：每个线程块先在片上共享内存（shared memory，速度接近寄存器的块内高速缓存）里建一份局部桶，把 `[0, 1, 2]` 和 `[1, 0]` 分别累加出局部计数 `[1, 1, 1]` 与 `[1, 1, 0]`。
2. **块间一次汇总**：所有输入扫完后，每个块只把非零局部桶往全局 `histogram` 累加一次，合并得到最终结果 `[2, 2, 1]`。

> baseline：块内单份共享内存直方图

基线版先用 `cudaMemsetAsync` 把全局 `histogram` 清零，再让每个线程块在共享内存开单份 `s_hist[1024]`。块内线程通过网格跨步循环（grid-stride loop，固定总线程数，每个线程每次跨过整个网格宽度 `gridDim.x * blockDim.x` 继续处理后续元素）遍历输入并在 `s_hist` 上做 `atomicAdd`，最后把块内结果写回全局显存：

```cuda
for (int i = global_tid; i < n; i += stride) {
    const unsigned v = static_cast<unsigned>(input[i]);  // 单次无符号比较同时过滤负数与 >= num_bins
    if (v < static_cast<unsigned>(num_bins)) {
        atomicAdd(&s_hist[v], 1);
    }
}
```

这版把全局显存原子排队转移到了片上共享内存，但还有三处开销：每次只读 4 字节标量浪费了显存带宽；块内 8 个线程束（warp，GPU 上 32 个锁步执行的线程组）挤在同一份 `s_hist` 上仍会产生共享内存存储体冲突（bank conflict，多个线程同时读写同一个共享内存通道导致硬件串行排队）；当输入全是同一个数（如全 `2`）时，同一线程束的 32 个线程还会发生 32 路原子串行。

> optimized：多副本私有分桶 + 固定常驻网格 + `int4` 向量化与同值快路径

为了逼近纯读取显存带宽下限并消除偏斜数据的冲突退化，优化版把访存、分桶和汇总压到最简：

- **按线程束拆分私有直方图**：当 `num_bins <= 256`（官方计时规模）时，块内 8 个线程束各自独占一行 `s_hist[8][256 + 1]`（末尾 `+ 1` 错开存储体映射），不同线程束之间零原子竞争；当 `256 < num_bins <= 1024` 时切到 4 份 `s_hist[4][1025]`（共 16.4 KB），兼顾低冲突与流多处理器（SM）驻留块数。
- **线程束同值快路径**：每个线程用 `int4` 一次从全局显存读入 4 个整数（16 字节合并访存，循环 4 路展开提升访存并发度）。写入共享内存前，用一条寄存器洗牌指令 `__shfl_sync` 和一条全束投票指令 `__all_sync` 检查整个线程束读到的 128 个整数是否全相等：均匀分布时两拍指令直接跳过；一旦遇到全相同输入，仅由 `lane 0` 对私有桶加一次 `128`，彻底绕开 32 路原子串行。
- **固定常驻网格跨步**：网格大小上限固定为 `160` 个常驻块（对应 40 个 SM 各驻留 4 个块，打满每 SM 32 个常驻线程束上限），扫完整个 `200 MB` 输入后全局最多只发生 `160` 次块间汇总，省掉成千上万个短命块的共享内存清零与同步开销；当 `N <= 2048` 时直接启动单块覆盖写出 `histogram`，连 `cudaMemsetAsync` 一并省去。

```cuda
const int same4 = (v.x == v.y) & (v.z == v.w) & (v.x == v.z);
const int ref = __shfl_sync(0xffffffffu, v.x, 0);
if (__all_sync(0xffffffffu, same4 & (v.x == ref))) {
    if (lane == 0 && static_cast<unsigned>(ref) < u_bins) s_hist[warp][ref] += 128;
} else {
    if (static_cast<unsigned>(v.x) < u_bins) atomicAdd(&s_hist[warp][v.x], 1);
    if (static_cast<unsigned>(v.y) < u_bins) atomicAdd(&s_hist[warp][v.y], 1);
    if (static_cast<unsigned>(v.z) < u_bins) atomicAdd(&s_hist[warp][v.z], 1);
    if (static_cast<unsigned>(v.w) < u_bins) atomicAdd(&s_hist[warp][v.w], 1);
}
```

0.81 ms 96.8th percentile (T4)

省掉标量读取开销、块内跨线程束原子冲突和海量短块同步后，在本地 Tesla T4（CUDA 12.4）上实测官方计时规模 `N = 50,000,000, num_bins = 256`（每次刷空 L2，15 轮中位数）：均匀分布下耗时由基线的 `0.698 ms` 降至 **`0.657 ms`**（有效读取带宽达 `304.4 GB/s`），全相同元素极端场景下更从 `1.647 ms` 降至 **`0.655 ms`**（提速 **2.51 倍**）。LeetGPU 官方 Tesla T4 首次提交即通过全部测试，耗时 **`0.81 ms`（第 `96.8` 百分位）**。

完整代码：[共享内存单桶基线](../../solutions/reduction/histogramming-baseline.cu) · [多副本向量化常驻网格](../../solutions/reduction/histogramming-opt.cu)
