### [k-means-clustering](https://leetgpu.com/challenges/k-means-clustering)
*环境: CUDA*

给定长度为 `sample_size` 的二维样本坐标数组 `data_x`、`data_y` 以及 `k` 个初始聚类中心 `initial_centroid_x`、`initial_centroid_y`，执行最多 `max_iterations` 轮二维 K-Means 迭代：每轮先将每个样本分配给欧氏距离平方最小的簇下标 `0 .. k - 1`（距离相同时取最小簇下标），再将每个非空簇的中心更新为簇内样本坐标均值（空簇保持原中心不变），最后输出最后一轮分配的 `labels` 与更新后的 `final_centroid_x`、`final_centroid_y`。

用官方四点两簇例子 `data_x = [1, 2, 8, 9]`、`data_y = [1, 2, 8, 9]`、初始中心 `[(1, 1), (8, 8)]`、`max_iterations = 10` 走一遍：
1. **首轮分配与均值更新**：前两个点 `(1, 1)`、`(2, 2)` 离 0 号中心 `(1, 1)` 更近，归入簇 `0`；后两个点 `(8, 8)`、`(9, 9)` 归入簇 `1`，得到 `labels = [0, 0, 1, 1]`。随后簇 `0` 中心更新为 `((1+2)/2, (1+2)/2) = (1.5, 1.5)`，簇 `1` 中心更新为 `((8+9)/2, (8+9)/2) = (8.5, 8.5)`。
2. **不动点收敛**：第 2 轮用新中心 `(1.5, 1.5)` 与 `(8.5, 8.5)` 重新分配，仍然得到 `labels = [0, 0, 1, 1]`，算出的新中心与上一轮逐比特相等，后续第 3~10 轮结果保持不变。注意当 `max_iterations` 较小尚未收敛时，输出的 `labels` 对应最后一轮更新中心**之前**的那次分配结果。

> baseline

基线版先把初始中心拷贝到输出数组，接着在主机端循环 `max_iterations` 轮，每轮依次启动两个核函数：第一个核函数为每个点扫描全部 `k` 个中心、写入 `labels[i]` 并用双精度原子加法累加各簇坐标和与计数；第二个核函数用计数相除更新非空簇中心并清零累加器：

```cuda
// 每轮启动一次分配核函数与一次中心更新核函数，共 2 * max_iterations 次内核启动
for (int it = 0; it < max_iterations; ++it) {
    baseline_assign_kernel<<<n_blocks, 256>>>(
        data_x, data_y, final_centroid_x, final_centroid_y, labels, sample_size, k);
    baseline_update_kernel<<<k_blocks, 256>>>(final_centroid_x, final_centroid_y, k);
}
```

> optimized：多流多处理器寄存器常驻协作单核 + 双精度无原子洗牌归约 + 收敛早退

看官方计时规模 `sample_size = 10,000, k = 5, max_iterations = 30`：
- 整份输入坐标 `data_x` 和 `data_y` 加起来只有 **`80 KB`**，却被基线版来回启动 **60 次核函数**、反复从显存重读 30 遍，且全局原子加法让 10,000 个线程争抢区区 5 个计数器；一旦中心在第 10 多轮提前收敛，后半程迭代更是在原地空转。
- 数值精度上，大样本单精度乱序原子累加会产生均值漂移，一旦某轮中心偏移 1 个最低有效位（ULP），就可能在下一轮把骑在中垂线上的边界样本翻到另一簇。

优化版按点数与簇数构建三档全覆盖流水线，在官方计时规模下把 **60 次内核启动压缩成 1 次**：
- **多 SM 寄存器常驻协作核函数（`sample_size <= 20,480, k <= 32`，默认主路径）**：通过 `cudaLaunchCooperativeKernel` 启动 `20` 个常驻线程块（每块 256 线程，每线程用寄存器永久保存 `2` 个样本点）。核函数入口只读 **1 次** `data_x`、`data_y`，后续全部 30 轮距离计算均在寄存器与片上共享内存（shared memory）中的 `s_cx`、`s_cy` 之间完成，零全局点坐标重读。当 `k <= 8` 时，每个线程直接在双精度寄存器数组 `tx[c]`、`ty[c]` 中累加各簇坐标，经线程束洗牌指令 `__shfl_down_sync` 归约后写出每块仅 `5` 组的局部和，彻底消灭样本级原子争用；跨块采用轻量级相位翻转（sense-reversing）二级缓存（L2 cache）栅栏同步——每块在栅栏后各自从一致性 L2 读取全部 `20` 个块的局部和更新本块共享内存中心，把每轮全网格同步次数压到仅 **1 次**：

```cuda
// 样本常驻寄存器，每轮算完局部双精度和后经单次网格同步即可由各块同步更新中心并检测收敛
coop_grid_sync(num_blocks, local_sense);
// 各块汇总 <= 40 份块局部和得到新中心 new_x, new_y；若全部 k 个中心逐比特未变则立即 break
if (__float_as_uint(new_x) != __float_as_uint(old_x) ||
    __float_as_uint(new_y) != __float_as_uint(old_y)) {
    atomicOr(&s_any_changed, 1);
}
```

- **收敛即停与单次标签落盘**：一旦某轮更新后的全部 `k` 个中心与上一轮逐比特一致，所有常驻块同步跳出循环；循环退出时才把寄存器里保存的最后一轮 `lbl[0..3]` 一次性合并写回 `labels`，既省去中间 29 轮的全局 `labels` 写流量，又严格对齐未收敛截断时的语义。
- **单块寄存器兜底与百万样本共享内存直方图路径**：不支持协作启动的环境自动切入 `1,024` 线程单块寄存器常驻核函数；超大样本（至 `1,000,000`）或大簇数（至 `k = 1,000`）走片上共享内存直方图归约与中心分块扫描核函数，并同样配备设备端零主机同步的 `g_converged` 早退标志。

在官方性能测试规模 `sample_size = 10,000, k = 5, max_iterations = 30` 下（每次计时前刷空二级缓存，取 15 轮中位数），本地 Tesla T4 多趟全局原子基线耗时 `0.596 ms`，寄存器常驻协作单核优化版降至 **`0.305 ms`**（提速 **`1.96x`**）。

完整代码：[多趟全局原子基线](../../solutions/reduction/k-means-clustering-baseline.cu) · [寄存器常驻协作单核优化](../../solutions/reduction/k-means-clustering-opt.cu)
