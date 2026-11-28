### [2d-subarray-sum](https://leetgpu.com/challenges/2d-subarray-sum)
*环境: CUDA*

给定尺寸为 `N × M` 的二维 32 位有符号整数矩阵 `input` 以及闭区间边界 `[S_ROW, E_ROW]` 与 `[S_COL, E_COL]`，计算子矩阵内所有元素的总和并写入 `output[0]`。

以官方示例 1 为例，输入为 `2 × 3` 矩阵 `[[1, 2, 3], [4, 5, 1]]`，查询区间 `S_ROW = 0, E_ROW = 1, S_COL = 1, E_COL = 2` 对应第 `0..1` 行、第 `1..2` 列的右下角 `2 × 2` 子块 `[[2, 3], [5, 1]]`，将这 4 个元素相加得到 `2 + 3 + 5 + 1 = 11`，写入 `output[0]`。评测框架传入的 `output` 可能带有未初始化的旧值，因此结果写出时直接覆盖 `output[0]`。

> baseline：二维标量分块归约 + 二次汇总

基线版只在有效子矩阵尺寸 `num_rows × col_len` 上展开一维网格跨步循环（grid-stride loop，固定总线程数并以整个网格宽度为步长推进），跳过子矩阵外的无关数据；每个线程块用 64 位整数 `long long` 归约块内局部和写入临时数组，再启动第二个单块核函数汇总各块结果并覆盖写入 `output[0]`：

```cuda
for (size_t idx = gtid; idx < total; idx += gstride) {
    const int r = s_row + static_cast<int>(idx / static_cast<size_t>(col_len));
    const int c = s_col + static_cast<int>(idx % static_cast<size_t>(col_len));
    local_sum += input[static_cast<size_t>(r) * static_cast<size_t>(m) + c];
}
```

在性能评测规模 `N = 10,000, M = 10,000, S_ROW = 0..9998, S_COL = 1..9999`（有效数据约 `399.92 MB`）下，这版做法存在两处损耗：每次循环都要执行 64 位整数除法与取模来反算行列坐标，而且全程只用 4 字节标量读取，显存请求发射密集，外加两次核函数启动间隔。

> optimized：逐行动态对齐剥离 + `int4` 向量化单次核函数归约

观察评测形状 `M = 10,000, S_COL = 1` 可以发现：每行起始地址 `&input[r * M + S_COL]` 相对 16 字节边界偏移了 4 字节，若简单要求 `S_COL % 4 == 0` 才启用 `int4`，就会退回慢速标量路径。优化版按子矩阵列宽自适应分派，在单次核函数启动内跑满显存带宽：

- **逐行头部剥离与 `int4` 四路展开**：当子矩阵列宽 `col_len >= 64` 时，启动最多 160 个常驻块（铺满 Tesla T4 的 40 个流多处理器），每个线程束（warp，32 个锁步执行的线程组）协作扫描整行。进入任意一行 `r` 时，先根据该行首指针 `row_ptr` 的运行期字节地址算出距下一个 16 字节对齐边界还差 `head`（`0..3`）个整数，由前 `head` 个线程一次性并行读入；对齐后的主体区间交给全束 32 个线程以 `int4`（单条指令合并加载 16 字节、4 个整数）做 4 路循环展开顺序流式读取，末尾剩余的 `0..3` 个元素同样一步收尾，无论 `S_COL`、`M` 奇偶或基地址偏移如何都能稳定走满 128 字节缓存行合并传输。
- **窄列无除法二维条带路径**：当 `col_len < 64`（例如单列 `S_COL == E_COL` 跨上万行）时，若仍让整个线程束负责一行会闲置大量线程。此时按 2 的幂次列宽（`1, 2, 4, 8, 16, 32`）在块内做位运算切分，单列场景下块内 256 个线程同时并行累加 256 行，全程零整数除法。
- **尾块原子收尾单次写出**：各块把 64 位局部和写入全局缓冲后通过 `__threadfence()` 刷出可见性，再由 `atomicAdd` 选出最后一个完成的线程块汇总所有分块和、重置计数器并直接覆盖写入 `output[0]`，省去第二次核函数启动与清零开销。

```cuda
const uintptr_t addr = reinterpret_cast<uintptr_t>(row_ptr);
int head = (4 - static_cast<int>((addr >> 2) & 3u)) & 3;
if (head > col_len) head = col_len;
if (lane < head) acc += __ldg(&row_ptr[lane]);

const int rem = col_len - head;
const int vec4_cnt = rem >> 2;
const int4* __restrict__ row4 = reinterpret_cast<const int4*>(row_ptr + head);
```

在 `10,000 × 10,000`（`S_ROW = 0..9998, S_COL = 1..9999`）基准评测中（每次计时前刷空二级缓存，取 15 轮中位数），基线版耗时 `1.74 ms`，优化版降至 **`1.29 ms`**（提速 **`1.35x`**，有效带宽达 **`309.0 GB/s`**）；在 LeetGPU 官方 T4 评测中一次通过全部测试，耗时 **1.57 ms**（第 **93.2** 百分位）。

完整代码：[二维标量两阶段基线](../../solutions/reduction/2d-subarray-sum-baseline.cu) · [逐行对齐 int4 单次归约优化版](../../solutions/reduction/2d-subarray-sum-opt.cu)
