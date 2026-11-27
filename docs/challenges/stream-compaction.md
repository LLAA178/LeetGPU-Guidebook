### [stream-compaction](https://leetgpu.com/challenges/stream-compaction)
*环境: CUDA*

按原顺序挑出输入数组中所有大于 `0` 的浮点数，紧凑写到输出数组开头（平台已将输出预置为 `0`，尾部无需补写）。

举个例子：输入 `[1, -2, 3, 0, -1, 4]`，挑出正数后输出 `[1, 3, 4, 0, 0, 0]`。并行化的核心难点在于：每个正数的写入下标等于**它前面一共有多少个正数**。如果把数组切成两块 `[1, -2, 3]` 和 `[0, -1, 4]`：
1. **块内统计**：第 0 块有 `2` 个正数，第 1 块有 `1` 个正数。
2. **扫描块偏移**：对块计数 `[2, 1]` 求排他前缀和（exclusive scan，即每项等于前面各项之和），得到每块的全局写入起点 `[0, 2]`。
3. **块内紧凑写回**：第 1 块知道前面共有 `2` 个正数，就把自己唯一的正数 `4` 写到 `output[2 + 0]`，各块互不干扰且严格保持原顺序。

> baseline

单线程从左到右扫描，用 `next` 记录下一个写入位置：

```cuda
int next = 0;
for (int i = 0; i < n; ++i) {
    float v = input[i];
    if (v > 0.0f) output[next++] = v;
}
```

> 分块双遍扫描

每块 256 个线程负责 1,024 个元素。第一遍用线程束投票指令 `__ballot_sync`（32 个线程同时判断 `v > 0` 并拼成一个 32 位掩码）配合 `__popc`（统计二进制中 `1` 的个数）算出每块的正数总数；中间对块计数做两级排他前缀和；第二遍重读输入，在每个线程（lane）上用位掩码算出前面线程的正数个数，直接定位写入地址：

```cuda
const unsigned lower_lanes = (1u << lane) - 1u;
if (values[row] > 0.0f) {
    output[base + __popc(masks[row] & lower_lanes)] = values[row];
}
base += __popc(masks[row]);
```

1.48 ms 66.7th percentile (A100)

> 向量化双遍 + 共享内存连续写回

双遍标量版的主要瓶颈有两处：按单个 `float` 读取没有跑满显存带宽，且第二遍过滤后幸存的正数直接零散写往全局内存，写合并效率低。优化版做了三点改进：

- **16 字节对齐 `float4` 读取**：先把开头零散的 1~3 个元素单独处理，主体部分每条指令一次读入 4 个浮点数，指令数减为原来的四分之一。
- **共享内存（shared memory）中转写回**：第二遍先把本块挑出的所有正数在片上共享内存 `s_out` 里排成连续数组，再由全块线程以 `float4` 向量化连续写回全局显存，把离散写变成满带宽合并写。
- **静态设备缓冲区**：预分配静态设备数组存放块偏移，一亿元素以内省去运行时动态分配（`cudaMallocAsync`）开销。

0.36 ms 96.5th percentile (A100)

这一版把显存读写双端都换成了 128 位向量化连续事务，在 A100 上耗时从 1.48 ms 降到 **0.36 ms**（提速 4.1 倍）。仓库中同时保留了一份[单遍解耦回看（Decoupled Look-back）实现](../../solutions/reduction/stream-compaction-lookback.cu)供对比：它在容量受限的 T4 上更快（5,000 万元素耗时从 1.69 ms 降到 1.04 ms），在带宽充裕的 A100 上则以向量化合并写回版（0.36 ms vs 1.66 ms）表现最优。

完整代码：[单线程基线](../../solutions/reduction/stream-compaction-baseline.cu) · [双遍分块扫描](../../solutions/reduction/stream-compaction.cu) · [单遍回看](../../solutions/reduction/stream-compaction-lookback.cu) · [向量化双遍扫描](../../solutions/reduction/stream-compaction-vectorized.cu)
