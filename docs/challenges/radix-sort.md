### [radix-sort](https://leetgpu.com/challenges/radix-sort)
*环境: CUDA*

使用基数排序（radix sort）将长度为 `N` 的 32 位无符号整数（`uint32_t`）数组按升序排列，并将有序结果写入 `output`。

以官方示例 1 为例，输入为 `[170, 45, 75, 90, 2, 802, 24, 66]`。采用最低有效位优先（LSD，从低位字节往高位字节逐趟稳定分桶）的 8 位基数排序（每趟按 `2^8 = 256` 个桶归类）：
1. **第 0 趟（位 `0..7`，即 `x & 255`）**：只有 `802 = 3×256 + 34` 跨过了 256，其低字节为 `34`，其余元素低字节等于自身；按低字节升序稳定重排后得到 `[2, 24, 802, 45, 66, 75, 90, 170]`。
2. **第 1 趟（位 `8..15`，即 `(x >> 8) & 255`）**：前 7 个高字节为 `0` 的元素保持第 0 趟已排好的先后顺序依次落入 `0` 号桶，`802`（高字节为 `3`）落入 `3` 号桶，得到 `[2, 24, 45, 66, 75, 90, 170, 802]`；更高两字节全为 `0`，最终有序结果即为该序列。

> baseline：四趟扫描-散射基数排序

传统的 GPU 8 位基数排序把每个字节趟拆成“分块统计 → 桶前缀扫描 → 重读散射”三步（4 趟共 16 次核函数启动）：
1. 每个线程块统计本块 `1,024` 个键在 256 个桶中的出现次数，按列主序写入全局分块直方图 `g_tile_hist[bin][tile]`。
2. 对每个桶跨所有块做排他前缀和（exclusive prefix sum，算出本块之前所有块落入该桶的元素总数），再对 256 个全局桶总量做一次前缀和，得到每块每桶的全局起始下标。
3. 散射核函数重新从全局显存读入这批键，在线程束（warp，32 个锁步执行的线程组）内用全束投票指令 `__ballot_sync` 逐比特匹配相同桶号并算出束内排名，最后直接写往目标位置：

```cuda
unsigned peer_mask = 0xffffffffu;
#pragma unroll
for (int bit = 0; bit < 9; ++bit) {
    const unsigned bit_val = (digit >> bit) & 1u;
    const unsigned b_mask = __ballot_sync(0xffffffffu, bit_val);
    peer_mask &= bit_val ? b_mask : ~b_mask;
}
const unsigned peer_rank = static_cast<unsigned>(__popc(peer_mask & ((1u << lane) - 1u)));
```

在性能评测规模 `N = 50,000,000`（单份数组 `200 MB`）下，这版做法每趟要读两次键、写一次键并读写一遍全局分块直方图，4 趟累计搬运约 `2.6 GB` 全局显存；同时每个线程从寄存器直接向全局显存的 256 个分散桶随机写出，同一线程束的 32 个线程往往打向几十条互不相邻的缓存行，写带宽利用率很低。

> optimized：单趟全局直方图 + Onesweep 解耦回看与片上重排

优化版实现单趟扫描基数排序（Onesweep），把 4 趟排序的总访存压到“1 次全局统计读取 + 每趟 1 读 1 写”（共 `9N = 1.8 GB`），并把随机全局散射改造成连续合并写出：

- **单趟 `uint4` 全局直方图与平凡字节跳过**：首个核函数用 160 个常驻块配合 `uint4`（单次 16 字节合并读取 4 个整数）只扫一遍 `input`，在片上共享内存（shared memory，块内高速缓存）中同时统计全部 4 个字节位段的 `4 × 256` 桶计数。收尾块在片上算出各趟的 256 桶全局起始偏移，并检查是否存在所有元素字节完全相同的平凡趟（例如全相等数组或高位全零数据），在设备端动态规划后续步骤的乒乓缓冲路径，自动跳过无须移动的字节趟；当 `N <= 2048` 时则直接走单块全片上基数排序，一次核函数启动即可排完所有位段。
- **`__match_any_sync` 硬件分桶与 64 位纪元解耦回看**：每个工作块（256 线程处理 `4,096` 个键）先通过 `atomicAdd` 动态领取单调递增的块号避免回看死锁。各线程束按连续子块顺序读入数据后，直接调用硬件等值匹配指令 `__match_any_sync`（单条指令找出同一线程束内桶号相同的全部线程掩码）完成稳定排名与束内计数。随后 256 个线程一人负责一个桶，将 32 位“轮次+状态”标记与 32 位桶计数组合成单个 64 位状态字发布到全局表，并行向前回看前驱块的完整前缀；凭借递增轮次号（epoch）区分不同趟次与不同调用，全程省去了状态表清零开销。
- **共享内存分桶重排与合并写回**：拿到本块 256 个桶的全局写起点后，线程先将寄存器中的 `4,096` 个键按桶序重排进片上共享内存 `s_keys[4096]`。此时同一桶的键在片上紧密相邻（平均每桶连续 16 个整数、64 字节），再由 256 个线程按连续下标顺序写出到全局显存，同一线程束的写出请求自然聚合到相邻的显存扇区上。

```cuda
const unsigned digit = (keys[r] >> shift) & 0xffu;
const unsigned peer_mask = __match_any_sync(0xffffffffu, digit);
const unsigned peer_rank = static_cast<unsigned>(__popc(peer_mask & ((1u << lane) - 1u)));
const int leader = __ffs(peer_mask) - 1;
unsigned wb = 0u;
if (lane == leader) {
    wb = s_warp_hist[warp][digit];
    s_warp_hist[warp][digit] = wb + static_cast<unsigned>(__popc(peer_mask));
}
wb = __shfl_sync(0xffffffffu, wb, leader);
ranks[r] = static_cast<uint16_t>(wb + peer_rank);
```

在 `N = 50,000,000` 基准评测中（每次计时前刷空二级缓存，取 15 轮中位数），四趟扫描散射基线耗时 `33.33 ms`，Onesweep 解耦回看优化版降至 **`12.92 ms`**（提速 **`2.58x`**，吞吐达 **`3,868.9 Mkeys/s`**）。

完整代码：[四趟扫描散射基线](../../solutions/reduction/radix-sort-baseline.cu) · [Onesweep 解耦回看优化版](../../solutions/reduction/radix-sort-opt.cu)
