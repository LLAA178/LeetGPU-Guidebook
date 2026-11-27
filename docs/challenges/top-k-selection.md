### [top-k-selection](https://leetgpu.com/challenges/top-k-selection)
*环境: CUDA*

从长度为 `N` 的单精度浮点数组 `input` 中选出最大的 `k` 个数，并按从大到小的顺序写入长度为 `k` 的 `output` 数组。

先看一个 6 元素找前 `k = 2` 大的小例子：`input = [-2.0, 5.0, 1.0, 8.0, -6.0, 4.0]`，最终结果是 `[8.0, 5.0]`。把两段分块各自保留前 `k = 2` 大再汇总：前半段 `[-2.0, 5.0, 1.0]` 留下 `[5.0, 1.0]`，后半段 `[8.0, 4.0, -6.0]` 留下 `[8.0, 4.0]`；两份局部前 `2` 大合并再取前 `2` 大恰好得到 `[8.0, 5.0]`。即便全部全局前 `k` 大都扎堆在同一个连续区间里，只要每个分区都完整保留本分区的前 `k` 大，汇总结果依然百分之百精确。配合 IEEE-754 保序映射（先把 `-0.0f` 规整为 `+0.0f`，正数翻转最高符号位 `u ^ 0x80000000u`，负数全位取反 `~u`），所有比较都简化为无符号整数 `uint32_t` 比较。

> baseline

先把 `input` 映射为 32 位保序无符号整数并补齐到 `2` 的幂次长度（补位填最小值 `0u`），在全局显存上执行双调排序（bitonic sort，一种比较位置固定、适合 GPU 并行的排序网络）将整张表按降序排好，最后把前 `k` 项还原回 `float` 写出。

```cuda
__global__ void bitonic_step_kernel(uint32_t* __restrict__ keys, int padded_n, int j, int stage_k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= padded_n) return;
  int ixj = i ^ j;
  if (ixj > i) {
    uint32_t a = keys[i];
    uint32_t b = keys[ixj];
    bool descending = ((i & stage_k) == 0);
    if ((descending && a < b) || (!descending && a > b)) {
      keys[i] = b;
      keys[ixj] = a;
    }
  }
}
```

> optimized：单遍流式块内 Top-K 过滤 + L2 二级归并

官方性能测试在 `N = 50,000,000`（200 MB）中取 `k = 100`。无论是全量排序还是对五千万个元素逐一发全局原子统计，都会受制于多趟访存或原子排队。当 `k <= 256` 时，优化版把 200 MB 输入均分给 160 个线程块（每块 256 个线程，仅占 8 KiB 共享内存，每个流式多处理器可常驻 4 个块），只顺序读取**一遍** `input` 就完成筛选：

- **256 点均匀采样预估下界**：正式扫描前，块内 256 个线程先在所属区间均匀读取 256 个样本，在共享内存（shared memory，线程块内高速片上内存）里排一次双调降序，取第 `k` 大样本作为初始门槛 `thresh`。因为本块真实包含至少 `k` 个大于等于 `thresh` 的元素，扫描一开始就能过滤掉超过 99.6% 的普通元素。
- **`float4` 满带宽流式扫描与 `__ballot_sync` 零写跳过**：主循环每步以 `float4` 读入 1,024 个元素并在寄存器中与 `thresh` 比较（集齐 `k` 个候选后立即切换为严格大于 `key > thresh`，遇到全相等数组后续零元素入队）。线程束（warp，锁步执行的 32 个线程）用硬件投票指令 `__ballot_sync` 检查是否有元素越过门槛：绝大多数线程束投票结果为 `0`，直接跳过共享内存写入继续读下一批数据；极少数越过门槛的元素才压入 `s_buf` 候选区，满 512 个才触发一次片上双调重排并收紧 `thresh`（31 万元素全程仅触发约 4 次片上重排）。
- **L2 二级归并写出（`k > 256` 自动切换双级基数桶兜底）**：第一阶段结束时，160 个块各自将前 `k` 大写入静态数组 `g_block_topk`（共 `160 * 100 * 4 B = 64 KB`，完全驻留 L2 缓存）；紧接着启动单个 `<<<1, 256>>>` 归并 kernel，复用同一套片上过滤器在几十微秒内选出全局前 `k` 大写回 `output`。当 `N <= 4096` 时单块一次排完；当 `k > 256` 时自动切到 16 位双级基数直方图路径，覆盖 `1 <= k <= N` 全范围。

```cuda
// 线程束投票：无元素超过当前第 k 大门槛 thresh 时，直接跳过共享内存写入
unsigned any_warp = __ballot_sync(0xffffffffu, p0 || p1 || p2 || p3);
if (any_warp != 0u) {
  warp_append_key(k0, p0, s_buf, &s_cand_cnt);
  warp_append_key(k1, p1, s_buf, &s_cand_cnt);
  warp_append_key(k2, p2, s_buf, &s_cand_cnt);
  warp_append_key(k3, p3, s_buf, &s_cand_cnt);
}
__syncthreads();
if ((!has_k && s_cand_cnt >= k) || (s_cand_cnt >= 512)) {
  flush_smem_candidates(s_buf, k, has_k, thresh, &s_cand_cnt);
}
```

全量双调排序需要在全局显存上反复读写数百趟，而单遍块内过滤零全局原子操作、零 `cudaMalloc`，把五千万元素的操作压成了单趟 `float4` 顺序流式读取外加约 4 次 8 KiB 片上小数组归约。优化版在 LeetGPU 官方 T4 上一次通过全部测试，耗时 **1.28 ms**（第 **99.7** 百分位）；在官方评测规模 `N = 50,000,000, k = 100`（`200 MB` 输入，每次计时前刷空 L2，15 轮中位数）下，本地 Tesla T4 实测耗时从基线的 `498.90 ms` 降至 **`0.8545 ms`**（提速 **583.9 倍**，等效单遍有效带宽达 **`234.1 GB/s`**）。

完整代码：[`top-k-selection-baseline.cu`](../../solutions/reduction/top-k-selection-baseline.cu) · [`top-k-selection-opt.cu`](../../solutions/reduction/top-k-selection-opt.cu)
