### [3d-subarray-sum](https://leetgpu.com/challenges/3d-subarray-sum)
*环境: CUDA*

给定行主序存储的三维 32 位整数张量 `input`（尺寸 `N x M x K`）以及三个维度的闭区间起止下标 `[S_DEP..E_DEP]`、`[S_ROW..E_ROW]`、`[S_COL..E_COL]`，计算三维子块 `input[S_DEP..E_DEP][S_ROW..E_ROW][S_COL..E_COL]` 内所有元素的精确总和并写入 `output[0]`。

拿官方样例 1（`2 x 2 x 3` 张量，`S_DEP = 0..1`、`S_ROW = 0..0`、`S_COL = 1..2`）走一遍：第 0 层 `[[1, 2, 3], [4, 5, 1]]` 取第 0 行的第 `1..2` 列得到 `[2, 3]`，第 1 层 `[[1, 1, 1], [2, 2, 2]]` 取第 0 行的第 `1..2` 列得到 `[1, 1]`，四个元素相加 `2 + 3 + 1 + 1 = 7`，直接覆盖写入 `output[0] = 7`（评测环境的 `output` 由 `torch.empty(1)` 分配，不能假定初值为 `0`）。

> baseline

将子块内的 `D_len * R_len * C_len` 个元素展平为一维编号 `idx`，第一个 kernel 用网格跨步循环通过整除和取模还原局部坐标 `(d_off, r_off, c_off)`，逐元素读取 `input[(d * M + r) * K + c]` 并以 `long long` 累加到共享内存（shared memory，线程块内高速片上内存）完成折半归约；第二个单块 kernel 汇总所有块和并覆盖写入 `output[0]`。

```cuda
for (long long idx = global_tid; idx < total_elems; idx += stride) {
  int d_off = static_cast<int>(idx / plane_elems);
  int rem   = static_cast<int>(idx - static_cast<long long>(d_off) * plane_elems);
  int r_off = rem / C_len, c_off = rem - r_off * C_len;
  acc += static_cast<long long>(input[((S_DEP + d_off) * M + (S_ROW + r_off)) * K + (S_COL + c_off)]);
}
```

> optimized：连续维度折叠 + 单线程束负责单行字节对齐 `int4` 流式归约

官方性能测试尺寸为 `500 x 500 x 500`，截取区间 `S_DEP = 11..498`、`S_ROW = 0..499`、`S_COL = 1..489`，子块总数据量约 `477.26 MB`，由 `488 * 500 = 244,000` 条长度为 `489` 的行片段组成，且列起点 `S_COL = 1` 除以 4 余 1（非 16 字节对齐）。朴素实现每个元素都要做两次整数除法拆坐标，且只能走 32 位标量访存。优化版围绕三维切片的内存几何布局做了三项关键设计：

- **连续维度折叠与零除法行步进**：主机端启动前先检查相邻维度是否收尾相接——若列区间占满 `[0..K-1]` 且行区间占满 `[0..M-1]`，整个三维子块直接折叠为单段一维连续数组；在官方性能测试这类 `S_ROW = 0..M-1`、`C_len < K` 的场景下，相邻深度层之间的行步长完全一致（恒为 `K = 500`），`(D, R)` 直接折叠为 `244,000` 条等距行片段。即便遇到 `R_len < M` 的一般三维子块，也在 CPU 侧预计算好每次跨步的偏移增量 `ptr_step`、`r_step` 与跨层补偿 `wrap_extra`，让 GPU 内层循环仅靠整数加法和一次比较推进指针，彻底消除循环体内的整除与取模指令。
- **单线程束负责单行（Warp-per-Row）与运行时 16 字节对齐剥离**：长度为 `489` 的行片段剥离首尾后包含 `121` 个 `int4`——若让 256 线程的整个线程块处理单行，超过一半线程会空转；改由 1 个线程束（warp，锁步执行的 32 个线程）协作处理单条行片段，所有 32 个通道都能跑满，每步连续发射 4 条 128 位 `__ldg` `int4` 加载（对于窄列 `16 <= C_len < 64` 自动切到 8 通道子线程束，`C_len < 16` 切到单线程单行）。每条片段根据实时字节地址动态剥离 `0..3` 个开头元素与末尾零头，保证无论 `S_COL` 或行宽 `K` 奇偶如何，`int4` 指针始终严格满足 16 字节硬件对齐。每个通道在寄存器内跨数百行持续累加，直到 kernel 末尾才做一次 `__shfl_down_sync` 线程束归约。
- **单次 kernel 启动与末块原子收尾**：各线程块将块和写入 `volatile` 修饰的静态数组 `g_partial_sums`，经 `__threadfence()` 后执行 `atomicAdd(&g_retired, 1u)`，由最后一个完成的线程块汇总全部块和、复位计数器并直接覆盖写入 `output[0]`，全程只需**单次 kernel 启动**。

```cuda
// 每条行片段根据实时字节地址剥离 0..3 个首元素，使 vec_ptr 严格满足 16 字节对齐
int head_elems = (4 - static_cast<int>((addr >> 2) & 3u)) & 3;
if (sub_lane < head_elems)       acc += __ldg(&seg_ptr[sub_lane]);
if (tail_start + sub_lane < C)   acc += __ldg(&seg_ptr[tail_start + sub_lane]);

const int4* __restrict__ vec_ptr = reinterpret_cast<const int4*>(seg_ptr + head_elems);
// 32 通道线程束一次并发发起 4 条 128 位只读加载，行间仅靠加法推进指针（零除法）
seg_offset += ptr_step;
r_off += r_step;
if (r_off >= R) { r_off -= R; seg_offset += wrap_extra; }
```

省去上亿次坐标整除取模、并用单线程束对齐向量化把 `244,000` 条非对齐行片段变成满带宽 128 位突发读取后，三维子块归约可以直接逼近显存物理带宽上限。在官方性能测试规模 `500 x 500 x 500`（`D = 11..498, R = 0..499, C = 1..489`）下，实测中位耗时从 `2.61 ms` 降至 **`1.57 ms`**（提速 **`1.66x`**，有效带宽达 **`303.2 GB/s`**）。

完整代码：[`3d-subarray-sum-baseline.cu`](../../solutions/reduction/3d-subarray-sum-baseline.cu) · [`3d-subarray-sum-opt.cu`](../../solutions/reduction/3d-subarray-sum-opt.cu)
