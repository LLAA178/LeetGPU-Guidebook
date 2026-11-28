### [count-2d-array-element](https://leetgpu.com/challenges/count-2d-array-element)
*环境: CUDA*

给定行主序连续存储的二维 32 位整数矩阵 `input`（尺寸 `N x M`）与目标整数 `K`，统计整个矩阵中值等于 `K` 的元素总个数并写入 `output[0]`。

拿官方样例 1（`2 x 3` 矩阵 `[[1, 2, 3], [4, 5, 1]]`，`K = 1`）走一遍：第 0 行首元素 `[0, 0]` 与第 1 行末元素 `[1, 2]` 等于 `1`，其余四个位置均不等于 `1`，合计命中 `2` 次，直接覆盖写入 `output[0] = 2`（评测环境的 `output` 由 `torch.empty(1)` 分配，不能假定初值为 `0`，即便像样例 2 那样命中数为 `0` 也必须显式写回 `0`）。

> baseline

二维矩阵在内存中紧密连续排布，总元素数为 `total_elems = N * M`（用 `size_t` 计算保证大尺寸安全）。分两步完成计数归约：第一个 kernel 用一维网格跨步循环扫描全部元素并在共享内存（shared memory，线程块内高速片上内存）中做树形折半归约；第二个单块 kernel 汇总所有线程块的部分和并覆盖写入 `output[0]`。

```cuda
long long cnt = 0;
for (size_t i = static_cast<size_t>(blockIdx.x) * kThreads + tid; i < total_elems; i += stride) {
  if (input[i] == K) ++cnt;
}
// 块内共享内存折半归约写入 g_base_partials，再由第 2 个 kernel 汇总覆盖写入 output[0]
```

> optimized：一维展平对齐剥离 + `int4` 八路展开只读加载 + 单核末块原子收尾

官方性能测试尺寸为 `10,000 x 10,000`（共 `1` 亿个 32 位整数，总读取量达 `400 MB`，`K = 1` 约命中一半元素）。由于统计全矩阵元素与行列坐标无关，优化的核心在于消除二维坐标拆解、拉满 128 位向量并发读取并省去二次内核启动：

- **展平一维与运行期 16 字节对齐剥离**：无需还原二维坐标 `(r, c)`，直接把矩阵视为长度 `N * M` 的一维流。先根据 `input` 的实时字节地址剥离开头和末尾 `0..3` 个非对齐元素（由 0 号线程块顺手统计），使主体 `int4` 指针严格满足 16 字节硬件对齐。
- **8 路 128 位只读加载 `__ldg` 与纯 32 位寄存器无分支累加**：启动最多 `640` 个线程块（256 线程/块），内层循环每步通过只读数据缓存通道 `__ldg` 并发读取 8 组 `int4`（共 32 个整数、128 字节），用无分支比较 `(v.x == K) + (v.y == K) + (v.z == K) + (v.w == K)` 在 32 位寄存器内高速累加，省掉内层循环的线程束同步开销。
- **循环外单次洗牌归约 + 单核末块原子收尾**：每个线程跑完跨步循环后仅做一次线程束（warp，锁步执行的 32 个线程）洗牌归约 `__shfl_down_sync`，写入 `volatile` 静态数组后经 `__threadfence()` 与 `atomicAdd(&g_retired, 1u)` 由最后完成的线程块汇总全局计数、复位计数器并直接覆盖写入 `output[0]`，全程只需**单次 kernel 启动**。

```cuda
// 展平 N * M 后每步通过 __ldg 并发发起 8 条 128 位 int4 加载，纯寄存器无分支累加
for (; idx + 7u * stride < num_vec4; idx += stride * 8u) {
  const int4 v0 = load_vec4(&vec_ptr[idx]);               const int4 v1 = load_vec4(&vec_ptr[idx + stride]);
  const int4 v2 = load_vec4(&vec_ptr[idx + 2u * stride]); const int4 v3 = load_vec4(&vec_ptr[idx + 3u * stride]);
  const int4 v4 = load_vec4(&vec_ptr[idx + 4u * stride]); const int4 v5 = load_vec4(&vec_ptr[idx + 5u * stride]);
  const int4 v6 = load_vec4(&vec_ptr[idx + 6u * stride]); const int4 v7 = load_vec4(&vec_ptr[idx + 7u * stride]);
  acc += (count_eq_int4(v0, target) + count_eq_int4(v1, target)) +
         (count_eq_int4(v2, target) + count_eq_int4(v3, target)) +
         (count_eq_int4(v4, target) + count_eq_int4(v5, target)) +
         (count_eq_int4(v6, target) + count_eq_int4(v7, target));
}
```

将二维矩阵展平为一维 128 位 8 路并发只读流、去掉内层循环同步并以单核原子收尾后，`10,000 x 10,000` 矩阵计数直接逼近显存物理读带宽上限。在官方性能测试规模 `N = 10,000, M = 10,000` 下（每次计时前刷空 L2 缓存，取 15 轮中位数），实测中位耗时从基线的 `1.48 ms` 降至 **`1.29 ms`**（有效显存读带宽达 **`309.5 GB/s`**）。

完整代码：[`count-2d-array-element-baseline.cu`](../../solutions/reduction/count-2d-array-element-baseline.cu) · [`count-2d-array-element-opt.cu`](../../solutions/reduction/count-2d-array-element-opt.cu)
