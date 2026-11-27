### [all-pairs-shortest-paths](https://leetgpu.com/challenges/all-pairs-shortest-paths)
*环境: CUDA*

给定含 `N` 个顶点的有向带权图邻接距离矩阵 `dist`（无直达边记为 `+∞`，主对角线为 `0`，不含负权环），使用 Floyd-Warshall 算法计算所有顶点对之间的最短路径距离并写入 `output`。

以官方 `N = 4` 示例为例，初始有向边为 `0 → 1 (5)`、`0 → 3 (10)`、`1 → 2 (3)`、`2 → 3 (1)`，其余非对角项全为 `+∞`：
1. **`k = 1`（允许借道顶点 1）**：原本不可达的 `0 → 2` 更新为 `min(+∞, 5 + 3) = 8`。
2. **`k = 2`（允许继续借道顶点 2）**：`0 → 3` 更新为 `min(10, 8 + 1) = 9`（路径 `0 → 1 → 2 → 3` 优于直达边 `10`），`1 → 3` 更新为 `min(+∞, 3 + 1) = 4`，得到最终最短路矩阵。

> baseline：未分块逐步松弛

最直接的并行写法先将 `dist` 拷入 `output`，随后外层循环按 `k = 0 .. N - 1` 顺序启动 `N` 次二维核函数，每次由 `N × N` 个线程独立松弛一对 `(i, j)`：

```cuda
const size_t ij = static_cast<size_t>(i) * static_cast<size_t>(n) + j;
const float cand = d[static_cast<size_t>(i) * n + k] + d[static_cast<size_t>(k) * n + j];
if (cand < d[ij]) {
    d[ij] = cand;
}
```

在性能评测规模 `N = 2,048`（单张距离矩阵 `16 MB`，总算术量 `2 × 2048^3 ≈ 171.8 亿` 次加法与取小）下，未分块做法需要启动 `2,048` 次核函数，每一步都把整张 `16 MB` 矩阵从全局显存完整读一遍并回写更新，累计产生数十 GB 的显存流量，算术强度极低。

> optimized：三阶段 `64×64` 分块 Floyd-Warshall + `4×4` 寄存器最小加矩阵乘

将 `N × N` 距离矩阵切分为 `64 × 64` 的瓦片（`N = 2,048` 时共 `32 × 32` 个瓦片，外层仅需推进 `32` 轮），把逐元素松弛转化为高算术强度的分块最小加矩阵乘（min-plus GEMM，`C = min(C, A ⊗ B)`）：

- **第一阶段（主元对角块 `D[r, r]` 与小图单次启动快路径）**：单块 256 个线程将 `64 × 64` 主元块直接装入各自的 `4 × 4` 私有寄存器，通过双缓冲片上共享内存（shared memory，块内高速缓存）交替广播每步所需的第 `k` 行与第 `k` 列，在寄存器内跑完 64 步顺序依赖松弛；当 `N <= 64` 时整道题只需这一次核函数启动即可直接从 `dist` 算完写出 `output`。
- **第二阶段（同行同列十字面板块融合更新）**：在不含负权环的图中，任意一条穿过主元块的最短简单路径在主元块内都有唯一的首个或末个中转点，因此行面板 `D[r, j]`（`j != r`）与列面板 `D[i, r]`（`i != r`）的更新可直接化归为无内部屏障的最小加矩阵乘。用二维网格一次性融合启动全部行、列面板块，把左操作数转置存入 `s_AT[64][68]`、右操作数存入 `s_B[64][64]`，直接从片上共享内存初始化累加器并一次性写回。
- **第三阶段（其余 `31×31` 独立瓦片的 `LDS.128` 向量化与脏块过滤写回）**：占总运算量约 `94%` 的非主元瓦片 `D[i, j]`（`i != r, j != r`）只读依赖本轮已定型的 `D[i, r]` 与 `D[r, j]`。块内 `16 × 16` 线程在每个内层 `k` 步仅用两条 16 字节片上宽读取指令 `float4`（`LDS.128`）无存储体冲突（bank conflict）地多播取出 4 行 `A` 与 4 列 `B`，在寄存器中完成 `16` 次 `fadd + fminf` 外积更新，64 步全程零线程块同步；写回全局显存前比对新旧 `float4`，仅在最短路实际变短时才触发全局写出，省掉收敛轮次中大量无变化的显存写流量。

```cuda
#pragma unroll 16
for (int k = 0; k < 64; ++k) {
    const float4 a = reinterpret_cast<const float4*>(s_AT[k])[ty];
    const float4 b = reinterpret_cast<const float4*>(s_B[k])[tx];

    c00 = fminf(c00, a.x + b.x); c01 = fminf(c01, a.x + b.y);
    c02 = fminf(c02, a.x + b.z); c03 = fminf(c03, a.x + b.w);
    c10 = fminf(c10, a.y + b.x); c11 = fminf(c11, a.y + b.y);
    c12 = fminf(c12, a.y + b.z); c13 = fminf(c13, a.y + b.w);
    c20 = fminf(c20, a.z + b.x); c21 = fminf(c21, a.z + b.y);
    c22 = fminf(c22, a.z + b.z); c23 = fminf(c23, a.z + b.w);
    c30 = fminf(c30, a.w + b.x); c31 = fminf(c31, a.w + b.y);
    c32 = fminf(c32, a.w + b.z); c33 = fminf(c33, a.w + b.w);
}
```

在官方性能评测规模 `N = 2,048` 下（每次计时前刷空二级缓存，取 15 轮中位数），本地 Tesla T4 未分块基线版耗时 `177.44 ms`，三阶段 `64 × 64` 分块优化版降至 **`8.79 ms`**（提速 **`20.18x`**）；在 LeetGPU 官方 T4 评测中一次通过全部测试，耗时 **7.04 ms**（第 **98.0** 百分位）。

完整代码：[未分块逐步松弛基线](../../solutions/reduction/all-pairs-shortest-paths-baseline.cu) · [三阶段 64×64 寄存器分块优化版](../../solutions/reduction/all-pairs-shortest-paths-opt.cu)
