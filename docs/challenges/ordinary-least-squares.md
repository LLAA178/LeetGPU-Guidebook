### [ordinary-least-squares](https://leetgpu.com/challenges/ordinary-least-squares)
*环境: CUDA*

给定列满秩的特征矩阵 `X`（尺寸 `n_samples x n_features`，行主序）与目标向量 `y`（长度 `n_samples`），按最小二乘闭式解公式 $\beta = (X^T X)^{-1} X^T y$ 计算回归系数向量并写入 `beta`（长度 `n_features`）。

拿官方样例（`n_samples = 5`，`n_features = 3`）走一遍：先将 `5 x 3` 的特征矩阵与自身相乘得到 `3 x 3` 对称正定格拉姆矩阵 $G = X^T X$，并算出三维向量 $b = X^T y$；随后对对称正定矩阵 $G$ 做乔列斯基分解（Cholesky 分解，把对称正定矩阵拆解为下三角矩阵 $L$ 与其转置乘积 $G = L L^T$ 的稳定数值方法），依次通过前向回代解 $L z = b$、后向回代解 $L^T \beta = z$，得到回归系数 `beta = [13.97, 29.12, 61.05]`。

> baseline

按标准正规方程流程拆成三个 kernel 顺序执行：第一个 kernel 每个线程计算 $G = X^T X$ 的一个矩阵元（双精度累加防止病态矩阵有效数字流失）；第二个 kernel 每个线程计算 $b = X^T y$ 的一个分量；第三个单块 kernel 在全局显存上执行并行乔列斯基分解与前后向三角回代并写出 `beta`。

```cuda
// 第 1 个 kernel：逐元素双精度累加对称矩阵 G = X^T X 的下三角并镜像写回
double sum = 0.0;
for (int s = 0; s < n_samples; ++s) {
  const float* row = X + static_cast<size_t>(s) * F;
  sum += static_cast<double>(row[i]) * static_cast<double>(row[j]);
}
g_base_xtx[i * F + j] = sum;  g_base_xtx[j * F + i] = sum;
```

> optimized：单核全片上融合求解 + 多尺度无原子分块格拉姆矩阵归约

官方性能测试尺寸为 `n_samples = 32, n_features = 32`（整张特征矩阵仅 `4 KiB`），同时约束上限允许扩展到 `100,000 x 1,000`。在小矩阵评测下，多次启动 kernel 与全局显存往返延迟占了大头；而在大矩阵场景下，既要跑满全部流式多处理器，又要用双精度正规方程抵抗条件数平方 $\kappa(X^T X) = \kappa(X)^2$ 带来的数值放大。优化版针对问题规模实施分层调度：

- **中小特征单次 kernel 全片上融合（覆盖 `32 x 32` 评测规模）**：当 `n_features <= 64` 且 `n_samples <= 256` 时，全程仅启动 **1 个线程块（单次 kernel 启动）**。对于 `32 x 32` 对齐输入，256 个线程各用一条 128 位 `float4` 加载即可一次把整张 `X` 搬进按步长 `65` 填充的共享内存（shared memory，线程块内高速片上内存，奇步长彻底避开存储体冲突），在片上直接算出紧凑下三角双精度数组 `s_L` 与 `s_rhs`，紧接着在同一共享内存内完成乔列斯基分解与三角回代，全程**零全局中间显存读写、零二次启动开销**。
- **前向回代与分解主循环融合**：在乔列斯基分解第 `k` 步求出对角元 $L_{kk}$ 及第 `k` 列消元乘子 $L_{ik}$ 的同一轮并行循环中，同步扣减右端项 `s_rhs[i] -= l_ik * zk`——当分解在 `k = F - 1` 收尾时，前向回代解向量 $z$ 也已同步就绪，只需再跑一轮并行后向回代即可直接写出 `beta`。
- **瘦高切分与大特征 `32 x 32` 分块对称矩阵乘**：当 `n_features <= 64` 而样本数较大时，启动最多 160 个线程块按样本段写入各自独立切片（零原子冲突），再由单块汇总求解；当 `n_features > 64` 时，用 `32 x 32` 共享内存分块配合每线程 `2 x 2` 寄存器微核仅计算下三角分块，每 32 行用单精度乘加后转入双精度主累加器，同时在消元阶段把主元列缓存到共享内存加速 $O(F^3)$ 尾矩阵更新。

```cuda
// 分解第 k 步融合前向回代：并行求出第 k 列乘子 l_ik 的同时直接更新 s_rhs[i]
for (int i = k + 1 + tid; i < F; i += kThreads) {
  const int ik_idx = (i * (i + 1)) / 2 + k;
  const double l_ik = s_L[ik_idx] * inv_lkk;
  s_L[ik_idx] = l_ik;
  s_rhs[i] -= l_ik * zk;
}
```

将三阶段全局往返压缩为单次 kernel 片上闭环求解、并为大规模超定系统配备无原子分块归约后，不同尺寸的最小二乘回归都能获得极低延迟与可靠的数值精度。在官方性能测试规模 `n_samples = 32, n_features = 32` 下（每次计时前刷空二级缓存，取 15 轮中位数），本地 Tesla T4 实测中位耗时从基线的 `0.084 ms` 降至 **`0.025 ms`**（提速 **`3.42x`**）。

完整代码：[`ordinary-least-squares-baseline.cu`](../../solutions/reduction/ordinary-least-squares-baseline.cu) · [`ordinary-least-squares-opt.cu`](../../solutions/reduction/ordinary-least-squares-opt.cu)
