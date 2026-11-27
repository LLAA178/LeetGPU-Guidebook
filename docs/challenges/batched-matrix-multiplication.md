### [batched-matrix-multiplication](https://leetgpu.com/challenges/batched-matrix-multiplication)
*环境: CUDA*

这题要算一批单精度浮点（FP32）矩阵乘法：给 `BATCH` 组 `A[M, K]` 和 `B[K, N]`，每组独立算出 `C[M, N] = A × B`。

举个小例子：假设 `BATCH = 32`、每组矩阵都是 `256 × 256`。最朴素的写法是一个线程算 `C` 里的 1 个数，沿着 `K = 256` 把 `A` 的一行和 `B` 的一列各读一遍；相邻线程会反复从全局显存读取同一行 `A` 和同一列 `B`，显存带宽很快成为瓶颈。

> baseline

用三维网格 `(N/16, M/16, BATCH)`，每个线程负责某一个批次里的 1 个输出元素，沿 `K` 累加乘加结果。

```CUDA
float acc = 0.0f;
for (int k = 0; k < K; ++k) {
    acc = fmaf(a_row[k], b_mat[static_cast<size_t>(k) * N + col], acc);
}
C[(static_cast<size_t>(b) * M + row) * N + col] = acc;
```

> tiled (64×64)

第一版优化把输出按 `64×64` 分块，每个线程块用 256 个线程、每线程算 `4×4` 个输出。沿 `K` 每次推进 16，线程用 `float4`（一次读写 4 个 FP32）把 `A/B` 搬进共享内存（shared memory，片上高速缓存），同时把 `A` 转置存放并加 4 个元素空位（padding）避免存储体冲突（bank conflict），在计算当前块时把下一块预取到寄存器里。这一版在 LeetGPU 官方 T4 上通过全部测试，耗时 **0.72 ms**（第 **79.2** 百分位）。

> optimized (128×128)

`64×64` 版里每个线程每读 8 个数只做 16 次乘加（平均每读 1 个数算 2 次），共享内存读取占比依然偏高。第二版把输出块放大到 `128×128`，256 个线程每人算 `8×8` 个输出，带来两处明显提速：
- **计算密度翻倍**：每个线程每读 16 个数能做 64 次乘加（每读 1 个数算 4 次），算力单元更饱和。
- **全局显存读取减半**：同样算完 `128×128` 个输出，搬运 `A/B` 的总数据量只有 `64×64` 版的一半，同时继续保留双缓冲预取和 `float4` 向量化读写。其余不整除或未对齐的尺寸自动走标量版。

这两步把 LeetGPU 官方 T4 实测耗时从 **0.72 ms** 压到 **0.54 ms**（缩短 25%，第 **92.5** 百分位）。

完整代码：[标量基线版](../../solutions/matmul/batched_matrix_multiplication_baseline.cu) · [64×64 分块版](../../solutions/matmul/batched_matrix_multiplication_tiled.cu) · [128×128 寄存器分块优化版](../../solutions/matmul/batched_matrix_multiplication_opt.cu)
