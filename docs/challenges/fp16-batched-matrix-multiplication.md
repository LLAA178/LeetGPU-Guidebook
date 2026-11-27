### [fp16-batched-matrix-multiplication](https://leetgpu.com/challenges/fp16-batched-matrix-multiplication)
*环境: CUDA，Tensor Core 版本需要 SM 7.0 或更高架构*

这题要算一批半精度浮点（FP16）矩阵乘法：给 `BATCH` 组 `A[M, K]` 和 `B[K, N]`，每组独立算出 `C[M, N] = A × B`。计算过程用单精度浮点（FP32）累加减少误差，写回 `C` 时再转回 FP16。

举个小例子：假设 `M = N = K = 256`，如果直接按最朴素的写法，每个线程算 `C` 里的 1 个数，要从全局显存把 `A` 的一行（256 个数）和 `B` 的一列（256 个数）各读一遍，相邻线程会反复读取同一批数据。

> baseline

每个线程负责 1 个输出元素，沿 `K` 累加点积。同一行里相邻线程的列号连续，因此读取 `B[k * N + col]` 时地址是连续的。

```CUDA
for (int k = 0; k < K; ++k) {
    sum = fmaf(__half2float(A[k]), __half2float(B[k * N]), sum);
}
```

> tensor core

为了用上 GPU 的矩阵计算单元（Tensor Core），可以让一个线程束（warp，32 个线程）共同负责一块 `32×32` 的输出，4 个线程束拼成一个 `64×64` 的线程块输出区域。每轮沿 `K` 推进 16，调用 `wmma`（矩阵乘加接口）把 4 块 `16×16` 的结果累加在 FP32 寄存器里，最后经共享内存（shared memory，片上高速缓存）转成 FP16 写回。其余不整除或未对齐的尺寸走标量版。

```CUDA
for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
        wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
}
```

这一版在 LeetGPU 官方 A100-80GB 上通过全部测试，耗时 **0.08 ms**（第 **72.6** 百分位）。

> optimized

WMMA 版有两个明显开销：一是每次从全局显存读数据时计算单元在等，二是写回时要绕道共享内存中转。优化版把这两处开销都省掉了：
- **双缓冲流水线**：在共享内存里准备两份缓冲区。A100 上用 `cp.async`（异步拷贝指令）在后台预取下一块 `A/B`，同时计算当前块，把搬数据的时间藏在计算后面。
- **直接寄存器写回**：改用底层 `ldmatrix`（片上矩阵加载指令）和 `mma.sync` 指令。它的输出布局里每个线程正好握着相邻的两个元素，可以直接用 `half2`（一次写 2 个 FP16）写回 `C`，省去共享内存中转。

```CUDA
load_stage(stage[0], A, B, 0);
for (int t = 0; t < kSize / kDepth; ++t) {
    wait_stage();
    if (t + 1 < kSize / kDepth) load_stage(stage[(t + 1) % 2], A, B, (t + 1) * kDepth);
    compute_stage(stage[t % 2], acc);
}
```

这两步优化把 LeetGPU A100-80GB 实测耗时从 **0.08 ms** 压到 **0.04 ms**（快了 2 倍，第 **98.4** 百分位）。

完整代码：[标量版](../../solutions/matmul/fp16_batched_matrix_multiplication_baseline.cu) · [WMMA 版](../../solutions/matmul/fp16_batched_matrix_multiplication_wmma.cu) · [双缓冲流水线版](../../solutions/matmul/fp16_batched_matrix_multiplication_pipelined.cu)
