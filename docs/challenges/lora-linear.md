### [lora-linear](https://leetgpu.com/challenges/lora-linear)
*环境: CUDA*

一句话：给定输入矩阵 `x`、主权重 `W` 以及低秩适配矩阵 `A`、`B`，一次算出 `output = x @ W^T + lora_scale * (x @ A^T) @ B^T`。

拿一个最小例子来看：设 `batch = 2, d_in = 4, d_out = 3, rank = 2, lora_scale = 0.5`。先算主分支 `x @ W^T` 得到 `2×3` 的基础结果；再算低秩分支，把 `x` 乘以 `A^T` 压缩成 `2×2` 的中间矩阵 `H`，乘上 `0.5` 后再乘以 `B^T` 升回 `2×3`，最后与主分支逐元素相加。在大模型常见形状（`batch = 256, d_in = 4096, d_out = 4096, rank = 64`）下，主分支占了约 97% 的计算量，低秩中间结果 `H` 只有 `256×64`（约 64 KB）。

> baseline

朴素写法按输出坐标直接循环累加：第一个 kernel 算出缩放后的低秩中间矩阵 `H = lora_scale * (x @ A^T)`，第二个 kernel 对每个输出元素分别遍历 `d_in` 和 `rank` 累加 `x @ W^T` 与 `H @ B^T`。每次调用还会临时 `cudaMalloc` / `cudaFree` 申请和释放 `H`。

> optimized

优化版主要省掉了三块开销：
- **复用低秩缓冲区 + Split-K=8 降维**：用静态设备指针缓存 `H`，避免每次调用反复分配显存；同时把 `x @ A^T` 沿 `d_in` 拆成 8 段并行计算，把小矩阵也铺满所有 SM。
- **`128×64` 分块 + 3-MMA FP16 拆分 Tensor Core（张量核心，专用矩阵乘硬件）**：按分块推算的字节数，将块形状从 `64×64` 调整为 `128×64` 后，`W` 的读取次数减半（`batch = 256` 仅读 2 遍，按分块推算约 128 MB）；块内把每个 `float32` 拆成高半部与低半部两个 `float16`（`x = x_hi + x_lo`），用 3 次 `mma.sync.m16n8k8` 计算 `A_hi @ B_hi + A_hi @ B_lo + A_lo @ B_hi`，在保持 `float32` 级别精度（`max_abs < 1e-4`）的同时把算力从 FP32 核心切换到高吞吐 Tensor Core。
- **融合低秩升维 `H @ B^T`**：主分支循环结束后，累加器还留在寄存器（register，线程私有高速存储）里，直接在同一个 kernel 里复用 shared memory（线程块内共享片上内存）继续沿 `rank` 维度累加 `H @ B^T`，最终只写回一次 `output`。

```cuda
// 3-MMA FP16 拆分：用 3 次 Tensor Core 矩阵乘保持 FP32 级精度
mma_m16n8k8_f32_f16(c[0], c[1], c[2], c[3], ah0, ah1, b_hi[in], c[0], c[1], c[2], c[3]);
mma_m16n8k8_f32_f16(c[0], c[1], c[2], c[3], ah0, ah1, b_lo[in], c[0], c[1], c[2], c[3]);
mma_m16n8k8_f32_f16(c[0], c[1], c[2], c[3], al0, al1, b_hi[in], c[0], c[1], c[2], c[3]);
```

优化版在 LeetGPU 官方 T4 平台实测：**全部测试通过，3.03 ms，提交时第 86.2 百分位**；本地标准基准形状（`batch = 256, d_in = 4096, d_out = 4096, rank = 64`）对照测试中，基线版平均耗时 `78.68 ms`，优化版降至 `1.00 ms`（提速约 `78.5×`，零寄存器溢出）。

完整代码：[基线版](../../solutions/matmul/lora_linear_baseline.cu) · [3-MMA Tensor Core 优化版](../../solutions/matmul/lora_linear_opt.cu)
