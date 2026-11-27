### [grouped-gemm](https://leetgpu.com/challenges/grouped-gemm)

这题要算的是**分组矩阵乘法（Grouped GEMM）**：把总行数 `M_total` 的输入矩阵 `A` 按 `group_offsets` 切成 `G` 段连续行，每段 `A[s:e]` 分别乘以自己对应的权重矩阵 `B[g]`（形状 `K × N`），写回输出 `C[s:e]`；这是混合专家模型（MoE，Mixture-of-Experts）推理里一次启动算完所有专家投影的核心算子。

举个小例子：假设 `G = 2, M_total = 4, K = 2, N = 3`，`group_offsets = [0, 2, 4]`。第 0 组占 `A` 的第 `[0, 2)` 行，与 `B[0]` 相乘写入 `C[0:2]`；第 1 组占第 `[2, 4)` 行，与 `B[1]` 相乘写入 `C[2:4]`。如果某组满足 `group_offsets[g] == group_offsets[g+1]`，说明该组为空，跳过不写即可。

**基础版（Baseline）** 直接按 `(M_total, N)` 启动二维线程网格，每个线程先对 `group_offsets` 做二分查找确定自己属于哪一组 `g`，再沿 `K` 维循环做点积。写法直观，但每个元素都要从显存重复读取整行 `A` 和整列 `B[g]`，带宽浪费严重。

**优化版（128×128 双缓冲寄存器分块）** 省掉了两类开销：
1. **设备端按组划分行块（Tile，分块），省掉主机同步**：因为 `G <= 64`，每个线程块（Block）在核函数开头直接扫描 `group_offsets` 把 `blockIdx.y` 映射到唯一的 `(group_id, local_tile_m)`，保证同一线程块只算同一组的数据，既不需要把 `group_offsets` 拷回 CPU，也不需要为每组单独启动核函数。
2. **共享内存（Shared Memory）双缓冲 + `float4` 向量化访存 + 每线程 `8×8` 寄存器微块**：每个 `256` 线程的块负责 `128×128` 输出块，沿 `K` 维以 `BK = 8` 推进；在计算当前缓冲块的同时预取下一块到寄存器，把大量重复的显存访问压缩为片上寄存器复用；对 `K % 8 != 0` 或 `N % 4 != 0` 的非对齐形状自动回退到通用 `16×16` 共享内存分块核函数。

在官方性能形状 `G = 8, M_total = 8192, K = 1024, N = 2048`（不均衡分组，FP32）上的实测结果：
- 本地 GPU 对比：Baseline `74.298 ms`（`462.46 GFLOP/s`）→ Opt `7.635 ms`（`4500.32 GFLOP/s`，加速 **9.73x**，与 Baseline 最大误差 `0.00e+00`）
- LeetGPU 平台提交（T4）：**13.44 ms，第 90.5 百分位**

完整代码：
- 基础版：[`solutions/matmul/grouped_gemm_baseline.cu`](../../solutions/matmul/grouped_gemm_baseline.cu)
- 优化版：[`solutions/matmul/grouped_gemm_opt.cu`](../../solutions/matmul/grouped_gemm_opt.cu)
