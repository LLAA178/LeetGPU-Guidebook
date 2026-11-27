### [linear-self-attention](https://leetgpu.com/challenges/linear-self-attention)
*环境: CUDA*

一句话：给定形状为 `M×d` 的 `Q`、`K`、`V`，先用逐元素特征映射 `φ(x) = ELU(x) + 1`（`x > 0` 时为 `x + 1`，否则为 `e^x`）把 `Q`、`K` 变成非负特征，再利用矩阵乘法结合律先算出 `d×d` 的汇总矩阵 `S = φ(K)^T V` 和长度为 `d` 的归一化向量 `z = Σ_j φ(K_j)`，最后计算 `output = (φ(Q) S) / (φ(Q) z)`，把时间复杂度从 `O(M^2 d)` 降到 `O(M d^2)`。

举个小例子（`M=2, d=2`）：设 `Q = [[0, 0], [1, 1]]`、`K = [[1, 0], [0, 1]]`、`V = [[3, 4], [5, 6]]`。经过 `φ(x)` 后，`φ(Q) = [[1, 1], [2, 2]]`、`φ(K) = [[2, 1], [1, 2]]`。先算汇总矩阵 `S = φ(K)^T V = [[11, 14], [13, 16]]` 和归一化向量 `z = [3, 3]`；再算分子 `φ(Q) S` 和分母 `φ(Q) z`，相除后两行输出都是 `[4.0, 5.0]`。

> baseline

分成两个 kernel（核函数）直接按定义计算：第一个 kernel 用二维线程块让每个线程负责 `S[row, col]` 的一个元素，顺序累加全部 `M` 行，并在 `col == 0` 时顺带累加 `z[row]`；第二个 kernel 让每个线程负责 `output[row, col]` 的一个元素，顺序遍历 `k = 0..d-1` 计算分子和分母后相除。

```CUDA
__global__ void compute_kv_z_baseline_kernel(
    const float* __restrict__ K, const float* __restrict__ V,
    float* __restrict__ S, float* __restrict__ z, int M, int d) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= d || col >= d) return;

  float s_acc = 0.0f, z_acc = 0.0f;
  for (int m = 0; m < M; ++m) {
    float pk = phi_elu_plus_1(K[m * d + row]);
    s_acc = fmaf(pk, V[m * d + col], s_acc);
    if (col == 0) z_acc += pk;
  }
  S[row * d + col] = s_acc;
  if (col == 0) z[row] = z_acc;
}
```

> optimized

在 `M=10000, d=128` 上，baseline 的两个阶段都存在明显的访存与并行度瓶颈：阶段一总共只有 `128×128` 个输出元素，按 `16×16` 分块只启动 64 个 block（线程块），每个线程还要沿着 `M=10000` 串行扫描全局显存；阶段二每个线程重复从 global memory（全局显存）读取整行 `Q` 和整列 `S`。优化版针对这两个阶段分别做了分块与融合：

- **阶段一（Split-M 分块 + 寄存器复用）**：将 `S` 划分为 `64×64` 的输出块，每个 256 线程的 block 采用 `4×4` register tiling（寄存器分块，每个线程在寄存器里同时累加 16 个输出元素），并沿 `M` 维切分成 `split_m` 份并行归约（80 个 block 填满 SM），最后用 `atomicAdd` 合并到 `S` 与 `z`。每次迭代用 `float4`（一次读写 4 个 `float` 的 128 位向量指令）把 `16×64` 的 `K`（顺带完成 `φ(x)` 变换和 `z` 累加）与 `V` 搬入 shared memory（片上共享内存，速度远快于全局显存）复用。
- **阶段二（单遍融合 `φ(Q) S` 与 `φ(Q) z`）**：同样按 `64×64` 分块计算输出，块内先将 `z` 缓存在片上；每搬入一块 `64×16` 的 `Q`（以转置布局 `sQ[16][65]` 存入片上，`+1` 列填充消除 shared memory 的 bank conflict，即同个线程束访问同一存储体导致的串行冲突）时，顺带用 `__shfl_xor_sync`（warp 线程束内寄存器直接交换指令）归约出该行的分母 `φ(Q) z` 并累加到 `sDenom`，循环结束时直接除以分母写回 `output`，避免额外扫描 `Q`。

```CUDA
// 阶段二：加载 Q 分块时同步计算分母 φ(Q)·z，循环结束后直接归一化写回
d_part += __shfl_xor_sync(0xffffffff, d_part, 1);
d_part += __shfl_xor_sync(0xffffffff, d_part, 2);
if (q_load_k4 == 0) sDenom[q_load_m] += d_part;
```

> 省了哪一步与实测效果
- **沿 `M` 维切分并行 + 片上复用**：阶段一把 `10000` 行的串行扫描拆给 80 个 block 并行完成，并用 `float4` 和 `4×4` 寄存器分块把 `K`/`V` 的全局显存读取次数大幅压缩。
- **单遍融合分母归约 + 消除 bank conflict**：阶段二在加载 `Q` 的同时完成 `φ(Q) z` 归约，并用 `sQ[16][65]` 填充布局消除片上存储体冲突，两个 kernel 均为 0 寄存器溢出（0 spill）。
- **本地实测（`M=10000, d=128`）**：baseline 平均 `2.1514 ms`，优化版降至 `0.2472 ms`（提速 `8.70x`），与 FP64 CPU 参考结果的最大绝对误差为 `9.537e-7`（全部功能与边界用例通过）。

完整代码：
- baseline：[`solutions/transformer/linear_self_attention_baseline.cu`](../../solutions/transformer/linear_self_attention_baseline.cu)
- optimized：[`solutions/transformer/linear_self_attention_opt.cu`](../../solutions/transformer/linear_self_attention_opt.cu)
