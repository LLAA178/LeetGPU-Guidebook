### [causal-depthwise-conv1d](https://leetgpu.com/challenges/causal-depthwise-conv1d)
*环境: CUDA*

给定按通道在最内维存储（channels-last，形状 `[B, L, D]`）的输入序列 `x`、逐通道卷积核 `weight`（形状 `[D, K]`）和偏置 `bias`（形状 `[D]`），对每个通道独立计算一维因果卷积：`output[b, l, d] = bias[d] + sum_{k=0}^{min(K-1, l)} weight[d, k] * x[b, l - k, d]`（序列起点左侧 `l - k < 0` 的位置按 `0` 处理）。

拿官方样例 `B = 1, L = 4, D = 2, K = 3` 演示：`x = [[[1, 2], [3, 4], [5, 6], [7, 8]]]`，`weight = [[1, 0, -1], [1, 1, 1]]`，`bias = [0, 0]`。这里 `weight[d, 0]` 乘在当前时刻 `x[b, l, d]` 上，`weight[d, 1]` 和 `weight[d, 2]` 依次乘在前一时刻 `l - 1` 和更早时刻 `l - 2` 上，越过左边界的时刻补零：
- `l = 0`（左侧补两个 `0`）：通道 0 为 `1*1 + 0*0 + (-1)*0 = 1`，通道 1 为 `1*2 = 2`；
- `l = 1`（左侧补一个 `0`）：通道 0 为 `1*3 + 0*1 + (-1)*0 = 3`，通道 1 为 `1*4 + 1*2 = 6`；
- `l = 2`（窗口完整覆盖 `l = 2, 1, 0`）：通道 0 为 `1*5 + 0*3 + (-1)*1 = 4`，通道 1 为 `1*6 + 1*4 + 1*2 = 12`；
- `l = 3`（窗口向右滑一步覆盖 `l = 3, 2, 1`）：通道 0 为 `1*7 + 0*5 + (-1)*3 = 4`，通道 1 为 `1*8 + 1*6 + 1*4 = 18`，输出 `[[[1, 2], [3, 6], [4, 12], [4, 18]]]`。

> baseline：单输出元素单线程直接卷积

基础版给每个 `(b, l, d)` 输出位置分配一个线程，每次直接从全局显存读取最多 `K` 个历史时刻的输入与对应的卷积权重：

```cuda
__global__ void causal_depthwise_conv1d_baseline_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int L, int D, int K) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(B) * L * D;
    if (idx >= total) return;

    const int d = static_cast<int>(idx % D);
    const size_t bl = idx / D;
    const int l = static_cast<int>(bl % L);
    const int b = static_cast<int>(bl / L);

    const size_t batch_base = static_cast<size_t>(b) * L * D;
    const int k_max = (K - 1 < l) ? (K - 1) : l;
    float acc = bias[d];
    for (int k = 0; k <= k_max; ++k) {
        acc = fmaf(weight[d * K + k], x[batch_base + static_cast<size_t>(l - k) * D + d], acc);
    }
    output[idx] = acc;
}
```

在官方评测规模 `B = 8, L = 2,048, D = 4,096, K = 4` 下，输入和输出张量各占 256 MiB。单元素一线程的写法在每个时刻都要重新去全局显存读 `x[l], x[l-1], x[l-2], x[l-3]` 四行，跨线程块缺乏时间维局部性，导致 256 MiB 的输入被放大成接近 4 倍的重复读取，每个输出位置还要重复读取 `weight`。

> optimized：通道维 `float4` 向量化 + 时间维寄存器滑动窗口

因为最内维正好是互不耦合的通道维 `D`，相邻通道在同一时刻内存连续，而相邻时刻只共享同一个通道自身的历史值，可以直接用寄存器在时间维做滑动窗口：

- **通道维 `float4` 合并读写与权重常驻寄存器**：每个线程负责 4 个连续通道（1 个 `float4`），一个 128 线程块覆盖 512 个连续通道；同一 warp（32 线程组成的硬件执行单元）每次读写恰好拼成连续 512 字节的满带宽合并访存。线程在进入时间循环前先把这 4 个通道的 `bias` 和 `4 * K` 个 `weight` 一次性读入寄存器（当 `K = 4` 时通过 4 次 `float4` 向量读取并在寄存器内完成 `4x4` 转置），整段序列推进期间零重复加载权重。
- **时间分片流式推进（零 shared memory、零同步）**：把序列维 `L` 切成长度为 `kLTile = 128` 的时间片。每个线程先预热分片左边界外的 `K - 1 = 3` 个历史 `float4`（越界位置填 `0`），随后沿时间维从前往后顺序扫过 128 步：每一步只发**一次** `float4` 全局读取拿到 `x_cur`，和寄存器里的 `hist[0..K-2]` 做乘加后立即发出**一次** `float4` 全局写回，并把窗口寄存器整体顺移一格。整个过程完全停留在单线程私有寄存器里，省掉块内共享内存（shared memory）中转与 `__syncthreads()` 同步，边界历史重读开销只剩 `3 / 128 = 2.3%`。

```cuda
// 预热分片左边界的 K - 1 个历史时刻（越过 l = 0 的位置填 0）
#pragma unroll
for (int h = 0; h < K - 1; ++h) {
    const int hl = l_start - 1 - h;
    hist[h] = (hl >= 0) ? __ldg(&x4[batch_offset4 + static_cast<size_t>(hl) * D4]) : zero_float4();
}

// 流式推进 kLTile 步：每步仅 1 次 float4 读取 + 1 次 float4 写回
#pragma unroll 8
for (int s = 0; s < kLTile; ++s) {
    const float4 x_cur = __ldg(&x4[ptr_idx]);
    float4 acc = fma_float4(w_tap[0], x_cur, b_vec);
    #pragma unroll
    for (int k = 1; k < K; ++k) acc = fma_float4(w_tap[k], hist[k - 1], acc);
    #pragma unroll
    for (int h = K - 2; h > 0; --h) hist[h] = hist[h - 1];
    hist[0] = x_cur;
    out4[ptr_idx] = acc;
    ptr_idx += D4;
}
```

2.44 ms 94.8th percentile (T4)

这一步省去了每个时刻对前 `K - 1` 行输入的重复全局显存读取以及逐位置重读权重的开销，让 `K = 4` 时的全局访存量逼近一读一写的 `512 MiB` 理论下限。在本地 Tesla T4（CUDA 12.4）上实测官方计时规模 `B = 8, L = 2,048, D = 4,096, K = 4`（每次刷空 L2，15 轮中位数）：单元素基线耗时 `2.833 ms`，优化版降至 **`1.940 ms`**（等效单遍有效带宽达 **`276.8 GB/s`**），两版均通过 31 组多尺寸、任意核宽 `K`、`K > L` 左端零填充与非对齐地址测试（最大绝对误差 `2.60e-6`）。LeetGPU 官方 Tesla T4 首次提交即通过全部测试，耗时 **`2.44 ms`（第 `94.8` 百分位）**。

完整代码：[单元素直接卷积基线](../../solutions/convolution/causal-depthwise-conv1d-baseline.cu) · [通道向量化寄存器滑窗优化版](../../solutions/convolution/causal-depthwise-conv1d-opt.cu)
