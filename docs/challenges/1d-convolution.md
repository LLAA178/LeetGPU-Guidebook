### [1d-convolution](https://leetgpu.com/challenges/1d-convolution)
*环境: CUDA*

给定长度为 `input_size` 的一维输入信号 `input` 与长度为 `kernel_size` 的卷积核 `kernel`，计算无填充的有效一维互相关（valid 1D cross-correlation）：仅在卷积核与输入完全重合的窗口上逐项相乘求和，生成长度为 `output_size = input_size - kernel_size + 1` 的输出数组 `output`。

用官方示例 1 走一遍窗口滑动过程：输入 `input = [1, 2, 3, 4, 5]`（`input_size = 5`），卷积核 `kernel = [1, 0, -1]`（`kernel_size = 3`），有效输出长度为 `5 - 3 + 1 = 3`：
- `output[0] = 1 × 1 + 2 × 0 + 3 × (-1) = -2`
- `output[1] = 2 × 1 + 3 × 0 + 4 × (-1) = -2`
- `output[2] = 3 × 1 + 4 × 0 + 5 × (-1) = -2`

最终输出为 `[-2, -2, -2]`。

> baseline

基线版让每个线程独立负责一个输出位置 `i`，直接从全局显存顺序读取 `input[i + j]` 与 `kernel[j]` 完成 `kernel_size` 次融合乘加（`fmaf`）：

```cuda
__global__ __launch_bounds__(256) void conv1d_valid_baseline_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int output_size,
    int kernel_size) {
    const int i = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
    if (i >= output_size) return;

    float sum = 0.0f;
    const float* in_ptr = input + i;
    for (int j = 0; j < kernel_size; ++j) {
        sum = fmaf(in_ptr[j], kernel[j], sum);
    }
    output[i] = sum;
}
```

> optimized：片上双缓冲 `LDS.128` 零冲突滑动窗口 + 16 路寄存器累加

在官方性能评测规模 `input_size = 1,500,000, kernel_size = 2,047`（有效输出 `1,497,954`）下，总浮点运算量高达 `2 × 1,497,954 × 2,047 ≈ 6.13 GFLOP`，而输入与输出总数据量仅有 `12 MB`，算术强度超过 `510 FLOP/Byte`。基线版每算一个输出都要穿透高速缓存发起 `4,094` 次标量访存指令，片上缓存端口与指令发射吞吐远早于浮点单元饱和。

优化版用单次核函数启动将算力推向硬件单精度峰值：
- **零主机拷贝开销的片上卷积核与光环加载**：省去主机端 `cudaMemcpyToSymbol` 驱动同步开销，每个 `256` 线程块在入口处利用只读缓存宽指令 `__ldg`（`float4`）协作把卷积核装入片上共享内存 `s_k4[512]`（`8.0 KiB`），并向上补零对齐到 4 的倍数（`2047 -> 2048`），同时把当前块所需的 `4,096 + 2,048 = 6,144` 个输入与光环边界（halo）元素装入 `s_in4[1536]`（`24.0 KiB`，单块静态共享内存合计 `32.0 KiB`），消除内层循环的全部尾部判断分支。
- **跨线程束步长对齐的无冲突 `LDS.128` 滑动窗口**：每个线程负责 `4` 组间隔 `1,024` 的 `float4` 输出（共 `16` 个输出累加器）。在内层每推进 4 个卷积核抽头时，全线程束广播读取 1 个 `s_k4[kv_idx]`，再由相邻线程按步长 1 连续读取 `s_in4[g * 256 + tid + kv_idx + 1]`，天然错开全部 32 个共享内存存储体（bank），用 1 次无冲突 `LDS.128` 片上宽读取喂饱 `16` 条单精度融合乘加指令（`64` 条 `fmaf` 仅需 `5` 条片上读取）：

```cuda
// 每步广播 4 个卷积核权重，对每组 4 路输出仅补读 1 个无存储体冲突的 float4 输入
const float4 kv = s_k4[kv_idx];
#pragma unroll
for (int g = 0; g < kGroups; ++g) {
    const float4 v1 = s_in4[g * kBlockThreads + tid + kv_idx + 1];
    acc[g][0] = fmaf(v0[g].x, kv.x, fmaf(v0[g].y, kv.y, fmaf(v0[g].z, kv.z, fmaf(v0[g].w, kv.w, acc[g][0]))));
    acc[g][1] = fmaf(v0[g].y, kv.x, fmaf(v0[g].z, kv.y, fmaf(v0[g].w, kv.z, fmaf(v1.x,    kv.w, acc[g][1]))));
    acc[g][2] = fmaf(v0[g].z, kv.x, fmaf(v0[g].w, kv.y, fmaf(v1.x,    kv.z, fmaf(v1.y,    kv.w, acc[g][2]))));
    acc[g][3] = fmaf(v0[g].w, kv.x, fmaf(v1.x,    kv.y, fmaf(v1.y,    kv.z, fmaf(v1.z,    kv.w, acc[g][3]))));
    v0[g] = v1;
}
```

- **直连流式提示向量写回与自适应分档**：由于每个线程的 `acc[g][0..3]` 本身就对应连续 4 个输出元素且线程间步长恰为 `16` 字节，无需二次共享内存中转或块同步，直接通过带缓存流式提示（cache-streaming hint）的 16 字节向量存储指令 `st.global.cs.v4.f32` 合并写回全局显存；小尺寸输入自动切至单组 `1,024` 输出块以铺满全部计算单元，超大卷积核（`> 2,048`）则通过分段共享内存循环兜底。

在官方评测规模 `input_size = 1,500,000, kernel_size = 2,047` 下（每次计时前刷空 L2 缓存，取 15 轮中位数）：标量直读基线耗时 `7.40 ms`，`LDS.128` 寄存器滑窗优化版降至 **`1.11 ms`**（提速 **`6.68x`**，有效单精度计算吞吐达 **`5.54 TFLOPS`**）；在 LeetGPU 官方 Tesla T4 评测中取得 **`0.97 ms`（第 `99.3` 百分位）**。

完整代码：[标量直读基线](../../solutions/convolution/1d-convolution-baseline.cu) · [`LDS.128` 零冲突滑窗优化](../../solutions/convolution/1d-convolution-opt.cu)
