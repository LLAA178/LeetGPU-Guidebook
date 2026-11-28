### [rms-normalization](https://leetgpu.com/challenges/rms-normalization)
*环境: CUDA*

对长度为 `N` 的一维单精度浮点数向量计算均方根归一化（RMS Normalization）：先求均方根 `rms = sqrt(mean(x^2) + eps)`，再按标量参数 `gamma` 与 `beta` 计算 `output[i] = gamma * (input[i] / rms) + beta`。

以官方示例 1（`input = [1.0, 2.0, 3.0, 4.0], N = 4, gamma = 1.0, beta = 0.0, eps = 1e-5`）为例：
1. **计算均方与均方根倒数**：各元素平方和为 `1 + 4 + 9 + 16 = 30`，均方值为 `30 / 4 = 7.5`，加上 `1e-5` 后求平方根倒数得到 `1 / sqrt(7.50001) ≈ 0.36514813`。
2. **缩放平移写出**：由于 `gamma = 1.0, beta = 0.0`，各输入乘上 `0.36514813` 后得到输出 `[0.36514813, 0.73029625, 1.0954444, 1.4605925]`。

> baseline：两级分块平方和归约 + 标量写出

基线版分三次核函数启动执行：第一遍各线程块用标量循环累加局部平方和写入临时数组，第二遍单块汇总全部局部和并算出融合缩放因子 `g_inv_rms_scale = gamma * rsqrtf(mean_sq + eps)`，第三遍重读 `input` 执行乘加写出：

```cuda
const float scale = g_inv_rms_scale;
for (int i = blockIdx.x * kThreads + threadIdx.x; i < n; i += gridDim.x * kThreads) {
    output[i] = fmaf(input[i], scale, beta);
}
```

在性能评测规模 `N = 100,000`（输入与输出各 `400 KB`）下，标量逐元素读取与双精度内层乘法指令密度高，且三次串行核函数启动在百微秒级算子中占据了可观的固定调度开销。

> optimized：小规模单块一次启动 + 8 路 `float4` 单精度累加与二级缓存流式写出

优化版按输入规模自适应分流，把平方累加留在高吞吐单精度融合乘加单元上，并将核函数启动次数压到最少：

- **中小规模单块片上一次启动（`N <= 8192`）**：单个线程块用 `float4`（单条指令读取 16 字节、4 个浮点数）把整条向量一次性读入 `32 KB` 片上共享内存（shared memory，块内高速缓存），块内归约出融合缩放系数 `scale = gamma * rsqrtf(mean_sq + eps)` 后直接从共享内存取出数据并以流式向量写出，整道题仅需 1 次核函数启动且全局显存只发生 1 读 1 写。
- **8 路展开 `float4` 累加与尾块原子收尾（`N > 8192` 第 1 趟）**：先剥离首部 `0..3` 个非对齐元素使主体对齐到 16 字节边界，并按输入规模精确匹配常驻网格块数（如 `N = 100,000` 时启动 40 个块，恰好对应 40 个流多处理器每核一块，消除多余空转线程与原子竞争）。内层循环做 8 路 `float4` 展开（每步处理 32 个浮点数），用 4 个独立的单精度累加器并行发射 `fmaf`，每满 32 个元素才向双精度总和归并一次，兼顾单精度峰值吞吐与大数组求和精度；最后由 `atomicAdd` 选出的收尾块直接算出全局 `g_scale`，省去独立的二次归约核函数。
- **二级缓存命中与缓存流式提示写出（第 2 趟）**：`400 KB` 的输入经第 1 趟加载后驻留在二级缓存（L2 cache）中；第 2 趟以 4 路展开 `float4` 高速重读输入，执行单条 `fmaf(x, scale, beta)` 后通过带缓存流式提示（cache-streaming hint）的 16 字节向量存储指令 `st.global.cs.v4.f32` 写回 `output`。

```cuda
float s0 = fmaf(v0.x, v0.x, v0.y * v0.y) + fmaf(v0.z, v0.z, v0.w * v0.w);
float s1 = fmaf(v1.x, v1.x, v1.y * v1.y) + fmaf(v1.z, v1.z, v1.w * v1.w);
float s2 = fmaf(v2.x, v2.x, v2.y * v2.y) + fmaf(v2.z, v2.z, v2.w * v2.w);
float s3 = fmaf(v3.x, v3.x, v3.y * v3.y) + fmaf(v3.z, v3.z, v3.w * v3.w);
```

在官方性能测试规模 `N = 100,000` 下（每次计时前刷空 L2 缓存，取 15 轮中位数），两级标量基线版耗时 `0.0199 ms`，向量化尾块收尾优化版降至 **`0.0124 ms`**（提速 **`1.60x`**）；在 LeetGPU 官方 T4 评测中一次通过全部测试，耗时 **0.03 ms**（第 **98.5** 百分位）。

完整代码：[两级归约标量基线](../../solutions/normalization/rms-normalization-baseline.cu) · [8 路 float4 与尾块收尾优化版](../../solutions/normalization/rms-normalization-opt.cu)
