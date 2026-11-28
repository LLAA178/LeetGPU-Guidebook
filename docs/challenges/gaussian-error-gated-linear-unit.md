### [gaussian-error-gated-linear-unit](https://leetgpu.com/challenges/gaussian-error-gated-linear-unit)
*环境: CUDA*

对长度为偶数 `N` 的一维单精度浮点数向量计算高斯误差门控线性单元（GEGLU）前向传播：将输入对半切分为前半段 `x_1 = input[0 .. N/2 - 1]` 与后半段 `x_2 = input[N/2 .. N - 1]`，对后半段计算精确高斯误差激活 `GELU(x_2) = 0.5 * x_2 * (1 + erf(x_2 / sqrt(2)))`，再与前半段 `x_1` 逐元素相乘写入长度为 `N / 2` 的 `output`。

以官方示例 2（`input = [2.0, -1.0, 1.0, 0.5], N = 4`）为例：
1. **前后对半切分**：前半段线性分量 `x_1 = [2.0, -1.0]`，后半段门控分量 `x_2 = [1.0, 0.5]`（注意激活函数作用在后半段 `x_2` 上）。
2. **计算 GELU 并相乘**：`GELU(1.0) = 0.5 * 1.0 * (1 + erf(1 / √2)) ≈ 0.84134475`，乘以 `x_1[0] = 2.0` 得 `1.6826895`；`GELU(0.5) = 0.5 * 0.5 * (1 + erf(0.5 / √2)) ≈ 0.3457312`，乘以 `x_1[1] = -1.0` 得 `-0.3457312`，最终输出 `[1.6826895, -0.3457312]`。

> baseline：单线程单输出标量计算

基线版按输出长度 `half_n = N / 2` 启动线程，每个线程读取一对标量 `input[idx]` 与 `input[idx + half_n]`，调用单精度误差函数 `erff` 算完后写回 `output[idx]`：

```cuda
const float x1 = input[idx];
const float x2 = input[idx + half_n];
const float gelu = 0.5f * x2 * (1.0f + erff(x2 * kInvSqrt2));
output[idx] = x1 * gelu;
```

在性能评测规模 `N = 1,000,000`（读入 `4 MB`、写出 `2 MB`，总计 `6 MB` 显存流量）下，逐元素 4 字节读取未能充分利用 16 字节宽加载吞吐，同时单线程只处理单个元素难以隐藏 `erff` 多指令多项式求值延迟。

> optimized：双路 `float4` 向量化并发求值 + 缓存流式提示写出

优化版结合显存合并读取、指令级并行与写回缓存策略将吞吐推向硬件带宽上限：

- **双路 `float4` 合并预取与 `erff` 延迟隐藏**：当半长 `half_n` 为 4 的倍数且地址满足 16 字节对齐（包括评测规模 `half_n = 500,000`）时，每个线程在单次循环内并发加载前半段与后半段各两个 `float4`（共 8 对元素），将 8 次独立的 `erff` 计算与显存预取交错流水执行，同时预乘标量系数 `0.5f * x1 * x2` 省去多余乘法；非 4 倍数半长（如 `N = 42`）或非对齐指针自动走 4 路展开的合并标量路径。
- **整波次网格配置**：网格上限控制在 `240` 个线程块，消除尾部不均波次与多余块启动开销。
- **带缓存流式提示的向量写出**：对只写的 `2 MB` 输出张量通过带缓存流式提示（cache-streaming hint）的 16 字节向量存储指令 `st.global.cs.v4.f32` 写回显存，降低输出写入对缓存的污染。

```cuda
__device__ __forceinline__ float eval_geglu(float x1, float x2) {
    return (0.5f * x1 * x2) * (1.0f + erff(x2 * kInvSqrt2));
}
```

在官方性能测试规模 `N = 1,000,000` 下（每次计时前刷空 L2 缓存，取 15 轮中位数），标量基线版耗时 `0.0254 ms`，双路 `float4` 流式优化版降至 **`0.0246 ms`**（等效读写吞吐 **`244.1 GB/s`**）；在 LeetGPU 官方 Tesla T4 评测中取得 **`0.04 ms`（第 `96.5` 百分位）**。

完整代码：[单元素标量基线](../../solutions/elementwise/geglu-baseline.cu) · [双路 float4 流式优化版](../../solutions/elementwise/geglu-opt.cu)
