### [distillation-kl-divergence](https://leetgpu.com/challenges/distillation-kl-divergence)
*环境: CUDA*

给定形状同为 `[N, V]` 的教师模型输出 `teacher_logits`、学生模型输出 `student_logits` 和温度系数 `temperature`，对每一行先除以温度得到 `a = teacher / T` 与 `b = student / T` 并各自归一化为概率分布 `p = softmax(a)`、`q = softmax(b)`，再把前向 KL 散度 `kl[i] = sum_{j=0}^{V-1} p_j * (log p_j - log q_j)` 写入输出向量。

拿官方样例 2（`N = 2, V = 3, temperature = 2.0`）演示：第 0 行 `teacher = [1, 2, 3]`、`student = [3, 2, 1]`，除以温度 `2.0` 后 `a = [0.5, 1.0, 1.5]`、`b = [1.5, 1.0, 0.5]`。教师概率 `p = softmax([0.5, 1.0, 1.5]) = [0.18632, 0.30720, 0.50648]`，由于 `b` 与 `a` 元素集合相同，两者的对数配分函数相等，逐位置对数概率差 `log p - log q = a - b = [-1.0, 0.0, 1.0]`，加权求和得 `kl[0] = 0.18632 * (-1) + 0.30720 * 0 + 0.50648 * 1 = 0.3202`；第 1 行全为 `0`，两分布完全一致，输出 `kl[1] = 0.0000`。

> baseline：单行三遍分步扫描

直接按定义计算时，因为单行词表长度可达 `V = 128,256`（两路输入合计约 `0.98 MiB`，远超单块 `48 KiB` 共享内存上限），基础版给每行分配一个线程块分三遍扫过全局显存：第一遍分别求 `m_a = max(a)` 和 `m_b = max(b)`，第二遍求平移后的指数和 `d_a`、`d_b`，第三遍累加 `p_j * (log p_j - log q_j)`：

```cuda
// 第 1 遍求各自最大值 m_a, m_b；第 2 遍求指数和 d_a, d_b；第 3 遍累加 KL 散度
float kl_sum = 0.0f;
for (int j = tid; j < V; j += blockDim.x) {
    const float u = t_row[j] * inv_temp - m_a;
    const float v = s_row[j] * inv_temp - m_b;
    kl_sum += (expf(u) * inv_d_a) * ((u - log_d_a) - (v - log_d_b));
}
```

在官方评测规模 `N = 2,048, V = 128,256` 下，教师与学生矩阵各占 `1.05 GB`。三遍扫描意味着整整 `6.30 GB` 的全局显存读取，耗时完全卡在重复搬运同一批词表数据上。

> optimized：五元组在线同步重标定 + `float4` 真正单遍归约

把 `log p_j - log q_j` 改写成各自减去最大值后的平移变量 `u_j = a_j - m_a` 与 `v_j = b_j - m_b`，有恒等式 `kl = (1 / d_a) * sum_j [exp(u_j) * (u_j - v_j)] + log(d_b / d_a)`。借助这条恒等式，只需维护五元组状态 `(m_a, d_a, m_b, d_b, w)`（其中 `w = sum exp(a_j - m_a) * ((a_j - m_a) - (b_j - m_b))`），就能在一遍扫描里同时完成两路归一化与加权交叉项统计：

- **五元组可结合合并律（防溢出且平移不变）**：当两段区间状态合并到新最大值 `M_a = max(m_a1, m_a2)`、`M_b = max(m_b1, m_b2)` 时，除了把 `d_a`、`d_b` 分别乘上缩放因子 `alpha = exp(m_a - M_a)`、`beta = exp(m_b - M_b)` 外，历史加权项同步按 `alpha * (w + d_a * ((m_a - M_a) - (m_b - M_b)))` 重标定。教师和学生各自拥有独立最大值，遇到一边在 `-300`、另一边在 `+500` 的分布照样稳定无溢出；当学生相对教师整体平移任意常数 `C` 时，`(m_a - M_a) - (m_b - M_b)` 精确归零，直接给出 `0.0`。
- **`float4` 四合一在线推进与双精度跨束归约**：每个线程用 `float4` 一次读入 4 个教师值和 4 个学生值，先求出当前 4 元组的局部最大并与运行最大取并，每 4 个词表项只做**一次**历史累加器重标定；循环结束后升为双精度五元组，用寄存器洗牌指令 `__shfl_down_sync`（同一线程束 32 个线程间直接交换寄存器）做树形合并。对于行数很少但词表极宽的形状（如 `4 x 128,256`），自动将单行拆成多个线程块并行并在末块合并，单次启动即可铺满全卡计算单元。

```cuda
// 两段在线五元组 (ma, da, mb, db, w) 的精确合并公式
const double ma = fmax(s1.ma, s2.ma), mb = fmax(s1.mb, s2.mb);
const double a1 = exp(s1.ma - ma),    a2 = exp(s2.ma - ma);
const double b1 = exp(s1.mb - mb),    b2 = exp(s2.mb - mb);
const double sh1 = (s1.ma - ma) - (s1.mb - mb);
const double sh2 = (s2.ma - ma) - (s2.mb - mb);
out.da = s1.da * a1 + s2.da * a2;
out.db = s1.db * b1 + s2.db * b2;
out.w  = a1 * (s1.w + s1.da * sh1) + a2 * (s2.w + s2.da * sh2);
```

这一步把全局显存读取从三遍（`6.30 GB`）直接砍成一遍（`2.10 GB`，省去三分之二的显存流量），同时无需任何临时显存分配。优化版在 LeetGPU 官方 T4 上一次通过全部测试，耗时 **10.24 ms**（第 **90.0** 百分位）；在官方评测规模 `N = 2,048, V = 128,256, temperature = 2.0`（每次计时前刷空二级缓存，取 15 轮中位数）下，本地实测耗时从三遍扫描基线的 `20.145 ms` 降至 **`6.928 ms`**（提速 **`2.91x`**，单遍等效带宽达 **`303.3 GB/s`**）。

完整代码：[三遍扫描基线](../../solutions/reduction/distillation-kl-divergence-baseline.cu) · [单遍五元组在线归约优化版](../../solutions/reduction/distillation-kl-divergence-opt.cu)
