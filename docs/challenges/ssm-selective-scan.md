### [ssm-selective-scan](https://leetgpu.com/challenges/ssm-selective-scan)
*环境: CUDA*

对形状为 `[batch, seq_len, d_model]` 的输入序列 `u` 和步长 `delta`，沿着时间步 `t = 0 .. seq_len - 1` 维护每个通道的 `d_state` 维隐状态 `h`（初值为 `0`），按 `h[n] = exp(delta * A[d, n]) * h[n] + delta * B[b, t, n] * u` 逐步更新，并把状态加权和 `sum_n C[b, t, n] * h[n]` 加上跳跃连接 `skip[d] * u` 写入输出 `y`。

用 `d_state = 2` 的单通道小例子看两步递推：设 `A = [-1, -2]`，`skip = 0.5`，初始隐状态 `h = [0, 0]`。
1. **`t = 0`（注入初始输入）**：读入 `u_0 = 1.0, delta_0 = 0.1` 与 `B_0 = [1.0, 0.5], C_0 = [1.0, 1.0]`。旧状态为 `0`，新状态直接装入缩放后的输入 `h = (0.1 * 1.0) * [1.0, 0.5] = [0.10, 0.05]`，输出 `y_0 = (1.0 * 0.10 + 1.0 * 0.05) + 0.5 * 1.0 = 0.65`。
2. **`t = 1`（指数衰减叠加新输入）**：读入 `u_1 = 0.5, delta_1 = 0.1` 与 `B_1 = [0.5, 1.0], C_1 = [0.5, -0.5]`。旧状态先乘衰减因子 `[exp(-0.1), exp(-0.2)] ≈ [0.9048, 0.8187]`，再加上 `(0.1 * 0.5) * [0.5, 1.0]` 得到 `h ≈ [0.1155, 0.0909]`，随即点乘 `C_1` 并加上 `0.5 * 0.5` 写出 `y_1 ≈ 0.2623`。

> baseline：单线程单通道串行递推

基线版让一个线程负责一条独立的 `(b, d)` 通道序列，在局部数组 `h[64]` 里顺着 `t = 0 .. seq_len - 1` 逐步读全局显存、算 `expf` 并累加写出 `y`：

```cuda
for (int t = 0; t < seq_len; ++t) {
    const float u_val = u[btd];
    const float dt = delta[btd];
    float y_acc = 0.0f;
    for (int n = 0; n < d_state; ++n) {
        h[n] = expf(dt * a_row[n]) * h[n] + (dt * b_row[n]) * u_val;
        y_acc += c_row[n] * h[n];
    }
    y[btd] = y_acc + skip_d * u_val;
}
```

这版在官方计时规模 `batch = 4, seq_len = 4096, d_model = 512, d_state = 16` 下有三处明显浪费：第一，同一批次的 512 个通道在每个时间步都要重复从全局显存读取完全相同的 `B[b, t, :]` 和 `C[b, t, :]`；第二，总共只有 `4 * 512 = 2,048` 条通道，按每块 256 线程只能分出 8 个线程块，40 个流多处理器（SM，GPU 上的核心计算单元组）里有 32 个彻底闲置；第三，内层每步调用 16 次标准库 `expf`，产生大量多指令超越函数开销。

> optimized：全张量 `float4` 共享内存分块预取 + 4 线程子束状态分片 + `exp2f` 单指令指数

看清递推与访存瓶颈后，把重复显存读取、逐步串行访存停顿和 SM 闲置一并消掉：

- **全张量 `float4` 分块预取与合并写回（消除逐步等待显存）**：在 `d_state = 16` 主路径中，每个 64 线程块负责 16 条连续通道，每次沿时间维推进 16 步。进入内层循环前，全块 64 个线程用 `float4` 向量化读取（单条指令搬运 16 字节）一次性把 `[16][16]` 的 `u`、`delta`、`B`、`C` 四个分块并行预取进片上共享内存（shared memory，速度接近寄存器的块内高速缓存）；内层 16 步递推全程只访问寄存器与共享内存，算出的 `y` 先暂存回 `s_u` 共享内存，再在分块末尾用 `float4` 合并写回全局显存。这样既让块内 16 条通道共享同一份 `B/C`，又把原本 4,096 次逐步串行等待显存延迟变成了高并发的整块流水预取。
- **4 线程子束拆分状态维铺满全卡 SM**：状态更新在各个状态下标 `n` 之间彼此独立，只有算输出 `y` 时才做一次跨 `n` 求和。让线程束（warp，GPU 上 32 个锁步执行的线程组）里每 4 个相邻线程组成一个子束，共同负责一条 `(b, d)` 通道：每个线程在寄存器里固定保管 4 个状态（恰好 1 个 `float4`），直接从共享内存广播读取当前步的 `u` 和 `delta`，更新完各自 4 个状态后再用 2 条异或洗牌指令 `__shfl_xor_sync`（同一线程束内直接交换寄存器值）归约出完整点积写入共享内存。全网活跃规模扩充到 128 个线程块（8,192 个线程），40 个 SM 全部满载运行；任意非 16 维或奇数通道尺寸则走通用子束掩码路径，覆盖全部 `1 <= d_state <= 64` 场景且零寄存器溢出。
- **预乘换底常数配单指令 `exp2f` 与乘加融合**：利用恒等式 `exp(dt * A) = 2^(dt * (A * log2(e)))`，进入时间循环前先把 `A[d, n] * log2(e)` 算好常驻寄存器，内层循环直接调用特殊函数单元（SFU，GPU 上专门执行超越函数的硬件流水线）的单指令 `exp2f`，并先算好标量 `du = dt * u_val` 配合单条乘加指令 `fmaf` 推进状态。

```cuda
const float u_val = s_u[s][d_local];
const float dt = s_dt[s][d_local];
const float du = dt * u_val;
const float4 bv = reinterpret_cast<const float4*>(&s_B[s][0])[sub];
const float4 cv = reinterpret_cast<const float4*>(&s_C[s][0])[sub];
h[0] = fmaf(exp2f(dt * a_log2e[0]), h[0], du * bv.x);
h[1] = fmaf(exp2f(dt * a_log2e[1]), h[1], du * bv.y);
h[2] = fmaf(exp2f(dt * a_log2e[2]), h[2], du * bv.z);
h[3] = fmaf(exp2f(dt * a_log2e[3]), h[3], du * bv.w);
float y_part = fmaf(cv.x, h[0], fmaf(cv.y, h[1], fmaf(cv.z, h[2], cv.w * h[3])));
y_part += __shfl_xor_sync(0xffffffffu, y_part, 2);
y_part += __shfl_xor_sync(0xffffffffu, y_part, 1);
if (sub == 0) s_u[s][d_local] = fmaf(skip_d, u_val, y_part);
```

2.29 ms 96.4th percentile (T4)

省掉 `B/C` 跨通道重复读取、逐步串行访存停顿以及多指令 `expf` 开销后，在本地 Tesla T4（CUDA 12.4）上实测官方计时规模 `batch = 4, seq_len = 4096, d_model = 512, d_state = 16`（每次刷空 L2，15 轮中位数）：单通道串行基线耗时 `8.9549 ms`，优化版降至 **`0.8719 ms`**（提速 **10.27 倍**），两版均通过 35 组多尺寸、奇数维度与非对齐边界测试（最大绝对误差 `5.77e-7`）。LeetGPU 官方 Tesla T4 首次提交即通过全部测试，耗时 **`2.29 ms`（第 `96.4` 百分位）**。

完整代码：[单通道串行基线](../../solutions/reduction/ssm-selective-scan-baseline.cu) · [共享内存预取与子束状态分片](../../solutions/reduction/ssm-selective-scan-opt.cu)
