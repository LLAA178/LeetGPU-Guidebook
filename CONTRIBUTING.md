# LeetGPU-Guidebook 维护规范

## 仓库结构

当前仓库建议保持轻量，题解正文保留在根 `README.md`，较长的专题学习笔记放到 `docs/`，完整代码放到 `solutions/`：

```text
LeetGPU-Guidebook/
|-- README.md
|-- CONTRIBUTING.md
|-- docs/
|   `-- cuda-best-practices/
`-- solutions/
    |-- normalization/
    |-- quantization/
    |-- reduction/
    `-- transformer/
```

推荐继续沿用这个形式：

- `README.md` 只放总说明、索引、题解正文，以及必要的核心代码片段
- `docs/` 放专题学习笔记、实验计划、长文档
- `CONTRIBUTING.md` 放仓库结构、模板、收录规范
- `solutions/` 放 README 里不适合完整展开的题目代码
- 同一题的多个版本放在同一节内，避免散落
- 新题默认追加到对应难度末尾，特殊大题放全文末尾也可以

## 条目模板

每题尽量统一成下面这个形状：

````md
### [problem-name](https://leetgpu.com/challenges/problem-name)
*环境: CUDA*

一句话：先说这版解法的主思路。

> scalar
```CUDA
extern "C" void solve(...) {
  ...
}
```

> 优化点
- 为什么这样分块
- 关键访存路径
- 为什么会更快

> dp4a / tensor core
```CUDA
extern "C" void solve(...) {
  ...
}
```
````

建议约束：

- **教程是给人看的**：用平实、直接的中文写，先用一句话说清楚题目要算什么，再用一个小例子讲清关键步骤，最后直接说明哪一步省了时间、实测快多少。
- **术语可读性**：专业术语（如 warp、shared memory、bank conflict、Tensor Core 等）首次出现时顺手补一句中文解释；避免“不是……而是……”、反复转折和重复报数。
- 标题统一用 `### [题目名](链接)`
- 先放最朴素版本，再放优化版本
- 版本小节统一写成 `> baseline`、`> optimized`、`> tensor core`，不要用 `#### baseline`
- 代码只保留和这一版直接相关的部分
- 不把整套公共代码在多个版本里重复贴
- 代码较长时，README 只保留核心片段，并在题末补 `完整代码` 链接到 `solutions/`

## 代码模板

最小 judge 模板：

```CUDA
#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;

__global__ void kernel(const float* in, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx];
  }
}

}  // namespace

extern "C" void solve(const float* in, float* out, int n) {
  int blocks = (n + kThreads - 1) / kThreads;
  kernel<<<blocks, kThreads>>>(in, out, n);
}
```

题解里的代码建议遵守：

- 只保留 `solve` 和必要 kernel
- 公共常量放匿名命名空间
- 能看懂就不额外加注释
- 优先贴核心 kernel，不贴测试 harness
- 如果是教学型大题，只贴核心路径代码，不强求可直接复制运行

## 收录规范

为了后续把所有题补齐，建议固定成下面的范式：

- 每题至少保留一个可讲清楚的版本
- 有明显性能台阶的题，再补第二版或第三版
- `scalar / 向量化 / shared memory / warp / dp4a / tensor core` 只保留真正有信息增量的版本
- 同一个优化如果只是常数调参，不单开小节
- 默认不贴跑分数字，避免 README 变成榜单快照
- 默认不贴过长工具代码，避免一题把页面撑开
- 当完整实现明显长于正文时，把完整代码放进 `solutions/<type>/`
- 当某题需要很多上下文时，优先压缩文字，再压缩代码
- 当 README 明显变长时，先补索引，不急着拆文件
- 只有当单文件检索成本已经明显高于维护收益时，再考虑拆分目录

## 新题添加流程

```text
1. 在索引补一个锚点
2. 在对应难度末尾新增题目小节
3. 先写最小可讲清的版本
4. 再决定是否补优化版本
5. 最后检查是否有重复代码和重复解释
```
