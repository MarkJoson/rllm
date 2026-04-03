---
name: kernel-generator
description: >
  Triton Ascend 算子代码生成——根据任务和可选 sketch 生成 @triton.jit kernel + ModelNew。
  严格禁止 PyTorch 退化。
triggers:
  - triton
  - kernel
  - generate
  - implement
---

# Triton Ascend 代码生成

## 核心约束：禁止 PyTorch 退化

**forward() 中所有核心计算必须在 `@triton.jit` kernel 中实现。**

### 禁止

| 操作 | 示例 |
|------|------|
| torch 计算函数 | `torch.matmul`, `torch.relu`, `torch.sum` |
| torch.nn.functional | `F.softmax`, `F.linear` |
| tensor 方法计算 | `x.sum()`, `x.mean()`, `x @ w` |
| nn.Module 调用 | `self.conv(x)`, `self.linear(x)` |

### 允许

| 操作 | 示例 |
|------|------|
| buffer 分配 | `torch.empty`, `torch.zeros` |
| 形状操作 | `.view`, `.reshape`, `.permute`, `.contiguous` |
| 元信息 | `.shape`, `.dtype`, `.numel()` |
| kernel 启动 | `kernel[grid](...)` |

## 输出格式

单个 Python 文件，包含：

```python
import torch, torch.nn as nn, triton, triton.language as tl

@triton.jit
def {op_name}_kernel(...):
    ...

class ModelNew(nn.Module):
    def __init__(self, <与原 Model 相同>):
        super().__init__()
        ...
    def forward(self, <与原 Model 相同>):
        ...  # 只调用 kernel，不使用 torch 计算
        return output
```

## 关键约束

- 类名必须为 `ModelNew`。
- `__init__` 和 `forward` 签名与原 `Model` 一致。
- 含随机权重的算子（Conv2d/Linear）：`__init__` 第一行 `torch.manual_seed(0)`，按原顺序创建参数。
- 所有 kernel 和辅助函数定义在同一文件内，可直接导入运行。

## 迭代修复模式

收到 `verifier_error` + `conductor_suggestion` 时：
1. 分析错误，理解具体原因。
2. 按 `conductor_suggestion` 方向修改，不做不必要的重构。
3. 保留正确部分，只改有问题的地方。
