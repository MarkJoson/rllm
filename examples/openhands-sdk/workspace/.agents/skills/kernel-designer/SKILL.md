---
name: kernel-designer
description: >
  Triton Ascend 算法草图设计——根据任务描述设计算法思路（sketch），
  用于指导后续代码生成。不生成可执行代码。
triggers:
  - sketch
  - design
  - algorithm
---

# Triton Ascend 算法草图设计

仅生成算法草图，不生成代码。草图用于指导 kernel-generator。

## 设计模式

1. 阅读 `INSTRUCTIONS.md` 中 `Model.forward()` 的参考实现。
2. 判断算子类型（elementwise / reduce / matmul / attention / 复合）。
3. 根据目标硬件架构选择并行化策略和内存访问模式。
4. 以 `sketch op_name { ... }` 格式写出算法草图，保存到 `src/sketch.txt`。

## 原则

- 高层抽象：关注算法逻辑和优化策略，不写实现细节。
- 标注优化点（并行度、内存层次、tiling 策略）。
- 数值正确性优先，性能次之。
