---
name: latency-optimizer
description: >
  Triton Ascend 性能优化——分析代码特征，应用优化策略，确保优化前后功能和精度一致。
triggers:
  - optimize
  - performance
  - latency
  - speedup
---

# Latency Optimizer

## 优化策略（按优先级）

1. **入参静态化**：将适合的参数声明为 `tl.constexpr`，减少运行时开销。
2. **Load 指令重排序**：将无依赖的 `tl.load` 集中到前面，增加流水线并行度。
3. **Int32 向量加法**：用 int32 替代 int64 索引计算，减少指令开销。
4. **向量比较优化**：涉及数值比较时，选择合适的比较指令。
5. **Kernel 融合**：多个独立 kernel 合并减少 launch 开销和内存带宽。
6. **Tiling 调优**：BLOCK_SIZE 选择需平衡并行度与寄存器/UB 容量。

## 使用方式

1. 读取当前通过正确性的 `{op_name}_triton_ascend_impl.py`。
2. 分析代码特征，选择适用的优化策略。
3. 生成优化版代码，保存并重跑 `bash tools/operator_pipeline.sh --op_name <op_name>`。
4. 对比 baseline 和优化版的 `speedup_vs_torch`。

## 约束

- 优化前后功能一致、精度一致。
- 不要回退到比 baseline 更慢的版本。
- Phase 4 最多迭代 3 轮。
