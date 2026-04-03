# Triton-Ascend 算子生成 Agent — 全局约定

> 容器内挂载路径一般为 `/opt/workspace`。

## 固定配置

- **framework**: torch
- **dsl**: triton_ascend
- **backend**: ascend

## 环境规则

- OpenHands SDK 运行在镜像自带的 **venv** 里；编译 / 验证 / 性能采集运行在 **conda 环境** 里。
- **Agent 禁止手动 `conda activate`**——所有需要 conda 的操作均由 `tools/` 脚本内部处理。
- **禁止修改** `tools/` 目录。

## 工作流（Phase 2 → 4，Phase 0/1 已在 host 侧完成）

```
Phase 2: 算法设计       (kernel-designer skill, 可选)
Phase 3: 代码生成与验证 (kernel-generator + kernel-verifier, 迭代)
Phase 4: 性能优化       (latency-optimizer + kernel-verifier, 迭代)
```

### Phase 2: 算法设计（可选）

根据 `INSTRUCTIONS.md` 中的任务，用 `kernel-designer` skill 的思路设计算法草图 → `src/sketch.txt`。
对于简单算子可跳过。

### Phase 3: 代码生成与验证（核心循环）

1. 阅读 `INSTRUCTIONS.md`（KernelBench 格式：`Model` + `get_inputs` + `get_init_inputs`）。
2. 在 `src/` 下生成实现文件，文件名须为 `{op_name}_triton_ascend_impl.py`，包含 `ModelNew` 类。
3. 运行流水线：
   ```bash
   bash tools/operator_pipeline.sh --op_name <op_name>
   ```
4. 读取 `metrics.json`，根据结果修复并重试。

**迭代上限**：5 轮。

### Phase 4: 性能优化

Phase 3 通过后，继续优化性能。对比 baseline Triton vs 优化版。
**迭代上限**：3 轮。

## 禁止 PyTorch 退化（最重要的约束）

`ModelNew.forward()` 中**所有核心计算**必须在 `@triton.jit` kernel 中实现。

### forward() 中 **禁止** 的操作

- `torch.matmul / torch.relu / torch.sum` 等计算函数
- `F.softmax / F.linear` 等 `torch.nn.functional`
- `x.sum() / x.mean() / x @ w` 等 tensor 方法/运算符
- `self.conv(x) / self.linear(x)` 等 nn.Module 调用

### forward() 中 **允许** 的操作

- buffer 分配：`torch.empty / torch.zeros / torch.ones`
- 形状操作：`.view / .reshape / .permute / .transpose / .contiguous`
- 元信息：`.shape / .dtype / .device / .numel()`
- kernel 启动：`kernel[grid](...)`

## Conductor 错误分类（迭代失败时使用）

| 类型 | 含义 | 决策 |
|------|------|------|
| **A 类** | 代码逻辑/算法错误（含 PyTorch 退化 Type 1/2/3） | 可修复，继续迭代 |
| **B 类** | 环境/基础设施错误（NPU OOM、设备不可用、超时） | 不可修复，终止 |
| **C 类** | 同一 A 类子类型连续 ≥ 3 次 | 终止 |

### PyTorch 退化子类型

| 子类型 | 含义 |
|--------|------|
| Type 1 | 完全无 `@triton.jit` kernel |
| Type 2 | 有 kernel 但 `forward()` 未调用 |
| Type 3 | `forward()` 调用了 kernel 但部分计算仍用 PyTorch |

## 文件布局

```
/opt/workspace/
├── INSTRUCTIONS.md                    # 当前任务（KernelBench 格式）
├── src/
│   ├── {op_name}_triton_ascend_impl.py   # Agent 实现（必须含 ModelNew）
│   └── sketch.txt                        # 可选：算法草图
├── tools/                             # 只读：流水线与验证脚本
│   ├── operator_pipeline.sh
│   ├── env.sh
│   └── scripts/
│       ├── validate_triton_impl.py    # AST 退化检查（无需 NPU）
│       ├── verify.py                  # 数值正确性（NPU）
│       └── benchmark.py              # 性能采集（torch_npu.profiler）
├── metrics.json                       # 流水线产出（机读指标）
└── profiling_results.json             # 兼容训练侧 reward
```

## metrics.json schema

```json
{
  "schema_version": 2,
  "op_name": "softmax",
  "success": true,
  "ast_check_ok": true,
  "correctness_ok": true,
  "perf_data": {
    "framework_latency_ms": 1.23,
    "impl_latency_ms": 0.56,
    "speedup_vs_torch": 2.17,
    "peak_memory_mb": 128.0
  },
  "error": null
}
```
