---
name: kernel-verifier
description: >
  算子验证——AST 退化检查 + NPU 上数值正确性 + torch_npu.profiler 性能采集。
  所有验证通过 tools/operator_pipeline.sh 统一执行。
triggers:
  - verify
  - benchmark
  - test
  - correctness
---

# Kernel Verifier

## 验证流程

由 `tools/operator_pipeline.sh --op_name <op_name>` 统一编排：

```
[1. AST 退化预检查]    → validate_triton_impl.py（无需 NPU，毫秒级）
        ↓ 通过
[2. 数值正确性]        → verify.py（NPU 上对比 Model vs ModelNew）
        ↓ 通过
[3. 性能测试]          → benchmark.py（torch_npu.profiler + fallback 计时）
        ↓
[4. 写入 metrics.json]
```

## 脚本说明

### validate_triton_impl.py

纯 AST 静态分析，检测 PyTorch 退化（Type 1/2/3）。
- 不需要 NPU / torch 运行时，可用 venv Python。
- `--json` 输出结构化结果。

### verify.py

NPU 上加载 `{op_name}_torch.py`（Model）和 `{op_name}_triton_ascend_impl.py`（ModelNew），
固定种子对比输出。按 dtype 选择精度阈值：

| dtype | 阈值 |
|-------|------|
| float16 | 0.004 |
| bfloat16 | 0.03 |
| int8 | 0.01 |
| 其他 | 0.02 |

### benchmark.py

优先使用 `torch_npu.profiler`（Level1），解析 `operator_details.csv`；
profiler 失败时 fallback 到 `time.perf_counter()`。
输出 JSON 包含 `framework` / `implementation` latency 和 `speedup_vs_torch`。

## Agent 使用方式

**不要手动调用这些脚本**，统一运行：

```bash
bash tools/operator_pipeline.sh --op_name <op_name>
```

然后读取 `metrics.json`。
