# 后端对比与选择 (Backend Comparison and Selection)

## 全面对比表

| 维度 | VERL | Tinker | Fireworks |
|------|------|--------|-----------|
| **推理方式** | vLLM/SGLang Python API | 本地/HTTP | Fireworks HTTP API |
| **训练方式** | FSDP 分布式 | 单机梯度 | 管道式远程 |
| **GPU 需求** | ≥1 GPU (推荐 4+) | 单 GPU / CPU | ❌ |
| **Python** | >= 3.10 | >= 3.11 | >= 3.10 |
| **分布式** | ✅ (Ray) | ❌ | ❌ |
| **LoRA** | ✅ | ✅ | ✅ |
| **全参数** | ✅ | ✅ | ❌ |
| **VLM** | ✅ (Qwen2VL/3VL) | 部分 | ❌ |
| **GPU 共享** | ✅ (wake_up/sleep) | N/A | N/A |
| **Agent/Env 模式** | ✅ | ✅ | ❌ |
| **Workflow 模式** | ✅ | ✅ | ✅ (仅此) |
| **SDK 模式** | ✅ | ❌ | ❌ |
| **调试难度** | 高 (多进程) | 低 (单进程) | 中 |
| **设置复杂度** | 高 | 低 | 中 |
| **训练速度** | 最快 | 慢 | 中 |
| **代码位置** | `rllm/trainer/verl/` | `rllm/trainer/deprecated/` | `rllm/trainer/verl/` |

## 选择决策树

```
开始
│
├── 需要分布式训练？
│   ├── 是 → VERL
│   └── 否 ↓
│
├── 有本地 GPU？
│   ├── 否 → Fireworks
│   └── 是 ↓
│
├── 正在开发/调试？
│   ├── 是 → Tinker
│   └── 否 ↓
│
├── 需要 SDK 模式？
│   ├── 是 → VERL
│   └── 否 ↓
│
├── 需要 Agent/Env 模式？
│   ├── 是 → VERL 或 Tinker
│   └── 否 → 三者皆可（推荐 VERL）
```

## 典型工作流程

1. **开发阶段**：使用 Tinker 后端在单 GPU 上快速迭代 Agent/Environment/Workflow
2. **验证阶段**：使用 Fireworks 或 VERL 后端进行小批量训练验证
3. **生产阶段**：使用 VERL 后端在多 GPU 集群上进行完整训练

## 后端切换

后端切换只需更改一个参数：

```python
# 开发 → Tinker
trainer = AgentTrainer(workflow_class=MyWorkflow, backend="tinker")

# 验证 → Fireworks
trainer = AgentTrainer(workflow_class=MyWorkflow, backend="fireworks")

# 生产 → VERL
trainer = AgentTrainer(workflow_class=MyWorkflow, backend="verl")
```

Agent 逻辑、Workflow 逻辑、奖励函数等全部不需要修改。
