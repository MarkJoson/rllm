# Tinker 后端 (Tinker Backend)

Tinker 是一个轻量级训练后端，适合单机开发、调试和小规模实验。

> 源码参考：[rllm/trainer/deprecated/](../rllm/trainer/deprecated/)

## 概述

| 维度 | 说明 |
|------|------|
| **适用场景** | 开发调试、快速原型、单 GPU 实验 |
| **Python** | >= 3.11（严格要求） |
| **模型管理** | 本地模型或 HTTP API |
| **训练方式** | 单机梯度更新 |
| **分布式** | ❌ 不支持 |
| **位置** | `rllm/trainer/deprecated/` |

> **注意**：Tinker 后端位于 `deprecated/` 目录，但仍然可用。建议新项目使用 VERL 后端。

## 架构

```
AgentTrainer._train_tinker()
    │
    ├── workflow_class 存在？
    │   ├── 是 → TinkerWorkflowTrainer
    │   └── 否 → TinkerAgentTrainer
    │
    └── trainer.fit_agent()
         ├── 初始化 TinkerEngine (本地/HTTP)
         ├── 训练循环:
         │   ├── 加载 batch
         │   ├── 执行 Agent/Workflow
         │   ├── 计算梯度
         │   └── 更新权重
         └── 保存 checkpoint
```

## 使用

```python
trainer = AgentTrainer(
    agent_class=MathAgent,
    env_class=MathEnv,
    config={"model_path": "Qwen/Qwen3-1.7B"},
    backend="tinker",  # ← 选择 tinker
)
trainer.train()
```

## 与 VERL 对比

| 维度 | Tinker | VERL |
|------|--------|------|
| 设置复杂度 | 低（pip install） | 高（Ray + vLLM + FSDP） |
| 训练速度 | 慢（单机） | 快（多 GPU 分布式） |
| 调试友好 | ✅ 单进程 | ❌ 多进程/Actor |
| 生产就绪 | ❌ | ✅ |
