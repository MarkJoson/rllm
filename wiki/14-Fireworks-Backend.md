# Fireworks 后端 (Fireworks Backend)

Fireworks 后端使用 Fireworks AI 的远程 GPU 进行推理，适合无本地 GPU 或需要远程训练的场景。

> 源码参考：
> - [rllm/trainer/agent_trainer.py L155-179](../rllm/trainer/agent_trainer.py)
> - [rllm/trainer/verl/train_workflow_pipeline.py](../rllm/trainer/verl/train_workflow_pipeline.py)
> - [rllm/engine/rollout/fireworks_engine.py](../rllm/engine/rollout/fireworks_engine.py)

## 概述

| 维度 | 说明 |
|------|------|
| **适用场景** | 远程推理、API-based训练、评估 |
| **推理方式** | Fireworks AI HTTP API |
| **训练方式** | 管道式（PipelineTaskRunner） |
| **模式限制** | 仅支持 Workflow 模式 |
| **本地 GPU** | ❌ 不需要 |

## 架构

```
AgentTrainer._train_fireworks()
    │
    ├── ray.init()
    │
    └── PipelineTaskRunner.remote()
         └── run()
              ├── 初始化 FireworksEngine
              │   └── HTTP 客户端 → Fireworks API
              ├── 初始化 AgentWorkflowEngine
              └── 训练循环:
                   ├── execute_tasks() → Episodes
                   ├── transform_results_for_verl() → DataProto
                   └── 训练更新（远程/管道式）
```

## 使用

```python
trainer = AgentTrainer(
    workflow_class=MathWorkflow,
    workflow_args={"reward_fn": math_reward_fn},
    config={
        "fireworks_api_key": "fw-xxxxx",
        "fireworks_model": "accounts/fireworks/models/qwen3-4b",
    },
    backend="fireworks",  # ← 选择 fireworks
)
trainer.train()
```

## 与 VERL 对比

| 维度 | Fireworks | VERL |
|------|-----------|------|
| 推理位置 | 远程 (Fireworks API) | 本地 (vLLM) |
| GPU 需求 | ❌ | ✅ |
| 延迟 | 高（网络） | 低（进程内） |
| 全参数训练 | ❌ | ✅ |
| LoRA | ✅ | ✅ |
| Agent/Env 模式 | ❌ | ✅ |
| SDK 模式 | ❌ | ✅ |
| 适用 | 评估/小规模 | 生产训练 |
