# Stepwise Advantage (逐步优势)

## 概述

Stepwise Advantage 是 rLLM 对多步 Agent 交互的核心优化——允许在多步轨迹中为每一步分配独立的优势值，改善多步任务中的信用分配问题。

> 源码参考：
> - [rllm/engine/agent_workflow_engine.py L346-375](../rllm/engine/agent_workflow_engine.py)
> - 配置：`rllm.stepwise_advantage`

## 动机

在标准 RL 训练中，多步轨迹的所有 step 共享同一个 trajectory-level reward：

```
Trajectory: step_0 → step_1 → step_2  (reward = 0.8)
所有 step 的优势 = f(0.8)  ← 哪个 step 贡献了奖励？
```

这导致信用分配模糊——好的 step 和差的 step 获得相同的梯度更新。

## 配置

```yaml
rllm:
  stepwise_advantage:
    enable: true         # 启用逐步优势
    mode: "broadcast"    # 或 "per_step"
```

## 两种模式

### broadcast 模式（推荐）

所有 step 共享 trajectory reward，但每步独立参与 GRPO 分组：

```
Task q1, rollout 0:
  Trajectory (3 steps):
    Step 0: traj_reward=0.8, step_reward=0.0
    Step 1: traj_reward=0.8, step_reward=0.0  
    Step 2: traj_reward=0.8, step_reward=0.8  ← is_last_step=True

Task q1, rollout 1:
  Trajectory (3 steps):
    Step 0: traj_reward=0.3, step_reward=0.0
    Step 1: traj_reward=0.3, step_reward=0.0
    Step 2: traj_reward=0.3, step_reward=0.3  ← is_last_step=True

GRPO 归一化（对 traj_reward）:
  rollout 0 的 step_0/1/2: A = (0.8 - 0.55) / σ
  rollout 1 的 step_0/1/2: A = (0.3 - 0.55) / σ
```

> 优点：每步都参与训练，步级 tokenization 避免多步拼接的 token 不一致问题。

### per_step 模式

每步使用自己的 step_reward：

```
Task q1, rollout 0:
  Step 0: step_reward=0.0
  Step 1: step_reward=0.3  
  Step 2: step_reward=0.8

Task q1, rollout 1:
  Step 0: step_reward=0.0
  Step 1: step_reward=0.1
  Step 2: step_reward=0.3

GRPO 按 step 分组归一化:
  step_0: A_0 = (0.0-0.0)/σ, A_1 = (0.0-0.0)/σ  # 无差异
  step_1: A_0 = (0.3-0.2)/σ, A_1 = (0.1-0.2)/σ  # 有差异
  step_2: A_0 = (0.8-0.55)/σ, A_1 = (0.3-0.55)/σ # 有差异
```

> 优点：更精细的信用分配。需要环境提供逐步奖励。

## 对 DataProto 的影响

| 维度 | stepwise=False（默认） | stepwise=True |
|------|---------------------|---------------|
| 展开方式 | 每 trajectory → 1 行 | 每 step → 1 行 |
| DataProto 行数 | = N_trajectories | = N_total_steps |
| Tokenization | 累积式（最后一步的完整对话）| 逐步独立 |
| repeat_counts | 每 episode 的 trajectory 数 | 每 episode 的 step 总数 |
| traj_rewards | ✅ | ✅ (broadcast 到每步) |
| step_rewards | ❌ | ✅ |
| is_last_step | ❌ | ✅ (最后一步标记) |

## 使用建议

| 场景 | 推荐模式 |
|------|---------|
| 单步任务（数学QA, 代码生成） | `enable=False` |
| 多步 Agent-Env（SWE, Web） | `enable=True, mode=broadcast` |
| 环境提供逐步奖励 | `enable=True, mode=per_step` |
| 调试/快速实验 | `enable=False` |

---

# 分布式训练与 Ray (Distributed Training with Ray)

## 概述

rLLM 通过 verl 后端使用 **Ray** 进行分布式训练，支持多 GPU、多节点训练。

## Ray 架构

```
Driver Process (AgentTrainer._train_verl())
    │
    ├── ray.init(runtime_env=get_ppo_ray_runtime_env())
    │
    └── TaskRunner.remote()          ← Ray Actor
         │
         ├── Actor Worker (FSDP)     ← 多 GPU 分片
         │   ├── GPU 0: shard 0
         │   ├── GPU 1: shard 1
         │   └── ...
         │
         ├── Critic Worker (FSDP)    ← 多 GPU 分片
         │
         ├── Ref Worker (FSDP)       ← 可选 param_offload
         │
         └── Rollout Worker          ← vLLM/SGLang
             ├── Replica 0           ← TP 并行
             ├── Replica 1
             └── ...
```

## 多节点配置

```bash
# Head 节点
ray start --head --port=6379 --num-gpus=4

# Worker 节点
ray start --address='192.168.1.100:6379' --num-gpus=4
```

## GPU 分配

```yaml
trainer:
  n_gpus_per_node: 4
actor_rollout_ref:
  actor:
    strategy: fsdp            # FSDP 分片策略
  rollout:
    tensor_model_parallel_size: 2  # TP 并行度
    gpu_memory_utilization: 0.3    # 推理 GPU 显存占比
  ref:
    fsdp_config:
      param_offload: true     # Ref 参数卸载到 CPU
```

## Runtime 环境传播

`get_ppo_ray_runtime_env()` 确保所有 Ray Worker 能导入 rLLM 包和用户自定义代码。

## 常见问题

| 问题 | 解决 |
|------|------|
| OOM | 减小 `gpu_memory_utilization`, 启用 `param_offload` |
| Ray 连接超时 | 检查防火墙，确保端口可达 |
| Import Error | 检查 `runtime_env` 配置 |
| Slow rollout | 增加 TP 并行度，减少 batch_size |
