# 蒸馏系统 (Distillation System)

## 概述

rLLM 的蒸馏系统支持 **on-policy 蒸馏**——在训练循环中使用当前模型的正确输出作为蒸馏目标，实现自我改进。

> 源码参考：[rllm/workflows/distillation_workflow.py](../rllm/workflows/distillation_workflow.py)

## DistillationWorkflow

```python
class DistillationWorkflow(Workflow):
    """
    On-policy 蒸馏工作流:
    
    1. 使用当前策略模型生成 response
    2. 使用奖励函数判断正确性
    3. 正确的 response 保留完整 chat_completions
    4. verl 训练器使用这些作为 SFT 目标
    
    关键: chat_completions 被保存在 DataProto 的
    non_tensor_batch 中，供后续 SFT loss 计算使用。
    """
    
    def __init__(self, system_prompt, reward_fn, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = system_prompt
        self.reward_fn = reward_fn
    
    async def run(self, task, uid, **kwargs):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task["question"]},
        ]
        
        model_output = await self.rollout_engine.get_model_response(messages)
        
        reward_output = self.reward_fn(task, model_output.content)
        
        step = Step.from_model_output(model_output, messages=messages)
        step.reward = reward_output.reward
        
        trajectory = Trajectory(steps=[step], reward=reward_output.reward)
        return Episode(
            trajectories=[trajectory],
            is_correct=reward_output.is_correct,
        )
```

## 蒸馏 vs. 标准 RL 训练

| 维度 | 标准 RL (GRPO) | On-Policy 蒸馏 |
|------|---------------|---------------|
| **Loss** | Policy Gradient | Cross-Entropy (SFT) |
| **数据来源** | 当前策略 rollout | 当前策略正确 rollout |
| **过滤** | 所有轨迹参与 | 只使用 is_correct=True |
| **优势** | GRPO 归一化 | 无（直接 SFT） |
| **目标** | 政策优化 | 行为克隆 (正确行为) |

## 使用

```python
trainer = AgentTrainer(
    workflow_class=DistillationWorkflow,
    workflow_args={
        "system_prompt": "Solve math problems...",
        "reward_fn": math_reward_fn,
    },
    config=config,
    backend="verl",
)
trainer.train()
```

---

# Trace 收集与调试 (Trace Collection and Debugging)

## 日志系统

### EpisodeLogger

`EpisodeLogger` 记录每个 Episode 的详细信息：

```python
episode_logger.log_episodes_batch(episodes, step=current_step)
```

输出内容：
- Episode ID、终止原因
- 每个 Trajectory 的名称和奖励
- 对话历史（所有消息）
- 步级指标（llm_time, env_time）

### SDK Trace 系统

SDK 模式通过 **SQLite + LiteLLM Proxy** 收集 Trace：

```
用户代码 (OpenAI API 调用)
    │ HTTP
    ▼
LiteLLM Proxy
    ├── 路由到 vLLM 后端
    ├── 记录 request/response
    └── 持久化到 SQLite
    │
    ▼
SqliteTraceStore
    └── get_by_session_uid(uid, since=timestamp)
```

### Wandb 集成

训练指标可自动上报到 Weights & Biases：

```yaml
training:
  logger:
    type: wandb
    project: rllm-experiment
    entity: my-team
```

## 调试指南

### 常见问题排查

| 症状 | 原因 | 解决 |
|------|------|------|
| 奖励始终为 0 | format_error_reward | 检查系统提示是否要求 `\boxed{}` 格式 |
| Token 不一致警告 | 多步 tokenization 不一致 | 启用 `filter_token_mismatch` |
| DataProto 行数为 0 | 所有 episode 被过滤 | 检查 compact_filtering 配置 |
| 训练不收敛 | 奖励信号稀疏 | 放宽 format_error_reward，增加 group_size |
| GPU OOM | batch 过大 | 减小 `train_batch_size`，启用梯度检查点 |
| 环境超时 | env.step() 过慢 | 增加 `trajectory_timeout`，优化环境 |

### 诊断命令

```python
# 检查 DataProto 内容
print(f"DataProto shape: {data_proto.batch['input_ids'].shape}")
print(f"Valid samples: {sum(data_proto.non_tensor_batch['is_valid'])}")
print(f"Repeat counts: {data_proto.meta_info['repeat_counts']}")

# 检查 Episode
for ep in episodes:
    print(f"Episode {ep.id}: correct={ep.is_correct}, "
          f"term={ep.termination_reason}, "
          f"trajs={len(ep.trajectories)}")
    for traj in ep.trajectories:
        print(f"  {traj.name}: reward={traj.reward}, steps={len(traj.steps)}")
```
