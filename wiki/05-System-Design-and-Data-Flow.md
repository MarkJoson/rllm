# 系统设计与数据流 (System Design and Data Flow)

## 整体架构

rLLM 的系统架构分为四层：**数据层**、**执行层**、**转换层**和**训练层**。

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据层 (Data Layer)                       │
│  Dataset → DataLoader → task dicts                              │
│  [rllm/data/dataset.py]                                         │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       执行层 (Execution Layer)                   │
│  ┌────────────────┐ ┌──────────────────┐ ┌──────────────────┐  │
│  │ExecutionEngine │ │ WorkflowEngine   │ │   SdkEngine      │  │
│  │Agent-Env循环   │ │ Workflow.run()   │ │ LiteLLM Proxy    │  │
│  │[exec_engine.py]│ │ [wf_engine.py]   │ │ [sdk_engine.py]  │  │
│  └──────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘  │
│         │                    │                     │             │
│         └────────────────────┼─────────────────────┘             │
│                              │                                   │
│                    ┌─────────┴────────────┐                      │
│                    │   Rollout Engines     │                      │
│                    │ VerlEngine/OpenAI/... │                      │
│                    │ [rollout/*.py]        │                      │
│                    └──────────────────────┘                      │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                │ list[Episode] + list[Trajectory]
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       转换层 (Transform Layer)                   │
│  transform_results_for_verl()                                   │
│  Episode → tokenize → pad → mask → DataProto                   │
│  [agent_workflow_engine.py L229-514]                            │
│  [agent_sdk_engine.py L464-712]                                 │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                │ DataProto (tensors + metadata)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       训练层 (Training Layer)                    │
│  verl PPO Trainer:                                              │
│  ├── Ref Model → ref_log_probs                                 │
│  ├── Critic → value estimates                                  │
│  ├── Advantage (GRPO/REINFORCE/RLOO)                           │
│  └── Actor 策略梯度更新 (PPO/GRPO)                              │
│  [train_agent_ppo.py]                                           │
└─────────────────────────────────────────────────────────────────┘
```

## GPU 时分复用

VERL 后端的核心创新是 GPU 显存的时分复用型——同一组 GPU 在**推理**和**训练**之间切换：

```
时间线:
├── 推理阶段 (GPU 被 vLLM 占用)
│   wake_up() → 加载 vLLM 权重 + 分配 KV Cache
│   ├── 并发执行 N 个 Agent-Env 交互
│   │   ├── prompt → vLLM → tokens + logprobs
│   │   ├── env.step() → observation
│   │   └── 重复...
│   sleep() → 释放 vLLM 权重 + KV Cache
│
├── 训练阶段 (GPU 被 FSDP 占用)
│   ├── Actor 前向 → actor_log_probs
│   ├── Ref 前向 → ref_log_probs
│   ├── Critic 前向 → values
│   ├── 优势计算 → advantages
│   └── PPO 梯度更新
│
└── 下一轮推理...
```

## 数据流详解

### 1. Dataset → Task Dicts

```python
# DataLoader 产出 batch
batch = DataProto(
    non_tensor_batch={
        "extra_info": np.array([task_dict_1, task_dict_2, ...]),
        "task_ids": np.array(["id1", "id2", ...]),
    }
)
```

### 2. Task Dict → Episode

执行引擎接收 task dicts，执行 Agent-Env 交互，产出 Episodes：

```python
task_dict = {
    "question": "What is 2+3?",
    "ground_truth": "5",
    "data_source": "gsm8k",
}
# 经过执行引擎 → 
episode = Episode(
    id="task1:0",
    is_correct=True,
    trajectories=[
        Trajectory(name="solver", reward=1.0, steps=[
            Step(prompt_ids=[...], response_ids=[...], logprobs=[...], reward=1.0),
        ]),
    ],
)
```

### 3. Episode → DataProto

`transform_results_for_verl()` 将 Episodes 转换为训练 tensor：

```python
DataProto(
    batch={
        "input_ids":      [B, P+R],     # prompt + response
        "attention_mask":  [B, P+R],     # 有效 token
        "position_ids":    [B, P+R],     # 位置编码
        "prompts":         [B, P],       # prompt 部分（左 pad）
        "responses":       [B, R],       # response 部分（右 pad）
        "response_mask":   [B, R],       # loss mask
        "traj_rewards":    [B, R],       # 轨迹奖励（最后 token）
        "step_rewards":    [B, R],       # 步级奖励（最后 token）
    },
    non_tensor_batch={
        "episode_ids":      [B],
        "trajectory_ids":   [B],
        "is_correct":       [B],
        "is_valid":         [B],
    },
    meta_info={
        "repeat_counts": [N_episodes],   # 每个 episode 的步数
    },
)
```

### 4. DataProto → 训练更新

```python
# Verl PPO Trainer 接收 DataProto:
ref_log_probs = ref_model.forward(input_ids)           # 参考模型
actor_log_probs = actor_model.forward(input_ids)       # 策略模型
values = critic_model.forward(input_ids)                # 价值模型

# 优势计算（取决于配置）
if algorithm == "grpo":
    advantages = group_normalized(traj_rewards)         # 组内归一化
elif algorithm == "reinforce":
    advantages = traj_rewards - baseline                # 基线减除
elif algorithm == "rloo":
    advantages = leave_one_out(traj_rewards)            # Leave-One-Out

# PPO 策略更新
ratio = exp(actor_log_probs - old_log_probs)
clipped_ratio = clip(ratio, 1-eps, 1+eps)
policy_loss = -min(ratio * advantages, clipped_ratio * advantages)
```

## Prompt Tokenization 与 Masking

### Tokenization 过程

```
消息历史:
[system: "You are a solver"]
[user: "What is 2+3?"]
[assistant: "Let me think... The answer is 5"]
[user: "Observation: Correct"]
[assistant: "Final answer: \\boxed{5}"]

↓ ChatTemplateParser.tokenize_and_mask_cumulative()

prompt_ids:  [sys, user_1]
response_ids: [asst_1, user_2, asst_2]
response_mask: [1,1,1,  0,0,    1,1,1]
                ^^^^^^^^        ^^^^^^^^ assistant tokens (loss=1)
                         ^^^^^^ user/env token (loss=0)
```

### Response Masking 规则

| Token 来源 | mask 值 | 是否计算 loss |
|-----------|---------|-------------|
| System prompt | N/A (在 prompt 中) | ❌ |
| User/Environment 消息 | 0 | ❌ |
| Assistant 回复 | 1 | ✅ |
| Padding | 0 | ❌ |

## 多 Agent 数据流

当一个 Episode 包含多个 Agent role（如 solver + judge）时：

```
Episode(task_id="q1:0")
├── Trajectory(name="solver", steps=[step_0, step_1], reward=0.8)
├── Trajectory(name="judge", steps=[step_0], reward=1.0)

→ DataProto 展开为 3 行（2步 solver + 1步 judge）:
Row 0: solver_step_0  (traj_reward=0.8, step_reward=0.0)
Row 1: solver_step_1  (traj_reward=0.8, step_reward=0.8)  ← is_last_step=True
Row 2: judge_step_0   (traj_reward=1.0, step_reward=1.0)  ← is_last_step=True

meta_info["repeat_counts"] = [3]  # 这个 episode 贡献了 3 行
```

## Stepwise vs. Trajectory-Level 优势计算

### Trajectory-Level（默认）

```
所有 step 共享同一个 traj_reward → GRPO 在 group 内对 traj_reward 归一化
```

### Stepwise Advantage

```yaml
rllm:
  stepwise_advantage:
    enable: true
    mode: "broadcast"  # 或 "per_step"
```

**broadcast 模式**：用 trajectory reward 广播到所有 step
**per_step 模式**：每个 step 使用自己的 step_reward

详见 [Stepwise Advantage](31-Stepwise-Advantage.md)。

## 错误处理与 Compact Filtering

```yaml
rllm:
  compact_filtering:
    enable: true
    mask_error: true              # ERROR 终止 → is_valid=False
    mask_timeout: true            # TIMEOUT → is_valid=False
    mask_max_prompt_length_exceeded: true
    mask_max_response_length_exceeded: false
    mask_env_done: false          # 正常结束不过滤
    mask_unknown: true
```

`is_valid=False` 的样本在优势计算和梯度更新中被排除。
