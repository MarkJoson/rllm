# 训练脚本 (Training Scripts)

## 训练入口

rLLM 提供两种训练启动方式：

### 方式 1：Python API

```python
from rllm.trainer import AgentTrainer
from rllm.data import Dataset

trainer = AgentTrainer(
    workflow_class=MyWorkflow,
    config=config,
    train_dataset=Dataset.from_parquet("data/train.parquet"),
    backend="verl",
)
trainer.train()
```

### 方式 2：命令行

```bash
# 使用 Hydra/verl 的配置系统
python -m rllm.trainer.verl.train_agent_ppo \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    rllm.workflow.class=my_module.MyWorkflow \
    data.train_files=data/train.parquet \
    training.train_batch_size=64
```

## 关键训练参数

```yaml
# 模型配置
actor_rollout_ref:
  model:
    path: Qwen/Qwen3-4B
    enable_gradient_checkpointing: true
  actor:
    strategy: fsdp
    ppo_mini_batch_size: 128
    ppo_micro_batch_size_per_gpu: 8
    ppo_epochs: 1
    clip_ratio: 0.2
  rollout:
    name: vllm
    temperature: 0.7
    gpu_memory_utilization: 0.4
    tensor_model_parallel_size: 1
  ref:
    fsdp_config:
      param_offload: true

# 数据配置
data:
  max_prompt_length: 1024
  max_response_length: 4096
  train_files: data/train.parquet
  val_files: data/val.parquet

# 训练配置
training:
  train_batch_size: 64
  save_freq: 100
  total_training_steps: 1000

# 算法配置
algorithm:
  adv_estimator: grpo
  group_size: 4

# rLLM 特定
rllm:
  agent:
    max_steps: 5
  workflow:
    use_workflow: true
```

## checkpoint 管理

```yaml
training:
  save_freq: 100           # 每 100 步保存
  save_path: checkpoints/  # 保存路径
  resume_from: null        # 恢复训练的 checkpoint 路径
```

---

# 配置系统 (Configuration System)

## Hydra 配置

rLLM 使用 **Hydra** 作为配置管理框架，支持 YAML 文件和命令行覆盖。

### 配置结构

```
config/
├── default.yaml           # 默认配置
├── model/
│   ├── qwen3_4b.yaml
│   └── qwen3_1_7b.yaml
├── algorithm/
│   ├── grpo.yaml
│   └── reinforce.yaml
└── data/
    ├── gsm8k.yaml
    └── leetcode.yaml
```

### 命令行覆盖

```bash
python train.py \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \     # 模型路径
    training.train_batch_size=64 \                     # batch 大小
    algorithm.adv_estimator=grpo \                     # RL 算法
    rllm.agent.max_steps=5 \                           # 最大交互步
    +new_config_key=value                              # 添加新配置
```

### OmegaConf 访问

```python
# 在代码中访问配置
config = OmegaConf.load("config/default.yaml")
model_path = config.actor_rollout_ref.model.path
max_steps = config.rllm.agent.max_steps
```

### 配置优先级

```
命令行覆盖 > 环境变量 > YAML 文件 > 代码默认值
```

---

# Stepwise Advantage (逐步优势)

## 概述

Stepwise Advantage 允许在多步轨迹中为每一步分配独立的优势值，而非使用统一的轨迹级优势。

## 配置

```yaml
rllm:
  stepwise_advantage:
    enable: true
    mode: "broadcast"  # 或 "per_step"
```

## 两种模式

### broadcast 模式

所有 step 共享 trajectory 的 reward，但每步独立 tokenize：

```
Trajectory (3 steps, traj_reward=0.8):
  Step 0: traj_reward=0.8, step_reward=0.0
  Step 1: traj_reward=0.8, step_reward=0.0
  Step 2: traj_reward=0.8, step_reward=0.8 (last step)
```

GRPO 在组内对 traj_reward 归一化。所有 step 的优势相同。

### per_step 模式

每步使用自己的 step_reward：

```
Trajectory (3 steps):
  Step 0: step_reward=0.0 → 独立优势
  Step 1: step_reward=0.3 → 独立优势
  Step 2: step_reward=0.8 → 独立优势
```

GRPO 在组内对每步的 step_reward 独立归一化。

## 对 DataProto 的影响

| 维度 | stepwise=False | stepwise=True |
|------|---------------|---------------|
| 展开方式 | 每 trajectory → 1 行 | 每 step → 1 行 |
| DataProto 行数 | = trajectories 总数 | = steps 总数 |
| Tokenization | 累积/直接 | 逐步独立 |
| repeat_counts | 每 episode 的 trajectory 数 | 每 episode 的 step 总数 |
| 奖励 | traj_rewards 只 | traj_rewards + step_rewards |

## 何时使用

| 场景 | 推荐 |
|------|------|
| 单步任务（数学QA） | `enable=False` |
| 多步交互（SWE, Web） | `enable=True, mode=broadcast` |
| 逐步奖励可用 | `enable=True, mode=per_step` |
