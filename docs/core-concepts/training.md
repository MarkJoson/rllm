# RL Training with AgentTrainer

!!! warning "Deprecation warning"
    The current `AgentTrainer` will be soon replaced with a newer, [`UnifiedTrainer`](../experimental/unified-trainer.md)-based agent trainer. If you are working with `Tinker`, we highly recommend you to migrate to the new [`UnifiedTrainer`](../experimental/unified-trainer.md)-based workflow trainer. See [RL Training with Tinker (with Unified Trainer)](../examples/tinker_rl.md) for an upgraded solver-judge workflow example.

Training in rLLM uses reinforcement learning algorithms to update agent policies based on rewards. This page explains the training architecture, available algorithms, and how to configure and run training jobs.

## Overview

The `AgentTrainer` is the high-level interface for training reinforcement learning agents in rLLM. It provides a simplified API that wraps the underlying training infrastructure (verl), allowing you to train custom agents in custom environments without directly managing the complex distributed training setup.

## Architecture

### Core Components

The AgentTrainer orchestrates several key components:

1. **Agent**: The learning policy that generates actions based on observations
2. **Environment**: The task environment that provides observations and rewards
3. **RL Trainer**: The underlying reinforcement learning algorithm implementation

### Training Flow

The AgentTrainer serves as a wrapper over the training engine `verl`. When `trainer.train()` is called, the following process occurs:

**Initialization**: The system initializes the `AgentPPOTrainer`, which inherits from `verl`'s `RayPPOTrainer`. We replace the original trajectory generation logic with rLLM's AgentExecutionEngine.

**Setup Phase**: The `AgentPPOTrainer` performs the following setup:

   - Sets up Ray workers for distributed model training
   - Initializes the AgentExecutionEngine
   - Loads the dataset and splits it into mini-batches

**Training Loop**: For each mini-batch:

   - Data is passed to rLLM's AgentExecutionEngine
   - The engine initializes agent-environment pairs to process the mini-batch in parallel
   - Agent trajectories are collected through environment interactions

**Update Phase**: After a mini-batch is sampled:

   - The trainer transforms trajectories into `verl`'s format
   - Gradient updates are performed using the collected trajectories

For more details, reference `rllm/trainer/agent_ppo_trainer.py`, where we implement our custom RL training flow for agents.

## Basic Usage

### Simple Training Setup

```python
import hydra
from rllm.train.agent_trainer import AgentTrainer
from rllm.agents import YourCustomAgent
from rllm.environments import YourCustomEnvironment
from rllm.data import DatasetRegistry

@hydra.main(config_path="pkg://rllm.train.config", config_name="ppo_trainer")
def main(config):
    # Load datasets
    train_dataset = DatasetRegistry.load_dataset("your_dataset", "train")
    val_dataset = DatasetRegistry.load_dataset("your_dataset", "test")
    
    # Initialize trainer
    trainer = AgentTrainer(
        agent_class=YourCustomAgent,
        env_class=YourCustomEnvironment,
        agent_args={},
        env_args={},
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    
    # Start training
    trainer.train()
```

## Configuration

### Main Configuration File

rLLM adopts the same configuration structure as verl's `ppo_trainer.yaml`, with additional rLLM-specific configurations for our AgentExecutionEngine.

#### Agent-Specific Configuration
```yaml
agent:
  max_steps: 10              # Maximum steps per episode
  n_parallel_agents: 8       # Number of parallel agent instances
  use_stepwise_advantage: true  # Enable step-wise advantage calculation
  trajectory_timeout: 300    # Timeout for trajectory collection (seconds)
```

## 批量大小(Batch Size)参数配置指南

在 RLHF 和 PPO 训练中（特别是在 `verl` 和 `rllm` 框架中），在流程的不同阶段存在多个维度的批量大小参数：经验采样 (Rollout)、旧策略/奖励计算 (Reward/Ref) 和 Actor 模型更新 (PPO Update)。理解它们之间的关系对保持训练稳定高效、避免由于显存不足导致 CUDA OOM 至关重要。

### 1. 宏观 / 算法视角：经验收集与经验池

这个层面的参数决定了单次 PPO Iteration（迭代）中能够筹集并构建的总经验池大小。

*   `data.train_batch_size` (或 Prompt 批量大小)
    *   **含义**: 每次迭代时从训练数据集中随机采样出的独立 prompt 的数量。
    *   **设置依据**: 保证采样任务的多样性。这个数值通常设为 128 到 512 之间，如果在做小规模功能测试则可能缩减为更小的数字。
*   `rollout.n` (每个 Prompt 的采样数量)
    *   **含义**: 对于每一个单独的 prompt，要求大模型 (Actor) 采样生成多少个不同的回答 (responses)。
    *   **设置依据**: 主要是为了提升探索能力（多见于 GRPO 中，通常 $n \geq 4$）。在标准 PPO 场景中，通常 $n=1, 2, \text{或 } 4$。
*   **核心关系公式**:
    *   `全局经验缓冲池大小 (Global_Experience_Buffer_Size)` = `train_batch_size` $\times$ `rollout.n`
    *   例如，如果 `train_batch_size` 是 8 并且 `rollout.n` 是 8，那么每次 PPO 迭代会向系统提供 64 条经验轨迹。这段包含 64 个完整采样的经验池便成为后续 PPO 更新训练的基底数据集。

### 2. 中观 / 优化器视角：PPO 的梯度更新

收集齐上一步计算出的 `全局经验缓冲池大小` 后，PPO 会运行几个完整的 Epoch 循环 (`ppo_epochs`) 进行神经网络 Actor 的优化更新，它不把数据一把塞入，而是切分成小块：

*   `actor.ppo_mini_batch_size`
    *   **含义**: 在执行一次梯度反向传播并在优化器上跨出一步时，所使用的全局逻辑批量大小。
    *   **设置依据**: 出于算法数学优化和收敛稳定性的必要考量，强化学习的更新阶段需要足够且平滑的批量以压制严重的数据噪声。
    *   **核心关系公式**:
        1. 它应当能被 `全局经验缓冲池大小 (Global_Experience_Buffer_Size)` 整除。
        2. 每 Epoch 的网络模型更新步数 = `全局经验缓冲池大小 (Global_Experience_Buffer_Size)` / `ppo_mini_batch_size`。
        3. 举例，如果当前收集了 64 条经验且 `ppo_mini_batch_size=16`，则每一个 epoch 会拆出 4 步完整的梯度下降计算。

### 3. 微观 / 硬件视角：严格的 VRAM (显存) 限制

LLM 模型十分庞大，原生框架无法硬吞整个 `ppo_mini_batch_size` 规模的数据并发往前馈/后馈而不崩溃（OOM）。

*   `micro_batch_size_per_gpu` (如 `log_prob_micro_batch_size_per_gpu`)
    *   **含义**: 一张独立的显卡上，每一次单次网络运算实际承接和消化的样本/轨迹数量。
    *   **设置依据**: **严格受物理显存 (VRAM) 瓶颈界定。** 当遇到 CUDA OOM 时，**这是唯一一处你需要调低的参数。** 当显卡空转率过高时，也应该调大它。下调此参数并不影响算法的实际数学效果，只会在底层引入更多的自动梯度累积步（使得前向计算化大为小）。
*   `ppo_max_token_len_per_gpu` (动态 Packing 封包)
    *   **含义**: 当启动 `use_dynamic_bsz=True` 选项时，底层将放弃条数思维，改用文本实际长短拼簇以保障最极致的吞吐量。这时不再关注条数限制，只要同一批被放上卡计算的数据的 Token 累加总量均匀地置于这个天花板极值以下，以避开 OOM。

### 调参排错自检单 (Tuning Checklist)
- **模型不好收敛 / KL 散度爆炸？** $\rightarrow$ 请调大或校验 `train_batch_size` 和 `ppo_mini_batch_size` 的值，保证训练有足够的稳定梯段和良好的探索多样性。
- **CUDA OOM 显存爆破？** $\rightarrow$ 千万别乱降前两者的数值。请直接且干脆地调降底层的 `*_micro_batch_size_per_gpu` 或 `ppo_max_token_len_per_gpu`。
- **Reward 奖励停滞 / 发现不到好路线？** $\rightarrow$ 尝试增加 `rollout.n` 这个参数为单题解锁多样态探索路径，以此增大碰撞到正向奖励解的几率。