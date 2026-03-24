# VERL 后端 (VERL Backend)

VERL（Versatile Efficient Reinforcement Learning for LLMs）是 rLLM 的主要训练后端，支持完整的分布式 PPO/GRPO 训练。

> 源码参考：
> - [rllm/trainer/verl/train_agent_ppo.py](../rllm/trainer/verl/train_agent_ppo.py)
> - [rllm/trainer/verl/ray_runtime_env.py](../rllm/trainer/verl/ray_runtime_env.py)

## 架构概览

```
AgentTrainer._train_verl()
    │
    ├── ray.init(runtime_env=get_ppo_ray_runtime_env())
    │
    └── TaskRunner.remote()  ← Ray Actor
         │
         └── run()
              ├── 加载 config + dataset
              ├── 初始化模型 (Actor/Critic/Ref)
              │   ├── FSDP 分片 → 多 GPU
              │   └── LoRA (可选)
              ├── 初始化 vLLM/SGLang rollout
              │   ├── Tensor Parallel (可选)
              │   └── GPU Memory Utilization 配置
              ├── 初始化执行引擎
              │   ├── AgentWorkflowEngine (workflow模式)
              │   ├── AgentExecutionEngine (agent-env模式)
              │   └── AgentSdkEngine (SDK模式)
              └── 训练循环
                   ├── wake_up() → 加载 vLLM 权重
                   ├── 生成轨迹 → Episodes
                   ├── sleep() → 释放 vLLM
                   ├── transform_results_for_verl() → DataProto
                   ├── Critic 前向 → values
                   ├── Ref 前向 → ref_log_probs
                   ├── 优势计算
                   ├── PPO/GRPO 梯度更新
                   │   ├── 多 mini-batch
                   │   └── KL 散度惩罚
                   └── 保存 checkpoint
```

## GPU 显存时分复用

VERL 最核心的设计是 **推理-训练 GPU 时分复用**：

```python
# 推理阶段：vLLM 占用 GPU
await self.rollout_engine.wake_up()   # 加载权重到 GPU
# ... 生成 trajectories ...
await self.rollout_engine.sleep()     # 释放权重+KV Cache

# 训练阶段：FSDP 占用 GPU
# ... 前向+反向+优化器更新 ...
```

`VerlEngine.wake_up()` ([verl_engine.py L108-110](../rllm/engine/rollout/verl_engine.py#L108-L110)) 并行唤醒所有 rollout replicas：
```python
async def wake_up(self):
    await asyncio.gather(*[replica.wake_up() 
                          for replica in self.rollout_manager.rollout_replicas])
```

## 模型配置

```yaml
actor_rollout_ref:
  model:
    path: "Qwen/Qwen3-4B"           # HuggingFace 模型路径
    enable_gradient_checkpointing: true  # 梯度检查点（省内存）
  actor:
    strategy: fsdp                    # 训练策略
    ppo_mini_batch_size: 128          # PPO mini-batch 大小
    ppo_micro_batch_size_per_gpu: 8   # 每 GPU micro-batch
    ppo_epochs: 1                     # PPO epoch 数
    clip_ratio: 0.2                   # PPO clip ratio
    kl_ctrl:
      type: fixed                     # KL 控制类型
      kl_coef: 0.0                    # KL 系数（0=不使用）
  rollout:
    name: vllm                        # 推理引擎
    temperature: 0.7                  # 采样温度
    top_k: -1                         # top-k 采样
    top_p: 1.0                        # top-p 采样
    tensor_model_parallel_size: 1     # TP 并行
    gpu_memory_utilization: 0.4       # GPU 显存占比
    do_sample: true
  ref:
    fsdp_config:
      param_offload: true             # Ref 模型参数卸载到 CPU
```

## 优势计算算法

| 算法 | config 值 | 说明 |
|------|-----------|------|
| GRPO | `grpo` | 组内归一化：`A_i = (R_i - mean(R)) / std(R)` |
| REINFORCE | `reinforce` | 基线减除：`A = R - baseline` |
| RLOO | `rloo` | Leave-One-Out：`A_i = R_i - mean(R_{-i})` |

```yaml
algorithm:
  adv_estimator: grpo    # 或 reinforce, rloo
  group_size: 4           # GRPO 组大小（每个 task 的 rollout 数）
```

## Ray Runtime 环境

`get_ppo_ray_runtime_env()` ([ray_runtime_env.py](../rllm/trainer/verl/ray_runtime_env.py))：

```python
def get_ppo_ray_runtime_env():
    import rllm
    rllm_path = Path(rllm.__file__).parent.parent
    return {
        "working_dir": str(rllm_path),
        "py_modules": [str(rllm_path / "rllm")],
    }
```

> 确保所有 Ray 工作节点能正确导入 `rllm` 和用户的自定义代码。

## 典型硬件配置

| 场景 | GPU | 模型大小 | 推荐配置 |
|------|-----|---------|---------|
| 开发调试 | 1×A100 40GB | ≤ 4B | `gpu_memory_utilization=0.4, ref.param_offload=true` |
| 标准训练 | 4×A100 80GB | ≤ 8B | `gpu_memory_utilization=0.5, TP=1` |
| 大模型 | 8×H100 80GB | ≤ 70B | `gpu_memory_utilization=0.3, TP=4, FSDP` |
