# 配置系统 (Configuration System)

## Hydra 配置框架

rLLM 使用 **OmegaConf** + **Hydra** 作为配置管理框架。配置通过三层叠加：YAML 文件 → Python dict → 命令行覆盖。

## 配置层次

```yaml
# 第一层：verl 默认配置
actor_rollout_ref:       # 模型/推理/参考配置
  model:                 # 模型路径、类型
  actor:                 # Actor 训练策略
  rollout:               # 推理引擎参数
  ref:                   # Ref 模型配置
data:                    # 数据路径、长度限制
training:                # batch size, 保存频率
algorithm:               # RL 算法选择
critic:                  # Critic 模型配置

# 第二层：rLLM 扩展配置  
rllm:
  agent:                  # Agent 交互参数
    max_steps: 5          # 最大步数
  workflow:               # Workflow 配置
    use_workflow: true    # 是否使用 Workflow
  stepwise_advantage:     # 逐步优势
    enable: false
    mode: "broadcast"
  compact_filtering:      # 无效样本过滤
    enable: true
    mask_error: true
  sdk:                    # SDK 模式配置
    processing:
      groupby_key: "session_name"
    store:
      path: "/tmp/traces.db"
  accumulate_reasoning: false  # 保留前步推理
  disable_thinking: false      # 移除 <think> 标签
  filter_token_mismatch: true  # 过滤 token 不一致

# 第三层：命令行覆盖
# python train.py actor_rollout_ref.model.path=Qwen/Qwen3-4B
```

## 常用配置范例

### 数学训练

```yaml
actor_rollout_ref:
  model.path: Qwen/Qwen3-4B
  rollout:
    temperature: 0.7
    gpu_memory_utilization: 0.4
data:
  max_prompt_length: 1024
  max_response_length: 4096
algorithm:
  adv_estimator: grpo
  group_size: 4
rllm:
  agent.max_steps: 1
```

### 代码训练

```yaml
actor_rollout_ref:
  model.path: Qwen/Qwen3-8B
  rollout:
    temperature: 0.6
    gpu_memory_utilization: 0.5
data:
  max_prompt_length: 2048
  max_response_length: 8192
rllm:
  agent.max_steps: 1
```

### 多步交互训练

```yaml
rllm:
  agent.max_steps: 10
  workflow:
    use_workflow: true
  stepwise_advantage:
    enable: true
    mode: broadcast
  accumulate_reasoning: true
```

## 在 AgentTrainer 中使用

```python
# 方式 1：dict
config = {
    "actor_rollout_ref.model.path": "Qwen/Qwen3-4B",
    "training.train_batch_size": 64,
}

# 方式 2：list（Hydra override 格式）
config = [
    "actor_rollout_ref.model.path=Qwen/Qwen3-4B",
    "training.train_batch_size=64",
]

# 方式 3：YAML 文件路径
config = "config/math_train.yaml"

trainer = AgentTrainer(config=config, ...)
```

---

# Stepwise Advantage (逐步优势)

详见 [28-Training-Scripts.md](28-Training-Scripts.md) 中的 Stepwise Advantage 部分。
