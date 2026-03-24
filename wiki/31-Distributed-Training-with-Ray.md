# 分布式训练与 Ray (Distributed Training with Ray)

详见 [30-Stepwise-Advantage.md](30-Stepwise-Advantage.md) 中的分布式训练部分。

## 核心概念

| 概念 | 说明 |
|------|------|
| **FSDP** | Fully Sharded Data Parallel — Actor/Critic/Ref 模型参数跨 GPU 分片 |
| **TP** | Tensor Parallel — vLLM 推理模型在多 GPU 间拆分 tensor |
| **Ray Actor** | 远程进程封装，管理训练循环 |
| **Runtime Env** | Ray 远程函数的 Python 环境配置 |
| **Wake/Sleep** | GPU 显存在推理和训练之间动态切换 |

## Ray 初始化

```python
# agent_trainer.py L127-135
if not ray.is_initialized():
    ray_init_settings = get_ray_init_settings(self.config)
    ray.init(
        runtime_env=get_ppo_ray_runtime_env(),
        **ray_init_settings
    )
```

`get_ray_init_settings()` 从 config 提取 Ray 初始化参数（如 `num_cpus`、`num_gpus`、`address`）。

## GPU 资源规划指南

### 单节点 4×A100 80GB

```yaml
trainer:
  n_gpus_per_node: 4
actor_rollout_ref:
  model.path: Qwen/Qwen3-4B
  actor:
    strategy: fsdp
  rollout:
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.4
  ref:
    fsdp_config:
      param_offload: true  # 省内存
```

### 8×H100 80GB (大模型)

```yaml
trainer:
  n_gpus_per_node: 8
actor_rollout_ref:
  model.path: Qwen/Qwen3-32B
  actor:
    strategy: fsdp
  rollout:
    tensor_model_parallel_size: 4
    gpu_memory_utilization: 0.3
```

---

# Rejection Sampling 与过滤 (Rejection Sampling and Filtering)

## 概述

Rejection Sampling 是一种训练策略——只使用"成功"的轨迹进行训练，丢弃失败的轨迹。

## 在 rLLM 中的实现

### Compact Filtering

rLLM 通过 `compact_filtering` 配置实现轨迹过滤：

```yaml
rllm:
  compact_filtering:
    enable: true
    mask_error: true          # 过滤 ERROR 终止
    mask_timeout: true        # 过滤 TIMEOUT
    mask_max_prompt_length_exceeded: true
    mask_unknown: true        # 过滤 UNKNOWN
    mask_env_done: false      # 保留正常结束
    mask_max_turns_exceeded: false
    mask_max_response_length_exceeded: false
```

被过滤的样本 `is_valid=False`，在优势计算中权重为 0。

### Overlong Filter

`AgentExecutionEngine` 的 `overlong_filter` 参数（[agent_execution_engine.py L363-368](../rllm/engine/agent_execution_engine.py#L363-L368)）：

```python
if self.overlong_filter:
    if termination_reason in [TRUNCATION, MAX_STEPS, TIMEOUT]:
        response_masks = zeros_like(response_tokens)  # 全零 → 不计算 loss
```

### GRPO 内置过滤

GRPO 算法在组内归一化时，如果组内所有轨迹奖励相同（如全部正确或全部错误），优势为 0，自动不更新梯度。

## 使用建议

| 场景 | 推荐 |
|------|------|
| 简单任务（high accuracy） | 关闭 overlong_filter |
| 困难任务（low accuracy） | 开启 compact_filtering |
| 多步交互 | 开启 overlong_filter |
| 代码生成（执行超时常见） | mask_timeout=true |

---

# 蒸馏系统 (Distillation System)

## 概述

rLLM 支持 **on-policy 蒸馏**——使用强模型生成的轨迹训练弱模型。

## DistillationWorkflow

`DistillationWorkflow` 是内置的蒸馏工作流：

```python
class DistillationWorkflow(Workflow):
    """
    On-policy 蒸馏:
    1. 使用当前模型生成 response
    2. 使用奖励函数评估
    3. 只保留正确的 (is_correct=True) 轨迹用于训练
    """
    async def run(self, task, uid, **kwargs):
        # ... 标准 QA 交互 ...
        episode.chat_completions = step.chat_completions  # 保留完整对话
        return episode
```

## 蒸馏数据流

```
教师模型或强 rollout
    │
    ├── 生成 response
    ├── 奖励评估
    ├── is_correct=True → 保留
    ├── is_correct=False → 丢弃
    │
    ▼
蒸馏数据集
    │
    └── 学生模型 SFT 训练
        └── loss = -log P(response | prompt)  (标准交叉熵)
```

## 配置

```yaml
rllm:
  distillation:
    enable: true
    filter_correct_only: true   # 只保留正确样本
    save_chat_completions: true # 保存完整对话（用于后续 SFT）
```

---

# Trace 收集与调试 (Trace Collection and Debugging)

## 概述

rLLM 提供多层次的 Trace 和日志系统，帮助开发者调试 Agent 行为和训练过程。

## EpisodeLogger

```python
class EpisodeLogger:
    def log_episodes_batch(self, episodes, step, ...):
        """将 Episode 批量写入日志"""
        for episode in episodes:
            self._log_episode(episode, step)
```

日志内容：
- Episode ID 和轨迹奖励
- 每步的 chat_completions（完整对话）
- 终止原因
- 指标（steps, llm_time, env_time）

## SDK Trace 系统

SDK 模式使用 **SQLite** 持久化 Trace 数据：

```python
self.store = SqliteTraceStore(db_path=config.rllm.sdk.store.path)

# 查询 Trace
traces = await store.get_by_session_uid(session_uid, since=start_time)
```

每个 Trace 记录：
- session_name（`{task_id}:{rollout_idx}:{retry_attempt}`）
- request（prompt 消息）
- response（完整 ModelOutput）
- 时间戳

## LiteLLM Proxy 日志

SDK 模式的 LiteLLM Proxy 记录所有通过它的 LLM 调用：

```python
# Flush traces 到 SQLite
success = await self.proxy_manager.flush_tracer(timeout=30.0)
```

## 调试技巧

| 问题 | 诊断方法 |
|------|---------|
| Agent 不调用工具 | 检查 system_prompt 是否包含工具说明 |
| 奖励始终为 0 | 检查 reward_fn 输入格式，检查 format_error_reward |
| Token 不一致 | 启用 `filter_token_mismatch`，检查 `token_mismatch` 指标 |
| 轨迹超时 | 增加 `trajectory_timeout`，检查 env.step() 耗时 |
| DataProto 为空 | 检查 `repeat_counts` 是否全为 0 |
| GPU OOM | 减小 `gpu_memory_utilization`，减小 batch_size |

## 指标监控

训练过程中记录的关键指标：

| 指标 | 说明 |
|------|------|
| `mean_reward` | 批次平均奖励 |
| `accuracy` | 批次正确率 |
| `mean_steps` | 平均交互步数 |
| `llm_time` | LLM 调用总耗时 |
| `env_time` | 环境执行总耗时 |
| `token_mismatch_rate` | token 不一致比例 |
| `filtered_rate` | compact filtering 过滤比例 |

---

# 测试基础设施 (Testing Infrastructure)

## 概述

rLLM 使用 pytest 进行测试，测试覆盖核心组件和域特定实现。

## 测试结构

```
tests/
├── test_agent.py             # Agent/Step/Trajectory 类型测试
├── test_execution_engine.py  # 执行引擎 E2E 测试
├── test_workflow.py          # Workflow 生命周期测试
├── test_reward.py            # 奖励函数测试
├── test_parser.py            # Chat 模板解析器测试
├── test_data.py              # 数据集加载测试
└── integration/
    ├── test_verl_training.py # VERL 训练 E2E 测试
    └── test_sdk_mode.py      # SDK 模式测试
```

## 运行测试

```bash
# 全部测试
pytest tests/

# 特定文件
pytest tests/test_reward.py

# 特定测试
pytest tests/test_reward.py::test_math_reward_correct

# 跳过需要 GPU 的测试
pytest tests/ -k "not gpu"

# 详细输出
pytest tests/ -v --tb=long
```

## 关键测试用例

### 奖励函数测试

```python
def test_math_reward_correct():
    reward_fn = RewardMathFn(RewardConfig())
    result = reward_fn(
        {"ground_truth": "42"},
        "<think>...</think>\nThe answer is \\boxed{42}"
    )
    assert result.reward == 1.0
    assert result.is_correct == True

def test_math_reward_format_error():
    config = RewardConfig(apply_format_reward=True)
    reward_fn = RewardMathFn(config)
    result = reward_fn({"ground_truth": "42"}, "The answer is 42")
    assert result.reward == 0.0  # 缺少 <think> 格式
```

### Agent 类型测试

```python
def test_trajectory_is_cumulative():
    traj = Trajectory(steps=[
        Step(chat_completions=[{"role": "user", "content": "Q1"}]),
        Step(chat_completions=[
            {"role": "user", "content": "Q1"},  # 前缀保留
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]),
    ])
    assert traj.is_cumulative() == True
```

## 持续集成

推荐在 CI/CD 中运行：

```yaml
# .github/workflows/test.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install -e ".[eval]"
      - run: pytest tests/ -k "not gpu" --tb=short
```
