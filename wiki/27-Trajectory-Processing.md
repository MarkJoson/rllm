# 轨迹处理 (Trajectory Processing)

## 概述

轨迹处理是将 Agent-Environment 交互记录（Episode/Trajectory/Step）转换为训练张量（DataProto）的过程。此过程在 `transform_results_for_verl()` 中实现。

> 源码参考：
> - [rllm/engine/agent_workflow_engine.py L229-514](../rllm/engine/agent_workflow_engine.py)
> - [rllm/engine/agent_sdk_engine.py L464-712](../rllm/engine/agent_sdk_engine.py)
> - [rllm/engine/agent_execution_engine.py L435-496](../rllm/engine/agent_execution_engine.py)

## Episode → DataProto 转换

### 展开规则

一个 Episode 可能包含多个 Trajectory（多角色），每个 Trajectory 可能包含多个 Step（多步交互）。展开为 DataProto 行的规则：

```
如果 stepwise_advantage.enable = False:
    每个 Trajectory → 1 行 DataProto
    (多步轨迹使用累积 tokenization)

如果 stepwise_advantage.enable = True:
    每个 Step → 1 行 DataProto
    (每步独立 tokenize)
```

### Tokenization 三种模式

| 条件 | 模式名 | 实现方式 |
|------|--------|---------|
| 多步 + stepwise=False | 累积 | `tokenize_and_mask_cumulative(last_step.chat)` |
| 单步 + 有 ModelOutput | 直接 | `prompt_ids=model_output.prompt_ids` |
| 单步 + 无 ModelOutput | 重新解析 | `tokenize_and_mask(chat_completions)` |
| stepwise=True | 逐步 | 每步独立 tokenize |

### Padding 策略

| 序列 | 填充方向 | 实现 |
|------|----------|------|
| Prompt | 左填充 (left pad) | `flip → pad → flip` |
| Response | 右填充 (right pad) | 标准 `pad_sequence` |

截断：`prompt[-max_prompt_length:]`，`response[:max_response_length]`

### 奖励放置

奖励值放在 response 序列的最后一个有效 token 位置：

```python
traj_rewards_batch[i, resp_len - 1] = traj_reward
step_rewards_batch[i, resp_len - 1] = step_reward
```

### Position IDs

标准模型：`position_ids = cumsum(attention_mask) - 1`
VLM (Qwen2VL/3VL)：`4D mrope position_ids = [text, temporal, height, width]`

## Compact Filtering

根据终止原因标记无效样本：

| 终止原因 | 默认过滤 | `is_valid` |
|----------|---------|------------|
| ERROR | ✅ 过滤 | False |
| TIMEOUT | ✅ 过滤 | False |
| MAX_PROMPT_LENGTH_EXCEEDED | ✅ 过滤 | False |
| UNKNOWN | ✅ 过滤 | False |
| ENV_DONE | ❌ 保留 | True |
| MAX_TURNS_EXCEEDED | ❌ 保留 | True |
| MAX_RESPONSE_LENGTH_EXCEEDED | ❌ 保留 | True |

## repeat_counts 机制

`meta_info["repeat_counts"]` 记录每个 Episode 展开为多少行 DataProto。这对 verl 训练器至关重要——它需要知道哪些行属于同一 Episode 以进行正确的 GRPO 分组。

```python
# 示例
# Episode 0: 1 trajectory × 3 steps → 3 行
# Episode 1: 2 trajectories × 1 step → 2 行
# Episode 2: dropped (no valid trajectories) → 0 行
repeat_counts = [3, 2, 0]
```

## Token 不一致处理

多步 tokenization 可能因中间步的 system/user 消息被重新 tokenize 而产生不一致。

`filter_token_mismatch` 配置控制处理方式：
- `True`：不一致时 `response_masks` 全部置零（不参与训练）
- `False`：忽略不一致，照常训练（可能有噪声）

> 详见 [Agent 执行引擎](06-Agent-Execution-Engine.md) 的 `assemble_steps()` 部分。
