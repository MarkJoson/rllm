# `Token` vs `Step` 模式详解

> 涉及文件：
> - `rllm/engine/agent_execution_engine.py` — `run_agent_trajectory_async(mode=...)`
> - `rllm/trainer/verl/agent_ppo_trainer.py` — `generate_agent_trajectory()` / `generate_agent_steps()`

---

## 何处决定使用哪种模式

| 调用方 | 传入 mode | 后续处理 | 启用条件 |
|--------|-----------|---------|---------|
| `generate_agent_trajectory()` | `"Token"` | `_transform_agent_trajectories()` | 默认 PPO/GRPO 训练 |
| `generate_agent_steps()` | `"Step"` | `_transform_agent_steps()` | `config.rllm.stepwise_advantage.enable = True` |

---

## `Token` 模式（整轨迹展平）

**核心思想**：把整个多轮交互拼成**一条**连续的 token 序列，视为一个标准的 seq2seq 样本。

```
[prompt_tokens] + [response_tokens]
```

- `prompt_tokens`：初始问题 token ids（左 padding 到 `max_prompt_length`）
- `response_tokens`：所有后续 token（LLM 输出 + 工具/环境返回）拼接
- `response_masks`：**1** = LLM 生成的 token（计入 loss），**0** = 环境/工具返回的文本（不计入 loss）
- `trajectory_reward`：整条轨迹奖励，放在**最后一个有效 response token**（通常是 EOS）上

```
图示（2-step 工具调用轨迹）：

prompt  │ A1 │ obs1 │ A2 │ obs2 │ A3 │
mask    │  1 │   0  │  1 │   0  │  1 │  ← 只有 LLM 输出计 loss
reward  │  0 │   0  │  0 │   0  │  r │  ← 只在最后一位放 reward
```

**返回 dict 结构**：
```python
{
    "prompt_tokens": Tensor,       # shape: (prompt_len,)
    "response_tokens": Tensor,     # shape: (response_len,)
    "response_masks": Tensor,      # shape: (response_len,)
    "trajectory_reward": float,
    "idx": int,
    "chat_completions": list[dict],
    "metrics": dict,               # llm_time, env_time, total_time, token_mismatch ...
}
```

**1 trajectory → 1 训练样本**

---

## `Step` 模式（逐步展开）

**核心思想**：把每个交互步骤拆成**独立**的训练样本，每步保存自己的 prompt/response，并计算 Monte Carlo return。

```
step 0: [sys + q + obs0 | A1]               mc_return[0]
step 1: [sys + q + obs0 + A1 + obs1 | A2]   mc_return[1]
step 2: [... + obs2 | A3]                   trajectory_reward  (last step)
```

- 每步各自 re-tokenize（在 `_transform_agent_steps` 中）
- `mc_returns[i]`：从第 i 步到终止的折扣累积回报
- `is_last_step`：布尔标记，标识最后一步（用于提取 final reward / advantage 计算）
- `step_ids`：格式 `"{uid}_step{i}"`，用于 group-wise advantage

**返回 dict 结构**：
```python
{
    "steps": [
        {"prompt": str, "response": str, "prompt_ids": list, "completion_ids": list, "logprobs": ...},
        ...   # 一个 step 一个 dict
    ],
    "trajectory_reward": float,
    "mc_returns": [float, ...],    # len == len(steps)
    "idx": int,
    "termination_reason": str,     # "ENV_DONE" / "MAX_STEPS" / "TRUNCATION" / ...
}
```

**1 trajectory → N 训练样本**（N = 步数）

---

## Stepwise Advantage 两种子模式

启用 `stepwise_advantage` 后，还有两种子策略：

| 子模式 | 说明 |
|--------|------|
| `per_step` | 每步用各自的 `mc_return` 作为 reward 独立更新，uid 替换为 `step_id` |
| `broadcast` | 只用最后一步计算 advantage，再广播给同 trajectory 的所有步骤 |

---

## 选择依据

```yaml
# config 示例
rllm:
  stepwise_advantage:
    enable: false    # true → Step 模式，false → Token 模式
    mode: "per_step" # 或 "broadcast"
    normalize_by_steps: false
```

- **默认（Token）**：适合整轨迹 GRPO/REINFORCE，实现简单，credit assignment 集中在末尾。
- **Step 模式**：适合长 horizon、多步工具调用场景，每步都有梯度信号，credit assignment 更细粒度。
