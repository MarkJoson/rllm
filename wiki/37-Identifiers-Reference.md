# rLLM 标识符参考手册

本文档梳理 rLLM 代码库中各种标识符（ID / Key）的含义、生成方式、格式规范和使用场景。

---

## 标识符层级总览

rLLM 的数据流围绕以下层次结构组织：

```
task_id          ← 一个「训练样本」的全局唯一标识
 └─ episode_id   ← 一次「完整 rollout」（含重试编号）
     └─ trajectory_id  ← episode 内一条「智能体轨迹」
         └─ step_id    ← 轨迹内一个「推理步骤」
```

辅助标识符：

| 标识符 | 所属模块 | 说明 |
|--------|----------|------|
| `uid` (batch) | AgentPPOTrainer | `task_id` 的别称，用于 GRPO rejection sampling |
| `application_id` | AgentExecutionEngine | 单次 rollout 的请求 ID（透传给 RolloutEngine） |
| `session_name` | AgentSdkEngine | 格式与 `episode_id` 相同，用于 LiteLLM proxy 日志追踪 |
| `session_uid` | AgentSdkEngine | LiteLLM proxy 返回的 SQLite 持久化追踪 ID |
| `batch_id` | 所有 Verl 引擎 | DataProto 中每行的随机 UUID（调试用途） |
| `group_id` | TrajectoryGroup | advantage 计算分组标识 |

---

## 详细说明

### 1. `task_id`

| 属性 | 内容 |
|------|------|
| **含义** | 数据集中一个训练样本的永久性唯一标识。同一道题在多次 rollout 中共享同一个 `task_id`。 |
| **格式** | UUID4 字符串，例如 `"a1b2c3d4-..."` |
| **生成位置** | `AgentPPOTrainer`（`agent_ppo_trainer.py`）、`AgentWorkflowTrainer`、`AgentSdkTrainer` 的 `fit()` 方法内，每次从数据集取出 batch 时动态生成：`np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))])` |
| **存储位置** | `DataProto.non_tensor_batch["uid"]`（PPO Trainer）/ `DataProto.non_tensor_batch["task_ids"]`（Workflow/SDK Trainer） |
| **引用位置** | `AgentWorkflowEngine.execute_tasks_verl`、`AgentSdkEngine.execute_tasks_verl` 读取 `task_ids` 并传入引擎；Rejection Sampling 按 `uid` 对同组 rollout 结果分组后筛选 |

> **注意**：`task_id` 在 `AgentPPOTrainer` 中写入字段名为 `"uid"`，在 Workflow/SDK Trainer 中写入 `"task_ids"`，二者语义等价。

---

### 2. `episode_id`

| 属性 | 内容 |
|------|------|
| **含义** | 一次完整 rollout 的唯一标识，包含 task 信息、rollout 序号和重试次数。 |
| **格式** | `"{task_id}:{rollout_idx}:{retry_attempt}"` 例如 `"abc-123:0:1"` |
| **生成位置** | `AgentSdkEngine._execute_tasks()`，在每次成功完成 rollout 后：`episode = Episode(id=session_name, ...)` |
| **基类定义** | `rllm/types.py` 中 `Episode.id`，默认为 `str(uuid.uuid4())` |
| **解析方法** | `episode.task_id` → `id.split(":")[0]`；`episode.rollout_idx` → `id.split(":")[1]` |
| **存储位置** | `DataProto.non_tensor_batch["episode_ids"]` |
| **引用位置** | Trainer 内去重统计 episode 指标（`seen_episodes` set）；`propagate_reward_to_last_step()` 中按 `episode_id` 对齐最后一步奖励 |

> **Workflow Engine**：`AgentWorkflowEngine` 中 episode.id 的格式为 `"{task_id}:{rollout_idx}"`（无重试编号），通过 `Workflow.run_with_termination_handling(uid=uid)` 传入。

---

### 3. `trajectory_id`

| 属性 | 内容 |
|------|------|
| **含义** | episode 内一条具名轨迹（例如 Solver-Judge 模式中的 `solver` 或 `judge`）的唯一标识。 |
| **格式** | `"{task_id}_{trajectory_name}"` 例如 `"abc-123_solver"` |
| **生成位置** | `AgentWorkflowEngine.transform_results_for_verl()` 和 `AgentSdkEngine.transform_results_for_verl()` 中：`trajectory_id = f"{task_ids[i]}_{name}"` |
| **名称来源** | `trajectory.name`，由 `Trajectory(name=...)` 在 Workflow 内设置（默认值 `"default_traj_name"` 或 `"agent"`） |
| **基类定义** | `rllm/types.py` 中 `Trajectory.uid`，默认 UUID4（与 `trajectory_id` 不同，后者是训练侧拼接的字符串） |
| **存储位置** | `DataProto.non_tensor_batch["trajectory_ids"]`，按步骤展开（每步一行共享同一 `trajectory_id`） |
| **引用位置** | `propagate_reward_to_last_step()`：通过 `trajectory_ids` 匹配 last-step 奖励并广播到非最后步；Rejection Sampling 中用 `trajectory_ids` 过滤无效的非最后步 |

---

### 4. `step_id`

| 属性 | 内容 |
|------|------|
| **含义** | 多步轨迹中单个推理步骤的唯一标识。启用 `stepwise_advantage` 时，每步独立成一行。 |
| **格式（stepwise）** | `"{trajectory_id}_step{step_idx}"` 例如 `"abc-123_solver_step0"` |
| **格式（非 stepwise）** | 等同于 `trajectory_id`（整条轨迹视为一步） |
| **生成位置** | `transform_results_for_verl()` 中：`step_ids.append(f"{trajectory_id}_step{step_idx}")` |
| **基类定义** | `rllm/types.py` 中 `Step.id`，默认 UUID4（traced via LiteLLM proxy，与训练侧 step_id 独立） |
| **存储位置** | `DataProto.non_tensor_batch["step_ids"]` |
| **引用位置** | `AgentPPOTrainer._transform_agent_steps()` 中：`step_ids = [f"{uids[idx]}_step{i}" for i in range(len(episode_steps))]`；之后 `batch.non_tensor_batch["uid"] = batch.non_tensor_batch["step_ids"]` 以便 stepwise advantage 按步聚合 |

---

### 5. `uid`（DataProto 字段）

| 属性 | 内容 |
|------|------|
| **含义** | `AgentPPOTrainer` 中用于 GRPO rejection sampling 的分组键。训练初期等于 `task_id`，stepwise 模式开启后被替换为 `step_id`。 |
| **格式** | 与 `task_id` 或 `step_id` 相同 |
| **切换位置** | `agent_ppo_trainer.py` 第 391 行：`batch.non_tensor_batch["uid"] = batch.non_tensor_batch["step_ids"]`（stepwise_advantage 开启时） |
| **引用位置** | `generate_agent_steps(uids=...)` 透传；Rejection Sampling 阶段用 `np.unique(uids)` 来对同一任务的多个 rollout 做统计 |

#### `uid_rewards` 的全零 / 全一与 Rejection Sampling

在 `AgentPPOTrainer.fit_agent()` 的 Rejection Sampling 阶段（`agent_ppo_trainer.py` 约 256–266 行），针对每个 `uid` 组计算：

```python
uid_rewards = reward_tensor[uid_mask].sum(-1)
# reward_tensor shape: [Batch, MaxTokens]
# 每条序列的 reward 仅放在最后一个有效 response token，
# 其余位置为 0，因此 sum(-1) == trajectory reward 标量
```

| 条件 | 含义 | 处理 | Metric |
|------|------|------|--------|
| `(uid_rewards <= 0).all()` | 同一题的 **所有 n 次 rollout 全部失败**（reward ≤ 0） | 整组丢弃 | `batch/solve_none` |
| `(uid_rewards >= 1).all()` | 同一题的 **所有 n 次 rollout 全部成功**（reward ≥ 1） | 整组丢弃 | `batch/solve_all` |
| 其他（混合）| 有成功有失败，组内 reward 存在方差 ✅ | **保留** | `batch/solve_partial` |

**为什么丢弃全零/全一组？**

- GRPO 等算法用组内 reward 差异估计 advantage；若同一组所有样本 reward 相同，advantage 归一化后为 0，梯度为 0，保留此组既浪费算力又引入噪声。
- 全零 → 模型对该题毫无解题能力，暂时跳过；全一 → 已完全掌握，无需再强化。
- 只有 `solve_partial`（部分正确）的样本才携带有效学习信号。

---

### 6. `application_id`

| 属性 | 内容 |
|------|------|
| **含义** | `AgentExecutionEngine`（旧版执行引擎）中透传给 RolloutEngine 的请求标识。用于 vLLM 等引擎的请求路由和日志追踪。 |
| **格式** | UUID4 字符串 |
| **生成位置** | `agent_execution_engine.py`：`application_id = str(uuid.uuid4())` |
| **引用位置** | 内部传递给 `get_model_response(prompt, application_id=..., ...)`；OpenAI/Verl Engine 在收到后可忽略（`kwargs.pop("application_id", None)`）或用于追踪 |

---

### 7. `session_name` / `session_uid`

| 属性 | 内容 |
|------|------|
| **`session_name` 含义** | AgentSdkEngine 中传给 LiteLLM proxy 的会话名称，用于将 LLM 调用日志与特定 rollout 关联。 |
| **`session_name` 格式** | `"{task_id}:{rollout_idx}:{retry_attempt}"` （与 `episode_id` 完全一致） |
| **`session_uid` 含义** | LiteLLM proxy / OpenTelemetry 返回的持久化存储 UID，用于从 SQLite 检索 Trace 记录。 |
| **`session_uid` 生成** | 由 `wrapped_agent_run_func()` 的返回值提供（即 prompt tracer 记录后的内部 ID） |
| **引用位置** | `_execute_tasks()` 用 `session_uid` 查询 SQLite：`store.get_by_session_uid(session_uid, since=...)`；用 `session_name` 反查 Trace 并组装 Episode |

---

### 8. `batch_id`

| 属性 | 内容 |
|------|------|
| **含义** | DataProto 中每一行的随机唯一标识，主要供调试和日志使用，不参与训练计算。 |
| **格式** | UUID4 字符串（每行独立） |
| **生成位置** | `transform_results_for_verl()` 中：`np.array([str(uuid.uuid4())] * len(episode_ids))` |
| **存储位置** | `DataProto.non_tensor_batch["batch_ids"]` |

---

### 9. `group_id`

| 属性 | 内容 |
|------|------|
| **含义** | `TrajectoryGroup` 的标识，用于 advantage 计算时将多条轨迹归为同一组（以便 GRPO 等算法做组内比较）。 |
| **格式** | `"{task_id}:{agent_role}"` 例如 `"abc-123:solver"`；或直接为 `task_id` 字符串 |
| **生成位置** | `rllm/experimental/common/transform.py`：`TrajectoryGroup(group_id=name, ...)` |
| **属性解析** | `group.group_role` → `group_id.split(":")[1]`；`group.task_id` → `group_id.split(":")[0]` |
| **引用位置** | `experimental/verl/transform.py`：`group_id = trajectory_group.group_id if ... else task_id`，然后作为 `episode_ids` 使用；`utils/tracking.py` 记录日志 |

---

## 标识符生成时序图

```
Dataset (DataProto)
  └─ non_tensor_batch["task_ids"] = [uuid4(), ...uuid4()]   ← Trainer.fit()
         │
         ▼
AgentWorkflowEngine / AgentSdkEngine.execute_tasks_verl()
  └─ task_ids → execute_tasks(tasks, task_ids)
       │
       ├─ per rollout: uid = f"{task_id}:{rollout_idx}:{retry}"
       │                      ↑ also used as "session_name" in SDK Engine
       │
       └─ Episode(id = session_name, ...)
            └─ episode_id = "{task_id}:{rollout_idx}:{retry}"

transform_results_for_verl()
  ├─ trajectory_id = f"{task_ids[i]}_{trajectory.name}"
  ├─ step_id       = f"{trajectory_id}_step{step_idx}"    (stepwise mode)
  │               OR  = trajectory_id                      (non-stepwise)
  └─ batch_id      = uuid4()   (per row)
```

---

## 各引擎的差异对比

| 引擎 | task_id 字段名 | episode_id 格式 | step_id 格式 |
|------|----------------|-----------------|--------------|
| `AgentWorkflowEngine` | `"task_ids"` | `"{task_id}:{rollout_idx}"` | `"{traj_id}_step{i}"` (stepwise) |
| `AgentSdkEngine` | `"task_ids"` | `"{task_id}:{rollout_idx}:{retry}"` | `"{traj_id}_step{i}"` (stepwise) |
| `AgentPPOTrainer`（旧执行引擎）| `"uid"` | N/A | `"{uid}_step{i}"` |

---

## 关键源码位置

| 标识符 | 主要定义 | 主要消费 |
|--------|----------|----------|
| `task_id` / `uid` | `agent_ppo_trainer.py:179`, `agent_workflow_trainer.py:181` | `agent_workflow_engine.py:220`, `agent_sdk_engine.py:454` |
| `episode_id` | `types.py:57`, `agent_sdk_engine.py:375` | `agent_sdk_trainer.py:640`, `agent_workflow_trainer.py:823` |
| `trajectory_id` | `agent_workflow_engine.py:295`, `agent_sdk_engine.py:513` | `agent_workflow_trainer.py:291–293`, `agent_sdk_trainer.py:311–313` |
| `step_id` | `agent_workflow_engine.py:373`, `agent_sdk_engine.py:586` | `agent_ppo_trainer.py:391` |
| `batch_id` | `agent_workflow_engine.py:479`, `agent_sdk_engine.py:700` | 仅调试日志 |
| `application_id` | `agent_execution_engine.py:556` | `agent_execution_engine.py:156–164` |
| `session_name` | `agent_sdk_engine.py:174` | SQLite Trace 查询 `agent_sdk_engine.py:307–311` |
| `group_id` | `agents/agent.py:260` | `experimental/verl/transform.py:306` |
