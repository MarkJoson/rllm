# rLLM 统计打点参考文档

> 梳理 rLLM 代码库中所有统计打点（metrics）的位置、含义、计算方式及特殊处理逻辑。

---

## 目录

1. [打点基础设施（Tracking 系统）](#1-打点基础设施)
2. [时序打点（Timing Mixin）](#2-时序打点)
3. [批次训练打点（batch/）](#3-批次训练打点)
4. [验证打点（val/）](#4-验证打点)
5. [Actor/Critic 模型打点（actor/ critic/）](#5-actorcritic-模型打点)
6. [拒绝采样打点（rejection_sample/）](#6-拒绝采样打点)
7. [全异步训练打点（fully_async/）](#7-全异步训练打点)
8. [滚动生成打点（rollouter/）](#8-滚动生成打点)
9. [奖励分解打点（reward/）](#9-奖励分解打点)
10. [蒸馏打点（val/distill/）](#10-蒸馏打点)
11. [自定义工作流打点（workflow metrics）](#11-自定义工作流打点)
12. [性能 / 吞吐打点（perf/ timing_s/）](#12-性能--吞吐打点)
13. [打点聚合规则（MetricsAggregator）](#13-打点聚合规则)

---

## 1. 打点基础设施

### 文件

[`rllm/utils/tracking.py`](file:///home/robomaster/Research/rllm/rllm/utils/tracking.py)

### `Tracking` 类

统一的多后端日志接口，支持以下后端：

| 后端 | 说明 |
|---|---|
| `wandb` | Weights & Biases |
| `mlflow` | MLflow（key 中的 `@` 替换为 `_at_`，其他非法字符替换为 `_`） |
| `swanlab` | SwanLab（国产 W&B 替代） |
| `vemlp_wandb` | 火山引擎 ML 平台 W&B |
| `tensorboard` | TensorBoard |
| `console` | 直接打印到控制台（调试用） |
| `clearml` | ClearML（key 按 `title/series` 分组） |
| `trackio` | Trackio |
| `file` | 写入 JSONL 文件（路径由 `VERL_FILE_LOGGER_PATH` 或 `VERL_FILE_LOGGER_ROOT` 控制） |
| `ui` | rLLM UI 在线监控（异步 HTTP 推送，会同时上传 episode 和 trajectory group） |

### 调用方式

```python
logger.log(data: dict, step: int, episodes=None, trajectory_groups=None)
```

- `data`：指标字典，所有 key 应遵循 `category/subcategory` 命名规范。
- `step`：全局训练步数（`global_steps` 或 `current_param_version`）。
- `episodes` / `trajectory_groups`：仅 `UILogger` 使用，用于上传可视化轨迹数据。

### `UILogger` 特殊处理

- 初始化时会替换 `sys.stdout` / `sys.stderr` 为 `TeeStream`，将打印内容同步转发到 UI 后端（每 20 行或 2 秒 flush 一次）。
- 通过异步队列（`queue.Queue(maxsize=64)`）保护训练不被 HTTP 阻塞；队列满时**丢弃**日志（不阻塞训练）。
- session 级别发送 heartbeat（每 30s），会话结束时 POST `/api/sessions/{session_id}/complete`。
- 上传 episode 时，自动剥离 `prompt_ids`、`response_ids`、`logprobs`、`model_output` 等大字段，仅保留可视化所需的信息。

---

## 2. 时序打点

### 文件

[`rllm/workflows/timing_mixin.py`](file:///home/robomaster/Research/rllm/rllm/workflows/timing_mixin.py)

### `TimingTrackingMixin` — episode/step 级别计时

| 字段 | 含义 | 层级 |
|---|---|---|
| `llm_time` | LLM 推理累计耗时（s） | episode / step |
| `env_time` | 环境交互累计耗时（s） | episode / step |
| `reward_time` | 奖励计算累计耗时（s） | episode |
| `total_time` | episode 总耗时（s） | episode |
| `start_timestamp` | ISO 8601 开始时间戳 | episode / trajectory / step |
| `end_timestamp` | ISO 8601 结束时间戳 | episode / trajectory / step |

**存放位置：**
- `episode.info["timing"]` — episode 级别
- `trajectory.info["timing"]` — trajectory 级别
- `step.info["timing"]` — step 级别（每次 `timed_llm_call()` 触发新 step 计时）

**特殊说明：**
- `TimingTrackingMixin.timed_llm_call()` 包装每次 LLM 请求，同时触发新 step 计时。
- 若 step 数超过实际采集到的 `_step_timings` 数量，后续 step 的时间会填 0。

---

## 3. 批次训练打点

### 文件

[`rllm/trainer/verl/agent_workflow_trainer.py`](file:///home/robomaster/Research/rllm/rllm/trainer/verl/agent_workflow_trainer.py)（主干）  
[`rllm/trainer/verl/agent_sdk_trainer.py`](file:///home/robomaster/Research/rllm/rllm/trainer/verl/agent_sdk_trainer.py)  
[`rllm/trainer/verl/agent_ppo_trainer_pipeline.py`](file:///home/robomaster/Research/rllm/rllm/trainer/verl/agent_ppo_trainer_pipeline.py)

### 每步打点（每个 `global_steps` 上报一次）

| Metric Key | 含义 | 计算方式 | 特殊说明 |
|---|---|---|---|
| `training/global_step` | 当前全局步数 | 直接赋值 | — |
| `training/epoch` | 当前 epoch | 直接赋值 | — |
| `batch/solve_none` | 全失败 task 占比 | `solve_none / num_tasks` | 该 task 所有 rollout 均错误 → 拒绝采样中丢弃 |
| `batch/solve_all` | 全正确 task 占比 | `solve_all / num_tasks` | 该 task 所有 rollout 均正确 → 拒绝采样中丢弃 |
| `batch/solve_partial` | 部分正确 task 占比 | `solve_partial / num_tasks` | 参与训练的有效批次 |
| `batch/num_tasks` | 本步实际 task 数 | 累计计数 | 包含被拒绝采样丢弃的 tasks |
| `batch/{termination_reason}` | 各终止原因占比 | `termination_counts[reason] / num_tasks` | `TerminationReason` 枚举值（如 `max_steps`, `tool_error` 等） |
| `batch/{workflow_metric}` | 工作流自定义指标均值 | `np.mean(workflow_metrics[key])` | 由各 Workflow 返回，通过 `episode.metrics` 传递 |
| `actor/entropy` | Actor 响应熵 | 聚合后的 entropy 均值（`agg_loss`） | 重计算 old_log_prob 时采集 |

**注意：** 在 Pipeline 模式（`PipelineAgentPPOTrainer`），`batch/solve_*` 以**绝对数量**（非占比）上报，并对列表值取 `sum`（`"batch/"` 前缀）或 `mean`（其他）。

---

## 4. 验证打点

### 文件

[`rllm/trainer/verl/agent_workflow_trainer.py`](file:///home/robomaster/Research/rllm/rllm/trainer/verl/agent_workflow_trainer.py) `_validate_agent()`  
[`rllm/experimental/fully_async/rollout_executor.py`](file:///home/robomaster/Research/rllm/rllm/experimental/fully_async/rollout_executor.py) `validate()`

### 同步训练验证打点

| Metric Key | 含义 | 计算方式 |
|---|---|---|
| `val/{data_source}/pass@1` | 该数据源的 pass@1 准确率 | `np.mean(is_correct_array[mask])` |
| `val/{data_source}/pass@{n}` | 该数据源的 pass@n 准确率 | `n` = `val_kwargs.n`；只要该 task 有一条正确即为 pass |
| `val/{data_source}/{workflow_metric}` | 工作流在验证集的自定义指标均值 | 按数据源分桶后取均值 |
| `val/test_score/{data_source}` | (Pipeline) 验证奖励均值 | `np.mean(rewards_for_source)` |
| `val/env_score/{data_source}` | (Pipeline) 环境原始分均值 | `np.mean(env_rewards_for_source)` |

**特殊处理：**  
验证时对"被 engine 提前丢弃的 episode"（`dropped_episodes`）补充 `is_correct=False`，避免 pass@k 被高估。

### 全异步训练验证打点（FullyAsync）

| Metric Key | 含义 |
|---|---|
| `val/avg_reward` | 验证集平均奖励 |
| `val/num_samples` | 验证集样本数 |
| `val/{user_key}` | 用户 `val_rollout_fn` 返回 metadata 中的数值指标均值 |
| `timing_s/validation` | 验证耗时（s） |
| `val/validated_version` | 实际验证所用的参数版本号（与 log step 可能不同） |

---

## 5. Actor/Critic 模型打点

### 来源

由 verl 的 `update_actor` / `update_critic` 方法产生，通过 `reduce_metrics(output.meta_info["metrics"])` 汇入。

### 常见 Key（由 verl 定义）

| Metric Key | 含义 |
|---|---|
| `actor/pg_loss` | PPO clip loss |
| `actor/pg_clipfrac` | clipped ratio 占比 |
| `actor/entropy` | Actor 输出熵 |
| `actor/kl` | KL 散度（若使用 KL 惩罚） |
| `critic/vf_loss` | Critic value function loss |
| `critic/vf_clipfrac` | Critic clip fraction |
| `actor/grad_norm` | Actor 梯度范数 |
| `critic/grad_norm` | Critic 梯度范数 |

**特殊说明：**  
在全异步场景（`FullyAsyncTrainer`），使用 `reduce_metrics_with_flatten()` 代替 `reduce_metrics()`，因各 worker 上报的可能是**嵌套列表**（variable micro-batching 导致），需先 flatten 再聚合。

---

## 6. 拒绝采样打点

### 文件

[`rllm/experimental/fully_async/utils.py`](file:///home/robomaster/Research/rllm/rllm/experimental/fully_async/utils.py) `apply_rejection_sampling()`

### 打点 Key

| Metric Key | 含义 | 聚合方式 |
|---|---|---|
| `rejection_sample/solve_none` | 全失败 task 组数 | sum（跨步）|
| `rejection_sample/solve_all` | 全正确 task 组数 | sum |
| `rejection_sample/solve_partial` | 部分正确 task 组数 | sum |
| `rejection_sample/total_groups` | 总 task 组数 | sum |
| `rejection_sample/filtered_groups` | 过滤后保留的 task 组数 | sum |
| `rejection_sample/enabled` | 是否启用拒绝采样 | last |
| `rejection_sample/unfiltered_reward_sum` | 过滤前奖励总和 | sum（用于加权均值） |
| `rejection_sample/unfiltered_reward_count` | 过滤前轨迹数量 | sum（用于加权均值） |
| `rejection_sample/unfiltered_reward_mean` | 过滤前平均奖励 | 由 sum/count 计算 |
| `rejection_sample/unfiltered_reward_std` | 过滤前奖励标准差 | avg |
| `rejection_sample/unfiltered_reward_min/max` | 过滤前奖励最值 | min/max |
| `rejection_sample/unfiltered_num_trajectories` | 过滤前轨迹总数 | sum |
| `rejection_sample/filtered_reward_sum/count` | 过滤后奖励总和/数量 | sum（用于加权均值） |
| `rejection_sample/filtered_reward_mean` | 过滤后平均奖励 | 由 sum/count 计算 |
| `rejection_sample/filtered_reward_std` | 过滤后奖励标准差 | avg |
| `rejection_sample/filtered_num_trajectories` | 过滤后轨迹数 | sum |
| `rejection_sample/all_filtered_fallback` | 是否触发"全过滤兜底"逻辑 | last（0 或 1）|

**计算细节：**  
`_mean` 不直接聚合 `mean` 值（可能有 sample count 不一致问题），而是存储 `_sum` 和 `_count` 分别累加，再由 `MetricsAggregator._special_metrics_aggergate` 计算加权均值：
```
filtered_reward_mean = filtered_reward_sum / filtered_reward_count
```

---

## 7. 全异步训练打点

### 文件

[`rllm/experimental/fully_async/fully_async_trainer.py`](file:///home/robomaster/Research/rllm/rllm/experimental/fully_async/fully_async_trainer.py) `_collect_metrics_from_samples()`  
[`rllm/experimental/fully_async/utils.py`](file:///home/robomaster/Research/rllm/rllm/experimental/fully_async/utils.py) `assemble_batch_from_trajectory_group_ls()`

### 样本处理打点

| Metric Key | 含义 | 聚合方式 |
|---|---|---|
| `fully_async/count/total_generated_samples` | 累计已生成样本总数 | last |
| `fully_async/count/stale_samples_processed` | 累计过期样本数（参数版本落后）| last |
| `fully_async/count/stale_trajectory_processed` | 累计过期轨迹数 | last |
| `fully_async/count/current_param_version` | 当前参数版本号 | last |
| `fully_async/count/dropped_stale_samples` | 被丢弃的过期样本数 | last |
| `fully_async/total_wait_time` | 从队列取样的总等待时间（s） | avg |

### 异步处理耗时打点

| Metric Key | 含义 | 来源 |
|---|---|---|
| `fully_async/processing_time/avg` | 轨迹处理时间均值 | `trajectory.metadata["processing_time"]` |
| `fully_async/processing_time/max` | 处理时间最大值 | — |
| `fully_async/processing_time/min` | 处理时间最小值 | — |
| `fully_async/processing_time/tp50` | 处理时间 P50 | `np.percentile` |
| `fully_async/processing_time/tp95` | 处理时间 P95 | — |
| `fully_async/processing_time/tp99` | 处理时间 P99 | — |

### 参数版本追踪（部分 rollout）打点

| Metric Key | 含义 |
|---|---|
| `fully_async/partial/total_partial_num` | 本批中跨参数版本轨迹数（param_version_start ≠ end）|
| `fully_async/partial/partial_ratio` | 跨版本轨迹比例 |
| `fully_async/partial/max_partial_span` | 最大版本跨度 |

### Tool Call 耗时打点

| Metric Key | 含义 | 聚合方式 |
|---|---|---|
| `timing_s/agent_loop/tool_calls/max` | tool 调用最大耗时（s）| max |
| `timing_s/agent_loop/tool_calls/min` | tool 调用最小耗时（s）| min |
| `timing_s/agent_loop/tool_calls/mean` | tool 调用平均耗时（s）| avg |

**数据来源：** `trajectory.metadata["tool_calls_time"]`，由各 rollout 函数在完成后写入 trajectory 的 metadata。

---

## 8. 滚动生成打点

### 文件

[`rllm/experimental/fully_async/rollout_executor.py`](file:///home/robomaster/Research/rllm/rllm/experimental/fully_async/rollout_executor.py) `update_param_version()`

| Metric Key | 含义 | 计算方式 |
|---|---|---|
| `rollouter/active_time` | 本版本周期中 rollouter 活跃生成的时长（s）| `idle_start_time - version_start_time` |
| `rollouter/version_time` | 本版本周期总时长（s）| `now() - version_start_time` |
| `rollouter/idle_ratio` | rollouter 空闲时间占比 | `1 - active_time / version_time` |

**打点时机：** 每次参数同步（`sync_weights`）时，`ParameterSynchronizer` 调用 `RolloutExecutor.update_param_version()`。

**参数同步计时：**
| Key | 含义 |
|---|---|
| `timing_s/wait_last_valid` | 等待上一轮验证完成的时长（s）|
| `timing_s/param_sync` | 本次参数同步耗时（s）|

---

## 9. 奖励分解打点

### 文件

[`rllm/experimental/common/metrics.py`](file:///home/robomaster/Research/rllm/rllm/experimental/common/metrics.py)  
[`rllm/experimental/unified_trainer.py`](file:///home/robomaster/Research/rllm/rllm/experimental/unified_trainer.py)

### `reduce_metrics_by_trajectory_name()`

| Metric Key | 含义 |
|---|---|
| `{prefix}/{traj_name}/mean` | 该轨迹类型的奖励均值 |
| `{prefix}/{traj_name}/max` | 最大奖励 |
| `{prefix}/{traj_name}/min` | 最小奖励 |
| `{prefix}/{traj_name}/std` | 奖励标准差 |
| `{prefix}/{traj_name}/fraction_zero` | 奖励为 0 的比例（仅 `include_fraction_zero=True` 时启用）|

> `prefix` 默认为 `"reward"`，`traj_name` 来自 `trajectory.name`（如 `"solver"` / `"judge"` 等）。

---

## 10. 蒸馏打点

### 文件

[`rllm/trainer/verl/agent_workflow_trainer.py`](file:///home/robomaster/Research/rllm/rllm/trainer/verl/agent_workflow_trainer.py) `_validate_agent()`

仅在 `distill_enabled=True` 且验证结束时上报：

| Metric Key | 含义 |
|---|---|
| `val/distill/mean_advantage` | Teacher-Student log prob 差均值（即反向 KL per token）|
| `val/distill/std_advantage` | 差的标准差 |
| `val/distill/min_advantage` | 最小差值 |
| `val/distill/max_advantage` | 最大差值 |

**计算方式：**  
`advantage[i] = teacher_logprob[i] - student_logprob[i]`，仅在 `response_mask==True` 的 token 上计算。

---

## 11. 自定义工作流打点

### 传递链路

```
Workflow.run() → episode.metrics[key] = value
→ AgentExecutionEngine (汇总) → non_tensor_batch["metrics"]
→ Trainer: workflow_metrics[key].append(value)
→ metrics[f"batch/{key}"] = np.mean(workflow_metrics[key])
```

### 全异步模式的自定义 trajectory 指标

在 trajectory 的 `metadata` 中以 `custom/` 前缀存储的任意数值指标，会被自动提取并上报：

```python
# 在 rollout_fn 中
trajectory.metadata["custom/my_metric"] = 0.5
```

上报 Key：
| Key | 含义 |
|---|---|
| `custom/{name}/avg` | 该批次该指标的均值 |
| `custom/{name}/max` | 最大值 |
| `custom/{name}/min` | 最小值 |

---

## 12. 性能 / 吞吐打点

### 来源

由 verl 的 `compute_timing_metrics()` 和 `compute_throughout_metrics()` 产生。

| Metric Key | 含义 |
|---|---|
| `timing_s/step` | 完整训练步耗时 |
| `timing_s/gen` | 轨迹生成耗时 |
| `timing_s/old_log_prob` | 重计算 old log prob 耗时 |
| `timing_s/ref` | Ref policy 计算耗时 |
| `timing_s/values` | Critic values 计算耗时 |
| `timing_s/adv` | Advantage 计算耗时 |
| `timing_s/update_actor` | Actor 更新耗时 |
| `timing_s/update_critic` | Critic 更新耗时 |
| `timing_s/testing` | 验证耗时 |
| `timing_s/save_checkpoint` | checkpoint 保存耗时 |
| `perf/throughput` | token/s/GPU（特殊聚合：`total_tokens / time / gpus`）|
| `perf/total_num_tokens` | 批次 token 总量 |
| `perf/time_per_step` | 每步耗时（用于吞吐计算）|
| `trainer/idle_ratio` | Trainer 空闲时间比（`gen_time / step_time`）|
| `global_seqlen/max` | 批次最大序列长度 |
| `global_seqlen/min` | 批次最小序列长度 |
| `global_seqlen/minmax_diff` | 最大最小差（由 `MetricsAggregator` 派生）|

---

## 13. 打点聚合规则

### 文件

[`rllm/experimental/fully_async/metric_utils.py`](file:///home/robomaster/Research/rllm/rllm/experimental/fully_async/metric_utils.py) `MetricsAggregator`

全异步训练中，`MetricsAggregator` 会跨多个 trigger step 聚合指标，然后在每次参数同步时上报。聚合规则如下：

| 规则 | 适用 Key 或模式 | 行为 |
|---|---|---|
| `last` | `fully_async/count/*`, `training/global_step` | 取最新值 |
| `time_sum` | 含 `timing_s/` 的 key | 累加（总时长）|
| `avg` | 含 `mean`/`avg`/`average` 的 key，默认 | 算术平均 |
| `max` | 含 `max`/`maximum` 的 key | 取最大值 |
| `min` | 含 `min`/`minimum` 的 key | 取最小值 |
| `sum` | `rejection_sample/*_sum`, `rejection_sample/*_count` | 累加（用于后续加权均值计算）|
| `weighted_avg` | 含 `weighted_avg` 的 key | 按 sample_count 加权平均 |

**派生指标（完成聚合后计算）：**
- `global_seqlen/minmax_diff = max - min`
- `perf/throughput = total_num_tokens / (time_per_step * total_gpus)`
- `trainer/idle_ratio = timing_s/gen / timing_s/step`
- `rejection_sample/unfiltered_reward_mean = sum / count`
- `rejection_sample/filtered_reward_mean = sum / count`

---

## 附录：Metric Key 命名规范

| 前缀 | 含义 |
|---|---|
| `training/` | 训练进度（step、epoch）|
| `batch/` | 本步批次统计（拒绝采样、终止原因、工作流指标）|
| `val/` | 验证集指标 |
| `actor/` | Actor 模型更新指标 |
| `critic/` | Critic 模型更新指标 |
| `rejection_sample/` | 拒绝采样统计 |
| `fully_async/` | 全异步训练统计 |
| `rollouter/` | Rollout Executor 计时 |
| `timing_s/` | 각 阶段耗时（秒）|
| `perf/` | 性能/吞吐指标 |
| `custom/` | 用户自定义 trajectory 指标 |
| `reward/` | 奖励分解（按轨迹名称）|
| `val/distill/` | 蒸馏验证指标 |
