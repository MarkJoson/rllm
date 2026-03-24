# 工作流引擎 (Workflow Engine)

`AgentWorkflowEngine` 编排 RL 训练期间多 Agent 工作流的执行。它维护 Workflow 实例池以实现并行任务处理，处理重试逻辑和终止条件，并将 Episode 结果转换为 verl 训练后端兼容的 `DataProto` 格式。

> 源码参考：
> - [rllm/engine/agent_workflow_engine.py L28-556](../rllm/engine/agent_workflow_engine.py)
> - [rllm/workflows/workflow.py L32-291](../rllm/workflows/workflow.py)

## 架构概览

```
AgentWorkflowEngine [agent_workflow_engine.py L28-556]
├── workflow_cls: type[Workflow]         # 工作流类（用于池化实例化）
├── workflow_args: dict                  # 工作流参数
├── rollout_engine: RolloutEngine        # 推理引擎（所有 workflow 共享）
├── workflow_queue: asyncio.Queue        # Workflow 实例池
├── executor: ThreadPoolExecutor         # 环境操作线程池
├── config: OmegaConf                    # Hydra 训练配置
├── episode_logger: EpisodeLogger|None   # Episode 日志记录器
│
├── initialize_pool()                    # 创建 workflow 实例池 [L75-88]
├── process_task_with_retry()            # 单任务重试逻辑 [L90-141]
├── execute_tasks()                      # 批量异步执行 [L143-199]
├── execute_tasks_verl()                 # verl 集成入口 [L201-227]
├── transform_results_for_verl()         # Episode → DataProto [L229-514]
└── _handle_multimodal_position_ids()    # VLM position_ids [L516-549]
```

## Workflow 基类详解

### 初始化参数

`Workflow.__init__()` ([workflow.py L33-48](../rllm/workflows/workflow.py#L33-L48))：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rollout_engine` | `RolloutEngine` | 必填 | 所有 LLM 调用通过此引擎 |
| `executor` | `ThreadPoolExecutor` | 必填 | 同步环境操作的线程池 |
| `timeout` | `float` | `1e6` | 单次 `run()` 超时（秒） |
| `gamma` | `float` | `0.0` | Monte Carlo return 折扣因子（0=不折扣） |
| `reward_bonus_coeff` | `float` | `0.0` | 差分奖励 bonus 系数（0=不启用） |

### 核心方法

#### `run()` — 用户实现的核心逻辑

```python
@abstractmethod
async def run(self, task: dict, uid: str, **kwargs) -> Episode | None:
    """
    用户必须实现的方法。接收一个任务 dict 和唯一标识符，
    返回 Episode 或 None（None 时由框架自动收集 trajectories）。
    """
```

#### `run_with_termination_handling()` — 包装器

([workflow.py L66-91](../rllm/workflows/workflow.py#L66-L91))

```python
async def run_with_termination_handling(self, task, uid, **kwargs) -> Episode:
    timeout = kwargs.pop("timeout", self.timeout)
    try:
        coro = self.run(task, uid, **kwargs)
        output = await asyncio.wait_for(coro, timeout=timeout)
        if output is not None and isinstance(output, Episode):
            return output  # 已由用户后处理
        return self.postprocess_episode(self.collect_trajectories(), TerminationReason.UNKNOWN)
    except asyncio.TimeoutError:
        return self.postprocess_episode(self.collect_trajectories(), TerminationReason.TIMEOUT)
    except TerminationEvent as e:
        return self.postprocess_episode(self.collect_trajectories(), e.reason)
    except Exception as e:
        error_details = {"error_message": str(e), "error_type": type(e).__name__,
                         "traceback": traceback.format_exc()}
        return self.postprocess_episode(self.collect_trajectories(), TerminationReason.ERROR,
                                        error=error_details)
```

> 无论 `run()` 正常返回、超时、抛出 `TerminationEvent` 还是崩溃，都能安全收集已有的 trajectory 数据。

#### `commit()` — 注册 Trajectory

([workflow.py L93-112](../rllm/workflows/workflow.py#L93-L112))

```python
def commit(self, name=None, agent=None, trajectory=None, reset=False):
    """
    将 Agent 或 Trajectory 注册到当前 episode。
    - agent 和 trajectory 互斥，只能提供一个
    - commit 后轨迹被 deepcopy 保存（防止后续修改影响）
    - reset=True 时自动重置 Agent
    """
    traj = agent.trajectory if agent is not None else trajectory
    if name: traj.name = name
    if traj.steps:
        self._completed_trajectories.append(deepcopy(traj))
    if agent is not None and reset:
        agent.reset()
```

#### `collect_trajectories()` — 自动发现 Agent

([workflow.py L114-137](../rllm/workflows/workflow.py#L114-L137))

```python
def collect_trajectories(self) -> Episode:
    episode = Episode()
    episode.trajectories.extend(self._completed_trajectories)
    completed_uids = {t.uid for t in self._completed_trajectories}

    # 自动发现 Workflow 属性中的 BaseAgent 实例
    for attr_name in dir(self):
        if attr_name.startswith("_"): continue
        attr = getattr(self, attr_name)
        if isinstance(attr, BaseAgent) and \
           attr.trajectory.uid not in completed_uids and \
           len(attr.trajectory.steps) > 0:
            episode.trajectories.append(deepcopy(attr.trajectory))
    return episode
```

> 机制：遍历 Workflow 的所有非下划线属性，找到类型为 `BaseAgent` 的属性，自动收集其轨迹。这样用户只需在 Workflow 上定义 `self.solver = MySolver()` 即可自动被收集。

### 后处理管线

`postprocess_episode()` ([workflow.py L198-238](../rllm/workflows/workflow.py#L198-L238)) 执行 7 步后处理：

```
1. episode.id = self.uid [L208]
   episode.task = self.task [L209]

2. 清理不完整 step [L211-215]
   if trajectory.steps[-1].chat_completions 为空: pop()

3. compute_trajectory_reward(trajectory) [L218]
   默认: trajectory.reward = sum(step.reward for step in steps) [L147]

4. adjust_step_rewards(trajectory) [L221-222]
   仅对多步轨迹（len(steps) > 1）调用

5. assign_episode_correctness(episode) [L225]
   默认: is_correct = (sum of trajectory rewards) > 0 [L180-183]

6. collect_metrics(episode) [L229]
   按 agent 名聚合: {f"{name}_acc": mean(rewards)} [L192-196]

7. 存储错误 + 终止原因 [L232-237]
```

### 奖励塑形

`adjust_step_rewards()` ([workflow.py L149-170](../rllm/workflows/workflow.py#L149-L170))：

**差分奖励 Bonus**（`reward_bonus_coeff > 0.0`）：
```python
# s[i].reward += bonus * (s[i].reward - s[i-1].reward)
# 鼓励逐步改进：如果 step 2 奖励比 step 1 高，给额外 bonus
for i in range(1, len(steps)):
    steps[i].reward += self.reward_bonus_coeff * (raw_rewards[i] - raw_rewards[i-1])
```

**Monte Carlo Return**（`gamma > 0.0`）：
```python
# G_t = R_{t+1} + γ * R_{t+2} + γ² * R_{t+3} + ... + γ^{T-t-1} * R_T
G = 0.0
for step in reversed(trajectory.steps):
    G = step.reward + self.gamma * G
    step.reward = G  # 原地替换为 MC return
```

### reset() — 自动重置

([workflow.py L240-267](../rllm/workflows/workflow.py#L240-L267))

```python
def reset(self, task=None, uid=None):
    self.uid = uid
    self.task = task
    self._completed_trajectories = []

    # 自动发现并重置所有 BaseAgent 属性
    for attr_name in dir(self):
        if attr_name.startswith("_"): continue
        attr = getattr(self, attr_name)
        if isinstance(attr, BaseAgent):
            attr.reset()
            attr.trajectory.task = task

    # 自动发现并重置所有 BaseEnv 属性
    for attr_name in dir(self):
        if attr_name.startswith("_"): continue
        attr = getattr(self, attr_name)
        if isinstance(attr, BaseEnv):
            attr.reset(task=task)
```

## AgentWorkflowEngine 任务执行

### Workflow 池初始化

`initialize_pool()` ([L75-88](../rllm/engine/agent_workflow_engine.py#L75-L88))：

```python
async def initialize_pool(self):
    if self.workflow_queue is not None: return  # 幂等
    self.workflow_queue = asyncio.Queue(maxsize=self.n_parallel_tasks)
    for i in range(self.n_parallel_tasks):
        workflow = self.workflow_cls(
            rollout_engine=self.rollout_engine,
            executor=self.executor,
            **self.workflow_args
        )
        assert workflow.is_multithread_safe()
        self.workflow_queue.put_nowait(workflow)
```

> 池化 N 个 Workflow 实例，每个实例共享同一个 `rollout_engine` 和 `executor`。

### 重试逻辑与终止处理

`process_task_with_retry()` ([L90-141](../rllm/engine/agent_workflow_engine.py#L90-L141))：

```python
async def process_task_with_retry(self, task, task_id, rollout_idx, **kwargs):
    workflow = await self.workflow_queue.get()  # 从池中获取 workflow
    try:
        for retry_attempt in range(1, self.retry_limit + 1):
            uid = f"{task_id}:{rollout_idx}"
            episode = await workflow.run_with_termination_handling(task=task, uid=uid)

            # 显示奖励（每个 trajectory 的 reward）
            reward_strs = [f"{t.name}: {t.reward:.1f}" for t in episode.trajectories]
            colorful_print(f"[{uid}] Rewards: {reward_strs}, Term: {episode.termination_reason}")

            # 非 ERROR 终止→成功
            if episode.termination_reason != TerminationReason.ERROR:
                return task_id, rollout_idx, episode

            # ERROR→重试或放弃
            if retry_attempt < self.retry_limit:
                print(f"[{uid}] Failed attempt {retry_attempt}/{self.retry_limit}, retrying...")
        return task_id, rollout_idx, episode  # 最终返回最后一次的 episode
    finally:
        await self.workflow_queue.put(workflow)  # 归还到池
```

### execute_tasks 批量执行

([L143-199](../rllm/engine/agent_workflow_engine.py#L143-L199))

```python
async def execute_tasks(self, tasks, task_ids=None, **kwargs):
    if self.workflow_queue is None:
        await self.initialize_pool()

    if task_ids is None:
        task_ids = [str(uuid.uuid4()) for _ in tasks]

    # 构建 futures 列表（每个 task × rollout_idx 一个 future）
    futures = []
    for task, task_id in zip(tasks, task_ids):
        futures.append(self.process_task_with_retry(task, task_id, rollout_idx))

    # as_completed 逐个收集结果 + tqdm 进度条
    with tqdm(total=len(tasks), desc="Generating trajectories") as pbar:
        for future in asyncio.as_completed(futures):
            task_id, rollout_idx, episode = await future
            task_states[task_id]["episodes"].append(episode)
            pbar.update(1)

    # 按原始顺序排列结果
    results = []
    for task_id in sorted(task_states.keys(), key=lambda tid: task_states[tid]["idx"]):
        results.extend(task_states[task_id]["episodes"])

    # Episode 日志
    if self.episode_logger is not None:
        self.episode_logger.log_episodes_batch(results, self.current_step, ...)

    return results
```

## DataProto 转换（Verl 集成）

### execute_tasks_verl()

([L201-227](../rllm/engine/agent_workflow_engine.py#L201-L227))——verl PPO 训练器的入口：

```python
async def execute_tasks_verl(self, batch: DataProto, **kwargs):
    await self.rollout_engine.wake_up()           # GPU 加载推理权重

    is_validation = batch.meta_info.get("validate", False)
    tasks = batch.non_tensor_batch["extra_info"].tolist()
    task_ids = batch.non_tensor_batch["task_ids"].tolist()

    results = await self.execute_tasks(tasks, task_ids)  # 执行所有 workflow

    await self.rollout_engine.sleep()              # GPU 卸载推理权重

    return self.transform_results_for_verl(results, task_ids)
```

### transform_results_for_verl() 详解

([L229-514](../rllm/engine/agent_workflow_engine.py#L229-L514))——将 Episode 列表转换为 verl 的 `DataProto`：

#### Tensor 构建

| 步骤 | 行号 | 操作 |
|------|------|------|
| Prompt 左填充 | L402-409 | `pad_sequence(flip → pad → flip)` + 截断到 `max_prompt_length` |
| Response 右填充 | L411-418 | `pad_sequence` + 截断到 `max_response_length` |
| input_ids 拼接 | L420 | `torch.concat([prompts, responses], dim=1)` |
| Attention mask | L422-430 | `prompt_mask` (左对齐) + `response_mask` (右对齐) |
| Position IDs | L440 | `cumsum(attention_mask) - 1` (标准) 或 mrope (VLM) |
| Response mask | L442-444 | `traj_mask` (只对 assistant 计算 loss) |
| Reward 放置 | L447-454 | 放在最后一个 response token 位置 |
| Rollout logprobs | L456-464 | 可选，用于重要性采样 |

#### 两种 Tokenization 模式

([L301-376](../rllm/engine/agent_workflow_engine.py#L301-L376))

| 条件 | 模式 | 行号 | 处理方式 |
|------|------|------|---------|
| `stepwise_advantage.enable=False` + 多步 | 累积模式 | L306-311 | `tokenize_and_mask_cumulative(last_step.chat_completions)` |
| `stepwise_advantage.enable=False` + 单步 + ModelOutput | 直接模式 | L314-331 | 直接用 `model_output.prompt_ids/completion_ids` |
| `stepwise_advantage.enable=False` + 单步 + 无 ModelOutput | 重新解析 | L332-339 | `tokenize_and_mask(chat_completions)` |
| `stepwise_advantage.enable=True` | 逐步模式 | L346-375 | 每个 step 独立 tokenize |

#### Compact Filtering

([L467-473](../rllm/engine/agent_workflow_engine.py#L467-L473))

根据终止原因设置 `is_valid` 标志：

```python
cf = self.config.rllm.compact_filtering
is_valid = [True] * len(episode_ids)
if cf.enable:
    for i in range(len(episode_ids)):
        termination_reason = termination_reasons[i]
        if (cf.mask_error and termination_reason == ERROR) or \
           (cf.mask_timeout and termination_reason == TIMEOUT) or \
           (cf.mask_max_prompt_length_exceeded and termination_reason == MAX_PROMPT_LENGTH_EXCEEDED) or ...:
            is_valid[i] = False
```

> `is_valid=False` 的样本在 verl 训练时不计算 loss。

#### 输出 DataProto 结构

**Tensor 字段** ([L493-505](../rllm/engine/agent_workflow_engine.py#L493-L505))：

| 字段 | 形状 | 说明 |
|------|------|------|
| `input_ids` | `[B, P+R]` | prompt + response 拼接 |
| `attention_mask` | `[B, P+R]` | 有效 token 标记 |
| `position_ids` | `[B, P+R]` 或 `[B, 4, P+R]` | 位置编码（VLM 为 4D mrope） |
| `prompts` | `[B, P]` | prompt 部分（左 pad） |
| `responses` | `[B, R]` | response 部分（右 pad） |
| `response_mask` | `[B, R]` | assistant token loss mask |
| `traj_rewards` | `[B, R]` | 轨迹奖励（最后 token） |
| `step_rewards` | `[B, R]` | 步级奖励（最后 token） |
| `rollout_log_probs` | `[B, R]` | rollout 时的 log probs（可选） |

**Non-Tensor 字段** ([L475-488](../rllm/engine/agent_workflow_engine.py#L475-L488))：

| 字段 | 格式 | 说明 |
|------|------|------|
| `episode_ids` | `"task_id:rollout_idx"` | Episode 唯一标识 |
| `trajectory_ids` | `"task_id_traj_name"` | Trajectory 唯一标识 |
| `step_ids` | `"task_id_traj_name_step{i}"` | Step 唯一标识 |
| `is_correct` | `bool` | 是否正确 |
| `termination_reasons` | `str` | 终止原因 |
| `is_valid` | `bool` | compact filtering 标志 |
| `is_last_step` | `bool` | 是否为轨迹最后一步 |
| `chat_completions` | `list[dict]` | 对话历史（蒸馏用） |

**Meta 信息** ([L510-513](../rllm/engine/agent_workflow_engine.py#L510-L513))：

| 字段 | 说明 |
|------|------|
| `repeat_counts` | 每个 episode 贡献的步数（关键元数据） |
| `dropped_episodes` | 被丢弃的 episode 列表 |

> `repeat_counts` 告诉 verl 哪些行属于同一 episode——多 trajectory episode 会在 DataProto 中展开为多行。

## 多模态支持

`_handle_multimodal_position_ids()` ([L516-549](../rllm/engine/agent_workflow_engine.py#L516-L549)) 处理 Qwen2VL/Qwen3VL 的 mrope position_ids：

```python
# 检测 Qwen2VL/Qwen3VL 处理器
if "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
    if "Qwen3VLProcessor" in processor.__class__.__name__:
        from verl.models.transformers.qwen3_vl import get_rope_index
    else:
        from verl.models.transformers.qwen2_vl import get_rope_index

    for i in range(batch_size):
        vision_position_ids = get_rope_index(
            processor, input_ids=input_ids[i],
            image_grid_thw=model_inputs.get("image_grid_thw"),
            video_grid_thw=model_inputs.get("video_grid_thw"),
        )  # (3, seq_length) — temporal/height/width
        # 添加文本 position_ids → (4, seq_length) — text/temporal/height/width
```

## TerminationReason 枚举

([workflow.py L16-23](../rllm/workflows/workflow.py#L16-L23))

```python
class TerminationReason(Enum):
    MAX_PROMPT_LENGTH_EXCEEDED = "max_prompt_length_exceeded"
    MAX_RESPONSE_LENGTH_EXCEEDED = "max_response_length_exceeded"
    ENV_DONE = "env_done"
    MAX_TURNS_EXCEEDED = "max_turns_exceeded"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"
    ERROR = "error"
```

### TerminationEvent

([workflow.py L26-29](../rllm/workflows/workflow.py#L26-L29))——Workflow 内部主动终止的异常：

```python
class TerminationEvent(Exception):
    def __init__(self, reason: TerminationReason = TerminationReason.UNKNOWN):
        super().__init__(f"Terminated: {reason}")
        self.reason = reason
```

用法：`raise TerminationEvent(TerminationReason.MAX_TURNS_EXCEEDED)`

## 配置参考

```yaml
rllm:
  workflow:
    use_workflow: true
  stepwise_advantage:
    enable: false
    mode: "broadcast"
  compact_filtering:
    enable: true
    mask_error: true
    mask_timeout: true
    mask_max_prompt_length_exceeded: true
    mask_max_response_length_exceeded: false
    mask_env_done: false
    mask_max_turns_exceeded: false
    mask_unknown: true
  filter_token_mismatch: true

training:
  n_parallel_tasks: 128
  retry_limit: 3
  timeout: 300
  gamma: 0.0
  reward_bonus_coeff: 0.0
```

## 线程安全要求

`Workflow.is_multithread_safe()` ([workflow.py L269-279](../rllm/workflows/workflow.py#L269-L279)) 遍历所有 `BaseEnv` 属性，确认都是线程安全的。池初始化时断言此条件（[L87](../rllm/engine/agent_workflow_engine.py#L87)）。
