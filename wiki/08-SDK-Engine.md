# SDK 引擎 (SDK Engine)

`AgentSdkEngine` 是 rLLM 三种执行引擎中最灵活的一种——它不要求用户使用特定的 Agent/Env 抽象，而是通过 **LiteLLM Proxy** 拦截任意框架（LangGraph、SmolAgents、原生代码等）发起的 LLM 调用，自动收集 Trace 数据并转换为可训练的 Episode。

> 源码参考：[rllm/engine/agent_sdk_engine.py L42-723](../rllm/engine/agent_sdk_engine.py)

## 架构概览

```
用户代码 (LangGraph/SmolAgents/自定义)
    │
    │ HTTP 请求 (OpenAI-compatible API)
    ▼
LiteLLM Proxy + VerlProxyManager [proxy_manager.py]
    │  ├── 路由到 vLLM 后端
    │  ├── Trace 采集 (request/response 记录)
    │  └── 持久化到 SQLite [sqlite_store.py]
    ▼
VerlEngine.get_model_response()
    │  Python API → vLLM → token_ids + logprobs
    ▼
AgentSdkEngine._execute_tasks()
    ├── 并发执行 agent_run_func (用户函数)
    ├── flush_traces() → 确保 Trace 持久化
    ├── SQLite 查询 Trace 数据
    ├── group_steps() → Trajectory 分组
    └── transform_results_for_verl() → DataProto
```

## 初始化参数

`AgentSdkEngine.__init__()` ([L43-91](../rllm/engine/agent_sdk_engine.py#L43-L91))：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `agent_run_func` | `Callable` | 必填 | 用户的 Agent 执行函数（同步或异步） |
| `rollout_engine` | `RolloutEngine` | 必填 | 必须是 `VerlEngine` |
| `config` | `OmegaConf` | `None` | Hydra 训练配置 |
| `n_parallel_tasks` | `int` | `128` | 最大并发任务数 |
| `retry_limit` | `int` | `3` | 失败重试次数 |
| `raise_on_error` | `bool` | `True` | 永久失败时是否抛异常 |
| `proxy_config` | `dict` | `None` | LiteLLM Proxy 配置 |
| `tracer` | `TracerProtocol` | `None` | 可选 Trace 记录器 |

### proxy_config 详解

| 键 | 默认值 | 说明 |
|---|--------|------|
| `model_name` | 必填 | 暴露给用户代码的模型名（如 `gpt-4`） |
| `proxy_host` | `"127.0.0.1"` | Proxy 绑定地址 |
| `proxy_port` | `4000` | Proxy 端口 |
| `mode` | `"external"` | `"subprocess"` (自动启动) 或 `"external"` (手动启动) |
| `admin_token` | `"my-shared-secret"` | Admin API token |
| `db_path` | `None` | SQLite DB 路径（subprocess 模式） |
| `project` | `"rllm-agent-sdk"` | Trace 项目名 |
| `add_logprobs` | `False` | 是否记录 logprobs |

## 代理初始化流程

### VERL Proxy 设置

`_setup_verl_proxy()` ([L92-157](../rllm/engine/agent_sdk_engine.py#L92-L157))：

```python
# 1. 创建 VerlProxyManager
self.proxy_manager = VerlProxyManager(
    rollout_engine=self.rollout_engine,
    model_name=model_name,
    proxy_host=proxy_host,
    proxy_port=proxy_port,
    admin_token=admin_token,
)

# 2. 获取 Proxy URL
self.rollout_engine_endpoint = self.proxy_manager.get_proxy_url()

# 3. 启动方式分支
if proxy_mode == "subprocess":
    self.proxy_manager.start_proxy_subprocess(
        config=config_payload,
        db_path=db_path, project=project,
        sync_tracer=requires_sync_storage,  # OpenTelemetry 需要同步
    )
elif proxy_mode == "external":
    self.proxy_manager.reload_proxy_config(config=config_payload)
```

## 任务执行数据流

### _execute_tasks() 详细流程

([L238-382](../rllm/engine/agent_sdk_engine.py#L238-L382))

```
┌── 准备阶段 ──────────────────────────────────────────────────┐
│  rollout_start_time = time.time()  # Trace 查询时间基线 [L261] │
│  为每个 task 创建 future [L263-273]                             │
└──────────────────────────────────────────────────────────────┘
            │
            ▼
┌── 并发执行 ──────────────────────────────────────────────────┐
│  with tqdm(total=N):                                         │
│    for future in asyncio.as_completed(futures):              │
│      task_id, rollout_idx, retry, output, session_uid        │
│      = await future                                          │
│      session_uids.add(session_uid)  [L282]                   │
│      outputs[session_name] = output  [L284]                  │
└──────────────────────────────────────────────────────────────┘
            │
            ▼
┌── Trace 收集 ────────────────────────────────────────────────┐
│  await self.flush_traces(timeout=60.0)  [L292]               │
│  # 确保所有 Trace 写入 SQLite                                 │
│                                                               │
│  for session_uid in session_uids:  [L297-299]                │
│    traces = await store.get_by_session_uid(uid, since=start) │
│    all_traces.extend(traces)                                 │
└──────────────────────────────────────────────────────────────┘
            │
            ▼
┌── Trace → Episode 转换 ──────────────────────────────────────┐
│  for session_name, traces in traces_by_session_name:          │
│    steps = [trace_to_step(trace) for trace in traces]  [L314]│
│                                                               │
│    if output is float:  [L327-332]                            │
│      # 用户返回单个奖励值                                      │
│      trajectories = group_steps(steps, by=groupby_key)       │
│      for traj in trajectories: traj.reward = output          │
│    else:  [L333-350]                                          │
│      # 用户返回 list[Trajectory] 带自定义奖励                   │
│      按 step.id 匹配 SQLite 数据 → 填充 prompt_ids等           │
│                                                               │
│  episode = Episode(id=session_name, trajectories=trajectories)│
│  episode.metrics = {retry_attempt, steps_collected, ...}     │
└──────────────────────────────────────────────────────────────┘
```

### 用户函数返回值类型

`process_task_with_retry()` ([L191-236](../rllm/engine/agent_sdk_engine.py#L191-L236)) 支持三种返回值格式：

| 返回类型 | 含义 | 处理方式 |
|----------|------|---------|
| `float \| int \| bool` | 标量奖励 | `group_steps()` 自动分组，所有轨迹共享奖励 |
| `list[Trajectory]` | 自定义轨迹 | 按 `step.id` 匹配 SQLite Trace 数据 |
| `tuple(float, dict)` | 奖励 + 指标 | 同 float，附加自定义 metrics |
| `tuple(list[Trajectory], dict)` | 轨迹 + 指标 | 同 list[Trajectory]，附加 metrics |

### 错误容忍

`execute_tasks()` ([L384-404](../rllm/engine/agent_sdk_engine.py#L384-L404)) 有三层重试：

```python
async def execute_tasks(self, tasks, task_ids=None):
    for _ in range(3):  # 整批重试最多 3 次
        results = await self._execute_tasks(tasks, task_ids)
        error_count = sum(1 for ep in results
                         if ep is None or not ep.trajectories)
        if error_count / len(results) > 0.01:
            # 错误率 > 1%，休眠 120 秒后重试
            await asyncio.sleep(120.0)
        else:
            return results
    raise Exception("Failed after 3 retries")
```

## Session 与 Trace 系统

### wrap_with_session_context

([L87](../rllm/engine/agent_sdk_engine.py#L87))——将用户函数包装为带 Session 上下文的版本：

```python
self.wrapped_agent_run_func = wrap_with_session_context(
    self.agent_run_func, tracer_service_name="agent-sdk-worker"
)
```

> 每次调用用户函数时自动创建 Session，Session 内的所有 LLM 调用通过 LiteLLM Proxy 被自动记录。

### SqliteTraceStore

([L90](../rllm/engine/agent_sdk_engine.py#L90))——Trace 持久化存储：

```python
self.store = SqliteTraceStore(db_path=self.config.rllm.sdk.store.path)
```

查询接口：`store.get_by_session_uid(session_uid, since=timestamp)` → 返回该 session 从 `since` 时间点起的所有 Trace。

## DataProto 转换

`transform_results_for_verl()` ([L464-712](../rllm/engine/agent_sdk_engine.py#L464-L712))——与 Workflow Engine 的实现类似，但有两个关键区别：

1. **始终使用 ModelOutput 模式**（[L558-588](../rllm/engine/agent_sdk_engine.py#L558-L588)）——因为 Trace 系统保证每步都有 `prompt_ids` 和 `completion_ids`
2. **Overlong Prompt 跳过**（[L561-565](../rllm/engine/agent_sdk_engine.py#L561-L565)）——如果单步 prompt 超过 `max_prompt_length`，静默跳过该步（避免训练/推理 OOD）

```python
if len(prompt_ids) > max_prompt_length:
    logger.warning(f"Skipping step {step_idx}: prompt {len(prompt_ids)} > {max_prompt_length}")
    continue
```

## 使用示例

```python
# 用户定义 Agent 函数（可使用任何框架）
def my_agent_func(metadata, question, answer, **kwargs):
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:4000")  # 指向 LiteLLM Proxy
    
    response = client.chat.completions.create(
        model="my-model",
        messages=[{"role": "user", "content": question}],
    )
    predicted = response.choices[0].message.content
    reward = 1.0 if predicted.strip() == answer.strip() else 0.0
    return reward

trainer = AgentTrainer(
    agent_run_func=my_agent_func,
    backend="verl",
    config={"rllm.sdk.processing.groupby_key": "session_name"},
)
trainer.train()
```

## 配置参考

```yaml
rllm:
  sdk:
    processing:
      groupby_key: "session_name"    # Trace 分组键
      traj_name_key: "model"         # 轨迹命名键
    store:
      path: "/tmp/rllm_traces.db"    # SQLite 路径

  compact_filtering:
    enable: true
    mask_error: true
    mask_timeout: true
```
