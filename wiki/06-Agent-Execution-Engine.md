# Agent 执行引擎 (Agent Execution Engine)

Agent 执行引擎是负责管理异步 Agent-环境交互的核心编排层，在轨迹生成期间协调多个 Agent 实例的并行执行、处理终止条件和重试逻辑，并将多步对话转换为训练可用的 token 序列。

> 源码参考：[rllm/engine/agent_execution_engine.py L29-627](../rllm/engine/agent_execution_engine.py)

## 核心组件

引擎由两个类实现：

- `AgentExecutionEngine` ([L29-622](../rllm/engine/agent_execution_engine.py#L29-L622))：主实现，管理异步轨迹生成
- `AsyncAgentExecutionEngine` ([L624-627](../rllm/engine/agent_execution_engine.py#L624-L627))：子类别名，向后兼容

### 类结构

```python
class AgentExecutionEngine:
    # 核心状态
    agents: list[BaseAgent | None]        # Agent 实例池
    envs: list[BaseEnv | None]            # 环境实例池
    rollout_engine: RolloutEngine          # 推理引擎
    chat_parser: ChatTemplateParser        # 模板解析器
    executor: ThreadPoolExecutor           # 环境操作线程池

    # 配置
    n_parallel_agents: int                 # 并发 Agent 数
    max_env_workers: int                   # 环境线程池大小
    max_steps: int                         # 最大交互步数
    max_response_length: int              # response 最大 token 数
    max_prompt_length: int                 # prompt 最大 token 数
    gamma: float                           # MC 折扣因子
    retry_limit: int                       # 重试次数
    trajectory_timeout: int                # 轨迹超时（秒）
    overlong_filter: bool                  # 过长轨迹过滤
    enforce_max_prompt_length: bool        # 逐步 prompt 长度检查
```

### 初始化参数

引擎在初始化时接受大量配置参数（[L30-54](../rllm/engine/agent_execution_engine.py#L30-L54)）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `engine_name` | `"openai"` | 推理引擎类型：`"openai"` / `"verl"` / `"tinker"` |
| `tokenizer` | `None` | HuggingFace tokenizer |
| `rollout_engine` | `None` | 预初始化的推理引擎（verl 模式需要，其他模式自动创建） |
| `chat_parser` | `None` | 聊天模板解析器（`None` 时自动从 tokenizer 推断） |
| `n_parallel_agents` | `128` | 并发活动 Agent 数（Semaphore 上限） |
| `max_workers` | `64` | 环境操作的线程池大小 |
| `trajectory_timeout` | `None` | 单条轨迹超时（秒），`None` → 近无限 `1e9` |
| `gamma` | `0.2` | Monte Carlo return 折扣因子 |
| `api_retries` | `3` | API 调用重试次数（传给 OpenAIEngine） |
| `retry_limit` | `3` | 轨迹级重试次数 |
| `max_steps` | `5` | Agent 最大交互步数 |
| `max_response_length` | `8192` | response 最大 token 数 |
| `max_prompt_length` | `1024` | prompt 最大 token 数 |
| `enforce_max_prompt_length` | `False` | 是否每步检查 prompt 长度 |
| `overlong_filter` | `False` | 是否对过长轨迹置零 response_masks |
| `agent_class` | `None` | Agent 类（用于 `execute_tasks`） |
| `env_class` | `None` | 环境类（用于 `execute_tasks`） |
| `agent_args` | `None` | Agent 初始化参数 |
| `env_args` | `None` | 环境初始化参数 |

### Rollout 引擎初始化

根据 `engine_name` 自动创建对应的推理引擎（[L101-128](../rllm/engine/agent_execution_engine.py#L101-L128)）：

```python
if self.engine_name == "openai":
    self.rollout_engine = OpenAIEngine(
        **rollout_engine_args,
        api_retries=api_retries,
        tokenizer=self.tokenizer,
        chat_parser=self.chat_parser,
        max_prompt_length=self.max_prompt_length,
        max_response_length=self.max_response_length,
        disable_thinking=self.disable_thinking,
    )
elif self.engine_name == "verl":
    self.rollout_engine = VerlEngine(
        config=self.config,
        rollout_manager=rollout_engine,  # 从 verl 传入的 rollout manager
        tokenizer=self.tokenizer,
        disable_thinking=self.disable_thinking,
    )
elif self.engine_name == "tinker":
    self.rollout_engine = TinkerEngine(**rollout_engine_args)
```

### ChatTemplateParser 集成

如果未提供 `chat_parser`，自动从 tokenizer 推断（[L93-96](../rllm/engine/agent_execution_engine.py#L93-L96)）：

```python
if chat_parser is None:
    self.chat_parser = ChatTemplateParser.get_parser(
        self.tokenizer, disable_thinking=self.disable_thinking
    )
```

## 轨迹执行生命周期

### 单条轨迹流程

`run_agent_trajectory_async()` ([L184-433](../rllm/engine/agent_execution_engine.py#L184-L433)) 实现核心 Agent-环境交互循环：

```
┌── 初始化 ─────────────────────────────────────────────────────┐
│  env.reset() → (observation, info)  [L206-207]                │
│  agent.reset()                       [L211]                    │
│  agent.update_from_env(obs, 0.0, False, info) [L213-218]      │
│  prompt_token_len = tokenize(agent.chat_completions) [L220-221]│
│  if prompt_token_len > max_prompt_length: raise [L223-225]     │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
┌── 主循环 for step_idx in range(max_steps) ────────────────────┐
│                                                                │
│  1. 准备 prompt [L229-242]                                     │
│     messages = agent.chat_completions.copy()                   │
│     if enforce_max_prompt_length:                              │
│         prompt_len = len(tokenizer.encode(messages))            │
│         if prompt_len > max_prompt_length: break [PROMPT_TRUNCATION] │
│     else:                                                      │
│         max_tokens = max_response_length - used_response_tokens │
│                                                                │
│  2. LLM 调用 [L246-251]                                       │
│     model_output = await get_model_response(messages, app_id)  │
│     response = model_output.text                               │
│     记录 {prompt_ids, completion_ids, logprobs} [L253-260]     │
│                                                                │
│  3. Agent 处理响应 [L263-264]                                   │
│     action: Action = agent.update_from_model(response)         │
│                                                                │
│  4. 环境执行 [L267-280]                                        │
│     (obs, reward, done, info) = await env.step(action)         │
│     with timeout = trajectory_timeout - elapsed                │
│     if TimeoutError: termination_reason = "ENV_TIMEOUT", break │
│                                                                │
│  5. 更新 Agent 状态 [L289-299]                                  │
│     agent.update_from_env(obs, reward, done, info)             │
│     cur_step = agent.get_current_state()                       │
│     cur_step.reward = reward                                   │
│     cur_step.done = done                                       │
│                                                                │
│  6. Token 累积 & 长度检查 [L301-358]                            │
│     将 assistant 和 env 消息转为 token                          │
│     response_token_len += new_tokens                           │
│     if response_token_len >= max_response_length:              │
│         截断最后一批 response tokens                            │
│         termination_reason = "TRUNCATION", break               │
│                                                                │
│  7. 超时/终止检查 [L345-361]                                    │
│     if total_time >= trajectory_timeout: "TIMEOUT"             │
│     if done: "ENV_DONE"                                        │
│     if step_idx == max_steps-1: "MAX_STEPS"                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
┌── 后处理 ──────────────────────────────────────────────────────┐
│  overlong_filter [L363-368]:                                   │
│    if TRUNCATION/MAX_STEPS/TIMEOUT: response_masks = all 0s    │
│                                                                │
│  compute_final_reward [L370-375]:                              │
│    if env has compute_final_reward():                           │
│        reward = await env.compute_final_reward()               │
│                                                                │
│  env.close() [L377]                                            │
│                                                                │
│  compute_trajectory_reward(trajectory) [L392]                  │
│  compute_mc_return(trajectory, gamma) [L393]                   │
│                                                                │
│  return trajectory / token_result / conversation / steps  [L395-433] │
└────────────────────────────────────────────────────────────────┘
```

### 四种执行模式

引擎支持四种输出模式（[L395-433](../rllm/engine/agent_execution_engine.py#L395-L433)）：

| 模式 | 返回类型 | 用途 | 关键内容 |
|------|---------|------|---------|
| `"Text"` | `Trajectory` | Workflow 引擎使用 | 完整的 Trajectory 对象 |
| `"Token"` | `dict` | 直接训练 | `{prompt_tokens, response_tokens, response_masks, trajectory_reward, metrics}` |
| `"Conversation"` | `list[dict]` | 对话日志 | `agent.chat_completions` |
| `"Step"` | `dict` | 步级分析 | `{steps, trajectory_reward, mc_returns, termination_reason}` |

**Token 模式输出结构**：
```python
{
    "prompt_tokens": torch.Tensor,         # [P] 初始 prompt token IDs
    "response_tokens": torch.Tensor,       # [R] 所有对话续接 token IDs
    "response_masks": torch.Tensor,        # [R] 1=assistant, 0=env/user
    "trajectory_reward": float,            # 聚合奖励
    "idx": int,                            # 环境索引
    "chat_completions": list[dict],        # 原始对话
    "metrics": {
        "steps": int,                      # 总步数
        "reward_time": float,              # 奖励计算耗时
        "env_time": float,                 # 环境执行总耗时
        "llm_time": float,                 # LLM 调用总耗时
        "total_time": float,               # 总耗时
        "token_mismatch": 0.0 | 1.0,      # 是否有 token 不一致
    },
}
```

## Step 拼装与 Tokenization

### 拼装流程

`assemble_steps()` ([L435-496](../rllm/engine/agent_execution_engine.py#L435-L496)) 将逐步收集的 prompt-response 对拼装为连续的训练 token 序列：

```python
def assemble_steps(self, steps: list[dict]):
    """
    核心逻辑:
    1. 第一步: prompt_ids 作为 initial_prompt，completion_ids 作为 response (mask=1)
    2. 后续步: 验证 current_prompt_ids 是否以 accumulated_sequence 为前缀
       - 是: 新增部分 = prompt_ids[len(accumulated):] → mask=0
                      + completion_ids → mask=1
       - 否: 警告 + 设置 is_valid_trajectory = False → response_masks 全 0
    """
```

**拼装确保的三个保证**：
1. **累积性验证**（[L466-481](../rllm/engine/agent_execution_engine.py#L466-L481)）：每步的 prompt 必须以前序累积 token 为前缀
2. **Loss Masking**：只对 assistant completion token 计算 loss（mask=1），user/env token mask=0
3. **Token 不一致检测**（[L476-478](../rllm/engine/agent_execution_engine.py#L476-L478)）：重新 tokenize 可能导致序列不一致，此时通过 `config.rllm.filter_token_mismatch` 决定是否全部置零

### 拼装示例

2 步轨迹的拼装过程：

```
Step 0:
  prompt_ids:     [1, 2, 3, 4]           # "User: Solve problem"
  completion_ids: [5, 6, 7]              # "Assistant: Let me think..."

Step 1:
  prompt_ids:     [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 前序 + "User: Continue"
  completion_ids: [10, 11, 12]                   # "Assistant: Answer is 42"

拼装结果:
  prompt_tokens:  [1, 2, 3, 4]                   # 初始 prompt
  response_tokens:[5, 6, 7, 8, 9, 10, 11, 12]    # 所有后续 token
  response_masks: [1, 1, 1, 0, 0,  1,  1,  1]    # assistant=1, env=0
```

> 第 1 步的 `current_prompt_ids[:7] == accumulated_sequence[:7]`，验证通过。
> 新增部分 `[8, 9]` 为环境消息（mask=0），`[10, 11, 12]` 为 assistant 回复（mask=1）。

## 并行执行架构

### trajectory_generator 模式

`trajectory_generator()` ([L509-556](../rllm/engine/agent_execution_engine.py#L509-L556)) 实现异步批量轨迹生成：

```python
async def trajectory_generator(self, reset_seed=0, timing_raw=None, mode="Text", **kwargs):
    # 1. 验证所有环境是 BaseEnv 且线程安全 [L512-513]
    # 2. 创建线程池 [L516]
    self.executor = ThreadPoolExecutor(max_workers=max_concurrency)

    # 3. Wake up (verl only) [L518-519]
    if self.engine_name == "verl":
        await self.rollout_engine.wake_up()

    # 4. Semaphore 限制并发 [L521]
    semaphore = asyncio.Semaphore(self.n_parallel_agents)

    # 5. 创建所有任务并通过 as_completed 逐个返回 [L541-549]
    tasks_to_run = [launch_one(i) for i in range(len(self.envs))]
    for coro in asyncio.as_completed(tasks_to_run):
        result = await coro
        yield result  # 异步生成器，完成一个返回一个

    # 6. Sleep (verl only) [L553-554]
    if self.engine_name == "verl":
        await self.rollout_engine.sleep()
```

**关键设计决策**：
1. **Semaphore 控制**（[L521](../rllm/engine/agent_execution_engine.py#L521)）：限制并发到 `n_parallel_agents`
2. **As-Completed 返回**（[L544-549](../rllm/engine/agent_execution_engine.py#L544-L549)）：结果一完成就返回，**不按顺序**
3. **Wake/Sleep 生命周期**（[L518-554](../rllm/engine/agent_execution_engine.py#L518-L554)）：管理 vLLM 服务器的 GPU 占用
4. **错误即时传播**（[L550-551](../rllm/engine/agent_execution_engine.py#L550-L551)）：异常立即抛出

### execute_tasks 模式

`execute_tasks()` ([L558-616](../rllm/engine/agent_execution_engine.py#L558-L616)) 提供替代执行模式——动态任务列表处理：

```python
async def execute_tasks(self, tasks: list[dict]):
    # 动态创建 Agent 和环境实例
    semaphore = asyncio.Semaphore(max_concurrent)
    index_queue = asyncio.Queue(maxsize=max_concurrent)

    async def sem_wrapper(task_id, task):
        async with semaphore:
            index = await index_queue.get()  # 获取可用索引
            try:
                self.envs[index] = self.env_class.from_dict({**task, **self.env_args})
                self.agents[index] = self.agent_class(**self.agent_args)
                res = await self.run_agent_trajectory_async(index, ...)
                return task_id, res
            finally:
                await index_queue.put(index)  # 归还索引

    results = await asyncio.gather(*[sem_wrapper(i, t) for i, t in enumerate(tasks)])
```

**与 trajectory_generator 的区别**：

| 维度 | `trajectory_generator` | `execute_tasks` |
|------|----------------------|----------------|
| Agent/Env | 预初始化（`update_envs_and_agents`） | 每任务动态创建 |
| 返回 | 异步生成器（逐个返回） | 全部完成后返回列表 |
| 索引 | 固定映射 | 通过 Queue 动态分配 |
| 适用 | verl 训练循环 | 独立推理/评估 |

## 重试与错误处理

### 重试包装器

`run_agent_trajectory_with_retry()` ([L498-507](../rllm/engine/agent_execution_engine.py#L498-L507))：

```python
async def run_agent_trajectory_with_retry(self, idx, seed=0, mode="Text", **kwargs):
    for _ in range(self.retry_limit):
        try:
            application_id = str(uuid.uuid4())
            return await asyncio.wait_for(
                self.run_agent_trajectory_async(idx, application_id=application_id, ...),
                timeout=7200  # 硬超时 2 小时
            )
        except Exception:
            traceback.print_exc()
            continue
    raise Exception(f"Trajectory {idx} cannot complete.")
```

### 终止原因

引擎追踪六种终止条件（[L186-361](../rllm/engine/agent_execution_engine.py#L186-L361)）：

| 终止原因 | 触发条件 | 行号 | 影响 |
|----------|---------|------|------|
| `PROMPT_TRUNCATION` | `enforce_max_prompt_length=True` 且 prompt 过长 | L241-242 | 跳过后续步 |
| `ENV_TIMEOUT` | `env.step()` 超过 `trajectory_timeout` | L271-280 | 奖励置 0 |
| `TRUNCATION` | `response_token_len >= max_response_length` | L317-338 | 截断并结束 |
| `TIMEOUT` | `total_time >= trajectory_timeout` | L345-350 | 标记结束 |
| `MAX_STEPS` | `step_idx == max_steps - 1` | L360-361 | 标记结束 |
| `ENV_DONE` | 环境返回 `done=True` | L353-355 | 正常结束 |

**Overlong Filtering**（[L363-368](../rllm/engine/agent_execution_engine.py#L363-L368)）：当 `overlong_filter=True` 时，`TRUNCATION`/`MAX_STEPS`/`TIMEOUT` 终止的轨迹的 `response_masks` 全部置零，从训练梯度中排除。

## 线程池管理

环境操作通过 `ThreadPoolExecutor` 在后台线程中执行（[L130-131](../rllm/engine/agent_execution_engine.py#L130-L131)）：

```python
self.executor = ThreadPoolExecutor(max_workers=max_workers)

# 环境 reset [L207]
observation, info = await loop.run_in_executor(self.executor, env.reset)

# 环境 step [L270]
next_observation, reward, done, info = await asyncio.wait_for(
    loop.run_in_executor(self.executor, env.step, action),
    timeout=(self.trajectory_timeout - total_time)
)

# 环境 close [L377]
await loop.run_in_executor(self.executor, env.close)

# 最终奖励 [L373]
reward = await loop.run_in_executor(self.executor, env.compute_final_reward)
```

> 同步环境的 `reset()`/`step()`/`close()` 被包装为异步操作，无需环境实现异步接口。

## Rollout 引擎集成

### 引擎分派

`get_model_response()` ([L133-167](../rllm/engine/agent_execution_engine.py#L133-L167)) 基于 `engine_name` 分派到不同引擎，所有引擎返回标准 `ModelOutput`：

```python
async def get_model_response(self, prompt, application_id, **kwargs) -> str:
    sampling_params = self.sampling_params.copy()
    sampling_params.update(kwargs)

    if self.engine_name == "openai":
        return await self.rollout_engine.get_model_response(
            prompt, application_id=application_id,
            enforce_max_prompt_length=False, **sampling_params
        )
    elif self.engine_name == "verl":
        meta_data = sampling_params.pop("meta_info", {})
        validate = meta_data.get("validate", False)
        return await self.rollout_engine.get_model_response(
            prompt, application_id=application_id,
            validate=validate, enforce_max_prompt_length=False, **sampling_params
        )
    elif self.engine_name == "tinker":
        return await self.rollout_engine.get_model_response(
            prompt, application_id=application_id,
            enforce_max_prompt_length=False, **sampling_params
        )
```

## 使用模式

### 模式 1：预初始化 Agent/Env

```python
engine = AgentExecutionEngine(engine_name="openai", tokenizer=tokenizer, ...)

# 外部创建 Agent 和 Env
agents = [MyAgent() for _ in range(batch_size)]
envs = [MyEnv(task) for task in tasks]
engine.update_envs_and_agents(envs, agents)

# 异步生成器
async for trajectory in engine.trajectory_generator(mode="Token"):
    process(trajectory)
```

### 模式 2：动态任务处理

```python
engine = AgentExecutionEngine(
    agent_class=ToolAgent, agent_args={"tools": ["python"]},
    env_class=ToolEnvironment, env_args={"reward_fn": math_reward_fn},
    engine_name="openai", tokenizer=tokenizer,
    n_parallel_agents=64,
)
# 自动创建 Agent 和 Env 实例
results = asyncio.run(engine.execute_tasks(tasks))
```

### 模式 3：verl 训练更新

```python
# 在 verl PPO 训练器内部
engine = AgentExecutionEngine(engine_name="verl", rollout_engine=rollout_manager, ...)
engine.update_envs_and_agents(envs, agents)
async for result in engine.trajectory_generator(mode="Token"):
    # wake_up/sleep 在 generator 内自动管理
    batch.append(result)
```

## 指标与计时

每条轨迹记录以下指标：

| 指标 | 计算方式 | 用途 |
|------|---------|------|
| `steps` | `len(trajectory.steps)` | 轨迹步数 |
| `llm_time` | 所有 `get_model_response()` 耗时之和 | LLM 调用耗时分析 |
| `env_time` | 所有 `env.step()` 耗时之和 | 环境瓶颈分析 |
| `reward_time` | `compute_final_reward()` 耗时 | 奖励计算耗时 |
| `total_time` | 轨迹总耗时 | 整体效率分析 |
| `token_mismatch` | `0.0` 或 `1.0` | token 拼装一致性 |
