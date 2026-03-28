# rLLM 架构分析补充：Review 疑问全面解答

> 针对 [rllm_architecture_review.md](file:///home/robomaster/Research/rllm/rllm_architecture_review.md) 中的 10 个疑问，逐一给出源码级解答。

---

## 🔴 1. RL 算法层细节

### 1.1 支持的四种优势估计算法

源码位于 [`rllm/experimental/common/rl_algo.py`](file:///home/robomaster/Research/rllm/rllm/experimental/common/rl_algo.py) 和 [`advantage.py`](file:///home/robomaster/Research/rllm/rllm/experimental/common/advantage.py)。

| 算法 | 公式 | 适用场景 |
|------|------|---------|
| **GRPO** | `A_i = (R_i - μ_group) / (σ_group + ε)` | 默认选项，同一 task 多 rollout 组内比较 |
| **REINFORCE** | `A_i = R_i` (无 baseline) | 最简单的策略梯度，单次 rollout |
| **REINFORCE++** | 组内中心化 + 全 batch 标准差白化 | 改进的 REINFORCE，带 baseline |
| **RLOO** | `A_i = n/(n-1) × (R_i - μ_group)` | Leave-one-out baseline，减少方差 |

**GRPO 实现**（28行，极简）：
```python
def calculate_grpo_advantages_per_group(rewards, norm_adv_by_std_in_grpo=True, epsilon=1e-6):
    group_mean = np.mean(rewards)
    group_std = np.std(rewards)
    if norm_adv_by_std_in_grpo:
        advantages = (rewards - group_mean) / (group_std + epsilon)
    else:
        advantages = rewards - group_mean
    return advantages, advantages
```

### 1.2 配置方式

通过 [`AlgorithmConfig`](file:///home/robomaster/Research/rllm/rllm/experimental/common/config.py) 配置（Hydra YAML）：

```yaml
rllm:
  algorithm:
    adv_estimator: "grpo"      # grpo | reinforce | reinforce_plus_plus_baseline | rloo
    use_rllm: true             # 使用 rLLM 原生优势计算
    use_precomputed_advantage: false  # 使用预计算优势（蒸馏用）
    loss_fn: "importance_sampling"    # importance_sampling | ppo | cispo | dro | cross_entropy
    lr_schedule: "cosine"             # linear | cosine | constant
    warmup_steps_ratio: 0.05
  stepwise_advantage:
    mode: "broadcast"          # broadcast（轨迹级） | per_step（已弃用）
    norm_adv_by_std_in_grpo: true
```

### 1.3 Reward Shaping（`adjust_step_rewards`）

源码 [`workflow.py:149-170`](file:///home/robomaster/Research/rllm/rllm/workflows/workflow.py#L149-L170)：

```python
def adjust_step_rewards(self, trajectory):
    # 1. 差分奖励塑形（鼓励进步）
    if self.reward_bonus_coeff > 0.0:
        for i in range(1, len(steps)):
            steps[i].reward += bonus * (raw_rewards[i] - raw_rewards[i-1])
    
    # 2. γ 折扣 MC 回报（反向迭代）
    if self.gamma > 0.0:
        G = 0.0
        for step in reversed(steps):
            G = step.reward + gamma * G
            step.reward = G   # 原地替换为 MC return
```

> **差分奖励**：如果 step 2 奖励比 step 1 高，给予额外 bonus，鼓励逐步改进。
> **MC return**：标准的蒙特卡洛回报 `G_t = R_{t+1} + γ·G_{t+1}`，默认 γ=0.0（不折扣）。

### 1.4 GRPO 在多步场景中的工作方式

关键理解：**GRPO 在轨迹(trajectory)层面比较，而非 step 层面**。

```
Task "Q1" 的 4 次 rollout → 4 个 trajectory → 4 个 trajectory.reward
GRPO: A_i = (reward_i - mean(rewards)) / std(rewards)
```

在 `broadcast` 模式下，每个 trajectory 内所有 step 共享同一个 advantage 值。即：
```python
for traj, advantage in zip(group.trajectories, advantages):
    for step in traj.steps:
        step.advantage = advantage  # 所有 step 同一个 scalar
```

> "组"的定义：同一 `task_id` + 同一 `trajectory.name` 的所有 rollout 构成一个 `TrajectoryGroup`。

---

## 🔴 2. BaseAgent 完整接口

源码 [`agents/agent.py:272-333`](file:///home/robomaster/Research/rllm/rllm/agents/agent.py#L272-L333)：

```python
class BaseAgent(ABC):
    # === 必须实现 ===
    @abstractmethod
    def reset(self):
        """重置内部状态，新 episode 开始时调用"""
    
    # === 通常需要实现（使用 AgentExecutionEngine 时） ===
    def update_from_env(self, observation, reward, done, info):
        """环境返回后更新 Agent 状态。将 obs 追加到内部 message 历史"""
    
    def update_from_model(self, response: str) -> Action:
        """模型生成后更新 Agent 状态。解析 response 提取 action"""
    
    # === 属性（核心接口） ===
    @property
    def chat_completions(self) -> list[dict]:
        """返回当前完整对话历史，用于调用 LLM"""
    
    @property
    def trajectory(self) -> Trajectory:
        """返回当前积累的 Trajectory 对象"""
    
    # === 可选 ===
    def get_current_state(self) -> Step | None:
        """返回最后一个 step 的快照"""
```

### 自定义 Agent 典型实现模式

```python
class MyAgent(BaseAgent):
    def __init__(self):
        self.messages = []      # 对话历史
        self._trajectory = Trajectory(name="solver")
    
    def reset(self):
        self.messages = []
        self._trajectory = Trajectory(name="solver")
    
    def update_from_env(self, obs, reward, done, info):
        self.messages.append({"role": "user", "content": str(obs)})
        # 可存入 trajectory step
    
    def update_from_model(self, response: str) -> Action:
        self.messages.append({"role": "assistant", "content": response})
        action = parse_action(response)
        # 构建 Step 加入 trajectory
        self._trajectory.steps.append(Step(
            chat_completions=list(self.messages),
            model_response=response,
            action=action,
        ))
        return Action(action=action)
    
    @property
    def chat_completions(self) -> list[dict]:
        return self.messages      # 执行引擎用来调 LLM
    
    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory   # 执行引擎用来收集训练数据
```

> **约束**：`chat_completions` 返回的消息列表必须是完整的对话历史（包含 system、所有 user 和 assistant 消息），因为执行引擎直接传给 `RolloutEngine.get_model_response()`。

---

## 🔴 3. 自定义 Environment 指南

### 3.1 基本接口

```python
class BaseEnv(ABC):
    def reset(self, task=None) -> tuple[observation, info_dict]:
        """初始化环境，返回初始观察"""
    
    def step(self, action) -> tuple[observation, reward, done, info]:
        """执行动作，返回 (观察, 奖励, 是否结束, 信息)"""
    
    @classmethod
    def is_multithread_safe(cls) -> bool:
        """声明环境是否线程安全（默认 True）"""
        return True
```

### 3.2 异步环境

执行引擎内部使用 `ThreadPoolExecutor` 运行环境调用：
```python
# agent_execution_engine.py
async def _step_env(env, action):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, env.step, action)
```

因此**同步的**环境可以直接使用，不需要特殊的异步实现。如果环境内部需要等待外部 API，直接在 `step()` 中阻塞即可。

### 3.3 `is_multithread_safe` 的含义

- 返回 `True`：同一个环境实例可在不同线程中并发调用（有状态隔离）
- 返回 `False`：执行引擎会为每个并发 workflow 创建独立的环境实例

### 3.4 多 Agent 环境

rLLM 当前设计为**每个 workflow 独立运行一个 agent-env 交互循环**。多 Agent 需通过 Solver-Judge 模式在 Workflow 层面编排（见第4节）。

---

## 🔴 4. 自定义 Workflow 与 Solver-Judge 模式

### 4.1 Solver-Judge 实现 — 真实示例

来自 [`cookbooks/solver_judge_flow/solver_judge_flow.py`](file:///home/robomaster/Research/rllm/cookbooks/solver_judge_flow/solver_judge_flow.py)：

```python
@rllm.rollout(name="solver-judge")
def solver_judge_flow(task, config) -> Episode:
    # Step 1: Solver 并行生成 N 个方案
    solver_trajectories = _generate_solutions(client, model, problem)
    # 每个 trajectory.name = "solver"
    
    # Step 2: Judge 选择最佳方案
    judge_trajectory = _judge_solutions(client, model, problem, solutions)
    # trajectory.name = "judge"
    
    return Episode(trajectories=[*solver_trajectories, judge_trajectory])
```

### 4.2 TrajectoryGroup `group_role` 机制

Episode 中的 Trajectory 通过 `name` 区分角色：

```
Episode (task_id="q1")
├── Trajectory(name="solver")  ← solver 角色
├── Trajectory(name="solver")  ← solver 角色（另一个方案）
└── Trajectory(name="judge")   ← judge 角色
```

`transform.py` 中 `_build_trajectory_groups()` 按 `task_id:trajectory.name` 分组：
```
TrajectoryGroup(group_id="q1:solver") → 包含该 task 所有 rollout 的 solver trajectories
TrajectoryGroup(group_id="q1:judge")  → 包含该 task 所有 rollout 的 judge trajectories
```

`group_role` 从 `group_id` 提取：`"q1:solver".split(":")[1]` = `"solver"`

### 4.3 不同角色使用不同优势算法

通过 `traj_group_adv_estimator_map` 配置：

```python
UnifiedTrainer(
    ...,
    traj_group_adv_estimator_map={
        "solver": "grpo",        # solver 用 GRPO
        "judge": "reinforce",    # judge 用 REINFORCE
    }
)
```

### 4.4 树搜索/MCTS

Workflow 本身**不直接支持分支回溯**。但用户可以在自定义 Workflow 的 `run()` 中实现树搜索，将最终选择的路径作为 trajectory 提交。框架不限制 `run()` 内的逻辑，只关心最终的 Episode 返回值。

---

## 🔴 5. 多步信用分配机制

### 5.1 两种优势分配模式

| 模式 | 配置 | 机制 | 适用场景 |
|------|------|------|---------|
| **broadcast** | `stepwise_advantage.mode: broadcast` | 同一 trajectory 所有 step 共享相同 advantage | 推荐，大多数场景 |
| **per_step** | 已弃用 | 每个 step 独立 advantage | 旧版兼容 |

### 5.2 MC Return 实现

[`env_utils.py:28-47`](file:///home/robomaster/Research/rllm/rllm/environments/env_utils.py#L28-L47)：

```python
def compute_mc_return(trajectory, gamma=0.95):
    G = 0.0
    for step in reversed(trajectory.steps):  # 从后向前
        G = step.reward + gamma * G
        step.mc_return = G
    return trajectory
```

MC return 在 verl 后端传入训练器后置于 response 的最后一个 token 位置：
```python
mc_return_batch[step_index, resp_len - 1] = all_mc_returns[step_index]
```

### 5.3 Critic-based Value Estimation

verl 后端的 PPO 训练器支持完整的 Critic 模型：
- `compute_ref_log_probs` — 参考模型前向计算 KL 正则项
- GAE (Generalized Advantage Estimation) — 通过 Critic 的 value 估计实现更精细的信用分配
- 这在 `agent_ppo_trainer.py` (51K行) 中实现，是 verl 原生能力

---

## 🟡 6. 端到端运行示例

### 可用示例

| 示例 | 路径 | 场景 |
|------|------|------|
| Solver-Judge Cookbook | `cookbooks/solver_judge_flow/` | 多 Agent 训练 |
| Solver-Judge (verl) | `examples/solver_judge/train_solver_judge_flow.py` | 分布式训练 |
| Solver-Judge (tinker) | `examples/solver_judge_tinker/` | 单机训练 |
| Solver-Judge (SDK) | `examples/sdk/solver_judge/` | SDK 模式 |
| Geo3K | `cookbooks/geo3k/` | VLM 训练 |
| Code | `examples/code/` | 代码生成 |
| Search | `examples/search/` | 搜索 Agent |

### 典型训练启动

```bash
# Tinker (单机)
python examples/solver_judge_tinker/train_solver_judge_flow_tinker.py

# Verl (分布式)
bash examples/solver_judge/train_solver_judge_flow.sh
```

---

## 🟡 7. 调试与开发体验

### EpisodeLogger 输出

每个 episode 保存为 JSON 文件到 `logs/{project}/{experiment}/episodes/`：
```json
{
  "id": "task1:0:1",
  "is_correct": true,
  "trajectories": [{
    "name": "solver",
    "steps": [{
      "chat_completions": [...],
      "model_response": "...",
      "reward": 1.0
    }]
  }]
}
```

### 开发迭代建议

```
1. 实现 Agent + Environment → 本地 Python 测试
2. 用 Tinker 后端 + SimpleWorkflow → 快速验证训练循环
3. 切换 verl 后端 → 分布式训练
```

Tinker → verl 切换只需改一行：`backend="tinker"` → `backend="verl"`。

---

## 🟡 8. 性能与扩展性

### 并发数 N 的决定因素

| 因素 | 约束 |
|------|------|
| `n_parallel_tasks` 配置 | 直接限制异步 Semaphore |
| vLLM batch size | GPU 显存限制推理批大小 |
| 环境资源 | 如 Docker/浏览器实例数 |
| CPU/内存 | 每个 workflow 的开销 |

典型值：128-256 个并发 workflow（远大于 GPU 数量，因为大部分时间在等待）。

### 工具调用延迟

`TimingTrackingMixin` 追踪每步的 LLM 和 env 耗时：
```python
async def timed_llm_call(self, ...):
    with timer("llm_time"):
        return await self.rollout_engine.get_model_response(...)

async def timed_env_call(self, func, ...):
    with timer("env_time"):
        return func(...)
```

指标自动记录到 episode.metrics 中。

---

## 🟡 9. 模型权重管理

### wake_up/sleep 权重同步

**verl 使用引用传递**——训练完成后，actor 权重原地更新（FSDP shard），vLLM 通过 `wake_up()` 从 actor 模型参数**直接拷贝**最新权重（完整拷贝），不是增量同步。

### 全异步训练的 param_sync

`param_sync.py` 实现从训练 GPU 到推理 GPU 的权重同步：
- 存在 **staleness**——推理使用的是上一轮或更早的权重
- 这是有意设计，论文（如 IMPALA）已证明适度的 staleness 不影响收敛
- 同步频率可配置

---

## 🟡 10. 与现有 Benchmark 集成

### 已内置支持

| Benchmark | 支持方式 |
|-----------|---------|
| SWE-bench | `rllm/environments/swe/` 专用环境 |
| BrowserGym | `rllm/environments/browsergym/` 封装 |
| AppWorld | `rllm/environments/appworld/` 封装 |
| GAIA | 可通过 QA transform + 搜索工具环境实现 |
| WebArena | 通过 BrowserGym 环境适配 |
| 50+ 评估数据集 | `rllm/data/transforms.py` 已覆盖 |

### Offline RL

rLLM 当前**不原生支持 offline RL**（从离线数据训练）。框架设计围绕在线 rollout 收集。但可以通过：
1. 将离线数据构造为 `Episode` 对象
2. 跳过 Stage 1（generate_episodes），直接进入 Stage 2
3. 使用预计算的 advantage（`use_precomputed_advantage: true`）

实现类 offline 的训练流程。

---

## 🟢 11. Examples 目录完整解析

> `examples/` 包含 20 个子项目，覆盖了从简单数学推理到 SWE-Bench 编程代理的全部场景。按**实现模式**分类如下：

### 11.1 模式全景图

| 模式 | 示例 | 核心组件 | 训练后端 |
|------|------|----------|----------|
| **单轮数学推理** | `simple_math`, `gsm8k_lora`, `deepscaler`, `countdown` | `MathAgent` + `SingleTurnEnv` | verl |
| **多步工具调用** | `math_tool`, `search`, `deepcoder` | `ToolAgent` + `ToolEnvironment` | verl |
| **环境交互 Agent** | `frozenlake` | `FrozenLakeAgent` + `FrozenLakeEnv` | verl |
| **多 Agent 协作** | `solver_judge`, `solver_judge_tinker`, `solver_judge_modal` | `SolverJudgeWorkflow` | verl / tinker |
| **第三方环境集成** | `verifiers_env`, `eval_protocol` | `VerifiersWorkflow` / `EvalProtocolWorkflow` | verl / tinker |
| **SWE 编程代理** | `swe` | `SWEEnv` + R2E-Gym | verl (K8s) |
| **Distillation** | `math_distill` | `DistillationWorkflow` | tinker |
| **完全异步训练** | `fully_async/deepresearch` | `AsyncAgentTrainer` + 自定义 `rollout_fn` | verl (experimental) |
| **SDK 模式** | `sdk` | LiteLLM Proxy + `SimpleWorkflow`/`SolverJudgeWorkflow` | verl |

---

### 11.2 单轮数学推理（最简模式）

**代表项目**：`simple_math`, `gsm8k_lora`, `deepscaler`, `countdown`

最小训练脚本结构（以 `gsm8k_lora` 为例）：

```python
# examples/gsm8k_lora/train_gsm8k_with_lora.py
@hydra.main(config_name="agent_ppo_trainer")
def main(config):
    train_dataset = DatasetRegistry.load_dataset("gsm8k", "train")
    test_dataset  = DatasetRegistry.load_dataset("gsm8k", "test")

    trainer = AgentTrainer(
        agent_class=MathAgent,
        env_class=SingleTurnEnvironment,
        env_args={"reward_fn": math_reward_fn},
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()
```

**关键点**：
- `SingleTurnEnvironment`：单步环境，`step()` 直接调用 `reward_fn` 评分，无需多轮交互
- `MathAgent`：内置 `\boxed{}` 格式解析逻辑
- **LoRA 支持**：在 YAML config 中设置 `actor_rollout_ref.model.lora_rank=32` 即可启用 LoRA

**DeepScaler 的迭代上下文增长**：

```bash
# 三阶段迭代训练，上下文从 8K 逐步扩展到 24K
bash examples/deepscaler/train_deepscaler_8k.sh   # 阶段 1
bash examples/deepscaler/train_deepscaler_16k.sh  # 阶段 2，需传入上一阶段 ckpt
bash examples/deepscaler/train_deepscaler_24k.sh  # 阶段 3
```

> **设计意图**：Iterative Context Lengthening (ICL)——先训练短上下文，再用 checkpoint 继续扩展，比直接训练长上下文更稳定。同样应用于 `deepcoder`（16K→32K→64K）。

---

### 11.3 多步工具调用 Agent

**代表项目**：`math_tool`, `search`, `deepcoder`

#### ToolAgent + ToolEnvironment 模式

```python
# examples/search/run_search_agent.py
tool_map = {"local_search": LocalRetrievalTool}  # 工具名 → 工具类

engine = AgentExecutionEngine(
    agent_class=ToolAgent,
    agent_args={"tool_map": tool_map, "system_prompt": SEARCH_SYSTEM_PROMPT, "parser_name": "qwen"},
    env_class=ToolEnvironment,
    env_args={"tool_map": tool_map, "reward_fn": search_reward_fn},
    ...
)
```

**ToolEnvironment 工作机制**：
1. Agent 生成包含 `<tool_call>...</tool_call>` XML 标签的响应
2. `ToolEnvironment.step()` 解析工具调用，执行 `tool_map[tool_name](**args)`
3. 工具结果作为新观察返回给 Agent
4. 最终轮次（无工具调用或达到 `max_steps`）计算奖励

**Search Agent 架构**：
```
HotpotQA 问题 → ToolAgent (qwen parser)
    → 发出 <tool_call>local_search</tool_call>
    → LocalRetrievalTool (E5 密集检索 Wikipedia)
    → 检索结果作为观察返回
    → 迭代直到得出最终答案
    → search_reward_fn 计算 F1/EM 奖励
```

**更换 parser 支持不同模型**：`parser_name` 控制工具调用格式解析，`"qwen"` 对应 Qwen3 的工具调用格式。

---

### 11.4 经典 RL 环境交互（FrozenLake）

**代表项目**：`frozenlake`

这是 rLLM 框架的**教学示例**，展示如何将经典 RL 环境接入 LLM 训练：

```python
# examples/frozenlake/run_frozenlake_agent.py
engine = AgentExecutionEngine(
    agent_class=FrozenLakeAgent,
    env_class=FrozenLakeEnv,
    agent_args={"max_steps": 10, "use_accumulate_history": True},
    env_args={"max_steps": 8, "is_slippery": False},
    ...
    n_parallel_agents=256,  # 256 并发 agent 同时运行
)
```

**数据集生成策略**：
```python
# prepare_frozenlake_data.py
# 随机生成 10,000 个训练环境 + 100 个测试环境
# 每个环境参数随机化：size(2-10), slip_prob(0.6-0.85), seed
```

**`FrozenLakeAgent` 的 `use_accumulate_history`**：为 True 时将完整历史（包含所有步骤的观察和动作）传给 LLM，为 False 时只传当前状态，This 控制是否使用 in-context learning 进行多步推理。

训练配置使用 `rllm/experimental/common/rl_algo.py` 的 GRPO 算法，在同一 FrozenLake 地图的 4 次 rollout 之间做组内优势归一化。

---

### 11.5 自定义 Agent 完整实现（MathAgentWithFewshot）

**代表项目**：`math_tinker`

`MathAgentWithFewshot` 是 [`BaseAgent`](file:///home/robomaster/Research/rllm/rllm/agents/agent.py) 的完整自定义实现，展示了所有关键接口的使用：

```python
# examples/math_tinker/math_agent_with_fewshot.py
class MathAgentWithFewshot(BaseAgent):
    STANDARD_FEWSHOT_PREFIX = [...]  # 标准 few-shot 示例，reset() 后保留

    def reset(self):
        self._trajectory = Trajectory()
        self.messages = []
        if self.use_fewshot:
            self.messages.extend(copy.deepcopy(self.STANDARD_FEWSHOT_PREFIX))  # 注意 deepcopy！

    def update_from_env(self, observation, reward, done, info, **kwargs):
        # 区分两种情况：
        # 1. observation=None → 只更新当前 step 的 reward/done
        # 2. observation=非None → 创建新 step，添加到 trajectory
        if observation is None:
            self.get_current_state().reward = reward
            return
        self.messages.append({"role": "user", "content": formatted_obs})
        self._trajectory.steps.append(Step(observation=formatted_obs))

    def update_from_model(self, response: str, **kwargs) -> Action:
        self.messages.append({"role": "assistant", "content": response})
        # 解析 <think>...</think> 思维链
        if "</think>" in response:
            thought, _, action_str = response.partition("</think>")
        cur_step.thought = thought
        cur_step.chat_completions = self.chat_completions
        return Action(response.strip())

    @property
    def chat_completions(self) -> list[dict]:
        # 可选：不积累历史中的 <think> 部分以减少上下文
        if not self.accumulate_thinking:
            # 清除历史 assistant 消息的 <think> 部分
            ...
        return messages
```

**关键细节**：
- `reset()` 保留了 few-shot 前缀——这是 multi-episode 训练中的常见需求
- `chat_completions` 可以过滤掉历史中的 `<think>` 标签，减少上下文长度
- `update_from_env` 区分"纯奖励更新"和"新观察"两种情形

---

### 11.6 多 Agent 协作：Solver-Judge 实战

**代表项目**：`solver_judge`, `solver_judge_tinker`, `solver_judge_modal`

#### 完整 Workflow 实现

```python
# examples/solver_judge/solver_judge_flow.py
class SolverJudgeWorkflow(Workflow):
    def __init__(self, rollout_engine, n_solutions=2, reward_function=None, **kwargs):
        super().__init__(rollout_engine, **kwargs)
        self.solver = Solver(rollout_engine)  # 使用同一个 rollout_engine
        self.judge  = Judge(rollout_engine)   # Solver 和 Judge 共享 LLM

    async def run(self, task: dict, uid: str) -> Episode:
        problem = task["question"]

        # Step 1: Solver 并行生成 N 个方案
        solver_trajectories = await self.solver.generate_solutions(problem, self.n_solutions)

        # Step 2: 逐个评分
        for traj in solver_trajectories:
            traj.steps[0].reward = self.reward_function(task, traj.steps[0].action).reward

        # Step 3: Judge 选最佳方案
        judge_traj = await self.judge.judge_solutions(problem, solutions)
        judge_traj.steps[0].reward = self.reward_function(task, selected_solution).reward

        # Step 4: 返回包含所有角色轨迹的 Episode
        return Episode(
            id=uid,
            task=task,
            trajectories=[*solver_trajectories, judge_traj],  # 角色通过 trajectory.name 区分
            is_correct=is_correct,
            metrics={"solver_acc": ..., "judge_acc": ...},
        )
```

#### 三种部署变体对比

| 变体 | 路径 | 特点 |
|------|------|------|
| **verl** | `solver_judge/` | 本地 GPU 分布式训练，支持 CoUntdown 类比 |
| **tinker** | `solver_judge_tinker/` | 远端推理，LoRA 微调，单机即可运行 |
| **modal** | `solver_judge_modal/` | 云端 Modal 部署，`modal_deploy.py` 管理 GPU 资源 |

**奖励分配策略**：solver 和 judge 轨迹分别使用独立的 `traj_group_adv_estimator_map` 配置，可以为每个角色选择不同的优势估计算法（如 solver 用 GRPO，judge 用 REINFORCE）。

---

### 11.7 第三方环境集成

#### 11.7.1 Verifiers 环境

**代表项目**：`verifiers_env`

核心：`VerifiersWorkflow` 将 rLLM 的 `RolloutEngine` 包装为 `AsyncOpenAI` 兼容接口供 Verifiers 使用。

```python
# examples/verifiers_env/workflow.py
class VerifiersWorkflow(Workflow):
    async def run(self, task, uid) -> Episode:
        # 1. 将 rLLM RolloutEngine 包装为 AsyncOpenAI 客户端
        client = RolloutEngineAsyncClient(rollout_engine=self.rollout_engine, ...)

        # 2. 执行 Verifiers 环境的 rollout
        state = await self.vf_env.rollout(input=rollout_input, client=client, ...)
        await self.vf_env.rubric.score_rollout(state, ...)

        # 3. 将 Verifiers State → rLLM Episode
        return self._convert_state_to_episode(state, uid)

    def _convert_state_to_episode(self, state, uid) -> Episode:
        # Verifiers 的 TrajectoryStep 包含 tokens: {prompt_ids, completion_ids, completion_logprobs}
        # 直接映射到 rLLM Step 的 prompt_ids / response_ids / logprobs
        steps = [Step(
            prompt_ids=traj_step["tokens"]["prompt_ids"],
            response_ids=traj_step["tokens"]["completion_ids"],
            logprobs=traj_step["tokens"]["completion_logprobs"],
            reward=traj_step.get("reward", 0.0),
        ) for traj_step in state["trajectory"]]
        ...
```

**架构图**：
```
AgentTrainer
    └─ VerifiersWorkflow.run()
           └─ RolloutEngineAsyncClient  ← 适配器层
                   └─ RolloutEngine     ← rLLM 推理引擎
           └─ vf_env.rollout()          ← Verifiers 管理对话轮次
           └─ vf_env.rubric.score()     ← Verifiers rubric 评分
```

#### 11.7.2 Eval Protocol 环境

**代表项目**：`eval_protocol`

`EvalProtocolWorkflow` 通过 MCP（Model Context Protocol）Server 接入 Eval Protocol 基准。特点：
- 使用 Fireworks AI 作为推理后端（非本地 vLLM）
- 支持 FrozenLake、WebArena、AppWorld 等标准 benchmark

```bash
# 推理
python examples/eval_protocol/run_frozen_lake_flow.py

# 训练
bash examples/eval_protocol/train_frozen_lake_flow.sh
```

---

### 11.8 SWE-Bench 编程代理（DeepSWE）

**代表项目**：`swe`

**成果**：DeepSWE-Preview 在 SWE-Bench-Verified 上达到 **59.2% Pass@16, 42.2% Pass@1**（开源 SOTA）。

**技术栈**：
```
rLLM + R2E-Gym + Kubernetes + Docker
  ├── 模型：Qwen3-32B（初始化）
  ├── 环境：R2E-Gym SWE-Bench Docker 容器（1000 个并行）
  ├── 基础设施：AWS/GCP K8s 集群（64+ GPUs，200 CPU/node, 6TB+ 磁盘）
  └── 规模：512 个并行 Docker 容器
```

**`SWEEnv` 使用方式**（见 [`rllm/environments/swe/swe.py`](file:///home/robomaster/Research/rllm/rllm/environments/swe/swe.py)）：

```python
from rllm.environments.swe.swe import SWEEnv
from datasets import load_dataset

ds = load_dataset("R2E-Gym/R2E-Gym-Subset", split="train")
env = SWEEnv(entry=ds[0], backend='kubernetes', scaffold='r2egym')
env.reset()   # 启动容器
env.step(patch_content)  # 提交代码修改
env.close()  # 销毁容器
```

**训练脚本**（需要 K8s 集群）：
```bash
bash examples/swe/train_deepswe_32b.sh
```

---

### 11.9 On-Policy Distillation（`math_distill`）

**代表项目**：`math_distill`

**核心思想**：用学生模型采样轨迹，用教师模型的 log-prob 计算 per-token 优势，而非稀疏的 RL 奖励。

```
advantage[t] = log P_teacher(token_t) - log P_student(token_t)
```

**配置关键参数**：

```bash
python -m examples.math_distill.train_deepmath_distill_tinker \
    rllm/backend=tinker \
    model.name=Qwen/Qwen3-8B-Base \
    model.lora_rank=128 \
    rllm.algorithm.use_precomputed_advantage=true  # 跳过 RL 优势计算，直接用 workflow 计算的 advantage
    rllm.algorithm.loss_fn=importance_sampling \
    training.group_size=4
```

**流程**：
1. `DistillationWorkflow.run()` 采样学生轨迹，调用教师模型计算 `clip(-5, advantage, +5)`
2. `use_precomputed_advantage=true` → 框架跳过 GRPO/REINFORCE，直接使用 trajectory 中存储的 advantage
3. 教师模型（Qwen3-32B）通过 Tinker API 调用，学生（Qwen3-8B）本地 LoRA 训练

> **OPD vs RL**：OPD 拥有 dense per-token feedback（每个 token 都有梯度信号），RL 只有 sparse trajectory-level reward。OPD 一般收敛更快但依赖教师模型质量。

---

### 11.10 完全异步训练（`fully_async/deepresearch`）

**代表项目**：`fully_async/deepresearch`

使用实验性的 `AsyncAgentTrainer`，rollout 生成与模型训练完全并行（类 IMPALA 架构）。

```python
# examples/fully_async/deepresearch/train.py
@hydra.main(config_name="fully_async_ppo_trainer")
def main(config):
    trainer = AsyncAgentTrainer(
        config=config,
        rollout_fn=rollout_fn,      # 自定义 async rollout 函数
        val_rollout_fn=val_rollout_fn,
    )
    trainer.train()

async def rollout_fn(client, tokenizer, **kwargs):
    reward, metric = await rollout(client=client, tool=train_retriever_tool, **kwargs)
    trajectory = metric.pop("trajectory")
    trajectory.reward = reward
    
    # 追踪 staleness（推理使用的权重版本）
    trajectory.metadata = {
        "param_version_start": client.cur_version,
        "is_partial": param_version_start != param_version_end,  # 是否使用了过期权重
        "tool_calls_time": tool_calls_count,
    }
    return trajectory
```

**与标准 `AgentTrainer` 的区别**：

| 特性 | `AgentTrainer` | `AsyncAgentTrainer` |
|------|--------------|---------------------|
| rollout/train 关系 | 串行（collect → train） | 完全并行（IMPALA 风格） |
| 权重 staleness | 无 | 有，需要 off-policy 校正 |
| 自定义 rollout | 通过 Workflow 类 | 直接传 `rollout_fn` 函数 |
| 适用场景 | 大多数 RL 场景 | 超长 rollout（深度研究型 Agent） |

**DeepResearch Agent**：多步 RAG 搜索（E5 检索 Wikipedia），每次 rollout 可能包含数十次工具调用，完全异步以最大化 GPU 利用率。

---

### 11.11 SDK 模式（`sdk`）

**代表项目**：`sdk`

SDK 模式通过 **LiteLLM Proxy** 将外部 LLM 服务（如 cloud API）接入 rLLM 训练框架，不依赖本地 GPU 推理。

```
AgentTrainer
    └─ rllm.sdk.proxy (LiteLLM Proxy Server)
           ├─ 自动管理 Proxy 生命周期（subprocess 模式）
           └─ 路由到外部 LLM API（OpenAI, Anthropic, vLLM, etc.）
```

**启动方式**：
```bash
# Proxy 自动管理（推荐）
# 训练脚本中设置: rllm.sdk.proxy.mode=subprocess

# 手动管理
python -m rllm.sdk.proxy.litellm_server \
  --config litellm_proxy_config_autogen.yaml \
  --port 4000 \
  --cs-endpoint http://localhost:8000
```

SDK 目录内还包含 `tutorial_quickstart.ipynb` 和 `sdk/simple_math/`, `sdk/solver_judge/` 两个完整示例，适合快速上手。

---

### 11.12 Examples 目录结构速查

```
examples/
├── simple_math/          # 最简单：MathAgent + SingleTurnEnv
├── gsm8k_lora/           # LoRA 微调示例
├── countdown/            # 数学倒计时：自定义 Dataset + 标准训练
├── deepscaler/           # DeepScaleR：ICL (8K→16K→24K)
├── deepcoder/            # DeepCoder：代码推理，ICL (16K→32K→64K)
├── math_tool/            # 工具调用：Python 解释器工具
├── search/               # 搜索 Agent：E5 检索 + HotpotQA
├── frozenlake/           # 经典 RL 环境：多步导航
├── math_tinker/          # Tinker 后端 + 自定义 Agent + few-shot
├── math_distill/         # On-Policy Distillation
├── geo3k/                # VLM 训练：几何视觉推理
├── solver_judge/         # 多 Agent：verl 后端
├── solver_judge_tinker/  # 多 Agent：tinker 后端
├── solver_judge_modal/   # 多 Agent：Modal 云部署
├── swe/                  # SWE-Bench：DeepSWE-Preview
├── verifiers_env/        # 第三方 Verifiers 环境集成
├── eval_protocol/        # Eval Protocol + MCP Server 集成
├── fully_async/          # 完全异步训练（experimental）
│   └── deepresearch/     # 深度研究型 Agent
├── sdk/                  # SDK 模式：LiteLLM Proxy
└── archive/              # 旧版示例存档
```

---

## 📋 Review 疑问解决状态

| # | 疑问 | 状态 | 关键源文件 |
|---|------|------|-----------|
| 1 | RL 算法细节 | ✅ 已解答 | `rl_algo.py`, `advantage.py`, `config.py` |
| 2 | BaseAgent 接口 | ✅ 已解答 | `agents/agent.py:272-333` |
| 3 | 自定义 Environment | ✅ 已解答 | `environments/base_env.py`, `agent_execution_engine.py` |
| 4 | 自定义 Workflow / Solver-Judge | ✅ 已解答 | `cookbooks/solver_judge_flow/`, `transform.py` |
| 5 | 多步信用分配 | ✅ 已解答 | `env_utils.py`, `workflow.py:149-170` |
| 6 | 端到端示例 | ✅ 已解答 | `cookbooks/`, `examples/` |
| 7 | 调试体验 | ✅ 已解答 | `utils/episode_logger.py`, Tinker 后端 |
| 8 | 性能扩展 | ✅ 已解答 | `timing_mixin.py`, Semaphore 机制 |
| 9 | 权重管理 | ✅ 已解答 | `param_sync.py`, verl 内部机制 |
| 10 | Benchmark 集成 | ✅ 已解答 | `environments/`, `data/transforms.py` |
