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
