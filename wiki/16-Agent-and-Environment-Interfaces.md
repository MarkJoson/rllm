# Agent 和环境接口 (Agent and Environment Interfaces)

本页详细介绍 rLLM 的 Agent 和环境抽象接口，以及它们如何在执行引擎中协同工作。

> 源码参考：
> - [rllm/agents/agent.py L272-333](../rllm/agents/agent.py)
> - [rllm/environments/base/base_env.py L1-80](../rllm/environments/base/base_env.py)

## BaseAgent 接口

([agent.py L272-333](../rllm/agents/agent.py#L272-L333))

`BaseAgent` 是所有 Agent 的抽象基类，定义了 Agent 与执行引擎交互的标准接口。

### 完整接口定义

```python
class BaseAgent(ABC):
    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """
        将 Agent 内部状态转为 OpenAI chat completions 格式。
        返回: [{"role": "system", "content": "..."}, 
               {"role": "user", "content": "..."}, 
               {"role": "assistant", "content": "..."}, ...]
        
        这是 Agent 与 LLM 交互的核心接口——
        执行引擎从此属性获取 prompt，输入到 rollout_engine。
        """
        return []

    @property
    def trajectory(self) -> Trajectory:
        """
        返回当前积累的 Trajectory 对象。
        Agent 应维护一个 Trajectory 实例，在每步 update 时添加 Step。
        """
        return Trajectory()

    def update_from_env(self, observation: Any, reward: float, 
                        done: bool, info: dict, **kwargs):
        """
        环境 step 后更新 Agent 状态。
        
        调用时机: env.reset() 后 和 env.step() 后
        职责:
        1. 存储新的 observation
        2. 更新内部对话历史（添加 user/environment 消息）
        3. 更新当前 Step 的 reward 和 done
        """
        raise NotImplementedError(
            "Subclasses must implement this if using AgentExecutionEngine"
        )

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        LLM 生成后更新 Agent 状态。
        
        调用时机: get_model_response() 返回后
        职责:
        1. 解析 LLM response（提取 thought、action）
        2. 添加 assistant 消息到对话历史
        3. 创建新的 Step 并追加到 trajectory
        4. 返回 Action（传给 env.step()）
        """
        raise NotImplementedError(
            "Subclasses must implement this if using AgentExecutionEngine"
        )

    @abstractmethod
    def reset(self):
        """
        重置 Agent 内部状态，每个新 episode 开始时调用。
        必须清空: 对话历史、trajectory、内部缓存。
        """
        return

    def get_current_state(self) -> Step | None:
        """返回最后一个 Step（可变引用），用于引擎更新 reward/done"""
        if not self.trajectory.steps:
            return None
        return self.trajectory.steps[-1]
```

### Agent 生命周期

在 `AgentExecutionEngine.run_agent_trajectory_async()` 中，Agent 的调用顺序：

```
agent.reset()                              # 每个 episode 开始
    │
agent.update_from_env(initial_obs, 0, False, info)  # 环境初始状态
    │
    ├─ messages = agent.chat_completions   # 构建初始 prompt
    │
    ▼ for each step:
    │
    ├─ prompt = agent.chat_completions.copy()
    │     ↓ LLM 推理
    ├─ action = agent.update_from_model(response)  # 解析响应
    │     ↓ 环境执行
    ├─ agent.update_from_env(obs, reward, done, info)  # 更新状态
    │     ↓
    ├─ cur_step = agent.get_current_state()  # 引擎修改 step
    │   cur_step.reward = reward
    │   cur_step.done = done
    │
    └─ 检查终止条件 → 继续或退出
```

### 自定义 Agent 示例

```python
class ToolAgent(BaseAgent):
    def __init__(self, system_prompt: str, tools: list[str]):
        self.system_prompt = system_prompt
        self.tools = tools
        self._trajectory = Trajectory()
        self._messages = []

    @property
    def chat_completions(self):
        return [{"role": "system", "content": self.system_prompt}] + self._messages

    @property
    def trajectory(self):
        return self._trajectory

    def reset(self):
        self._trajectory = Trajectory()
        self._messages = []

    def update_from_env(self, observation, reward, done, info, **kwargs):
        self._messages.append({"role": "user", "content": str(observation)})

    def update_from_model(self, response, **kwargs) -> Action:
        self._messages.append({"role": "assistant", "content": response})
        # 解析 action
        parsed_action = self._parse_action(response)
        # 创建 Step
        step = Step(
            observation=self._messages[-2]["content"],
            model_response=response,
            action=parsed_action,
            chat_completions=list(self.chat_completions),
        )
        self._trajectory.steps.append(step)
        return Action(action=parsed_action)
```

## BaseEnv 接口

([base_env.py L5-80](../rllm/environments/base/base_env.py#L5-L80))

`BaseEnv` 遵循 Gymnasium 标准接口，为 RL Agent 提供交互环境。

### 完整接口定义

```python
class BaseEnv(ABC):
    @property
    def idx(self) -> Any:
        """环境在 batch 中的索引（引擎自动设置）"""
        return getattr(self, "_idx", None)

    @idx.setter
    def idx(self, value: Any):
        self._idx = value

    @abstractmethod
    def reset(self) -> tuple[dict, dict]:
        """
        标准 Gym reset。返回 (初始观察, info_dict)。
        
        初始观察: 任何类型，通常是字符串（如问题文本或环境状态描述）。
        info: metadata 字典，通常包含 task 信息。
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        """
        标准 Gym step。返回 (观察, 奖励, 是否结束, info)。
        
        action: Agent 的行动（通常是解析后的字符串）
        观察: 新的环境状态
        奖励: 标量奖励
        是否结束: True 表示 episode 结束
        info: 额外元数据
        """
        pass

    def close(self):
        """清理资源（如 Docker 容器、浏览器实例）"""
        return

    @staticmethod
    @abstractmethod
    def from_dict(info: dict) -> "BaseEnv":
        """
        工厂方法——从字典创建环境实例。
        AgentExecutionEngine.execute_tasks() 使用此方法动态实例化环境。
        
        Args:
            info: 通常是 dataset 中的一条记录 (task dict)
        Returns:
            BaseEnv 实例
        """
        raise NotImplementedError

    @staticmethod
    def is_multithread_safe() -> bool:
        """
        声明环境是否线程安全。
        - True: 可由 AgentExecutionEngine 并发使用
        - False: 不允许——初始化时会抛出 AssertionError
        
        重要: 线程安全不意味着所有方法都是原子的，
        而是环境实例无共享可变状态。
        """
        return True
```

### 重要设计说明

**`is_multithread_safe()` 的含义**：

rLLM 的执行引擎使用 `asyncio` + `ThreadPoolExecutor` 并发执行多个环境。`is_multithread_safe()` 声明的是：这个环境类的**不同实例**可以安全地在不同线程中**并发**运行。

✅ 线程安全：每个实例有独立的状态（如独立的 Docker 容器、独立的文件路径）
❌ 非线程安全：多个实例共享全局资源（如全局变量、单例模型、共享 GPU 显存）

```python
# 线程安全的示例
class DockerSWEEnv(BaseEnv):
    def __init__(self, task):
        self.container = docker.create_container(...)  # 每实例独立容器

    @staticmethod
    def is_multithread_safe() -> bool:
        return True  # 不同容器互不干扰

# 非线程安全的示例
class SharedGPUEnv(BaseEnv):
    shared_model = load_model()  # 全局共享模型 ← 危险！

    @staticmethod
    def is_multithread_safe() -> bool:
        return False  # 共享可变状态
```

**引擎检查**（[agent_execution_engine.py L90-91](../rllm/engine/agent_execution_engine.py#L90-L91)）：

```python
if env_class is not None:
    assert env_class.is_multithread_safe(), "Environment must be multithread safe"
```

### compute_final_reward 可选方法

某些环境在 `step()` 过程中给出的是**中间奖励**，而在所有步完成后需要计算**最终奖励**：

```python
class MyEnv(BaseEnv):
    def compute_final_reward(self) -> float:
        """可选方法。step 循环结束后由引擎调用。"""
        return self._final_evaluation()
```

引擎调用（[agent_execution_engine.py L370-375](../rllm/engine/agent_execution_engine.py#L370-L375)）：

```python
if hasattr(env, "compute_final_reward") and not masked_out:
    reward = await loop.run_in_executor(self.executor, env.compute_final_reward)
    cur_step.reward = reward
```

### from_dict 工厂方法

当使用 `execute_tasks()` 时，引擎会从 task dict 动态创建环境：

```python
# agent_execution_engine.py L595
self.envs[index] = self.env_class.from_dict({**task, **self.env_args})
```

示例实现：

```python
class MathEnv(BaseEnv):
    def __init__(self, question: str, ground_truth: str, reward_fn=None):
        self.question = question
        self.ground_truth = ground_truth
        self.reward_fn = reward_fn or math_reward_fn

    @staticmethod
    def from_dict(info: dict) -> "MathEnv":
        return MathEnv(
            question=info["question"],
            ground_truth=info["ground_truth"],
            reward_fn=info.get("reward_fn", None),
        )

    def reset(self):
        return self.question, {"ground_truth": self.ground_truth}

    def step(self, action):
        result = self.reward_fn(
            {"ground_truth": self.ground_truth}, action
        )
        return "", result.reward, True, {"is_correct": result.is_correct}
```

## 与 Workflow 的集成

在 Workflow 模式下，Agent 和 Env 不再通过 `AgentExecutionEngine` 的循环交互，而是在 Workflow 的 `run()` 方法中由用户自由组合：

```python
class SolverJudgeWorkflow(Workflow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solver = MathSolverAgent(system_prompt="...")
        self.judge = MathJudgeAgent(system_prompt="...")
        self.env = MathEnv()

    async def run(self, task, uid, **kwargs):
        self.reset(task=task, uid=uid)  # 自动重置所有 Agent 和 Env
        
        obs, info = await self.run_in_executor(self.env.reset)
        
        # Solver 生成解答
        self.solver.update_from_env(obs, 0, False, info)
        response = await self.rollout_engine.get_model_response(
            self.solver.chat_completions
        )
        self.solver.update_from_model(response.content)
        self.commit(name="solver", agent=self.solver, reset=True)
        
        # Judge 评判
        self.judge.update_from_env(response.content, 0, False, info)
        judge_resp = await self.rollout_engine.get_model_response(
            self.judge.chat_completions
        )
        self.judge.update_from_model(judge_resp.content)
        self.judge.trajectory.steps[-1].reward = \
            1.0 if "correct" in judge_resp.content.lower() else 0.0
        self.commit(name="judge", agent=self.judge)
```

## 域特定实现

| 域 | Agent | Env | 详情页 |
|----|-------|-----|--------|
| Math | `ToolMathAgent` | `ToolMathEnv` | [数学与代码](19-Math-and-Code-Domains.md) |
| Code | `CodeAgent` | `CodeEnv` | [数学与代码](19-Math-and-Code-Domains.md) |
| SWE | `SWEAgent` | `SWEEnv` (Docker) | [SWE 域](17-SWE-Domain.md) |
| Web | `WebAgent` | `BrowserEnv` | [Web 导航](18-Web-Navigation-Domain.md) |
| Classic RL | `FrozenLakeAgent` | `FrozenLakeEnv` | [经典 RL](20-Classic-RL-Environments.md) |
