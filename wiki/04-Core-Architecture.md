# 核心架构 (Core Architecture)

## 类型系统：Step → Trajectory → Episode

rLLM 的数据模型由三级嵌套结构组成，是整个框架的基石。这些类型定义在两层继承中——**轻量级 SDK 类型**（`rllm/types.py`）和**训练类型**（`rllm/agents/agent.py`），前者不依赖 PyTorch，后者添加训练所需的 tensor 字段。

> 源码参考：[rllm/agents/agent.py L1-333](../rllm/agents/agent.py)、[rllm/types.py](../rllm/types.py)

### 层级结构

```
Episode（一次任务的完整记录）[agent.py L189-243]
├── id: str                        # 格式 "task_id:rollout_idx"
├── task: dict                     # 原始任务数据
├── is_correct: bool               # 是否正确
├── termination_reason: TerminationReason  # 终止原因枚举
├── metrics: dict                  # 聚合指标
├── trajectories: list[Trajectory] # 所有 Agent 角色的轨迹
│   ├── Trajectory [agent.py L127-187]
│   │   ├── uid: str               # 唯一标识（UUID）
│   │   ├── name: str              # Agent 角色名 (如 "solver", "judge")
│   │   ├── task: dict             # 该轨迹对应的任务
│   │   ├── reward: float          # 轨迹级聚合奖励
│   │   └── steps: list[Step]      # 每步 LLM 调用的记录
│   │       ├── Step [agent.py L17-117]
│   │       │   ├── prompt_ids: list[int]     # prompt token IDs
│   │       │   ├── response_ids: list[int]   # response token IDs
│   │       │   ├── logprobs: list[float]     # 每个 token 的 log prob
│   │       │   ├── chat_completions: list[dict]  # 完整对话历史
│   │       │   ├── observation: Any          # 环境观察
│   │       │   ├── thought: str              # <think> 推理内容
│   │       │   ├── action: Any               # Agent 执行的动作
│   │       │   ├── model_response: str       # LLM 原始输出
│   │       │   ├── model_output: ModelOutput # 完整推理输出对象
│   │       │   ├── reward: float             # 步级奖励
│   │       │   ├── done: bool                # 是否结束
│   │       │   ├── mc_return: float          # 蒙特卡洛回报
│   │       │   └── advantage: list[float]|float|None  # 优势值
│   │       └── ...
│   └── ...
└── artifacts: dict                # 附加输出（如最终答案）
```

### model_post_init 自动回填

当 `Step` 对象从 `ModelOutput` 构建时，训练所需字段被自动填充（[agent.py L52-65](../rllm/agents/agent.py)）：

```python
def model_post_init(self, __context: Any) -> None:
    if self.model_output is None:
        return
    # 自动从 ModelOutput 回填 prompt_ids, response_ids, logprobs
    if len(self.prompt_ids) == 0 and self.model_output.prompt_ids is not None:
        self.prompt_ids = self.model_output.prompt_ids
    if len(self.response_ids) == 0 and self.model_output.completion_ids is not None:
        self.response_ids = self.model_output.completion_ids
    if len(self.logprobs) == 0 and self.model_output.logprobs is not None:
        self.logprobs = self.model_output.logprobs

    # 长度一致性校验
    if len(self.logprobs) > 0:
        assert len(self.response_ids) == len(self.logprobs), \
            f"length mismatch between response_ids and logprobs"
```

> 这个自动回填机制使得用户创建 Step 时只需传入 `model_output`，其余训练字段自动填充。

### Step.from_model_output 工厂方法

提供便捷的 Step 构建方式（[agent.py L106-117](../rllm/agents/agent.py)）：

```python
@classmethod
def from_model_output(cls, model_output: ModelOutput,
                      messages: list[dict] | None = None,
                      action: Any | None = None) -> Step:
    return cls(
        prompt_ids=model_output.prompt_ids or [],
        response_ids=model_output.completion_ids or [],
        logprobs=model_output.logprobs or [],
        chat_completions=(messages or []) + [
            {"role": "assistant", "content": model_output.content,
             "reasoning": model_output.reasoning}
        ],
        thought=model_output.reasoning or "",
        action=action,
        model_response=model_output.content or "",
        model_output=model_output,
    )
```

### TrajectoryGroup：优势计算分组

`TrajectoryGroup` ([agent.py L245-269](../rllm/agents/agent.py)) 是专为优势计算设计的分组结构——将同一 task 的同一 Agent 角色的多次 rollout 聚在一起，用于 GRPO 等算法在组内比较 reward：

```python
class TrajectoryGroup(BaseModel):
    trajectories: list[Trajectory]   # 同组轨迹列表
    group_id: str = ""               # 格式 "task_id:traj_name"
    metadata: list[dict] = Field(default_factory=list)

    @property
    def group_role(self) -> str:
        """提取角色名，如 'solver' 或 'judge'"""
        return self.group_id.split(":")[1] if ":" in self.group_id[:-1] else "all_groups"

    @property
    def task_id(self) -> str:
        return self.group_id.split(":")[0]
```

#### 分组示例

```
Episode (task_id="q1", rollout_idx=0)
├── Trajectory(name="solver", reward=0.8)  → group "q1:solver"
├── Trajectory(name="judge", reward=1.0)   → group "q1:judge"

Episode (task_id="q1", rollout_idx=1)
├── Trajectory(name="solver", reward=0.3)  → group "q1:solver"
├── Trajectory(name="judge", reward=0.0)   → group "q1:judge"

最终分组:
TrajectoryGroup(group_id="q1:solver")
  trajectories = [solver_traj_0(reward=0.8), solver_traj_1(reward=0.3)]
  → GRPO: A_0 = (0.8-0.55)/σ,  A_1 = (0.3-0.55)/σ

TrajectoryGroup(group_id="q1:judge")
  trajectories = [judge_traj_0(reward=1.0), judge_traj_1(reward=0.0)]
  → GRPO: A_0 = (1.0-0.5)/σ,  A_1 = (0.0-0.5)/σ
```

### Trajectory.is_cumulative()

验证多步对话中 `chat_completions` 是否严格累加（[agent.py L173-186](../rllm/agents/agent.py)）：

```python
def is_cumulative(self) -> bool:
    """
    返回 True 如果每步的 chat_completions 是前一步的严格超集（前缀关系）。
    
    这对多步 tokenize_and_mask_cumulative() 至关重要——
    如果不满足累积性，拼接的 token 序列可能不一致。
    """
    prev = None
    for step in self.steps:
        if prev is not None:
            prev_cc = prev.chat_completions
            curr_cc = step.chat_completions
            if not (len(curr_cc) >= len(prev_cc) and curr_cc[:len(prev_cc)] == prev_cc):
                return False
        prev = step
    return True
```

## 执行引擎概览

| 引擎 | 文件 | 行数 | 核心入口 | 适用场景 |
|------|------|------|---------|---------|
| `AgentExecutionEngine` | `agent_execution_engine.py` | 627 | `run_agent_trajectory_async()` | 标准 Agent-Env 交互 |
| `AgentWorkflowEngine` | `agent_workflow_engine.py` | 556 | `execute_tasks()` | 自定义工作流逻辑 |
| `AgentSdkEngine` | `agent_sdk_engine.py` | 723 | `execute_workflow()` | 任意框架 SDK 拦截 |

详细分析：
- [Agent 执行引擎](06-Agent-Execution-Engine.md)
- [工作流引擎](07-Workflow-Engine.md)
- [SDK 引擎](08-SDK-Engine.md)

## 工作流系统

`Workflow` 基类 ([workflow.py L32-291](../rllm/workflows/workflow.py)) 定义了自定义交互逻辑的标准接口：

```python
class Workflow(ABC):
    def __init__(self, rollout_engine: RolloutEngine, executor: ThreadPoolExecutor,
                 timeout=1e6, gamma=0.0, reward_bonus_coeff=0.0, **kwargs):
        self.rollout_engine = rollout_engine
        self.executor = executor
        self.timeout = int(timeout)
        self.gamma = gamma                    # MC 折扣因子
        self.reward_bonus_coeff = reward_bonus_coeff  # 差分奖励系数
        self._completed_trajectories: list[Trajectory] = []

    @abstractmethod
    async def run(self, task: dict, uid: str, **kwargs) -> Episode | None:
        """用户必须实现的核心方法"""
        ...
```

五种内置变体：

| 变体 | 文件 | 行数 | 用途 |
|------|------|------|------|
| `SimpleWorkflow` | `simple_workflow.py` | 68 | 纯问答，无环境 |
| `SingleTurnWorkflow` | `single_turn_workflow.py` | 58 | 单步 Agent-Env |
| `MultiTurnWorkflow` | `multi_turn_workflow.py` | 61 | 多步 Agent-Env |
| `CumulativeWorkflow` | `cumulative_workflow.py` | 72 | 多步 + token 预算 |
| `DistillationWorkflow` | `distillation_workflow.py` | 79 | on-policy 蒸馏 |

## 训练器

| 训练器 | 文件 | 行号 | 支持后端 |
|--------|------|------|----------|
| `AgentTrainer` | `agent_trainer.py` | L7-180 | verl, fireworks, tinker |
| `UnifiedTrainer` | `unified_trainer.py` | 25.8K | verl, tinker |

## 设计原则

| 原则 | 代码体现 |
|------|---------|
| **框架无关** | SDK 引擎通过 LiteLLM Proxy 拦截任意框架的 LLM 调用 |
| **渐进式复杂度** | `SimpleWorkflow` → `MultiTurnWorkflow` → 自定义 `Workflow` |
| **后端抽象** | `BackendProtocol` / `AgentTrainer` 隔离训练后端差异 |
| **异步优先** | 所有执行引擎基于 `asyncio` + `Semaphore` 并发 |
| **可组合** | 奖励函数、工作流、环境均可独立替换 |
| **显存高效** | `wake_up()`/`sleep()` 实现推理-训练 GPU 时分复用 |
| **双层类型** | SDK 不依赖 PyTorch；训练层添加 tensor 字段 |
