# 概览 (Overview)

## 目的与范围

本文档是 rLLM（Reinforcement Learning for Language Models）框架的全面技术指南。它涵盖核心架构组件、训练管线和关键抽象，帮助读者深入理解如何通过强化学习实现语言模型 Agent 的后训练。

- 安装指引见 [安装与配置](02-Installation-and-Setup.md)
- 动手实例见 [快速入门](03-Quick-Start-Guide.md)
- 子系统详情见 [核心架构](04-Core-Architecture.md)、[训练后端](12-VERL-Backend.md)、[Agent 与环境](16-Agent-and-Environment-Interfaces.md)、[奖励系统](21-Reward-Function-Architecture.md)

> 源码参考：[README.md L1-131](../README.md)、[pyproject.toml L1-192](../pyproject.toml)

## 什么是 rLLM

rLLM 是一个用于**通过强化学习训练语言模型 Agent** 的开源框架。它提供：

- **统一训练 API**：跨多种后端训练 Agent——VERL（分布式训练）、Tinker（基于服务的训练）、Fireworks（基于 API 的训练）
- **灵活的 Agent-环境抽象**：支持多种交互范式（单步、多步、工作流制、SDK 制）
- **垂直领域实现**：代码生成、数学解题、软件工程和网页导航
- **完整的奖励系统**：沙盒化代码执行、符号/数值评分和任务定制评估

框架构建在标准 RL 库（Ray、PyTorch、FSDP）之上，集成现代 LLM 推理引擎（vLLM、SGLang），实现从单 GPU 开发到多节点分布式训练的平滑扩展。

> 源码参考：[README.md L24-40](../README.md)、[pyproject.toml L6-8](../pyproject.toml)

## 核心架构：代码实体映射

下图展示了 rLLM 的概念组件到代码实现的精确映射关系：

```
概念组件                          代码实现
─────────────────────────────────────────────────────────
训练入口                          AgentTrainer
                                  [rllm/trainer/agent_trainer.py L7-180]

执行引擎                          AgentExecutionEngine
                                  [rllm/engine/agent_execution_engine.py L29-622]
                                  AgentWorkflowEngine
                                  [rllm/engine/agent_workflow_engine.py L28-556]
                                  AgentSdkEngine
                                  [rllm/engine/agent_sdk_engine.py]

Agent 抽象                        BaseAgent
                                  [rllm/agents/agent.py L272-333]

环境抽象                          BaseEnv
                                  [rllm/environments/base/base_env.py]

推理引擎                          RolloutEngine → OpenAI/Verl/Tinker
                                  [rllm/engine/rollout/rollout_engine.py L55-67]

数据类型                          Step → Trajectory → Episode
                                  [rllm/agents/agent.py L17-243]

工作流系统                        Workflow (基类)
                                  [rllm/workflows/workflow.py L32-291]

奖励系统                          RewardFunction (Protocol)
                                  [rllm/rewards/reward_fn.py L13-28]
```

## 训练管线：从任务到模型更新

以下是端到端的训练流程，标注了每个阶段涉及的代码组件和具体行号：

```
┌─ 阶段 1: 数据加载 ─────────────────────────────────────────────┐
│  DatasetRegistry.load_dataset("gsm8k", "train")               │
│  [rllm/data/dataset.py]                                        │
│  ↓ parquet 文件                                                 │
│  DataLoader 创建 batch of task dicts                            │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ 阶段 2: Rollout (轨迹生成) ──────────────────────────────────┐
│  AgentExecutionEngine.trajectory_generator()                   │
│  [agent_execution_engine.py L509-556]                          │
│                                                                │
│  for 每个 task:                                                │
│    ├── rollout_engine.wake_up()  [L518-519]                    │
│    ├── semaphore 控制并发 [L521]                                │
│    ├── run_agent_trajectory_async() [L184-433]                 │
│    │   ├── env.reset() → agent.update_from_env() [L207-218]   │
│    │   ├── for step in max_steps:                              │
│    │   │   ├── get_model_response() [L247]                     │
│    │   │   ├── agent.update_from_model() [L263]                │
│    │   │   ├── env.step(action) [L270]                         │
│    │   │   ├── agent.update_from_env() [L289-294]              │
│    │   │   └── 终止条件检查 [L317-361]                          │
│    │   ├── compute_trajectory_reward() [L392]                  │
│    │   └── compute_mc_return(gamma) [L393]                     │
│    └── rollout_engine.sleep()  [L553-554]                      │
│                                                                │
│  输出: list[Trajectory] (Text 模式) 或 dict (Token 模式)        │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ 阶段 3: DataProto 转换 ──────────────────────────────────────┐
│  transform_results_for_verl(episodes, task_ids)                │
│  [agent_workflow_engine.py L229-514]                           │
│                                                                │
│  for 每个 episode:                                             │
│    for 每个 trajectory:                                        │
│      for 每个 step:                                            │
│        ├── 提取 prompt_ids, completion_ids [L319-323]          │
│        ├── 构建 response_mask (只对 assistant 计算 loss)        │
│        └── 收集 rollout_log_probs [L329-331]                   │
│                                                                │
│  Tensor 构建:                                                  │
│    ├── 左填充 prompt → pad_sequence_to_length [L402-409]       │
│    ├── 右填充 response [L411-418]                              │
│    ├── input_ids = concat(prompt, response) [L420]             │
│    ├── attention_mask [L424-430]                               │
│    ├── position_ids [L440]                                     │
│    ├── traj_rewards → 放在最后一个 token 位置 [L447-454]        │
│    └── compact_filtering [L467-473]                            │
│                                                                │
│  输出: verl.DataProto                                          │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ 阶段 4: RL 训练更新 ─────────────────────────────────────────┐
│  verl PPO Trainer 接管 DataProto:                              │
│    ├── 参考模型前向 → ref_log_probs                             │
│    ├── Critic 前向 → value estimates                           │
│    ├── 优势计算 (GRPO/REINFORCE/RLOO)                          │
│    └── PPO/GRPO 策略梯度更新                                    │
│                                                                │
│  [rllm/trainer/verl/train_agent_ppo.py]                        │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ 阶段 5: 权重更新 → 下一轮迭代 ──────────────────────────────┐
│  VerlEngine.wake_up() → vLLM 加载新权重                        │
│  → 回到阶段 2                                                  │
└────────────────────────────────────────────────────────────────┘
```

## 关键组件概览

### AgentTrainer：统一训练接口

`AgentTrainer` 类 ([rllm/trainer/agent_trainer.py L7-180](../rllm/trainer/agent_trainer.py)) 提供单一入口进行 Agent 训练，无论使用何种后端：

```python
class AgentTrainer:
    def __init__(self,
        workflow_class: type | None = None,      # 工作流类
        workflow_args: dict | None = None,        # 工作流参数
        agent_class: type | None = None,          # Agent 类
        env_class: type | None = None,            # 环境类
        agent_args: dict | None = None,           # Agent 参数
        env_args: dict | None = None,             # 环境参数
        config: dict | list[str] | None = None,   # 训练配置（Hydra override）
        train_dataset: Dataset | None = None,     # 训练数据集
        val_dataset: Dataset | None = None,       # 验证数据集
        backend: Literal["verl", "fireworks", "tinker"] = "verl",  # 后端选择
        agent_run_func: Callable | None = None,   # SDK 模式函数
    ): ...
```

训练器基于 `backend` 参数分派到后端特定实现：

| 后端 | 分派方法 | 实际训练器 | 行号 |
|------|---------|-----------|------|
| `verl` | `_train_verl()` | Ray Actor → `TaskRunner` → verl PPO Trainer | L123-153 |
| `tinker` | `_train_tinker()` | `TinkerWorkflowTrainer` / `TinkerAgentTrainer` | L98-121 |
| `fireworks` | `_train_fireworks()` | `PipelineTaskRunner` | L155-179 |

> 源码参考：[rllm/trainer/agent_trainer.py L9-179](../rllm/trainer/agent_trainer.py)

### 执行引擎：轨迹生成

三种执行引擎处理不同的交互范式：

| 引擎 | 代码位置 | 核心方法 | 特点 |
|------|---------|---------|------|
| `AgentExecutionEngine` | [agent_execution_engine.py L29-622](../rllm/engine/agent_execution_engine.py) | `run_agent_trajectory_async()` | 直接控制 Agent-Env 循环 |
| `AgentWorkflowEngine` | [agent_workflow_engine.py L28-556](../rllm/engine/agent_workflow_engine.py) | `execute_tasks()` | 委托给 `Workflow.run()` |
| `AgentSdkEngine` | [agent_sdk_engine.py](../rllm/engine/agent_sdk_engine.py) | `execute_workflow()` | LiteLLM Proxy 拦截 |

所有引擎使用**异步执行 + Semaphore 并发控制**。

> 源码参考：[agent_execution_engine.py L29-622](../rllm/engine/agent_execution_engine.py)

### Rollout 引擎：LLM 推理抽象

`RolloutEngine` 接口 ([rollout_engine.py L55-67](../rllm/engine/rollout/rollout_engine.py)) 抽象了不同 LLM 推理后端：

```python
class RolloutEngine:
    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        raise NotImplementedError
    async def wake_up(self): pass
    async def sleep(self): pass
```

**`ModelOutput` 数据结构** ([rollout_engine.py L7-52](../rllm/engine/rollout/rollout_engine.py))：

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | `str \| None` | 完整文本输出 |
| `content` | `str \| None` | 内容部分（去除 `<think>` 推理标签） |
| `reasoning` | `str \| None` | 思考过程（`<think>` 标签内容） |
| `tool_calls` | `list[ToolCall] \| None` | 解析出的工具调用 |
| `prompt_ids` | `list[int] \| None` | prompt 的 token ID 序列 |
| `completion_ids` | `list[int] \| None` | completion 的 token ID 序列 |
| `multi_modal_inputs` | `dict \| None` | 多模态输入（图像等） |
| `logprobs` | `list[float] \| None` | 每个 completion token 的 log probability |
| `prompt_logprobs` | `list[float] \| None` | 每个 prompt token 的 log probability |
| `prompt_length` | `int` | prompt token 数 |
| `completion_length` | `int` | completion token 数 |
| `finish_reason` | `str \| None` | `"stop"` / `"length"` |

四种实现：

| 引擎 | 代码路径 | 推理方式 | 关键特性 |
|------|---------|---------|---------|
| `VerlEngine` | [rollout/verl_engine.py](../rllm/engine/rollout/verl_engine.py) (5.7K) | Python API → vLLM | `wake_up()`/`sleep()` GPU 共享 |
| `OpenAIEngine` | [rollout/openai_engine.py](../rllm/engine/rollout/openai_engine.py) (12K) | HTTP API | 兼容任何 OpenAI API |
| `TinkerEngine` | [rollout/tinker_engine.py](../rllm/engine/rollout/tinker_engine.py) (14.6K) | 本地/HTTP | 单机/CPU |
| `FireworksEngine` | [rollout/fireworks_engine.py](../rllm/engine/rollout/fireworks_engine.py) (8.3K) | HTTP API → Fireworks | 远程 GPU |

### Agent-环境框架

**BaseAgent** ([agent.py L272-333](../rllm/agents/agent.py))：

```python
class BaseAgent(ABC):
    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """将 Agent 内部状态转为 OpenAI chat completions 格式消息列表"""
        return []

    @property
    def trajectory(self) -> Trajectory:
        """返回当前积累的 Trajectory 对象"""
        return Trajectory()

    def update_from_env(self, observation, reward, done, info, **kwargs):
        """环境 step 后更新 Agent 状态"""
        raise NotImplementedError

    def update_from_model(self, response: str, **kwargs) -> Action:
        """模型生成后更新 Agent 状态，返回下一步 Action"""
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """重置 Agent 内部状态，每个新 episode 开始时调用"""
        return

    def get_current_state(self) -> Step | None:
        """返回最后一个 step 的快照"""
        if not self.trajectory.steps: return None
        return self.trajectory.steps[-1]
```

**BaseEnv** ([base_env.py](../rllm/environments/base/base_env.py))：

```python
class BaseEnv(ABC):
    @abstractmethod
    def reset(self, task=None) -> tuple[Any, dict]:
        """初始化环境，返回 (初始观察, info_dict)"""

    @abstractmethod
    def step(self, action) -> tuple[Any, float, bool, dict]:
        """执行动作，返回 (观察, 奖励, 是否结束, 信息)"""

    @classmethod
    def is_multithread_safe(cls) -> bool:
        """声明环境是否线程安全"""
        return True
```

### 配置系统

rLLM 使用 **Hydra** 作为配置管理，支持 YAML 文件 + 命令行覆盖：

```bash
python train.py \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    rllm.algorithm.adv_estimator=grpo \
    training.group_size=4
```

### 奖励系统

**RewardFunction 协议** ([reward_fn.py L13-28](../rllm/rewards/reward_fn.py))：

```python
@runtime_checkable
class RewardFunction(Protocol):
    def __call__(self, task_info: dict, action: str) -> RewardOutput: ...
```

**RewardOutput 数据结构** ([reward_types.py L77-90](../rllm/rewards/reward_types.py))：

```python
@dataclass(slots=True, kw_only=True)
class RewardOutput:
    reward: float                      # 奖励值 (0.0-1.0)
    metadata: dict = field(default_factory=dict)  # 计算元数据
    is_correct: bool | None = None     # 正确性标志
```

**RewardConfig** ([reward_types.py L11-35](../rllm/rewards/reward_types.py))：

```python
@dataclass
class RewardConfig:
    apply_format_reward: bool = False       # 是否应用格式奖励
    math_reward_weight: float = 1.0         # 数学奖励权重
    code_reward_weight: float = 1.0         # 代码奖励权重
    correct_reward: float = 1.0             # 正确奖励值
    incorrect_reward: float = 0.0           # 错误奖励值
    format_error_reward: float = 0.0        # 格式错误奖励
    toolcall_bonus: float = 0.5             # 工具调用 bonus
    use_together_code_interpreter: bool = False  # Together 代码沙盒
```

内置实现：

| 奖励函数 | 行号 | 说明 |
|----------|------|------|
| `math_reward_fn()` | L47-62 | 数学等价评估（精确→符号→数值） |
| `search_reward_fn()` | L65-84 | 搜索/QA 答案匹配 |
| `code_reward_fn()` | L87-102 | 沙盒化代码执行 + 测试验证 |
| `f1_reward_fn()` | L105-176 | Token 级 F1 分数（精确/召回） |
| `zero_reward()` | L32-44 | 恒返回 0 的占位奖励 |

## Ray 运行时环境

verl 后端的 Ray 初始化 ([agent_trainer.py L123-153](../rllm/trainer/agent_trainer.py))：

```python
def _train_verl(self):
    import ray
    from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env
    from rllm.trainer.verl.train_agent_ppo import TaskRunner

    if not ray.is_initialized():
        ray_init_settings = get_ray_init_settings(self.config)
        ray.init(runtime_env=get_ppo_ray_runtime_env(), **ray_init_settings)

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(
        config=self.config,
        workflow_class=self.workflow_class,
        workflow_args=self.workflow_args,
        agent_class=self.agent_class,
        env_class=self.env_class,
        agent_args=self.agent_args,
        env_args=self.env_args,
        agent_run_func=self.agent_run_func,
    ))
```

`get_ppo_ray_runtime_env()` ([ray_runtime_env.py L1-78](../rllm/trainer/verl/ray_runtime_env.py)) 确保 Ray 远程函数能正确导入 rLLM 包，通过设置 `working_dir` 和 `py_modules` 将当前项目目录传播到所有 Ray 工作节点。

## 多后端架构对比

| 维度 | VERL | Tinker | Fireworks |
|------|------|--------|-----------|
| **推理方式** | vLLM Python API（本地） | 本地/远程 HTTP | Fireworks HTTP API |
| **训练方式** | FSDP 分布式 | 单机梯度更新 | 管道式远程 |
| **GPU 需求** | 多 GPU 集群 | 单 GPU / CPU | 远程 GPU |
| **LoRA** | ✅ | ✅ | ✅ |
| **全参数** | ✅ | ✅ | ❌ |
| **分布式** | ✅ (Ray) | ❌ | ❌ |
| **VLM 多模态** | ✅ | 部分 | ❌ |
| **GPU 共享** | ✅ (wake_up/sleep) | N/A | N/A |
| **适用阶段** | 生产训练 | 开发/调试 | 评估/小规模 |
| **Python** | >= 3.10 | >= 3.11 | >= 3.10 |

## 数据流示例：FrozenLake 训练

以 FrozenLake 经典 RL 环境为例展示完整数据流：

```
1. prepare_frozenlake_data.py → 生成 10,000 个 FrozenLake 地图 → parquet 文件
2. AgentTrainer(
       agent_class=FrozenLakeAgent,
       env_class=FrozenLakeEnv,
       backend="verl",
       config=config,
       train_dataset=train_dataset,
   ).train()
3. _train_verl() → ray.init() + TaskRunner.remote()
4. 每个训练步:
   a. 从 DataLoader 取 batch_size 个 FrozenLake 地图
   b. VerlEngine.wake_up() → vLLM 加载权重
   c. 并发 N 个 run_agent_trajectory_async():
      - env.reset(map) → "S_FF\n_HFH\nFFFH\n_HFG"
      - agent.update_from_env(grid_observation)
      - LLM → "move_down"
      - env.step("move_down") → (new_grid, +0.1, False, {})
      - ... 重复直到 done/max_steps
      - compute_trajectory_reward() → 0.0 或 1.0
   d. VerlEngine.sleep()
   e. transform_results_for_verl() → DataProto
   f. PPO/GRPO → 策略更新
5. 保存 checkpoint → 回到步骤 4
```

## 安装简览

| 后端 | 安装命令 | 最低要求 |
|------|---------|---------|
| 基础 | `uv pip install -e .` | Python >= 3.10 |
| Tinker | `uv pip install -e .[tinker]` | Python >= 3.11 |
| VERL | `uv pip install -e .[verl]` | Python >= 3.10, CUDA |

> 详见 [安装与配置](02-Installation-and-Setup.md)。

## 下一步

| 目标 | 推荐页面 |
|------|---------|
| 安装框架 | [安装与配置](02-Installation-and-Setup.md) |
| 动手训练一个 Agent | [快速入门](03-Quick-Start-Guide.md) |
| 理解核心抽象 | [核心架构](04-Core-Architecture.md) |
| 搭建分布式训练 | [VERL 后端](12-VERL-Backend.md) |
| 集成自定义 Agent | [Agent 和环境接口](16-Agent-and-Environment-Interfaces.md) |
