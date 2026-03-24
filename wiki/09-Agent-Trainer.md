# Agent 训练器 (Agent Trainer)

`AgentTrainer` 是 rLLM 的统一训练入口，提供单一类即可完成 Agent 的 RL 后训练，无论使用何种后端（VERL / Tinker / Fireworks）。

> 源码参考：[rllm/trainer/agent_trainer.py L7-180](../rllm/trainer/agent_trainer.py)

## 类定义

```python
class AgentTrainer:
    """
    支持两种训练范式：
    1. Workflow 模式：自定义 Workflow 子类控制 Agent-Env 交互逻辑
    2. Agent/Env 模式：标准 Agent-Env 解耦交互（仅 verl/tinker 后端）

    支持三种后端：
    - 'verl' (默认)：标准训练后端，同时支持 Workflow 和 Agent/Env 模式
    - 'tinker'：基于服务的训练后端
    - 'fireworks'：基于管道的训练后端，仅支持 Workflow 模式
    """
```

## 初始化参数

([L17-83](../rllm/trainer/agent_trainer.py#L17-L83))

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `workflow_class` | `type \| None` | `None` | Workflow 子类 |
| `workflow_args` | `dict \| None` | `None` | 传给 Workflow 构造器 |
| `agent_class` | `type \| None` | `None` | Agent 子类（与 workflow_class 互斥） |
| `env_class` | `type \| None` | `None` | Environment 子类 |
| `agent_args` | `dict \| None` | `None` | Agent 初始化参数 |
| `env_args` | `dict \| None` | `None` | Environment 初始化参数 |
| `config` | `dict \| list[str] \| None` | `None` | Hydra 配置覆盖 |
| `train_dataset` | `Dataset \| None` | `None` | 训练数据集 |
| `val_dataset` | `Dataset \| None` | `None` | 验证数据集 |
| `backend` | `Literal["verl","fireworks","tinker"]` | `"verl"` | 训练后端 |
| `agent_run_func` | `Callable \| None` | `None` | SDK 模式的 Agent 函数 |

### 参数验证规则

1. **Fireworks 后端**只支持 `workflow_class`，不允许 `agent_class` / `env_class` / `agent_args` / `env_args`
2. **Workflow 模式**（`use_workflow=True`）不允许同时传 `agent_class` / `env_class`——必须通过 `workflow_args` 传入
3. 数据集路径自动同步到 config：`config.data.train_files = dataset.get_verl_data_path()`

## train() 分派逻辑

```python
def train(self):
    if self.backend == "verl":      self._train_verl()
    elif self.backend == "fireworks": self._train_fireworks()
    elif self.backend == "tinker":  self._train_tinker()
```

## VERL 后端训练

`_train_verl()` ([L123-153](../rllm/trainer/agent_trainer.py#L123-L153))：

```python
def _train_verl(self):
    import ray
    from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env
    from rllm.trainer.verl.train_agent_ppo import TaskRunner

    # 1. Ray 初始化
    if not ray.is_initialized():
        from rllm.trainer.ray_init_utils import get_ray_init_settings
        ray_init_settings = get_ray_init_settings(self.config)
        ray.init(runtime_env=get_ppo_ray_runtime_env(), **ray_init_settings)

    # 2. 在 Ray 远程 Actor 中运行训练
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

**TaskRunner** 是 Ray remote Actor，在 `run()` 内部:
1. 加载 Hydra config + 数据集
2. 初始化 verl PPO Trainer（含 Actor/Critic/Ref 模型 FSDP worker）
3. 根据是否有 `workflow_class` 决定使用 `AgentWorkflowEngine` 还是 `AgentExecutionEngine`
4. 执行训练循环

## Tinker 后端训练

`_train_tinker()` ([L98-121](../rllm/trainer/agent_trainer.py#L98-L121))：

```python
def _train_tinker(self):
    if self.workflow_class is not None:
        from rllm.trainer.deprecated.tinker_workflow_trainer import TinkerWorkflowTrainer
        trainer = TinkerWorkflowTrainer(
            config=self.config,
            workflow_class=self.workflow_class,
            workflow_args=self.workflow_args,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
        )
    else:
        from rllm.trainer.deprecated.tinker_agent_trainer import TinkerAgentTrainer
        trainer = TinkerAgentTrainer(
            config=self.config,
            agent_class=self.agent_class, env_class=self.env_class,
            agent_args=self.agent_args, env_args=self.env_args,
            train_dataset=self.train_dataset, val_dataset=self.val_dataset,
        )
    trainer.fit_agent()
```

> Tinker 后端位于 `rllm/trainer/deprecated/`，使用 Tinker 的 HTTP API 而非 Ray。

## Fireworks 后端训练

`_train_fireworks()` ([L155-179](../rllm/trainer/agent_trainer.py#L155-L179))：

```python
def _train_fireworks(self):
    import ray
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
        ray.init(runtime_env=get_ppo_ray_runtime_env(),
                 num_cpus=self.config.ray_init.num_cpus)

    from rllm.trainer.verl.train_workflow_pipeline import PipelineTaskRunner
    runner = PipelineTaskRunner.remote()
    ray.get(runner.run.remote(
        config=self.config,
        workflow_class=self.workflow_class,
        workflow_args=self.workflow_args,
    ))
```

> 使用 Fireworks API 进行远程推理，本地只做训练。仅支持 Workflow 模式。

## 使用示例

### Agent/Env 模式

```python
from rllm.trainer import AgentTrainer
from rllm.data import Dataset

trainer = AgentTrainer(
    agent_class=MathToolAgent,
    env_class=MathEnv,
    agent_args={"tools": ["python_interpreter"]},
    env_args={"reward_fn": math_reward_fn},
    config=["data.train_batch_size=64", "actor_rollout_ref.model.path=Qwen/Qwen3-4B"],
    train_dataset=Dataset.from_parquet("data/gsm8k_train.parquet"),
    backend="verl",
)
trainer.train()
```

### Workflow 模式

```python
trainer = AgentTrainer(
    workflow_class=SolverJudgeWorkflow,
    workflow_args={
        "solver_cls": MathSolver,
        "judge_cls": MathJudge,
        "max_turns": 3,
    },
    config=config,
    backend="verl",
)
trainer.train()
```

### SDK 模式

```python
def my_agent(metadata, question, **kwargs):
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:4000")
    resp = client.chat.completions.create(model="m", messages=[...])
    return 1.0 if correct(resp) else 0.0

trainer = AgentTrainer(agent_run_func=my_agent, backend="verl", config=config)
trainer.train()
```

## 后端选择决策树

```
需要分布式多 GPU 训练？ ──是──→ verl
  │                              │
  否                              └─── 需要 SDK 模式？ ──是──→ verl + agent_run_func
  │                                    │
  ▼                                    否
需要远程 GPU 推理？ ──是──→ fireworks
  │
  否
  │
  ▼
开发/调试/单 GPU ──→ tinker
```
