# 快速入门 (Quick Start Guide)

本指南通过三个递进的示例帮助您快速上手 rLLM。

## 示例 1：SimpleWorkflow 数学训练

最简单的入门——纯问答，无环境交互。

### 1.1 准备数据

```bash
cd examples/math
python prepare_gsm8k_data.py
# 生成 data/gsm8k_train.parquet 和 data/gsm8k_test.parquet
```

**数据格式**（parquet 文件的每一行）：

| 列 | 类型 | 说明 |
|----|------|------|
| `extra_info` | `dict` | 包含 `question`、`answer`（ground_truth）等 |
| `data_source` | `str` | 数据集来源标识 |

### 1.2 定义 Workflow

```python
from rllm.workflows import SimpleWorkflow
from rllm.rewards.reward_fn import math_reward_fn

class MathSimpleWorkflow(SimpleWorkflow):
    """
    SimpleWorkflow 继承自 Workflow，自动处理:
    1. 构建 prompt (system_prompt + question)
    2. 调用 LLM 生成 response
    3. 创建 Step 并追加到 Trajectory
    4. 使用 reward_fn 计算奖励
    
    用户只需提供 system_prompt 和 reward_fn
    """
    def __init__(self, **kwargs):
        super().__init__(
            system_prompt="你是一名数学专家。请一步步思考，然后在 \\boxed{} 中给出最终答案。",
            reward_fn=math_reward_fn,
            **kwargs,
        )
```

### 1.3 启动训练

```python
from rllm.trainer import AgentTrainer
from rllm.data import Dataset

trainer = AgentTrainer(
    workflow_class=MathSimpleWorkflow,
    config=[
        "actor_rollout_ref.model.path=Qwen/Qwen3-1.7B",
        "actor_rollout_ref.rollout.temperature=0.7",
        "training.train_batch_size=32",
        "data.max_prompt_length=1024",
        "data.max_response_length=4096",
    ],
    train_dataset=Dataset.from_parquet("data/gsm8k_train.parquet"),
    val_dataset=Dataset.from_parquet("data/gsm8k_test.parquet"),
    backend="verl",
)
trainer.train()
```

或使用命令行：

```bash
python -m rllm.trainer.verl.train_agent_ppo \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    rllm.workflow.class=examples.math.workflow.MathSimpleWorkflow
```

### 1.4 观察训练

```
[q1:0] Rewards: [solver: 1.0], Term: ENV_DONE
[q2:0] Rewards: [solver: 0.0], Term: ENV_DONE
[q3:0] Rewards: [solver: 1.0], Term: ENV_DONE
...
Steps: 32/1000 | Accuracy: 0.45 | Mean Reward: 0.45
```

## 示例 2：Agent-Env 交互训练

多步工具调用——Agent 使用代码解释器解题。

### 2.1 定义 Agent

```python
from rllm.agents.agent import BaseAgent, Step, Trajectory

class ToolMathAgent(BaseAgent):
    SYSTEM_PROMPT = """你是数学助手，可以使用 Python 工具。
格式要求：
<tool_call>
{"name": "python", "arguments": {"code": "your_code_here"}}
</tool_call>
最终答案放在 \\boxed{} 中。"""

    def __init__(self):
        self._trajectory = Trajectory()
        self._messages = []

    @property
    def chat_completions(self):
        return [{"role": "system", "content": self.SYSTEM_PROMPT}] + self._messages

    @property
    def trajectory(self):
        return self._trajectory

    def reset(self):
        self._trajectory = Trajectory()
        self._messages = []

    def update_from_env(self, observation, reward, done, info, **kwargs):
        self._messages.append({"role": "user", "content": str(observation)})

    def update_from_model(self, response, **kwargs):
        self._messages.append({"role": "assistant", "content": response})
        step = Step(
            model_response=response,
            chat_completions=list(self.chat_completions),
        )
        self._trajectory.steps.append(step)
        
        # 解析 tool_call 或最终答案
        if "<tool_call>" in response:
            return self._parse_tool_call(response)
        return response
```

### 2.2 定义 Environment

```python
from rllm.environments.base import BaseEnv
from rllm.rewards.reward_fn import math_reward_fn

class ToolMathEnv(BaseEnv):
    def __init__(self, question, ground_truth, **kwargs):
        self.question = question
        self.ground_truth = ground_truth
        self.code_tool = CodeTool()

    @staticmethod
    def from_dict(info):
        return ToolMathEnv(
            question=info["question"],
            ground_truth=info["ground_truth"],
        )

    def reset(self):
        return self.question, {"ground_truth": self.ground_truth}

    def step(self, action):
        if isinstance(action, dict) and action.get("name") == "python":
            result = self.code_tool(action["arguments"]["code"])
            return f"执行结果: {result}", 0.0, False, {}
        
        reward = math_reward_fn(
            {"ground_truth": self.ground_truth}, action
        )
        return "", reward.reward, True, {"is_correct": reward.is_correct}
```

### 2.3 启动训练

```python
trainer = AgentTrainer(
    agent_class=ToolMathAgent,
    env_class=ToolMathEnv,
    config=[
        "actor_rollout_ref.model.path=Qwen/Qwen3-4B",
        "rllm.agent.max_steps=5",
        "training.train_batch_size=64",
    ],
    train_dataset=Dataset.from_parquet("data/gsm8k_train.parquet"),
    backend="verl",
)
trainer.train()
```

## 示例 3：SDK 模式训练

使用任意框架——以原生 OpenAI 客户端为例。

### 3.1 定义 Agent 函数

```python
def math_agent(metadata, question, answer, **kwargs):
    """
    SDK 模式下的 agent 函数:
    - metadata: 包含 session_name 等上下文
    - **kwargs: task dict 的所有字段
    - 返回值: float (reward) 或 list[Trajectory]
    """
    from openai import OpenAI
    
    # 指向 LiteLLM Proxy（自动路由到 vLLM）
    client = OpenAI(base_url="http://localhost:4000", api_key="dummy")
    
    response = client.chat.completions.create(
        model="my-model",
        messages=[{"role": "user", "content": f"解题: {question}"}],
        temperature=0.7,
    )
    
    predicted = response.choices[0].message.content
    
    # 直接计算奖励
    from rllm.rewards.reward_fn import math_reward_fn
    result = math_reward_fn({"ground_truth": answer}, predicted)
    return result.reward  # 返回 float → SDK 引擎自动处理
```

### 3.2 启动训练

```python
trainer = AgentTrainer(
    agent_run_func=math_agent,
    config={
        "rllm.sdk.processing.groupby_key": "session_name",
        "rllm.sdk.store.path": "/tmp/traces.db",
    },
    backend="verl",
)
trainer.train()
```

## 关键概念对应

| 概念 | 示例 1 (Simple) | 示例 2 (Agent/Env) | 示例 3 (SDK) |
|------|-----------------|-------------------|-------------|
| **训练方式** | `workflow_class=` | `agent_class= + env_class=` | `agent_run_func=` |
| **交互逻辑** | Workflow 内置 | 引擎控制循环 | 用户自由编写 |
| **LLM 调用** | `rollout_engine` | `rollout_engine` | OpenAI API (proxy) |
| **数据收集** | 自动建 Step | Agent 手动管理 | Trace 自动收集 |
| **奖励计算** | `reward_fn` 参数 | `env.step()` 返回 | `return float` |
| **适合场景** | 纯QA | 有环境交互 | 已有 Agent 框架 |

## 训练脚本示例

参见 [examples/](../examples/) 目录：

| 示例 | 脚本 | 用途 |
|------|------|------|
| FrozenLake | `examples/frozen_lake/` | 经典 RL 环境学习 |
| GSM8K | `examples/math/` | 数学问题训练 |
| SWE-bench | `examples/swe_bench/` | 软件工程任务 |
| WebArena | `examples/webarena/` | 网页导航训练 |
| Code | `examples/code/` | 编程训练 |
