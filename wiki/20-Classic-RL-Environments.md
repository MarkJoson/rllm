# 经典 RL 环境 (Classic RL Environments)

## 概述

rLLM 包含经典 RL 环境（如 FrozenLake）作为教学示例和框架验证工具，展示如何将传统 RL 问题作为 LLM Agent 训练任务。

> 源码参考：[examples/frozen_lake/](../examples/frozen_lake/)

## FrozenLake 示例

### 环境设计

```
S _ F F
F H F H    S = Start, G = Goal
F F F H    F = Frozen (安全), H = Hole (掉落)
_ H F G
```

Agent 观察到网格状态，输出方向（up/down/left/right），目标是从 S 到达 G。

### Agent 实现

```python
class FrozenLakeAgent(BaseAgent):
    SYSTEM_PROMPT = """
    You are navigating a FrozenLake grid.
    S=Start, G=Goal, F=Frozen (safe), H=Hole (game over).
    Your position is marked with 'P'.
    Available actions: up, down, left, right
    """
    
    def update_from_model(self, response, **kwargs):
        direction = response.strip().lower()
        return Action(action=direction)
```

### Environment 实现

```python
class FrozenLakeEnv(BaseEnv):
    def __init__(self, map_str, **kwargs):
        self.grid = parse_map(map_str)
        self.pos = find_start(self.grid)
    
    def reset(self):
        self.pos = find_start(self.grid)
        return self._render(), {}
    
    def step(self, action):
        new_pos = move(self.pos, action)
        cell = self.grid[new_pos]
        
        if cell == "H":
            return self._render(), 0.0, True, {}  # 掉入洞
        elif cell == "G":
            return self._render(), 1.0, True, {}  # 到达终点
        else:
            self.pos = new_pos
            return self._render(), 0.0, False, {}  # 继续

    @staticmethod
    def from_dict(info):
        return FrozenLakeEnv(map_str=info["map"])
```

### 训练

```python
from rllm.trainer import AgentTrainer
from rllm.data import Dataset

trainer = AgentTrainer(
    agent_class=FrozenLakeAgent,
    env_class=FrozenLakeEnv,
    config=["rllm.agent.max_steps=20"],
    train_dataset=Dataset.from_parquet("data/frozenlake_train.parquet"),
    backend="verl",
)
trainer.train()
```

### 教学价值

FrozenLake 示例展示了 rLLM 框架的核心概念：

1. **Agent-Env 分离**：Agent 负责决策，Environment 负责状态转移
2. **多步交互**：Agent 在多步中积累 trajectory
3. **稀疏奖励**：只有到达终点才获得奖励
4. **MC Return**：通过 `gamma` 折扣因子将终点奖励回传到前序 step
5. **Token-Level Training**：每步的 action token 都参与训练

---

# 代码评估 (Code Evaluation)

> 详见 [奖励函数架构](21-Reward-Function-Architecture.md) 中的 RewardCodeFn 部分。

代码评估系统通过沙盒化执行测试用例来判断 Agent 生成代码的正确性。

## 评估流程

```
1. extract_code_from_model(response)     # 正则提取 ``` 代码块
2. clean_code_main_block(code)           # 移除 __main__ 块
3. 数据集分派 → check_correctness()       # 按数据集选择评判方法
4. multiprocessing 沙盒执行               # 独立进程运行
5. 超时保护 (timeout_per_test × num_tests) # 防止死循环
6. 结果汇总 → RewardOutput               # 是否全部通过
```

## 安全机制

| 层级 | 机制 | 保护范围 |
|------|------|---------|
| 进程隔离 | `multiprocessing.Process` | 主进程不受代码注入影响 |
| 超时保护 | `process.join(timeout)` | 防止无限循环 |
| 沙盒 | `firejail` (LeetCode) | 文件系统/网络隔离 |
| 测试限制 | `max_tests=15` | 防止测试过多导致超时 |
| 远程沙盒 | Together Code Interpreter | 完全隔离 |

---

# 数学评估 (Math Evaluation)

> 详见 [奖励函数架构](21-Reward-Function-Architecture.md) 中的 RewardMathFn 部分。

## 评判级联

```
extract_answer(model_solution)          # 1. 从 \boxed{} 提取答案
    ↓
grade_answer_mathd(pred, gold)          # 2. MATH 数据集式匹配
    ├── normalize(pred) == normalize(gold)
    └── 数值/分数/小数转换后比较
    ↓ 不匹配
grade_answer_sympy(pred, gold)          # 3. SymPy 符号计算
    ├── sympy.simplify(pred - gold) == 0
    └── 支持代数式、方程等
    ↓
结论: is_correct = True/False
```

## 答案提取

`extract_answer()` 从 `\boxed{answer}` 格式提取答案，支持嵌套大括号：

```python
# 输入: "The answer is \boxed{x^2 + 2x + 1}"
# 输出: "x^2 + 2x + 1"

# 输入: "Therefore \boxed{\frac{3}{4}}"  
# 输出: "\frac{3}{4}"
```
