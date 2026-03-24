# 数学与代码域 (Math and Code Domains)

## 数学域

### 概述

数学域训练 LLM 通过多步推理和可选的工具调用（Python 解释器）解决数学问题。

### 工作流变体

| 变体 | 特点 | 示例 |
|------|------|------|
| SimpleWorkflow | 纯推理，无工具 | 直接输出 `\boxed{answer}` |
| SingleTurnWorkflow | 单步 + 工具 | 一次 Python 执行 + 答案 |
| MultiTurnWorkflow | 多步 + 工具 | 多次 Python 执行 |
| CumulativeWorkflow | 多步 + token 预算 | 受 response 长度限制 |

### 数学 Agent 实现

```python
class MathToolAgent(BaseAgent):
    """支持 Python 工具的数学 Agent"""
    
    SYSTEM_PROMPT = """
    Solve the math problem step by step.
    You can use Python code by wrapping it in <tool_call>...</tool_call>.
    Put your final answer in \\boxed{}.
    """
    
    def update_from_model(self, response, **kwargs):
        if "<tool_call>" in response:
            # 提取 Python 代码
            code = extract_code(response)
            return Action(action={"name": "python", "arguments": {"code": code}})
        return Action(action=response)
```

### 数学 Environment

```python
class MathToolEnv(BaseEnv):
    def __init__(self, question, ground_truth, reward_fn=math_reward_fn):
        self.question = question
        self.ground_truth = ground_truth
        self.reward_fn = reward_fn
        self.code_tool = CodeTool()  # Python 解释器

    def step(self, action):
        if isinstance(action, dict) and action.get("name") == "python":
            result = self.code_tool(action["arguments"]["code"])
            return f"Output: {result}", 0.0, False, {}
        
        # 最终答案评估
        reward = self.reward_fn({"ground_truth": self.ground_truth}, action)
        return "", reward.reward, True, reward.metadata
```

### 支持的数学数据集

| 数据集 | 题数 | 难度 |
|--------|------|------|
| GSM8K | 8.8K | 小学数学 |
| MATH | 12.5K | 竞赛数学 |
| AIME | ~50 | 高难度竞赛 |
| Gaokao | 1K+ | 高考数学 |
| MathBench | 多种 | 综合评估 |

---

## 代码域

### 概述

代码域训练 LLM 生成通过测试用例的程序代码。奖励由沙盒化代码执行决定。

### 代码评估流程

```
LLM Response
    │
    ├── extract_code_from_model(response)  # 从 ``` 提取代码
    │   └── 返回代码字符串
    │
    ├── 按数据集分派评判
    │   ├── taco/apps → lcb_check_correctness_v2()
    │   ├── leetcode → leetcode_check_correctness()
    │   ├── codeforces → lcb_check_correctness_v2()
    │   ├── kodcode → kodcode_check_correctness()
    │   └── humanevalplus → humanevalplus_check_correctness()
    │
    └── RewardOutput(reward=1.0 if all_passed else 0.0)
```

### 沙盒执行机制

```python
def check_correctness(tests, code, test_fn, timeout_per_test=12, max_tests=15):
    """
    安全执行用户代码:
    1. 限制测试数量（选最长的 max_tests 个）
    2. multiprocessing.Process 沙盒
    3. 超时自动 kill
    """
    # 代码在独立进程中运行
    process = multiprocessing.Process(target=evaluate_code, args=(...))
    process.start()
    process.join()  # 等待完成或超时
    if process.is_alive():
        process.kill()  # 超时强制终止
```

### 支持的代码数据集

| 数据集 | 格式 | 特点 |
|--------|------|------|
| TACO/APPS | `{inputs: [], outputs: []}` | stdin/stdout |
| LiveCodeBench | `[{input, output}]` | stdin + functional |
| LeetCode | `{functional: test_code}` | 函数调用 |
| KodCode | `test_file_content` | pytest 风格 |
| HumanEvalPlus | `test_file_content` | 扩展测试 |
| CodeForces | `[{input, output}]` | 竞赛题 |
| PrimeIntellect | `[{input, output, fn_name}]` | 混合格式 |

### 代码工具 (Together Code Interpreter)

```python
class TogetherCodeTool(CodeTool):
    """使用 Together AI 的远程沙盒执行代码"""
    def __call__(self, code: str, timeout: int = 30):
        response = self.client.run(code=code, timeout=timeout)
        return ToolResponse(output=response.output, error=response.error)
```

当 `RewardConfig.use_together_code_interpreter=True` 时，代码在远程沙盒而非本地进程中执行。
