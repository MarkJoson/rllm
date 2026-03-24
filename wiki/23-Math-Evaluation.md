# 数学评估 (Math Evaluation)

数学评估系统从 LLM 响应中提取答案，并通过多级评判方法与标准答案比较。

> 源码参考：
> - [rllm/rewards/math_reward.py L1-142](../rllm/rewards/math_reward.py)
> - [rllm/rewards/math_utils/](../rllm/rewards/math_utils/)

## 评判级联

```
extract_answer(model_solution)
    │
    │ 从 \boxed{answer} 提取答案
    │ 支持嵌套大括号: \boxed{\frac{3}{4}}
    ▼
grade_answer_mathd(prediction, reference)
    │ MATH 数据集式评判:
    │ ├── 文本归一化 (去空格/大小写)
    │ ├── 数值转换 (3/4 → 0.75)
    │ ├── LaTeX 简化 (\frac{3}{4} → 0.75)
    │ └── 精确匹配
    ▼ 不匹配？
grade_answer_sympy(prediction, reference)
    │ SymPy 符号计算:
    │ ├── 解析为 sympy 表达式
    │ ├── simplify(pred - ref) == 0
    │ └── 支持代数式、方程、集合等
    ▼
is_correct: bool
```

## RewardMathFn 完整流程

([math_reward.py L29-94](../rllm/rewards/math_reward.py#L29-L94))

```python
def __call__(self, task_info, action):
    # ① 空响应 → format_error_reward (0.0)
    if model_response is None or model_response == "":
        return RewardOutput(reward=self.config.format_error_reward)

    # ② 提取 <think>...</think> 后的解答部分
    if THOUGHT_DELIMITER_END in model_response:
        model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
    else:
        if self.config.apply_format_reward:
            return RewardOutput(reward=self.config.format_error_reward)
        model_solution = model_response

    # ③ 从 \boxed{} 提取答案
    model_answer = extract_answer(model_solution)
    if model_answer is None:
        return RewardOutput(reward=self.config.format_error_reward)

    # ④ 支持多个正确答案
    ground_truths = task_info.get("ground_truth")
    if isinstance(ground_truths, str|float|int):
        ground_truths = [ground_truths]

    # ⑤ 逐一评判（精确 → 符号）
    for ground_truth in processed_ground_truths:
        is_correct = (
            grade_answer_mathd(model_answer, ground_truth) or
            grade_answer_sympy(model_answer, ground_truth)
        )
        if is_correct:
            reward = self.config.correct_reward
            if task_info.get("has_toolcall", False):
                reward += self.config.toolcall_bonus
            return RewardOutput(reward=reward, is_correct=True)

    return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)
```

## 工具调用 Bonus

当 `has_toolcall=True`（Agent 使用了 Python 解释器等工具）且答案正确时，额外奖励 `toolcall_bonus`（默认 0.5）：

```
总奖励 = correct_reward (1.0) + toolcall_bonus (0.5) = 1.5
```

> 这鼓励 Agent 学习使用工具而非纯推理。

## 便捷接口

```python
from rllm.rewards.math_reward import rllm_reward_fn_math

result = rllm_reward_fn_math(
    data_source="gsm8k",
    llm_solution="<think>Step by step...</think>\nThe answer is \\boxed{42}",
    ground_truth="42",
)
# result.reward = 1.0, result.is_correct = True
```
