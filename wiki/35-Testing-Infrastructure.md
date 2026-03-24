# 测试基础设施 (Testing Infrastructure)

详见 [34-Trace-Collection-and-Debugging.md](34-Trace-Collection-and-Debugging.md) 中的测试基础设施部分。

## 补充：域特定测试

### 数学奖励测试

```python
def test_math_reward_correct_boxed():
    """测试标准 \boxed{} 格式"""
    fn = RewardMathFn(RewardConfig())
    result = fn(
        {"ground_truth": "42", "data_source": "gsm8k"},
        "<think>Let me think...</think>\nThe answer is \\boxed{42}"
    )
    assert result.reward == 1.0
    assert result.is_correct == True

def test_math_reward_sympy_equivalence():
    """测试 SymPy 符号等价"""
    fn = RewardMathFn(RewardConfig())
    result = fn(
        {"ground_truth": "x^2 + 2*x + 1"},
        "<think>...</think>\n\\boxed{(x+1)^2}"
    )
    assert result.is_correct == True

def test_math_reward_multiple_ground_truths():
    """测试多个正确答案"""
    fn = RewardMathFn(RewardConfig())
    result = fn(
        {"ground_truth": ["42", "\\frac{84}{2}"]},
        "<think>...</think>\n\\boxed{42}"
    )
    assert result.is_correct == True

def test_math_reward_format_error():
    """测试格式检查"""
    fn = RewardMathFn(RewardConfig(apply_format_reward=True))
    result = fn(
        {"ground_truth": "42"},
        "The answer is 42"  # 缺少 <think> 和 \boxed{}
    )
    assert result.reward == 0.0

def test_math_reward_toolcall_bonus():
    """测试工具调用 bonus"""
    fn = RewardMathFn(RewardConfig(toolcall_bonus=0.5))
    result = fn(
        {"ground_truth": "42", "has_toolcall": True},
        "<think>...</think>\n\\boxed{42}"
    )
    assert result.reward == 1.5  # 1.0 + 0.5
```

### 代码奖励测试

```python
def test_code_reward_correct():
    """测试正确代码"""
    fn = RewardCodeFn(RewardConfig())
    result = fn(
        {
            "data_source": "leetcode",
            "ground_truth": {"functional": "assert add(2,3)==5\nassert add(-1,1)==0"},
        },
        "```python\ndef add(a, b): return a + b\n```"
    )
    assert result.is_correct == True

def test_code_reward_no_code_block():
    """测试无代码块"""
    fn = RewardCodeFn(RewardConfig())
    result = fn(
        {"data_source": "leetcode", "ground_truth": {"functional": "..."}},
        "Here is my solution: def add(a, b): return a + b"
    )
    assert result.reward == 0.0  # 无 ``` 代码块
```

### Trajectory 类型测试

```python
def test_step_from_model_output():
    """测试 Step 工厂方法"""
    mo = ModelOutput(
        text="answer", content="answer", reasoning="thinking",
        prompt_ids=[1,2,3], completion_ids=[4,5,6],
        logprobs=[-0.1, -0.2, -0.3],
    )
    step = Step.from_model_output(mo, messages=[{"role": "user", "content": "q"}])
    assert step.prompt_ids == [1,2,3]
    assert step.response_ids == [4,5,6]
    assert step.model_response == "answer"
    assert step.thought == "thinking"

def test_trajectory_is_cumulative_true():
    """测试累积性验证—正确情况"""
    traj = Trajectory(steps=[
        Step(chat_completions=[{"role": "user", "content": "Q1"}]),
        Step(chat_completions=[
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]),
    ])
    assert traj.is_cumulative() == True

def test_trajectory_is_cumulative_false():
    """测试累积性验证—错误情况"""
    traj = Trajectory(steps=[
        Step(chat_completions=[{"role": "user", "content": "Q1"}]),
        Step(chat_completions=[{"role": "user", "content": "Q2"}]),  # 不是前缀
    ])
    assert traj.is_cumulative() == False

def test_trajectory_group():
    """测试 TrajectoryGroup 分组"""
    group = TrajectoryGroup(
        group_id="task1:solver",
        trajectories=[
            Trajectory(name="solver", reward=0.8),
            Trajectory(name="solver", reward=0.3),
        ],
    )
    assert group.group_role == "solver"
    assert group.task_id == "task1"
```
