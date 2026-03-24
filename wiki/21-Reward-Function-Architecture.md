# 奖励函数架构 (Reward Function Architecture)

本页详细介绍 rLLM 的奖励函数系统——包括 `RewardFunction` 协议、`RewardOutput` 数据结构、配置系统，以及奖励函数如何与训练管线集成。

> 源码参考：
> - [rllm/rewards/reward_fn.py L1-177](../rllm/rewards/reward_fn.py)
> - [rllm/rewards/reward_types.py L1-90](../rllm/rewards/reward_types.py)
> - [rllm/rewards/math_reward.py L1-142](../rllm/rewards/math_reward.py)
> - [rllm/rewards/code_reward.py L1-520](../rllm/rewards/code_reward.py)

## RewardFunction 协议

([reward_fn.py L13-28](../rllm/rewards/reward_fn.py#L13-L28))

```python
@runtime_checkable
class RewardFunction(Protocol):
    """奖励函数协议——任何满足该签名的 callable 都是合法奖励函数"""
    def __call__(self, task_info: dict, action: str) -> RewardOutput:
        """
        Args:
            task_info: 任务字典，包含 question、answer、data_source 等元数据
            action: Agent 的解答/响应字符串
        Returns:
            RewardOutput: 计算出的奖励值和元数据
        """
        ...
```

**设计理念**：
1. 使用 Python `Protocol`（而非抽象基类），任何 callable 只要签名匹配即可作为奖励函数
2. `@runtime_checkable` 允许运行时检查 `isinstance(fn, RewardFunction)`
3. 输入简洁——只需 task 字典和 action 字符串
4. 输出结构化——通过 `RewardOutput` 返回奖励值、正确性标志和元数据

## RewardOutput 结构

([reward_types.py L77-90](../rllm/rewards/reward_types.py#L77-L90))

```python
@dataclass(slots=True, kw_only=True)
class RewardOutput:
    reward: float                                  # 奖励值（通常 0.0-1.0）
    metadata: dict = field(default_factory=dict)   # 计算过程的元数据
    is_correct: bool | None = None                 # 正确性标志
```

### 字段详解

| 字段 | 类型 | 用途 |
|------|------|------|
| `reward` | `float` | 标量奖励值，直接参与 RL 优势/梯度计算 |
| `metadata` | `dict` | 日志/调试信息：如 test_details、error 信息、解析中间结果 |
| `is_correct` | `bool \| None` | 二值正确性（用于 accuracy 指标和 rejection sampling）；`None` 表示未评估 |

### metadata 内容示例

**数学奖励**：
```python
{"model_answer": "42", "ground_truth": "42", "match_method": "sympy"}
```

**代码奖励**：
```python
{
    "all_passed": True,
    "total_tests": 5,
    "passed_tests": 5,
    "test_results": [
        {"input": "3 30\n2 2 1", "expected": "5", "passed": True},
        ...
    ]
}
```

## RewardConfig 系统

([reward_types.py L11-35](../rllm/rewards/reward_types.py#L11-L35))

```python
@dataclass
class RewardConfig:
    apply_format_reward: bool = False       # 检查 <think>...</think> 格式
    math_reward_weight: float = 1.0         # 数学奖励权重
    use_math_orm: bool = False              # 使用 ORM 模型评判
    code_reward_weight: float = 1.0         # 代码奖励权重
    cot_reward_weight: float = 0.0          # CoT 奖励权重
    correct_reward: float = 1.0             # 正确时的奖励值
    incorrect_reward: float = 0.0           # 错误时的奖励值
    format_error_reward: float = 0.0        # 格式错误时的奖励值
    unk_error_reward: float = 0.0           # 无法判断时的奖励值
    toolcall_bonus: float = 0.5             # 使用工具的额外 bonus
    use_together_code_interpreter: bool = False  # Together Code Interpreter
```

### 奖励值语义

| 情况 | 默认值 | config 键 |
|------|--------|-----------|
| 回答正确 | `1.0` | `correct_reward` |
| 回答错误 | `0.0` | `incorrect_reward` |
| 格式错误（无 `\boxed{}`） | `0.0` | `format_error_reward` |
| 无法判断 | `0.0` | `unk_error_reward` |
| 使用了工具调用 | `+0.5` | `toolcall_bonus` |

### Format Reward 应用

当 `apply_format_reward=True` 时，response 必须包含 `<think>...</think>` 格式的推理过程。否则直接返回 `format_error_reward`。

## 奖励函数实现

### 函数工厂模式

([reward_fn.py L32-176](../rllm/rewards/reward_fn.py#L32-L176))

rLLM 提供五个内置奖励函数，全部遵循 `RewardFunction` 协议：

| 函数 | 行号 | 说明 | 评判方法 |
|------|------|------|---------|
| `zero_reward()` | L32-44 | 恒返回 0 的占位奖励 | 无 |
| `math_reward_fn()` | L47-62 | 数学等价评估 | `RewardMathFn` |
| `search_reward_fn()` | L65-84 | 搜索/QA 答案匹配 | `RewardSearchFn` |
| `code_reward_fn()` | L87-102 | 沙盒化代码执行 | `RewardCodeFn` |
| `f1_reward_fn()` | L105-176 | Token 级 F1 分数 | 精确率/召回率/F1 |

每个函数封装对应的奖励类实例化和调用：

```python
def math_reward_fn(task_info: dict, action: str) -> RewardOutput:
    reward_config = RewardConfig()
    reward_fn = RewardMathFn(reward_config)
    if isinstance(action, Action):
        action = action.action  # Action 对象解包
    return reward_fn(task_info, action)
```

## 奖励计算数据流

### 逐步过程

```
1. 环境 env.step(action) 或 env.compute_final_reward()
   ↓
2. 调用 reward_fn(task_info, action)
   ↓
3. 奖励函数内部:
   a. 提取 model 响应中的答案
      - 数学: extract_answer(model_solution) → 从 \boxed{} 提取
      - 代码: extract_code_from_model(response) → 从 ```code``` 提取
   b. 处理 ground_truth
      - 数学: 支持 str/float/int/list，自动 extract_answer
      - 代码: 解析为测试用例格式
   c. 评判
      - 数学: grade_answer_mathd() → grade_answer_sympy()
      - 代码: check_correctness() → multiprocessing 沙盒
   d. 构建 RewardOutput
   ↓
4. 返回 RewardOutput(reward=r, is_correct=c, metadata=m)
   ↓
5. Step.reward = output.reward
   Trajectory.reward = sum(step.reward for step in steps)  (默认)
```

## RewardMathFn 详解

([math_reward.py L18-94](../rllm/rewards/math_reward.py#L18-L94))

### 评判流程

```python
def __call__(self, task_info, action) -> RewardOutput:
    # 1. 空响应检查
    if model_response is None or model_response == "":
        return RewardOutput(reward=format_error_reward)

    # 2. 提取解答（<think>...</think> 后的内容）
    if THOUGHT_DELIMITER_END in model_response:
        model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
    else:
        if apply_format_reward:
            return RewardOutput(reward=format_error_reward)  # 格式不符
        model_solution = model_response  # 容错：直接用全文

    # 3. 从 \boxed{} 提取答案
    model_answer = extract_answer(model_solution)
    if model_answer is None:
        return RewardOutput(reward=format_error_reward)

    # 4. 处理 ground_truth（支持多个正确答案）
    ground_truths = task_info.get("ground_truth", None)
    if isinstance(ground_truths, str|float|int):
        ground_truths = [ground_truths]

    # 5. 逐一比较（任一匹配即正确）
    for ground_truth in processed_ground_truths:
        is_correct = (
            grade_answer_mathd(model_answer, ground_truth) or  # MATH 数据集方法
            grade_answer_sympy(model_answer, ground_truth)     # SymPy 符号方法
        )
        if is_correct:
            reward = correct_reward
            if task_info.get("has_toolcall", False):
                reward += toolcall_bonus  # 使用工具的额外 bonus
            return RewardOutput(reward=reward, is_correct=True)

    return RewardOutput(reward=incorrect_reward, is_correct=False)
```

### 评判级联

| 级别 | 函数 | 方法 |
|------|------|------|
| 1 | `grade_answer_mathd()` | MATH 数据集的精确匹配（normalize 后比较） |
| 2 | `grade_answer_sympy()` | SymPy 符号等价判断（`sympy.simplify(a-b) == 0`） |

## RewardCodeFn 详解

([code_reward.py L397-470](../rllm/rewards/code_reward.py#L397-L470))

### 数据集分派

```python
def __call__(self, task_info, action) -> RewardOutput:
    model_code = extract_code_from_model(model_response)  # 从 ``` 提取
    if model_code is None:
        return RewardOutput(reward=format_error_reward)

    # 按数据集分派评判方法
    if dataset_name in ["taco", "apps", "code_contests"]:
        tests = taco_to_lcb_format(tests)        # 统一为 LCB 格式
        is_correct, details = lcb_check_correctness_v2(tests, model_code)
    elif dataset_name == "leetcode":
        is_correct, details = leetcode_check_correctness(tests, model_code)
    elif dataset_name in ["livecodebench", "codeforces", "primeintellect"]:
        is_correct, details = lcb_check_correctness_v2(tests, model_code)
    elif dataset_name == "kodcode":
        is_correct, details = kodcode_check_correctness(tests, model_code)
    elif dataset_name == "humanevalplus":
        is_correct, details = humanevalplus_check_correctness(tests, model_code)
```

### 沙盒执行

`check_correctness()` ([code_reward.py L73-148](../rllm/rewards/code_reward.py#L73-L148))：

```python
def check_correctness(tests, code, test_fn, timeout_per_test=12, max_tests=15):
    # 1. 限制测试数量（选最长的 max_tests 个）
    if total_tests > max_tests:
        selected_indices = sorted(range(total_tests),
            key=lambda i: len(tests[i]["input"]), reverse=True)[:max_tests]

    # 2. 在独立进程中执行（防止代码注入影响主进程）
    process = multiprocessing.Process(target=evaluate_code, args=(...))
    process.start()
    process.join()
    if process.is_alive(): process.kill()  # 超时杀进程

    # 3. 返回详细结果
    return all(passed), {"all_passed": ..., "passed_tests": ..., "test_results": [...]}
```

> 安全机制：代码执行在独立 `multiprocessing.Process` 中，主进程不受影响。部分数据集还使用 `firejail` 沙盒。

## F1 奖励函数

([reward_fn.py L105-176](../rllm/rewards/reward_fn.py#L105-L176))

```python
def f1_reward_fn(task_info, action) -> RewardOutput:
    # 1. 文本归一化（小写、去标点、去冠词、去多余空格）
    predicted = normalize_text(str(action))
    gold = normalize_text(str(gold_text))

    # 2. Tokenize
    predicted_tokens = predicted.split()
    gold_tokens = gold.split()

    # 3. 计算 F1
    common = Counter(predicted_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    precision = num_same / len(predicted_tokens)
    recall = num_same / len(gold_tokens)
    f1_score = (2 * precision * recall) / (precision + recall)

    return RewardOutput(reward=f1_score, metadata={})
```

## RewardType 枚举

([reward_types.py L37-50](../rllm/rewards/reward_types.py#L37-L50))

```python
class RewardType(Enum):
    MATH = "MATH"    # 数学问题
    CODE = "CODE"    # 编程问题
    WEB = "WEB"      # 网页导航
    UNK = "UNK"      # 未知/未分类
```

## 与训练管线的集成

### 在 Workflow 中使用

```python
class MathWorkflow(Workflow):
    async def run(self, task, uid, **kwargs):
        # ... Agent 交互 ...
        
        # 计算奖励
        reward_output = math_reward_fn(task, agent.trajectory.steps[-1].action)
        agent.trajectory.steps[-1].reward = reward_output.reward
        
        self.commit(name="solver", agent=agent)
```

### 在 Environment 中使用

```python
class MathEnv(BaseEnv):
    def step(self, action):
        reward_output = math_reward_fn(self.task_info, action)
        return "", reward_output.reward, True, {"is_correct": reward_output.is_correct}
```

### task_info 标准字段

| 字段 | 类型 | 说明 | 数学 | 代码 |
|------|------|------|------|------|
| `data_source` | `str` | 数据集名 | ✅ | ✅ |
| `problem` | `str` | 问题文本 | ✅ | ✅ |
| `problem_type` | `RewardType` | 问题类型 | ✅ | ✅ |
| `ground_truth` | `str \| list \| dict` | 标准答案/测试用例 | `str/list` | `dict/list` |
| `has_toolcall` | `bool` | 是否使用了工具 | ✅ (optional) | ❌ |

## 错误处理

| 错误类别 | 表现 | 奖励值 |
|----------|------|--------|
| 空响应 | `action is None or ""` | `format_error_reward` (0.0) |
| 格式错误 | 无 `\boxed{}` / 无 ``` 代码块 | `format_error_reward` (0.0) |
| 无 ground_truth | `task_info["ground_truth"]` 为 None | `unk_error_reward` (0.0) |
| 代码执行超时 | `multiprocessing` 超时被 kill | `incorrect_reward` (0.0) |
| 代码执行异常 | RuntimeError / AssertionError | `incorrect_reward` (0.0) |

## 元数据用途

| 场景 | 使用的 metadata |
|------|----------------|
| 日志和调试 | `error_message`、`test_results` |
| Rejection Sampling | `is_correct` 作为过滤条件 |
| Metric 计算 | `passed_tests` / `total_tests` → pass@1 |
| 蒸馏 | `is_correct` 决定是否入选蒸馏数据集 |
