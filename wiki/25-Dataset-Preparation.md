# 数据集准备 (Dataset Preparation)

## 数据格式要求

rLLM 使用 **Parquet** 格式存储训练数据。每条记录的核心字段：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `extra_info` | `dict` | ✅ | 任务信息字典，传给 Agent/Env/Workflow |

### extra_info 字段约定

| 域 | 必需字段 | 可选字段 |
|----|---------|---------|
| 数学 | `question`, `ground_truth` | `data_source`, `problem_type`, `has_toolcall` |
| 代码 | `problem`, `ground_truth` (tests) | `data_source`, `fn_name` |
| SWE | `instance_id`, `repo`, `base_commit` | `test_patch`, `hints` |
| Web | `task_description`, `start_url` | `expected_url`, `expected_content` |
| 自定义 | 任意 | 任意 |

## Dataset 类

```python
from rllm.data import Dataset

# 从 Parquet 文件加载
train_dataset = Dataset.from_parquet("data/train.parquet")
val_dataset = Dataset.from_parquet("data/val.parquet")

# 获取 verl 兼容路径
path = train_dataset.get_verl_data_path()
```

## 数据准备示例

### GSM8K 数学

```python
from datasets import load_dataset
import pandas as pd

ds = load_dataset("openai/gsm8k", "main", split="train")
records = []
for item in ds:
    records.append({
        "extra_info": {
            "question": item["question"],
            "ground_truth": item["answer"].split("####")[-1].strip(),
            "data_source": "gsm8k",
        }
    })
pd.DataFrame(records).to_parquet("data/gsm8k_train.parquet")
```

### FrozenLake

```python
import random
import pandas as pd

def generate_map(size=4):
    grid = [["F"] * size for _ in range(size)]
    grid[0][0] = "S"
    grid[size-1][size-1] = "G"
    # 随机放置 holes
    for _ in range(size):
        r, c = random.randint(0, size-1), random.randint(0, size-1)
        if grid[r][c] == "F":
            grid[r][c] = "H"
    return "\n".join("".join(row) for row in grid)

records = [{"extra_info": {"map": generate_map()}} for _ in range(10000)]
pd.DataFrame(records).to_parquet("data/frozenlake_train.parquet")
```

### 自定义数据集

```python
records = [
    {"extra_info": {"question": q, "ground_truth": a, "data_source": "custom"}}
    for q, a in your_data
]
pd.DataFrame(records).to_parquet("data/custom_train.parquet")
```

## 数据集与 AgentTrainer 集成

```python
trainer = AgentTrainer(
    workflow_class=MyWorkflow,
    train_dataset=Dataset.from_parquet("data/train.parquet"),
    val_dataset=Dataset.from_parquet("data/val.parquet"),
    config=config,
    backend="verl",
)
```

> `AgentTrainer.__init__` 自动将数据集路径注入到 config：
> `config.data.train_files = dataset.get_verl_data_path()`

---

# 数据集类型与格式 (Dataset Types and Formats)

## 概览

| 类别 | 数据集 | 任务数 | 格式特点 |
|------|--------|--------|---------|
| **数学** | GSM8K | 8.8K | 小学数学，链式推理 |
| | MATH | 12.5K | 竞赛数学，分类型 |
| | AIME | ~50 | 高难度竞赛 |
| | Gaokao | 1K+ | 中文高考数学 |
| **代码** | LeetCode | 2K+ | 函数式测试 |
| | TACO/APPS | 10K+ | stdin/stdout 测试 |
| | LiveCodeBench | 400+ | 混合测试 |
| | HumanEvalPlus | 164 | pytest式测试 |
| | KodCode | 6K+ | pytest式测试 |
| | CodeForces | 12K+ | stdin/stdout |
| | PrimeIntellect | 多种 | 可验证编程 |
| **SWE** | SWE-bench | 2K+ | GitHub issue + 测试补丁 |
| **Web** | WebArena | 812 | 网站交互任务 |
| **对话** | Search/QA | 多种 | 问答对 |

## ground_truth 格式

| 域 | 类型 | 示例 |
|----|------|------|
| 数学 | `str` | `"42"` |
| 数学 | `list[str]` | `["42", "\\frac{84}{2}"]` |
| 代码 (stdin) | `dict` | `{"inputs": ["3\n1 2 3"], "outputs": ["6"]}` |
| 代码 (函数) | `dict` | `{"functional": "assert f(2,3)==5"}` |
| 代码 (LCB) | `list[dict]` | `[{"input": "3", "output": "6"}]` |
| SWE | `str` | test_patch 内容 |

---

# 轨迹处理 (Trajectory Processing)

## 概述

轨迹处理是将 Agent-Environment 交互产生的 Episode/Trajectory 数据转换为可用于 RL 训练的张量格式的过程。

## 处理管线

```
Episode
├── Trajectory[0] (solver, 3 steps)
│   ├── Step 0: prompt_ids=[...], response_ids=[...], reward=0.0
│   ├── Step 1: prompt_ids=[...], response_ids=[...], reward=0.0
│   └── Step 2: prompt_ids=[...], response_ids=[...], reward=1.0
└── Trajectory[1] (judge, 1 step)
    └── Step 0: prompt_ids=[...], response_ids=[...], reward=1.0

    ↓ transform_results_for_verl()

DataProto (4 行):
Row 0: solver_step0  prompt=[P0], response=[R0], mask=[1,1,...], reward=0
Row 1: solver_step1  prompt=[P1], response=[R1], mask=[1,1,...], reward=0
Row 2: solver_step2  prompt=[P2], response=[R2], mask=[1,1,...], reward=1.0
Row 3: judge_step0   prompt=[P3], response=[R3], mask=[1,1,...], reward=1.0

meta_info["repeat_counts"] = [4]  # episode 贡献了 4 行
```

## Tokenization 模式

### 模式 1：累积 Tokenization

当 `stepwise_advantage.enable=False` 且轨迹有多步时：

```python
# 使用最后一步的完整对话历史
chat_completions = trajectory.steps[-1].chat_completions
prompt, response, mask = chat_parser.tokenize_and_mask_cumulative(chat_completions)
```

### 模式 2：直接模式

当单步轨迹且 Step 有 `model_output` 时：

```python
prompt_ids = step.model_output.prompt_ids
completion_ids = step.model_output.completion_ids
mask = ones_like(completion_ids)  # 全为 1（单步无 env token）
```

### 模式 3：逐步模式

当 `stepwise_advantage.enable=True` 时：

```python
for step in trajectory.steps:
    prompt, response, mask = tokenize_step(step)
    # 每步独立一行 DataProto
```

## Padding 策略

| 目标 | 方向 | 公式 | 原因 |
|------|------|------|------|
| Prompt | 左填充 | `pad(flip(ids)) → flip()` | 右对齐，使有效 token 靠近 response |
| Response | 右填充 | `pad(ids)` | 左对齐，奖励在最后有效 token |
| 截断 | 两侧 | `prompt[:max_P]`, `response[:max_R]` | 防止 OOM |

## Compact Filtering

根据终止原因过滤不可靠样本：

```yaml
compact_filtering:
  enable: true
  mask_error: true                    # RuntimeError 等
  mask_timeout: true                  # 超时
  mask_max_prompt_length_exceeded: true
  mask_unknown: true                  # 未知终止
  mask_env_done: false                # 正常结束不过滤
```

过滤后的样本 `is_valid=False`，在优势计算中权重为 0。
