# 系统提示 (System Prompts)

系统提示是 Agent 行为的蓝图，决定了 LLM 如何理解任务、使用工具和格式化输出。

> 源码参考：各域的 Agent 实现

## 设计原则

1. **角色定义**：明确 Agent 的身份和能力
2. **格式约束**：指定输出格式（如 `\boxed{}`、`<tool_call>`）
3. **工具描述**：列出可用工具及其调用方式
4. **示例引导**：通过 few-shot 示例引导目标行为

## 域特定系统提示

### 数学域

```
你是一个数学问题求解专家。
请按以下步骤解题：
1. 仔细阅读题目
2. 列出已知条件
3. 逐步推理
4. 将最终答案放在 \boxed{} 中

如果需要计算，可以使用 Python 工具：
<tool_call>
{"name": "python", "arguments": {"code": "your_code"}}
</tool_call>
```

### SWE 域

```
You are a software engineer working on fixing a GitHub issue.
You have access to a terminal. Available commands:
- find_file <filename>: Search for files
- open_file <path> [line_number]: Open a file
- edit_file <path> <start_line> <end_line>: Edit a file
- bash <command>: Run a bash command
- submit: Submit your changes

Always start by understanding the issue, then locate the relevant code.
```

### Web 导航域

```
You are a web navigation agent. Your goal is to complete tasks on websites.
You can see the current page as an accessibility tree.
Available actions:
- click(element_id): Click an element
- type(element_id, text): Type text into an element
- scroll(direction): Scroll the page
- goto(url): Navigate to a URL
```

## 系统提示与训练的关系

系统提示在训练中被包含在 **prompt** 部分（不计算 loss），但它影响 Agent 生成的质量，因此间接影响训练效果：

- 好的系统提示 → Agent 更容易生成正确格式的输出 → 更多正奖励 → 更有效的训练
- 差的系统提示 → Agent 输出格式混乱 → format_error_reward → 训练信号稀疏

## 动态系统提示

某些 Agent 根据任务动态构建系统提示：

```python
class DynamicAgent(BaseAgent):
    def reset(self):
        self._system_prompt = self.base_prompt
    
    def update_from_env(self, observation, reward, done, info):
        # 根据环境信息动态调整
        if "available_tools" in info:
            self._system_prompt += f"\nTools: {info['available_tools']}"
```

---

# 数据集准备 (Dataset Preparation)

## 数据格式

rLLM 使用 **Parquet** 格式存储训练和验证数据集。每条记录至少包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `extra_info` | `dict` | 任务信息（传给 Agent/Env/Workflow） |

`extra_info` 的典型内容：

| 域 | 关键字段 |
|----|---------|
| 数学 | `question`, `ground_truth`, `data_source` |
| 代码 | `problem`, `tests`, `data_source` |
| SWE | `instance_id`, `repo`, `base_commit`, `test_patch` |
| Web | `task_description`, `expected_url`, `start_url` |

## Dataset 类

```python
from rllm.data import Dataset

# 从 Parquet 加载
dataset = Dataset.from_parquet("data/train.parquet")

# 获取 verl 兼容路径
path = dataset.get_verl_data_path()  # → "data/train.parquet"

# 从 HuggingFace 加载
dataset = Dataset.from_huggingface("gsm8k", split="train")
```

## 数据准备脚本

各示例提供数据准备脚本：

```bash
python examples/math/prepare_gsm8k_data.py
python examples/code/prepare_leetcode_data.py
python examples/frozen_lake/prepare_frozenlake_data.py
```

## 自定义数据集

```python
import pandas as pd

data = [
    {"extra_info": {"question": "2+3=?", "ground_truth": "5", "data_source": "custom"}},
    {"extra_info": {"question": "7*8=?", "ground_truth": "56", "data_source": "custom"}},
]
df = pd.DataFrame(data)
df.to_parquet("data/custom_train.parquet")
```

---

# 数据集类型与格式 (Dataset Types and Formats)

## 支持的数据集

| 域 | 数据集 | 格式 | 大小 |
|----|--------|------|------|
| 数学 | GSM8K | question + answer | 8.8K |
| 数学 | MATH | question + answer + type | 12.5K |
| 数学 | AIME | question + answer | ~50 |
| 代码 | LeetCode | problem + functional_tests | 2K+ |
| 代码 | TACO/APPS | problem + stdin/stdout tests | 10K+ |
| 代码 | LiveCodeBench | problem + tests + metadata | 400+ |
| 代码 | HumanEvalPlus | problem + test_file | 164 |
| 代码 | KodCode | problem + test_file | 6K+ |
| SWE | SWE-bench | instance_id + repo + tests | 2K+ |
| Web | WebArena | task + website config | 812 |
| RL | FrozenLake | map configuration | 自定义 |

## Parquet Schema

所有数据集统一为：

```
 ├── extra_info (dict)
 │   ├── question / problem (str)
 │   ├── ground_truth (str | list | dict)
 │   ├── data_source (str)
 │   └── [域特定字段...]
 └── [可选的其他列]
```

---

# 轨迹处理 (Trajectory Processing)

## 从 Episode 到训练数据

```
Episode(trajectories=[Trajectory(steps=[Step(...), Step(...)]), ...])
    │
    ├── Compact Filtering
    │   ├── ERROR → drop
    │   ├── TIMEOUT → drop
    │   └── ENV_DONE → keep
    │
    ├── Tokenization
    │   ├── 累积模式: tokenize_and_mask_cumulative()
    │   ├── 直接模式: 使用 ModelOutput 的 prompt_ids/completion_ids
    │   └── 逐步模式: 每步独立 tokenize
    │
    ├── Padding & Truncation
    │   ├── Prompt: 左填充到 max_prompt_length
    │   ├── Response: 右填充到 max_response_length
    │   └── 超长截断
    │
    ├── Masking
    │   ├── Response Mask: assistant=1, env/user=0
    │   ├── Attention Mask: valid=1, padding=0
    │   └── Overlong Filter: 超长轨迹全零 mask
    │
    └── DataProto 打包
        ├── Tensors: input_ids, masks, rewards, logprobs
        └── Non-Tensors: ids, correctness, metrics
```

## 奖励放置

在 `transform_results_for_verl()` 中，奖励被放在每条轨迹 response 的最后一个有效 token 位置：

```python
for i, (traj_reward, step_reward) in enumerate(zip(traj_rewards, step_rewards)):
    resp_len = response_lengths[i]
    if resp_len > 0:
        traj_rewards_batch[i, resp_len - 1] = traj_reward
        step_rewards_batch[i, resp_len - 1] = step_reward
```

> 这种放置确保了 GRPO/REINFORCE 等算法能正确在最后 token 处进行信用分配。
