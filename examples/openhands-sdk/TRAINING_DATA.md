# 算子训练数据格式说明

本文描述使用 `train_openhands.py` + `OPENHANDS_DATASET=mock_npu`（或等价模式）时，**Parquet 训练数据**应满足的字段与 `extra_info` 约定。`rollout()` 会把 `extra_info` 展成关键字参数，用于布置 workspace、写 `INSTRUCTIONS.md`、向 OpenHands 容器注入 `OPERATOR_*` 环境变量，并在结束后根据 `metrics.json` 计算 reward。

---

## 1. 文件形态

- **格式**：Parquet。
- **默认路径**（当前代码）：与 `train_openhands.py` 同目录下的 `mock_npu_operator.parquet`。首次运行且文件不存在时，会由 `create_mock_npu_operator_data.py` 自动生成一份 mock。
- **接入自有数据**：将符合下文的 Parquet 放到上述路径，或修改 `train_openhands.py` 中 `_MockNPUOperatorParquetDataset.get_verl_data_path()` 的返回值指向你的文件。

---

## 2. 列要求

| 列名 | 类型 | 说明 |
|------|------|------|
| `prompt` | string（建议为 JSON 字符串） | 与 verl/rLLM 数据管线兼容；内容可与 `extra_info.instruction` 对齐，便于日志与调试。 |
| `extra_info` | **dict**（Parquet 中常存为 struct/map，读入后为字典） | **核心**：必须包含下列算子任务字段。 |

> 若你方数据管线还有其它列（如 `data_source`），只要不影响读取 `prompt` / `extra_info` 即可；以 `AgentTrainer` / verl 实际 schema 为准。

---

## 3. `extra_info` 字段（与 `openhands_agent.rollout` 对齐）

### 3.1 必填 / 强烈建议

| 键 | 类型 | 说明 |
|----|------|------|
| `scenario` | string | 建议固定为 `npu_operator`，与 `OPENHANDS_DATASET=mock_npu` 语义一致，便于区分非算子任务（若将来扩展）。 |
| `op_name` | string | 算子短名，**仅字母数字下划线**。决定 `src/{op_name}.py`、`src/{op_name}_triton_ascend_impl.py` 及 pipeline 的 `--op_name`。 |
| `instruction` | string | 自然语言任务说明，写入 `INSTRUCTIONS.md` 并作为容器内用户消息摘要来源之一。 |
| `task_code` | string | **参考实现**源码（KernelBench 风格）：需包含 `Model`、`get_inputs()`、`get_init_inputs()`，将写入 `src/{op_name}.py`。 |

### 3.2 可选

| 键 | 类型 | 默认值 | 说明 |
|----|------|--------|------|
| `arch` | string | `ascend910b1` | 写入 `INSTRUCTIONS.md`，并作为容器环境变量 `OPERATOR_ARCH`。 |
| `operator_backend` | string | `triton` | 容器内 `OPERATOR_BACKEND`。 |

---

## 4. `task_code` 约定（参考算子）

须为合法 Python 源码，且至少包含：

- `class Model`：PyTorch 参考实现（通常在 NPU 上与 `ModelNew` 对比精度）。
- `def get_inputs()`：返回 forward 所需输入列表（tensor 等）。
- `def get_init_inputs()`：返回 `Model(...)` / `ModelNew(...)` 构造函数参数（无则 `return []`）。

与 `verify.py` / `benchmark.py` 的加载方式一致；dtype、shape 需与真实验证环境匹配。

---

## 5. 示例：逻辑一行（JSON 视图）

```json
{
  "prompt": "[{\"role\": \"user\", \"content\": \"在 Ascend 910B 上用 Triton 实现向量加……\"}]",
  "extra_info": {
    "scenario": "npu_operator",
    "op_name": "vector_add",
    "arch": "ascend910b1",
    "instruction": "实现 FP16 向量逐元素相加的 Triton kernel，与参考 Model 数值一致。",
    "operator_backend": "triton",
    "task_code": "import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n    def forward(self, a, b):\n        return a + b\n\ndef get_inputs():\n    return [torch.randn(4096, dtype=torch.float16), torch.randn(4096, dtype=torch.float16)]\n\ndef get_init_inputs():\n    return []\n"
  }
}
```

`prompt` 在 mock 脚本里通常用 `json.dumps([{"role":"user","content": instruction}])` 生成，与多轮对话数据格式一致。

---

## 6. 用 pandas 生成一条并写入 Parquet（示例）

```python
import json
import pandas as pd

instruction = "实现 vector_add（FP16，4096 维）。"
row = {
    "prompt": json.dumps([{"role": "user", "content": instruction}]),
    "extra_info": {
        "scenario": "npu_operator",
        "op_name": "vector_add",
        "arch": "ascend910b1",
        "instruction": instruction,
        "task_code": '''import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a, b):
        return a + b

def get_inputs():
    return [torch.randn(4096, dtype=torch.float16), torch.randn(4096, dtype=torch.float16)]

def get_init_inputs():
    return []
''',
    },
}
pd.DataFrame([row]).to_parquet("my_ops.parquet", index=False)
```

将 `my_ops.parquet` 换名为工程约定的路径，或改 `train_openhands.py` 中的数据路径。

---

## 7. 启动训练前检查清单

1. `export OPENHANDS_DATASET=mock_npu`（或代码中识别的等价取值）。
2. Parquet 存在且 `extra_info` 含 `op_name`、`instruction`、`task_code`。
3. `op_name` 与 `task_code` 内类名无冲突；实现文件由 agent 生成：`src/{op_name}_triton_ascend_impl.py`。
4. OpenHands 镜像、`OPENHANDS_IMAGE`、`LLM` 代理已按 `train_openhands.sh` 配置。

---

## 8. 参考实现

同目录下 `create_mock_npu_operator_data.py` 中 `_MOCK_TASKS` 为多任务完整示例，可直接运行生成 `mock_npu_operator.parquet`：

```bash
python3 create_mock_npu_operator_data.py
```

---

## 9. 若重新设计数据格式：适配时应关注的脚本

数据从 **Parquet → rLLM/verl 读盘 → `rollout` 收到与 `extra_info` 对齐的 kwargs → workspace / Docker / reward** 一条链；改列名、嵌套结构或语义时，按下面表格在 **本仓库（openhands-sdk）** 内排查即可。

**与 rLLM 的关系**：在常见设计下 **不需要** 对 rLLM 源码做侵入式修改。只要数据管线仍把样本里的 `extra_info` **展平成** 传给 `rollout` 的关键字参数（与当前行为一致），适配工作集中在 `openhands_agent.py`、`train_openhands.py` 与数据生成脚本即可。此前文中提到 `AgentSdkEngine` 仅用于说明「框架里存在从样本到 `rollout` 的衔接」这一**概念位置**，不是要求你去改 Engine；若将来 `extra_info` 改成深层嵌套对象，优先在 **本目录** 用 `rollout` 包装函数或自定义 Dataset 做 **展平/映射**，仍应避免 fork rLLM。

| 层级 | 路径 | 关注点 |
|------|------|--------|
| 数据生成 / 样例 | `create_mock_npu_operator_data.py` | `extra_info` 键、`prompt` 构造、与 Parquet schema 一致。 |
| 训练入口与数据集绑定 | `train_openhands.py` | `OPENHANDS_DATASET` 分支、`get_verl_data_path()` 返回的 parquet 路径；若列名不再是 `prompt`/`extra_info`，需与 verl **配置**或本仓库自定义 `Dataset` 对齐（仍不必改 rLLM 内部实现，除非你们选择上游贡献）。 |
| Rollout 与任务归一化 | `openhands_agent.py` | `_rollout_task_and_config()`：哪些键从 kwargs 进入 `task`、哪些进 `_ROLLOUT_CONFIG_KEYS`；`_setup_npu_operator_workspace()`：`task.get("op_name")` 等所有键；`_NPU_INSTRUCTION_TEMPLATE` 占位符；`_run_openhands_container()` 传入容器的 `OPERATOR_*`；`_npu_operator_reward()` / `_archive_npu_artifacts()` 是否依赖 `op_name`、`arch` 等路径与 manifest。 |
| 指令与规范文案 | `workspace/INSTRUCTIONS.md`（模板）、`workspace/AGENTS.md`、`.agents/skills/*/SKILL.md` | 若产物路径、Phase、文件名规则随数据字段而变，需同步改文案中的占位符与示例命令。 |
| 流水线与脚本约定 | `workspace/tools/operator_pipeline.sh` | `--op_name`、`IMPL_NAME`、期望的 `src/{op_name}.py` / `src/{op_name}_*_impl.py` 命名；改命名规则必改此处。 |
| 数值与性能验证 | `workspace/tools/scripts/verify.py`、`benchmark.py` | `--verify_dir` 下模块命名 `{op_name}_torch`、 `{op_name}_{triton_impl_name}`；若 `task_code` 产出结构变化，需与这两支脚本的 import 约定一致。 |
| Mock 管线（无 NPU） | `openhands_agent.py` 内 `_MOCK_PIPELINE_SCRIPT` | 若真实 pipeline 的 `metrics.json` schema 或 impl 文件名规则变更，mock 脚本需同步，避免 reward 与训练逻辑漂移。 |

**建议**：先在 **`openhands_agent.py` 中集中定义「数据行 → `task: dict`」的单一映射函数**（或 dataclass），再让 `_setup_npu_operator_workspace` 等只依赖该结构；这样日后改 Parquet 只需改映射层与 `train_openhands`/mock 生成脚本，避免散落多处 `task.get(...)`。（当前实现为分散 `task.get`，重设计时优先收敛。）
