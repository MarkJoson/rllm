# 安装与配置 (Installation and Setup)

## 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **Python** | 3.10+ | 3.11 (tinker 后端需 3.11+) |
| **操作系统** | Linux (Ubuntu 20.04+) | Ubuntu 22.04+ |
| **CUDA** | 12.1+ | 12.8 |
| **GPU** | NVIDIA A100 40GB (verl) | 4×A100/H100 80GB |
| **RAM** | 32GB | 64GB+ |
| **磁盘** | 50GB 可用空间 | 100GB+ (模型权重) |

> 源码参考：[pyproject.toml L1-192](../pyproject.toml)、[README.md](../README.md)

## 安装方法

### 方法一：uv（推荐）

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆代码
git clone https://github.com/rllm-org/rllm.git
cd rllm

# 创建虚拟环境（Python >= 3.10）
uv venv --python 3.10
source .venv/bin/activate

# 安装基础版（无训练后端）
uv pip install -e .

# 安装 VERL 后端（推荐，完整功能）
uv pip install -e ".[verl]"

# 安装 Tinker 后端（开发/调试）
uv pip install -e ".[tinker]"

# 安装所有后端
uv pip install -e ".[all]"
```

### 方法二：pip

```bash
pip install -e .
pip install -e ".[verl]"
```

### 方法三：从 PyPI

```bash
pip install rllm           # 基础
pip install rllm[verl]     # 带 VERL
pip install rllm[tinker]   # 带 Tinker
```

## 依赖组说明

从 [pyproject.toml](../pyproject.toml) 提取的依赖组：

| 安装组 | 核心依赖 | 适用场景 |
|--------|---------|---------|
| `base` | `pydantic`, `jinja2`, `httpx`, `tqdm` | 数据处理、Agent 开发 |
| `verl` | `verl`, `vllm`, `ray`, `torch`, `fsdp` | 分布式训练 |
| `tinker` | `tinker` | 本地单机训练/调试 |
| `eval` | `evalplus`, `datasets` | 代码/数学评估 |
| `swe` | `docker`, `swebench` | Software Engineering 环境 |
| `web` | `playwright`, `selenium` | Web Navigation 环境 |
| `sdk` | `litellm`, `opentelemetry` | SDK 模式（框架无关） |

## VERL 后端环境配置

### CUDA 环境

```bash
# 检查 CUDA 版本
nvcc --version

# 安装 PyTorch（CUDA 12.x）
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 安装 vLLM
pip install vllm
```

### Ray 集群配置

```bash
# 单机模式（开发）
ray start --head --port=6379

# 多机模式（生产）
# 主节点
ray start --head --port=6379 --num-gpus=4
# 工作节点
ray start --address='<head-ip>:6379' --num-gpus=4
```

### Ray Runtime 环境

`get_ppo_ray_runtime_env()` ([ray_runtime_env.py](../rllm/trainer/verl/ray_runtime_env.py)) 自动配置：

```python
def get_ppo_ray_runtime_env():
    """确保 Ray 远程工作节点能导入 rllm 包"""
    import rllm
    rllm_path = Path(rllm.__file__).parent.parent
    return {
        "working_dir": str(rllm_path),
        "py_modules": [str(rllm_path / "rllm")],
    }
```

### GPU 资源分配

```yaml
# 典型的 4×A100 配置
trainer:
  n_gpus_per_node: 4
actor_rollout_ref:
  model:
    path: "Qwen/Qwen3-4B"
  actor:
    strategy: fsdp                    # 全参数分布式
    fsdp_config:
      param_offload: false
  rollout:
    name: vllm                        # 推理引擎
    tensor_model_parallel_size: 1     # TP 并行度
    gpu_memory_utilization: 0.4       # KV Cache 占 GPU 比例
```

## Tinker 后端配置

```bash
# Python >= 3.11 必需
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[tinker]"

# 启动 Tinker 服务
tinker serve --model Qwen/Qwen3-4B --port 8000
```

## 验证安装

```bash
# 验证基础安装
python -c "import rllm; print(rllm.__version__)"

# 验证 VERL 安装
python -c "from rllm.trainer import AgentTrainer; print('VERL OK')"

# 验证 Ray
python -c "import ray; ray.init(); print(ray.cluster_resources()); ray.shutdown()"

# 验证 vLLM
python -c "from vllm import LLM; print('vLLM OK')"
```

## 常见问题

| 问题 | 解决方案 |
|------|---------|
| `CUDA out of memory` | 减小 `gpu_memory_utilization`，使用更小模型 |
| `Ray init timeout` | 检查防火墙，确保 6379 端口可达 |
| `ModuleNotFoundError: verl` | `pip install -e ".[verl]"` |
| `Python version < 3.11 for tinker` | 使用 `uv venv --python 3.11` |
| `tokenizer not found` | 设置 `HF_TOKEN` 或下载模型到本地 |
