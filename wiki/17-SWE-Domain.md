# SWE 域 (SWE Domain)

软件工程（SWE）域实现了在 Docker 容器化环境中训练 Agent 解决真实 GitHub issue 的能力。

> 源码参考：[rllm/environments/swe/](../rllm/environments/swe/)、[examples/swe_bench/](../examples/swe_bench/)

## SWE Agent-Environment 交互

```
┌─────────────┐     action (bash cmd)     ┌──────────────────┐
│  SWE Agent  │────────────────────────→   │  SWE Environment │
│  (解析LLM   │                            │  (Docker 容器)    │
│   输出为     │   ←────────────────────    │  ├── git clone    │
│   bash 命令) │     obs (命令输出)         │  ├── bash exec    │
└─────────────┘                            │  └── test runner  │
                                           └──────────────────┘
```

## SWE Environment

```python
class SWEEnv(BaseEnv):
    def __init__(self, instance_id, repo, base_commit, test_patch, ...):
        self.container = docker.create_container(...)
        
    def reset(self):
        """克隆仓库到容器内, checkout base_commit"""
        self.container.exec("git clone ... && git checkout ...")
        return initial_observation, info
    
    def step(self, action: str):
        """在容器内执行 bash 命令"""
        output = self.container.exec(f"bash -c '{action}'")
        return output, 0.0, False, {}  # 中间步无奖励
    
    def compute_final_reward(self):
        """执行测试套件判断补丁是否正确"""
        self.container.exec("python -m pytest ...")
        return 1.0 if tests_pass else 0.0
    
    @staticmethod
    def is_multithread_safe() -> bool:
        return True  # 每个实例独立容器
```

## 工作流

SWE 任务通常使用 `MultiTurnWorkflow`（最多 10-20 步交互），支持：
- 文件浏览和编辑
- bash 命令执行
- 测试运行和结果分析
- 补丁生成和提交

## 数据集

| 数据集 | 说明 |
|--------|------|
| SWE-bench Lite | 300 个精选 GitHub issue |
| SWE-bench Full | 2000+ GitHub issue |
| SWE-bench Verified | 人工验证子集 |

## 安装

```bash
uv pip install -e ".[swe]"
# 需要 Docker 守护进程运行
docker pull python:3.11-slim
```

---

# Web 导航域 (Web Navigation Domain)

Web 导航域训练 Agent 在浏览器环境中完成网页交互任务。

> 源码参考：[rllm/environments/web/](../rllm/environments/web/)、[examples/webarena/](../examples/webarena/)

## Web Agent-Environment 交互

```
┌──────────────┐   action (click/type/nav)  ┌───────────────────┐
│  Web Agent   │─────────────────────────→   │  Browser Env      │
│  (解析LLM    │                             │  (Playwright/     │
│   输出为      │   ←─────────────────────   │   Selenium)       │
│   浏览器动作) │    obs (DOM/screenshot)     │  ├── 页面渲染      │
└──────────────┘                             │  ├── 元素交互      │
                                             │  └── 状态提取      │
                                             └───────────────────┘
```

## 支持的动作

| 动作类型 | 示例 |
|----------|------|
| 点击 | `click(element_id=123)` |
| 输入 | `type(element_id=456, text="hello")` |
| 导航 | `goto(url="https://...")` |
| 滚动 | `scroll(direction="down")` |
| 等待 | `wait(seconds=2)` |

## 数据集

| 数据集 | 说明 |
|--------|------|
| WebArena | 真实网站交互（GitLab、Reddit 等） |
| MiniWoB++ | 简化的 Web 任务 |

## 安装

```bash
uv pip install -e ".[web]"
playwright install chromium
```
