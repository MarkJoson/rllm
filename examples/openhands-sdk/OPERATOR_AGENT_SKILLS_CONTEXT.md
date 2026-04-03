# 算子 Agent · Skills / 上下文注入 — 讨论历史总结

> 供新对话窗口快速对齐背景；路径相对于本仓库 `rllm-private-repo/rllm-szh/examples/openhands-sdk/`。

## OpenHands 官方文档（Skills 总览 · SDK 详解）

OpenHands 当前文档站点为 **[docs.openhands.dev](https://docs.openhands.dev/)**（Mintlify）；口语里说的「wiki」一般即指该站，而非本仓库内的 Markdown。

| 主题 | 链接 |
|------|------|
| **Skills 总览**（加载模型、`AGENTS.md`、`.agents/skills/`、触发式 skill、与 AgentSkills 标准的关系） | [https://docs.openhands.dev/overview/skills](https://docs.openhands.dev/overview/skills) |
| **SDK Skills 指南**（`Skill`、`AgentContext`、`load_project_skills`、`load_skills_from_dir`、`SKILL.md` 格式等） | [https://docs.openhands.dev/sdk/guides/skill](https://docs.openhands.dev/sdk/guides/skill) |
| **SDK 中 Skill 架构**（设计细节） | [https://docs.openhands.dev/sdk/arch/skill](https://docs.openhands.dev/sdk/arch/skill) |
| 仓库常驻上下文（Repository / `AGENTS.md`） | [https://docs.openhands.dev/overview/skills/repo](https://docs.openhands.dev/overview/skills/repo) |
| 关键词触发类 Skills | [https://docs.openhands.dev/overview/skills/keyword](https://docs.openhands.dev/overview/skills/keyword) |
| **AgentSkills 规范**（OpenHands 扩展版带可选 keyword triggers） | [https://agentskills.io/specification](https://agentskills.io/specification) |
| 社区技能注册表（示例与贡献入口） | [https://github.com/OpenHands/extensions](https://github.com/OpenHands/extensions) |
| 全站文档索引（给 LLM / 检索用） | [https://docs.openhands.dev/llms.txt](https://docs.openhands.dev/llms.txt) |

## 1. 目标与场景

- **算子 / NPU kernel agent**：在挂载到容器的 workspace（通常为 `/opt/workspace`）里写实现、跑 `tools/` 下检查或 profiling，按 `INSTRUCTIONS.md` 迭代。
- **OpenHands Python SDK**（`workspace/entrypoint.py`）：通过 `Agent`、`AgentContext`、`Skill` 把「全局约定 + 领域技能 + 本次任务」注入系统提示，而非仅依赖超长默认 `system_prompt.j2`。
- **rllm 侧**：`openhands_agent.py` 用 Docker 起容器，`NPU_OPERATOR_TASK=1` 时走算子任务 workspace 模板与更贴近 kernel 的 `task_scope` 文案。

## 2. 模板目录结构（`operator_workspace_template/`）

| 路径 | 作用 |
|------|------|
| [`AGENTS.md`](operator_workspace_template/AGENTS.md) | 仓库级全局约定（数值精度、目录约定、提交前检查等），挂载后由 OpenHands 的 **project skills** 加载（若接入 `load_project_skills`）。 |
| [`INSTRUCTIONS.md`](operator_workspace_template/INSTRUCTIONS.md) | 当前任务步骤占位；可被 host 侧 `_setup_*_workspace` 或环境变量 `TASK_INSTRUCTION` 覆盖。 |
| `.agents/skills/<name>/SKILL.md` | **AgentSkills** 风格：每个子目录一个 skill，YAML frontmatter（`name` / `description` / `triggers`）+ 正文。 |
| [`merge_skills_example.py`](operator_workspace_template/merge_skills_example.py) | **三种上下文合并示例**（见下节），注释说明如何嵌进 `entrypoint.py`。 |

示例 skills（两个示例文档，可直接点开）：

- **[AscendC — `SKILL.md`](operator_workspace_template/.agents/skills/ascendc-kernel/SKILL.md)**：`triggers: ascendc, ascend, npu`，AscendC / CANN 约定与编译检查流程。
- **[Triton — `SKILL.md`](operator_workspace_template/.agents/skills/triton-kernel/SKILL.md)**：`triggers: triton, tile`，Triton tile/mask/dtype 与 reference 对齐。

### 文档链接索引（相对本文件所在目录 `examples/openhands-sdk/`）

| 说明 | 链接 |
|------|------|
| 全局约定 | [`operator_workspace_template/AGENTS.md`](operator_workspace_template/AGENTS.md) |
| 任务说明模板 | [`operator_workspace_template/INSTRUCTIONS.md`](operator_workspace_template/INSTRUCTIONS.md) |
| Skills 合并示例代码 | [`operator_workspace_template/merge_skills_example.py`](operator_workspace_template/merge_skills_example.py) |
| AscendC skill | [`operator_workspace_template/.agents/skills/ascendc-kernel/SKILL.md`](operator_workspace_template/.agents/skills/ascendc-kernel/SKILL.md) |
| Triton skill | [`operator_workspace_template/.agents/skills/triton-kernel/SKILL.md`](operator_workspace_template/.agents/skills/triton-kernel/SKILL.md) |
| 容器内入口（生产） | [`workspace/entrypoint.py`](workspace/entrypoint.py) |
| rllm 侧 Docker / 任务 flag | [`openhands_agent.py`](openhands_agent.py) |

## 3. 三种上下文注入方式（概念对齐）

讨论中归纳的三种来源（与 [`merge_skills_example.py`](operator_workspace_template/merge_skills_example.py) 一致）：

1. **`AGENTS.md`（及 `CLAUDE.md` / `GEMINI.md`）**  
   - API：`load_project_skills(workspace_dir=...)`  
   - 语义：**常驻**项目级规则，适合「所有算子任务都成立」的约定。

2. **`.agents/skills/*/SKILL.md`**  
   - API：`load_skills_from_dir(...)` → 得到 `agent_skills` 等，取 `agent_skills.values()` 并入列表。  
   - 语义：**渐进披露**；`description` 供路由/摘要，`triggers` 表示用户或对话中出现关键词时再强调该 skill（具体行为以所用 OpenHands SDK 版本为准）。

3. **代码内联 `Skill(..., trigger=None)`**  
   - 在 `entrypoint.py`（或 rllm 注入处）手写 `task_scope`。  
   - 语义：**本次 rollout** 的短约束（例如「按 INSTRUCTIONS 写 kernel、跑某脚本」）；`trigger=None` 表示始终参与。

**合并顺序建议**：磁盘 ①② 先加入，`task_scope` **放最后**，便于在最终 prompt 里更靠后、强调当前任务。

## 4. `task_scope` 能否挪到 `AGENTS.md`？

- **原则上可以**：把「算子任务总述」写进 `AGENTS.md` 或单独 skill，减少 entrypoint 硬编码。  
- **实际上**仍常见保留一条 **`trigger=None` 的 `task_scope`**，用于 host 按 `NPU_OPERATOR_TASK` / 任务类型切换**当次**说明，而不改整个仓库的 `AGENTS.md`。

## 5. 与当前生产 `entrypoint.py` 的关系（避免误解）

- **当前仓库中的 [`workspace/entrypoint.py`](workspace/entrypoint.py)**（Docker 默认入口）：使用 **硬编码极简 system 模板** + **单个内联 `Skill(name="task_scope", ...)`**，**尚未**调用 `merge_workspace_skills` / `load_project_skills` / `load_skills_from_dir`。  
- **[`merge_skills_example.py`](operator_workspace_template/merge_skills_example.py)** 是 **集成参考**：要在生产里启用 [`AGENTS.md`](operator_workspace_template/AGENTS.md) + `.agents/skills`，需要在 `entrypoint.py` 里按示例改 `AgentContext(skills=...)`，并注意 OpenHands 版本是否支持 `load_public_skills=False` 等参数。

## 6. 多轮对话与 trigger 的直观理解

- **`trigger=None`**：该 skill 内容在多轮中持续作为上下文的一部分（除非 SDK 另有裁剪策略）。  
- **带 `triggers` 的 skill**：偏向「相关时再展开」，减轻每轮 token；**具体触发语义**以当前 OpenHands SDK 文档/实现为准（讨论中仅作架构层对齐）。

## 7. 与 rllm 训练侧的衔接（仅列相关点）

- **轨迹与 reward**：SDK 路径下 rollout 返回值、`per_step` 与 `step.reward` / `trajectory.reward`、以及 `Episode.termination_reason` 未设置导致日志里 `unknown` 等，属于 **训练管线** 话题；本文件专注 **算子 workspace + skills 上下文**。  
- **算子任务 flag**：`openhands_agent.py` 对 NPU 任务设 `NPU_OPERATOR_TASK=1`，entrypoint 据此切换 `_task_scope` 文案（kernel / profiler 导向）。

## 8. 关键文件速查

```
examples/openhands-sdk/
├── openhands_agent.py              # docker run、proxied URL、NPU_OPERATOR_TASK
├── workspace/entrypoint.py         # 当前生产：极简 system + 单 skill task_scope
├── operator_workspace_template/
│   ├── AGENTS.md
│   ├── INSTRUCTIONS.md
│   ├── merge_skills_example.py    # 三种 skills 合并示例
│   └── .agents/skills/
│       ├── ascendc-kernel/SKILL.md
│       └── triton-kernel/SKILL.md
```

---

*本文档由历史讨论整理，若实现与 OpenHands 版本升级不一致，以 SDK 官方行为为准。*
