# Skill 迁移指南

将外部 Agent 技能体系（如 Claude Code / OpenCode 多智能体 skill）迁移到 OpenHands SDK 单智能体 + rllm RL 训练框架的要点总结。

基于 `Just-it/AscendOpGenAgent` 仓库 `br_claudecode` 分支 → 本仓库 `workspace/` 的迁移实践整理。

---

## 1. SKILL.md 前置元数据（YAML Frontmatter）

### 1.1 `argument-hint` → `triggers`

Claude Code 使用 `argument-hint` 字段在多 subagent 间路由 skill。OpenHands SDK 使用 `triggers` 关键词列表做 skill 自动加载匹配。

```yaml
# ❌ 源格式（Claude Code / OpenCode）
name: kernel-generator
description: ...
argument-hint: generate triton kernel code

# ✅ 目标格式（OpenHands SDK）
name: kernel-generator
description: ...
triggers:
  - generate
```

**注意**：`triggers` 是列表类型；每个 trigger 应为单一动词或短语，避免放完整句子。Skill 按 trigger 命中频率自动注入 agent context。

### 1.2 字段保留

| 字段 | 是否保留 | 说明 |
|------|---------|------|
| `name` | ✅ 必须 | skill 唯一标识 |
| `description` | ✅ 必须 | 简短功能描述 |
| `triggers` | ✅ 必须 | 替代 `argument-hint` |
| `argument-hint` | ❌ 删除 | OpenHands 不识别 |
| `allowed-tools` | ❌ 删除 | OpenHands 无 tool 白名单机制 |

---

## 2. 路径规范化

### 2.1 Skill 内 reference 路径

源仓库的 skill 通常使用**相对于 skill 自身目录**的路径。迁移到 OpenHands 后，agent 的 cwd 是 workspace 根目录（`/opt/workspace`），相对路径会解析失败。

```markdown
# ❌ 相对于 skill 目录（源格式）
详见 `references/hint-mode.md`

# ✅ 相对于 workspace 根目录（目标格式）
详见 `.agents/skills/kernel-designer/references/hint-mode.md`
```

**检查清单**：迁移后用 `grep -r 'references/' workspace/.agents/` 扫描所有 skill，确认无残留的裸相对路径。

### 2.2 目录命名：下划线 vs 连字符

Python 模块目录必须用**下划线**（`openhands_sdk`），否则无法 import。非 Python 目录（如 skill 名 `kernel-generator`）可以用连字符。

在一个仓库中统一风格：若顶层目录已用下划线（如 `examples/openhands_sdk/`），则脚本与文档中的路径注释也应保持一致，避免混用 `openhands_sdk` / `openhands-sdk`。

---

## 3. 内容压缩策略

OpenHands 会把所有匹配的 skill 内容注入 agent 的 system prompt，token 成本直接影响训练效率。压缩原则：

### 3.1 可安全删除的内容

| 类型 | 示例 | 原因 |
|------|------|------|
| Claude Code 专用指令 | `Use Bash tool to run ...`、`Call Read tool ...` | OpenHands 有自己的 tool 抽象 |
| 重复的角色声明 | 每个 section 开头的"你是一个 Triton 专家" | 保留一次即可 |
| 完整命令模板 | `python3 scripts/verify.py --impl ... --ref ...` | 已封装进 `operator_pipeline.sh` |
| 参数表格 | 脚本参数的详细说明表 | 由 pipeline 脚本内部处理 |
| 中间 JSON 输出格式 | 各步骤的输出 schema 说明 | agent 只需关注最终 `metrics.json` |
| 硬件规格详细表格 | AI Core 频率、带宽等完整参数 | 移到 `references/` 按需引用 |

### 3.2 必须保留的内容

- **Role 声明**（每个 skill 开头一句话定义角色）
- **核心约束**（如 PyTorch 退化禁止规则、AST 检查要求）
- **分类枚举**（如 A/B/C 类错误分类、修复策略表）
- **Reference 文件索引**（告诉 agent 有哪些参考资料可用）

### 3.3 压缩效果参考

| Skill | 压缩前（行） | 压缩后（行） | 缩减率 |
|-------|-------------|-------------|--------|
| kernel-generator | 280 | 147 | 47% |
| kernel-designer | 135 | 72 | 47% |
| kernel-verifier | 276 | 70 | 75% |
| latency-optimizer | 36 | 31 | 14% |

---

## 4. 多智能体 → 单智能体工作流适配

### 4.1 架构差异

```
Claude Code / OpenCode（源）          OpenHands + rllm（目标）
┌──────────────────────┐            ┌──────────────────────┐
│  orchestrator        │            │  单一 agent          │
│  ├─ designer agent   │    →       │  ├─ Phase 2: 设计    │
│  ├─ generator agent  │            │  ├─ Phase 3: 生成    │
│  ├─ verifier agent   │            │  ├─ Phase 3: 验证    │
│  └─ optimizer agent  │            │  └─ Phase 4: 优化    │
└──────────────────────┘            └──────────────────────┘
```

- 源框架中每个 subagent 有独立 context，skill 只注入对应 agent
- 目标框架中所有 skill 可能同时注入同一个 agent，因此**去重和压缩更重要**

### 4.2 Skill 间职责边界

即使是单 agent，skill 之间的职责仍需清晰划分，避免指令冲突：

| Skill | 职责边界 | 不应包含 |
|-------|---------|---------|
| kernel-designer | 分析任务 → 输出设计方案 / sketch | 具体代码生成 |
| kernel-generator | 根据设计生成代码 | 验证/性能测试命令 |
| kernel-verifier | 调用 pipeline 验证 + 解读结果 | 代码修改建议 |
| latency-optimizer | 性能调优策略 | 正确性修复 |

### 4.3 工作流编排

在 `AGENTS.md`（全局约定文件）中定义 Phase 流程，各 skill 只描述"做什么"，不描述"何时做"。

---

## 5. 工具链封装

### 5.1 将分散脚本合并为统一 Pipeline

源仓库的 skill 通常直接指导 agent 调用各个独立脚本。迁移时应封装为**统一入口**：

```
# ❌ 源格式：skill 内写 3 条命令
1. python3 scripts/validate_triton_impl.py ...
2. python3 scripts/verify.py ...
3. python3 scripts/benchmark.py ...

# ✅ 目标格式：一条命令
bash tools/operator_pipeline.sh --op_name {op_name}
```

好处：
- Skill 内容大幅缩减（删除所有命令模板和参数说明）
- 环境切换（venv ↔ conda）在 pipeline 内部透明处理
- 新增步骤不需要改 skill

### 5.2 环境隔离

当容器内存在多个 Python 环境时，通过 `tools/env.sh` 统一定义解释器路径：

```bash
# AST 静态检查 —— 在 venv 里跑（不需要硬件库）
AST_CHECK_PYTHON="/opt/venv/bin/python3"

# 编译/验证/性能 —— 在 conda 环境里跑（需要 torch_npu 等）
OPERATOR_PYTHON="/opt/conda/envs/triton/bin/python3"
```

`AGENTS.md` 中明确禁止 agent 手动 `conda activate`，一切由脚本处理。

---

## 6. 参数传递链路

从训练数据到容器内 agent，参数经过多层传递，必须保持一致：

```
parquet (extra_info)
  ├─ op_name         ─┐
  ├─ arch            ─┤ openhands_agent.py 提取
  ├─ task_code       ─┤
  └─ instruction     ─┘
       │
       ▼
  _setup_npu_operator_workspace()
  ├─ INSTRUCTIONS.md ← 模板 format(op_name, arch, instruction)
  ├─ src/{op_name}.py ← task_code 写入
  └─ Docker env vars: OPERATOR_ARCH={arch}
       │
       ▼
  entrypoint.py（容器内）
  ├─ 读取 OPERATOR_ARCH 环境变量
  └─ 注入 agent system prompt
```

**检查要点**：
- `create_mock_npu_operator_data.py` 的 `extra_info` 字段必须与 `_setup_npu_operator_workspace` 的 `task.get()` key 一一对应
- `_NPU_INSTRUCTION_TEMPLATE` 的 `{占位符}` 必须在 `.format()` 调用中全部填充
- `entrypoint.py` 读取的环境变量名必须与 `docker run -e` 传入的一致

---

## 7. Reward 与最佳版本追踪

### 7.1 渐进式 Reward

为 RL 训练提供连续梯度信号，而非 0/1 二值：

| 阶段 | 条件 | Reward |
|------|------|--------|
| 无实现文件 | `impl.py` 不存在 | 0.0 |
| 有实现但无 metrics | 文件存在但 pipeline 未跑/全部失败 | 0.2 |
| AST 通过 | `ast_check_ok: true` | 0.3 |
| 正确性通过 | `correctness_ok: true` | 0.4 |
| 性能达标 | `success: true` | 0.5 + f(speedup)，上限 1.0 |

### 7.2 最佳版本追踪

`AGENTS.md` 中指示 agent 在每次 pipeline 成功后比较并保存最佳版本：

- `src/{op_name}_triton_ascend_impl_best.py` — 最佳代码备份
- `metrics_best.json` — 最佳指标备份

host 侧 `_npu_operator_reward` 在当前 metrics 不佳时自动 fallback 到 `metrics_best.json` 取 reward，确保优化过程中的回退不会惩罚已取得的成绩。

### 7.3 产物归档

训练结束后 temp workspace 会被删除。通过 `OPENHANDS_ARTIFACT_DIR` 环境变量启用归档，在删除前保存关键产物（代码、metrics、manifest），用于训练后分析和 case study。

---

## 8. 迁移检查清单

新 skill 迁移时按此清单逐项确认：

- [ ] YAML frontmatter: `argument-hint` → `triggers`，删除 `allowed-tools`
- [ ] 每个 skill 开头有一句 role 声明
- [ ] 所有 `references/` 路径改为 workspace 根目录相对路径（`.agents/skills/xxx/references/...`）
- [ ] 删除 Claude Code / OpenCode 专用的 tool 调用指令
- [ ] 删除已封装进 pipeline 的命令模板和参数表
- [ ] 删除与其他 skill 重复的内容（如硬件规格、通用约束）
- [ ] `AGENTS.md` 中注册新 skill 的 Phase 和迭代上限
- [ ] `entrypoint.py` 的 `merge_workspace_skills` 能发现新 skill 目录
- [ ] `_OPERATOR_SEED_NAMES` 包含新 skill 的父目录（通常是 `.agents`，已覆盖）
- [ ] Mock 数据（parquet）的 `extra_info` 包含新 skill 需要的所有字段
- [ ] Pipeline 脚本能处理新 skill 引入的产物文件
- [ ] 归档逻辑覆盖新 skill 产生的关键文件

---

## 9. 文件结构参考

```
workspace/
├── AGENTS.md                          # 全局约定、Phase 工作流、错误分类
├── INSTRUCTIONS.md                    # 任务模板（host 侧动态填充）
├── entrypoint.py                      # 容器入口（host 侧常只读挂载进 OpenHands 镜像）
├── .agents/skills/
│   ├── kernel-designer/
│   │   ├── SKILL.md
│   │   └── references/                # 设计案例、sketch 模板
│   ├── kernel-generator/
│   │   ├── SKILL.md
│   │   └── references/                # 硬件规格、Triton API 参考
│   ├── kernel-verifier/
│   │   └── SKILL.md
│   └── latency-optimizer/
│       ├── SKILL.md
│       └── references/                # 优化案例
├── tools/
│   ├── env.sh                         # 环境变量定义（Python 解释器路径）
│   ├── operator_pipeline.sh           # 统一验证入口
│   └── scripts/
│       ├── validate_triton_impl.py    # AST 静态检查
│       ├── verify.py                  # 数值正确性验证
│       └── benchmark.py              # 性能 profiling
└── src/                               # 运行时生成（host 写入 task code，agent 写入 impl）
```
