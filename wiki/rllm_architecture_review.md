# rLLM 架构分析 Review：Agentic RL 研究视角的待解疑问

> 基于 [rllm_architecture_analysis.md](file:///home/robomaster/Research/rllm/rllm_architecture_analysis.md) 阅读后，从 Agentic RL 研究开发者视角整理的疑问清单。

---

## 🔴 核心研究相关疑问

### 1. RL 算法层细节缺失

- 文档提到 GRPO/REINFORCE/RLOO/PPO，但**未说明各算法在 rLLM 中的具体实现**。`compute_advantages`（Stage 6）仅一行描述——超参数（KL 系数、clip range、组大小）如何配置？各算法适用场景和经验性能差异？
- **Reward shaping**（`adjust_step_rewards`）的具体实现？MC return 的 γ 折扣如何应用到多步 Agent 场景？

### 2. 自定义 Agent 开发指南

- `BaseAgent` 完整接口（`update_from_model`、`update_from_env`、`chat_completions` 属性）**无详细文档**。实现自定义 agentic agent（多工具调用、树搜索、self-reflection）需要实现哪些方法？有哪些约束？
- Agent 的**状态管理**——多步交互中如何维护内部状态（记忆、计划树）？

### 3. 自定义 Environment 扩展指南

- `BaseEnv` 仅展示 `reset/step` 接口，实际开发 agentic 环境时：
  - **异步环境**如何实现？（需等待外部 API 返回的环境）
  - **多 agent 环境**如何支持？（多个 agent 共享一个环境）
  - `is_multithread_safe` 的具体含义和实现要求？

### 4. 自定义 Workflow 深度指南

- 内置 5 种 Workflow 之外，研究场景的需求：
  - **Solver-Judge** 模式（一个 agent 生成，另一个评判），`TrajectoryGroup` 的 `group_role` 如何配合？
  - **树搜索/MCTS** 式探索——Workflow 是否允许**分支和回溯**？
  - **多 Agent 协作**——一个 episode 中多个 trajectory 来自不同 agent 时的优势计算逻辑？

### 5. 多步奖励与信用分配

- step-level reward 和 trajectory reward 的**信用分配**（credit assignment）机制不清楚。`mc_return` 如何分配到每个 step？是否有更精细的信用分配方法（基于 critic 的 value estimation）？
- **GRPO 在多步场景中如何工作？** GRPO 原设计用于单步场景（同一 prompt 多 response 组内比较），多步 agentic 场景中"组"的定义是什么？

---

## 🟡 工程实践疑问

### 6. 端到端运行示例

- 缺少**完整 agentic 训练配置示例**——YAML 配置、环境定义、agent 定义、奖励函数、训练启动的全流程。`cookbooks/` 和 `examples/` 中是否有相关端到端示例？

### 7. 调试与开发体验

- 如何**调试 Agent 行为**？`EpisodeLogger` 和 `trajectory_visualizer` 的具体输出和使用方式？
- **开发迭代循环**建议：先用 Tinker 后端本地调试 → 切换 verl 分布式训练，切换是否 seamless？

### 8. 性能与扩展性

- 并发 N 个 Workflow 的 N 如何确定？与 GPU 数量、vLLM 批处理能力的关系？
- 多步 Agent 场景下每步 rollout 的**延迟瓶颈**？工具调用等待时间如何处理？

### 9. 模型权重管理

- `wake_up/sleep` 中训练后新权重如何同步到 vLLM？完整拷贝还是增量同步？
- 全异步训练中 `param_sync` 的一致性保证——推理使用哪个版本的权重？是否存在 staleness 问题？

### 10. 与现有研究管线集成

- 如何将 rLLM 与现有 evaluation benchmark（WebArena、SWE-bench、GAIA）集成？需要实现自定义 Environment 还是已有现成支持？
- 是否支持 **offline RL**（从已有 trajectory 数据集训练，非在线收集）？

---

## 📋 建议下一步行动

| 优先级 | 行动 | 目标 |
|--------|------|------|
| **P0** | 深入阅读 `compute_advantages` 和 GRPO 实现源码 | 理解多步场景下的优势计算 |
| **P0** | 阅读 `BaseAgent` 及其子类实现 | 掌握自定义 Agent 的接口规范 |
| **P1** | 运行 `cookbooks/` 中的完整示例 | 建立端到端实操经验 |
| **P1** | 阅读 `adjust_step_rewards` 和 MC return 实现 | 理解信用分配机制 |
| **P2** | 阅读全异步训练和 `param_sync` 实现 | 评估训练效率优化空间 |
| **P2** | 调研 Solver-Judge 和多 Agent Workflow 示例 | 为多 Agent 研究做准备 |
