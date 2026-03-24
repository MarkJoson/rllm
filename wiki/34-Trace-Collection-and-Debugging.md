# Trace 收集与调试 (Trace Collection and Debugging)

详见 [33-Distillation-System.md](33-Distillation-System.md) 中的 Trace 收集与调试部分。

## 关键调试流程

### 1. 检查 Agent 行为

```python
# 运行单个 trajectory 并检查
engine = AgentExecutionEngine(engine_name="openai", ...)
result = await engine.run_agent_trajectory_async(0, mode="Step")
for step in result["steps"]:
    print(f"Observation: {step.observation[:100]}")
    print(f"Action: {step.action}")
    print(f"Reward: {step.reward}")
```

### 2. 检查 Tokenization

```python
chat_parser = ChatTemplateParser.get_parser(tokenizer)
messages = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
]
prompt, response, mask = chat_parser.tokenize_and_mask(messages)
print(f"Prompt tokens: {len(prompt)}")
print(f"Response tokens: {len(response)}")
print(f"Mask sum: {mask.sum()} / {len(mask)}")
```

### 3. 检查 DataProto

```python
# 验证 DataProto 格式
batch = transform_results_for_verl(episodes, task_ids)
print(f"Batch size: {batch.batch['input_ids'].shape[0]}")
print(f"Max prompt len: {batch.batch['prompts'].shape[1]}")
print(f"Max response len: {batch.batch['responses'].shape[1]}")
print(f"Valid: {sum(batch.non_tensor_batch['is_valid'])}")
print(f"Correct: {sum(batch.non_tensor_batch['is_correct'])}")
```

### 4. 指标监控

| 指标 | 含义 | 预期范围 |
|------|------|---------|
| `mean_reward` | 批次平均奖励 | 应随训练逐步上升 |
| `accuracy` | 批次正确率 | 0.0-1.0 |
| `mean_steps` | 平均交互步数 | 取决于 max_steps |
| `llm_time` | LLM 推理耗时 | < env_time (通常) |
| `env_time` | 环境执行耗时 | 取决于环境复杂度 |
| `filtered_rate` | compact 过滤率 | < 50%（否则数据不足） |
| `token_mismatch` | tokenization 不一致 | 应接近 0 |

---

# 测试基础设施 (Testing Infrastructure)

## 测试组织

```
tests/
├── unit/
│   ├── test_types.py          # Step/Trajectory/Episode 数据类型
│   ├── test_reward_math.py    # 数学奖励函数
│   ├── test_reward_code.py    # 代码奖励函数
│   └── test_parser.py         # Chat 模板解析器
├── integration/
│   ├── test_execution.py      # 执行引擎 E2E
│   ├── test_workflow.py       # Workflow 生命周期
│   └── test_transform.py     # DataProto 转换
└── e2e/
    ├── test_verl_train.py     # VERL 完整训练步
    └── test_sdk_mode.py       # SDK 模式完整流程
```

## 运行

```bash
# 所有测试
pytest tests/ -v

# 单元测试（无 GPU）
pytest tests/unit/ -v

# 集成测试（需 GPU）
pytest tests/integration/ -v --gpu

# 特定测试
pytest tests/unit/test_reward_math.py::test_sympy_grading -v
```

## 常用断言

```python
# 奖励函数
assert reward_fn(task, correct_answer).is_correct == True
assert reward_fn(task, wrong_answer).reward == 0.0

# 类型系统
assert traj.is_cumulative() == True
assert len(episode.trajectories) > 0

# DataProto
assert data.batch["input_ids"].shape[0] > 0
assert data.batch["attention_mask"].sum() > 0
```
