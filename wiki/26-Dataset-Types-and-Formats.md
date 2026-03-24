# 数据集类型与格式 (Dataset Types and Formats)

## 概览

rLLM 支持多种任务域的数据集，所有数据集统一为 Parquet 格式。

### 数据集完整列表

| 类别 | 数据集 | 任务数 | ground_truth 格式 | 评估方式 |
|------|--------|--------|-------------------|---------|
| 数学 | GSM8K | 8.8K | `str` | grade_answer_mathd + sympy |
| 数学 | MATH | 12.5K | `str \| list[str]` | grade_answer_mathd + sympy |
| 数学 | AIME | ~50 | `str` | 同上 |
| 代码 | LeetCode | 2K+ | `{"functional": test_code}` | firejail_exec |
| 代码 | TACO/APPS | 10K+ | `{"inputs":[], "outputs":[]}` | taco_run_test |
| 代码 | LiveCodeBench | 400+ | `[{"input","output"}]` | lcb_run_test |
| 代码 | HumanEvalPlus | 164 | test_file (str) | humanevalplus_run_test |
| 代码 | KodCode | 6K+ | test_file (str) | kod_code_exec |
| 代码 | CodeForces | 12K+ | `[{"input","output"}]` | lcb_run_test |
| 代码 | PrimeIntellect | 多种 | `[{"input","output","fn_name"}]` | taco_run_test |
| SWE | SWE-bench | 2K+ | test_patch (str) | Docker pytest |
| Web | WebArena | 812 | URL/content match | Playwright |
| RL | FrozenLake | 自定义 | map (str) | 到达终点 |
| 对话 | Search/QA | 多种 | answer (str) | F1 score |

### Parquet 统一 Schema

```python
{
    "extra_info": {       # dict，所有任务信息
        "question": str,      # 问题/任务描述
        "ground_truth": Any,  # 标准答案（格式因域而异）
        "data_source": str,   # 数据集标识
        # ... 域特定字段
    }
}
```

---

# 轨迹处理 (Trajectory Processing)

详见 [05-System-Design-and-Data-Flow.md](05-System-Design-and-Data-Flow.md) 和 [07-Workflow-Engine.md](07-Workflow-Engine.md) 中的 `transform_results_for_verl()` 部分。
