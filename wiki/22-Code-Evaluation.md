# 代码评估 (Code Evaluation)

代码评估系统负责沙盒化执行 Agent 生成的代码，并通过测试用例验证正确性。

> 源码参考：[rllm/rewards/code_reward.py L1-520](../rllm/rewards/code_reward.py)

## 评估架构

```
Agent 生成的代码响应
    │
    ▼
extract_code_from_model(response)         [L28-41]
    │ 正则匹配: ```(\w+)?\n(.*?)```
    │ 取最后一个代码块
    ▼
clean_code_main_block(code)               [L44-70]
    │ 移除 if __name__ == "__main__" 块
    ▼
数据集分派                                 [L442-462]
    ├── taco/apps → taco_to_lcb_format() → lcb_check_correctness_v2()
    ├── leetcode → leetcode_check_correctness()
    ├── livecodebench/codeforces → lcb_check_correctness_v2()
    ├── kodcode → kodcode_check_correctness()
    └── humanevalplus → humanevalplus_check_correctness()
    │
    ▼
check_correctness(tests, code, test_fn)    [L73-148]
    │ multiprocessing.Process 沙盒执行
    │ 超时保护 + 进程 kill
    ▼
RewardOutput(reward=1.0|0.0, metadata={test_details})
```

## 沙盒执行详解

### check_correctness() 函数

([code_reward.py L73-148](../rllm/rewards/code_reward.py#L73-L148))

```python
def check_correctness(tests, code, test_fn, timeout_per_test=12, max_tests=15):
    # 1. 测试数量限制
    #    选择最长输入的 max_tests 个测试（覆盖更多边界）
    if total_tests > max_tests:
        selected = sorted(range(total_tests), 
                         key=lambda i: len(tests[i]["input"]), 
                         reverse=True)[:max_tests]

    # 2. 进程隔离执行
    manager = Manager()
    test_results = manager.list()
    process = multiprocessing.Process(
        target=evaluate_code, 
        args=(tests, code, False, test_results, test_fn)
    )
    process.start()
    process.join()
    if process.is_alive():
        process.kill()

    # 3. 返回详细结果
    return all(passed), {
        "all_passed": bool,
        "total_tests": int,
        "passed_tests": int,
        "test_results": [
            {"input": str, "expected": str, "passed": bool}
        ]
    }
```

### LiveCodeBench 评估

`lcb_check_correctness_v2()` ([L208-250](../rllm/rewards/code_reward.py#L208-L250))：

```python
def lcb_check_correctness_v2(sample, generation, timeout=6):
    sample = postprocess_lcb_sample(sample)  # 统一格式
    
    # 全局超时 = (timeout+1) × 测试数 + 5秒
    global_timeout = (timeout + 1) * num_tests + 5
    
    process = multiprocessing.Process(target=_temp_run, args=(...))
    process.start()
    process.join(timeout=global_timeout)
    
    if process.is_alive():
        process.kill()
        # 所有测试视为失败
```

### LeetCode 评估

`leetcode_check_correctness()` ([L253-273](../rllm/rewards/code_reward.py#L253-L273))：

```python
def leetcode_check_correctness(tests, code):
    # 使用 firejail 沙盒执行
    succ, output = firejail_code_exec(code + "\n" + tests["functional"])
    return succ, {"all_passed": succ, "output": output}
```

### Together Code Interpreter

`codetool_check_correctness()` ([L365-394](../rllm/rewards/code_reward.py#L365-L394))——远程代码执行：

```python
def codetool_check_correctness(tests, code, codetool, is_taco_format=True):
    # 包装代码 + 测试为可执行脚本
    if call_based:
        test_wrapped = call_based_test_code_wrapper(code, tests)
    else:
        test_wrapped = stdin_test_code_wrapper(code, tests)
    
    # 远程执行
    tool_response = codetool(code=test_wrapped, timeout=30)
    return not tool_response.error, {details}
```

## 安全保证

| 层级 | 机制 | 风险缓解 |
|------|------|---------|
| **进程隔离** | `multiprocessing.Process` | 恶意代码无法访问主进程内存 |
| **超时保护** | `process.join(timeout); process.kill()` | 无限循环/死锁 |
| **Firejail** | 系统级沙盒 (LeetCode) | 文件系统/网络访问 |
| **测试限制** | `max_tests=15` | 测试过多导致超时 |
| **远程沙盒** | Together Code Interpreter | 完全隔离 |

## 测试格式

### stdin/stdout 格式 (TACO/APPS/CodeForces)

```python
tests = {
    "inputs": ["3 30\n2 2 1", "3 10\n3 2 1"],
    "outputs": ["5", "5"]
}
```

### 函数调用格式 (LeetCode/HumanEval)

```python
tests = {
    "functional": """
def test():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
test()
"""
}
```

### LCB 格式 (LiveCodeBench)

```python
tests = [
    {"input": "3 30\n2 2 1", "output": "5", "testtype": "functional",
     "metadata": {"func_name": "solve"}}
]
```
