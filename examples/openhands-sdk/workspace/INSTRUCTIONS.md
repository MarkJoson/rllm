# 当前任务

（由 host 侧 `_setup_npu_operator_workspace` 写入；此处为默认占位。）

## 任务格式（KernelBench）

任务文件会以 `{op_name}.py` 形式放在 `src/` 下或直接嵌入此文件。
包含 `Model`（PyTorch 参考）、`get_inputs()`、`get_init_inputs()`。

## 要求

1. 在 `src/{op_name}_triton_ascend_impl.py` 中实现 `ModelNew` 类。
2. 运行 `bash tools/operator_pipeline.sh --op_name <op_name>` 验证。
3. 迭代修复直到 `metrics.json` 报 `"success": true`。
