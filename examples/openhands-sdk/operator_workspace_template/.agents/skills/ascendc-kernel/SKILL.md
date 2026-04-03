---
name: ascendc-kernel
description: >
  AscendC 算子约定：与 CANN/算子 API 相关的注意点与编译检查流程。
  在用户提到 ascend、ascendc、npu kernel 时使用。
triggers:
  - ascendc
  - ascend
  - npu
---

# AscendC 算子

1. 遵守项目规定的入口与内存层次（UB/L1 等）约束。
2. 按 `INSTRUCTIONS.md` 调用指定编译器或 mock profiler。
3. 长参考文档可放在本目录 `references/` 供按需读取。
