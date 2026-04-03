---
name: triton-kernel
description: >
  Triton 算子编写约定：tile、mask、dtype 与与 PyTorch reference 对齐的检查步骤。
  在用户提到 triton、tile、kernel 优化时使用。
triggers:
  - triton
  - tile
---

# Triton 算子（渐进披露 — 正文可在触发或 agent 按需阅读时注入）

1. 先写/确认 reference（同 shape/dtype）。
2. 处理边界：0 长度、非连续 stride、边界 block。
3. 改完后运行 `INSTRUCTIONS.md` 或本 skill 同目录下 `scripts/` 中列出的检查命令（可在此目录添加 `scripts/check.sh`）。
