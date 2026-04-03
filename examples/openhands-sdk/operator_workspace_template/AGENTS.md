# 算子 workspace 全局约定（挂载到容器后路径为 `/opt/workspace`）

- 所有 kernel 实现放在本目录约定路径；改前先读 `INSTRUCTIONS.md`。
- 数值：与 reference 对比须写明 dtype、rtol/atol；禁止无说明的精度降级。
- 提交前至少通过 workspace 内提供的编译/检查命令（若有 `tools/` 脚本则按 `INSTRUCTIONS.md` 执行）。
