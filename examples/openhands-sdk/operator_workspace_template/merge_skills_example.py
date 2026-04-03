"""
三种上下文注入方式合并示例（复制到 workspace/entrypoint.py 中按需使用）。

挂载关系（与 openhands_agent.py 一致）:
    宿主机目录  ->  容器内 WORKSPACE_BASE=/opt/workspace

1) AGENTS.md          -> load_project_skills(workspace_dir)   常驻
2) .agents/skills/*/SKILL.md -> load_skills_from_dir(...)     渐进披露 + 可选 triggers
3) Skill(..., trigger=None)   -> 代码里拼接                    本次 rollout 短约束
"""
from __future__ import annotations

import os
from pathlib import Path

from openhands.sdk.context import Skill
from openhands.sdk.context.skills import load_project_skills, load_skills_from_dir


def merge_workspace_skills(
    workspace_base: str,
    task_scope: Skill,
) -> list:
    """把磁盘上的 ①② 与代码里的 ③ 合成 AgentContext(skills=...) 的列表。"""
    ws = Path(workspace_base)
    skills: list = []

    # ① 仓库根 AGENTS.md / CLAUDE.md / GEMINI.md（存在则加载）
    if any((ws / name).exists() for name in ("AGENTS.md", "CLAUDE.md", "GEMINI.md")):
        loaded = load_project_skills(workspace_dir=str(ws))
        if loaded:
            skills.extend(loaded if isinstance(loaded, list) else list(loaded))

    # ② AgentSkills：父目录下每个子目录一个 skill（内含 SKILL.md）
    agents_skills_root = ws / ".agents" / "skills"
    if agents_skills_root.is_dir():
        _repo, _knowledge, agent_skills = load_skills_from_dir(str(agents_skills_root))
        skills.extend(agent_skills.values())

    # ③ 本次任务短提示（放最后便于在 prompt 里靠后强调）
    skills.append(task_scope)
    return skills


# ----- 在 entrypoint.py 中的用法（示意） -----
#
# from openhands.sdk import AgentContext
#
# task_scope = Skill(
#     name="task_scope",
#     content="本次 NPU 任务：按 INSTRUCTIONS.md 写 kernel.cpp 并跑 tools/profile_wrapper.sh。",
#     trigger=None,
# )
# agent_context = AgentContext(
#     skills=merge_workspace_skills(WORKSPACE_BASE, task_scope),
#     load_public_skills=False,
#     system_message_suffix=f"Workspace: {WORKSPACE_BASE}. Max iterations: {MAX_ITERATIONS}.",
# )
