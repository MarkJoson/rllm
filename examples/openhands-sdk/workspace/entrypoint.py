#!/usr/bin/env python3
"""
rllm OpenHands entrypoint — runs inside the OpenHands Docker container.

Launched by openhands_agent.py (rllm side) via ``docker run``.
Uses the new OpenHands Python SDK (LLM, Agent, Conversation, Tool) directly.

Configuration via environment variables (set by openhands_agent.py):

    LLM_BASE_URL        Proxied rllm LiteLLM URL with embedded metadata slug.
    LLM_API_KEY         API key for the proxy (default: EMPTY)
    LLM_MODEL           Model name on the LiteLLM proxy
    WORKSPACE_BASE      Workspace directory (default: /opt/workspace)
    MAX_ITERATIONS      Max agent iterations (default: 30)
    OPERATOR_BACKEND    triton (default)
    OPERATOR_ARCH       Target NPU architecture (default: ascend910b1)
    OPERATOR_NAME       Operator name from task data (default: operator)

Exit codes:
    0   Completed
    1   Fatal error
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pydantic import SecretStr

from openhands.sdk import LLM, Agent, AgentContext, Conversation, get_logger
from openhands.sdk.context import Skill
from openhands.sdk.context.skills import load_project_skills, load_skills_from_dir
from openhands.sdk.tool import Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool

logger = get_logger(__name__)

_HARDCODED_SYSTEM_PROMPT_J2 = """You are a helpful assistant. Use the provided tools to complete the task. Be concise.
"""

_MINIMAL_SYSTEM_J2_PATH = "/tmp/rllm_minimal_system.j2"


def merge_workspace_skills(workspace_base: str, task_scope: Skill) -> list:
    """Merge AGENTS.md + .agents/skills/* + inline task_scope."""
    ws = Path(workspace_base)
    skills: list = []

    if any((ws / name).exists() for name in ("AGENTS.md", "CLAUDE.md", "GEMINI.md")):
        loaded = load_project_skills(workspace_dir=str(ws))
        if loaded:
            skills.extend(loaded if isinstance(loaded, list) else list(loaded))

    agents_skills_root = ws / ".agents" / "skills"
    if agents_skills_root.is_dir():
        _repo, _knowledge, agent_skills = load_skills_from_dir(str(agents_skills_root))
        skills.extend(agent_skills.values())

    skills.append(task_scope)
    return skills


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_BASE_URL: str = os.environ.get("LLM_BASE_URL", "")
LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "EMPTY")
LLM_MODEL: str = os.environ.get("LLM_MODEL", "openai/openhands-model")
WORKSPACE_BASE: str = os.environ.get("WORKSPACE_BASE", "/opt/workspace")
MAX_ITERATIONS: int = int(os.environ.get("MAX_ITERATIONS", "30"))
OPERATOR_ARCH: str = os.environ.get("OPERATOR_ARCH", "ascend910b1")
OPERATOR_NAME: str = os.environ.get("OPERATOR_NAME", "operator")

TASK_INSTRUCTION: str = os.environ.get("TASK_INSTRUCTION", "")
if not TASK_INSTRUCTION:
    _md = os.path.join(WORKSPACE_BASE, "INSTRUCTIONS.md")
    if os.path.exists(_md):
        with open(_md) as _f:
            _lines = [l for l in _f.read().splitlines()
                      if l.strip() and not l.startswith("#")]
        TASK_INSTRUCTION = "\n".join(_lines).strip()

if not LLM_BASE_URL:
    logger.error("LLM_BASE_URL is not set. Exiting.")
    sys.exit(1)

if not TASK_INSTRUCTION:
    logger.error("No task instruction provided. Exiting.")
    sys.exit(1)

with open(_MINIMAL_SYSTEM_J2_PATH, "w", encoding="utf-8") as _f:
    _f.write(_HARDCODED_SYSTEM_PROMPT_J2)

logger.info("LLM_BASE_URL : %s...", LLM_BASE_URL[:80])
logger.info("LLM_MODEL    : %s", LLM_MODEL)
logger.info("WORKSPACE    : %s", WORKSPACE_BASE)
logger.info("MAX_ITER     : %d", MAX_ITERATIONS)
logger.info("TASK         : %.120s", TASK_INSTRUCTION)


# ---------------------------------------------------------------------------
# Build SDK objects
# ---------------------------------------------------------------------------

llm = LLM(
    usage_id="rllm-openhands",
    model=LLM_MODEL,
    api_key=SecretStr(LLM_API_KEY),
    base_url=LLM_BASE_URL if LLM_BASE_URL else None,
    max_output_tokens=4096,
)

_task_scope = (
    "You are a Triton-Ascend kernel generation agent. "
    f"Target architecture: {OPERATOR_ARCH}. "
    "Follow AGENTS.md and INSTRUCTIONS.md strictly. "
    f"Implement ModelNew with @triton.jit kernels in src/{OPERATOR_NAME}_triton_ascend_impl.py. "
    "All core computation MUST be in Triton kernels — no PyTorch ops in forward(). "
    f"Do NOT modify tools/. Verify by running: bash tools/operator_pipeline.sh --op_name {OPERATOR_NAME}. "
    "Iterate until metrics.json reports success. Summarize results when done."
)

_task_skill = Skill(
    name="task_scope",
    content=_task_scope,
    trigger=None,
)

try:
    _merged_skills = merge_workspace_skills(WORKSPACE_BASE, _task_skill)
except Exception:
    logger.exception("merge_workspace_skills failed; falling back to task_scope only")
    _merged_skills = [_task_skill]

agent_context = AgentContext(
    skills=_merged_skills,
    load_public_skills=False,
    system_message_suffix=(
        f"Workspace directory: {WORKSPACE_BASE}. "
        f"Maximum iterations budget: {MAX_ITERATIONS}."
    ),
)

agent = Agent(
    llm=llm,
    tools=[
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
    ],
    agent_context=agent_context,
    system_prompt_filename=_MINIMAL_SYSTEM_J2_PATH,
)

conversation = Conversation(
    agent=agent,
    workspace=WORKSPACE_BASE,
    max_iteration_per_run=MAX_ITERATIONS,
)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    exit_code = 0
    try:
        conversation.send_message(TASK_INSTRUCTION)
        conversation.run()

        if llm.metrics is not None:
            cost = llm.metrics.accumulated_cost
            logger.info("EXAMPLE_COST: %s", cost)

        logger.info("Conversation completed successfully.")

    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        exit_code = 1
    except Exception:
        logger.exception("Unhandled exception in entrypoint.")
        exit_code = 1

    sys.exit(exit_code)
