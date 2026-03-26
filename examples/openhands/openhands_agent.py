"""
OpenHands agent — rollout function for sandbox execution.

This module defines ``rollout(task, config)`` which is the entry point
called by ``worker_server.py`` inside a Docker sandbox. It:

1. Creates a per-rollout workspace with the task instruction
2. Optionally clones a git repository into the workspace
3. Invokes OpenHands headless to solve the coding task
4. Evaluates the result via pytest or keyword heuristics
5. Returns reward as an rllm Trajectory

The function does NOT import rllm.sdk.session — session tracking is
handled externally by worker_server.py via the metadata slug mechanism.

Environment Variables
---------------------
OPENHANDS_MODEL_NAME : str, default "openhands-model"
    Model name exposed by the proxy (must match proxy_config["model_name"]).
OPENHANDS_MAX_ITERATIONS : int, default 30
    Maximum agent iterations per episode.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from typing import Any

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = int(os.environ.get("OPENHANDS_MAX_ITERATIONS", "30"))


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------

def _default_reward(task: dict[str, Any], workspace_dir: str, agent_output: str) -> float:
    """Evaluate the agent's result and compute a scalar reward.

    Tries to run ``pytest`` inside the workspace; falls back to checking
    whether the agent explicitly reported success. Returns 1.0 on success,
    0.0 otherwise.

    Args:
        task: The raw task dict from the dataset.
        workspace_dir: Path to the sandbox directory the agent operated in.
        agent_output: Final text output emitted by the agent.

    Returns:
        Scalar reward in [0.0, 1.0].
    """
    # --- attempt pytest ---
    test_target = task.get("test_file") or task.get("test_dir")
    if test_target:
        test_path = os.path.join(workspace_dir, test_target)
        if os.path.exists(test_path):
            try:
                result = subprocess.run(
                    ["python", "-m", "pytest", test_path, "-x", "-q", "--tb=no"],
                    capture_output=True,
                    cwd=workspace_dir,
                    timeout=60,
                )
                return 1.0 if result.returncode == 0 else 0.0
            except subprocess.TimeoutExpired:
                logger.warning("pytest timed out in %s", workspace_dir)
                return 0.0

    # --- fallback: keyword heuristic in agent output ---
    success_keywords = task.get("success_keywords", [])
    if success_keywords:
        agent_output_lower = agent_output.lower()
        if any(kw.lower() in agent_output_lower for kw in success_keywords):
            return 1.0

    # If no evaluation criteria are provided, default to 0.0 so training
    # doesn't trivially converge; replace with a domain-specific reward.
    logger.warning(
        "No evaluation criteria found in task (expected 'test_file', "
        "'test_dir', or 'success_keywords'). Reward defaults to 0.0. "
        "Set these fields in your dataset or provide a custom reward function."
    )
    return 0.0


# ---------------------------------------------------------------------------
# Workspace setup
# ---------------------------------------------------------------------------

def _setup_workspace(task: dict[str, Any]) -> str:
    """Create a per-rollout workspace with task files.

    If the task contains a ``repo_url``, it will be cloned into the workspace.
    Otherwise, an empty workspace with ``INSTRUCTIONS.md`` is created.
    """
    workspace = tempfile.mkdtemp(prefix=f"openhands-{uuid.uuid4().hex[:8]}-")

    # Optionally clone a repo into the workspace
    repo_url = task.get("repo_url")
    if repo_url:
        subprocess.run(
            ["git", "clone", "--depth=1", repo_url, workspace],
            check=True,
            capture_output=True,
            timeout=120,
        )

    # Write task instruction
    instruction = task.get("instruction", "Fix the bug in the repository.")
    with open(os.path.join(workspace, "INSTRUCTIONS.md"), "w") as f:
        f.write(f"# Task\n\n{instruction}\n")

    return workspace


# ---------------------------------------------------------------------------
# OpenHands invocation
# ---------------------------------------------------------------------------

def _run_openhands(workspace: str, base_url: str, instruction: str) -> str:
    """Run OpenHands headless with the LiteLLM proxy endpoint.

    Args:
        workspace: Path to the working directory.
        base_url: Proxied LiteLLM URL (contains metadata slug for session tracking).
        instruction: Task instruction text.

    Returns:
        Agent's final output text.
    """
    try:
        from openhands.core.config import AppConfig, LLMConfig  # type: ignore[import]
        from openhands.headless import run_openhands  # type: ignore[import]
    except ImportError:
        logger.warning(
            "OpenHands not installed. Falling back to no-op. "
            "Install with: pip install openhands-ai"
        )
        return "OpenHands not available"

    model_name = os.environ.get("OPENHANDS_MODEL_NAME", "openhands-model")

    llm_cfg = LLMConfig(
        model=model_name,
        base_url=base_url,
        api_key="EMPTY",
    )
    app_cfg = AppConfig(
        workspace_base=workspace,
        max_iterations=_MAX_ITERATIONS,
        headless_mode=True,
        # Use local runtime inside the Docker sandbox — the sandbox container
        # IS the sandbox, so no nested Docker is required.
        runtime="local",
    )
    app_cfg.set_llm_config(llm_cfg)

    result = run_openhands(
        task_str=instruction,
        config=app_cfg,
    )
    output_text = getattr(result, "final_output", "") or str(result)
    return output_text


# ---------------------------------------------------------------------------
# Sandbox rollout entry point
# ---------------------------------------------------------------------------

def rollout(task: dict, config: dict) -> list[dict]:
    """Sandbox rollout entry point.

    Called by ``worker_server.py`` with the proxied base_url already
    containing the metadata slug for session tracking.

    Args:
        task: Task dict from the dataset (instruction, repo_url, test_file, ...).
        config: Agent config dict with ``base_url`` and ``session_uid``
                (injected by worker_server.py).

    Returns:
        List of trajectory dicts (single trajectory for OpenHands agent).
    """
    base_url = config.get("base_url", "http://localhost:4000/v1")
    instruction = task.get("instruction", "Fix the bug in the repository.")

    workspace = _setup_workspace(task)

    try:
        # Run OpenHands agent
        agent_output = _run_openhands(
            workspace=workspace,
            base_url=base_url,
            instruction=instruction,
        )

        # Evaluate the result and compute reward
        reward = _default_reward(task, workspace, agent_output)
        logger.info(
            "[openhands] reward=%.2f | instruction=%s",
            reward,
            instruction[:80],
        )

    except Exception:
        logger.exception("[openhands] Rollout failed")
        reward = 0.0

    finally:
        # Clean up workspace
        try:
            shutil.rmtree(workspace, ignore_errors=True)
        except Exception:
            pass

    # Return trajectory in the format expected by worker_server.py
    return [
        {
            "name": "openhands",
            "steps": [],  # steps are tracked via LiteLLM proxy traces
            "reward": reward,
        }
    ]
