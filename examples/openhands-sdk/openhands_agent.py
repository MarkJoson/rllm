"""
OpenHands agent — rollout function for rllm training.

Architecture:
    rllm training process
    └─ rollout(...)  # AgentSdkEngine passes extra_info as kwargs
         ├─ assemble_routing_metadata + build_proxied_base_url (rllm.sdk.proxy.metadata_slug)
         └─ docker run rllm-openhands  (workspace/Dockerfile)
              └─ workspace/entrypoint.py
                   └─ OpenHands SDK (LLM, Agent, Conversation, Tool)
                        └─ LLM calls → LLM_BASE_URL (proxied) → rllm proxy

No openhands Python library is imported in this file. OpenHands runs entirely
inside its own container (built from workspace/Dockerfile). The proxied base
URL with embedded metadata slug is passed as LLM_BASE_URL into the container
so the rllm proxy can attribute all LLM calls to the correct session.

Depends on ``rllm.sdk.proxy.metadata_slug`` (``assemble_routing_metadata`` /
``build_proxied_base_url``) for URL slug construction.

Environment Variables:
    OPENHANDS_IMAGE             : Custom rllm-openhands image
                                  (built from workspace/Dockerfile)
                                  Default: rllm-openhands
    OPENHANDS_SANDBOX_IMAGE     : Runtime image used by OpenHands internally
                                  Default: docker.all-hands.dev/all-hands-ai/runtime:0.28-nikolaik
    OPENHANDS_MODEL_NAME        : Model name on the LiteLLM proxy
                                  Default: openai/openhands-model
    OPENHANDS_MAX_ITERATIONS    : Max agent iterations, default 30
    OPENHANDS_CONTAINER_TIMEOUT : Seconds to wait for container, default 600
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import subprocess
import tempfile
import uuid
from urllib.parse import urlparse, urlunparse
from typing import Any

from rllm.sdk.proxy.metadata_slug import assemble_routing_metadata, build_proxied_base_url
from rllm.types import Trajectory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NPU operator mock (align with openhands-npu bring-up)
# ---------------------------------------------------------------------------

# Workspace package (Docker build context) — seeded into temp dir for operator tasks.
_SDK_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_PKG = os.path.join(_SDK_DIR, "workspace")
_OPERATOR_SEED_NAMES = (
    "AGENTS.md",
    "INSTRUCTIONS.md",
    ".agents",
    "tools",
    "src",
    "refs",
)


def _chmod_executable_scripts(tools_dir: str) -> None:
    if not os.path.isdir(tools_dir):
        return
    for name in os.listdir(tools_dir):
        path = os.path.join(tools_dir, name)
        if os.path.isfile(path) and (name.endswith(".sh") or name.endswith(".py")):
            try:
                os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            except OSError:
                pass


_NPU_INSTRUCTION_TEMPLATE = """# NPU / 算子任务

{instruction}

## 要求

1. 阅读 `AGENTS.md`，在 `INSTRUCTIONS.md` 指定路径实现算子（默认：`src/triton/operator.py` 或 `src/ascendc/kernel.cpp`，取决于后端）。
2. **不要修改** `tools/`。编译与评测一律通过 conda 封装脚本（内部已处理环境）：
   ```bash
   bash /opt/workspace/tools/operator_pipeline.sh
   ```
3. 检查根目录 `metrics.json` 与 `profiling_results.json`，直至 `"success": true`。
4. 失败时根据终端输出与 JSON 中的 `error` 字段修复并重新运行流水线。
"""


def _is_npu_operator_task(task: dict[str, Any]) -> bool:
    return task.get("scenario") == "npu_operator" or task.get("task_type") == "npu_operator"


def _setup_npu_operator_workspace(task: dict[str, Any]) -> str:
    workspace = tempfile.mkdtemp(prefix=f"openhands-npu-{uuid.uuid4().hex[:8]}-")
    instruction = task.get("instruction", "Implement a simple vector_add-style operator.")
    for name in _OPERATOR_SEED_NAMES:
        src = os.path.join(_WORKSPACE_PKG, name)
        dst = os.path.join(workspace, name)
        if not os.path.exists(src):
            continue
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    _chmod_executable_scripts(os.path.join(workspace, "tools"))
    with open(os.path.join(workspace, "INSTRUCTIONS.md"), "w") as f:
        f.write(_NPU_INSTRUCTION_TEMPLATE.format(instruction=instruction))
    return workspace


def _npu_operator_reward(task: dict[str, Any], workspace_dir: str, output: str) -> float:
    del output  # optional hooks for logging extensions
    triton_py = os.path.join(workspace_dir, "src", "triton", "operator.py")
    ascend_cpp = os.path.join(workspace_dir, "src", "ascendc", "kernel.cpp")
    legacy_cpp = os.path.join(workspace_dir, "kernel.cpp")
    has_impl = any(os.path.exists(p) for p in (triton_py, ascend_cpp, legacy_cpp))
    if not has_impl:
        logger.info("[openhands-npu] no operator source found -> reward=0.0")
        return 0.0

    metrics_path = os.path.join(workspace_dir, "metrics.json")
    profiler_json = os.path.join(workspace_dir, "profiling_results.json")
    perf: dict[str, Any] | None = None
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                perf = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[openhands-npu] bad metrics.json: %s", exc)
    if perf is None and os.path.exists(profiler_json):
        try:
            with open(profiler_json) as f:
                perf = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[openhands-npu] bad profiling_results.json: %s", exc)

    if not perf:
        logger.info("[openhands-npu] implementation present, no metrics -> reward=0.2")
        return 0.2
    if not bool(perf.get("success", False)):
        logger.info("[openhands-npu] metrics report failure -> reward=0.2")
        return 0.2
    try:
        bandwidth = float(perf.get("bandwidth_gbps", 0.0))
        theoretical_peak = float(task.get("theoretical_peak_gbps", 1200.0))
        if theoretical_peak <= 0:
            theoretical_peak = 1200.0
        optimization_ratio = min(bandwidth / theoretical_peak, 1.0)
        reward = 0.5 + 0.5 * optimization_ratio
        logger.info("[openhands-npu] success bandwidth=%.1f GB/s reward=%.3f", bandwidth, reward)
        return reward
    except (ValueError, TypeError) as exc:
        logger.warning("[openhands-npu] bad bandwidth in metrics: %s", exc)
        return 0.1


# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

# Custom image built from workspace/Dockerfile (based on official OpenHands
# image but with workspace/entrypoint.py pre-installed as ENTRYPOINT).
_OPENHANDS_IMAGE = os.environ.get("OPENHANDS_IMAGE", "rllm-openhands")
_MODEL_NAME = os.environ.get("OPENHANDS_MODEL_NAME", "openai/openhands-model")
_MAX_ITERATIONS = int(os.environ.get("OPENHANDS_MAX_ITERATIONS", "30"))
_CONTAINER_TIMEOUT = int(os.environ.get("OPENHANDS_CONTAINER_TIMEOUT", "600"))


# ---------------------------------------------------------------------------
# LiteLLM metadata slug (rllm.sdk.proxy.metadata_slug)
# ---------------------------------------------------------------------------


def _trace_label_from_routing_metadata(metadata: dict[str, Any]) -> str:
    uids = metadata.get("session_uids") or []
    if uids:
        return str(uids[-1])[:18]
    name = metadata.get("session_name")
    return str(name or "none")[:18]


def _routing_metadata_for_rollout(
    explicit_uids: list[str] | None,
    explicit_name: str | None,
) -> dict[str, Any]:
    """Slug payload: ``assemble_routing_metadata`` + optional ``_rllm_proxy_session_*`` overrides."""
    extra: dict[str, Any] = {}
    if explicit_uids is not None:
        extra["session_uids"] = list(explicit_uids)
    if explicit_name is not None:
        extra["session_name"] = explicit_name
    return assemble_routing_metadata(extra=extra if extra else None)


def _to_container_url(url: str) -> str:
    """Replace localhost/127.0.0.1 with host.docker.internal so the URL
    is reachable from inside an OpenHands Docker container."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if host in ("localhost", "127.0.0.1"):
        netloc = parsed.netloc.replace(host, "host.docker.internal", 1)
        url = urlunparse(parsed._replace(netloc=netloc))
    return url


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------

def _setup_workspace(task: dict[str, Any]) -> str:
    if _is_npu_operator_task(task):
        return _setup_npu_operator_workspace(task)
    workspace = tempfile.mkdtemp(prefix=f"openhands-{uuid.uuid4().hex[:8]}-")
    repo_url = task.get("repo_url")
    if repo_url:
        subprocess.run(
            ["git", "clone", "--depth=1", repo_url, workspace],
            check=True, capture_output=True, timeout=120,
        )
    instruction = task.get("instruction", "Fix the bug in the repository.")
    with open(os.path.join(workspace, "INSTRUCTIONS.md"), "w") as f:
        f.write(f"# Task\n\n{instruction}\n")
    return workspace


# ---------------------------------------------------------------------------
# Reward evaluation
# ---------------------------------------------------------------------------

def _default_reward(task: dict[str, Any], workspace_dir: str, output: str) -> float:
    if _is_npu_operator_task(task):
        return _npu_operator_reward(task, workspace_dir, output)
    test_target = task.get("test_file") or task.get("test_dir")
    if test_target:
        test_path = os.path.join(workspace_dir, test_target)
        if os.path.exists(test_path):
            try:
                result = subprocess.run(
                    ["python", "-m", "pytest", test_path, "-x", "-q", "--tb=no"],
                    capture_output=True, cwd=workspace_dir, timeout=60,
                )
                return 1.0 if result.returncode == 0 else 0.0
            except subprocess.TimeoutExpired:
                return 0.0

    success_keywords = task.get("success_keywords", [])
    if success_keywords:
        if any(kw.lower() in output.lower() for kw in success_keywords):
            return 1.0

    logger.warning("[openhands] No evaluation criteria in task; reward=0.0")
    return 0.0


# ---------------------------------------------------------------------------
# OpenHands container launch
# ---------------------------------------------------------------------------

def _run_openhands_container(
    workspace: str,
    proxied_url: str,
    instruction: str,
    *,
    npu_operator: bool = False,
    operator_backend: str = "triton",
) -> str:
    """Start an OpenHands headless container, wait for completion, return logs.

    Args:
        workspace:    Host-side workspace directory (mounted into container).
        proxied_url:  LiteLLM proxy URL with embedded rllm metadata slug.
                      Passed as LLM_BASE_URL so rllm can track all LLM calls.
        instruction:  Task instruction (also written to INSTRUCTIONS.md).
    """
    container_name = f"rllm-openhands-{uuid.uuid4().hex[:12]}"

    cmd = [
        "docker", "run",
        "--rm",
        "--name", container_name,

        # --- LLM routing ---
        # proxied_url carries the rllm metadata slug so every LLM call made
        # by OpenHands SDK (via workspace/entrypoint.py) is attributed to
        # this training session by the rllm LiteLLM proxy.
        "-e", f"LLM_BASE_URL={proxied_url}",
        "-e", "LLM_API_KEY=EMPTY",
        "-e", f"LLM_MODEL={_MODEL_NAME}",
        "-e", f"NPU_OPERATOR_TASK={'1' if npu_operator else '0'}",
    ]
    if npu_operator:
        cmd.extend(["-e", f"OPERATOR_BACKEND={operator_backend}"])
    if not npu_operator:
        cmd.extend(["-e", f"TASK_INSTRUCTION={instruction}"])

    cmd.extend(
        [
        # --- Workspace ---
        # The host workspace dir is mounted; the agent operates directly
        # inside the container — no inner sandbox is created.
        "-e", "WORKSPACE_BASE=/opt/workspace",
        "-v", f"{workspace}:/opt/workspace",

        # --- Agent iterations ---
        "-e", f"MAX_ITERATIONS={_MAX_ITERATIONS}",

        # --- Network ---
        # host.docker.internal resolves to the training host so the agent
        # can reach the LiteLLM proxy.
        "--add-host", "host.docker.internal:host-gateway",

        # Custom image built from workspace/Dockerfile.
        # ENTRYPOINT = workspace/entrypoint.py (new OpenHands SDK).
        # No docker.sock mount needed — no inner sandbox.
        _OPENHANDS_IMAGE,
        ]
    )

    logger.info(
        "[openhands] Launching container %s (image=%s, proxied_url=%s...)",
        container_name, _OPENHANDS_IMAGE, proxied_url[:70],
    )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=_CONTAINER_TIMEOUT,
        )
        output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
        if result.returncode != 0:
            logger.warning(
                "[openhands] Container %s exited with code %d",
                container_name, result.returncode,
            )
        return output
    except subprocess.TimeoutExpired:
        logger.error("[openhands] Container %s timed out", container_name)
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
        return ""
    except Exception:
        logger.exception("[openhands] Failed to run container %s", container_name)
        return ""


# ---------------------------------------------------------------------------
# Rollout entry point
# ---------------------------------------------------------------------------

# Keys passed through AgentSdkEngine / session wrapper as kwargs alongside
# extra_info fields; they must not be treated as task payload.
_ROLLOUT_CONFIG_KEYS = frozenset({
    "config",
    "base_url",
    "session_uid",
    "is_validation",
    # Set by rllm.sdk.session.wrap_with_session_context (rllm ≥ patched)
    "_rllm_proxy_session_uids",
    "_rllm_proxy_session_name",
})


def _rollout_task_and_config(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize call shapes.

    - AgentSdkEngine: ``partial(wrapped, metadata, **extra_info)`` → session
      calls ``rollout(instruction=..., scenario=..., ...)`` (kwargs only).
    - Legacy / manual: ``rollout(task_dict, config_dict)``.
    """
    config: dict[str, Any] = {}
    raw_cfg = kwargs.get("config")
    if isinstance(raw_cfg, dict):
        config = dict(raw_cfg)
    if "base_url" in kwargs:
        config["base_url"] = kwargs["base_url"]
    if "session_uid" in kwargs:
        config["session_uid"] = kwargs["session_uid"]

    if len(args) >= 1 and isinstance(args[0], dict):
        task = dict(args[0])
        if len(args) >= 2 and isinstance(args[1], dict):
            config = {**config, **args[1]}
        return task, config

    task = {k: v for k, v in kwargs.items() if k not in _ROLLOUT_CONFIG_KEYS}
    return task, config


def rollout(*args: Any, **kwargs: Any) -> list[dict]:
    """rllm rollout entry point for OpenHands.

    Generates a session-specific metadata slug, builds a proxied LiteLLM URL,
    and launches OpenHands in an isolated Docker container. All LLM calls made
    by OpenHands will carry the metadata slug so the rllm proxy can attribute
    them to this training session.

    Compatible with:

    - ``AgentSdkEngine`` (task fields as keyword args from ``extra_info``).
    - Legacy ``rollout(task_dict, config_dict)``.

    Config may include ``base_url`` (LiteLLM proxy, e.g. ``http://127.0.0.1:4000/v1``).

    Returns:
        List of one ``rllm.types.Trajectory`` (required by ``AgentSdkEngine``).
    """
    slug_uids = kwargs.get("_rllm_proxy_session_uids")
    slug_name = kwargs.get("_rllm_proxy_session_name")
    if slug_uids is not None and not isinstance(slug_uids, list):
        slug_uids = list(slug_uids) if slug_uids else None

    task, config = _rollout_task_and_config(args, kwargs)

    # Raw proxy URL — rllm passes this before any slug is applied
    proxy_url = config.get("base_url", "http://127.0.0.1:4000/v1")

    metadata = _routing_metadata_for_rollout(slug_uids, slug_name)
    trace_label = _trace_label_from_routing_metadata(metadata)
    _uids = metadata.get("session_uids") or []
    logger.info(
        "[openhands] proxy slug: n_uids=%d session_name=%r trace_tail=%s",
        len(_uids),
        metadata.get("session_name"),
        _uids[-1][-12:] if _uids else "",
    )

    proxied_url = build_proxied_base_url(proxy_url, metadata)

    # Rewrite localhost/127.0.0.1 → host.docker.internal so the URL is
    # reachable from inside the OpenHands Docker container.
    proxied_url = _to_container_url(proxied_url)

    npu = _is_npu_operator_task(task)
    workspace = _setup_workspace(task)
    instruction = task.get("instruction", "Fix the bug in the repository.")
    reward = 0.0

    try:
        output = _run_openhands_container(
            workspace,
            proxied_url,
            instruction,
            npu_operator=npu,
            operator_backend=str(task.get("operator_backend", "triton")),
        )
        reward = _default_reward(task, workspace, output)
        logger.info(
            "[openhands] trace=%s reward=%.2f instruction=%s",
            trace_label, reward, instruction[:80],
        )
    except Exception:
        logger.exception("[openhands] Rollout failed (trace=%s)", trace_label)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    # AgentSdkEngine requires list[rllm.types.Trajectory], not plain dicts.
    return [
        Trajectory(
            name="openhands",
            steps=[],
            reward=reward,
            metadata={"trace_label": trace_label},
        )
    ]
