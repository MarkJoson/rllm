"""
CUDA kernel agent — rollout function for sandbox execution.

This module defines ``rollout(task, config)`` which is the entry point
called by ``worker_server.py`` inside a sandbox (Docker container or
local subprocess). It:

1. Sets up a workspace with the test harness and task instruction
2. Invokes OpenHands headless to generate a CUDA kernel
3. Evaluates the kernel via ``cuda_reward.evaluate_cuda_kernel``
4. Returns reward as an rllm Trajectory

The function does NOT import rllm.sdk.session — session tracking is
handled externally by worker_server.py via the metadata slug mechanism.

Environment Variables
---------------------
CUDA_AGENT_MAX_ITERATIONS : int, default 20
    Maximum OpenHands agent iterations.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = int(os.environ.get("CUDA_AGENT_MAX_ITERATIONS", "20"))

# Path to templates shipped alongside this module
_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def _setup_workspace(task: dict) -> str:
    """Create a per-rollout workspace with task files.

    Contents written:
      - test_harness.cu  (from template)
      - INSTRUCTIONS.md  (task instruction)
      - If reference_impl exists, write reference.cu
    """
    workspace = tempfile.mkdtemp(prefix=f"cuda-agent-{uuid.uuid4().hex[:8]}-")

    # Copy test harness template
    harness_src = os.path.join(_TEMPLATE_DIR, "test_harness_template.cu")
    harness_dst = os.path.join(workspace, "test_harness.cu")
    if os.path.exists(harness_src):
        shutil.copy2(harness_src, harness_dst)
    else:
        logger.warning("Test harness template not found at %s", harness_src)

    # Write task instruction
    instruction = task.get("instruction", "Implement the target_kernel function in kernel.cu")
    test_cases = task.get("test_cases", [{"M": 128, "N": 128}])

    test_case_desc = "\n".join(
        f"  - M={tc.get('M', 128)}, N={tc.get('N', 128)}, dtype={tc.get('dtype', 'float32')}"
        for tc in test_cases
    )

    instructions = f"""# CUDA Kernel Task

## Objective
{instruction}

## Requirements
1. Create `kernel.cu` in the current directory
2. Implement `__global__ void target_kernel(const float* input, float* output, int M, int N)`
3. Optionally implement `void launch_kernel(const float* d_input, float* d_output, int M, int N)` for custom launch config
4. The kernel will be compiled with: `nvcc -O2 -std=c++17 -o kernel_test kernel.cu test_harness.cu`
5. Correctness is verified against a reference CPU implementation

## Test Cases
{test_case_desc}

## Commands
- Compile: `nvcc -O2 -std=c++17 -o kernel_test kernel.cu test_harness.cu`
- Test:    `./kernel_test <M> <N>`
- Bench:   `./kernel_test --benchmark <M> <N> 5`

## Evaluation
- Compilation must succeed
- All test cases must print "PASS" (max abs error < 1e-5)
- Performance is measured relative to a baseline
"""
    with open(os.path.join(workspace, "INSTRUCTIONS.md"), "w") as f:
        f.write(instructions)

    # Copy reference implementation if provided
    ref_impl = task.get("reference_impl")
    if ref_impl and os.path.exists(ref_impl):
        shutil.copy2(ref_impl, os.path.join(workspace, "reference.cu"))

    return workspace


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

    model_name = os.environ.get("CUDA_AGENT_MODEL_NAME", "cuda-agent-model")

    llm_cfg = LLMConfig(
        model=model_name,
        base_url=base_url,
        api_key="EMPTY",
    )
    app_cfg = AppConfig(
        workspace_base=workspace,
        max_iterations=_MAX_ITERATIONS,
        headless_mode=True,
        runtime="local",
    )
    app_cfg.set_llm_config(llm_cfg)

    result = run_openhands(
        task_str=instruction,
        config=app_cfg,
    )
    output_text = getattr(result, "final_output", "") or str(result)
    return output_text


def rollout(task: dict, config: dict) -> list[dict]:
    """Sandbox rollout entry point.

    Called by ``worker_server.py`` with the proxied base_url already
    containing the metadata slug for session tracking.

    Args:
        task: Task dict from the dataset (instruction, test_cases, ...).
        config: Agent config dict with ``base_url`` and ``session_uid``.

    Returns:
        List of trajectory dicts (single trajectory for CUDA agent).
    """
    from examples.cuda_agent.cuda_reward import evaluate_cuda_kernel

    base_url = config.get("base_url", "http://localhost:4000/v1")
    instruction = task.get("instruction", "Implement a CUDA softmax kernel")
    test_cases = task.get("test_cases", [{"M": 128, "N": 128}])
    baseline_ms = task.get("performance_baseline")

    workspace = _setup_workspace(task)

    try:
        # Run OpenHands agent to generate kernel.cu
        agent_output = _run_openhands(
            workspace=workspace,
            base_url=base_url,
            instruction=instruction,
        )

        # Evaluate the generated kernel
        breakdown = evaluate_cuda_kernel(
            workspace=workspace,
            test_cases=test_cases,
            baseline_ms=baseline_ms,
        )

        reward = breakdown.total
        logger.info(
            "[cuda-agent] reward=%.3f (compile=%s, correct=%s, perf=%.2f) | %s",
            reward,
            breakdown.compile_ok,
            breakdown.correctness_ok,
            breakdown.perf_ratio,
            instruction[:60],
        )

    except Exception:
        logger.exception("[cuda-agent] Rollout failed")
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
            "name": "cuda-kernel",
            "steps": [],  # steps are tracked via LiteLLM proxy traces
            "reward": reward,
        }
    ]
