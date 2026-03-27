"""KernelGYM environment for rllm.

Integrates KernelGYM's HTTP evaluation server with rllm's MultiTurnEnvironment
interface. The LLM iteratively writes and refines CUDA/Triton kernels, receiving
compilation/correctness/speedup feedback from KernelGYM after each attempt.

Reference: DrKernel's AsyncKernelRewardManager for reward formulation.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, Optional, Tuple

from rllm.environments.base.multi_turn_env import MultiTurnEnvironment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reward weights (mirrors DrKernel's AsyncKernelRewardManager)
# ---------------------------------------------------------------------------
_W_COMPILED = 0.1
_W_CORRECT = 0.3
_W_SPEEDUP = 0.6
_SPEEDUP_CLIP_MAX = 10.0  # cap speedup contribution


def _compute_reward(
    compiled: bool,
    correctness: Optional[bool],
    speedup: Optional[float],
) -> float:
    """Compute a scalar reward from KernelGYM evaluation results.

    Reward formula:
        r = 0.1 * compiled
          + 0.3 * correct
          + 0.6 * clip(speedup, 0, 10) / 10
    """
    r = _W_COMPILED * float(compiled)
    if correctness:
        r += _W_CORRECT
    if speedup is not None and speedup > 0:
        clipped = min(speedup, _SPEEDUP_CLIP_MAX)
        r += _W_SPEEDUP * (clipped / _SPEEDUP_CLIP_MAX)
    return round(r, 4)


def _extract_kernel_code(text: str) -> str:
    """Extract kernel code from an LLM response.

    Priority:
    1. ``<kernel>...</kernel>`` tags.
    2. Last ```python ... ``` (or bare ``` ... ```) fenced block.
    3. Entire text as fallback.
    """
    # 1. Explicit tags
    tag_match = re.search(r"<kernel>(.*?)</kernel>", text, re.DOTALL)
    if tag_match:
        return tag_match.group(1).strip()

    # 2. Fenced code blocks (last one wins – usually the final answer)
    fence_matches = re.findall(r"```(?:python|cuda|triton)?\n?(.*?)```", text, re.DOTALL)
    if fence_matches:
        return fence_matches[-1].strip()

    # 3. Fallback
    return text.strip()


class KernelGymEnv(MultiTurnEnvironment):
    """Multi-turn RL environment backed by the KernelGYM evaluation server.

    Each episode:
    - ``reset()``  → returns the task description (reference PyTorch code).
    - ``step(action)`` → submits the kernel code to KernelGYM, receives
      compiled / correctness / speedup metrics, compute reward.
      On failure the next observation contains the error message so the agent
      can revise its kernel in subsequent turns.

    Args:
        task: Task dict with fields:
            - ``problem_id`` (str): Unique problem identifier.
            - ``reference_code`` (str): PyTorch reference implementation.
            - ``description`` (str, optional): Human-readable problem description.
            - ``entry_point`` (str, optional): Class name to evaluate (default "Model").
            - ``toolkit`` (str, optional): KernelGYM toolkit name (default "kernelbench").
            - ``backend_adapter`` (str, optional): Backend adapter (default "kernelbench").
            - ``backend`` (str, optional): Backend type, e.g. "cuda" or "triton".
            - ``kernel_server_url`` (str, optional): Override server URL for this task.
        kernel_server_url: Base URL of the running KernelGYM API server,
            e.g. ``"http://localhost:8000"``.
        max_turns: Maximum LLM refinement rounds per episode (default 3).
        num_correct_trials: Correctness trials passed to KernelGYM (default 5).
        num_perf_trials: Performance timing trials (default 50).
        timeout: Per-evaluation timeout in seconds (default 300).
        toolkit: Default toolkit name (default "kernelbench").
        backend_adapter: Default backend adapter (default "kernelbench").
        backend: Default backend type (default "cuda").
        request_timeout: HTTP request timeout in seconds (default 360).
    """

    def __init__(
        self,
        task: dict | None = None,
        kernel_server_url: str = "http://localhost:8000",
        max_turns: int = 3,
        num_correct_trials: int = 5,
        num_perf_trials: int = 50,
        timeout: int = 300,
        toolkit: str = "kernelbench",
        backend_adapter: str = "kernelbench",
        backend: str = "cuda",
        request_timeout: int = 360,
    ):
        super().__init__(task=task, max_turns=max_turns)
        self.kernel_server_url = kernel_server_url.rstrip("/")
        self.num_correct_trials = num_correct_trials
        self.num_perf_trials = num_perf_trials
        self.timeout = timeout
        self.toolkit = toolkit
        self.backend_adapter = backend_adapter
        self.backend = backend
        self.request_timeout = request_timeout

        # State reset by reset()
        self._last_error: Optional[str] = None
        self._last_result: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, task: dict | None = None, seed: int | None = None) -> Tuple[dict, dict]:
        """Reset the environment and return the initial observation.

        Returns:
            (observation, info) where observation is the task dict.
        """
        if task is not None:
            self.task = task

        assert self.task is not None, "Task must be set before calling reset()"

        self.done = False
        self.current_turn = 0
        self.history = []
        self._last_error = None
        self._last_result = None

        # Initial observation: the task itself (reference code + description)
        return self.task, {}

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, dict]:
        """Submit the kernel code to KernelGYM and compute the reward.

        Args:
            action: Raw LLM response (may contain ``<kernel>`` tags or fenced
                    code blocks). Kernel code is extracted automatically.

        Returns:
            (next_observation, reward, done, info)
        """
        self.history.append(action)

        kernel_code = _extract_kernel_code(action)
        reward, next_obs = self.get_reward_and_next_obs(self.task, kernel_code)

        self.current_turn += 1
        if self.current_turn >= self.max_turns:
            self.done = True
            return {}, reward, self.done, self.task

        return next_obs, reward, self.done, self.task

    def get_reward_and_next_obs(
        self, task: dict, action: str
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate the kernel against KernelGYM and return (reward, next_obs).

        Args:
            task: Task dict (see class docstring).
            action: Extracted kernel code (already stripped of tags/fences).

        Returns:
            Tuple of ``(reward: float, next_observation: dict)``.
        """
        result = self._evaluate_kernel(task, action)
        self._last_result = result

        compiled: bool = result.get("compiled", False)
        correctness: Optional[bool] = result.get("correctness")
        speedup: Optional[float] = result.get("speedup")
        error_msg: Optional[str] = result.get("error_message")

        reward = _compute_reward(compiled, correctness, speedup)

        # Build next observation for the agent
        if self.done or (compiled and correctness):
            next_obs: Dict[str, Any] = {}
        else:
            # Feed back the error / failure mode so the agent can revise
            feedback = self._build_feedback(compiled, correctness, speedup, error_msg)
            next_obs = {
                "feedback": feedback,
                "compiled": compiled,
                "correctness": correctness,
                "speedup": speedup,
                "error_message": error_msg,
            }

        return reward, next_obs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_kernel(self, task: dict, kernel_code: str) -> Dict[str, Any]:
        """POST to KernelGYM /evaluate and wait for the result.

        Uses httpx for synchronous HTTP (already required by kernelgym).
        Falls back to a minimal error dict on any network/timeout error.
        """
        try:
            import httpx
        except ImportError as exc:
            raise ImportError("httpx is required for KernelGymEnv. Install it via 'pip install httpx'.") from exc

        task_id = f"{task.get('problem_id', 'task')}_{uuid.uuid4().hex[:8]}"

        # Resolve task-level server URL override
        server_url = task.get("kernel_server_url", self.kernel_server_url).rstrip("/")

        payload = {
            "task_id": task_id,
            "reference_code": task.get("reference_code", ""),
            "kernel_code": kernel_code,
            "toolkit": task.get("toolkit", self.toolkit),
            "backend_adapter": task.get("backend_adapter", self.backend_adapter),
            "backend": task.get("backend", self.backend),
            "entry_point": task.get("entry_point", "Model"),
            "num_correct_trials": self.num_correct_trials,
            "num_perf_trials": self.num_perf_trials,
            "timeout": self.timeout,
            "priority": "normal",
            "workflow": task.get("workflow", "kernelbench"),
        }

        try:
            with httpx.Client(timeout=self.request_timeout) as client:
                resp = client.post(f"{server_url}/evaluate", json=payload)
                resp.raise_for_status()
                return resp.json()
        except httpx.TimeoutException as exc:
            logger.warning("KernelGYM evaluation timed out for task %s: %s", task_id, exc)
            return {
                "compiled": False,
                "correctness": False,
                "speedup": None,
                "error_message": f"Evaluation timed out after {self.request_timeout}s",
            }
        except Exception as exc:
            logger.warning("KernelGYM evaluation failed for task %s: %s", task_id, exc)
            return {
                "compiled": False,
                "correctness": False,
                "speedup": None,
                "error_message": str(exc),
            }

    @staticmethod
    def _build_feedback(
        compiled: bool,
        correctness: Optional[bool],
        speedup: Optional[float],
        error_message: Optional[str],
    ) -> str:
        """Human-readable feedback string to feed back to the agent."""
        lines = []
        if not compiled:
            lines.append("❌ Compilation FAILED.")
            if error_message:
                lines.append(f"Error:\n{error_message}")
            lines.append("Please fix the compilation error and resubmit your kernel.")
        elif not correctness:
            lines.append("✅ Compilation succeeded.")
            lines.append("❌ Correctness check FAILED — outputs do not match the reference.")
            if error_message:
                lines.append(f"Error:\n{error_message}")
            lines.append("Please fix the numerical correctness issue and resubmit.")
        else:
            speedup_str = f"{speedup:.2f}x" if speedup is not None else "N/A"
            lines.append("✅ Compilation succeeded.")
            lines.append("✅ Correctness check PASSED.")
            lines.append(f"Speedup over reference: {speedup_str}")
            lines.append("Can you further optimize the kernel?")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Factory method
    # ------------------------------------------------------------------

    @staticmethod
    def from_dict(env_args: dict) -> "KernelGymEnv":
        """Create a KernelGymEnv from a configuration dictionary.

        Expected keys (all optional except ``task``):
            task, kernel_server_url, max_turns, num_correct_trials,
            num_perf_trials, timeout, toolkit, backend_adapter, backend,
            request_timeout.
        """
        task = env_args.pop("task", None)
        return KernelGymEnv(task=task, **env_args)

    @staticmethod
    def is_multithread_safe() -> bool:
        """KernelGymEnv is stateless per instance; safe to use across threads."""
        return True
