"""
Multi-stage reward function for CUDA kernel generation tasks.

Evaluates agent-generated CUDA kernels through three progressive stages:
  1. Compilation  — does ``nvcc`` succeed? (+0.2)
  2. Correctness  — do all test cases pass within tolerance? (+0.5)
  3. Performance  — is the kernel faster than baseline? (+0.0–0.3)

Each stage gates the next: if compilation fails the total reward is 0.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults (overridable via environment)
# ---------------------------------------------------------------------------
_NVCC = os.environ.get("CUDA_AGENT_NVCC", "nvcc")
_COMPILE_TIMEOUT = int(os.environ.get("CUDA_AGENT_COMPILE_TIMEOUT", "120"))
_RUN_TIMEOUT = int(os.environ.get("CUDA_AGENT_RUN_TIMEOUT", "60"))
_BENCH_REPEATS = int(os.environ.get("CUDA_AGENT_BENCH_REPEATS", "5"))

# Reward weights
_W_COMPILE = 0.2
_W_CORRECT = 0.5
_W_PERF = 0.3


@dataclass
class RewardBreakdown:
    """Detailed reward breakdown for logging / metrics."""

    compile_ok: bool = False
    compile_reward: float = 0.0
    correctness_ok: bool = False
    correctness_reward: float = 0.0
    perf_ratio: float = 0.0
    perf_reward: float = 0.0
    total: float = 0.0
    error: str | None = None


# ---------------------------------------------------------------------------
# Stage 1 — Compilation
# ---------------------------------------------------------------------------

def compile_kernel(
    workspace: str,
    kernel_file: str = "kernel.cu",
    harness_file: str = "test_harness.cu",
    output: str = "kernel_test",
    extra_flags: list[str] | None = None,
) -> tuple[bool, str]:
    """Compile the agent's kernel with the test harness.

    Returns:
        (success, stderr/stdout combined output)
    """
    cmd = [
        _NVCC,
        "-O2",
        "-std=c++17",
        "-o", output,
        kernel_file,
        harness_file,
    ]
    if extra_flags:
        cmd.extend(extra_flags)

    try:
        result = subprocess.run(
            cmd,
            cwd=workspace,
            capture_output=True,
            timeout=_COMPILE_TIMEOUT,
        )
        combined = (result.stdout + result.stderr).decode("utf-8", errors="replace")
        return result.returncode == 0, combined
    except subprocess.TimeoutExpired:
        return False, f"nvcc compilation timed out after {_COMPILE_TIMEOUT}s"
    except FileNotFoundError:
        return False, f"nvcc not found at: {_NVCC}"


# ---------------------------------------------------------------------------
# Stage 2 — Correctness
# ---------------------------------------------------------------------------

def run_correctness_tests(
    workspace: str,
    test_cases: list[dict],
    executable: str = "kernel_test",
) -> tuple[bool, list[dict]]:
    """Run the compiled binary against each test case.

    The test harness is expected to:
      - Accept CLI args:  ./kernel_test <M> <N> [dtype]
      - Print "PASS" on stdout if max absolute error < threshold
      - Print "FAIL" followed by error details otherwise
      - Exit 0 on pass, non-zero on fail

    Returns:
        (all_passed, per-case results)
    """
    results = []
    all_passed = True

    for tc in test_cases:
        args = [os.path.join(workspace, executable)]
        args.append(str(tc.get("M", 128)))
        args.append(str(tc.get("N", 128)))
        if "dtype" in tc:
            args.append(tc["dtype"])

        try:
            result = subprocess.run(
                args,
                cwd=workspace,
                capture_output=True,
                timeout=_RUN_TIMEOUT,
            )
            stdout = result.stdout.decode("utf-8", errors="replace")
            stderr = result.stderr.decode("utf-8", errors="replace")
            passed = result.returncode == 0 and "PASS" in stdout
            results.append({
                "test_case": tc,
                "passed": passed,
                "stdout": stdout[:500],
                "stderr": stderr[:500],
            })
            if not passed:
                all_passed = False
        except subprocess.TimeoutExpired:
            results.append({
                "test_case": tc,
                "passed": False,
                "stdout": "",
                "stderr": f"Test timed out after {_RUN_TIMEOUT}s",
            })
            all_passed = False

    return all_passed, results


# ---------------------------------------------------------------------------
# Stage 3 — Performance benchmarking
# ---------------------------------------------------------------------------

def benchmark_kernel(
    workspace: str,
    test_cases: list[dict],
    executable: str = "kernel_test",
    baseline_ms: float | None = None,
) -> float:
    """Benchmark the kernel using the largest test case.

    The test harness should print a line like ``TIME_MS: 0.123`` to stdout
    when receiving the ``--benchmark`` flag.

    Returns:
        Performance ratio (baseline / actual). Values > 1.0 mean the agent
        kernel is *faster* than baseline. Returns 0.0 on failure.
    """
    if not baseline_ms or baseline_ms <= 0:
        return 1.0  # no baseline → neutral score

    # Use the largest test case for benchmarking
    largest = max(test_cases, key=lambda tc: tc.get("M", 0) * tc.get("N", 0))
    args = [
        os.path.join(workspace, executable),
        "--benchmark",
        str(largest.get("M", 1024)),
        str(largest.get("N", 1024)),
        str(_BENCH_REPEATS),
    ]

    try:
        result = subprocess.run(
            args,
            cwd=workspace,
            capture_output=True,
            timeout=_RUN_TIMEOUT * _BENCH_REPEATS,
        )
        stdout = result.stdout.decode("utf-8", errors="replace")
        for line in stdout.splitlines():
            if "TIME_MS:" in line:
                actual_ms = float(line.split("TIME_MS:")[1].strip())
                if actual_ms > 0:
                    return baseline_ms / actual_ms
    except Exception as e:
        logger.warning("Benchmark failed: %s", e)

    return 0.0


# ---------------------------------------------------------------------------
# Combined evaluator
# ---------------------------------------------------------------------------

def evaluate_cuda_kernel(
    workspace: str,
    test_cases: list[dict],
    baseline_ms: float | None = None,
    kernel_file: str = "kernel.cu",
    harness_file: str = "test_harness.cu",
) -> RewardBreakdown:
    """Evaluate a CUDA kernel through all three stages.

    Args:
        workspace: Directory containing kernel.cu and test_harness.cu.
        test_cases: List of test case dicts with M, N, dtype fields.
        baseline_ms: Optional baseline runtime in milliseconds.
        kernel_file: Name of the agent's kernel source file.
        harness_file: Name of the test harness file.

    Returns:
        RewardBreakdown with per-stage scores and total.
    """
    breakdown = RewardBreakdown()

    # Stage 1: Compilation
    compile_ok, compile_output = compile_kernel(
        workspace, kernel_file, harness_file
    )
    breakdown.compile_ok = compile_ok
    if not compile_ok:
        breakdown.error = f"Compilation failed: {compile_output[:200]}"
        logger.info("[reward] compile FAIL: %s", compile_output[:200])
        return breakdown
    breakdown.compile_reward = _W_COMPILE
    logger.info("[reward] compile PASS")

    # Stage 2: Correctness
    correct_ok, test_results = run_correctness_tests(workspace, test_cases)
    breakdown.correctness_ok = correct_ok
    if not correct_ok:
        breakdown.total = breakdown.compile_reward
        failed = [r for r in test_results if not r["passed"]]
        breakdown.error = f"Correctness failed: {len(failed)}/{len(test_results)} tests failed"
        logger.info("[reward] correctness FAIL: %d/%d", len(failed), len(test_results))
        return breakdown
    breakdown.correctness_reward = _W_CORRECT
    logger.info("[reward] correctness PASS (%d tests)", len(test_results))

    # Stage 3: Performance
    perf_ratio = benchmark_kernel(workspace, test_cases, baseline_ms=baseline_ms)
    breakdown.perf_ratio = perf_ratio
    # Scale: ratio=1.0 → 0.15, ratio=2.0+ → 0.3 (full perf score)
    breakdown.perf_reward = _W_PERF * min(perf_ratio, 2.0) / 2.0
    logger.info("[reward] performance ratio=%.2f", perf_ratio)

    breakdown.total = min(
        breakdown.compile_reward + breakdown.correctness_reward + breakdown.perf_reward,
        1.0,
    )
    return breakdown
