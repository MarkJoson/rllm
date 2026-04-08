"""
Multi-turn baseline evaluation: vLLM + KernelGYM.

Usage:
    python eval_baseline.py \
        --vllm-url http://localhost:8000/v1 \
        --kernelgym-url http://localhost:10907 \
        --data hkust-nlp/drkernel-validation-data \
        --max-turns 3 \
        --output results.json

Prerequisites:
    pip install openai httpx pandas pyarrow
    # Optional: pip install datasets  (for HuggingFace datasets)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_baseline")

# ---------------------------------------------------------------------------
# Code extraction (from KernelGYM KernelAgent)
# ---------------------------------------------------------------------------

_KERNEL_MARKERS = [
    re.compile(r"#\s*Kernel\s+Implementation\s*\n(.*?)(?=#\s*End\b|$)", re.I | re.S),
    re.compile(r"```python\s*#\s*Kernel\s*\n(.*?)```", re.I | re.S),
    re.compile(r"#\s*Your\s+implementation:\s*\n(.*?)(?=#\s*End\b|$)", re.I | re.S),
    re.compile(r"#\s*Generated\s+kernel:\s*\n(.*?)(?=#\s*End\b|$)", re.I | re.S),
]
_CODE_BLOCK_RE = re.compile(r"```(?:[\w+-]+)?\s*\n?(.*?)```", re.S)


def extract_kernel_code(response: str) -> str | None:
    for pat in _KERNEL_MARKERS:
        m = pat.search(response)
        if m:
            return m.group(1).strip()
    blocks = _CODE_BLOCK_RE.findall(response)
    if blocks:
        return blocks[-1].strip()
    return None


# ---------------------------------------------------------------------------
# KernelGYM client (minimal: submit → poll → results)
# ---------------------------------------------------------------------------

class KernelGymClient:
    def __init__(self, base_url: str, timeout: int = 600, poll_interval: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=float(timeout), write=10.0, pool=5.0),
            headers={"Content-Type": "application/json"},
        )

    def evaluate(
        self,
        reference_code: str,
        kernel_code: str,
        entry_point: str = "Model",
        task_timeout: int = 300,
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
        verbose_errors: bool = True,
        enable_profiling: bool = False,
    ) -> dict[str, Any]:
        task_id = f"baseline_{uuid4().hex[:12]}"
        payload = {
            "task_id": task_id,
            "reference_code": reference_code,
            "kernel_code": kernel_code,
            "backend": "triton",
            "entry_point": entry_point,
            "timeout": task_timeout,
            "num_correct_trials": num_correct_trials,
            "num_perf_trials": num_perf_trials,
            "verbose_errors": verbose_errors,
            "enable_profiling": enable_profiling,
        }

        resp = self._client.post(f"{self.base_url}/evaluate", json=payload)
        if resp.status_code != 200:
            return {"status": "failed", "error": f"submit HTTP {resp.status_code}: {resp.text[:200]}"}

        deadline = time.time() + task_timeout + 60
        while time.time() < deadline:
            try:
                s = self._client.get(f"{self.base_url}/status/{task_id}")
                if s.status_code == 200:
                    status = s.json().get("status", "unknown")
                    if status in ("completed", "failed", "timeout", "cancelled"):
                        if status == "completed":
                            r = self._client.get(f"{self.base_url}/results/{task_id}")
                            if r.status_code == 200:
                                result = r.json()
                                result["status"] = status
                                return result
                            return {"status": status, "error": f"fetch results HTTP {r.status_code}"}
                        return {"status": status, "error": s.json().get("error_message", status)}
            except httpx.HTTPError:
                pass
            time.sleep(self.poll_interval)

        return {"status": "timeout", "error": "client-side polling timeout"}


# ---------------------------------------------------------------------------
# Feedback formatting (from KernelGYM multi_turn_kernel.yaml)
# ---------------------------------------------------------------------------

_FEEDBACK_TEMPLATE = """\
Now you have received the server feedback for your last implementation. \
Based on that and all your previous responses, improve the implementation.

Here is the server feedback. Please refer to this feedback to improve the implementation:
Server feedback (status/metrics/errors):
{feedback}

Return an improved Triton implementation named `ModelNew` as a single ```python``` block. Let's think step by step."""


def format_feedback(eval_result: dict[str, Any]) -> str:
    """Serialize full eval result as feedback — aligned with DR.Kernel's env_state injection."""
    feedback_json = json.dumps(eval_result, ensure_ascii=False, indent=2, default=str)
    return _FEEDBACK_TEMPLATE.format(feedback=feedback_json)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class Task:
    task_id: str
    reference_code: str
    entry_point: str = "Model"
    prompt: str = ""


def _parse_row(row: dict, idx: int) -> Task | None:
    """Parse a single data row into a Task, auto-detecting the format.

    Supported formats:
      A) KernelBench raw:  code, name, level, problem_id
      B) rllm JSONL:       reference_code, problem_id, entry_point, description
      C) DR.Kernel parquet: prompt, extra_info.{ground_truth, entry_point, uuid}
    """
    # --- Format A: KernelBench raw (HuggingFace ScalingIntelligence/KernelBench) ---
    if "code" in row and "name" in row:
        level = row.get("level", "")
        pid = row.get("problem_id", idx)
        name = row.get("name", "")
        tid = f"level{level}_{pid}_{name}" if level else f"task_{pid}"
        desc = f"Optimise the PyTorch module '{name}' (Level {level}, Problem {pid})."
        return Task(task_id=tid, reference_code=row["code"],
                    entry_point="Model", prompt=desc)

    # --- Format B: rllm JSONL (from prepare_kernelbench_data.py) ---
    if "reference_code" in row:
        tid = row.get("problem_id", f"task_{idx}")
        entry = row.get("entry_point", "Model")
        prompt = row.get("description", "")
        return Task(task_id=str(tid), reference_code=row["reference_code"],
                    entry_point=entry, prompt=prompt)

    # --- Format C: DR.Kernel parquet (extra_info dict) ---
    extra = row.get("extra_info", {}) or {}
    if isinstance(extra, str):
        extra = json.loads(extra)
    ref_code = extra.get("ground_truth", extra.get("task_code", ""))
    if not ref_code:
        return None
    entry = extra.get("entry_point", "Model")
    tid = extra.get("uuid", extra.get("op_name", f"task_{idx}"))
    prompt_raw = row.get("prompt", "")
    if isinstance(prompt_raw, str) and prompt_raw.startswith("["):
        msgs = json.loads(prompt_raw)
        prompt_text = msgs[-1]["content"] if msgs else ""
    else:
        prompt_text = str(prompt_raw)
    return Task(task_id=str(tid), reference_code=ref_code,
                entry_point=entry, prompt=prompt_text)


def load_tasks(data_path: str, hf_split: str = "level_1") -> list[Task]:
    """Load tasks from parquet, jsonl, or HuggingFace dataset id.

    Args:
        data_path: Local file path (.parquet/.jsonl) or HuggingFace dataset id.
        hf_split: Split name when loading from HuggingFace (default "level_1").
                  KernelBench splits: level_1, level_2, level_3, level_4.
    """
    tasks: list[Task] = []

    path = Path(data_path)
    if path.suffix in (".parquet", ".jsonl") or path.exists():
        import pandas as pd
        if path.suffix == ".jsonl":
            df = pd.read_json(data_path, lines=True)
        else:
            df = pd.read_parquet(data_path)

        for idx, row in df.iterrows():
            t = _parse_row(row.to_dict(), idx)
            if t:
                tasks.append(t)
    else:
        try:
            from datasets import load_dataset
            ds = load_dataset(data_path, split=hf_split)
            for idx, row in enumerate(ds):
                t = _parse_row(dict(row), idx)
                if t:
                    tasks.append(t)
        except Exception as e:
            log.error("Cannot load data from %s (split=%s): %s", data_path, hf_split, e)
            sys.exit(1)

    log.info("Loaded %d tasks from %s", len(tasks), data_path)
    return tasks


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a GPU kernel optimization expert. Given a PyTorch reference implementation (class Model), \
generate an optimized Triton kernel implementation (class ModelNew) that produces identical results \
but runs faster. Return your implementation as a single ```python``` code block containing the \
complete ModelNew class with all necessary imports."""


# ---------------------------------------------------------------------------
# Multi-turn evaluation loop
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    turn: int
    compiled: bool = False
    correctness: bool = False
    speedup: float = 0.0
    error: str | None = None
    kernel_code: str | None = None


@dataclass
class TaskResult:
    task_id: str
    turns: list[TurnResult] = field(default_factory=list)
    best_speedup: float = 0.0
    best_turn: int = -1
    final_correct: bool = False


def run_multi_turn(
    task: Task,
    llm: OpenAI,
    gym: KernelGymClient,
    model_name: str,
    max_turns: int = 3,
    temperature: float = 0.6,
    max_tokens: int = 4096,
    task_timeout: int = 300,
) -> TaskResult:
    result = TaskResult(task_id=task.task_id)

    user_content = task.prompt if task.prompt else (
        f"Here is the PyTorch reference implementation:\n\n```python\n{task.reference_code}\n```\n\n"
        "Generate an optimized Triton kernel implementation named `ModelNew`. "
        "Return a single ```python``` block."
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    for turn_idx in range(max_turns):
        log.info("[%s] Turn %d/%d — generating...", task.task_id, turn_idx + 1, max_turns)

        try:
            completion = llm.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            log.error("[%s] LLM error: %s", task.task_id, e)
            result.turns.append(TurnResult(turn=turn_idx, error=f"LLM error: {e}"))
            break

        kernel_code = extract_kernel_code(response_text)
        if not kernel_code:
            log.warning("[%s] Turn %d: no code extracted", task.task_id, turn_idx + 1)
            result.turns.append(TurnResult(turn=turn_idx, error="no code block found"))
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": format_feedback(
                {"status": "failed", "error_message": "No valid Python code block found in response."}
            )})
            continue

        log.info("[%s] Turn %d — submitting to KernelGYM...", task.task_id, turn_idx + 1)
        eval_result = gym.evaluate(
            reference_code=task.reference_code,
            kernel_code=kernel_code,
            entry_point=task.entry_point,
            task_timeout=task_timeout,
        )

        tr = TurnResult(
            turn=turn_idx,
            compiled=bool(eval_result.get("compiled", False)),
            correctness=bool(eval_result.get("correctness", False)),
            speedup=float(eval_result.get("speedup") or 0.0),
            error=eval_result.get("error_message") or eval_result.get("error"),
            kernel_code=kernel_code,
        )
        result.turns.append(tr)

        log.info(
            "[%s] Turn %d result: compiled=%s correct=%s speedup=%.2fx",
            task.task_id, turn_idx + 1, tr.compiled, tr.correctness, tr.speedup,
        )

        if tr.correctness and tr.speedup > result.best_speedup:
            result.best_speedup = tr.speedup
            result.best_turn = turn_idx
            result.final_correct = True

        if turn_idx < max_turns - 1:
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": format_feedback(eval_result)})

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-turn kernel generation baseline evaluation")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1", help="vLLM OpenAI-compatible API base URL")
    parser.add_argument("--vllm-api-key", default="EMPTY", help="API key for vLLM")
    parser.add_argument("--model", default=None, help="Model name for vLLM (auto-detect if not set)")
    parser.add_argument("--kernelgym-url", default="http://localhost:10907", help="KernelGYM server URL")
    parser.add_argument("--data", required=True, help="Path to parquet/jsonl or HuggingFace dataset id")
    parser.add_argument("--hf-split", default="level_1", help="HuggingFace dataset split (e.g. level_1, level_2, level_3)")
    parser.add_argument("--max-turns", type=int, default=3, help="Max turns per task")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit number of tasks (for quick test)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--task-timeout", type=int, default=300, help="KernelGYM per-task timeout (seconds)")
    parser.add_argument(
        "--max-concurrent-tasks",
        type=int,
        default=1,
        help="Run this many tasks in parallel (each uses its own HTTP clients). "
        "Match to vLLM capacity / KernelGYM worker count (e.g. 4). Default 1 = sequential.",
    )
    parser.add_argument("--output", default="results.json", help="Output file path")
    args = parser.parse_args()

    llm_probe = OpenAI(base_url=args.vllm_url, api_key=args.vllm_api_key)
    model_name = args.model
    if not model_name:
        models = llm_probe.models.list()
        model_name = models.data[0].id if models.data else "default"
        log.info("Auto-detected model: %s", model_name)

    try:
        health = httpx.get(f"{args.kernelgym_url}/health", timeout=10)
        log.info("KernelGYM health: %s", health.status_code)
    except Exception as e:
        log.warning("KernelGYM health check failed: %s (continuing anyway)", e)

    tasks = load_tasks(args.data, hf_split=args.hf_split)
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]

    def _run_one_task(task: Task) -> TaskResult:
        """Separate OpenAI + httpx clients per task (thread-safe for parallel runs)."""
        llm_local = OpenAI(base_url=args.vllm_url, api_key=args.vllm_api_key)
        gym_local = KernelGymClient(args.kernelgym_url, timeout=args.task_timeout + 120)
        return run_multi_turn(
            task,
            llm_local,
            gym_local,
            model_name,
            max_turns=args.max_turns,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            task_timeout=args.task_timeout,
        )

    results: list[TaskResult] = []
    if args.max_concurrent_tasks <= 1:
        for i, task in enumerate(tasks):
            log.info("=== Task %d/%d: %s ===", i + 1, len(tasks), task.task_id)
            results.append(_run_one_task(task))
    else:
        w = min(args.max_concurrent_tasks, len(tasks))
        log.info("Running %d tasks with up to %d workers in parallel", len(tasks), w)
        with concurrent.futures.ThreadPoolExecutor(max_workers=w) as pool:
            # map preserves task order in results
            results = list(pool.map(_run_one_task, tasks))

    # Aggregate metrics
    total = len(results)
    correct_any = sum(1 for r in results if r.final_correct)
    fast_1_0 = sum(1 for r in results if r.final_correct and r.best_speedup >= 1.0)
    fast_1_2 = sum(1 for r in results if r.final_correct and r.best_speedup >= 1.2)
    avg_speedup = (sum(r.best_speedup for r in results if r.final_correct) / correct_any) if correct_any else 0

    summary = {
        "model": model_name,
        "max_turns": args.max_turns,
        "max_concurrent_tasks": args.max_concurrent_tasks,
        "total_tasks": total,
        "correct_any_turn": correct_any,
        "correct_rate": f"{correct_any / total:.1%}" if total else "N/A",
        "fast@1.0": fast_1_0,
        "fast@1.0_rate": f"{fast_1_0 / total:.1%}" if total else "N/A",
        "fast@1.2": fast_1_2,
        "fast@1.2_rate": f"{fast_1_2 / total:.1%}" if total else "N/A",
        "avg_speedup_correct": round(avg_speedup, 3),
    }

    output = {
        "summary": summary,
        "results": [asdict(r) for r in results],
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    log.info("Results saved to %s", output_path)

    print("\n" + "=" * 50)
    print("BASELINE EVALUATION SUMMARY")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()
