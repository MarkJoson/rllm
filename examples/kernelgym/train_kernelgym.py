"""Training entry point for CUDA/Triton kernel RL with KernelGYM.

Usage:
    cd /path/to/rllm
    python examples/kernelgym/train_kernelgym.py \\
        --kernel_server_url http://localhost:8000 \\
        --train_data     data/kernelbench_train.jsonl \\
        --val_data       data/kernelbench_val.jsonl

This mirrors the pattern used by DrKernel's main_kernel.py, but runs inside
rllm's AgentTrainer interface instead of the verl-based RayKernelTrainer.

A minimal JSONL record looks like::

    {
        "problem_id": "relu",
        "reference_code": "import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    def forward(self, x):\n        return torch.relu(x)\n",
        "description": "Implement a fast ReLU kernel.",
        "entry_point": "Model",
        "backend": "cuda"
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a kernel generation agent with rllm + KernelGYM"
    )
    parser.add_argument(
        "--kernel_server_url",
        default="http://localhost:8000",
        help="KernelGYM API server base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--train_data",
        default="data/kernelbench_train.jsonl",
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--val_data",
        default="data/kernelbench_val.jsonl",
        help="Path to validation JSONL file",
    )
    parser.add_argument("--max_turns", type=int, default=3, help="Max revision turns per episode")
    parser.add_argument("--backend", default="cuda", choices=["cuda", "triton"], help="Kernel backend")
    parser.add_argument("--toolkit", default="kernelbench", help="KernelGYM toolkit name")
    parser.add_argument(
        "--config",
        nargs="*",
        default=[],
        help="Extra rllm config overrides e.g. trainer.total_episodes=2000",
    )
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    args = parse_args()

    # --- Validate data files exist ----------------------------------------
    for p in (args.train_data, args.val_data):
        if not Path(p).exists():
            print(f"[ERROR] Data file not found: {p}", file=sys.stderr)
            sys.exit(1)

    # --- Check KernelGYM server health ------------------------------------
    import httpx

    health_url = f"{args.kernel_server_url.rstrip('/')}/health"
    try:
        resp = httpx.get(health_url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "healthy":
            raise RuntimeError(f"Server reports unhealthy: {data}")
        print(f"✅ KernelGYM server healthy at {health_url}")
    except Exception as exc:
        print(f"❌ Cannot reach KernelGYM server at {health_url}: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- Build rllm Dataset objects ---------------------------------------
    from rllm.data import Dataset

    train_records = load_jsonl(args.train_data)
    val_records = load_jsonl(args.val_data)

    train_dataset = Dataset.from_list(train_records)
    val_dataset = Dataset.from_list(val_records)

    # --- Build AgentTrainer -----------------------------------------------
    from rllm import AgentTrainer
    from rllm.agents.kernelgym_agent import KernelAgent
    from rllm.environments.kernelgym.kernelgym_env import KernelGymEnv

    env_args = {
        "kernel_server_url": args.kernel_server_url,
        "max_turns": args.max_turns,
        "toolkit": args.toolkit,
        "backend_adapter": args.toolkit,
        "backend": args.backend,
    }

    config_overrides = list(args.config)

    trainer = AgentTrainer(
        agent_class=KernelAgent,
        env_class=KernelGymEnv,
        agent_args={},
        env_args=env_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config_overrides or None,
    )

    print("🚀 Starting KernelGYM RL training via rllm AgentTrainer …")
    trainer.train()


if __name__ == "__main__":
    main()
