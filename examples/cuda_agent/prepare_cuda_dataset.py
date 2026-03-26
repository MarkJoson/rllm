"""
Prepare CUDA kernel training dataset and register with DatasetRegistry.

Creates a set of CUDA kernel generation tasks with varying difficulty,
covering common GPU compute patterns (softmax, gemm, layernorm, reduce, etc.).

Usage::

    python3 -m examples.cuda_agent.prepare_cuda_dataset

This registers two datasets:
  - ``cuda_kernels/train`` — training tasks
  - ``cuda_kernels/val``   — validation tasks (held-out kernel types)
"""

from __future__ import annotations

import json
import os

from rllm.data.dataset import DatasetRegistry


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
# Each task specifies what kernel the agent should write.
# The test harness ships with a default softmax reference; for other kernels
# you'd need to provide a matching reference_impl or modify the harness.

TRAIN_TASKS = [
    # --- Softmax variants ---
    {
        "instruction": "Implement a CUDA kernel for row-wise softmax on a float32 M×N matrix. "
        "Use shared memory to find the row max and compute the sum of exponentials efficiently.",
        "test_cases": [
            {"M": 128, "N": 256},
            {"M": 512, "N": 512},
            {"M": 1024, "N": 1024},
        ],
        "performance_baseline": 0.5,
        "tags": ["softmax", "shared_memory", "reduction"],
    },
    {
        "instruction": "Implement a numerically stable online softmax CUDA kernel. "
        "Process each row in a single pass using the online softmax trick "
        "(track running max and sum simultaneously).",
        "test_cases": [
            {"M": 256, "N": 512},
            {"M": 1024, "N": 2048},
        ],
        "performance_baseline": 0.8,
        "tags": ["softmax", "online", "numerical_stability"],
    },
    # --- Reduction ---
    {
        "instruction": "Implement a CUDA kernel that computes the sum reduction of each row "
        "in a float32 M×N matrix. The output should be a vector of M elements, "
        "each being the sum of the corresponding row. Use warp shuffles for the final reduction.",
        "test_cases": [
            {"M": 256, "N": 1024},
            {"M": 1024, "N": 4096},
        ],
        "performance_baseline": 0.2,
        "tags": ["reduction", "warp_shuffle"],
    },
    {
        "instruction": "Implement a CUDA kernel for parallel prefix sum (inclusive scan) "
        "on each row of a float32 M×N matrix. Use the Blelloch algorithm with "
        "shared memory. Each row should be scanned independently.",
        "test_cases": [
            {"M": 128, "N": 256},
            {"M": 512, "N": 512},
        ],
        "performance_baseline": 0.6,
        "tags": ["scan", "prefix_sum", "shared_memory"],
    },
    # --- Element-wise ---
    {
        "instruction": "Implement a CUDA kernel for GELU activation on a float32 M×N matrix. "
        "Use the approximation: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³))). "
        "Maximize throughput via vectorized loads (float4).",
        "test_cases": [
            {"M": 512, "N": 512},
            {"M": 2048, "N": 2048},
        ],
        "performance_baseline": 0.1,
        "tags": ["activation", "gelu", "vectorized"],
    },
    {
        "instruction": "Implement a CUDA kernel for ReLU activation: output[i] = max(0, input[i]). "
        "Use vectorized memory access (float4) for maximum memory bandwidth utilization.",
        "test_cases": [
            {"M": 1024, "N": 1024},
            {"M": 4096, "N": 4096},
        ],
        "performance_baseline": 0.05,
        "tags": ["activation", "relu", "vectorized"],
    },
    # --- Normalization ---
    {
        "instruction": "Implement a CUDA kernel for Layer Normalization over the last dimension. "
        "For each row, compute mean and variance, then normalize: "
        "output[i][j] = (input[i][j] - mean[i]) / sqrt(var[i] + epsilon). "
        "Use epsilon=1e-5. Use shared memory for the reduction.",
        "test_cases": [
            {"M": 256, "N": 768},
            {"M": 512, "N": 1024},
        ],
        "performance_baseline": 0.4,
        "tags": ["layernorm", "normalization", "shared_memory"],
    },
    # --- Transpose ---
    {
        "instruction": "Implement a CUDA kernel for matrix transpose of a float32 M×N matrix. "
        "Use shared memory tiling (32×32 tiles) to avoid uncoalesced global memory accesses. "
        "Handle the case where M or N is not a multiple of 32.",
        "test_cases": [
            {"M": 256, "N": 512},
            {"M": 1024, "N": 1024},
            {"M": 1000, "N": 777},
        ],
        "performance_baseline": 0.3,
        "tags": ["transpose", "tiling", "shared_memory"],
    },
]

VAL_TASKS = [
    # Held-out kernel types for validation
    {
        "instruction": "Implement a CUDA kernel for row-wise L2 normalization of a float32 M×N matrix. "
        "For each row, compute the L2 norm and divide each element by it: "
        "output[i][j] = input[i][j] / sqrt(sum(input[i][:]²) + 1e-12).",
        "test_cases": [
            {"M": 256, "N": 512},
            {"M": 1024, "N": 1024},
        ],
        "performance_baseline": 0.3,
        "tags": ["normalization", "l2norm"],
    },
    {
        "instruction": "Implement a CUDA kernel for element-wise sigmoid: "
        "output[i] = 1.0 / (1.0 + exp(-input[i])). "
        "Ensure numerical stability for large negative inputs.",
        "test_cases": [
            {"M": 512, "N": 512},
            {"M": 2048, "N": 2048},
        ],
        "performance_baseline": 0.08,
        "tags": ["activation", "sigmoid"],
    },
]


def _task_to_row(task: dict, idx: int) -> dict:
    """Convert a task spec to a DatasetRegistry-compatible row."""
    return {
        "extra_info": task,
        "task_id": f"cuda_kernel_{idx:04d}",
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare CUDA kernel dataset")
    parser.add_argument("--output-dir", default=None, help="Directory to write parquet files")
    args = parser.parse_args()

    train_rows = [_task_to_row(t, i) for i, t in enumerate(TRAIN_TASKS)]
    val_rows = [_task_to_row(t, i + len(TRAIN_TASKS)) for i, t in enumerate(VAL_TASKS)]

    # Register with DatasetRegistry
    DatasetRegistry.register("cuda_kernels", "train", train_rows)
    DatasetRegistry.register("cuda_kernels", "val", val_rows)

    print(f"Registered datasets:")
    print(f"  cuda_kernels/train: {len(train_rows)} tasks")
    print(f"  cuda_kernels/val:   {len(val_rows)} tasks")

    # Optionally write to disk
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for split, rows in [("train", train_rows), ("val", val_rows)]:
            path = os.path.join(args.output_dir, f"cuda_kernels_{split}.json")
            with open(path, "w") as f:
                json.dump(rows, f, indent=2)
            print(f"  Written to: {path}")

    print("\nExample task:")
    print(json.dumps(TRAIN_TASKS[0], indent=2))


if __name__ == "__main__":
    main()
