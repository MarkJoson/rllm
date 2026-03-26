"""
Training entry point: CUDA Kernel Agent + AgentSDKEngine (PPO/GRPO).

Uses SandboxOrchestrator to run the agent in Docker containers with GPU
access (方案 B). Each container runs worker_server.py which forwards LLM
requests through the LiteLLM Proxy with metadata slug for session tracking.

Run via train_cuda_agent.sh or directly::

    python3 -m examples.cuda_agent.train_cuda_agent \\
        actor_rollout_ref.model.path=<your_model>

Dataset format
--------------
Each row must have extra_info containing::

    {
        "instruction": "Implement a CUDA kernel for row-wise softmax",
        "test_cases": [{"M": 128, "N": 256}, {"M": 1024, "N": 1024}],
        "performance_baseline": 0.5,  # optional, in ms
        "reference_impl": "path/to/reference.cu",  # optional
    }
"""

import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    try:
        train_dataset = DatasetRegistry.load_dataset("cuda_kernels", "train")
        val_dataset = DatasetRegistry.load_dataset("cuda_kernels", "val")
    except Exception:
        import warnings

        warnings.warn(
            "cuda_kernels dataset not found in DatasetRegistry. "
            "Falling back to 'countdown' for a quick smoke-test. "
            "To use real CUDA tasks, run prepare_cuda_dataset.py first "
            "or override data.train_files / data.val_files.",
            stacklevel=1,
        )
        train_dataset = DatasetRegistry.load_dataset("countdown", "train")
        val_dataset = DatasetRegistry.load_dataset("countdown", "test")

    assert train_dataset, (
        "Train dataset not found. Run prepare_cuda_dataset.py or pass "
        "data.train_files=<path> to provide training data."
    )

    # ------------------------------------------------------------------
    # For sandbox mode (方案 B), the agent_run_func is the sandbox-aware
    # rollout function. The SandboxOrchestrator handles container lifecycle,
    # metadata slug injection, and result collection.
    #
    # For in-process mode (方案 A), you would directly import rollout:
    #   from examples.cuda_agent.cuda_agent import rollout
    #   trainer = AgentTrainer(agent_run_func=rollout, ...)
    # ------------------------------------------------------------------
    from examples.cuda_agent.cuda_agent import rollout

    trainer = AgentTrainer(
        agent_run_func=rollout,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
