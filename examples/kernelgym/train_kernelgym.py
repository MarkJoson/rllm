import hydra

from rllm.agents.kernelgym_agent import KernelAgent
from rllm.data.dataset import Dataset
from rllm.environments.kernelgym.kernelgym_env import KernelGymEnv
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = Dataset.load_data(config.get("data", {}).get("train_files", "data/kernelbench_train.jsonl"))
    test_dataset = Dataset.load_data(config.get("data", {}).get("val_files", "data/kernelbench_val.jsonl"))

    agent_args = {
        "system_prompt": (
            "You are an expert GPU kernel engineer. Your task is to write a "
            "high-performance CUDA or Triton kernel that is functionally equivalent "
            "to the given PyTorch reference implementation, but runs faster.\n\n"
            "Instructions:\n"
            "1. Study the reference PyTorch implementation carefully.\n"
            "2. Implement a custom kernel as a Python class named `ModelNew`.\n"
            "3. Your implementation must pass correctness checks.\n"
            "4. Optimise for speed.\n"
            "5. Wrap your final code inside <kernel> ... </kernel> tags."
        ),
    }

    kernel_cfg = config.get("kernel", {})
    env_args = {
        "kernel_server_url": kernel_cfg.get("server_url", "http://localhost:8000"),
        "max_turns": config.get("rllm", {}).get("agent", {}).get("max_steps", 3),
        "backend": kernel_cfg.get("backend", "cuda"),
        "toolkit": kernel_cfg.get("toolkit", "kernelbench"),
        "backend_adapter": kernel_cfg.get("toolkit", "kernelbench"),
    }

    trainer = AgentTrainer(
        agent_class=KernelAgent,
        env_class=KernelGymEnv,
        agent_args=agent_args,
        env_args=env_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
