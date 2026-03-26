# Dockerfile for OpenHands sandbox in rllm training
#
# This image is used by the SandboxOrchestrator to run OpenHands agent
# rollouts inside isolated Docker containers.
#
# Build:
#   docker build -t rllm-openhands-sandbox -f examples/openhands/Dockerfile .
#
# The orchestrator will:
#   1. Upload agent code to /app/agent
#   2. Upload worker_server.py to /app/runner
#   3. Install rllm[sdk] if configured
#   4. Start worker_server.py as the agent process
#
FROM python:3.11-slim

# System dependencies for OpenHands and git operations
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install OpenHands and dependencies needed by worker_server.py
RUN pip install --no-cache-dir \
        openhands-ai \
        aiohttp \
        pytest

# Working directory for agent code (uploaded by orchestrator)
WORKDIR /app/agent

# Default command — overridden by orchestrator's start_agent_process
CMD ["bash"]
