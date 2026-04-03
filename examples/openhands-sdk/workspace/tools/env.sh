# shellcheck shell=bash
# 算子工具链环境配置
# OpenHands Agent 禁止手动 conda activate。

# --- Conda ---
: "${CONDA_BASE:=/opt/conda}"
: "${OPERATOR_CONDA_ENV:=operator-build}"

_CONDA_PYTHON="${CONDA_BASE}/envs/${OPERATOR_CONDA_ENV}/bin/python"
if [[ -x "${_CONDA_PYTHON}" ]]; then
  export OPERATOR_PYTHON="${_CONDA_PYTHON}"
elif [[ -n "${OPERATOR_PYTHON:-}" ]]; then
  echo "[env.sh] WARN: conda python not found at ${_CONDA_PYTHON}, using OPERATOR_PYTHON=${OPERATOR_PYTHON}" >&2
else
  echo "[env.sh] ERROR: conda python not found at ${_CONDA_PYTHON} and OPERATOR_PYTHON not set." >&2
  echo "[env.sh] Set CONDA_BASE or OPERATOR_PYTHON in the container environment." >&2
  exit 1
fi

# validate_triton_impl.py 是纯 AST，不需要 torch；用 venv python 即可
export AST_CHECK_PYTHON="${AST_CHECK_PYTHON:-python3}"

# --- Workspace ---
: "${WORKSPACE_BASE:=/opt/workspace}"
