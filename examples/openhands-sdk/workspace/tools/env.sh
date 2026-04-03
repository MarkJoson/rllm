# shellcheck shell=bash
# 算子工具链环境 — 占位符路径在镜像构建或 docker run 时替换。
# OpenHands Agent 禁止手动 conda activate。

# --- Conda ---
: "${CONDA_BASE:=/path/to/conda}"
: "${OPERATOR_CONDA_ENV:=operator-build}"

if [[ -x "${CONDA_BASE}/envs/${OPERATOR_CONDA_ENV}/bin/python" ]]; then
  export OPERATOR_PYTHON="${CONDA_BASE}/envs/${OPERATOR_CONDA_ENV}/bin/python"
else
  export OPERATOR_PYTHON="${OPERATOR_PYTHON:-python3}"
fi

# validate_triton_impl.py 是纯 AST，不需要 torch；用 venv python 即可
export AST_CHECK_PYTHON="${AST_CHECK_PYTHON:-python3}"

# --- Workspace ---
: "${WORKSPACE_BASE:=/opt/workspace}"

operator_activate_conda() {
  if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${OPERATOR_CONDA_ENV}" 2>/dev/null || true
  fi
}
