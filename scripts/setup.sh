#!/usr/bin/env bash
set -euo pipefail

# Creates a conda env (Python 3.10) to provide the interpreter, then installs
# the project and its dependencies into that env with `uv sync` (reading
# pyproject.toml / uv.lock).
# Usage:
#   bash scripts/setup.sh             # uses default env name 'ars'
#   ENV_NAME=myenv bash scripts/setup.sh

ENV_NAME="${ENV_NAME:-ars}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# Resolve repo root (this script lives in scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found on PATH. Install Miniconda/Anaconda first." >&2
  exit 1
fi

# Ensure 'conda activate' works in a non-interactive shell
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists. Skipping create."
else
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip uv

# Install the project + dependencies into the active conda env from
# pyproject.toml / uv.lock. --active targets the conda env instead of a .venv.
cd "${REPO_ROOT}"
VIRTUAL_ENV="${CONDA_PREFIX}" uv sync --active

echo "Done."
echo "To activate later: conda activate ${ENV_NAME}"
