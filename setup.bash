#!/usr/bin/env bash
# Installs requirements.txt into the current uv/conda/.venv environment.
# Priority: activated VIRTUAL_ENV -> activated CONDA_PREFIX -> project .venv
# (created with uv venv or python -m venv if missing).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python_from_prefix() {
  local prefix="$1"
  if [[ -x "$prefix/bin/python" ]]; then
    echo "$prefix/bin/python"
  elif [[ -x "$prefix/bin/python3" ]]; then
    echo "$prefix/bin/python3"
  else
    return 1
  fi
}

ensure_venv() {
  if [[ -x "$SCRIPT_DIR/.venv/bin/python" || -x "$SCRIPT_DIR/.venv/bin/python3" ]]; then
    return 0
  fi
  echo "No project .venv found; creating $SCRIPT_DIR/.venv"
  if command -v uv >/dev/null 2>&1; then
    uv venv "$SCRIPT_DIR/.venv"
  elif command -v python3 >/dev/null 2>&1; then
    python3 -m venv "$SCRIPT_DIR/.venv"
  elif command -v python >/dev/null 2>&1; then
    python -m venv "$SCRIPT_DIR/.venv"
  else
    echo "Error: cannot create .venv (need uv, python3, or python)." >&2
    exit 1
  fi
}

resolve_python() {
  local py=""
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    py="$(python_from_prefix "$VIRTUAL_ENV")" && { echo "$py"; return; }
  fi
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    py="$(python_from_prefix "$CONDA_PREFIX")" && { echo "$py"; return; }
  fi
  ensure_venv
  py="$(python_from_prefix "$SCRIPT_DIR/.venv")" && { echo "$py"; return; }
  echo "Error: no usable Python interpreter found." >&2
  exit 1
}

install_requirements() {
  local py="$1"
  if command -v uv >/dev/null 2>&1; then
    # Works for uv venvs (no pip), conda envs, and stdlib venvs.
    uv pip install --python "$py" -r requirements.txt
    return
  fi
  if "$py" -m pip --version >/dev/null 2>&1; then
    "$py" -m pip install --upgrade pip
    "$py" -m pip install -r requirements.txt
    return
  fi
  if [[ -n "${CONDA_PREFIX:-}" ]] && command -v conda >/dev/null 2>&1; then
    conda install -y pip
    "$py" -m pip install -r requirements.txt
    return
  fi
  "$py" -m ensurepip --upgrade
  "$py" -m pip install --upgrade pip
  "$py" -m pip install -r requirements.txt
}

PYBIN="$(resolve_python)"
echo "Using $("$PYBIN" --version) at $PYBIN"
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "Detected activated venv: $VIRTUAL_ENV"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "Detected activated conda env: ${CONDA_DEFAULT_ENV:-$CONDA_PREFIX}"
else
  echo "Using project .venv (activate with: source .venv/bin/activate)"
fi

install_requirements "$PYBIN"

echo "Setup complete. Run: $PYBIN test.py --test-path <absolute-path-to-dataset>"
