#!/usr/bin/env bash
# Bootstrap a conda environment for the pipeline and install requirements.txt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="${ENV_NAME:-piano-corpus}"
PYTHON_VER="${PYTHON_VER:-3.10}"

_pip_pkg_names() {
  conda run -n "$ENV_NAME" python -m pip list --format=freeze \
    | sed -E 's/[<>=].*$//' \
    | sed -E 's/@.*$//' \
    | tr '[:upper:]' '[:lower:]' \
    | tr '_' '-'
}

_has_pkg() {
  local pkg="$1"
  _pip_pkg_names | grep -qx "$(echo "$pkg" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"
}

_install_missing_deps() {
  echo "Checking missing packages from requirements.txt..."

  # madmom builds require Cython at metadata stage; preinstall toolchain first
  echo "Pre-installing build prerequisites (Cython<3, wheel)..."
  conda run -n "$ENV_NAME" python -m pip install "Cython<3" wheel

  mapfile -t installed < <(_pip_pkg_names)
  local installed_blob=" ${installed[*]} "
  local missing_specs=()

  while IFS= read -r raw || [[ -n "$raw" ]]; do
    local spec
    spec="$(echo "$raw" | sed -E 's/[[:space:]]*#.*$//' | xargs)"
    [[ -z "$spec" ]] && continue
    [[ "$spec" == -* ]] && continue

    local pkg
    pkg="$(echo "$spec" \
      | sed -E 's/\[.*\]//' \
      | sed -E 's/[<>=!~ ].*$//' \
      | sed -E 's/@.*$//' \
      | tr '[:upper:]' '[:lower:]' \
      | tr '_' '-')"

    [[ -z "$pkg" ]] && continue
    if [[ "$installed_blob" != *" $pkg "* ]]; then
      missing_specs+=("$spec")
    fi
  done < requirements.txt

  if [[ ${#missing_specs[@]} -gt 0 ]]; then
    echo "Installing missing packages (${#missing_specs[@]}): ${missing_specs[*]}"
    conda run -n "$ENV_NAME" python -m pip install "${missing_specs[@]}"
  else
    echo "No missing packages in requirements.txt."
  fi

  # DGL CUDA. Set DGL_CPU=1 to keep the CPU-only build.
  if [[ "${DGL_CPU:-0}" == "1" ]]; then
    echo "DGL_CPU=1: keeping current DGL state."
  elif _has_pkg dgl; then
    echo "DGL already installed. Skipping reinstall."
  else
    echo "DGL not found. Installing DGL 1.1.3 with CUDA 12.1 support..."
    # Pin to 1.1.3: DGL 2.x drops torchdata.datapipes used by upstream code.
    if ! conda run -n "$ENV_NAME" python -m pip install "dgl==1.1.3" -f https://data.dgl.ai/wheels/cu121/repo.html; then
      echo "DGL CUDA install failed, falling back to CPU build..."
      conda run -n "$ENV_NAME" python -m pip install "dgl>=1.0,<2.0"
    fi
  fi
}

if conda env list | grep -q "^${ENV_NAME} "; then
  echo "Conda env '${ENV_NAME}' already exists."
  echo "Installing only missing packages into the existing env..."
  _install_missing_deps
  echo "Done. Activate with: conda activate ${ENV_NAME}"
  exit 0
fi

echo "Conda env '${ENV_NAME}' not found. Creating it from scratch (Python ${PYTHON_VER})..."
conda create -n "$ENV_NAME" python="$PYTHON_VER" -y

echo "Installing initial dependencies into the new env..."
_install_missing_deps

echo "Done. Activate with: conda activate ${ENV_NAME}"