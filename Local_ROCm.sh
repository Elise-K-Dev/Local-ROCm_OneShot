#!/usr/bin/env bash
set -euo pipefail

# Arch Linux + RX 7900 XT + ROCm + source-built vLLM + Gemma 4 E4B-it
# This script recreates the working setup and starts the server.
#
# Assumptions:
# - Miniconda is installed at ~/miniconda3
# - ROCm is installed under /opt/rocm
# - You want a conda env named vllm-src
# - You want to build vLLM from source in ~/src/vllm
# - You already have access to google/gemma-4-E4B-it via Hugging Face
#
# Usage:
#   bash vllm_gemma_rocm_setup_and_serve.sh
#
# Optional env vars:
#   ENV_NAME=vllm-src
#   VLLM_SRC_DIR=$HOME/src/vllm
#   MODEL_ID=google/gemma-4-E4B-it
#   MAX_MODEL_LEN=1024
#   GPU_MEMORY_UTILIZATION=0.95
#   RECREATE_ENV=1
#   CLONE_VLLM=1

ENV_NAME="${ENV_NAME:-vllm-src}"
VLLM_SRC_DIR="${VLLM_SRC_DIR:-$HOME/src/vllm}"
MODEL_ID="${MODEL_ID:-google/gemma-4-E4B-it}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
RECREATE_ENV="${RECREATE_ENV:-0}"
CLONE_VLLM="${CLONE_VLLM:-0}"

MINICONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
if [[ ! -f "$MINICONDA_SH" ]]; then
  echo "[ERROR] Miniconda not found at: $MINICONDA_SH"
  exit 1
fi

source "$MINICONDA_SH"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[ERROR] Required command not found: $1"
    exit 1
  }
}

need_cmd sudo
need_cmd pacman
need_cmd git
need_cmd python

if [[ "$RECREATE_ENV" == "1" ]]; then
  conda deactivate >/dev/null 2>&1 || true
  conda remove -n "$ENV_NAME" --all -y || true
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.12 -y
fi

conda activate "$ENV_NAME"

# System packages required for ROCm/vLLM source build.
sudo pacman -Sy --needed \
  base-devel cmake ninja git openmpi amdsmi hipsparselt python-pip

# Core Python build tools.
pip install -U pip setuptools wheel
pip install "numpy<2" ninja cmake pybind11 packaging

# MPI ABI expected by earlier attempts and source build workflow.
conda install -c conda-forge "openmpi=4.1.*" -y

# ROCm PyTorch must be installed first and must stay ROCm, not CUDA.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2

# Persistent ROCm env vars for this conda env.
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/rocm_env.sh" <<'EOF'
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-}
export CMAKE_PREFIX_PATH=/opt/rocm:/opt/rocm/lib/cmake:${CMAKE_PREFIX_PATH:-}
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export VLLM_TARGET_DEVICE=rocm
export PYTORCH_ROCM_ARCH=gfx1100
export VLLM_LOGGING_LEVEL=DEBUG
EOF
source "$CONDA_PREFIX/etc/conda/activate.d/rocm_env.sh"

# Quick validation: torch must be ROCm, not CUDA.
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('hip:', torch.version.hip)
print('cuda:', torch.version.cuda)
print('available:', torch.cuda.is_available())
if not torch.version.hip:
    raise SystemExit('ROCm torch is not active. Aborting.')
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
PY

# AMD SMI python bindings + runtime test.
pip install amdsmi
python -c "import amdsmi; print('amdsmi ok')"

# Get vLLM source.
mkdir -p "$(dirname "$VLLM_SRC_DIR")"
if [[ ! -d "$VLLM_SRC_DIR/.git" ]]; then
  if [[ "$CLONE_VLLM" != "1" ]]; then
    echo "[INFO] vLLM repo not found at $VLLM_SRC_DIR"
    echo "[INFO] Set CLONE_VLLM=1 to clone automatically."
    exit 1
  fi
  git clone https://github.com/vllm-project/vllm.git "$VLLM_SRC_DIR"
fi

cd "$VLLM_SRC_DIR"

# Clean old build artifacts.
git clean -xfd

# Install ROCm/source build requirements.
if [[ -f requirements/rocm.txt ]]; then
  pip install -r requirements/rocm.txt
elif [[ -f requirements-rocm.txt ]]; then
  pip install -r requirements-rocm.txt
fi

# Build/install from source without build isolation so current ROCm torch is used.
pip install -e . --no-build-isolation -v

# Final import validation.
python - <<'PY'
import torch, vllm
print('torch:', torch.__version__)
print('hip:', torch.version.hip)
print('gpu:', torch.cuda.get_device_name(0))
print('vllm:', vllm.__version__)
from vllm.platforms import current_platform
print('platform:', current_platform.device_type if current_platform else 'unknown')
PY

echo
printf '[INFO] If not already authenticated with Hugging Face, run:\n  hf auth login\n\n'

# Start server.
exec vllm serve "$MODEL_ID" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
