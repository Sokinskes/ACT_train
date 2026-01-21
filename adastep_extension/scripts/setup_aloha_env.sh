#!/usr/bin/env bash
# One-shot environment setup for ACT-Plus-Plus / Mobile ALOHA reproduction (safe, non-destructive)
set -euo pipefail
ENV_NAME=aloha
PY_VER=${1:-3.8.10}  # pass 3.10 for newer GPUs
echo "Creating conda env: $ENV_NAME (python $PY_VER)"
conda create -y -n "$ENV_NAME" python="$PY_VER"
echo "Activate the environment and install Python deps:"
echo "  conda activate $ENV_NAME && pip install -r scripts/requirements_aloha.txt"

cat <<'EOF'
GPU notes:
- If you have an NVIDIA 50-series GPU you may need to install a matching PyTorch wheel (article suggests a nightly install for CUDA 12.8):
  pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
- Mujoco: install mujoco binaries per mujoco docs and set MUJOCO_PY_MUJOCO_PATH / LD_LIBRARY_PATH as needed.
EOF

echo "Done. Next: activate the env and run the conversion / training scripts in scripts/README or docs/REPRODUCE_ALOHA.md."