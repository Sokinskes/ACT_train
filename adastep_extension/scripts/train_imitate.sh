#!/usr/bin/env bash
# Minimal wrapper to run ACT imitation training (example from article)
set -euo pipefail
CFG_TASK=${1:-sim_transfer_cube_scripted}
CKP_DIR=${2:-../ckp}
NUM_STEPS=${3:-2000}

conda activate aloha
python3 imitate_episodes.py \
  --task_name ${CFG_TASK} \
  --ckpt_dir "${CKP_DIR}" \
  --policy_class ACT \
  --kl_weight 1 \
  --chunk_size 10 \
  --hidden_dim 512 \
  --batch_size 1 \
  --dim_feedforward 3200 \
  --lr 1e-5 \
  --seed 0 \
  --num_steps ${NUM_STEPS}
