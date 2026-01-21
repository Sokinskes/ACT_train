#!/usr/bin/env bash
# Wrapper: evaluate / visualize episodes
set -euo pipefail
TASK=${1:-sim_transfer_cube_scripted}
CKP_DIR=${2:-../ckp}

conda activate aloha
# evaluation (uses latest checkpoint by default)
python3 imitate_episodes.py --eval --task_name ${TASK} --ckpt_dir "${CKP_DIR}" --num_steps 20

# visualize a saved episode (example)
python3 visualize_episodes.py --dataset_dir "../datas/gen_data" --episode_idx 0
