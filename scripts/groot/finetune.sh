#!/usr/bin/env bash
# Finetune GR00T on a LeRobot v2.1 VLA dataset via train/train_groot.py (dispatches torchrun).
# Runs in the unified env. Export GROOT_BACKBONE_PATH for token-free training and
# CUDA_VISIBLE_DEVICES to pick GPUs. Set \$GROOT_PYTHON to override the interpreter.
#
# Usage: bash scripts/groot/finetune.sh <dataset_dir> <env_id> [output_dir] [num_gpus]
#   <env_id> keys into utils/groot_utils/modality_configs.py for the state/action/video splits.
#   e.g. CUDA_VISIBLE_DEVICES=1,4,6,7 bash scripts/groot/finetune.sh \
#          data_train/bottlecap_gripper_vla BottleCapUnscrewing-gripper-demo \
#          runs/groot_bottlecap_ft 4
set -euo pipefail
DS="${1:?usage: finetune.sh <dataset_dir> <env_id> [output_dir] [num_gpus]}"
ENV_ID="${2:?need an env_id (key into utils/groot_utils/modality_configs.py)}"
OUT="${3:-runs/groot_ft}"
NGPU="${4:-4}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${GROOT_PYTHON:-$REPO_ROOT/libs/Isaac-GR00T/.venv/bin/python}"
cd "$REPO_ROOT"
exec "$PY" train/train_groot.py \
  --dataset_path "$DS" --env_id "$ENV_ID" \
  --output_dir "$OUT" --num_gpus "$NGPU"
