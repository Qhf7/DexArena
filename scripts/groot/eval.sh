#!/usr/bin/env bash
# Evaluate a finetuned GR00T checkpoint through our ManiSkill evaluate_policy loop.
# Runs in the unified env (gr00t + ManiSkill in one venv); GrootAgent loads the checkpoint
# IN-PROCESS — no inference server. Set \$GROOT_PYTHON to override the interpreter.
#
# Usage: bash scripts/groot/eval.sh <checkpoint_dir> [env_id] [episodes]
#   e.g. bash scripts/groot/eval.sh runs/groot_bottlecap_ft/checkpoint-10000 BottleCapUnscrewing-gripper-demo 50
set -euo pipefail
CKPT="${1:?usage: eval.sh <checkpoint_dir> [env_id] [episodes]}"
ENV_ID="${2:-BottleCapUnscrewing-gripper-demo}"
EPISODES="${3:-50}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${GROOT_PYTHON:-$REPO_ROOT/libs/Isaac-GR00T/.venv/bin/python}"
cd "$REPO_ROOT"
exec "$PY" train/train_groot.py --evaluate --model_path "$CKPT" \
  --env_id "$ENV_ID" --control_mode pd_joint_delta_pos --robot_uids panda_wristcam \
  --exp_name "unscrew the bottle cap" \
  --num_eval_episodes "$EPISODES" --num_eval_steps 200 --action_horizon 8
