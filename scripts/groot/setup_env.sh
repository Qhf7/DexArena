#!/usr/bin/env bash
# Build the single UNIFIED GR00T + ManiSkill env on a GPU host (e.g. server5).
#
# GR00T pins its deps via libs/Isaac-GR00T/uv.lock (vendored submodule at a fixed commit), so
# `uv sync` is the source of truth. Those pins (torch 2.7.1, transformers 4.57.3, gymnasium
# 1.2.2, ...) also satisfy ManiSkill and lerobot, so gr00t + mani_skill + lerobot live in ONE
# venv (libs/Isaac-GR00T/.venv) — no socket bridge, no separate ManiSkill env needed.
#
# Usage:  bash scripts/groot/setup_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GR00T_DIR="$REPO_ROOT/libs/Isaac-GR00T"
VENV_PY="$GR00T_DIR/.venv/bin/python"

# --- uv (any recent uv works) ---
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "[setup] uv: $(uv --version)"

# --- vendored submodule at its pinned commit ---
echo "[setup] init Isaac-GR00T submodule ..."
git -C "$REPO_ROOT" submodule update --init libs/Isaac-GR00T
echo "[setup] GR00T commit: $(git -C "$GR00T_DIR" rev-parse --short HEAD)  (expected ab88b50)"

# --- let the hardcoded backbone name be overridden by $GROOT_BACKBONE_PATH (token-free) ---
LF="$GR00T_DIR/gr00t/experiment/launch_finetune.py"
if ! grep -q GROOT_BACKBONE_PATH "$LF"; then
  sed -i 's|config.model.model_name = "nvidia/Cosmos-Reason2-2B"|config.model.model_name = os.environ.get("GROOT_BACKBONE_PATH") or "nvidia/Cosmos-Reason2-2B"|' "$LF"
  echo "[setup] patched launch_finetune to honor \$GROOT_BACKBONE_PATH"
fi

# --- 1) GR00T deps (uv sync) ---
if "$VENV_PY" -c "import gr00t" >/dev/null 2>&1; then
  echo "[setup] GR00T deps already present — skipping uv sync."
else
  echo "[setup] building GR00T env (uv sync --all-extras) ..."
  ( cd "$GR00T_DIR" && uv sync --all-extras )
fi

# --- 2) add ManiSkill + lerobot + repo logging deps into the SAME venv (makes it unified) ---
if "$VENV_PY" -c "import mani_skill, lerobot, swanlab, tensorboard" >/dev/null 2>&1; then
  echo "[setup] ManiSkill/lerobot/logging already present — skipping."
else
  echo "[setup] adding mani_skill + lerobot + swanlab + tensorboard ..."
  uv pip install --python "$VENV_PY" "mani_skill==3.0.1" "lerobot==0.4.0" swanlab tensorboard
fi

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  "$VENV_PY" -c "import gr00t, mani_skill, lerobot, torch; print('unified env ok | gr00t + mani_skill', mani_skill.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

echo "[setup] DONE. Unified env: $GR00T_DIR/.venv"
echo "[setup]   export GROOT_PYTHON=$VENV_PY   # the interpreter for the commands in docs/GROOT_DEXARENA.md"
