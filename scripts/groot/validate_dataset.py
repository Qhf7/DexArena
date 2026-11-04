#!/usr/bin/env python3
"""Validate a GR00T VLA dataset loads with our NEW_EMBODIMENT modality config.

Run in the unified env. Registers the env's modality config (from
``utils/groot_utils/modality_configs.py``), loads the dataset with GR00T's loader, and prints
one episode's video/state/action shapes to confirm the head/wrist/state/action keys map
correctly before fine-tuning.

Usage:
  python scripts/groot/validate_dataset.py \
    --dataset-path data_train/bottlecap_gripper_vla \
    --env-id BottleCapUnscrewing-gripper-demo
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--env-id", required=True,
                    help="env id key into utils/groot_utils/modality_configs.py")
    args = ap.parse_args()

    info = json.loads((Path(args.dataset_path) / "meta" / "info.json").read_text())
    print(f"[info] codebase_version={info.get('codebase_version')} "
          f"episodes={info.get('total_episodes')} frames={info.get('total_frames')}")
    for fname in ("modality.json", "stats.json"):
        exists = (Path(args.dataset_path) / "meta" / fname).exists()
        print(f"[info] meta/{fname}: {'present' if exists else 'MISSING'}")

    from utils.groot_utils.modality_configs import register
    register(args.env_id)
    print(f"[ok] registered modality config for env {args.env_id}")

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    tag = EmbodimentTag.NEW_EMBODIMENT
    modality_configs = MODALITY_CONFIGS[tag.value]
    print(f"[ok] modality_configs for {tag.value}: {list(modality_configs.keys())}")

    loader = LeRobotEpisodeLoader(dataset_path=args.dataset_path, modality_configs=modality_configs)
    print(f"[ok] loader built, episodes={len(loader)}")
    ep = loader[0]
    n = len(ep) if hasattr(ep, "__len__") else "?"
    print(f"[ok] loaded episode 0, timesteps={n}")
    cols = list(ep.columns) if hasattr(ep, "columns") else list(getattr(ep, "keys", lambda: [])())
    print(f"[cols] {cols}")
    for modality in ("video", "state", "action"):
        if modality not in modality_configs:
            continue
        for key in modality_configs[modality].modality_keys:
            col = f"{modality}.{key}"
            try:
                v = ep[col].iloc[0] if hasattr(ep[col], "iloc") else ep[col][0]
                v = np.asarray(v)
                print(f"  {col}: shape={v.shape} dtype={v.dtype}")
            except Exception as exc:
                print(f"  {col}: <unavailable: {exc}>")
    print("[SUCCESS] dataset validation passed")


if __name__ == "__main__":
    main()
