"""Convert rendered demonstrations into the multi-camera NumPy layout that
train/train_ACT.py and train/train_DP.py read.

Input: the HDF5 file written by ``data_gen/render_pixel_demos.py`` (one group
per episode with ``action``, ``proprio`` and one ``cam_<i>`` image array per
camera of the preset; ``meta`` records which sensor each ``cam_<i>`` is).

Output directory (``act_multicamera_numpy_v1``):

    dataset_info.json   cameras (name, output_file, output_shape) and provenance
    state.npy           (N, state_dim) float32, the robot joint positions
    action.npy          (N, action_dim) float32, absolute joint targets
    images/<name>.npy   (N, H, W, 3) uint8, one array per camera and history offset
    episodes.csv        episode_index, start, end_exclusive, success
    episode_index.npy, frame_index.npy

The image policies look at every camera at several history offsets (frames
t, t-2 and t-4 by default). Each offset is materialised as its own array named
``<sensor>_t0``, ``<sensor>_tm2``, ``<sensor>_tm4``; before the offset is
reachable the first frame of the episode is repeated, which is what the
evaluation loop does at the start of an episode.

Example (OpenCabinet, 256x256, the released checkpoints' cameras):

    python -m data_gen.render_pixel_demos \\
        --demo-path data/sharpa/CabinetPouringOpenCabinet-Sharpa-v1 \\
        --env-id CabinetPouringOpenCabinet-Sharpa-v1 \\
        --robot-uids panda_sharpa_hand_left_angled_mobile_base --control-mode pd_joint_pos \\
        --camera-preset cabinet_fixed --image-size 256 --max-episode-steps 600 \\
        --out data/pixels/CabinetPouringOpenCabinet-Sharpa-v1_256.h5
    python train/prepare_image_dataset.py \\
        --pixels data/pixels/CabinetPouringOpenCabinet-Sharpa-v1_256.h5 \\
        --out data/act/CabinetPouringOpenCabinet-Sharpa-v1

The script prints the ``--camera-names``, ``--camera-map`` and
``--camera-history-offsets`` flags that match the written arrays.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pixels", required=True, help="HDF5 from data_gen/render_pixel_demos.py")
    parser.add_argument("--out", required=True, help="output dataset directory")
    parser.add_argument(
        "--history-offsets",
        type=int,
        nargs="+",
        default=(0, 2, 4),
        help="history offsets in control steps, one image array per camera and offset",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=None,
        help="leading proprio dims to keep as the state (default: the action dim, i.e. the joint positions)",
    )
    parser.add_argument("--episodes", type=int, default=None, help="keep only the first N episodes")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def history_name(sensor: str, offset: int) -> str:
    return f"{sensor}_t0" if offset == 0 else f"{sensor}_tm{offset}"


def main(argv=None) -> int:
    args = parse_args(argv)
    offsets = tuple(int(value) for value in args.history_offsets)
    if any(value < 0 for value in offsets) or len(set(offsets)) != len(offsets):
        raise ValueError("history offsets must be distinct and non-negative")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{out} is not empty; pass --overwrite")
    (out / "images").mkdir(parents=True, exist_ok=True)

    with h5py.File(args.pixels, "r") as source:
        meta = dict(source["meta"].attrs)
        camera_config = ast.literal_eval(str(meta["camera_config"]))
        sensors = [(spec["obs_key"], spec["sensor_name"]) for spec in camera_config["cameras"]]
        keys = sorted(source["episodes"].keys(), key=int)
        if args.episodes is not None:
            keys = keys[: args.episodes]
        if not keys:
            raise ValueError("no episodes in the input file")

        lengths = [int(source[f"episodes/{key}/action"].shape[0]) for key in keys]
        total = int(sum(lengths))
        action_dim = int(source[f"episodes/{keys[0]}/action"].shape[1])
        proprio_dim = int(source[f"episodes/{keys[0]}/proprio"].shape[1])
        state_dim = int(args.state_dim) if args.state_dim is not None else action_dim
        if state_dim > proprio_dim:
            raise ValueError(f"state dim {state_dim} exceeds the proprio width {proprio_dim}")
        first_image = source[f"episodes/{keys[0]}/{sensors[0][0]}"]
        height, width = int(first_image.shape[1]), int(first_image.shape[2])

        state = np.lib.format.open_memmap(out / "state.npy", mode="w+", dtype=np.float32, shape=(total, state_dim))
        action = np.lib.format.open_memmap(out / "action.npy", mode="w+", dtype=np.float32, shape=(total, action_dim))
        images = {}
        for obs_key, sensor in sensors:
            for offset in offsets:
                name = history_name(sensor, offset)
                images[name] = np.lib.format.open_memmap(
                    out / "images" / f"{name}.npy", mode="w+", dtype=np.uint8, shape=(total, height, width, 3)
                )
        episode_index = np.zeros(total, dtype=np.int64)
        frame_index = np.zeros(total, dtype=np.int64)
        rows = []
        start = 0
        for episode_id, (key, length) in enumerate(zip(keys, lengths)):
            group = source[f"episodes/{key}"]
            end = start + length
            proprio = np.asarray(group["proprio"][:length], dtype=np.float32)
            state[start:end] = proprio[:, :state_dim]
            action[start:end] = np.asarray(group["action"][:length], dtype=np.float32)
            episode_index[start:end] = episode_id
            frame_index[start:end] = np.arange(length)
            for obs_key, sensor in sensors:
                frames = np.asarray(group[obs_key][:length])
                if frames.shape[1:] != (height, width, 3) or frames.dtype != np.uint8:
                    raise ValueError(f"{key}/{obs_key}: unexpected image array {frames.shape} {frames.dtype}")
                for offset in offsets:
                    source_frames = np.maximum(np.arange(length) - offset, 0)
                    images[history_name(sensor, offset)][start:end] = frames[source_frames]
            rows.append(
                dict(
                    episode_index=episode_id,
                    start=start,
                    end_exclusive=end,
                    success=bool(group.attrs.get("success", True)),
                    source_episode=key,
                )
            )
            print(f"[episode] {episode_id + 1}/{len(keys)} frames={length}", flush=True)
            start = end

        for array in [state, action, *images.values()]:
            array.flush()
        np.save(out / "episode_index.npy", episode_index)
        np.save(out / "frame_index.npy", frame_index)
        pd.DataFrame(rows).to_csv(out / "episodes.csv", index=False)

        camera_entries = [
            dict(
                name=history_name(sensor, offset),
                output_file=f"images/{history_name(sensor, offset)}.npy",
                output_shape=[height, width, 3],
                source_sensor=sensor,
                history_offset_frames=offset,
            )
            for _, sensor in sensors
            for offset in offsets
        ]
        info = dict(
            format="act_multicamera_numpy_v1",
            num_frames=total,
            num_episodes=len(keys),
            state_dim=state_dim,
            action_dim=action_dim,
            state="the first state_dim entries of the rendered proprio vector (joint positions)",
            cameras=camera_entries,
            files=dict(
                state="state.npy",
                action="action.npy",
                episodes="episodes.csv",
                episode_index="episode_index.npy",
                frame_index="frame_index.npy",
            ),
            source=dict(
                pixels=str(Path(args.pixels).resolve()),
                env_id=str(meta.get("env_id")),
                robot_uids=str(meta.get("robot_uids")),
                control_mode=str(meta.get("control_mode")),
                camera_config_hash=str(meta.get("camera_config_hash")),
                image_size=[height, width],
            ),
        )
        (out / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")

    names = [entry["name"] for entry in camera_entries]
    print(f"[done] {total} frames, {len(keys)} episodes, state {state_dim}, action {action_dim}, cameras {len(names)} -> {out}")
    print("training / evaluation flags for these arrays:")
    print("  --camera-names " + " ".join(names))
    print("  --camera-map " + " ".join(f"{e['name']}={e['source_sensor']}" for e in camera_entries))
    print("  --camera-history-offsets " + " ".join(f"{e['name']}={e['history_offset_frames']}" for e in camera_entries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
