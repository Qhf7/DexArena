#!/usr/bin/env python3
"""Plan/run the complete 30-task x 3-background matrix in isolated processes.

`plan` is simulator-free. `preview` and `smoke` verify native tabletop support,
image observations, stepping and repeatable resets via GR00T's env factory.
No policy weights are loaded and no policy success rates are measured.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.dex_scene_eval import BACKGROUND_IDS, background_manifest
from utils.dex_scene_eval import registered_task_settings


def subtasks():
    # Read the authoritative literal without importing tasks/__init__ (SAPIEN).
    tree = ast.parse((ROOT / "tasks/config/collection_targets.py").read_text())
    names = next(ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.AnnAssign) and n.target.id == "SUBTASKS")
    return [f"{name}-Sharpa-v1" for name in names]


def matrix():
    return [{"env_id": env, "scene_background": scene,
             "key": f"{env}__{scene}"}
            for env in subtasks() for scene in BACKGROUND_IDS]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def smoke_one(row, output, seed, steps):
    # Exercise the same GR00T factory and native-table checks as the preview.
    # Native support geometry intentionally differs from the old generic table.
    preview_one(row, output, seed, steps)
    record = json.loads((output / "preview.json").read_text())
    write_json(output / "smoke.json", record)


def preview_one(row, output, seed, steps):
    """Use GR00T's real env factory, without loading GR00T or measuring success."""
    import numpy as np
    from PIL import Image
    from mani_skill.utils import sapien_utils
    from train.train_configs.groot_config import Args
    from train.train_groot import make_eval_env
    def state_arrays(value, prefix=""):
        if isinstance(value, dict):
            return {p:a for k,v in value.items() for p,a in state_arrays(v,prefix+"/"+k).items()}
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return {prefix:np.asarray(value).copy()}

    spec = registered_task_settings(row["env_id"])
    args = Args(env_id=row["env_id"], robot_uids=spec["robot_uids"],
                control_mode=spec["control_mode"], sim_backend="physx_cpu",
                camera_size=256, scene_background=row["scene_background"])
    args.env_max_episode_steps = spec["max_episode_steps"]
    env = make_eval_env(args, num_envs=1, reconfiguration_freq=1)
    try:
        obs, _ = env.reset(seed=seed)
        initial_state = state_arrays(env.unwrapped.get_state_dict())
        for camera, data in obs["sensor_data"].items():
            rgb = data["rgb"][0].cpu().numpy()
            Image.fromarray(rgb).save(output / f"{camera}.png")
        u = env.unwrapped
        # Diagnostic views never enter the policy observation camera set.
        for name, eye, target in (
            ("overview", [-1.25, 0, 1.5], [.25, 0, .15]),
            ("clearance_side", [-.3, -2.8, .4], [-.3, 0, 0]),
        ):
            camera = u.scene.sub_scenes[0].add_camera("placement_" + name, 800, 600, 1.0, .01, 100)
            camera.set_local_pose(sapien_utils.look_at(eye, target).sp)
            u.scene.update_render()
            camera.take_picture()
            rgb = np.asarray(camera.get_picture("Color"))[..., :3]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb * (255 if rgb.max() <= 1.001 else 1), 0, 255).astype(np.uint8)
            Image.fromarray(rgb).save(output / (name + ".png"))
        from utils.dex_scene_checks import inspect_native_scene
        checks = []
        for reset_index in range(2):
            if reset_index:
                obs, _ = env.reset(seed=seed+reset_index)
            check = inspect_native_scene(env)
            checks.append(check)
            write_json(output / "placement.json", checks)
            if check["errors"]:
                raise AssertionError("; ".join(check["errors"]))
            for camera_name, data in obs["sensor_data"].items():
                pixels = data["rgb"][0].cpu().numpy()
                if not np.isfinite(pixels).all() or pixels.std() < 1:
                    raise AssertionError(f"Blank/invalid camera {camera_name}")
            action = u.agent.robot.qpos.clone()
            for _ in range(steps):
                obs, reward, terminated, truncated, info = env.step(action)
                if not np.isfinite(u.agent.robot.qpos.cpu().numpy()).all():
                    raise AssertionError("Non-finite robot state")
            after = inspect_native_scene(env)
            # Keep the post-step record as evidence; contacts on native surfaces
            # are intentionally not required to match the old work table.
            for item in check["table_supported_objects"]:
                box = after["task_aabbs"].get(item["entity"])
                if box is None or box[0][2] < -.04:
                    raise AssertionError(f"Object lost native tabletop support: {item['entity']}")
            if after["errors"]:
                raise AssertionError("; ".join(after["errors"]))
            check["after_hold"] = after
        env.reset(seed=seed)
        final_state = state_arrays(u.get_state_dict())
        if final_state.keys() != initial_state.keys():
            raise AssertionError("Reset changed task state registry")
        for key in initial_state:
            if not key.startswith(("/actors/", "/articulations/")):
                continue  # Task-specific history caches are not physical layout.
            np.testing.assert_allclose(final_state[key], initial_state[key], atol=1e-6, rtol=0,
                                       err_msg=f"Non-repeatable reset: {key}")
        repeated = inspect_native_scene(env)
        if repeated["surface_aabb"] != checks[0]["surface_aabb"]:
            raise AssertionError("Native table placement changed between resets")
        write_json(output / "preview.json", {**row, "status": "passed", "seed": seed,
                   "resets": 3, "steps_per_reset": steps,
                   "source": "train.train_groot.make_eval_env", "policy_loaded": False,
                   "native_surface_checks": checks, "background": u.eval_background_manifest})

    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "smoke", "smoke-one", "preview", "preview-one"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--index", type=int)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--resume", action="store_true", help="skip passed combinations with matching scene source hashes")
    p.add_argument("--seed", type=int, default=10001)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--timeout", type=int, default=900)
    args = p.parse_args()
    rows = matrix()
    if args.index is not None and not 0 <= args.index < len(rows):
        p.error("--index must be in 0..89")
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        p.error("--shard must be in 0..shards-1")
    if args.steps < 1 or args.timeout < 1 or args.seed < 0:
        p.error("steps/timeout must be positive and seed must be nonnegative")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "plan":
        write_json(args.out / "matrix.json", {"tasks": 30, "backgrounds": 3, "combinations": rows,
                   "scenes": [background_manifest(s) for s in BACKGROUND_IDS]})
        print(f"Wrote {len(rows)} combinations to {args.out / 'matrix.json'}")
        return 0
    if args.mode in ("smoke-one", "preview-one"):
        if args.index is None:
            p.error("single-combination modes require --index")
        row = rows[args.index]
        output = args.out / row["key"]
        output.mkdir(parents=True, exist_ok=True)
        try:
            (smoke_one if args.mode == "smoke-one" else preview_one)(row, output, args.seed, args.steps)
        except Exception:
            write_json(output / (args.mode.removesuffix("-one")+".json"),
                       {**row, "status": "failed", "error": traceback.format_exc()})
            raise
        return 0
    selected = list(enumerate(rows))[args.shard::args.shards] if args.index is None else [(args.index, rows[args.index])]
    results = []
    suffix = f"_{args.index}" if args.index is not None else f"_shard{args.shard}"
    status_path = args.out / f"{args.mode}_status{suffix}.json"
    for index, row in selected:
        output = args.out / row["key"]
        record_path = output / f"{args.mode}.json"
        if args.resume and record_path.is_file():
            previous = json.loads(record_path.read_text())
            expected = background_manifest(row["scene_background"])["source_sha256"]
            if (previous.get("status") == "passed" and previous.get("seed") == args.seed and
                    previous.get("background", {}).get("source_sha256") == expected):
                results.append({**row, "returncode": 0, "resumed": True})
                continue
        command = [sys.executable, str(Path(__file__).resolve()), args.mode + "-one", "--index", str(index),
                   "--out", str(args.out), "--seed", str(args.seed), "--steps", str(args.steps)]
        log = args.out / (row["key"] + ".log")
        print(f"[{index+1}/90] {row['key']}", flush=True)
        with log.open("w") as stream:
            try:
                process_env = {**os.environ, "GROOT_MODALITY_ENV": row["env_id"], "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
                # Task asset loaders write caches under gettempdir(); isolate
                # each child so parallel workers never share or delete a cache.
                with tempfile.TemporaryDirectory(prefix="dex-scene-preview-") as cache_dir:
                    process_env["TMPDIR"] = cache_dir
                    result = subprocess.run(command, cwd=ROOT, env=process_env, stdout=stream, stderr=subprocess.STDOUT,
                                            timeout=args.timeout)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
        results.append({**row, "returncode": code, "log": str(log)})
        write_json(status_path, results)
    write_json(status_path, results)
    return int(any(r["returncode"] != 0 for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
