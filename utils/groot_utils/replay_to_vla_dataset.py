"""Replay state-only demos into a VLA LeRobot dataset with rendered base + wrist views.

The teleop demos store proprio/env state + actions but **no images** (recorded in
``obs_mode="state"``). VLA policies need camera views. This script re-opens the SAME env
in ``obs_mode="rgb"``, restores each recorded ``observation.environment_state`` (drift-free:
no open-loop action replay), renders the ``base`` + ``wrist`` views, and writes a new LeRobot
dataset with:

    observation.state                    -> proprio (see --proprio)
    action                               -> the demo's actions (unchanged)
    observation.images.base_0_rgb        -> env base_camera (the "head" view)
    observation.images.left_wrist_0_rgb  -> the robot's wrist camera

Two embodiment families:

  --proprio franka_gripper   7 arm joints + gripper width (8-dim). The demos were recorded on
                             ``robot_uids="panda"`` (no wrist cam), so pass
                             ``--robot_uids panda_wristcam`` and the recorded articulation is
                             renamed to that uid when restoring (same DoF, camera is a sensor).
  --proprio controller_qpos  every actuated joint in controller order (arm 7 + hand 22 = 29 for
                             the Sharpa hand). The dex robots carry their own wrist camera, so
                             the env's registered robot is used as-is.

State restore (``--restore``):

  set_state   one ``env.reset(seed=<episode seed>)`` per episode so load-time randomisation
              (textures, lighting, layout) matches the recording, then ``set_state_dict`` per
              frame. Fast: no reconfigure per frame.
  reset       ``env.reset(seed, options={"reset_to_env_states": ...})`` per frame -- the
              slow path the gripper datasets were built with; kept for cross-checking.
``--verify_frames N`` renders N frames of the first episode through BOTH paths and reports
the pixel/state disagreement, so a dataset can be built on the fast path with evidence.

Usage (dex task; per-episode instructions and robot come from utils/dex_tasks.py):
  python utils/groot_utils/replay_to_vla_dataset.py \
    --src data/CabinetPouringOpenCabinet-Sharpa-v1 \
    --dst $DEX_OUT/datasets/cabinet_pour_door_open_rig_vla \
    --env_id CabinetPouringOpenCabinet-Sharpa-v1 --control_mode pd_joint_pos \
    --proprio controller_qpos --camera_size 256 --verify_frames 8
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RECORDING_CONTEXT_KEY = "__recording_context__"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="input state-only LeRobot dataset root")
    ap.add_argument("--dst", required=True, help="output VLA LeRobot dataset root")
    ap.add_argument("--env_id", required=True,
                    help="env to re-open for rendering; must match the demo's env")
    ap.add_argument("--control_mode", required=True,
                    help="must match what the demo was recorded with")
    ap.add_argument("--robot_uids", default=None,
                    help="override the env's robot (gripper demos: panda_wristcam to add a wrist "
                         "cam; must share the demo's DoF). Default: the env's registered robot")
    ap.add_argument("--src_robot_name", default=None,
                    help="robot articulation key in the demo env_state, remapped to --robot_uids "
                         "when they differ (default: the single articulation whose name differs)")
    ap.add_argument("--proprio", choices=("franka_gripper", "controller_qpos"), default="franka_gripper",
                    help="observation.state layout (see module docstring)")
    ap.add_argument("--restore", choices=("set_state", "reset"), default="set_state",
                    help="how each recorded frame is restored before rendering (see docstring)")
    ap.add_argument("--verify_frames", type=int, default=0,
                    help="cross-check this many frames of the first episode through both restore "
                         "paths and print the disagreement (0 = skip)")
    ap.add_argument("--goal_in_state", action="store_true",
                    help="append the 3-dim goal position to observation.state (8 -> 11). The "
                         "goal is re-randomized per episode, so without it (or a reliably "
                         "visible marker) the policy cannot know where to place the object")
    ap.add_argument("--goal_site_visual_radius", type=float, default=None,
                    help="draw the goal marker at this radius; success tolerance is unchanged")
    ap.add_argument("--head_camera_target", type=float, nargs=3, default=None,
                    help="re-aim the front camera, e.g. -0.15 -0.15 0.25 to include the goal")
    ap.add_argument("--visible_goal_site", action="store_true",
                    help="expose the goal marker to the sensor cameras. Needed when the goal is "
                         "randomized per episode and absent from the 8-dim proprio state, since "
                         "the policy otherwise has no way to know where to place the object")
    ap.add_argument("--camera_size", type=int, default=None,
                    help="override ALL sensor cameras to camera_size x camera_size; needed for envs "
                         "whose views differ in resolution (GR00T stacks views and needs equal sizes)")
    ap.add_argument("--camera_profile", choices=("none", "robodex_fixed"), default="none",
                    help="robodex_fixed: render the accepted fixed base + wrist views "
                         "(utils/camera_configs.py via utils/dex_cameras.py) instead of the task's "
                         "own cameras; evaluation must then pass the same profile")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--sim_backend", default=None,
                    help="ManiSkill sim backend; pass physx_cpu to use no GPU at all "
                         "(the render is software rasterised either way on these hosts)")
    ap.add_argument("--episodes", type=int, default=None, help="limit episodes (default: all)")
    ap.add_argument("--episode_start", type=int, default=0,
                    help="index of the first episode to replay. Combined with --episodes this "
                         "selects a slice, so several processes can render disjoint shards in "
                         "parallel and the shards be combined with lerobot.datasets.aggregate")
    ap.add_argument("--prompt", default=None,
                    help="pin every episode to this one instruction. Required unless --env_id "
                         "is a registered dex task (utils/dex_tasks.py), which otherwise "
                         "labels each episode with one of its registered instructions; for "
                         "such a task the value must be one the task registers, since a "
                         "wrong prompt silently mislabels the whole dataset")
    ap.add_argument("--instruction_index", type=int, default=None,
                    help="pin every episode to the task's instruction at this position "
                         "(utils/dex_tasks.py). Default is no pinning: episodes are labelled "
                         "round-robin over the whole list, so a 100-episode dataset carries "
                         "all of them and training sees every phrasing")
    ap.add_argument("--repo_id", default=None, help="LeRobot repo id (default: dst dir name)")
    ap.add_argument("--base_camera_field", default="observation.images.base_0_rgb")
    ap.add_argument("--wrist_camera_field", default="observation.images.left_wrist_0_rgb")
    ap.add_argument("--fps", type=int, default=None, help="override fps (default: from src)")
    ap.add_argument("--image_features", action="store_true",
                    help="store camera views as per-frame image features instead of video "
                         "(GR00T's training loader needs VIDEO features, which is the default)")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def _import_task_modules():
    import importlib
    import pkgutil

    import tasks as _task_pkg

    for _tmod in pkgutil.iter_modules(_task_pkg.__path__):
        try:
            importlib.import_module(f"task.{_tmod.name}")
        except Exception as _texc:  # a task package may have unmet deps; skip it
            print(f"[replay] skipped task.{_tmod.name}: {_texc}")


def resolve_dataset_instruction(env_id, prompt, instruction_index=None):
    """How this dataset labels its episodes -> (instruction_of(episode_index), manifest).

    For a registered dex task, each episode is labelled with one of the instructions the
    shared table registers (utils/dex_tasks.py), so a 100-episode dataset carries all of
    them and training conditions on a different phrasing from episode to episode. The
    assignment is a function of the episode index alone, so the parallel per-shard renders
    agree without coordinating.

    ``--prompt`` or ``--instruction_index`` pins every episode to one instruction instead.
    A ``--prompt`` for a registered task is checked against the table rather than trusted:
    an unregistered wording produces a dataset only its author can train and evaluate on,
    and nothing downstream would notice. Envs outside the table keep plain free text.
    """
    from utils.dex_tasks import (
        DEX_INSTRUCTIONS_VERSION,
        DEX_TASKS_BY_ENV,
        dataset_episode_instruction,
        validate_dex_instruction,
    )

    task = DEX_TASKS_BY_ENV.get(env_id)
    if task is None:
        if prompt is None:
            raise SystemExit(f"--prompt is required: {env_id} is not a registered dex task.")
        if instruction_index is not None:
            raise SystemExit(
                f"--instruction_index only applies to a registered dex task; {env_id} is not one."
            )
        return (lambda _episode_index: prompt), {
            "instruction_assignment": "fixed",
            "instructions": [prompt],
            "instructions_version": None,
        }

    if prompt is not None:
        validate_dex_instruction(env_id, prompt)
        instruction_index = task.instructions.index(prompt)
    if instruction_index is not None:
        pinned = task.instruction(instruction_index)
        print(f"[dex] every episode pinned to instruction [{instruction_index}]: {pinned!r}")
        return (lambda _episode_index: pinned), {
            "instruction_assignment": "fixed",
            "instruction_index": instruction_index,
            "instructions": [pinned],
            "instructions_version": DEX_INSTRUCTIONS_VERSION,
        }

    print(f"[dex] per-episode instructions ({len(task.instructions)} registered):")
    for index, text in enumerate(task.instructions):
        print(f"[dex]   [{index}] {text!r}")
    return (lambda episode_index: dataset_episode_instruction(env_id, episode_index)), {
        "instruction_assignment": "per_episode",
        "instructions": list(task.instructions),
        "instructions_version": DEX_INSTRUCTIONS_VERSION,
    }


def main():
    args = parse_args()
    import gymnasium as gym
    import pyarrow.parquet as pq
    import torch

    import mani_skill.envs  # noqa: F401
    # Register task envs from whatever task packages exist on the current branch (task-agnostic,
    # same mechanism as train/train_groot.py).
    _import_task_modules()
    from utils.demo_dataset import LeRobotEpisodeRecorder, unflatten_env_state_batch
    from utils.image_observation import (
        BASE_CAMERA_ALIASES,
        WRIST_CAMERA_ALIASES,
        extract_camera_rgb,
        extract_controller_qpos,
        extract_franka_proprio,
    )

    instruction_of, instruction_manifest = resolve_dataset_instruction(
        args.env_id, args.prompt, args.instruction_index
    )

    src = Path(args.src)
    dst = Path(args.dst)
    use_cpu = str(args.sim_backend or "").startswith("physx_cpu")
    device = "cpu" if use_cpu else f"cuda:{args.gpu_id}"
    repo_id = args.repo_id or dst.name

    # ---- source dataset metadata ----
    info = json.loads((src / "meta" / "info.json").read_text())
    src_fps = int(info.get("fps", 20))
    fps = args.fps or src_fps
    feats = info["features"]
    src_obs_dim = int(np.prod(feats["observation.state"]["shape"]))
    src_act_dim = int(np.prod(feats["action"]["shape"]))
    env_state_names = feats["observation.environment_state"]["names"]
    print(f"[src] {src}: obs_dim={src_obs_dim} action_dim={src_act_dim} fps={src_fps} "
          f"robot_type={info.get('robot_type')}")

    # ---- env (single, rgb) ----
    env_kwargs = dict(
        num_envs=1, obs_mode="rgb", render_mode="rgb_array",
        control_mode=args.control_mode,
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = args.robot_uids
    if args.sim_backend is not None:
        env_kwargs["sim_backend"] = args.sim_backend
    if args.visible_goal_site:
        env_kwargs["visible_goal_site"] = True
    if args.goal_site_visual_radius is not None:
        env_kwargs["goal_site_visual_radius"] = args.goal_site_visual_radius
    if args.head_camera_target is not None:
        env_kwargs["head_camera_target"] = list(args.head_camera_target)
    if args.camera_size is not None:
        env_kwargs["sensor_configs"] = dict(width=args.camera_size, height=args.camera_size)
    from utils.dex_cameras import install_camera_profile
    camera_manifest = install_camera_profile(args.env_id, args.camera_profile)
    env = gym.make(args.env_id, **env_kwargs)
    base = env.unwrapped
    obs0, _ = env.reset(seed=0)
    robot_uid = str(base.robot_uids if isinstance(base.robot_uids, str) else base.agent.uid)
    env_act_dim = int(np.prod(env.action_space.shape[-1:]))
    if env_act_dim != src_act_dim:
        raise RuntimeError(f"env action_dim {env_act_dim} != demo action_dim {src_act_dim}")
    # Which rig cameras this env actually built, and the dataset field each goes to.
    if camera_manifest is not None:
        from utils.dex_cameras import camera_fields, install_fixed_cameras

        spec = install_fixed_cameras(args.env_id)  # now filled in with the built wrist cameras
        camera_manifest.update(spec.as_dict())
        view_fields = [(name, f"observation.images.{field}", (name,))
                       for name, field in camera_fields(spec).items()]
    else:
        view_fields = [("base", args.base_camera_field, BASE_CAMERA_ALIASES),
                       ("wrist", args.wrist_camera_field, WRIST_CAMERA_ALIASES)]
    # Confirm every view is present before doing any work.
    for label, field, aliases in view_fields:
        try:
            img = extract_camera_rgb(obs0, aliases)
        except KeyError as exc:
            raise RuntimeError(
                f"{label} camera not found in obs with robot_uids={robot_uid}. {exc}"
            ) from exc
        print(f"[env] {label} -> {field}, shape={img.shape}")
    print(f"[env] robot_uids={robot_uid} action_dim={env_act_dim} restore={args.restore} "
          f"proprio={args.proprio}")

    def proprio_of(obs):
        if args.proprio == "controller_qpos":
            return extract_controller_qpos(obs, env=env)
        state = extract_franka_proprio(obs, env=env)
        if args.goal_in_state:
            goal = base.goal_site.pose.p[0].cpu().numpy().astype(np.float32)
            state = np.concatenate([state, goal])
        return state

    # The demo may key the robot articulation by another uid (gripper demos: "panda",
    # rendered on "panda_wristcam"). Same DoF -> identical state layout, so rename the key.
    scene_articulations = set(base.scene.articulations.keys())

    def restorable_state(es_flat_row):
        st = unflatten_env_state_batch(es_flat_row[None, :], env_state_names)
        context = st.pop(RECORDING_CONTEXT_KEY, None)
        arts = st.get("articulations")
        if arts:
            src_name = args.src_robot_name
            if src_name is None:
                unknown = [name for name in arts if name not in scene_articulations]
                if len(unknown) == 1 and robot_uid not in arts:
                    src_name = unknown[0]
            if src_name is not None and src_name in arts and src_name != robot_uid:
                arts[robot_uid] = arts.pop(src_name)
        es_torch = {cat: {n: torch.as_tensor(v, device=device) for n, v in d.items()}
                    for cat, d in st.items()}
        return es_torch, context

    def apply_context(context):
        if not context or "elapsed_steps" not in context:
            return
        target = getattr(base, "_elapsed_steps", None)
        if target is None:
            return
        restored = torch.as_tensor(np.asarray(context["elapsed_steps"]).reshape(-1), device=base.device)
        if torch.is_tensor(target):
            target.copy_(restored.to(dtype=target.dtype).reshape_as(target))
        else:
            base._elapsed_steps = restored

    def restore_reset(es_flat_row, seed):
        es_torch, context = restorable_state(es_flat_row)
        obs, _ = env.reset(seed=seed, options={"reset_to_env_states": {"env_states": es_torch}})
        apply_context(context)
        return obs

    def restore_set_state(es_flat_row):
        es_torch, context = restorable_state(es_flat_row)
        base.set_state_dict(es_torch)
        apply_context(context)
        return base.get_obs()

    # ---- load source frames grouped by episode ----
    # Read ALL v3.0 data files (a large dataset spans multiple file-*.parquet), not just the
    # first -- otherwise episodes beyond the first file are silently dropped.
    data_paths = sorted((src / "data").rglob("*.parquet"))
    table = pq.read_table(data_paths)
    cols = table.to_pydict()
    ei = np.asarray(cols["episode_index"], dtype=np.int64)
    fi = np.asarray(cols["frame_index"], dtype=np.int64)
    order = np.lexsort((fi, ei))
    ei_s = ei[order]
    actions_all = np.asarray([cols["action"][i] for i in order], dtype=np.float32)
    es_all = np.asarray([cols["observation.environment_state"][i] for i in order], dtype=np.float32)
    seeds_all = (
        np.asarray([cols["episode.seed"][i] for i in order])
        if "episode.seed" in cols else None
    )

    episode_ids = np.unique(ei_s).tolist()
    episode_ids = episode_ids[args.episode_start :]
    if args.episodes is not None:
        episode_ids = episode_ids[: args.episodes]

    def episode_seed(idxs):
        if seeds_all is None:
            return None
        value = seeds_all[idxs[0]]
        return None if value is None else int(value)

    # ---- optional cross-check of the two restore paths ----
    if args.verify_frames > 0 and episode_ids:
        idxs = np.nonzero(ei_s == episode_ids[0])[0]
        picks = np.unique(np.linspace(0, len(idxs) - 1, num=min(args.verify_frames, len(idxs))).astype(int))
        seed = episode_seed(idxs)
        worst_pix, worst_state = 0.0, 0.0
        env.reset(seed=seed)
        for k in picks:
            t = idxs[k]
            obs_a = restore_set_state(es_all[t])
            a_base, a_wrist = extract_camera_rgb(obs_a, BASE_CAMERA_ALIASES), extract_camera_rgb(obs_a, WRIST_CAMERA_ALIASES)
            a_state = proprio_of(obs_a)
            obs_b = restore_reset(es_all[t], seed)
            b_base, b_wrist = extract_camera_rgb(obs_b, BASE_CAMERA_ALIASES), extract_camera_rgb(obs_b, WRIST_CAMERA_ALIASES)
            b_state = proprio_of(obs_b)
            pix = max(np.abs(a_base.astype(int) - b_base.astype(int)).max(),
                      np.abs(a_wrist.astype(int) - b_wrist.astype(int)).max())
            frac = float(np.mean(np.abs(a_base.astype(int) - b_base.astype(int)) > 8))
            st = float(np.abs(a_state - b_state).max())
            worst_pix, worst_state = max(worst_pix, pix), max(worst_state, st)
            print(f"[verify] ep {episode_ids[0]} frame {k}: max|pix| set_state vs reset = {pix} "
                  f"(frac>8: {frac:.4f}), max|state| = {st:.2e}")
        print(f"[verify] worst: pixels {worst_pix}, state {worst_state:.2e} "
              f"({'OK' if worst_state < 1e-4 else 'STATE MISMATCH'})")
        env.reset(seed=0)

    recorder = LeRobotEpisodeRecorder(
        dataset_root=dst, repo_id=repo_id, fps=fps, robot_type=robot_uid,
        # Per-episode labels override this; it only names the dataset's default task.
        task=instruction_manifest["instructions"][0],
        include_env_state=False, include_success=False,
        # include_seed=True: the manifest writer reads an `episode.seed` column
        # unconditionally; GR00T ignores the extra column anyway.
        include_seed=True, overwrite=args.overwrite,
        # GR00T's training loader maps modality "video" keys from VIDEO (mp4) features;
        # image features get dropped (video.<key> not found). Default to video.
        use_videos=not args.image_features,
    )

    total_frames = 0
    t_start = time.time()
    for e in episode_ids:
        idxs = np.nonzero(ei_s == e)[0]
        n = len(idxs)
        seed = episode_seed(idxs)
        states = []
        views = {field: [] for _label, field, _aliases in view_fields}
        if args.restore == "set_state":
            # One reconfigure per episode with the recorded seed, so load-time
            # randomisation (textures, lighting, layout) matches the recording.
            env.reset(seed=seed)
        t_ep = time.time()
        for t in idxs:
            obs = restore_set_state(es_all[t]) if args.restore == "set_state" else restore_reset(es_all[t], seed)
            states.append(proprio_of(obs))
            for _label, field, aliases in view_fields:
                views[field].append(extract_camera_rgb(obs, aliases))
        states = np.asarray(states, dtype=np.float32)          # (n, D)
        views = {f: np.asarray(v, dtype=np.uint8) for f, v in views.items()}  # (n, H, W, 3)
        actions = actions_all[idxs][: n - 1]                   # (n-1, A); recorder wants obs=actions+1
        recorder.add_episode(
            obs=states,
            actions=actions,
            rewards=np.zeros((n - 1,), dtype=np.float32),
            terminated=np.zeros((n - 1,), dtype=np.bool_),
            truncated=np.zeros((n - 1,), dtype=np.bool_),
            # The source episode id keys the label, not the output index, so a shard can
            # compute its own labels and still agree with the aggregated dataset.
            task=instruction_of(int(e)),
            episode_seed=seed,
            image_sequences=views,
        )
        total_frames += n
        dt = time.time() - t_ep
        shapes = " ".join(f"{f.rsplit('.', 1)[-1]}{v.shape[1:]}" for f, v in views.items())
        print(f"[ep {e}] {n} frames in {dt:.1f}s ({n / max(dt, 1e-6):.1f} fps) -> {shapes} "
              f"state{states.shape[1:]} seed={seed}", flush=True)

    summary = recorder.finalize()
    env.close()
    # Record how the views were produced next to the dataset (survives shard aggregation only
    # if copied).
    render_manifest = {
        "env_id": args.env_id, "control_mode": args.control_mode, "restore": args.restore,
        "proprio": args.proprio, "camera_size": args.camera_size,
        "camera_profile": args.camera_profile, "camera": camera_manifest,
        "src": str(src), "episode_start": args.episode_start, "episodes": args.episodes,
        # Which instruction this dataset is labelled with. Training reads the string from the
        # dataset itself; this says where in the shared table it came from, so a rendered
        # dataset can be traced back to the table version that produced it.
        **instruction_manifest,
    }
    (Path(dst) / "meta" / "robodex_render.json").write_text(json.dumps(render_manifest, indent=2) + "\n")
    print(f"[done] wrote {summary.total_episodes} episodes / {total_frames} frames -> {dst} "
          f"in {(time.time() - t_start) / 60:.1f} min")
    print(f"       state_dim={summary.obs_dim} action_dim={summary.action_dim} "
          f"images={[f for _l, f, _a in view_fields]}")


if __name__ == "__main__":
    main()
