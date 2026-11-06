"""Version-2 long-horizon evaluation, family half: run one FAMILY checkpoint in one SUBTASK env.

Version 2 ("per-subtask AND") scores a long task per seed: for seed s, each subtask of the family is
evaluated on its own, in its own subtask environment, with that seed; the long task passes seed
s only if every subtask passed it. For the specialist checkpoints those per-subtask runs already
exist (evals/groot/<run>/checkpoint-12000/episodes.jsonl, seeds 10001-10050). This script
produces the same thing for a family checkpoint: it loads the family model, sends it one of
the subtask's registered instructions per episode (--instruction-sampling registered, the
default; `canonical` sends the subtask's canonical one every time), and rolls out 50 episodes
in the subtask environment.

    python scripts/dex/eval_family_on_subtask.py --family drawer --subtask DrawerPlugPickCharger-Sharpa-v1 \\
        --family-ckpt .../groot_drawer_family_rig/checkpoint-12000 --episodes 50 --out ...

Layouts: nine families are 29-joint single-arm policies and run in the subtask env as is. The
package family is a 58-joint two-view policy (utils/groot_utils/build_package_family_dataset.py):
in the dual-arm OpenPackage env it runs as is; in the single-arm ExtractAndPlaceApple env its
29-joint proprio is padded into the 58-joint layout with the dataset's left-arm filler and only
its right-arm slots are executed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dex_long_tasks import DUAL_SLOTS, resolve_long_task  # noqa: E402
from utils.dex_tasks import draw_instruction, resolve_dex_task  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True)
    ap.add_argument("--subtask", required=True)
    ap.add_argument("--family-ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=10001)
    ap.add_argument("--action-horizon", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--instruction-sampling", choices=("registered", "canonical"), default="registered",
                    help="registered: one of the subtask's instructions per episode, from the episode seed "
                         "(utils/dex_tasks.draw_instruction); canonical: the subtask's canonical one every time")
    args = ap.parse_args()

    import torch
    from train.train_groot import make_eval_env, _success_from_info
    from algo.groot import build_dex_agent

    long_task = resolve_long_task(args.family)
    stage = next(s for s in long_task.stages if s.subtask == args.subtask)
    sub = resolve_dex_task(args.subtask)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from mani_skill.utils.registration import REGISTERED_ENVS
    import tasks as _task_pkg  # noqa: F401
    horizon = int(REGISTERED_ENVS[args.subtask].max_episode_steps)

    env_args = SimpleNamespace(
        env_id=args.subtask, robot_uids=sub.robot_uids, control_mode="pd_joint_pos",
        camera_size=256, camera_profile="robodex_fixed", sim_backend="physx_cpu",
        render_backend=None, gpu_id=None, reward_mode=None, eval_env_extra=None,
        env_max_episode_steps=horizon, num_eval_envs=1, eval_partial_reset=False, scene_background=None,
    )
    env = make_eval_env(env_args, num_envs=1, reconfiguration_freq=1,
                        record_dir=None if args.no_video else str(out / "videos"), max_steps_per_video=horizon)
    import gymnasium as gym
    assert isinstance(env.single_action_space, gym.spaces.Box)
    env_dim = int(np.prod(env.single_action_space.shape))
    low, high = env.single_action_space.low, env.single_action_space.high

    dual_family = long_task.family == "package"
    from utils.image_observation import extract_controller_qpos
    if dual_family:
        os.environ["GROOT_VIDEO_KEYS"] = "head:base_0_rgb,right_wrist:right_wrist_0_rgb"
        modality_env = "PackageOpeningOpenPackage-Sharpa-v1"
        policy_dim = 58
    else:
        modality_env = args.subtask
        policy_dim = 29
    # The agent's arm only matters for the wrist alias / proprio slice inside the LONG dual env;
    # in a subtask env the wrist camera is the env's own, so use the stage's arm for the alias
    # and override the proprio function below.
    agent = build_dex_agent(args.family_ckpt, modality_env, stage.arm if not dual_family else "dual",
                        env, device, args.action_horizon, f"v2_family_{long_task.family}", policy_dim)
    pad_filler = None
    if dual_family and env_dim == 29:
        # The left-arm filler the package family dataset was built with; identical in every
        # build of that dataset, so whichever copy exists under $DEX_OUT serves.
        datasets = Path(os.environ.get("DEX_OUT", "dex_out")) / "datasets"
        manifest_path = next((datasets / name / "meta" / "robodex_family.json"
                              for name in ("package_family_rig_v4_vla", "package_family_rig_vla")
                              if (datasets / name / "meta" / "robodex_family.json").exists()), None)
        if manifest_path is None:
            raise SystemExit(f"no package family dataset manifest under {datasets}")
        fam_manifest = json.loads(manifest_path.read_text())
        pad_filler = np.zeros(58, dtype=np.float32)
        pad_filler[0:7] = fam_manifest["left_arm_filler"]; pad_filler[14:36] = fam_manifest["left_hand_filler"]
        r = DUAL_SLOTS["right"]

        def state_fn(obs, _env=env):
            q = np.asarray(extract_controller_qpos(obs, env=_env), dtype=np.float32).reshape(-1)
            s = pad_filler.copy(); s[r["arm"][0]:r["arm"][1]] = q[:7]; s[r["hand"][0]:r["hand"][1]] = q[7:]
            return s
        agent._state_fn = state_fn
        from algo.groot import DEFAULT_VIDEO_KEY_ALIASES, WRIST_CAMERA_ALIASES
        aliases = dict(DEFAULT_VIDEO_KEY_ALIASES); aliases["right_wrist"] = WRIST_CAMERA_ALIASES
        agent._video_key_aliases = aliases
    else:
        agent._state_fn = lambda obs, _env=env: np.asarray(extract_controller_qpos(obs, env=_env), dtype=np.float32).reshape(-1)
        if not dual_family:
            from algo.groot import DEFAULT_VIDEO_KEY_ALIASES
            agent._video_key_aliases = dict(DEFAULT_VIDEO_KEY_ALIASES)   # "wrist" -> the env's one wrist camera

    def to_env(a):
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        if a.size == env_dim:
            return a
        if a.size == 58 and env_dim == 29:
            r = DUAL_SLOTS["right"]; return np.concatenate([a[r["arm"][0]:r["arm"][1]], a[r["hand"][0]:r["hand"][1]]])
        raise ValueError(f"policy emitted {a.size} dims for a {env_dim}-dim env")

    choices = sub.instructions if args.instruction_sampling == "registered" else (sub.prompt,)
    print(f"[v2] instruction_sampling={args.instruction_sampling} over {len(choices)}: "
          + ", ".join(repr(c) for c in choices), flush=True)
    records, t0 = [], time.time()
    for ep in range(args.episodes):
        seed = args.seed + ep
        instruction = draw_instruction(sub.env_id, seed, choices)
        agent.set_instruction(instruction)
        obs, _ = env.reset(seed=seed)
        agent.reset_rollout(num_envs=1)
        steps, done, success = 0, False, False
        while steps < horizon and not done:
            chunk = agent.predict_action_chunk(obs)
            chunk = np.asarray(chunk.detach().cpu().numpy())[0]
            for a in chunk:
                obs, reward, term, trunc, info = env.step(np.clip(to_env(a), low, high))
                steps += 1
                success = success or _success_from_info(info)
                done = bool(np.asarray(term).any() or np.asarray(trunc).any())
                if done or steps >= horizon:
                    break
        records.append({"episode_index": ep, "environment_seed": seed, "instruction": instruction,
                        "success": bool(success), "episode_length": steps, "video": f"{ep}.mp4"})
        print(f"[episode] idx={ep} seed={seed} success={int(success)} steps={steps} instruction={instruction!r}", flush=True)
    summary = {"family": long_task.family, "subtask": args.subtask, "mode": "family_on_subtask",
               "checkpoint": args.family_ckpt, "instruction_sampling": args.instruction_sampling,
               "instructions": list(choices), "num_episodes": len(records),
               "num_successes": int(sum(r["success"] for r in records)), "horizon": horizon,
               "seed": args.seed, "action_horizon": args.action_horizon, "elapsed_min": round((time.time() - t0) / 60, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out / "episodes.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"[v2] {long_task.family}/{args.subtask}: {summary['num_successes']}/{summary['num_episodes']} -> {out}", flush=True)


if __name__ == "__main__":
    main()
