"""Long-horizon evaluation: run a long task stage by stage with subtask policies.

Two modes, same environment, same seeds, same success rule (every stage's gate must latch
inside the long task's own horizon; one failed stage fails the episode):

  chain   one specialist checkpoint per stage; control switches to the next checkpoint the
          moment the current stage's gate latches (utils/dex_long_tasks.py says which key).
  family  one checkpoint trained on the whole family; only the INSTRUCTION switches per stage.

Each stage's instruction is drawn per episode from the subtask's registered instructions
(--instruction-sampling registered, the default, from the episode seed; `canonical` uses each
subtask's canonical one), and the summary records what every episode ran under.

Nine of the ten long tasks run the 58-joint dual-arm robot while the policies are 29-joint
single-arm, so every action is routed into the acting arm's slots and the idle arm is held at
its current joint position (utils/dex_long_tasks.route_action). The policy's "wrist" view is
the acting arm's wrist camera, and its state is that arm's 29 joints cut out of the 58.

    python scripts/dex/eval_long_chain.py --long cabinet --mode chain --ckpt-root $DEX_OUT/runs \\
        --episodes 50 --out $DEX_OUT/evals/long/chain/cabinet
    python scripts/dex/eval_long_chain.py --long cabinet --mode family \\
        --family-ckpt $DEX_OUT/runs/groot_cabinet_family_rig/checkpoint-12000 --episodes 50 --out ...
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

from utils.dex_long_tasks import (  # noqa: E402
    DUAL, DUAL_ACTION_DIM, DUAL_SLOTS, SINGLE_ACTION_DIM, LongTask, resolve_long_task,
    route_action, stage_reached,
)
from utils.dex_tasks import draw_instruction, resolve_dex_task  # noqa: E402


# ----------------------------------------------------------------------------- core loop
def run_episode(env, stages, agents, instructions, *, horizon, dual, qpos_fn, clamp, seed,
                info_hook=None, router=route_action, on_stage=None):
    """One long-task episode. `agents[k]` drives stage k (chain) or is the same object for
    every k (family); `instructions[k]` is sent to the agent when stage k starts.

    Returns (stages_done, steps, success_flag, stage_steps). Pure control flow, so it can be
    exercised with fake envs and agents before a GPU is spent on it.
    """
    obs, _ = env.reset(seed=seed)
    k, steps, success = 0, 0, False
    stage_steps = []
    stage_start = 0
    agent = agents[0]
    if on_stage is not None:
        on_stage(agent, stages[0])
    agent.set_instruction(instructions[0])
    agent.reset_rollout(num_envs=1)
    done = False
    while steps < horizon and not done and k < len(stages):
        chunk = agent.predict_action_chunk(obs)           # (1, <=H, dim)
        chunk = np.asarray(chunk.detach().cpu().numpy() if hasattr(chunk, "detach") else chunk)[0]
        for a in chunk:
            held = qpos_fn(obs)
            routed = clamp(router(a, stages[k].arm, held, dual))
            obs, reward, term, trunc, info = env.step(routed)
            steps += 1
            if info_hook is not None:
                info_hook(info)
            done = bool(np.asarray(term).any() or np.asarray(trunc).any())
            if stage_reached(info, stages[k]):
                stage_steps.append(steps - stage_start); stage_start = steps
                k += 1
                if k == len(stages):
                    success = True
                    break
                agent = agents[k]
                if on_stage is not None:
                    on_stage(agent, stages[k])       # family mode: point the one policy at the new arm
                agent.set_instruction(instructions[k])
                agent.reset_rollout(num_envs=1)
                break                                    # new stage: ask the (new) policy afresh
            if done or steps >= horizon:
                break
    return k, steps, success, stage_steps


# ----------------------------------------------------------------------------- GPU residency
def _park(agent, where):
    """Move an agent's GR00T model to `where` ("cpu" or the cuda device) and free the cache.

    Both A100 nodes are full of 62 GB trainings, leaving ~18 GB per card; a chain of four stage
    policies needs ~20 GB if they all sit on the GPU. Only the acting stage's policy lives there.
    """
    import torch
    model = getattr(getattr(agent, "policy", None), "model", None)
    if model is not None:
        model.to(where)
    if str(where) == "cpu":
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--long", required=True, help="long task env id or family name")
    ap.add_argument("--mode", choices=("chain", "family"), required=True)
    ap.add_argument("--ckpt-root", default=os.path.join(os.environ.get("DEX_OUT", "dex_out"), "runs"),
                    help="chain mode: directory of the specialist runs (groot_<stem>_rig/checkpoint-12000)")
    ap.add_argument("--ckpt-suffix", default="_rig")
    ap.add_argument("--family-ckpt", default=None)
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--seed", type=int, default=10001)
    ap.add_argument("--action-horizon", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--instruction-sampling", choices=("registered", "canonical"), default="registered",
                    help="registered: each stage's instruction drawn per episode from its subtask's registered "
                         "instructions (utils/dex_tasks.draw_instruction); canonical: each subtask's canonical one")
    args = ap.parse_args()

    import torch
    from train.train_groot import make_eval_env
    from algo.groot import build_dex_agent, dex_video_aliases, dex_state_fn

    task: LongTask = resolve_long_task(args.long)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from mani_skill.utils.registration import REGISTERED_ENVS
    import tasks as _task_pkg  # noqa: F401  (registers the envs)
    horizon = int(REGISTERED_ENVS[task.env_id].max_episode_steps)

    env_args = SimpleNamespace(
        env_id=task.env_id, robot_uids=task.robot_uids, control_mode="pd_joint_pos",
        camera_size=256, camera_profile="robodex_fixed", sim_backend="physx_cpu",
        render_backend=None, gpu_id=None, reward_mode=None, eval_env_extra=None,
        env_max_episode_steps=horizon, num_eval_envs=1, eval_partial_reset=False,
        scene_background=None,
    )
    record_dir = None if args.no_video else str(out / "videos")
    env = make_eval_env(env_args, num_envs=1, reconfiguration_freq=1,
                        record_dir=record_dir, max_steps_per_video=horizon)
    import gymnasium as gym
    assert isinstance(env.single_action_space, gym.spaces.Box)
    action_dim = int(np.prod(env.single_action_space.shape))
    assert action_dim == task.action_dim, f"env action_dim {action_dim} != registry {task.action_dim}"
    low, high = env.single_action_space.low, env.single_action_space.high
    clamp = lambda a: np.clip(np.asarray(a, dtype=np.float32), low, high)  # noqa: E731

    from utils.image_observation import extract_controller_qpos
    qpos_fn = lambda obs: extract_controller_qpos(obs, env=env)  # noqa: E731

    # Per stage: the instructions this evaluation may use; the draw happens per episode below.
    stage_choices = [resolve_dex_task(s.subtask).instructions if args.instruction_sampling == "registered"
                     else (resolve_dex_task(s.subtask).prompt,) for s in task.stages]
    print(f"[long] instruction_sampling={args.instruction_sampling}: "
          + "; ".join(f"{s.subtask} x{len(c)}" for s, c in zip(task.stages, stage_choices)), flush=True)
    if args.mode == "chain":
        agents, ckpts = [], []
        for s in task.stages:
            stem = resolve_dex_task(s.subtask).short_name
            ck = None
            for suffix in (args.ckpt_suffix, "_fixcam", ""):
                cand = Path(args.ckpt_root) / f"groot_{stem}{suffix}" / "checkpoint-12000"
                if (cand / "config.json").exists():
                    ck = cand; break
            if ck is None:
                raise SystemExit(f"no checkpoint for {s.subtask} under {args.ckpt_root}")
            dim = DUAL_ACTION_DIM if s.arm == DUAL else SINGLE_ACTION_DIM
            print(f"[chain] stage {len(agents)}: {s.subtask} arm={s.arm} <- {ck}", flush=True)
            agents.append(build_dex_agent(str(ck), s.subtask, s.arm, env, device, args.action_horizon,
                                      f"long_chain_{task.family}", dim))
            ckpts.append(str(ck))
            _park(agents[-1], "cpu")
    else:
        if not args.family_ckpt:
            raise SystemExit("--mode family needs --family-ckpt")
        # The family checkpoint's layout is the family dataset's: 58-dim two-view for package,
        # 29-dim otherwise. Its modality env is any subtask with that layout.
        dual_family = task.family == "package"
        if dual_family:
            os.environ["GROOT_VIDEO_KEYS"] = "head:base_0_rgb,right_wrist:right_wrist_0_rgb"
            modality_env = task.stages[0].subtask
        else:
            modality_env = next(s.subtask for s in task.stages if s.arm != DUAL)
        dim = DUAL_ACTION_DIM if dual_family else SINGLE_ACTION_DIM
        shared = build_dex_agent(args.family_ckpt, modality_env, DUAL if dual_family else task.stages[0].arm,
                             env, device, args.action_horizon, f"long_family_{task.family}", dim)
        agents = [shared] * len(task.stages)
        ckpts = [args.family_ckpt]

    resident = {"agent": None}

    def point_at_arm(agent, stage):
        """Chain mode: put this stage's policy on the GPU and park the previous one.
        Family mode with a 29-dim policy: its wrist view and proprio follow the stage's arm."""
        if args.mode == "chain":
            if resident["agent"] is not agent:
                if resident["agent"] is not None:
                    _park(resident["agent"], "cpu")
                _park(agent, device)
                resident["agent"] = agent
            return
        if task.family == "package" or stage.arm == DUAL:
            return
        agent._video_key_aliases = dex_video_aliases(stage.arm)
        agent._state_fn = dex_state_fn(stage.arm, env)

    def family_route(a, arm, held, dual):
        """package family: a 58-dim output on a right-arm stage keeps only the right slots."""
        a = np.asarray(a).reshape(-1)
        if dual and a.size == DUAL_ACTION_DIM and arm != DUAL:
            s = DUAL_SLOTS[arm]; out_ = np.asarray(held, dtype=np.float32).copy()
            out_[s["arm"][0]:s["arm"][1]] = a[s["arm"][0]:s["arm"][1]]
            out_[s["hand"][0]:s["hand"][1]] = a[s["hand"][0]:s["hand"][1]]
            return out_
        return route_action(a, arm, held, dual)

    router = family_route if (args.mode == "family" and task.family == "package") else route_action

    records, t0 = [], time.time()
    n_stages = len(task.stages)
    for ep in range(args.episodes):
        seed = args.seed + ep
        instructions = [draw_instruction(s.subtask, seed, c) for s, c in zip(task.stages, stage_choices)]
        k, steps, success, stage_steps = run_episode(
            env, task.stages, agents, instructions, horizon=horizon, dual=task.is_dual,
            qpos_fn=qpos_fn, clamp=clamp, seed=seed, router=router, on_stage=point_at_arm)
        records.append({"episode_index": ep, "environment_seed": seed, "stages_done": k,
                        "success": bool(success), "steps": steps, "stage_steps": stage_steps,
                        "instructions": instructions, "video": f"{ep}.mp4"})
        print(f"[episode] idx={ep} seed={seed} stages={k}/{n_stages} success={int(success)} steps={steps}",
              flush=True)
    stage_pass = [sum(1 for r in records if r["stages_done"] > i) for i in range(n_stages)]
    summary = {
        "long_task": task.env_id, "family": task.family, "mode": args.mode, "robot_uids": task.robot_uids,
        "num_episodes": len(records), "num_successes": int(sum(r["success"] for r in records)),
        "success_rate": float(np.mean([r["success"] for r in records])),
        "stages": [s.subtask for s in task.stages], "stage_arms": [s.arm for s in task.stages],
        "stage_reached_counts": stage_pass, "instruction_sampling": args.instruction_sampling,
        "stage_instructions": [list(c) for c in stage_choices], "checkpoints": ckpts,
        "horizon": horizon, "seed": args.seed, "action_horizon": args.action_horizon,
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out / "episodes.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"[long] {task.env_id} {args.mode}: {summary['num_successes']}/{summary['num_episodes']} "
          f"stage counts {stage_pass} -> {out}", flush=True)


if __name__ == "__main__":
    main()
