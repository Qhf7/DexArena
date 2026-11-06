"""Batch state-replay vs action-replay comparison videos over LeRobot datasets.

For every selected dataset this renders, per episode, a side-by-side mp4:

  LEFT   the demonstration itself: every frame restored from the recorded
         simulator state (bit-exact by construction);
  RIGHT  open-loop ACTION replay from frame 0, run through the zero-error
         recipe (1-step warm-up + free-object re-pin + task-counter/clock
         write-back; see data_gen/recordings/replay_compare_video.py:_warm_up_from_frame0),

plus a per-dataset summary.csv and a cross-dataset summary table answering the
question the training team actually asks: does replaying the recorded actions
reproduce task success, and where does the trajectory first drift.

Usage:
  # one dataset
  python -m data_gen.replay.batch_compare data/teleop_lerobot/NutBoltTurnNut-Sharpa-v1 --episodes 3

  # every certified dataset under data/teleop_lerobot
  python -m data_gen.replay.batch_compare --all --episodes 2

  # show what the OLD cold-restore replay looked like, for comparison
  python -m data_gen.replay.batch_compare <root> --episodes 2 --no-warmup
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_gen.recordings.dataset_layout import iter_dataset_roots
from data_gen.recordings.replay_compare_video import (  # noqa: E402
    _episode_dirs,
    _load,
    render_episode,
)


def _dataset_roots(args) -> list[pathlib.Path]:
    if args.roots:
        return [pathlib.Path(r) for r in args.roots]
    if not args.all:
        raise SystemExit("give dataset roots, or --all for every dataset under --lerobot-root")
    base = pathlib.Path(args.lerobot_root)
    roots = iter_dataset_roots(base)
    if not roots:
        raise SystemExit(f"no LeRobot datasets with replay sidecars under {base}")
    return roots


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("roots", nargs="*", help="LeRobot dataset roots (data/teleop_lerobot/<hand>/<env id>)")
    ap.add_argument("--all", action="store_true",
                    help="process every dataset under --lerobot-root")
    ap.add_argument("--lerobot-root", default="data/teleop_lerobot")
    ap.add_argument("--episodes", type=int, default=2, help="episodes per dataset (0 = all)")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/replay_compare"))
    ap.add_argument("--res", type=int, default=448)
    ap.add_argument("--stride", type=int, default=2, help="render every Nth frame")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = whole episode")
    ap.add_argument("--hold-steps", type=int, default=1)
    ap.add_argument("--no-warmup", action="store_true",
                    help="old cold-restore replay (no recipe), to demonstrate the difference")
    args = ap.parse_args()

    roots = _dataset_roots(args)
    all_rows: list[dict] = []
    aggregate: list[dict] = []
    started = time.time()
    for root in roots:
        out_dir = args.out / root.name
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            dirs = _episode_dirs(root, args.episodes)
        except SystemExit as exc:
            print(f"[batch] SKIP {root.name}: {exc}", flush=True)
            continue
        print(f"\n[batch] {root.name}: {len(dirs)} episode(s) -> {out_dir}", flush=True)
        rows = []
        for i, d in enumerate(dirs, 1):
            ep = _load(d)
            name = f"{d.name}.mp4"
            print(f"  [{i}/{len(dirs)}] {d.name} ({len(ep['actions'])} steps) ...", flush=True)
            try:
                row = render_episode(
                    ep, out_dir / name, args.res, args.max_steps, args.stride,
                    hold_steps=args.hold_steps, warmup=not args.no_warmup,
                )
            except Exception as exc:  # keep the batch going; report at the end
                row = {"episode": d.name, "env_id": None, "error": str(exc)[:200]}
                print(f"        ERROR: {exc}", flush=True)
                rows.append(row)
                continue
            row["dataset"] = root.name
            rows.append(row)
            print(f"        success replay@{row['replay_success_step']} "
                  f"rec@{row['recorded_success_step']}  "
                  f"contact@{row['first_hand_object_contact']}  "
                  f"robot-drift@{row['first_robot_drift_step']}"
                  f"{'(PRE-CONTACT!)' if row['robot_drift_before_contact'] else ''}  "
                  f"final drift: obj {row['final_object_drift_mm']:.2f} mm / "
                  f"task-joint {row['final_joint_drift_rad']:.2e} rad / "
                  f"robot {row['final_robot_drift_rad']:.2e} rad", flush=True)
        ok_rows = [r for r in rows if "error" not in r]
        if ok_rows:
            with open(out_dir / "summary.csv", "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(ok_rows[0]))
                writer.writeheader()
                writer.writerows(ok_rows)
        succ_action = sum(1 for r in ok_rows if r.get("replay_task_success"))
        succ_state = sum(1 for r in ok_rows if r.get("recorded_task_success"))
        aggregate.append({
            "dataset": root.name,
            "episodes": len(rows),
            "errors": len(rows) - len(ok_rows),
            "state_success": f"{succ_state}/{len(ok_rows)}" if ok_rows else "0/0",
            "action_success": f"{succ_action}/{len(ok_rows)}" if ok_rows else "0/0",
            "median_final_object_drift_mm": (
                sorted(r["final_object_drift_mm"] for r in ok_rows)[len(ok_rows) // 2]
                if ok_rows else None
            ),
        })
        all_rows.extend(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps({"aggregate": aggregate, "episodes": all_rows}, indent=2, default=str)
    )
    print(f"\n[batch] {len(all_rows)} episode(s) in {time.time() - started:.0f}s. Aggregate:", flush=True)
    print(f"  {'dataset':44s} {'eps':>4} {'err':>4} {'state succ':>11} {'action succ':>12} "
          f"{'med drift mm':>13}", flush=True)
    for a in aggregate:
        drift = a["median_final_object_drift_mm"]
        print(f"  {a['dataset']:44s} {a['episodes']:4d} {a['errors']:4d} "
              f"{a['state_success']:>11} {a['action_success']:>12} "
              f"{drift if drift is None else f'{drift:13.2f}'}", flush=True)
    ok_all = [r for r in all_rows if "error" not in r]
    n = len(ok_all)
    state_ok = sum(1 for r in ok_all if r.get("recorded_task_success"))
    action_ok = sum(1 for r in ok_all if r.get("replay_task_success"))
    print(f"  {'OVERALL':44s} {n:4d}      "
          f"{state_ok}/{n} ({state_ok / max(n, 1):.0%})   "
          f"{action_ok}/{n} ({action_ok / max(n, 1):.0%})", flush=True)
    print(f"[batch] videos + per-dataset summary.csv under {args.out}; "
          f"cross-dataset summary.json written.", flush=True)


if __name__ == "__main__":
    main()
