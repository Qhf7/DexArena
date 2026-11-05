"""Do today's success predicates still agree with what was recorded?

WHY THIS EXISTS, AND WHY IT IS NOT ACTION REPLAY
------------------------------------------------
The restoration check proves the INTERFACE is unchanged -- same observation
width, same action width, same state schema. That says nothing about whether the
task still JUDGES the same world the same way. A threshold moved, a predicate
rewritten, a stage reordered: all of those leave the schema identical and quietly
invalidate every success label already on disk.

Action replay cannot answer this. It feeds recorded actions into an open loop, so
a disagreement at frame 300 could equally be trajectory divergence -- and this
project measures 54% end-to-end action-replay success even on data that is fine.

So this replays the STATE, not the actions. Every recorded frame is restored
exactly and `evaluate()` is re-run on it. Physics never advances, so there is no
divergence to confuse the result: any disagreement is the predicate itself
changing its mind about a world it has been handed verbatim.

TWO THINGS A BARE RESTORE CANNOT MEASURE, AND HOW THIS HANDLES THEM
-------------------------------------------------------------------
1. CONTACT-DERIVED PREDICATES. Right after a restore the engine's contact report
   is EMPTY -- it is rebuilt by the first `simulate()`. So `is_grasped`,
   `contact_count`, `grasp_achieved` and friends read as "nothing is touching"
   on a cold restore. Measured on furniture/PanelLayFlat: `grasp_achieved`
   disagreed on 90 of 119 frames cold and on 0 of 119 after a single warming
   step. Reading that as a rewritten predicate would have been wrong.

   So every frame is evaluated TWICE, cold and after `--warm-steps` hold steps,
   and channels are classified by what the warming does:
     * agrees cold                          -> fine
     * disagrees cold, agrees warm          -> contact-dependent, and resolved
     * disagrees both, and cold != warm     -> contact-dependent and NOT settled
                                               by one step: UNVERIFIABLE here,
                                               reported separately, never as drift
     * disagrees both, and cold == warm     -> insensitive to the contact state.
                                               THIS is the real signal.

2. THE FINAL FRAME. An episode cut by hand at the instant of success ends exactly
   on a debounce threshold, so the last frame can latch under one build and not
   the other with no predicate having changed. Measured across the first test
   sessions: every single `success` disagreement sat on the LAST frame.
   Final-frame-only disagreements are therefore reported as a note. A settle tail
   (`--success-tail-steps`, default 20) removes the ambiguity for new recordings.

Reported per episode:
  * channels that disagree mid-episode and are insensitive to warming -- drift;
  * channels that only a warming step settles, or that warming cannot settle;
  * whether the LATCH FRAME moved, and whether it moved only on the final frame.

USAGE
    python -m data_gen.replay.check_label_parity data/teleop_pending/session_*_test
    python -m data_gen.replay.check_label_parity <lerobot_dataset_root> --episodes 3
    python -m data_gen.replay.check_label_parity <root> --stride 5    # sample frames
Run from the repository root, in the ManiSkill conda env. For the office and
table_clean families, preload the libgomp shipped in the coacd wheel
(LD_PRELOAD=<site-packages>/coacd.libs/libgomp-*.so).
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from data_gen.recordings.episode_store import restore_seed

def _scalars(info: dict) -> dict[str, np.ndarray]:
    """Flatten an info dict to comparable arrays, one level deep."""
    out: dict[str, np.ndarray] = {}
    for key, value in info.items():
        if key == "elapsed_steps":
            continue
        try:
            arr = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        except Exception:
            continue
        if arr.dtype.kind in "biu f".replace(" ", ""):
            out[key] = arr.reshape(-1)
    return out


def _first_true(flags: list[bool]) -> int | None:
    hits = np.flatnonzero(np.asarray(flags, dtype=bool))
    return int(hits[0]) if len(hits) else None


def _hold_action(base, art_name: str):
    import torch

    art = base.scene.articulations[art_name]
    joint_names = [j.name for j in art.get_active_joints()]
    qpos = art.get_qpos().detach().cpu().numpy().reshape(-1)
    order = [j.name
             for controller in base.agent.controller.controllers.values()
             for j in controller.joints]
    return torch.as_tensor(
        np.asarray([qpos[joint_names.index(n)] for n in order], np.float32)
    )[None]


def check_episode(env, episode, stride: int, warm_steps: int = 1) -> dict:
    from data_gen.recordings import replay_recording as RR

    base = env.unwrapped
    raw = episode["raw"]
    recorded_infos = episode.get("infos")
    if not recorded_infos:
        return {"skipped": "no infos.pkl -- nothing to compare against"}
    robot_key = next(iter(base.scene.articulations))
    for name in base.scene.articulations:
        if name in raw[0].get("articulations", {}):
            robot_key = name
            break

    # FRAME ALIGNMENT. `raw_states` holds n_actions + 1 states -- the initial one
    # plus the state after each step -- while `infos` holds one entry per STEP.
    # So `infos[i]` describes `raw[i + 1]`, not `raw[i]`. Comparing them index to
    # index looks exactly like a drifting predicate: measured on DrawerPlugOpen,
    # `drawer_travel` disagreed by up to 12 mm and the error GREW with travel,
    # because it was reading the drawer one control step before it was measured.
    # With the correct alignment the same episode agrees to 0.000e+00.
    frames = list(range(0, min(len(raw) - 1, len(recorded_infos)), max(1, stride)))
    def agrees(new, old):
        if new is None or new.shape != old.shape:
            return None
        return (np.array_equal(new, old) if old.dtype.kind in "biu"
                else bool(np.allclose(new, old, rtol=1e-4, atol=1e-6)))

    # Two passes. Per-frame classification is not enough: on some frames a
    # contact channel's cold and warm values happen to coincide while both
    # differ from the recording, and calling that "insensitive to warming" reads
    # a contact-reproduction failure as predicate drift. So a channel is judged
    # CONTACT-SENSITIVE if warming moved it on ANY checked frame of the episode,
    # and a contact-sensitive channel is never reported as drift.
    bad_frames: dict[str, list[int]] = {}
    contact_sensitive: set[str] = set()
    non_idempotent: set[str] = set()
    rec_success, now_success = [], []

    for index in frames:
        RR._set_raw_state(env, raw[index + 1])
        cold = _scalars(base.get_info())
        if warm_steps > 0:
            for _ in range(warm_steps):
                env.step(_hold_action(base, robot_key))
            RR._set_raw_state(env, raw[index + 1])
            warm = _scalars(base.get_info())
        else:
            warm = cold
        # Some counters are written as `counter = where(cond, counter + 1, 0)`
        # with no per-control-step gate, so `evaluate()` is not idempotent for
        # them: calling it twice on one step counts twice. Recording calls it
        # once per step, so the DATA is fine -- but this checker necessarily
        # calls it again on a restored state that already includes the
        # recording's increment, and reads one count high. Detect it by
        # evaluating the same restored state twice: anything that moves is
        # unverifiable here, and must not be reported as a changed predicate.
        again = _scalars(base.get_info())
        was = _scalars(recorded_infos[index])
        for key in was:
            if agrees(again.get(key), warm.get(key)) is False:
                non_idempotent.add(key)
        for key, old in was.items():
            if agrees(warm.get(key), cold.get(key)) is False:
                contact_sensitive.add(key)
            if agrees(cold.get(key), old) or agrees(warm.get(key), old):
                continue
            if agrees(cold.get(key), old) is None:
                continue
            bad_frames.setdefault(key, []).append(index)
        if "success" in was:
            rec_success.append(bool(was["success"][0]))
            now_success.append(bool(warm.get("success", was["success"])[0]))

    # An episode cut by hand at the instant of success ends ON a debounce
    # threshold, so disagreements confined to its last few checked frames are
    # that boundary, not a rewritten predicate.
    tail_start = frames[max(0, len(frames) - 5)] if frames else 0
    drift, tail_only, unverifiable = {}, {}, {}
    unstable = contact_sensitive | non_idempotent
    for key, hits in bad_frames.items():
        target = (unverifiable if key in unstable
                  else tail_only if min(hits) >= tail_start
                  else drift)
        target[key] = len(hits)
    return {
        "frames": len(frames),
        "drift": drift,
        "final_frame_only": tail_only,
        "unverifiable": unverifiable,
        "first_bad": {k: min(v) for k, v in bad_frames.items()},
        "latch_recorded": _first_true(rec_success),
        "latch_now": _first_true(now_success),
    }


def check_root(root: pathlib.Path, args) -> bool:
    from data_gen.recordings import replay_recording as RR
    from data_gen.recordings.replay_compare_video import _episode_dirs, _load

    label = root.name
    print(f"\n=== {label}")
    dirs = _episode_dirs(root, max(1, args.episodes))
    episodes = []
    for directory in dirs:
        episode = _load(directory)
        infos_path = directory / "infos.pkl"
        episode["infos"] = (pickle.loads(infos_path.read_bytes())
                            if infos_path.exists() else None)
        episodes.append(episode)

    env = RR._make_env(episodes[0]["meta"]["identity_manifest"])
    ok = True
    # A recording made before the task learned to serialise a state key cannot
    # restore whatever that key held, so every predicate downstream of it is
    # unverifiable -- not drifted. This is the furniture case: the family had no
    # `get_state_dict` in its early recordings, so their milestone latches come
    # back False on restore and `evaluate()` re-derives them from a cold contact
    # report. Say so once, up front, instead of printing it as drift.
    env.reset(seed=restore_seed(episodes[0]["meta"]))
    recorded_state = {k: v for k, v in episodes[0]["raw"][0].items()
                      if k != "__recording_context__"}
    current_state = env.unwrapped.get_state_dict()
    missing_keys = sorted(set(current_state) - set(recorded_state))
    # A key can be PRESENT and still predate a field: task state is one packed
    # tensor, so a build that added a column writes a wider one than the
    # recording holds. Pipette's `primed` / `dipped` are exactly that -- 4 fields
    # recorded, 6 written now -- and without this the latches they drive look
    # like drifting predicates.
    def _width(value):
        if isinstance(value, dict):
            return sum(int(np.asarray(v).reshape(1, -1).shape[-1]) for v in value.values())
        return int(np.asarray(
            value.detach().cpu().numpy() if hasattr(value, "detach") else value
        ).reshape(1, -1).shape[-1])

    for key, value in current_state.items():
        if key in ("actors", "articulations") or key not in recorded_state:
            continue
        now, then = _width(value), _width(recorded_state[key])
        if now > then:
            missing_keys.append(f"{key} (recorded {then} fields, now {now})")
    missing_keys = sorted(missing_keys)
    if missing_keys:
        print(f"  NOTE  this recording predates {', '.join(missing_keys)}; "
              "predicates that depend on it cannot be verified from it")
    try:
        for index, episode in enumerate(episodes):
            env.reset(seed=restore_seed(episode["meta"]))
            result = check_episode(env, episode, args.stride, args.warm_steps)
            if result.get("skipped"):
                print(f"  ep{index}: skipped -- {result['skipped']}")
                continue
            frames = result["frames"]
            latch_at_end = (result["latch_now"] is not None
                            and result["latch_recorded"] is None
                            and result["latch_now"] >= frames - 1)
            latch_moved = (result["latch_recorded"] != result["latch_now"]
                           and not latch_at_end)
            if missing_keys:
                # Fold everything into the unverifiable bucket: without the
                # missing state there is no ground truth to compare against.
                result["unverifiable"].update(result["drift"])
                result["unverifiable"].update(result["final_frame_only"])
                result["drift"], result["final_frame_only"] = {}, {}
                latch_moved = False
            if result["drift"] or latch_moved:
                ok = False
            if not result["drift"] and not latch_moved:
                extra = []
                if result["final_frame_only"] or latch_at_end:
                    extra.append("final frame only")
                if result["unverifiable"]:
                    extra.append(f"{len(result['unverifiable'])} channel(s) unverifiable")
                suffix = f"  [{'; '.join(extra)}]" if extra else ""
                print(f"  ep{index}: {frames} frames, predicates agree "
                      f"(latch @ {result['latch_recorded']}){suffix}")
                continue
            if latch_moved:
                print(f"  ep{index}: LATCH MOVED {result['latch_recorded']} -> "
                      f"{result['latch_now']} (not the final frame)")
            for key, count in sorted(result["drift"].items(), key=lambda kv: -kv[1]):
                print(f"  ep{index}: {key} disagrees on {count}/{frames} frames "
                      f"regardless of warming, first at frame {result['first_bad'][key]}")
            if result["unverifiable"]:
                print(f"  ep{index}: not verifiable by state restore (contact-derived, "
                      f"or evaluate() not idempotent), NOT read as drift: "
                      + ", ".join(sorted(result["unverifiable"])[:8]))
    finally:
        env.close()
    print(f"  => {'OK' if ok else 'ATTENTION'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=pathlib.Path,
                    help="pending session dirs, single pending episodes, or LeRobot roots")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--stride", type=int, default=1,
                    help="check every Nth frame (1 = every frame)")
    ap.add_argument("--warm-steps", type=int, default=1,
                    help="hold steps after each restore, to rebuild the contact "
                         "report before evaluating (0 = cold restore only)")
    args = ap.parse_args()

    bad = 0
    for root in args.roots:
        try:
            if not check_root(root, args):
                bad += 1
        except Exception as exc:
            bad += 1
            print(f"\n=== {root.name}\n  ERROR {type(exc).__name__}: {exc}")
    total = len(args.roots)
    print(f"\n[label-parity] {total - bad}/{total} agree with today's predicates")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
