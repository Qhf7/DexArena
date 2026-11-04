"""Action-replay success rate over the SUCCESSFUL human episodes of a session.

`verify_recording_fidelity.py --action-replay` replays the first N episodes of
a root, successful or not, and reports each one. For the number people actually
ask for -- "of the demos we kept, how many replay to success open-loop?" -- the
denominator has to be the kept (final_success) episodes, and the numerator has
to be counted two ways, because they mean different things:

  ever     success became True at some step of the replay
           (the metric a success-terminating evaluator would report)
  final    success is True on the replay's last step
           (what the recorded episode itself looks like: it ends in success)

Both are printed, with the step at which the replay first left the recorded
trajectory by more than 1e-2 and the final state error, so a low number can be
read for what it is: an open-loop divergence property, not a data defect.
Only meaningful on sessions whose code fingerprint matches the current build;
the script prints that check first and refuses to report a rate otherwise.

USAGE
    python -m data_gen.replay.replay_success_rate <pending session | LeRobot dataset root>... --jobs 2
    python -m data_gen.replay.replay_success_rate data/teleop_lerobot/<hand>/<env id>

Every episode is replayed in a FRESH environment (see the note in the child
script) and the episodes of a session are spread over --jobs
worker processes.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import uuid

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CHILD = r'''
import json, pathlib, sys, pickle
import numpy as np, torch
sys.path.insert(0, %(repo)r)
import tasks
from data_gen.recordings import replay_recording as RR
from data_gen.recordings.replay_compare_video import _warm_up_from_frame0
from data_gen.replay.verify_recording_fidelity import robot_key_of, flatten_tree, max_err, as_array, CONTEXT_KEY
from data_gen.recordings.review_demos import _identity_code_version
from data_gen.recordings.episode_store import restore_seed
from tasks.config.code_fingerprint import fingerprint_worktree

session = pathlib.Path(sys.argv[1]); hold = int(sys.argv[2]); shard = int(sys.argv[3]); nshards = int(sys.argv[4]); fresh = sys.argv[5] == "1"
# A pending session holds episode_*/; an exported LeRobot dataset holds the same
# per-episode payload under replay/source_*/ (the replay sidecar). Accept both.
ep_dirs = sorted(session.glob("episode_*")) or sorted((session / "replay").glob("source_*"))
eps = []
for d in ep_dirs:
    m = json.loads((d / "meta.json").read_text())
    if not m.get("final_success"): continue
    z = np.load(d / "episode.npz")
    eps.append({"dir": d, "meta": m, "raw": pickle.loads((d / "raw_states.pkl").read_bytes()),
                "actions": np.asarray(z["actions"]), "success": np.asarray(z["success"]).reshape(-1).astype(bool)})
eps = eps[shard::nshards]
if not eps:
    print("@@JSON@@" + json.dumps({"session": session.name, "n": 0})); sys.exit(0)
idn = eps[0]["meta"]["identity_manifest"]; env_id = idn["env_id"]
ver = _identity_code_version(idn); now = "fp:" + fingerprint_worktree(env_id)[:12]
# One FRESH environment per episode. Measured on TakeAppleToBlender:
# the same episode replayed in a fresh env vs. in an env that had just replayed
# another episode ends in different states (digests dd4f8f... vs 2ccc59...),
# because set_state_dict does not reset PhysX's contact caches. A marginal
# friction grasp then flips on that residue, so the rate depended on the order
# the episodes were run in. A fresh env per episode makes each verdict a pure
# function of (code, episode); two fresh processes agree bit for bit.
env = None if fresh else RR._make_env(idn)
rows = []
for ep in eps:
    raw, actions = ep["raw"], ep["actions"]
    if fresh:
        if env is not None: env.close()
        env = RR._make_env(idn)
    base = env.unwrapped
    env.reset(seed=restore_seed(ep["meta"]))
    rkey = robot_key_of(base, raw[0])
    RR._set_raw_state(env, raw[0]); _warm_up_from_frame0(env, raw[0], rkey, hold)
    ever = False; final = False; first_div = None; worst = 0.0
    for t in range(len(actions)):
        _, _, _, _, info = env.step(torch.as_tensor(np.asarray(actions[t], np.float32))[None])
        s = bool(as_array(info.get("success", False)).reshape(-1)[0]); ever |= s; final = s
        if t + 1 < len(raw):
            now_s = flatten_tree(base.get_state_dict()); was = flatten_tree({k: v for k, v in raw[t + 1].items() if k != CONTEXT_KEY})
            worst = max((max_err(now_s[k], v) for k, v in was.items() if k in now_s), default=0.0)
            if first_div is None and worst > 1e-2: first_div = t + 1
    rows.append({"episode": ep["dir"].name, "steps": len(actions), "ever": ever, "final": final,
                 "first_divergence": first_div, "final_error": worst})
    print(f"  {ep['dir'].name} {len(actions):4d} steps  ever={ever} final={final} first_div={first_div} err={worst:.2e}", flush=True)
env.close()
print("@@JSON@@" + json.dumps({"session": session.name, "env_id": env_id, "version": ver, "current": now,
                               "matches": ver == now, "n": len(rows), "ever": sum(r["ever"] for r in rows),
                               "final": sum(r["final"] for r in rows), "rows": rows}))
'''


def run(session: pathlib.Path, hold: int, timeout: int, shard: int = 0, nshards: int = 1, fresh: bool = True) -> dict:
    env = dict(os.environ); env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    # Every shard gets its OWN temp root. Several task modules stage a
    # SAPIEN-safe copy of a RoboCasa MJCF under `tempfile.gettempdir()` with a
    # per-ASSET (not per-process) name, and they clear it first:
    #     cache_dir = gettempdir()/"dexarena_blender023"/cache_tag
    #     if cache_dir.exists(): shutil.rmtree(cache_dir)      # juice_place_blender_lid.py:592
    # Two shards of the same environment therefore race on one directory, and the
    # loser reads a file the winner has just deleted. Measured on
    # PressBlenderButton with --jobs 3: one shard died with
    # `FileNotFoundError: /tmp/dexarena_blender023/press_button_blender023/model.xml`
    # and the run reported 67 of 100 episodes. `tempfile.gettempdir()` honours
    # TMPDIR, so isolating the shards needs no change to the task code (that file
    # is in the code fingerprint of six collection targets).
    shard_tmp = pathlib.Path(tempfile.gettempdir()) / f"replay_shard_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    shard_tmp.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = env["TEMP"] = env["TMP"] = str(shard_tmp)
    try:
        proc = subprocess.run([sys.executable, "-c", CHILD % {"repo": str(REPO_ROOT)}, str(session), str(hold),
                               str(shard), str(nshards), "1" if fresh else "0"],
                              cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=timeout)
    finally:
        shutil.rmtree(shard_tmp, ignore_errors=True)
    for line in proc.stdout.splitlines():
        if line.startswith("@@JSON@@"):
            return json.loads(line[len("@@JSON@@"):])
    return {"session": session.name, "error": (proc.stderr or proc.stdout)[-600:]}


def expected_episode_count(session: pathlib.Path) -> int | None:
    """How many kept (final_success) episodes the shards together should have replayed.

    The denominator has to be checked against something the shards did not
    produce: a shard that dies takes its episodes with it, and without this the
    rate is reported over whatever survived. Measured on a TakeAppleToBlender
    session: one of three shards failed and the run printed
    `n=65` for a session with 98 kept episodes, with no other sign of trouble.
    """
    dirs = sorted(session.glob("episode_*")) or sorted((session / "replay").glob("source_*"))
    if not dirs:
        return None
    kept = 0
    for d in dirs:
        try:
            if json.loads((d / "meta.json").read_text()).get("final_success"):
                kept += 1
        except Exception:
            return None
    return kept


def _merge(parts: list[dict], expected: int | None = None) -> dict:
    """Combine the shards of one session into one result.

    A failed shard is NOT dropped quietly: its error is kept and the merged
    result carries `incomplete`, so the caller can refuse to report a rate whose
    denominator is missing episodes.
    """
    failed = [p for p in parts if "error" in p]
    parts = [p for p in parts if "error" not in p]
    if not parts:
        return {"session": failed[0]["session"] if failed else "?",
                "error": "every shard failed: " + (failed[0]["error"] if failed else "")}
    rows = sorted((r for p in parts for r in p.get("rows", [])), key=lambda r: r["episode"])
    head = next((p for p in parts if p.get("n")), parts[0])
    out = {k: head.get(k) for k in ("session", "env_id", "version", "current", "matches")}
    out.update({"n": len(rows), "ever": sum(r["ever"] for r in rows), "final": sum(r["final"] for r in rows), "rows": rows})
    out["expected_n"] = expected
    out["failed_shards"] = [p.get("error", "")[-300:] for p in failed]
    out["incomplete"] = bool(failed) or (expected is not None and len(rows) != expected)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", type=pathlib.Path)
    ap.add_argument("--hold-steps", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=2, help="parallel worker processes (each session is split into this many shards)")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--json-out")
    ap.add_argument("--fresh-env", dest="fresh_env", action="store_true", default=True,
                    help="(default) build a new environment for every episode, so no episode inherits "
                         "another's engine contact caches; --no-fresh-env reuses one env per shard (old behaviour)")
    ap.add_argument("--no-fresh-env", dest="fresh_env", action="store_false")
    args = ap.parse_args()
    from concurrent.futures import ThreadPoolExecutor, as_completed

    nshards = max(1, args.jobs)
    results = []
    with ThreadPoolExecutor(max_workers=nshards) as pool:
        futs = {pool.submit(run, s, args.hold_steps, args.timeout, k, nshards, args.fresh_env): (s, k)
                for s in args.sessions for k in range(nshards)}
        by_session: dict[str, list] = {}
        for f in as_completed(futs):
            s, _k = futs[f]; by_session.setdefault(str(s), []).append(f.result())
        exit_code = 0
        for s in args.sessions:
            r = _merge(by_session.get(str(s), []), expected_episode_count(s)); results.append(r)
            if "error" in r:
                print(f"ERROR {r['session']}: {r['error'][-300:]}", flush=True); exit_code = 1; continue
            if r["n"] == 0:
                print(f"{r['session']}: no successful episodes", flush=True); continue
            tag = "fingerprint match" if r["matches"] else "fingerprint DRIFT (not a success rate)"
            print(f"{r['env_id']:46s} {tag}  n={r['n']}  ever-success {r['ever']}/{r['n']} ({100*r['ever']/r['n']:.0f}%)  "
                  f"final-success {r['final']}/{r['n']} ({100*r['final']/r['n']:.0f}%)", flush=True)
            if r["incomplete"]:
                exit_code = 1
                print(f"  !! INCOMPLETE (not a success rate): {len(r['failed_shards'])} shard(s) failed, "
                      f"replayed {r['n']} of {r['expected_n']} episodes. Re-run this session before reporting a rate.", flush=True)
                for e in r["failed_shards"]:
                    print(f"     shard error: {e[-200:]}", flush=True)
    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(results, indent=1))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
