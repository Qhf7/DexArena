"""Build the per-subtask `data/` subset: N episodes that replay to success.

WHAT AND WHY
    `data_all/<env id>`      every episode `review_demos` kept  -> local `data/teleop_lerobot/`
    `data/<hand>/<env id>`          exactly N of those that replay to success, a STRICT
                             SUBSET of the same dataset  -> local `data/teleop_lerobot_100/`
    One number of episodes, one quality bar, so the algorithm side can consume
    every subtask the same way.

SELECTION (deterministic, recorded in meta/subset_manifest.json)
    among the episodes whose open-loop replay reached success, the N with the
    SHORTEST COMPLETION TIME -- the recorded frame at which `success` first turns
    true, not the file length, so a long settle tail does not penalise a fast
    demo. Ties break on total length, then on the source hash, so two runs over
    the same inputs pick the same episodes.

HOW IT IS BUILT
    Not by slicing parquet: every selected episode is re-exported from its replay
    sidecar (the untouched pending payload) through `review_demos._export_group`
    -- the same certified path that produced `data_all` -- with the same
    `trim_start_frames` read off the source certificate. So a subset episode is
    byte-for-byte the `data_all` episode, and the subset gets its own certificate,
    round-trip check and replay sidecars.

USAGE
    P=python   # the ManiSkill environment's interpreter
    # show the selection and whether each dataset has enough episodes (writes nothing)
    $P -m data_gen.replay.build_replay_success_subset --replay-json data/gate_logs/20260911/replay.json
    # build (only datasets with at least N replayable episodes)
    $P -m data_gen.replay.build_replay_success_subset --replay-json data/gate_logs/20260911/replay.json --apply
    # a few datasets only / export even when fewer than N are available
    $P -m data_gen.replay.build_replay_success_subset --replay-json ... --only NutBoltTurnNut-Sharpa-v1 --apply
    $P -m data_gen.replay.build_replay_success_subset --replay-json ... --allow-partial --apply
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import types

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SOURCE = "data/teleop_lerobot"
DEFAULT_OUT = "data/teleop_lerobot_100"
DEFAULT_OUT_COMPLEMENT = "data/teleop_lerobot_append"


def completion_time(episode: dict) -> int:
    """The recorded frame where `success` first turns true; length if it never does.

    This is the demo's completion time, not its file length: the recorder keeps a
    settle tail after the latch, and a fast demo with a long tail should not lose
    to a slow demo with a short one.
    """
    success = np.asarray(episode["success"]).reshape(-1).astype(bool)
    hit = np.flatnonzero(success)
    return int(hit[0]) if hit.size else int(success.size)


def replay_verdicts(replay_json: pathlib.Path) -> dict[str, dict[str, bool]]:
    """{env_id: {sidecar name: reached success}} from a replay_success_rate json."""
    rows = json.loads(replay_json.read_text())
    rows = rows if isinstance(rows, list) else [rows]
    out: dict[str, dict[str, bool]] = {}
    for row in rows:
        if "error" in row or not row.get("env_id"):
            continue
        if row.get("incomplete"):
            raise SystemExit(
                f"{replay_json}: {row['env_id']} replay result is incomplete "
                f"({row['n']} replayed of {row.get('expected_n')} expected); re-run that dataset first")
        if not row.get("matches", True):
            raise SystemExit(f"{replay_json}: {row['env_id']} fingerprint drift, result unusable")
        out[row["env_id"]] = {r["episode"]: bool(r["ever"]) for r in row["rows"]}
    return out


def complement(dataset: pathlib.Path, chosen: list):
    """Every episode of `dataset` that the subset did NOT take, in sidecar order.

    `data/` + `data_append/` partition `data_all`: no episode is published twice, and
    the two together are still everything review kept. A LeRobot dataset is not a bag
    of episodes (its `info.json` counts, its episode/frame indices and its certificate
    cover the whole thing), so the remainder has to be exported as its own dataset
    rather than carved out of the parent.
    """
    from data_gen.recordings.episode_store import load_pending_episode
    from data_gen.recordings.lerobot_integrity import episode_source_hash

    taken = {e["source_hash"] for e in chosen}
    rest = []
    for sidecar in sorted((dataset / "replay").glob("source_*")):
        episode = load_pending_episode(sidecar)
        source_hash = f"{episode_source_hash(episode):015x}"
        if source_hash in taken:
            continue
        rest.append({"sidecar": sidecar, "episode": episode, "source_hash": source_hash,
                     "completion": completion_time(episode),
                     "length": int(np.asarray(episode["actions"]).shape[0])})
    return rest


def select(dataset: pathlib.Path, verdicts: dict[str, bool], count: int):
    """(chosen, considered) -- chosen is the first `count` episodes sorted by completion time."""
    from data_gen.recordings.episode_store import load_pending_episode
    from data_gen.recordings.lerobot_integrity import episode_source_hash

    considered = []
    for sidecar in sorted((dataset / "replay").glob("source_*")):
        if not verdicts.get(sidecar.name):
            continue
        episode = load_pending_episode(sidecar)
        considered.append({
            "sidecar": sidecar,
            "episode": episode,
            "source_hash": f"{episode_source_hash(episode):015x}",
            "completion": completion_time(episode),
            "length": int(np.asarray(episode["actions"]).shape[0]),
        })
    considered.sort(key=lambda e: (e["completion"], e["length"], e["source_hash"]))
    return considered[:count], considered


def build(dataset: pathlib.Path, out_root: pathlib.Path, chosen: list, trim: int,
          reward_from_success: bool = False) -> None:
    from data_gen.recordings.dataset_layout import dataset_root
    from data_gen.recordings.review_demos import _export_group, _trim_episode_start

    meta = chosen[0]["episode"]["meta"]
    episodes = [_trim_episode_start(e["episode"], trim) for e in chosen]
    image_keys = tuple(sorted(k for k, _shape in (meta.get("image_keys") or [])))
    group_key = (
        str(dataset_root(out_root, dataset.name)),
        meta.get("repo_prefix", "dexarena"),
        meta.get("robot_type", "robot"),
        int(meta.get("record_lerobot_fps", 10)),
        bool(meta.get("record_env_state", False)) and episodes[0]["env_state_sequence"] is not None,
        image_keys,
    )
    args = types.SimpleNamespace(overwrite=True, trim_start_frames=trim,
                                 replace_existing_seed=False, mode="keep-success",
                                 dry_run=False, delete_after=False, session="", pending_dir="",
                                 keep_unstable_tail=False, recheck_success=False,
                                 allow_mixed_code_versions=False, dataset_root_override=None,
                                 reward_from_success=bool(reward_from_success))
    _export_group(group_key, episodes, args)


def write_manifest(out: pathlib.Path, dataset: pathlib.Path, chosen: list, considered: list,
                   replay_json: pathlib.Path, trim: int, role: str = "subset") -> None:
    """Record the selection, then RE-ISSUE the certificate so it covers it.

    The certificate hashes the dataset's whole artifact SET, so a file added
    after `_export_group` wrote it makes it invalid ("certified artifact set
    changed"). The first version of this tool produced 12 subsets whose
    certificates all failed for exactly that reason.
    """
    from data_gen.recordings.lerobot_integrity import (
        assert_valid_certificate, certificate_path, write_certificate,
    )

    (out / "meta").mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_dataset": str(dataset),
        "role": role,
        "selection": ("replay reached success, then shortest completion time "
                      "(first recorded success frame), ties by length then source hash")
        if role == "subset" else
        ("every episode the 100-episode subset did not take; data/ + data_append/ "
         "partition data_all with no episode published twice"),
        "replay_results": str(replay_json),
        "trim_start_frames": trim,
        "replay_successful_available": len(considered),
        "chosen": [{"source_hash": e["source_hash"], "completion_frame": e["completion"],
                    "length": e["length"]} for e in chosen],
    }
    (out / "meta" / "subset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    certificate = json.loads(certificate_path(out).read_text(encoding="utf-8"))
    diagnostics = dict(certificate.get("diagnostics") or {})
    diagnostics[role] = {k: manifest[k] for k in
                         ("source_dataset", "role", "selection", "replay_results",
                          "replay_successful_available")}
    diagnostics[role]["episodes"] = len(chosen)
    write_certificate(out, status=certificate["status"],
                      source_hashes=certificate["source_hashes"], diagnostics=diagnostics)
    assert_valid_certificate(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay-json", required=True, type=pathlib.Path,
                    help="the --json-out of replay_success_rate (must be complete and fingerprint-matched)")
    ap.add_argument("--source-root", default=DEFAULT_SOURCE, type=pathlib.Path)
    ap.add_argument("--out-root", type=pathlib.Path, default=None,
                    help="default by --mode: data/teleop_lerobot_100 or data/teleop_lerobot_append")
    ap.add_argument("--mode", choices=("subset", "complement"), default="subset",
                    help="subset = the released N episodes; complement = the rest (data_append/)")
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--only", nargs="*", default=None, help="env ids; default all")
    ap.add_argument("--allow-partial", action="store_true", help="export even when fewer than --count episodes replay to success")
    ap.add_argument("--apply", action="store_true", help="write the datasets; default is a dry run that only prints")
    args = ap.parse_args()
    if args.out_root is None:
        args.out_root = pathlib.Path(DEFAULT_OUT if args.mode == "subset" else DEFAULT_OUT_COMPLEMENT)

    from data_gen.recordings.dataset_layout import dataset_root, iter_dataset_roots
    from data_gen.recordings.lerobot_integrity import assert_valid_certificate

    verdicts = replay_verdicts(args.replay_json)
    datasets = iter_dataset_roots(args.source_root)
    done, short, skipped = [], [], []
    for dataset in datasets:
        env_id = dataset.name
        if args.only and env_id not in args.only:
            continue
        if env_id not in verdicts:
            skipped.append((env_id, "no replay result")); continue
        cert = assert_valid_certificate(dataset)
        trim = (cert.get("diagnostics") or {}).get("trim_start_frames")
        if trim is None:
            skipped.append((env_id, "certificate has no trim_start_frames; recertify first")); continue
        chosen, considered = select(dataset, verdicts[env_id], args.count)
        enough = len(chosen) >= args.count
        if not enough and not args.allow_partial:
            short.append((env_id, len(considered)))
            print(f"{env_id:46s} replayable {len(considered):3d}   <- fewer than {args.count}, skipped"
                  + (" (complement = the whole dataset; use the master directory)" if args.mode == "complement" else ""))
            continue
        if not chosen:
            short.append((env_id, 0))
            print(f"{env_id:46s} replayable   0   <- nothing replays to success, skipped"); continue
        if args.mode == "complement":
            chosen = complement(dataset, chosen)
            if not chosen:
                print(f"{env_id:46s} empty complement, skipped"); continue
            print(f"{env_id:46s} complement {len(chosen):3d} episodes (of {len(chosen) + args.count})")
        else:
            print(f"{env_id:46s} replayable {len(considered):3d}  chosen {len(chosen):3d}  "
                  f"completion {chosen[0]['completion']}-{chosen[-1]['completion']} frames"
                  + ("" if enough else f"   <- only {len(considered)} available (--allow-partial)"))
        if args.apply:
            # the subset carries whatever reward definition its master carries
            build(dataset, args.out_root.resolve(), chosen, int(trim),
                  reward_from_success=bool((cert.get("diagnostics") or {}).get("reward_relabel")))
            write_manifest(dataset_root(args.out_root, dataset.name).resolve(), dataset, chosen, considered,
                           args.replay_json, int(trim), role=args.mode)
            done.append(env_id)

    print(f"\n{len(datasets)} datasets; " + (f"{len(done)} written -> {args.out_root}" if args.apply
                                            else "dry run (add --apply to write)"))
    if short:
        print(f"{len(short)} with fewer than {args.count}: " + ", ".join(f"{e}({n})" for e, n in short))
    if skipped:
        print("skipped: " + ", ".join(f"{e}({why})" for e, why in skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
