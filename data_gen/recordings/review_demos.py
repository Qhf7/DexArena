"""Batch review + export of staged demonstration episodes into a trainable LeRobot dataset.

Workflow:
  1. A recording session buffers each auto-segmented episode into the pending
     store (``data/teleop_pending`` by default, see ``episode_store``), with an
     mp4 you can watch.
  2. ``python -m data_gen.recordings.review_demos`` (this tool) lists every pending
     episode, lets you keep/discard each (after watching its video), and exports
     only the kept ones into the final LeRobot dataset via the SAME
     ``LeRobotEpisodeRecorder`` a recording session uses.

The exported dataset lands directly under its ``dataset_root`` (default
``data/teleop_lerobot/<hand>/<env id>``), which is exactly the path the
training configs default to -- so ``train_ppo.py`` / ``train_diffusion.py`` can
consume it with ZERO file moving or post-processing (only matching CLI args).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_gen.recordings.dataset_layout import collection_root, dataset_root as _dataset_root_of
from data_gen.recordings.episode_store import discover_pending_episodes, load_pending_episode
from data_gen.recordings.lerobot_integrity import (
    assert_valid_certificate,
    certificate_path,
    episode_source_hash,
    stage_replay_sidecar,
    verify_lerobot_roundtrip,
    verify_replay_sidecar,
    write_certificate,
)
from data_gen.recordings.recording_integrity import EpisodeIntegrityError


DECISION_CHOICES = ("interactive", "keep-all", "keep-success", "discard-all")


from data_gen.recordings.episode_store import restore_seed

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review staged demonstration episodes and export kept ones to LeRobot."
    )
    parser.add_argument(
        "--pending-dir",
        default="data/teleop_pending",
        help="Pending store directory written by `--record-demo --record-staged`.",
    )
    parser.add_argument(
        "--session",
        default="latest",
        help="Which session_* subfolder under --pending-dir to review. `latest` "
             "(default) = only the most recent run, so old sessions don't get re-exported "
             "and you don't have to delete them first. `all` = every session under the dir "
             "(old behavior). Or pass a specific name, e.g. `session_20260617_120000`.",
    )
    parser.add_argument(
        "--mode",
        choices=DECISION_CHOICES,
        default="interactive",
        help="How to decide keep/discard. `interactive` prompts per episode (watch "
             "the printed video path first); `keep-success` keeps only successful "
             "episodes; `keep-all` / `discard-all` skip prompting.",
    )
    parser.add_argument(
        "--trim-start-frames",
        type=int,
        default=3,
        help="Drop the first N transitions of every kept episode at export. "
             "Removes the record-arm trigger-posture snap (frame-1 |qvel| peaks "
             "5-10 rad/s; measured in 24/24 sampled episodes). Default 3: on 889 "
             "exported episodes frame 2 still rang at 1-4 rad/s (median; DoorOpen "
             "p95 5.2) and frame 3 is the first at operating level (0.1-0.9). A dataset records its trim in the "
             "certificate and refuses appends with a different value. Selection, not "
             "fabrication: remaining pairs are untouched recorded pairs and the "
             "full untrimmed payload still goes to the replay sidecar. "
             "0 disables the trim.",
    )
    parser.add_argument(
        "--keep-unstable-tail",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export successful episodes whose final settle-tail frame no longer "
             "satisfies the task's raw success_condition (default: discard them "
             "with a printed reason -- the achieved state decayed on camera).",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recreate the target LeRobot dataset instead of appending to it.",
    )
    parser.add_argument(
        "--replace-existing-seed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace an existing LeRobot episode that shares the same episode seed.",
    )
    parser.add_argument(
        "--delete-after",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete each pending episode folder after it is exported or discarded.",
    )
    parser.add_argument(
        "--allow-mixed-code-versions",
        action="store_true",
        default=False,
        help="Export even though the batch (or the target dataset) carries more "
             "than one recording build. Success and reward are computed at record "
             "time, so the resulting success column means two different things; "
             "only use this to reproduce a historical dataset, never for new data.",
    )
    parser.add_argument(
        "--reward-from-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export `next.reward` as the per-step success indicator (1.0 where `next.success`, "
             "else 0.0), one reward definition across every task. --no-reward-from-success "
             "exports the reward the environment returned at record time instead. The recorded "
             "reward always stays in the replay sidecar. A dataset is either entirely relabeled "
             "or not: appending to a relabeled dataset relabels automatically, appending "
             "relabeled episodes to a plain dataset is refused.",
    )
    parser.add_argument(
        "--recheck-success",
        action="store_true",
        help="Re-evaluate every episode's success under the CURRENT code before deciding: "
             "restore the recorded physical state of each frame, step the recorded action, "
             "and take the environment's own `success` (latches accumulate along the "
             "trajectory; nothing is replayed open-loop). The keep/discard decision, the "
             "exported `success` channel, the certificate and a `success_recheck.npz` next "
             "to each replay sidecar then follow the current predicate instead of the label "
             "written at recording time. Needs raw states in the pending episodes. Meant "
             "for a predicate fix that changes what counts as success for already-recorded "
             "episodes.",
    )
    parser.add_argument(
        "--dataset-root-override",
        default=None,
        help="Write the exported datasets under this directory instead of the "
             "collection-time path recorded in each pending payload, keeping the "
             "per-environment <env id> directory name. For "
             "verifying the export end to end without touching the production "
             "corpus.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="List decisions without writing the LeRobot dataset or deleting anything.",
    )
    return parser


def _scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _tail_witness(episode: dict) -> tuple[bool, str]:
    """Settle-tail stability witness.

    Question it answers: after success latched, did the ACHIEVEMENT decay during
    the recorded settle tail (the bottle toppled), or is it still there?

    ⚠️ It must not read ``success`` / ``success_condition`` at the final frame:
    measured on a CabinetPourPour session, those are
    the LATCHED value (`cabinet_pour.py:3541` sets `success=success_condition`),
    and a debounce counter resets on a single transient frame. Four episodes
    ended with `success=0` while `pour_ratio` had climbed from 0.938 to 1.000 --
    the pour was strictly BETTER than at latch time. Judging on the latched flag
    would have discarded good data.

    So judge on the task's own RAW signals at the final frame, in order:
      1. ``*_candidate`` booleans -- the undebounced predicate;
      2. ``*_hold_count`` counters -- >0 means the raw predicate holds right now
         (the counter only increments while it holds and resets to 0 when it
         fails), so it is a raw-truth proxy even mid-reaccumulation;
      3. no such field -> cannot judge, keep (pre-tail recordings end exactly at
         the latch instant, where the condition is true by construction).
    """
    if not np.asarray(episode.get("success", [])).any():
        return True, "episode not marked successful"
    infos = episode.get("info_sequence")
    if not infos:
        return True, "no info sequence recorded"
    last = infos[-1]
    success = np.asarray(episode["success"]).reshape(-1).astype(bool)
    if bool(np.asarray(last.get("success", False)).reshape(-1)[0]):
        return True, "success still latched at the final frame"
    latch = int(np.flatnonzero(success)[0])

    # Classify every 0/1 info channel using the episode itself, so the rule does
    # not depend on per-family field NAMES (`pour_candidate` in CabinetPour vs
    # `nut_turned` in Screwing):
    #   * a LATCH never goes 1 -> 0 (`stage_turned`), and a latch ALIAS tracks
    #     `success` exactly (`poured`, `success_condition`) -- both say nothing
    #     about present physical truth, so they are excluded;
    #   * whatever remains is a RAW per-frame predicate.
    # An episode decayed if every raw predicate that was TRUE at the latch frame
    # is FALSE at the final frame.
    series: dict[str, np.ndarray] = {}
    for key in last:
        try:
            col = np.asarray([_scalar(info[key]) for info in infos])
        except Exception:
            continue
        if col.shape != success.shape or not np.isin(col, (0.0, 1.0)).all():
            continue
        b = col.astype(bool)
        if not (b[:-1] & ~b[1:]).any():          # monotone: a latch
            continue
        if np.array_equal(b, success):           # tracks success exactly: an alias
            continue
        series[key] = b
    held_at_latch = {k: v for k, v in series.items() if v[latch]}
    if not held_at_latch:
        return True, "task exposes no raw per-frame predicate to judge"
    dropped = sorted(k for k, v in held_at_latch.items() if not v[-1])
    if not dropped:
        return True, (
            "every raw predicate true at the latch instant still holds "
            f"({', '.join(sorted(held_at_latch))})"
        )
    # Conjunction, not disjunction: the latch fires on a CONJUNCTION of raw
    # predicates, so one of them going false means the achievement is gone even
    # while its siblings still hold. Measured on a ScrewingTurn session:
    # three episodes latched and then
    # the nut backed off during the tail (cum_rotation 3.185/3.250/3.200 ->
    # 3.136/3.111/2.992 against the pi threshold). Requiring only ONE surviving
    # predicate caught two of the three; requiring ALL catches all three, with
    # zero extra discards across the other five sessions checked.
    return False, f"raw predicate(s) dropped during the settle tail: {', '.join(dropped)}"


def _trim_episode_start(episode: dict, k: int) -> dict:
    """Drop the first ``k`` transitions of a pending episode before export.

    Motivation (measured, 24/24 sampled episodes): recording arms from a fully
    held pose (qvel exactly 0) while action[0] is the operator's live retargeted
    hand pose, so the first transition is a trigger-posture SNAP (frame-1 |qvel|
    peaks 5-10 rad/s, ringing down over ~5 frames). That snap is rig-docking, not
    task skill, and it is a poor teacher signal for BC/VLA fine-tuning.

    This is *selection*, not fabrication: every exported (s, a) pair remains an
    exactly recorded pair; the T/T+1 contract is preserved by dropping the same
    ``k`` leading entries from every stream. The replay sidecar still stores the
    FULL untrimmed pending payload (provenance + replay use the sidecar), and
    ``episode.source_hash`` is unchanged because it hashes the pending file.
    """
    if k <= 0:
        return episode
    num_actions = int(episode["actions"].shape[0])
    if num_actions - k < 10:
        print(
            f"[Review] trim-start skipped for {episode['dir']}: only "
            f"{num_actions} actions (need > {k + 10}).",
            flush=True,
        )
        return episode
    out = dict(episode)
    out["obs"] = episode["obs"][k:]
    for key in ("actions", "rewards", "terminated", "truncated", "success"):
        out[key] = episode[key][k:]
    for key in ("env_state_sequence", "raw_state_sequence"):
        if episode.get(key) is not None:
            out[key] = list(episode[key])[k:]
    if episode.get("info_sequence") is not None:
        out["info_sequence"] = list(episode["info_sequence"])[k:]
    if episode.get("image_sequences"):
        out["image_sequences"] = {
            key: values[k:] for key, values in episode["image_sequences"].items()
        }
    meta = dict(episode["meta"])
    meta["trim_start_frames"] = int(k)
    meta["num_actions"] = num_actions - k
    meta["num_frames"] = num_actions - k + 1
    out["meta"] = meta
    return out


def _recheck_identity_key(identity: dict) -> str:
    return json.dumps({k: identity.get(k) for k in ("env_id", "env_kwargs", "robot_uids",
                                                    "control_mode", "obs_mode", "reward_mode")},
                      sort_keys=True)


def _recheck_success(env, episode: dict) -> dict:
    """Success under the CURRENT code along the recorded trajectory.

    Frame t's recorded PHYSICAL state (actors + articulations + the episode clock)
    is restored, the recorded action t is stepped, and the environment's own info
    is read. Task latches are deliberately NOT restored, so they accumulate from
    reset exactly as they would have live; the physics never drifts because every
    frame starts from the recording. Returns per-frame `success` and infos.
    """
    from data_gen.recordings import replay_recording as RR
    from data_gen.recordings.replay_recording import clone_tree_to_numpy

    meta = episode["meta"]
    identity = meta["identity_manifest"]
    raw = episode.get("raw_state_sequence")
    if raw is None:
        raise EpisodeIntegrityError(
            f"{episode['dir']}: --recheck-success needs raw_states.pkl (record with --record-raw-state)")
    actions = np.asarray(episode["actions"])
    steps = int(actions.shape[0])
    if len(raw) < steps:
        raise EpisodeIntegrityError(
            f"{episode['dir']}: {len(raw)} raw states for {steps} actions")
    encoding = identity.get("action_encoding") or {}
    env.reset(seed=restore_seed(meta))
    success = np.zeros(steps, dtype=bool)
    infos = []
    for t in range(steps):
        frame = raw[t]
        physical = {k: v for k, v in frame.items() if k in ("actors", "articulations")}
        if RR.RECORDING_CONTEXT_KEY in frame:
            physical[RR.RECORDING_CONTEXT_KEY] = frame[RR.RECORDING_CONTEXT_KEY]
        RR._set_raw_state(env, physical)
        _obs, _rew, _term, _trunc, info = env.step(RR._reconstruct_action(actions[t], encoding))
        info_np = clone_tree_to_numpy(info)
        success[t] = bool(np.asarray(info_np.get("success", False)).reshape(-1)[0])
        infos.append(info_np)
    return {"success": success, "info_sequence": infos}


def _apply_success_recheck(env, episode: dict, code_fingerprint: str) -> None:
    """Replace the recorded labels with the rechecked ones; keep the recorded ones alongside."""
    result = _recheck_success(env, episode)
    recorded = np.asarray(episode["success"]).reshape(-1).astype(bool)
    now = result["success"]
    first = lambda a: int(np.flatnonzero(a)[0]) if a.any() else -1   # noqa: E731
    meta = episode["meta"]
    meta["final_success_recorded"] = bool(meta.get("final_success"))
    meta["final_success"] = bool(now[-1])
    episode["success_recorded"] = recorded
    episode["success"] = now
    episode["info_sequence_recorded"] = episode.get("info_sequence")
    episode["info_sequence"] = result["info_sequence"]
    episode["success_recheck"] = {
        "code_fingerprint": code_fingerprint,
        "final_recorded": bool(recorded[-1]),
        "final_now": bool(now[-1]),
        "first_success_recorded": first(recorded),
        "first_success_now": first(now),
        "success_untrimmed": now.copy(),
    }


def _resolve_scan_root(pending_dir: Path, session: str) -> Path:
    """Pick which directory `discover_pending_episodes` scans.

    Sessions are written as ``pending_dir/session_<stamp>/episode_NNNN``. By default we
    target only the LATEST session so old runs aren't re-exported (and you don't have to
    delete them first). ``all`` keeps the old whole-dir behavior; a specific name targets
    one session.
    """
    pending_dir = Path(pending_dir)
    session_dirs = sorted(d for d in pending_dir.glob("session_*") if d.is_dir())
    if session == "all" or not session_dirs:
        # `all`, or a flat/legacy layout with no session_* nesting -> scan everything.
        return pending_dir
    if session == "latest":
        # Names are session_YYYYMMDD_HHMMSS, so lexicographic max == most recent.
        chosen = session_dirs[-1]
        print(f"[Review] Reviewing latest session: {chosen.name} "
              f"(of {len(session_dirs)}; use --session all for every session).", flush=True)
        return chosen
    cand = pending_dir / session
    if not cand.exists():
        cand = pending_dir / f"session_{session}"
    if not cand.exists():
        avail = ", ".join(d.name for d in session_dirs) or "(none)"
        raise SystemExit(f"[Review] session '{session}' not found under {pending_dir}. "
                         f"Available: {avail}")
    return cand


def _decide(meta: dict, mode: str, video: Path | None) -> bool | str:
    if mode == "keep-all":
        return True
    if mode == "discard-all":
        return False
    if mode == "keep-success":
        return bool(meta.get("final_success"))
    # interactive
    print(
        f"\n  env={meta.get('env_id')} "
        f"control_mode={meta.get('control_mode')} "
        f"steps={meta.get('num_actions')} success={meta.get('final_success')} "
        f"auto={meta.get('auto_success')}",
        flush=True,
    )
    if video is not None and Path(video).exists():
        print(f"  video: {video}", flush=True)
    else:
        print("  video: (none)", flush=True)
    while True:
        try:
            ans = input("  Keep this demo? [y/N/a=keep all remaining/q to stop]: ").strip().lower()
        except EOFError:
            return False
        if ans in ("y", "yes"):
            return True
        if ans in ("", "n", "no"):
            return False
        if ans in ("a", "all"):
            return "all"   # keep this one AND every remaining episode, no more prompts
        if ans in ("q", "quit"):
            raise KeyboardInterrupt
        print("  Please answer y / n / a / q.", flush=True)


def _resolve_dataset_root(meta: dict, args) -> str:
    """Where this episode's LeRobot dataset goes: `<root>/<hand>/<env id>`.

    The root is the collection-time one the recorder wrote into the payload, or
    `--dataset-root-override`, so a set of pending episodes can be exported
    somewhere disposable and verified end to end WITHOUT writing into the
    production corpus.
    """
    env_id = str(meta.get("env_id") or (meta.get("identity_manifest") or {}).get("env_id") or "export")
    recorded = str(meta.get("dataset_root") or "")
    override = getattr(args, "dataset_root_override", None)
    base = Path(override) if override else collection_root(Path(recorded))
    return str(_dataset_root_of(base, env_id))


NO_SHA = "(no git identity)"


def _identity_code_version(identity: dict) -> str:
    """Which build recorded this episode, AS FAR AS THIS ENVIRONMENT CARES.

    The repo sha is the wrong unit. A collection run spans weeks, and editing a
    family nobody has collected yet moves the sha -- so a sha-based check would
    fork the dataset of every OTHER environment, whose code did not move. The
    per-environment fingerprint (`tasks/config/code_fingerprint.py`) hashes only
    the sources that determine what this environment records.

    Three tiers, most trustworthy first:

      1. the fingerprint the recorder wrote into the payload;
      2. for payloads recorded before fingerprints existed, the same fingerprint
         recomputed out of git at the recorded sha -- so old and new episodes of
         an unchanged environment still compare equal instead of being forced
         apart;
      3. the raw sha, when neither is available. Stricter, never looser.

    Tier 2 is skipped for a dirty tree: git cannot tell us what the uncommitted
    files said, so the sha is all there is.
    """
    env_id = str(identity.get("env_id") or "").strip()
    recorded = str(identity.get("code_fingerprint") or "").strip()
    if recorded:
        return "fp:" + recorded[:12]

    sha = str(identity.get("git_sha") or "").strip()
    if not sha:
        return NO_SHA
    dirty = bool(identity.get("git_dirty"))
    if env_id and not dirty:
        try:
            from tasks.config.code_fingerprint import fingerprint_at_sha

            recovered = fingerprint_at_sha(env_id, sha)
        except Exception:
            recovered = None
        if recovered:
            return "fp:" + recovered[:12]
    return sha[:12] + ("+dirty" if dirty else "")


def _episode_code_version(episode: dict) -> str:
    return _identity_code_version(episode["meta"].get("identity_manifest") or {})


def _existing_code_versions(dataset_root: Path) -> dict[str, int]:
    """The build(s) that recorded what is already in this dataset."""
    found: dict[str, int] = {}
    for meta_path in sorted((dataset_root / "replay").glob("*/meta.json")):
        try:
            identity = json.loads(meta_path.read_text()).get("identity_manifest") or {}
        except Exception:
            continue
        key = _identity_code_version(identity)
        found[key] = found.get(key, 0) + 1
    return found


def _assert_one_code_version(dataset_root: Path, episodes: list, args) -> str:
    """Refuse to put two builds' episodes in one dataset.

    A success predicate, a threshold or a stage boundary can change between
    builds, and `next.success` / `next.reward` are computed at RECORD time. Two
    builds in one dataset therefore means one column with two meanings, and any
    success rate computed over it is a number about nothing. The segmentation is
    baked in too: an episode is cut 20 steps after the latch of the build that
    recorded it, so a later, stricter predicate cannot be recovered from it
    offline -- that data has to be collected again.

    Measured before this check existed: four production datasets
    already held two builds each (CabinetPourPour 56 + 59, ScrewingTurn 102 + 50,
    ScrewingNutPickPlace 70 + 75, SharpaOpenFridge 105 + 47).

    The fix when this fires is a NEW dataset directory, not a bypass. Change the
    predicate, change the pin, change the directory.
    """
    incoming: dict[str, int] = {}
    for episode in episodes:
        key = _episode_code_version(episode)
        incoming[key] = incoming.get(key, 0) + 1
    existing = ({} if bool(args.overwrite)
                else _existing_code_versions(dataset_root))
    combined = dict(existing)
    for key, count in incoming.items():
        combined[key] = combined.get(key, 0) + count
    if len(combined) <= 1:
        return next(iter(combined), NO_SHA)

    def render(counts: dict[str, int]) -> str:
        return ", ".join(f"{k[:12]}={v}" for k, v in sorted(counts.items()))

    message = (
        "this export would put more than one code version in "
        f"{dataset_root.name}: {render(combined)}"
        + (f" (already there: {render(existing)}; this batch: {render(incoming)})"
           if existing else "")
        + ". `next.success` and `next.reward` are computed when the episode is "
          "RECORDED, so mixing builds gives one column two meanings. Export this "
          "batch to a NEW dataset directory, or re-export the whole set with "
          "--overwrite. `fp:` is a fingerprint of THIS environment's own "
          "sources, so this fired because code that changes what this "
          "environment records actually moved -- not merely because some other "
          "task was edited."
    )
    if getattr(args, "allow_mixed_code_versions", False):
        print(f"[Review] WARNING (--allow-mixed-code-versions): {message}", flush=True)
        return "mixed: " + render(combined)
    raise EpisodeIntegrityError(message)


REWARD_RELABEL_SUCCESS = "success_indicator_v1"


def relabel_reward_from_success(episode: dict) -> dict:
    """`rewards` := 1.0 on every step whose `success` is true, else 0.0.

    The per-step `info["success"]` the recorder stored, not a latched version: a step
    where the object slipped out again scores 0 (measured: the environments'
    own sparse reward disagreed with `success` on 0.9% of CabinetPourPour rows, all
    after the first success, so even the sparse recordings change slightly).
    """
    out = dict(episode)
    success = np.asarray(episode["success"], dtype=np.bool_).reshape(-1)
    out["rewards"] = success.astype(np.float32)
    meta = dict(episode["meta"])
    meta["reward_relabel"] = REWARD_RELABEL_SUCCESS
    out["meta"] = meta
    return out


def _previous_episodes(dataset_root: Path, trim: int, new_hashes, relabel: bool = False) -> list:
    """Every episode already in the dataset, loaded from its replay sidecar and
    trimmed (and relabeled) exactly as it was exported, so the round trip can cover
    the whole dataset rather than the batch being appended."""
    from data_gen.recordings.episode_store import load_pending_episode

    skip = {int(h) for h in new_hashes}
    out = []
    for sidecar in sorted((dataset_root / "replay").glob("source_*")):
        episode = load_pending_episode(sidecar)
        if int(episode_source_hash(episode)) in skip:
            continue
        episode = _trim_episode_start(episode, trim)
        out.append(relabel_reward_from_success(episode) if relabel else episode)
    return out


def _export_group(group_key, episodes, args) -> None:
    dataset_root, repo_prefix, robot_type, fps, include_env_state, _image_keys = group_key
    from utils.demo_dataset import (
        LeRobotEpisodeRecorder,
        load_demo_dataset,
        load_demo_episodes,
        require_lerobot,
    )
    from utils.offline_replay_buffer import load_lerobot_replay_dataset

    # Pending metadata stores the collection-time output path, which is usually
    # repo-relative (``data/teleop_lerobot/...``). The production demo loaders
    # interpret any relative path as relative to DATA_ROOT, so handing that value
    # through unchanged turns it into ``<repo>/data/data/teleop_lerobot/...``.
    # Resolve once against the same repo cwd used by collection, then keep this
    # absolute path for writing, certificates, replay sidecars and loader smoke tests.
    dataset_root = Path(dataset_root).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root
    dataset_root = dataset_root.resolve()
    require_lerobot(cache_root=dataset_root)
    previous_hashes: list[int] = []
    previous: dict = {}
    cert = certificate_path(dataset_root)
    appending = (dataset_root / "meta" / "info.json").exists() and not bool(args.overwrite)
    if appending:
        previous = assert_valid_certificate(dataset_root)
        previous_hashes = [int(value) for value in previous.get("source_hashes", [])]
    new_hashes = [episode_source_hash(ep) for ep in episodes]
    if len(set(new_hashes)) != len(new_hashes):
        raise EpisodeIntegrityError("duplicate pending source in one export group")
    duplicate_hashes = sorted(set(previous_hashes).intersection(new_hashes))
    if duplicate_hashes:
        rendered = ", ".join(f"{value:015x}" for value in duplicate_hashes)
        raise EpisodeIntegrityError(
            f"pending episode was already exported: {rendered}"
        )
    code_version = _assert_one_code_version(dataset_root, episodes, args)
    trim = int(args.trim_start_frames)
    if appending:
        # Frame 0 of every exported episode is `trim` transitions into the
        # recording. Two trims in one dataset means two different "frame 0"s
        # with no column saying which is which, so the certificate carries the
        # value and an append has to match it. A certificate from before this
        # field existed cannot say what it holds; re-export it whole.
        prev_trim = (previous.get("diagnostics") or {}).get("trim_start_frames")
        if prev_trim is None:
            raise EpisodeIntegrityError(
                f"{dataset_root.name} predates trim recording (certificate has no "
                "trim_start_frames); re-export the whole dataset with --overwrite "
                "instead of appending to it")
        if int(prev_trim) != trim:
            raise EpisodeIntegrityError(
                f"{dataset_root.name} was exported with --trim-start-frames "
                f"{prev_trim}; this batch uses {trim}. One trim per dataset: "
                "re-export the whole dataset with --overwrite, or match the value.")
    # Reward relabel: `next.reward` := per-step success indicator, the
    # same definition for all 30 subtasks. The environment's own reward is untouched in the
    # replay sidecar (episode.npz). Inherited on append; never mixed within one dataset.
    prev_relabel = ((previous.get("diagnostics") or {}).get("reward_relabel") if appending else None) or None
    wants_relabel = bool(getattr(args, "reward_from_success", False))
    if appending and wants_relabel and prev_relabel is None:
        raise EpisodeIntegrityError(
            f"{dataset_root.name} holds the recorded reward; appending success-indicator rewards "
            "would give `next.reward` two meanings. Re-export the whole dataset with "
            "--reward-from-success, or drop the flag for this batch.")
    reward_relabel = prev_relabel if prev_relabel else (REWARD_RELABEL_SUCCESS if wants_relabel else None)
    if reward_relabel:
        episodes = [relabel_reward_from_success(ep) for ep in episodes]
    cert.unlink(missing_ok=True)

    recorder = LeRobotEpisodeRecorder(
        dataset_root=dataset_root,
        repo_id=f"{repo_prefix}_live",
        fps=int(fps),
        robot_type=str(robot_type),
        task=str(episodes[0]["meta"].get("env_id")),
        include_env_state=bool(include_env_state),
        include_success=True,
        include_seed=True,
        include_source_hash=True,
        overwrite=bool(args.overwrite),
        # Offline re-render (data_gen/render) exports its frames as video features;
        # every other caller records state only and keeps the defaults.
        use_videos=bool(getattr(args, "use_videos", False)),
        next_images=bool(getattr(args, "next_images", True)),
    )
    for ep, source_hash in zip(episodes, new_hashes):
        recorder.replace_episode_seed = (
            ep["episode_seed"] if args.replace_existing_seed else None
        )
        recorder.add_episode(
            obs=ep["obs"],
            actions=ep["actions"],
            rewards=ep["rewards"],
            terminated=ep["terminated"],
            truncated=ep["truncated"],
            success=ep["success"],
            episode_seed=ep["episode_seed"],
            source_hash=source_hash,
            env_state_sequence=ep["env_state_sequence"],
            image_sequences=ep.get("image_sequences"),
        )
    summary = recorder.finalize()

    rechecks = [ep.get("success_recheck") for ep in episodes]
    recheck_diag = {}
    if any(rechecks):
        recheck_diag = {"success_recheck": {
            "code_fingerprint": next(rc["code_fingerprint"] for rc in rechecks if rc),
            "episodes": sum(1 for rc in rechecks if rc),
            "verdict_changed": sum(1 for rc in rechecks if rc and rc["final_recorded"] != rc["final_now"]),
        }}
    try:
        sidecars = [stage_replay_sidecar(dataset_root, ep) for ep in episodes]
        for sidecar, ep in zip(sidecars, episodes):
            rc = ep.get("success_recheck")
            if rc:
                # The sidecar is the untouched recording; the rechecked labels ride next to it.
                np.savez(Path(sidecar) / "success_recheck.npz",
                         success=rc["success_untrimmed"],
                         code_fingerprint=np.array(rc["code_fingerprint"]),
                         final_recorded=np.array(rc["final_recorded"]),
                         final_now=np.array(rc["final_now"]))
        for sidecar in sidecars:
            verify_replay_sidecar(sidecar)
        # The certificate is REWRITTEN on every append, so a round trip over this
        # batch alone would leave the certificate vouching only for the last
        # batch (ExtractApple read 70/152 after its second session). Verify the
        # whole dataset: every earlier episode is reloaded from its replay
        # sidecar and trimmed the way it was exported.
        roundtrip = verify_lerobot_roundtrip(
            dataset_root, _previous_episodes(dataset_root, trim, new_hashes, bool(reward_relabel)) + list(episodes))
        all_hashes = sorted(set(previous_hashes + new_hashes))
        write_certificate(
            dataset_root,
            status="valid",
            source_hashes=all_hashes,
            diagnostics={"roundtrip": roundtrip, "training_loader_smoke": "running",
                         "code_version": code_version, "trim_start_frames": trim,
                         "reward_relabel": reward_relabel, **recheck_diag},
        )

        obs_dim = int(episodes[0]["obs"].shape[1])
        action_dim = int(episodes[0]["actions"].shape[1])
        loaded = load_demo_dataset(
            str(dataset_root),
            expected_obs_dim=obs_dim,
            expected_act_dim=action_dim,
            load_bc_data=True,
            load_reset_states=bool(include_env_state),
        )
        diffusion_episodes = load_demo_episodes(
            str(dataset_root),
            expected_obs_dim=obs_dim,
            expected_act_dim=action_dim,
        )
        replay = load_lerobot_replay_dataset(
            str(dataset_root),
            expected_obs_dim=obs_dim,
            expected_action_dim=action_dim,
        )
        if loaded.bc_obs is None or loaded.bc_actions is None:
            raise RuntimeError("production BC loader returned no samples")
        if not diffusion_episodes:
            raise RuntimeError("production diffusion loader returned no episodes")
        if len(replay) <= 0:
            raise RuntimeError("production replay loader returned no transitions")
        loader_diagnostics = {
            "bc_samples": int(loaded.bc_actions.shape[0]),
            "diffusion_episodes": len(diffusion_episodes),
            "replay_transitions": len(replay),
        }
        write_certificate(
            dataset_root,
            status="valid",
            source_hashes=all_hashes,
            diagnostics={
                "roundtrip": roundtrip,
                "training_loader_smoke": loader_diagnostics,
                "code_version": code_version,
                "trim_start_frames": trim,
                "reward_relabel": reward_relabel,
                **recheck_diag,
            },
        )
    except Exception as exc:
        write_certificate(
            dataset_root,
            status="invalid",
            source_hashes=sorted(set(previous_hashes + new_hashes)),
            diagnostics={"error": str(exc), "trim_start_frames": trim},
        )
        raise RuntimeError(
            f"LeRobot export failed mandatory integrity validation: {exc}"
        ) from exc

    print(
        f"\n[Review] CERTIFIED export {len(episodes)} episode(s) -> {summary.dataset_root} "
        f"({summary.total_episodes} episodes, {summary.total_frames} frames; "
        f"roundtrip + replay sidecar + production loaders passed).",
        flush=True,
    )
    _print_training_commands(episodes[0]["meta"], summary.dataset_root)


def _print_training_commands(meta: dict, dataset_root: Path) -> None:
    env_id = meta.get("env_id")
    control_mode = meta.get("control_mode")
    print(
        "\n[Review] This dataset is directly trainable (no file moving needed). "
        "Match env_id / control_mode at training time:",
        flush=True,
    )
    print(
        "  # Behavior cloning / demo-init in PPO:\n"
        f"  python -m train.train_ppo --env-id {env_id} "
        f"--control-mode {control_mode} --bc-data-path {dataset_root}",
        flush=True,
    )
    print(
        "  # Diffusion policy:\n"
        f"  python -m train.train_diffusion --env-id {env_id} "
        f"--control-mode {control_mode} --demo-data-path {dataset_root}",
        flush=True,
    )


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    scan_root = _resolve_scan_root(Path(args.pending_dir), args.session)
    ep_dirs = discover_pending_episodes(scan_root)
    if not ep_dirs:
        print(f"[Review] No pending episodes found under {scan_root}.", flush=True)
        return

    print(f"[Review] Found {len(ep_dirs)} pending episode(s) under {scan_root}.", flush=True)
    print(f"[Review] Decision mode: {args.mode}.", flush=True)

    groups: dict[tuple, list] = defaultdict(list)
    kept_dirs: list[Path] = []
    discarded_dirs: list[Path] = []
    stopped = False
    keep_all_rest = False   # set once the reviewer picks 'a' (keep all remaining) interactively

    recheck_envs: dict[str, object] = {}
    recheck_changed = 0
    for ep_dir in ep_dirs:
        episode = load_pending_episode(ep_dir)
        meta = episode["meta"]
        if args.recheck_success:
            from data_gen.recordings import replay_recording as RR
            from tasks.config.code_fingerprint import fingerprint_worktree

            identity = meta["identity_manifest"]
            key = _recheck_identity_key(identity)
            if key not in recheck_envs:
                recheck_envs[key] = RR._make_env(identity)
            fp_now = fingerprint_worktree(str(identity.get("env_id")))
            _apply_success_recheck(recheck_envs[key], episode, fp_now)
            rc = episode["success_recheck"]
            if rc["final_recorded"] != rc["final_now"]:
                recheck_changed += 1
                print(
                    f"[Review] RECHECK {ep_dir.name}: recorded final_success={rc['final_recorded']} "
                    f"-> now {rc['final_now']} (first success frame {rc['first_success_recorded']} -> "
                    f"{rc['first_success_now']}) under fp:{fp_now[:12]}",
                    flush=True,
                )
        if keep_all_rest:
            keep = True
        else:
            try:
                decision = _decide(meta, args.mode, episode["video"])
            except KeyboardInterrupt:
                print("\n[Review] Stopped by user; remaining episodes left untouched.", flush=True)
                stopped = True
                break
            if decision == "all":
                keep_all_rest = True
                keep = True
                print("[Review] Keeping this and ALL remaining episodes (no more prompts).", flush=True)
            else:
                keep = bool(decision)
        if keep:
            episode = _trim_episode_start(episode, int(args.trim_start_frames))
            stable, why = _tail_witness(episode)
            if not stable and not args.keep_unstable_tail:
                print(
                    f"[Review] UNSTABLE TAIL -> discarding {ep_dir.name}: {why} "
                    "(the success state decayed during the recorded settle tail; "
                    "--keep-unstable-tail to export anyway).",
                    flush=True,
                )
                discarded_dirs.append(ep_dir)
                continue
            # Bucket episodes by the FULL schema fingerprint: any difference in
            # dataset_root / robot / fps / env-state availability / camera setup
            # would make `LeRobotEpisodeRecorder` reject the second episode with
            # a schema-mismatch error, so we route those into separate buckets
            # and emit independent datasets per bucket.
            raw_image_keys = meta.get("image_keys") or []
            image_keys_tuple = tuple(
                (str(key), tuple(int(d) for d in shape))
                for key, shape in raw_image_keys
            )
            group_key = (
                _resolve_dataset_root(meta, args),
                meta.get("repo_prefix", "dexarena"),
                meta.get("robot_type", "robot"),
                int(meta.get("record_lerobot_fps", 10)),
                bool(meta.get("record_env_state", False)) and episode["env_state_sequence"] is not None,
                image_keys_tuple,
            )
            groups[group_key].append(episode)
            kept_dirs.append(ep_dir)
        else:
            discarded_dirs.append(ep_dir)

    for env in recheck_envs.values():
        try:
            env.close()
        except Exception:
            pass
    if args.recheck_success:
        print(f"[Review] Success rechecked under the current code for {len(ep_dirs)} episode(s); "
              f"final verdict changed on {recheck_changed}.", flush=True)

    print(
        f"\n[Review] Decisions: keep={len(kept_dirs)}, discard={len(discarded_dirs)}"
        + (" (stopped early)" if stopped else "")
        + ".",
        flush=True,
    )

    if args.dry_run:
        print("[Review] Dry run: no dataset written, no files deleted.", flush=True)
        return

    for group_key, episodes in groups.items():
        _export_group(group_key, episodes, args)

    if args.delete_after:
        for ep_dir in kept_dirs + discarded_dirs:
            shutil.rmtree(ep_dir, ignore_errors=True)
        print(
            f"[Review] Deleted {len(kept_dirs) + len(discarded_dirs)} pending "
            "episode folder(s).",
            flush=True,
        )


if __name__ == "__main__":
    main()
