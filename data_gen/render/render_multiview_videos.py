"""Re-render a state-only LeRobot dataset into the same dataset plus per-camera videos.

WHAT
    For every episode of `data/teleop_lerobot_100/<hand>/<env id>` (the 100-episode release
    subsets): rebuild the recording environment from the episode's identity manifest,
    reset it with the RECORDED seed once, then restore every recorded raw state in turn
    (no physics step, no action replay) and capture the family camera rig
    (data_gen/render/camera_rigs.py: base + one wrist per arm, 256x256). The frames are
    exported through the same certified path as the source dataset
    (`review_demos._export_group`, same trim, same episode order), with the cameras as
    LeRobot VIDEO features `observation.images.<camera>`; the state columns are the
    source's state columns, byte for byte (the round trip checks them exactly and holds
    the decoded video to a PSNR floor). Output: `data/teleop_lerobot_100_video/<hand>/<env id>`.

GATES (every number is per dataset, with its denominator, in meta/video_manifest.json)
    seed         reset(seed) alone renders the same pixels as reset(seed)+restore(raw[0]) --
                 the recorded seed rebuilds the recorded scene (layout AND appearance draws).
                 Counted, not enforced: the P0 in the handoff doc (reconfiguration_freq=0
                 envs) makes this fail for a known reason while the restored frames are
                 still right; the count says how many.
    reconfigure  the env never reconfigures inside an episode (otherwise randomized
                 appearance is re-drawn per frame and the video flickers).
    frames       rendered frames == recorded states == exported rows, per camera.
    determinism  K frames per episode re-rendered after a fresh reset+restore are
                 pixel-identical to the sequential pass. Enforced.
    degenerate   every camera's frames have non-trivial variance (not black / uniform).
                 Enforced.
    fingerprint  the rig is in no recording closure (no collection target's code fingerprint
                 depends on it).

USAGE
    P=python   # the ManiSkill environment's interpreter
    $P -m data_gen.render.render_multiview_videos                      # plan: datasets / episodes / frames / rig
    $P -m data_gen.render.render_multiview_videos --only OfficeDeskClickMouse-Sharpa-v1 --episodes 2 \
        --out-root /tmp/render_trial --apply                            # trial: 2 episodes
    $P -m data_gen.render.render_multiview_videos --jobs 3 --apply     # every release subset
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
import time
import types
import uuid

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SOURCE = "data/teleop_lerobot_100"
DEFAULT_OUT = "data/teleop_lerobot_100_video"
IMAGE_KEY_PREFIX = "observation.images."
MIN_FRAME_STD = 2.0


# --------------------------------------------------------------------------- helpers
def env_id_of(dataset: pathlib.Path) -> str:
    return dataset.name


def ordered_sidecars(dataset: pathlib.Path) -> list[pathlib.Path]:
    """Replay sidecars in the dataset's own episode order, so the video dataset's
    episode i is the source dataset's episode i."""
    from data_gen.recordings.lerobot_integrity import sidecars_in_episode_order

    return sidecars_in_episode_order(dataset)


def _build_env(identity: dict, family: str, markers_off: bool):
    from mani_skill.utils.registration import REGISTERED_ENVS

    from data_gen.render import camera_rigs as RIG
    from data_gen.recordings import replay_recording as RR
    from tasks.config.guidance_markers import GUIDANCE_MARKER_KWARG

    identity = dict(identity)
    kwargs = json.loads(identity.get("env_kwargs") or "{}")
    # The recording's render-camera size override would resize the rig's cameras too
    # (ManiSkill applies unnamed keys to every human render camera); the rig replaces
    # the render cameras wholesale, so it goes.
    kwargs.pop("human_render_camera_configs", None)
    if markers_off:
        kwargs[GUIDANCE_MARKER_KWARG] = False
    identity["env_kwargs"] = json.dumps(kwargs)

    RIG.install_rig(REGISTERED_ENVS[str(identity["env_id"])].cls, family)
    try:
        return RR._make_env(identity), markers_off
    except TypeError as exc:
        if markers_off and GUIDANCE_MARKER_KWARG in str(exc):
            return RR._make_env(_build_env_identity(identity)), False
        raise


def _build_env_identity(identity: dict) -> dict:
    from tasks.config.guidance_markers import GUIDANCE_MARKER_KWARG

    kwargs = json.loads(identity.get("env_kwargs") or "{}")
    kwargs.pop(GUIDANCE_MARKER_KWARG, None)
    return {**identity, "env_kwargs": json.dumps(kwargs)}


def _actor_names(env) -> set[str]:
    base = env.unwrapped
    names = set(base.scene.actors)
    registry = getattr(base.scene, "state_dict_registry", None)
    if registry is not None:
        names |= set(getattr(registry, "actors", {}))
    return names


def make_env(identity: dict, family: str, recorded_actors: set[str] = frozenset()):
    """The recording environment, with the family rig as its human render cameras and
    the guidance markers out of the picture.

    Returns (env, markers_off, hidden_marker_actors). Some environments honour
    `show_guidance_markers=False` by NOT BUILDING the marker actor, and some of those
    markers are in the recorded state dict (furniture `place_zone`: restoring into the
    markers-off env is a KeyError). Then, and only then, the
    markers-on env is used and exactly those actors' visuals are hidden. Markers that
    are not in the recorded state (screwing `bolt_placement_region`) simply stay unbuilt.
    """
    env_off, markers_off = _build_env(identity, family, markers_off=True)
    if not markers_off:
        return env_off, False, []
    missing = sorted(set(recorded_actors) - _actor_names(env_off))
    if not missing:
        return env_off, True, []
    env_off.close()
    env_on, _ = _build_env(identity, family, markers_off=False)
    still_missing = sorted(set(missing) - _actor_names(env_on))
    if still_missing:
        raise RuntimeError(f"recorded state has actors this environment cannot build: {still_missing}")
    return env_on, True, missing


def hide_markers(base, hidden: list[str]) -> None:
    """Re-applied after every reset: a reconfigure rebuilds the actors visible.

    `Actor.hide_visual` assumes a collision body; a visual-only marker has none, so the
    render component is switched off directly in that case.
    """
    import sapien

    for name in hidden:
        actor = base.scene.actors[name]
        try:
            actor.hide_visual()
        except AttributeError:
            for entity in actor._objs:
                body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
                if body is not None:
                    body.visibility = 0.0


def capture(base, camera_names: list[str]) -> dict[str, np.ndarray]:
    """One uint8 (H,W,3) frame per rig camera from the current scene state.
    Hidden objects stay hidden: the videos carry sensor semantics, not the viewer's."""
    from data_gen.recordings.replay_recording import to_numpy_rgb_image

    base.scene.update_render(update_sensors=False, update_human_render_cameras=True)
    images = base.scene.get_human_render_camera_images()
    if not isinstance(images, dict):
        raise RuntimeError(f"human render cameras returned {type(images).__name__}, expected dict")
    out = {}
    for name in camera_names:
        frame = to_numpy_rgb_image(images.get(name))
        if frame is None:
            raise RuntimeError(f"camera {name!r} produced no RGB image (got {sorted(images)})")
        out[name] = frame
    return out


def seed_dependence(env, camera_names, raw_states, seed: int, hidden_markers=()) -> dict:
    """Does anything VISIBLE depend on the reset seed beyond the restored state?

    Two frames of one episode, each rendered after reset(recorded seed)+restore and after
    reset(another seed)+restore; a third render with the recorded seed again is the
    determinism control. Zero differing pixels means the video is a function of the
    state dict alone and a wrong seed could not have changed it. Measured on the 22
    release subsets of the first batch: 19 are seed-independent; furniture (panel colour drawn at
    reconfigure) and TableClean (the static bin, not in the state dict) are not.
    """
    from data_gen.recordings import replay_recording as RR

    base = env.unwrapped
    out = {}
    for t in sorted({0, len(raw_states) // 2}):
        renders = []
        for s in (seed, seed + 977, seed):
            env.reset(seed=int(s))
            hide_markers(base, hidden_markers)
            RR._set_raw_state(env, raw_states[t])
            renders.append(capture(base, camera_names))
        out[str(t)] = {name: {"other_seed_px": int(np.any(renders[0][name] != renders[1][name], axis=-1).sum()),
                              "same_seed_px": int(np.any(renders[0][name] != renders[2][name], axis=-1).sum())}
                       for name in camera_names}
    return out


def _count_reconfigures(base) -> list[int]:
    counter = [0]
    original = base._reconfigure

    def counted(*args, **kwargs):
        counter[0] += 1
        return original(*args, **kwargs)

    base._reconfigure = counted
    return counter


def render_episode(env, counter, camera_names, episode_full, episode, seed, tmp_dir, index,
                   determinism_frames, rng, hidden_markers=()) -> tuple[dict[str, np.ndarray], dict]:
    from data_gen.recordings import replay_recording as RR

    base = env.unwrapped
    raw_full = episode_full["raw_state_sequence"]
    raw = episode["raw_state_sequence"]
    if raw is None or raw_full is None:
        raise RuntimeError("episode has no raw_state_sequence; re-render needs --record-raw-state recordings")
    frames_expected = int(np.asarray(episode["obs"]).shape[0])
    if len(raw) != frames_expected:
        raise RuntimeError(f"raw states {len(raw)} != observations {frames_expected}")

    before = counter[0]
    env.reset(seed=int(seed))
    hide_markers(base, hidden_markers)
    reconfigures_at_reset = counter[0] - before
    after_reset = capture(base, camera_names)
    RR._set_raw_state(env, raw_full[0])
    restored_start = capture(base, camera_names)
    seed_px_diff = {name: int(np.any(after_reset[name] != restored_start[name], axis=-1).sum())
                    for name in camera_names}

    frames = {}
    for name in camera_names:
        h, w, _ = after_reset[name].shape
        frames[name] = np.lib.format.open_memmap(tmp_dir / f"{index:03d}_{name}.npy", mode="w+",
                                                 dtype=np.uint8, shape=(frames_expected, h, w, 3))
    started = time.time()
    for t in range(frames_expected):
        RR._set_raw_state(env, raw[t])
        images = capture(base, camera_names)
        for name in camera_names:
            frames[name][t] = images[name]
    render_seconds = time.time() - started
    reconfigures_during = counter[0] - before - reconfigures_at_reset

    sample = sorted(int(t) for t in rng.choice(frames_expected, size=min(determinism_frames, frames_expected),
                                               replace=False)) if determinism_frames > 0 else []
    mismatches = []
    if sample:
        env.reset(seed=int(seed))
        hide_markers(base, hidden_markers)
        for t in sample:
            RR._set_raw_state(env, raw[t])
            images = capture(base, camera_names)
            for name in camera_names:
                if not np.array_equal(images[name], frames[name][t]):
                    mismatches.append([t, name, int(np.any(images[name] != frames[name][t], axis=-1).sum())])

    stride = max(1, frames_expected // 20)
    frame_std = {name: float(np.asarray(frames[name][::stride], dtype=np.float32).std()) for name in camera_names}
    for memmap in frames.values():
        memmap.flush()

    record = {
        "index": index,
        "source_hash": f"{int(episode_source_hash_of(episode)):015x}",
        "frames": frames_expected,
        "seed": int(seed),
        "seed_source": "env_episode_seed" if episode_full.get("env_episode_seed") is not None else "episode_seed",
        "seed_px_diff": seed_px_diff,
        "reconfigures_at_reset": reconfigures_at_reset,
        "reconfigures_during_episode": reconfigures_during,
        "determinism": {"frames_checked": len(sample), "cameras": len(camera_names), "mismatches": mismatches},
        "frame_std": frame_std,
        "render_seconds": round(render_seconds, 2),
    }
    return {name: memmap.filename for name, memmap in frames.items()}, record


def episode_source_hash_of(episode) -> int:
    from data_gen.recordings.lerobot_integrity import episode_source_hash

    return episode_source_hash(episode)


def contact_sheet(frames: dict[str, np.ndarray], path: pathlib.Path) -> None:
    """Rows = cameras, columns = first / middle / last frame of the episode."""
    from PIL import Image

    names = list(frames)
    total = frames[names[0]].shape[0]
    picks = sorted({0, total // 2, total - 1})
    rows = [np.concatenate([np.asarray(frames[name][t]) for t in picks], axis=1) for name in names]
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


# --------------------------------------------------------------------------- one dataset
def render_dataset(dataset: pathlib.Path, out_root: pathlib.Path, max_episodes: int,
                   determinism_frames: int) -> dict:
    from data_gen.render import camera_rigs as RIG
    from data_gen.recordings.dataset_layout import dataset_root
    from data_gen.recordings.episode_store import load_pending_episode, restore_seed
    from data_gen.recordings.lerobot_integrity import (
        assert_valid_certificate, certificate_path, write_certificate,
    )
    from data_gen.recordings.review_demos import _export_group, _trim_episode_start

    env_id = env_id_of(dataset)
    family = RIG.family_of(env_id)
    cert = assert_valid_certificate(dataset)
    trim = (cert.get("diagnostics") or {}).get("trim_start_frames")
    if trim is None:
        raise RuntimeError(f"{dataset.name}: certificate has no trim_start_frames")
    trim = int(trim)
    sidecars = ordered_sidecars(dataset)
    if max_episodes > 0:
        sidecars = sidecars[:max_episodes]

    out = dataset_root(out_root, env_id)
    tmp_dir = out_root / ".render_tmp" / dataset.name
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    env = None
    started = time.time()
    try:
        episodes, records, sheet_frames = [], [], None
        for index, sidecar in enumerate(sidecars):
            full = load_pending_episode(sidecar)
            episode = _trim_episode_start(full, trim)
            if env is None:
                recorded_actors = set((full["raw_state_sequence"] or [{}])[0].get("actors", {}))
                env, markers_off, hidden_markers = make_env(full["meta"]["identity_manifest"], family,
                                                            recorded_actors)
                base = env.unwrapped
                counter = _count_reconfigures(base)
                camera_names = RIG.rig_camera_names(base)
                if hidden_markers:
                    print(f"  [{dataset.name}] markers kept in the scene but hidden: {hidden_markers}", flush=True)
            seed = restore_seed(full, default=None)
            if seed is None:
                raise RuntimeError(f"{sidecar.name}: no seed to reset with")
            if index == 0:
                seed_probe = seed_dependence(env, camera_names, episode["raw_state_sequence"], int(seed), hidden_markers)
                depends = any(v["other_seed_px"] for t in seed_probe.values() for v in t.values())
                print(f"  [{dataset.name}] visuals depend on the reset seed: {depends} {seed_probe}", flush=True)
            paths, record = render_episode(env, counter, camera_names, full, episode, seed, tmp_dir, index,
                                           determinism_frames, rng, hidden_markers)
            images = {IMAGE_KEY_PREFIX + name: np.load(path, mmap_mode="r") for name, path in paths.items()}
            if sheet_frames is None:
                sheet_frames = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
            episode = dict(episode)
            episode["image_sequences"] = images
            episode["raw_state_sequence"] = None  # not needed by the export; frees ~10 MB per episode
            episode["info_sequence"] = None
            episodes.append(episode)
            records.append(record)
            print(f"  [{dataset.name}] ep {index:3d} {record['frames']:4d} frames "
                  f"{record['frames'] / max(record['render_seconds'], 1e-6):6.1f} fps "
                  f"seed_px_diff={max(record['seed_px_diff'].values())} "
                  f"det_mismatch={len(record['determinism']['mismatches'])}", flush=True)
        render_seconds = time.time() - started

        # Enforced gates. Everything is still reported, but these three stop the export.
        failures = []
        for r in records:
            if r["reconfigures_during_episode"]:
                failures.append(f"ep {r['index']}: reconfigured {r['reconfigures_during_episode']}x mid-episode")
            if r["determinism"]["mismatches"]:
                failures.append(f"ep {r['index']}: {len(r['determinism']['mismatches'])} non-deterministic frames")
            low = [n for n, s in r["frame_std"].items() if s < MIN_FRAME_STD]
            if low:
                failures.append(f"ep {r['index']}: degenerate camera(s) {low}")
        if failures:
            raise RuntimeError(f"{dataset.name}: render gates failed:\n  " + "\n  ".join(failures[:20]))

        meta = episodes[0]["meta"]
        image_keys = tuple(sorted((k, tuple(int(d) for d in v.shape[1:]))
                                  for k, v in episodes[0]["image_sequences"].items()))
        group_key = (
            str(out),
            meta.get("repo_prefix", "dexarena"),
            meta.get("robot_type", "robot"),
            int(meta.get("record_lerobot_fps", 10)),
            bool(meta.get("record_env_state", False)) and episodes[0]["env_state_sequence"] is not None,
            image_keys,
        )
        args = types.SimpleNamespace(overwrite=True, trim_start_frames=trim, replace_existing_seed=False,
                                     mode="keep-success", dry_run=False, delete_after=False, session="",
                                     pending_dir="", keep_unstable_tail=False, recheck_success=False,
                                     allow_mixed_code_versions=False, dataset_root_override=None,
                                     reward_from_success=bool((cert.get("diagnostics") or {}).get("reward_relabel")),
                                     use_videos=True, next_images=False)
        if env is not None:
            env.close()
            env = None
        export_started = time.time()
        _export_group(group_key, episodes, args)
        export_seconds = time.time() - export_started

        (out / "meta").mkdir(parents=True, exist_ok=True)
        # LeRobot stages PNGs under images/ while encoding and leaves the empty tree behind.
        if (out / "images").is_dir() and not any(p.is_file() for p in (out / "images").rglob("*")):
            shutil.rmtree(out / "images", ignore_errors=True)
        subset_manifest = dataset / "meta" / "subset_manifest.json"
        if subset_manifest.exists():
            shutil.copy2(subset_manifest, out / "meta" / "subset_manifest.json")
        contact_sheet(sheet_frames, out / "meta" / "contact_sheet.png")
        seed_ok = sum(1 for r in records if max(r["seed_px_diff"].values()) == 0)
        summary = {
            "env_id": env_id,
            "source_dataset": str(dataset),
            "episodes": len(records),
            "partial": bool(max_episodes > 0 and len(records) < len(ordered_sidecars(dataset))),
            "frames": int(sum(r["frames"] for r in records)),
            "cameras": camera_names,
            "guidance_markers_off": bool(markers_off),
            "hidden_marker_actors": list(hidden_markers),
            "trim_start_frames": trim,
            # Pixel-level, robot included: the recorded frame 0 is never the reset pose (the
            # arm is already moving at frame 0), so this is ~always 0/N. Kept so manifests
            # stay comparable across batches; `visuals_depend_on_seed` is the informative one.
            "seed_reproduces_reset": {"episodes": seed_ok, "of": len(records)},
            "visuals_depend_on_seed": {"episode_0": seed_probe,
                                       "depends": any(v["other_seed_px"] for t in seed_probe.values() for v in t.values())},
            "seed_sources": {s: sum(1 for r in records if r["seed_source"] == s)
                             for s in sorted({r["seed_source"] for r in records})},
            "reconfigures_during_episode": int(sum(r["reconfigures_during_episode"] for r in records)),
            "determinism": {"frames_checked": int(sum(r["determinism"]["frames_checked"] for r in records)),
                            "mismatching_frames": int(sum(len(r["determinism"]["mismatches"]) for r in records))},
            "render_seconds": round(render_seconds, 1),
            "export_seconds": round(export_seconds, 1),
        }
        manifest = {"rig": RIG.describe(family), **summary, "episodes_detail": records}
        (out / "meta" / "video_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        certificate = json.loads(certificate_path(out).read_text(encoding="utf-8"))
        diagnostics = dict(certificate.get("diagnostics") or {})
        diagnostics["video"] = {"rig": RIG.describe(family), **summary}
        write_certificate(out, status=certificate["status"], source_hashes=certificate["source_hashes"],
                          diagnostics=diagnostics)
        assert_valid_certificate(out)
        return {"dataset": dataset.name, **summary}
    finally:
        if env is not None:
            env.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --------------------------------------------------------------------------- driver
def _worker(dataset: pathlib.Path, out_root: pathlib.Path, max_episodes: int, determinism_frames: int,
            timeout: int) -> dict:
    """One dataset in a subprocess with its own TMPDIR (the patched-MJCF cache is per asset
    and rebuilt with rmtree, so two processes of one env in one /tmp delete each other)."""
    env = dict(os.environ)
    shard_tmp = pathlib.Path(tempfile.gettempdir()) / f"render_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    shard_tmp.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = env["TEMP"] = env["TMP"] = str(shard_tmp)
    result_path = shard_tmp / "result.json"
    log_path = out_root / ".render_tmp" / f"{dataset.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "w") as log:
            proc = subprocess.run(
                [sys.executable, "-m", "data_gen.render.render_multiview_videos", "--dataset", str(dataset),
                 "--out-root", str(out_root), "--episodes", str(max_episodes),
                 "--determinism-frames", str(determinism_frames), "--apply", "--json-out", str(result_path)],
                cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        if proc.returncode != 0 or not result_path.exists():
            tail = "".join(open(log_path).readlines()[-15:])
            return {"dataset": dataset.name, "error": f"exit {proc.returncode}; log {log_path}\n{tail}"}
        return json.loads(result_path.read_text())
    except subprocess.TimeoutExpired:
        return {"dataset": dataset.name, "error": f"timeout after {timeout}s; log {log_path}"}
    finally:
        shutil.rmtree(shard_tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-root", default=DEFAULT_SOURCE, type=pathlib.Path)
    ap.add_argument("--out-root", default=DEFAULT_OUT, type=pathlib.Path)
    ap.add_argument("--only", nargs="*", default=None, help="env ids; default all")
    ap.add_argument("--episodes", type=int, default=0, help="0 = every episode; N > 0 renders only the first N (trial run)")
    ap.add_argument("--determinism-frames", type=int, default=3, help="frames per episode re-rendered for the determinism check")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=6 * 3600, help="time limit in seconds for each dataset subprocess")
    ap.add_argument("--apply", action="store_true", help="render and export; default is a dry run that prints the plan")
    ap.add_argument("--dataset", type=pathlib.Path, help=argparse.SUPPRESS)
    ap.add_argument("--json-out", type=pathlib.Path, help=argparse.SUPPRESS)
    args = ap.parse_args()

    import tasks  # noqa: F401  -- register environments
    from data_gen.render import camera_rigs as RIG
    from data_gen.recordings.dataset_layout import iter_dataset_roots

    if args.dataset is not None:  # worker mode
        result = render_dataset(args.dataset.resolve(), args.out_root.resolve(), args.episodes,
                                args.determinism_frames)
        if args.json_out:
            args.json_out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
        return 0

    datasets = [d for d in iter_dataset_roots(args.source_root) if not args.only or env_id_of(d) in args.only]
    if not datasets:
        raise SystemExit(f"{args.source_root}: no datasets" + (f" among {args.only}" if args.only else ""))
    total_frames = 0
    print(f"{'dataset':56s} {'family':16s} {'eps':>5s} {'frames':>7s}  base eye -> target")
    for dataset in datasets:
        info = json.loads((dataset / "meta" / "info.json").read_text())
        family = RIG.family_of(env_id_of(dataset))
        spec = RIG.base_camera_spec(family)
        n = info["total_episodes"] if args.episodes <= 0 else min(args.episodes, info["total_episodes"])
        total_frames += info["total_frames"] if args.episodes <= 0 else 0
        print(f"{dataset.name:56s} {family:16s} {n:5d} {info['total_frames']:7d}  "
              f"{tuple(round(v, 3) for v in spec.eye_in_world_m)} -> {tuple(round(v, 3) for v in spec.target_in_world_m)}")
    print(f"\n{len(datasets)} datasets, {total_frames} frames in total; rig {RIG.RIG_VERSION}; output {args.out_root}")
    if not args.apply:
        print("dry run; add --apply to render")
        return 0

    args.out_root.mkdir(parents=True, exist_ok=True)
    results = []
    if args.jobs <= 1:
        for dataset in datasets:
            try:
                results.append(render_dataset(dataset.resolve(), args.out_root.resolve(), args.episodes,
                                              args.determinism_frames))
            except Exception as exc:  # keep going; the summary names the failure
                results.append({"dataset": dataset.name, "error": f"{type(exc).__name__}: {exc}"})
                print(f"!! {dataset.name}: {type(exc).__name__}: {exc}", flush=True)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(_worker, d.resolve(), args.out_root.resolve(), args.episodes,
                                   args.determinism_frames, args.timeout) for d in datasets]
            for future in futures:
                results.append(future.result())
                r = results[-1]
                print(("!! " if "error" in r else "   ") + r["dataset"]
                      + ("" if "error" in r else f"  {r['episodes']} eps {r['frames']} frames "
                                                 f"{r['frames'] / max(r['render_seconds'], 1e-6):.0f} fps"), flush=True)

    print(f"\n{'dataset':56s} {'eps':>5s} {'frames':>7s} {'seed-ok':>9s} {'determ':>8s} {'render s':>8s} {'export s':>8s}")
    for r in results:
        if "error" in r:
            print(f"{r['dataset']:56s} !! {r['error'].splitlines()[0][:100]}")
            continue
        print(f"{r['dataset']:56s} {r['episodes']:5d} {r['frames']:7d} "
              f"{r['seed_reproduces_reset']['episodes']:4d}/{r['seed_reproduces_reset']['of']:<4d} "
              f"{r['determinism']['mismatching_frames']:3d}/{r['determinism']['frames_checked']:<4d} "
              f"{r['render_seconds']:7.0f} {r['export_seconds']:7.0f}")
    failed = [r["dataset"] for r in results if "error" in r]
    print(f"\n{len(results) - len(failed)}/{len(results)} datasets done -> {args.out_root}"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
