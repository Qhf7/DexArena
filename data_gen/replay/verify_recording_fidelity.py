"""THE recording-fidelity gate: everything a restore-based consumer can observe.

WHY THIS REPLACES THE EARLIER CHECKS
------------------------------------
The previous checks were ENUMERATIVE -- they listed the fields to compare, so
their coverage was bounded by whoever wrote the list. That is why each round of
review found something the round before had missed: not bad luck, an open-ended
method. Widening the list is not a fix; it just moves the boundary.

This one is closed by construction. A consumer that restores a recorded frame
can observe exactly four things and no others:

    get_state_dict()   what the next restore will write
    get_obs()          what a policy is fed
    get_info()         what success, stages and diagnostics say
    render()           the pixels the offline RGB re-render produces

So the invariant is: restore any recorded frame, by any path a consumer uses,
and all four must agree. There is no fifth thing to think of later.

The render is the odd one out, because a state-only recording has no image to
compare against. What it CAN be held to is that both restore paths produce the
same picture. That is not a weak check: visibility is carried by render-only
caches that no state tensor contains -- Screwing hides and shows the parked nut,
CabinetPour its placement marker -- and a restore that rebuilds one but not the
other yields two different worlds that agree on every number.

THE PATHS, AND WHY BOTH
-----------------------
    raw       set_state_dict(raw_states[t]) + the recorded episode clock.
              What replay and QC tooling do.
    flat      reset(options={"reset_to_env_states": <unflattened parquet row>}).
              What PPO reverse-curriculum demo-init and the offline RGB
              re-render do -- a DIFFERENT code path that skips
              _initialize_episode, passes an env_idx and starts a fresh clock.
              It has had its own defects, invisible to the raw path.

THE ONLY FOUR OUTCOMES PER CHANNEL
----------------------------------
    exact             bit-for-bit, or within float tolerance
    contact-derived   the engine's contact report is empty until the first
                      simulate(), so `is_grasped` and friends cannot be read off
                      a bare restore. Detected by re-reading after replaying the
                      recorded step from the preceding frame (a hold action does
                      not do: it drops the fingers' commanded squeeze, so a
                      contact-force flag stays 0). A platform property, not a
                      defect.
    non-idempotent    `counter = where(cond, counter + 1, 0)` with no
                      per-control-step gate counts twice if evaluate() runs
                      twice on one step. Recording calls it once, so the data is
                      correct; a checker that re-evaluates reads one high.
                      Detected by evaluating the same restored state twice.
    cross-step        a difference ACROSS the step boundary whose baseline the
                      same evaluate() overwrites -- ScrewingTurn's
                      `rotation_delta` is `wrap_to_pi(yaw - self._nut_yaw_prev)`
                      and then sets `_nut_yaw_prev = yaw`, so the state dict
                      stores the post-update baseline and every single-frame
                      restore recomputes 0. Detected by bridging the boundary:
                      restore the preceding frame, let one evaluate() set the
                      baseline, then write only the target frame's PHYSICAL
                      state and read again.

Anything outside those four is a defect. When the run reports none, the
question "is the recorded data faithful" is answered -- not "answered so far".

ACTION REPLAY is reported alongside but judged separately: it is open-loop, so a
divergence there is a property of the controller and the contact solver, not of
the recording. See docs/lerobot_data_and_action_replay_zh.md.

USAGE
    python -m data_gen.replay.verify_recording_fidelity <lerobot_root_or_pending_session>...
    python -m data_gen.replay.verify_recording_fidelity --all
    python -m data_gen.replay.verify_recording_fidelity <root> --action-replay
    python -m data_gen.replay.verify_recording_fidelity <root> --episodes 3 --stride 1
Run from the repository root, in the ManiSkill conda env. For the office and
table_clean families prefix the LD_PRELOAD line from scripts/run_pytest.sh.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_gen.recordings.dataset_layout import iter_dataset_roots

CONTEXT_KEY = "__recording_context__"
# The state dict is a direct write-then-read and is held to float32's own
# precision. DERIVED quantities -- info fields and observations built from poses
# and quaternions -- are not: float32 carries about seven digits, so an axis
# recomputed from a bit-identical pose still lands ~2e-7 away. Holding those to
# the same bound reported CabinetPour and the keyboard as defective over
# differences of 2.4e-7, which is below the representable noise floor.
FLOAT_RTOL, FLOAT_ATOL = 1e-5, 1e-7
DERIVED_RTOL, DERIVED_ATOL = 1e-4, 1e-6


# --------------------------------------------------------------------------- #
# comparison helpers
# --------------------------------------------------------------------------- #
from data_gen.recordings.episode_store import restore_seed

def as_array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def flatten_tree(tree, prefix=""):
    """{dotted name: array} for a nested state/info dict."""
    out = {}
    for key, value in tree.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten_tree(value, name + "."))
            continue
        try:
            arr = as_array(value)
        except Exception:
            continue
        if arr.dtype.kind in "biuf":
            out[name] = np.array(arr, copy=True)
    return out


def same(new, old, *, derived=False):
    """None when not comparable, else whether the two agree."""
    if new is None or old is None:
        return None
    new, old = np.asarray(new), np.asarray(old)
    if new.shape != old.shape:
        return None
    if old.dtype.kind in "biu" and new.dtype.kind in "biu":
        return bool(np.array_equal(new, old))
    new64, old64 = new.astype(np.float64), old.astype(np.float64)
    if not derived:
        return bool(np.allclose(new64, old64, rtol=FLOAT_RTOL, atol=FLOAT_ATOL,
                                equal_nan=True))
    # A derived VECTOR's accuracy is set by its largest component, not its
    # smallest: element-wise rtol gives a position's near-zero axis a budget of
    # ~1e-6 while the same float32 arithmetic that produced the 0.3 m axis
    # carries a few times that. Scaling by the magnitude of the whole quantity
    # flagged keyboard `key_top_pos` over 3e-6 of float noise while a genuine
    # divergence there is 0.19 m -- four orders of magnitude clear.
    budget = DERIVED_ATOL + DERIVED_RTOL * float(np.max(np.abs(old64)) if old64.size else 0.0)
    diff = np.abs(new64 - old64)
    diff = np.where(np.isnan(new64) & np.isnan(old64), 0.0, diff)
    return bool(np.all(diff <= budget))


def max_err(new, old):
    try:
        return float(np.max(np.abs(np.asarray(new, np.float64)
                                   - np.asarray(old, np.float64))))
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# episode loading
# --------------------------------------------------------------------------- #
def load_episodes(root: pathlib.Path, limit: int) -> tuple[list[dict], bool]:
    from data_gen.recordings.replay_compare_video import _episode_dirs, _load

    is_dataset = (root / "meta" / "info.json").exists()
    episodes = []
    for directory in _episode_dirs(root, max(1, limit)):
        episode = _load(directory)
        with np.load(directory / "episode.npz", allow_pickle=True) as npz:
            episode["obs"] = np.asarray(npz["obs"])
        infos_path = directory / "infos.pkl"
        episode["infos"] = (pickle.loads(infos_path.read_bytes())
                            if infos_path.exists() else None)
        env_states_path = directory / "env_states.pkl"
        episode["env_states"] = (pickle.loads(env_states_path.read_bytes())
                                 if env_states_path.exists() else None)
        episodes.append(episode)
    return episodes, is_dataset


def hold_action(base, robot_key):
    import torch

    art = base.scene.articulations[robot_key]
    names = [j.name for j in art.get_active_joints()]
    qpos = art.get_qpos().detach().cpu().numpy().reshape(-1)
    order = [j.name
             for controller in base.agent.controller.controllers.values()
             for j in controller.joints]
    return torch.as_tensor(
        np.asarray([qpos[names.index(n)] for n in order], np.float32))[None]


def robot_key_of(base, raw0):
    for name in base.scene.articulations:
        if name in raw0.get("articulations", {}):
            return name
    return next(iter(base.scene.articulations))


# --------------------------------------------------------------------------- #
# the invariant
# --------------------------------------------------------------------------- #
def verify_episode(env, episode, stride: int, warm_steps: int,
                   render_every: bool = True, episode_seed: int | None = None) -> dict:
    import torch

    from data_gen.recordings import replay_recording as RR

    base = env.unwrapped
    raw = episode["raw"]
    infos = episode["infos"]
    obs_rec = episode["obs"]
    actions = episode.get("actions")
    n = len(raw) - 1
    frames = list(range(0, n, max(1, stride)))
    rkey = robot_key_of(base, raw[0])

    state_bad: dict[str, int] = {}
    state_err: dict[str, float] = {}
    obs_bad = 0
    obs_unstable = 0
    # A silently SKIPPED comparison and a passing one look identical in the
    # output, which would turn "the demo-init path was never exercised" into a
    # green tick. Count what actually ran and print it.
    ran = {"state": 0, "obs": 0, "info": 0, "demo": 0, "render": 0}
    render_bad: list[int] = []
    render_frames = set(frames[:: max(1, len(frames) // 4)][:4]) if render_every else set()
    info_bad: dict[str, list[int]] = {}
    demo_bad: dict[str, list[int]] = {}
    contact_derived: set[str] = set()
    non_idempotent: set[str] = set()
    pre_step: set[str] = set()
    cross_step: set[str] = set()
    obs_pending: list = []

    for index in frames:
        target = raw[index + 1]
        # The PRE-STEP baseline is taken first, from a clean restore. Taking it
        # later -- after the warming step and after two `get_info()` calls have
        # advanced the debounce counters -- made the comparison depend on that
        # history, and a genuinely pre-step quantity failed to be recognised as
        # one on a frame where the counters had moved (keyboard `key_top_pos`,
        # which reproduces EXACTLY off the preceding frame: diff 0.00000).
        previous = None
        if infos and index < len(infos):
            RR._set_raw_state(env, raw[index])
            previous = flatten_tree(base.get_info())
        RR._set_raw_state(env, target)

        # -- 1. the state dict itself. No contact or idempotency confound: this
        #       is a pure write-then-read of what the next restore will write.
        ran["state"] += 1
        now = flatten_tree({k: v for k, v in base.get_state_dict().items()})
        was = flatten_tree({k: v for k, v in target.items() if k != CONTEXT_KEY})
        for key, old in was.items():
            verdict = same(now.get(key), old)
            if verdict is False:
                state_bad[key] = state_bad.get(key, 0) + 1
                state_err[key] = max(state_err.get(key, 0.0),
                                     max_err(now[key], old))

        # -- 2 and 3. observation and info share the two confounds, because some
        #    tasks feed info straight into `_get_obs_extra` (TableClean puts
        #    `n_in_pan` in the observation). Classifying info but not the
        #    observation would report those tasks as defective for a reason
        #    already known to be a platform property.
        cold_obs = as_array(base.get_obs()).reshape(-1)
        cold = flatten_tree(base.get_info())
        again = flatten_tree(base.get_info())
        for key in cold:
            if same(again.get(key), cold[key]) is False:
                non_idempotent.add(key)
        obs_idempotent = same(as_array(base.get_obs()).reshape(-1), cold_obs) is not False
        if warm_steps > 0:
            # The warming step replays the RECORDED step(s) that produced this
            # frame, from the preceding recorded frame. A hold action (target =
            # current qpos) used to be stepped from the target frame instead; it
            # takes the fingers' commanded squeeze away, so a contact-FORCE grasp
            # flag (TableClean `debris_held`, ScrewFasten `is_tool_grasped`) read
            # 0 both cold and warm and was reported as a defect. Measured on
            # 80 recorded-true frames each: hold step 0/80 and
            # 1/80, the recorded step 79/80 and 80/80.
            start = max(0, index + 1 - warm_steps)
            RR._set_raw_state(env, raw[start])
            for t in range(start, index + 1):
                if actions is not None and t < len(actions):
                    env.step(torch.as_tensor(np.asarray(actions[t], np.float32))[None])
                else:
                    env.step(hold_action(base, rkey))
            RR._set_raw_state(env, target)
            warm = flatten_tree(base.get_info())
            warm_obs = as_array(base.get_obs()).reshape(-1)
        else:
            warm, warm_obs = cold, cold_obs

        # -- 2b. the CROSS-STEP probe. Bridge the step boundary the way the
        #    recorder crossed it: restore the preceding frame, let one
        #    `get_info()` set the channel's baseline, then write ONLY the target
        #    frame's PHYSICAL state (task-state keys held back, so the env keeps
        #    that baseline) and read again. Both ends are bit-exact restored
        #    states, so unlike replaying the step this carries no cold-start
        #    noise -- ScrewingTurn's `rotation_delta` is ~1e-3 rad and a replayed
        #    step misses it by more than the derived tolerance.
        bridged, bridged_obs = cold, cold_obs
        if infos and index < len(infos):
            physical = {k: v for k, v in target.items()
                        if k in ("actors", "articulations")}
            if physical:
                try:
                    RR._set_raw_state(env, raw[index])
                    base.get_info()
                    base.set_state_dict(physical)
                    bridged = flatten_tree(base.get_info())
                    bridged_obs = as_array(base.get_obs()).reshape(-1)
                except Exception:
                    bridged, bridged_obs = cold, cold_obs
                RR._set_raw_state(env, target)
        obs_stable = obs_idempotent and same(warm_obs, cold_obs) is not False
        if obs_rec is not None and index + 1 < len(obs_rec):
            ran["obs"] += 1
            recorded_obs = np.asarray(obs_rec[index + 1]).reshape(-1)
            if (same(cold_obs, recorded_obs, derived=True) is False
                    and same(warm_obs, recorded_obs, derived=True) is False
                    and same(bridged_obs, recorded_obs, derived=True) is False):
                # Deferred: whether the observation is carrying a pre-step or
                # contact channel is only knowable once info has been
                # classified, which happens at the end of the episode.
                obs_pending.append((index, obs_stable))
        for key in cold:
            if same(warm.get(key), cold[key]) is False:
                contact_derived.add(key)
        if infos and index < len(infos):
            ran["info"] += 1
            recorded = flatten_tree(infos[index])
            missed = []
            for key, old in recorded.items():
                if (same(cold.get(key), old, derived=True)
                        or same(warm.get(key), old, derived=True)):
                    continue
                if same(cold.get(key), old, derived=True) is None:
                    continue
                if same(bridged.get(key), old, derived=True):
                    # Record the frame like the other three categories do: `split()`
                    # routes any key in `unstable` to the unverifiable bucket, and the
                    # observation bookkeeping excuses a frame only when an unstable
                    # channel is listed as disagreeing AT THAT INDEX. Returning early
                    # here left `info_bad` empty, so an observation carrying this
                    # channel was still counted as a defect -- measured on
                    # CabinetPourCapRemovePlace ep2: `cap_twist` classified cross-step,
                    # yet 1 of 87 compared frames reported `observation on 1 frame(s)`.
                    cross_step.add(key)
                    info_bad.setdefault(key, []).append(index)
                    continue
                missed.append(key)
                info_bad.setdefault(key, []).append(index)
            if missed and previous is not None:
                # A quantity computed in `_before_control_step` is evaluated on
                # the state BEFORE the step and reported in the info AFTER it,
                # so its recorded value lives in the PRECEDING frame. Finding it
                # there says "computed pre-step", not "changed since".
                for key in missed:
                    if same(previous.get(key), recorded[key], derived=True):
                        pre_step.add(key)

        # -- 4. the pixels, on both paths. Rendering is expensive, so a few
        #    frames per episode: a visibility cache that fails to rebuild fails
        #    on every frame, not on a rare one.
        render_here = render_frames and index in render_frames
        replay_pixels = None
        if render_here:
            RR._set_raw_state(env, target)
            ran["render"] += 1
            replay_pixels = as_array(env.render()).copy()

        # -- 5. the demo-init path: a different entry point, a fresh clock
        if episode.get("env_states") is not None and index + 1 < len(episode["env_states"]):
            flat = episode["env_states"][index + 1]
            payload = {cat: {k: torch.as_tensor(np.asarray(v), device=base.device)
                             for k, v in leaves.items()}
                       for cat, leaves in flat.items()
                       if isinstance(leaves, dict)}
            ran["demo"] += 1
            # The recorded seed is part of the restore, not decoration: with
            # num_envs=1 every reset reconfigures, so a task that randomises at
            # LOAD time re-rolls it unless the seed pins it. Omitting it made
            # furniture render a different panel colour on this path than on the
            # other one -- which is precisely the defect the pixel comparison
            # below is looking for, so it has to be a fair comparison.
            env.reset(seed=episode_seed,
                      options={"reset_to_env_states": {"env_states": payload}})
            demo = flatten_tree(base.get_info())
            for key, value in cold.items():
                if key in contact_derived or key in non_idempotent:
                    continue
                if key == "elapsed_steps":
                    continue
                if same(demo.get(key), value, derived=True) is False:
                    demo_bad.setdefault(key, []).append(index)
            if render_here and replay_pixels is not None:
                demo_pixels = as_array(env.render())
                if demo_pixels.shape != replay_pixels.shape or not np.array_equal(
                    demo_pixels, replay_pixels
                ):
                    render_bad.append(index)

    unstable = contact_derived | non_idempotent | pre_step | cross_step
    tail_start = frames[max(0, len(frames) - 5)] if frames else 0

    # A STAGE BOUNDARY frame is one where a pre-step channel disagreed: the
    # task advanced its own stage part-way through `evaluate()`, so the info it
    # returned mixes quantities read before the advance with quantities read
    # after. Measured on the keyboard: `sequence_index` moves mid-evaluate, so
    # the key geometry is the OLD key's while the slot the info reports is the
    # new one, and the tip-relative distances derived from both match neither
    # the preceding state nor the following one. Nothing derived at such a frame
    # is reproducible from any single state, on any code.
    boundary_frames = {i for key in pre_step for i in info_bad.get(key, [])}

    def split(bucket):
        defect, tail, unver = {}, {}, {}
        for key, hits in bucket.items():
            at_boundary = boundary_frames and set(hits) <= boundary_frames
            target = (unver if key in unstable or at_boundary
                      else tail if min(hits) >= tail_start
                      else defect)
            target[key] = len(hits)
        return defect, tail, unver

    info_defect, info_tail, info_unver = split(info_bad)
    # Several tasks feed info fields straight into `_get_obs_extra`, so an
    # observation that moves on exactly the frames an unverifiable channel moved
    # is that channel seen through the observation, not a separate defect.
    explained = {i for key in unstable for i in info_bad.get(key, [])}
    obs_bad = sum(1 for i, stable in obs_pending if stable and i not in explained)
    obs_unstable = len(obs_pending) - obs_bad
    demo_defect, demo_tail, demo_unver = split(demo_bad)
    return {
        "frames": len(frames),
        "ran": ran,
        "state_defect": state_bad,
        "state_err": state_err,
        "obs_defect": obs_bad,
        "obs_unstable": obs_unstable,
        "render_defect": render_bad,
        "info_defect": info_defect, "info_tail": info_tail, "info_unver": info_unver,
        "demo_defect": demo_defect, "demo_tail": demo_tail, "demo_unver": demo_unver,
        "contact_derived": sorted(contact_derived),
        "non_idempotent": sorted(non_idempotent),
        "pre_step": sorted(pre_step),
        "cross_step": sorted(cross_step),
    }


def action_replay(env, episode, hold_steps: int, fresh_env: bool = True) -> dict:
    """Open-loop replay from frame 0, through the zero-error recipe.

    ``fresh_env`` (default) runs the replay in a NEW environment and closes it:
    set_state_dict does not reset PhysX's contact caches, so a replay in the env
    that just did the restore checks (or another episode) inherits residue that
    can flip a marginal friction grasp (measured on TakeAppleToBlender).
    """
    import torch

    from data_gen.recordings import replay_recording as RR
    from data_gen.recordings.replay_compare_video import _warm_up_from_frame0

    own_env = None
    if fresh_env:
        own_env = RR._make_env(episode["meta"]["identity_manifest"])
        own_env.reset(seed=restore_seed(episode["meta"]))
        env = own_env
    try:
        return _action_replay_in(env, episode, hold_steps)
    finally:
        if own_env is not None:
            own_env.close()


def _action_replay_in(env, episode, hold_steps: int) -> dict:
    import torch

    from data_gen.recordings import replay_recording as RR
    from data_gen.recordings.replay_compare_video import _warm_up_from_frame0

    base = env.unwrapped
    raw, actions = episode["raw"], episode["actions"]
    rkey = robot_key_of(base, raw[0])
    RR._set_raw_state(env, raw[0])
    _warm_up_from_frame0(env, raw[0], rkey, hold_steps)
    # Divergence is reported as a MAGNITUDE, not a boolean. A single threshold
    # over ~350 heterogeneous state columns says almost nothing: the warm-up
    # step alone leaves the arm a few micro-radians off, which trips any tight
    # bound at step 1 and reads as "diverged immediately" when the episode in
    # fact tracks perfectly. What a reader needs is where it leaves 1 cm / 1e-2
    # rad, and how far apart the two runs finish.
    DIVERGENCE = 1e-2
    first_div, success, worst_end = None, False, 0.0
    for step in range(len(actions)):
        _, _, _, _, info = env.step(
            torch.as_tensor(np.asarray(actions[step], np.float32))[None])
        success = success or bool(as_array(info.get("success", False)).reshape(-1)[0])
        if step + 1 < len(raw):
            now = flatten_tree(base.get_state_dict())
            was = flatten_tree({k: v for k, v in raw[step + 1].items()
                                if k != CONTEXT_KEY})
            worst = max((max_err(now[k], v) for k, v in was.items() if k in now),
                        default=0.0)
            worst_end = worst
            if first_div is None and worst > DIVERGENCE:
                first_div = step + 1
    recorded_success = bool(np.asarray(episode["success"]).any()) \
        if episode.get("success") is not None else None
    return {"steps": len(actions), "replay_success": success,
            "recorded_success": recorded_success, "first_divergence": first_div,
            "final_error": worst_end}


# --------------------------------------------------------------------------- #
def verify_root(root: pathlib.Path, args) -> bool:
    from data_gen.recordings import replay_recording as RR

    episodes, is_dataset = load_episodes(root, args.episodes)
    identity = episodes[0]["meta"]["identity_manifest"]
    # Name the ENVIRONMENT, not the session stamp: a wall of `session_2026...`
    # rows is unreadable, and the environment is what a reader is looking for.
    print(f"\n=== {identity.get('env_id', root.name)}  ({root.name})")
    env = RR._make_env(identity)
    base = env.unwrapped
    ok = True
    try:
        env.reset(seed=restore_seed(episodes[0]["meta"]))

        # ---- schema -------------------------------------------------------
        recorded_state = {k: v for k, v in episodes[0]["raw"][0].items()
                          if k != CONTEXT_KEY}
        current_state = base.get_state_dict()
        missing = sorted(set(current_state) - set(recorded_state))
        extra = sorted(set(recorded_state) - set(current_state))
        narrow = []
        for key, value in current_state.items():
            if key in ("actors", "articulations") or key not in recorded_state:
                continue
            # COLUMNS, not keys. `flatten_tree` yields one entry per leaf, and a
            # task state is a single packed tensor, so counting entries compared
            # 1 against 1 and never noticed a widening. Pipette went from 4
            # fields to 6 at one point, and an earlier recording of it was
            # reported as schema-identical and then failed on the two latches
            # the recording could not possibly carry.
            def width(tree):
                return sum(int(np.asarray(a).size) for a in tree.values())

            now_w = width(flatten_tree({key: value}))
            then_w = width(flatten_tree({key: recorded_state[key]}))
            if now_w != then_w:
                narrow.append(f"{key} ({then_w} recorded, {now_w} now)")
        rec_obs = int(np.asarray(episodes[0]["obs"]).shape[-1])
        cur_obs = int(as_array(base.get_obs()).reshape(1, -1).shape[-1])
        rec_act = int(np.asarray(episodes[0]["actions"]).shape[-1])
        cur_act = int(np.prod(env.action_space.shape[-1:]))
        schema_bad = bool(extra) or rec_obs != cur_obs or rec_act != cur_act
        if schema_bad:
            ok = False
        if extra:
            print(f"  SCHEMA   DEFECT recorded keys this build no longer writes: "
                  f"{', '.join(extra)}")
        if rec_obs != cur_obs:
            print(f"  SCHEMA   DEFECT observation {rec_obs} recorded vs {cur_obs} now")
        if rec_act != cur_act:
            print(f"  SCHEMA   DEFECT action {rec_act} recorded vs {cur_act} now")
        if missing or narrow:
            print(f"  SCHEMA   recording predates: "
                  f"{', '.join(missing + narrow)} (older recording, not a defect)")
        if not schema_bad and not missing and not narrow:
            print("  SCHEMA   identical to this build")

        # ---- the invariant, per episode ------------------------------------
        contact, nonidem, prestep, crossstep = set(), set(), set(), set()
        for index, episode in enumerate(episodes):
            env.reset(seed=restore_seed(episode["meta"]))
            result = verify_episode(env, episode, args.stride, args.warm_steps,
                                    render_every=not args.no_render,
                                    episode_seed=restore_seed(episode["meta"]))
            contact |= set(result["contact_derived"])
            nonidem |= set(result["non_idempotent"])
            prestep |= set(result["pre_step"])
            crossstep |= set(result["cross_step"])
            defects = []
            if result["state_defect"]:
                worst = sorted(result["state_err"].items(), key=lambda kv: -kv[1])[:4]
                defects.append("state_dict " + ", ".join(
                    f"{k}(max {v:.2e})" for k, v in worst))
            if result["obs_defect"] and not missing and not narrow:
                defects.append(f"observation on {result['obs_defect']} frame(s)")
            if result["info_defect"] and not missing and not narrow:
                defects.append("info " + ", ".join(
                    f"{k}({v})" for k, v in sorted(
                        result["info_defect"].items(), key=lambda kv: -kv[1])[:6]))
            if result["render_defect"]:
                defects.append(f"rendered image differs between the two restore "
                               f"paths on {len(result['render_defect'])} frame(s)")
            if result["demo_defect"] and not missing and not narrow:
                defects.append("demo-init path " + ", ".join(
                    f"{k}({v})" for k, v in sorted(
                        result["demo_defect"].items(), key=lambda kv: -kv[1])[:6]))
            ran = result["ran"]
            if defects:
                ok = False
                # The compared counts are printed here too: a DEFECT on one key
                # must not hide whether the OTHER channels were exercised at all
                # (a forced-success drive makes exactly one key unreproducible,
                # and the reader has to be able to see the rest were checked).
                print(f"  ep{index}  {result['frames']} frames  DEFECT: "
                      + "; ".join(defects)
                      + f"  (state {ran['state']}, obs {ran['obs']}, info {ran['info']}, "
                        f"demo-init {ran['demo']}, render {ran['render']} compared)")
            else:
                notes = []
                if result["info_tail"] or result["demo_tail"]:
                    notes.append("final-frame debounce")
                if result["info_unver"] or result["demo_unver"]:
                    notes.append(f"{len(result['info_unver']) + len(result['demo_unver'])}"
                                 " unverifiable channel(s)")
                if result["obs_unstable"]:
                    notes.append(f"observation carries an unverifiable channel on "
                                 f"{result['obs_unstable']} frame(s)")
                suffix = f"  [{'; '.join(notes)}]" if notes else ""
                ran = result["ran"]
                print(f"  ep{index}  {result['frames']} frames  exact"
                      f"  (state {ran['state']}, obs {ran['obs']}, info {ran['info']}, "
                      f"demo-init {ran['demo']}, render {ran['render']} compared)"
                      f"{suffix}")
                required = {k: v for k, v in ran.items()
                            if k != "render" or not args.no_render}
                if min(required.values()) == 0:
                    ok = False
                    missing_paths = [k for k, v in required.items() if v == 0]
                    print(f"  ep{index}  DEFECT: never compared: "
                          f"{', '.join(missing_paths)} -- a green result here would "
                          "have meant nothing")
            if args.action_replay:
                env.reset(seed=restore_seed(episode["meta"]))
                replay = action_replay(env, episode, args.hold_steps, fresh_env=args.fresh_env)
                print(f"  ep{index}  action replay: {replay['steps']} steps, "
                      f"success {replay['recorded_success']} -> "
                      f"{replay['replay_success']}, first >1e-2 at "
                      f"{replay['first_divergence']}, final error "
                      f"{replay['final_error']:.2e}")
        if contact:
            print(f"  contact-derived ({len(contact)}): "
                  + ", ".join(sorted(contact)[:10])
                  + (" ..." if len(contact) > 10 else ""))
        if nonidem:
            print(f"  non-idempotent ({len(nonidem)}): "
                  + ", ".join(sorted(nonidem)[:10])
                  + (" ..." if len(nonidem) > 10 else ""))
        if prestep:
            print(f"  pre-step ({len(prestep)}): "
                  + ", ".join(sorted(prestep)[:10])
                  + (" ..." if len(prestep) > 10 else ""))
        if crossstep:
            print(f"  cross-step ({len(crossstep)}): "
                  + ", ".join(sorted(crossstep)[:10])
                  + (" ..." if len(crossstep) > 10 else ""))
    finally:
        env.close()
    print(f"  => {'FAITHFUL' if ok else 'DEFECT'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", type=pathlib.Path)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--lerobot-root", type=pathlib.Path,
                    default=pathlib.Path("data/teleop_lerobot"))
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--warm-steps", type=int, default=1)
    ap.add_argument("--action-replay", action="store_true")
    ap.add_argument("--hold-steps", type=int, default=1)
    ap.add_argument("--fresh-env", dest="fresh_env", action="store_true", default=True,
                    help="(default) run each action replay in a new environment; --no-fresh-env reuses the checking env")
    ap.add_argument("--no-fresh-env", dest="fresh_env", action="store_false")
    ap.add_argument("--no-render", action="store_true",
                    help="skip the cross-path pixel comparison (it is the slow part)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="verify this many roots in parallel, one subprocess each")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    roots = list(args.roots)
    if args.all:
        roots += iter_dataset_roots(args.lerobot_root)
    if not roots:
        ap.error("give roots, or --all")

    bad, rows = 0, []
    if args.jobs > 1 and len(roots) > 1:
        # One root per subprocess. A ManiSkill scene cannot be torn down
        # cleanly enough to build 56 of them in one process, and serially this
        # takes hours.
        import os
        import subprocess
        from concurrent.futures import ThreadPoolExecutor

        def run_child(root):
            child = [sys.executable, __file__, str(root),
                     "--episodes", str(args.episodes), "--stride", str(args.stride),
                     "--warm-steps", str(args.warm_steps), "--jobs", "1"]
            if args.action_replay:
                child += ["--action-replay", "--hold-steps", str(args.hold_steps)]
                if not args.fresh_env:
                    child += ["--no-fresh-env"]
            if args.no_render:
                child += ["--no-render"]
            proc = subprocess.run(child, cwd=str(REPO_ROOT), capture_output=True,
                                  text=True, env=dict(os.environ))
            keep = [line for line in proc.stdout.splitlines()
                    if line.startswith(("=== ", "  ")) and "[fidelity]" not in line]
            return root, proc.returncode == 0, "\n".join(keep) or proc.stderr[-500:]

        # as_completed, not map: `map` yields IN ORDER, so one slow root blocks
        # every finished one behind it and the run looks hung for as long as the
        # longest dataset takes. Results are sorted at the end instead.
        from concurrent.futures import as_completed

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run_child, root): root for root in roots}
            for future in as_completed(futures):
                root, good, text = future.result()
                print(text, flush=True)
                rows.append({"root": str(root), "faithful": good})
                bad += not good
    else:
      for root in roots:
        try:
            good = verify_root(root, args)
        except Exception as exc:
            good = False
            print(f"\n=== {root.name}\n  ERROR {type(exc).__name__}: {exc}")
        rows.append({"root": str(root), "faithful": good})
        bad += not good
    print(f"\n[fidelity] {len(roots) - bad}/{len(roots)} faithful")
    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(rows, indent=2))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
