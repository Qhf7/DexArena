# Replay — validation and release tools

Check a recorded dataset and build the release subsets. Every tool reads the raw recordings under `replay/source_*`, rebuilds the environment from the episode's identity manifest and restores the recorded state into it; nothing here touches the training table under `data/`. Run everything from the repository root in the project's conda environment.

## 1. File structure

```
data_gen/replay/
├── verify_recording_fidelity.py    # restoration check: restored frame == recording
├── replay_success_rate.py          # open-loop replay of the actions (release criterion)
├── batch_compare.py                # side-by-side video of a replay next to its recording
├── check_label_parity.py           # stored success labels vs the current predicates
└── build_replay_success_subset.py  # selects and re-exports the 100 released episodes
```

The restore and replay primitives these tools share live in `data_gen/recordings/replay_recording.py`; the certificate reader and the round-trip check in `data_gen/recordings/lerobot_integrity.py`.

## 2. Data layout

```
data/
├── teleop_pending/<session>/episode_*/       # raw recording (meta, states, actions, ...)
│        │  python -m data_gen.recordings.review_demos --session <name> --mode keep-success
│        v
├── teleop_lerobot/<hand>/<env id>/
│   ├── data/chunk-000/*.parquet              # training table
│   ├── meta/recording_integrity.json         # certificate: fingerprint, trim, reward
│   └── replay/source_<hash>/                 # raw untrimmed copy of every episode
│        │  python -m data_gen.replay.build_replay_success_subset --replay-json ... --apply
│        v
└── teleop_lerobot_100/<hand>/<env id>/   # release: 100 replayable episodes
```

## 3. Validation and release

Four checks, run in this order.

### 3.1 Frame restoration

`verify_recording_fidelity.py` writes each sampled frame into a fresh environment and compares the state, the observation, the task's own quantities and the rendered image with the values stored at record time. It checks the restoration mechanism, not the task logic: the released videos and every evaluation that starts from a demonstration state depend on it.

```bash
# every dataset under data/teleop_lerobot: 2 episodes each, plus one action replay per dataset
python -m data_gen.replay.verify_recording_fidelity --all --episodes 2 --action-replay --jobs 3

# one dataset
python -m data_gen.replay.verify_recording_fidelity --episodes 2 --action-replay \
  data/teleop_lerobot/<hand>/<env id>
```

- positional `roots`: one or more dataset roots (or pending session directories) to check.
- `--all`: instead of `roots`, every dataset under `--lerobot-root` (default `data/teleop_lerobot`).
- `--episodes N` (default 1): the first N episodes of each dataset, in directory order.
- `--stride N` (default 4): restore and compare every Nth frame of each episode.
- `--action-replay`: additionally replay the recorded actions of each checked episode open-loop and report, per episode, the recorded and replayed success, the first step at which the replayed state deviates from the recording by more than 1e-2, and the final deviation (`ep<i>  action replay: <steps> steps, success <recorded> -> <replayed>, first >1e-2 at <step>, final error <e>`). This line is a diagnostic and does not change the verdict; the success rate itself is 3.2.
- `--no-render`: skip the rendered-image comparison, the slow part.
- `--jobs N`: check that many datasets in parallel, one subprocess each.
- `--json-out FILE`: also write every result to a JSON file.

The output has one block per dataset, i.e. per task. Every `ep` line should show non-zero compared counts, e.g. `(state 70, obs 70, info 70, demo-init 70, render 4 compared)`, and the block ends with `=> FAITHFUL`. `=> DEFECT` names the channel and the frames that differ. Quantities that depend on the solver's contact state cannot be reproduced from a restored frame; they are listed as notes, not defects.

### 3.2 Open-loop replay

`replay_success_rate.py` restores the first frame of each kept episode and executes the recorded action stream open-loop in a fresh environment. An episode is replayable when the success condition is reached at any step. This is the release criterion.

```bash
# one dataset
python -m data_gen.replay.replay_success_rate --jobs 3 \
  data/teleop_lerobot/<hand>/<env id>

# every dataset, with the per-episode results saved for the subset builder
python -m data_gen.replay.replay_success_rate --jobs 3 --json-out data/gate_logs/replay.json \
  data/teleop_lerobot/*
```

- positional `sessions`: one or more dataset roots or pending session directories; every kept episode of each is replayed.
- `--jobs N`: split the episodes of each dataset over N worker processes.
- `--json-out FILE`: write the per-episode results (needed by the subset builder).
- `--timeout S` (default 3600): time limit per worker process.
- `--no-fresh-env`: reuse one environment per worker instead of building a fresh one per episode. The default fresh environment is the validated setting; this switch is for diagnosis only.

The output has one line per dataset, i.e. per task: `<env id>  fingerprint match  n=<kept>  ever-success <k>/<n> (<pct>%)  final-success <k>/<n>`. `ever-success` counts episodes where success became true at some step, `final-success` those where it holds on the last step.

The rate is only meaningful with `fingerprint match`; `fingerprint DRIFT` means the dataset was recorded by a different build of that environment. `!! INCOMPLETE (not a success rate)` is not a verdict on the data: with `--jobs` the episodes are spread over worker processes, and if one of them dies (timeout, crash) its episodes were never replayed, so the tool withholds the rate rather than compute it over the survivors, prints the worker's error under the line and exits non-zero. Re-run the dataset; if it happens again, the printed error says why.

`batch_compare.py` renders the replay next to the recording for inspecting a failure:

```bash
# the first 3 episodes of one dataset
python -m data_gen.replay.batch_compare --episodes 3 data/teleop_lerobot/<hand>/<env id>

# every episode of every dataset
python -m data_gen.replay.batch_compare --all --episodes 0
```

- positional `roots` or `--all`: as above.
- `--episodes N` (default 2): the first N episodes of each dataset in directory order; `0` = all.
- `--out DIR` (default `data/replay_compare`): where the videos and `summary.csv` go.
- `--res PX` (default 448), `--stride N` (default 2: render every Nth frame), `--max-steps N` (0 = whole episode): video size and length.
- `--no-warmup`: replay with a cold restore instead of the validated warm-up recipe, to show the difference.

Output: one side-by-side `.mp4` per episode and a `summary.csv` with, per episode, the first hand-object contact step, the first step at which the replay drifts more than 1 mm from the recording, and the final object and joint drift.

### 3.3 Label parity

`check_label_parity.py` is the check to run after a change to a task's success predicates: it restores every recorded frame, re-evaluates it with today's `evaluate()`, and counts the frames whose `success` changed and whether the first success moved. On the build that recorded the data it agrees with 3.1 by construction. The `next.success` and `next.reward` columns derive from these labels.

```bash
python -m data_gen.replay.check_label_parity --episodes 3 \
  data/teleop_lerobot/<hand>/<env id>
```

- positional `roots`: dataset roots, pending session directories or single pending episodes.
- `--episodes N` (default 1): the first N episodes of each root.
- `--stride N` (default 1): evaluate every Nth frame.

The output has one block per dataset, with one line per episode: `ep<i>: <frames> frames, predicates agree`, or `<key> disagrees on <k>/<frames> frames`; each dataset's block ends with `=> OK` or `=> ATTENTION`. After the last dataset one summary line counts the episodes: `[label-parity] <agreeing>/<total> agree with today's predicates`.

### 3.4 Release subset selection

`build_replay_success_subset.py` reads the replay results from 3.2, selects per task the 100 episodes with the shortest completion time (the recorded frame at which success first holds) among the replayable ones, and re-exports them through the same certified path as the full dataset. Each subset episode is byte-identical to the full dataset's, and the subset carries its own certificate and `meta/subset_manifest.json`.

```bash
# selection only, writes nothing
python -m data_gen.replay.build_replay_success_subset --replay-json data/gate_logs/replay.json
# export
python -m data_gen.replay.build_replay_success_subset --apply \
  --replay-json data/gate_logs/replay.json
```

- `--replay-json FILE`: the `--json-out` of `replay_success_rate` over every dataset; it must be complete and fingerprint-matched.
- `--count N` (default 100): episodes per task.
- `--only ENV_ID ...`: restrict to these datasets.
- `--allow-partial`: export a dataset even when fewer than N episodes replay to success.
- `--apply`: write the datasets; without it the run only prints the selection.
- `--mode complement`: export the remaining replayable episodes instead (the ones not in the subset).
- `--source-root DIR`, `--out-root DIR`: input and output roots (defaults `data/teleop_lerobot` and `data/teleop_lerobot_100`).

One line per dataset: `<env id> replayable <n>  chosen 100  completion <min>-<max> frames`, or `replayable <n> <- fewer than 100, skipped`; the run ends with `<N> datasets; <k> written -> data/teleop_lerobot_100`.
