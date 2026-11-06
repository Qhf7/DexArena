# Recordings — review, export and the dataset contract

The tools the released demonstrations were exported and validated with. The demonstrations were recorded by teleoperation, which is not part of this repository; this package starts where the recorder stops: a recording session is reviewed, exported into a LeRobot dataset with an integrity certificate, and can be restored or replayed frame by frame. The validation and release-subset tools in [data_gen/replay](../replay/README.md) and the renderer in [data_gen/render](../render/README.md) build on these modules.

## 1. File structure

```
data_gen/recordings/
├── recording_integrity.py     # the recording contract: schema version, payload validation, hashes
├── episode_store.py           # pending episodes on disk (episode.npz + meta.json) and seed restore
├── dataset_layout.py          # where pending sessions and exported datasets live, per end effector
├── lerobot_integrity.py       # dataset certificate: writing, round trip, video PSNR
├── review_demos.py            # reviews pending episodes and exports a LeRobot dataset
├── replay_recording.py        # restore and replay primitives used by data_gen/replay and data_gen/render
└── replay_compare_video.py    # side-by-side video of a recording and its replay
```

## 2. Review and export

A recording session is a directory `data/teleop_pending/session_<time>_<env id>/episode_*/`; each episode holds the recorded states, actions, infos and seed exactly as the recorder wrote them (`episode.npz` + `meta.json`), with a preview `.mp4`.

```bash
python -m data_gen.recordings.review_demos --mode keep-success --session latest
```

- `--session`: which `session_*` folder under `data/teleop_pending` to review. Default `latest`; `all` takes every session.
- `--mode`: `keep-success` keeps the episodes the environment marked successful; `interactive` shows each clip's path and asks keep / discard; `keep-all` and `discard-all` skip the prompt.
- `--reward-from-success` (default on): `next.reward` is 1 on every step where the success condition holds, else 0, the same definition for every task. `--no-reward-from-success` exports the reward the environment returned at record time instead. A dataset is either entirely relabeled or not; appending the other kind is refused.
- `--trim-start-frames` (default 3): drops the first frames of every episode, which hold the record-arm snap.
- `--overwrite`: recreate the target dataset instead of appending to it.
- `--dry-run`: print the keep / discard decisions and write nothing.

The dataset is written to `data/teleop_lerobot/<hand>/<env id>/`. `data/` holds the training table. `meta/recording_integrity.json` is the certificate: the code fingerprint of the environment build that recorded the episodes, the trim, the reward definition, a hash of every file and the result of the round-trip check between the table and the raw recording; replay and append refuse a dataset whose certificate does not match. `replay/source_<hash>/` keeps the raw untrimmed recording of every episode (states, actions, infos, seed) exactly as the recorder wrote it; the validation, subset and rendering tools work from these, not from the training table under `data/`. Each frame of the table carries:


| Column                          | Content                                                                                                                                                                                                                                   |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `observation.state`             | what a policy observes: robot joint positions and velocities, then the task's own quantities (object poses, stage flags)                                                                                                                  |
| `action`                        | absolute joint targets, 29 (single arm) or 58 (dual arm)                                                                                                                                                                                  |
| `next.reward`                   | 1 while the success condition holds, else 0 (`--reward-from-success`)                                                                                                                                                                     |
| `next.success`, `next.done`     | success and episode-end flags                                                                                                                                                                                                             |
| `observation.environment_state` | the full simulator state, flattened: the episode step counter, then pose and velocity of every body, the joint state of every articulation and the task's latched state; the entry names are in `meta/info.json`; used to restore a frame |


Validation (restoration check, open-loop replay rate) and the 100-episode release subsets: [data_gen/replay/README.md](../replay/README.md). Offline multi-view rendering: [data_gen/render/README.md](../render/README.md).