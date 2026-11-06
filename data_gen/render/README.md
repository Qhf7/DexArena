# Render — offline multi-view rendering

Turn a state-only LeRobot dataset into the same dataset plus one video feature per camera. Every recorded frame is restored into a fresh environment and captured by the family camera rig: one fixed scene camera (1.08 m above the table, tilted 20 degrees) and one wrist camera per arm in use, 256x256, 90 degree vertical field of view, guidance markers hidden. No physics step and no action replay are involved; the state columns are copied byte for byte. Run from the repository root in the project's conda environment.

## 1. File structure

```
data_gen/render/
├── camera_rigs.py               # the camera rig: scene camera and one wrist camera per arm
└── render_multiview_videos.py   # the batch renderer
```

## 2. Usage

```bash
# plan only: datasets, episodes, frames and the rig per dataset (writes nothing)
python -m data_gen.render.render_multiview_videos

# trial: two episodes of one dataset into a scratch directory
python -m data_gen.render.render_multiview_videos --apply --episodes 2 \
  --only OfficeDeskClickMouse-Sharpa-v1 --out-root /tmp/render_trial

# every dataset under the source root, three in parallel
python -m data_gen.render.render_multiview_videos --apply --jobs 3
```

- `--source-root DIR` (default `data/teleop_lerobot_100`): the state-only datasets to render.
- `--out-root DIR` (default `data/teleop_lerobot_100_video`): where the rendered datasets go.
- `--only ENV_ID ...`: restrict to these datasets; default all under the source root.
- `--episodes N` (default 0 = every episode): render only the first N episodes of each dataset, for a trial run.
- `--determinism-frames N` (default 3): frames per episode re-rendered a second time for the pixel-identity check.
- `--jobs N` (default 1): datasets rendered in parallel, one subprocess each.
- `--timeout S` (default 21600): time limit per dataset subprocess.
- `--apply`: render and export; without it the run only prints the plan.

Output: `data/teleop_lerobot_100_video/<hand>/<env id>/` with the cameras as `observation.images.<camera>` video features (AV1), the source's state columns, a new certificate, and `meta/video_manifest.json` with the per-dataset counts: rendered frames vs recorded states, re-rendered frames that are pixel-identical, and the decoded-video PSNR check (25 dB floor; the certificate records the minimum per camera).
