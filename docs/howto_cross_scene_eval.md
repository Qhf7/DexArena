# Shared cross-scene evaluation

Use one environment factory for the 30 canonical subtasks and three native scene
surfaces. Algorithms retain their own checkpoints, action processing, and rollout loops.

| `scene_background` | Native surface |
| --- | --- |
| `robocasa_l3_s2` | Blue kitchen stone island |
| `robocasa_l3_s1` | Light wood kitchen island |
| `robocasa_l3_s6` | Warm wood kitchen stone island (candidate E) |

The optional [LabUtopia laboratory](howto_labutopia.md) is also available as
`scene_background="labutopia_lab003"`. Its background includes an empty collection
work table, and tasks add their own objects. It is listed in
`AVAILABLE_BACKGROUND_IDS`; the three-kitchen `BACKGROUND_IDS` and default matrix
are unchanged. The native-kitchen details below apply to those three backgrounds.

## Integration

```python
from utils.dex_scene_eval import BACKGROUND_IDS, make_scene_eval_env

env = make_scene_eval_env(
    "OfficeDeskClickMouse-Sharpa-v1",
    scene_background=BACKGROUND_IDS[0],
    obs_mode="rgb",  # Use "state" for state-based policies.
)
obs, info = env.reset(seed=10001)
# Continue with the algorithm's existing policy and rollout loop.
settings = env.unwrapped.eval_background_manifest
```

Evaluate every task in `tasks/config/collection_targets.py:SUBTASKS` on each
background. Use the same episode seeds across algorithms and a separate process
per tasks/background combination. Save `eval_background_manifest` with results.

The factory enforces one CPU-physics environment, the registered task robot and
time limit, `pd_joint_pos`, reset-time reconfiguration, and the `robodex_fixed`
256 × 256 base/wrist camera rig. Incompatible overrides fail explicitly. The
camera rig need not match the task's default cameras. Registration is restored
after construction, so subsequent ordinary `gym.make` calls remain unaffected.

Tasks use the native room's unscaled table and collision geometry. The original
workspace table and ground have rendering and collisions disabled. At reset,
the room is translated to match the original task's visible robot-facing table
edge; task object positions remain unchanged. Robot root pose and joint reset
positions remain exactly as initialized by the original task. No floor-height
compensation or arm IK is applied.

All three native surfaces are 0.92 m high, within 0.36 mm of the 0.9196429 m
collection workspace. The factory rejects surface-height differences above
1 mm and verifies the tabletop elevation at reset. Lower dining tables are
not supported. Native table dimensions and collision shapes still differ;
horizontal working-edge alignment does not imply identical physics. The original
table's visible mesh and collision box also have different edges.

## GR00T

`train.train_groot` accepts `--scene-background` and uses the same factory.
Without this option, its existing environment construction is unchanged.

```bash
python -m train.train_groot --evaluate \
  --env-id OfficeDeskClickMouse-Sharpa-v1 --model-path /path/to/checkpoint \
  --scene-background robocasa_l3_s2 --sim-backend physx_cpu \
  --num-eval-envs 1 --eval-reconfiguration-freq 1 --camera-size 256 \
  --control-mode pd_joint_pos --max-episode-steps 500
```

The example loads a policy. Other tasks must use their own robot and horizon;
query `registered_task_settings(env_id)` rather than copying this task's values.
When recording video, GR00T also writes `scene_background.json` beside the video directory.

## Placement checks without a policy

Install the project's robot/task assets and the ManiSkill `RoboCasa` scene assets. Run rendering on a GPU/Vulkan-capable machine.

```bash
python utils/scene_matrix.py plan --out output/scene90
python utils/scene_matrix.py preview --out output/scene90/preview
# A single combination; indices follow the generated matrix order.
python utils/scene_matrix.py preview --index 2 --out output/scene90/preview
python -m pytest -q tests/test_scene_backgrounds.py tests/test_robodex_camera_configs.py
```

The preview uses GR00T's environment factory without loading weights. It saves
policy camera frames, diagnostic overview/side views, logs, and per-combination
checks under the requested output directory. Diagnostic cameras do not enter
policy observations. Preview artifacts are local outputs, not repository files.

Checks cover native support, disabled original surfaces, working-edge alignment,
tabletop height, trolley/table separation, short joint-hold steps, and repeatable
resets. They do not measure policy success or prove collision-free policy trajectories.

Use `--shard 0 --shards 3` to split work. `--resume` skips passed records with
matching source hashes and seeds. Each child gets an isolated temporary asset
cache. Failures return a nonzero exit status and retain their logs.
