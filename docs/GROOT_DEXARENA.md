# GR00T on DexArena: subtask and long-horizon training and evaluation

How the GR00T N1.7 numbers of the paper are produced, end to end, with the code in this
repository: render a dexterous-hand subtask into a video dataset, finetune one checkpoint per
subtask, evaluate it, and then the long-horizon setting — one checkpoint per long task trained
on the union of its stages' datasets, evaluated by chaining the stages. Everything is driven by
the task table (`utils/dex_tasks.py`, 30 subtasks) and the long-task registry
(`utils/dex_long_tasks.py`, 10 long tasks); nothing is spelled out per task twice.

The GR00T environment itself (one interpreter with `gr00t`, `mani_skill` and `lerobot`) is built
by `bash scripts/groot/setup_env.sh`: it checks out the vendored `libs/Isaac-GR00T` submodule at its
pinned commit, runs `uv sync` there and installs ManiSkill and lerobot into the same virtual
environment, `libs/Isaac-GR00T/.venv`.

## What is where

| Piece | File | Role |
|---|---|---|
| Task table | `utils/dex_tasks.py` | env id, robot, joint layout (29 single-arm / 58 dual-arm), horizon, four instructions per subtask |
| Long tasks | `utils/dex_long_tasks.py` | the 10 long tasks as ordered stages: the subtask of each stage, the arm it uses, the `info` gate that marks it done |
| Cameras | `utils/dex_cameras.py`, `utils/dex_camera_rigs.py` | the published fixed rig (one base view per family + the wrist camera of each arm), installed at env construction |
| Renderer | `utils/groot_utils/replay_to_vla_dataset.py`, `aggregate_shards.py` | replays a state-only demonstration dataset and records the rig's views as a LeRobot dataset |
| Labels | `utils/groot_utils/relabel_instructions.py`, `scripts/dex/ops/verify_labels.py` | per-episode instruction protocol ([DEX_TASK_INSTRUCTIONS.md](DEX_TASK_INSTRUCTIONS.md)) |
| Modality configs | `utils/groot_utils/modality_configs.py` | GR00T's state/action/video splits, generated from the task table |
| Finetune / evaluate | `train/train_groot.py`, `algo/groot.py`, `train/train_configs/groot_config.py` | one entry point for both |
| Family datasets | `utils/groot_utils/merge_family_dataset.py`, `build_package_family_dataset.py`, `scripts/dex/ops/verify_family_labels.py` | one dataset per long task from its subtasks' datasets |
| Long-horizon evaluation | `scripts/dex/eval_long_chain.py`, `scripts/dex/eval_family_on_subtask.py`, `scripts/dex/ops/long_v2_table.py` | version 1 (chained) and version 2 (per-subtask AND) |

Every command below runs from the repository root in the GR00T interpreter. `DEX_OUT` is the
directory that holds the pipeline's outputs (`datasets/`, `runs/`, `evals/`); the tools default
to `./dex_out` when it is unset.

```bash
export DEX_OUT=/data/dex                                   # datasets/, runs/, evals/
export GROOT_BACKBONE_PATH=/models/nvidia/Cosmos-Reason2-2B   # optional: a local copy of the backbone
```

`GROOT_BACKBONE_PATH` lets training and evaluation load the gated backbone from a local copy
instead of the Hub: `scripts/groot/setup_env.sh` patches GR00T's finetune launcher to honour it,
and `algo/groot.py` repoints each checkpoint's baked-in reference at load time (the path must
contain `nvidia/Cosmos-Reason2`, which is how GR00T picks the backbone class). Without it, a
logged-in Hub token with access to `nvidia/Cosmos-Reason2-2B` is needed.

## 1. Render a subtask dataset

The released demonstrations (`SIPAIR/DexArena`, `data/<hand>/<ENV_ID>`) are
state-only. The renderer replays each episode by restoring the simulator state frame by frame
(`--restore set_state`; `--verify_frames N` cross-checks N frames of the first episode through
both restore paths and prints any disagreement), records the fixed rig's views at 256 × 256, and
writes the state and action as the controller-ordered joint vector (`--proprio controller_qpos`:
arm 7 + hand 22 for one arm, 58 for the dual-arm robot). Each episode is labelled with one of the
subtask's four registered instructions, 25 episodes each (the default; `--prompt` pins the whole
dataset to one phrasing).

The conversion and validation steps below need an `ffmpeg` executable on `PATH` and the FFmpeg
shared libraries that `torchcodec` loads (`libavutil`, `libavcodec`, `libavformat`, `libavdevice`);
on a host without them, a conda-forge `ffmpeg` on `PATH` and `LD_LIBRARY_PATH` is enough.

```bash
ENV=CabinetPouringOpenCabinet-Sharpa-v1
STEM=$(python -c "from utils.dex_tasks import resolve_dex_task as r; print(r('$ENV').short_name)")
DS=$DEX_OUT/datasets/${STEM}_rig_vla

python utils/groot_utils/replay_to_vla_dataset.py \
  --src data/sharpa/${ENV} --dst "$DS" --env_id $ENV --control_mode pd_joint_pos \
  --proprio controller_qpos --camera_size 256 --camera_profile robodex_fixed --verify_frames 6 \
  --gpu_id 0 --overwrite
```

The renderer runs one process; `--episode_start / --episodes` renders a slice, and
`utils/groot_utils/aggregate_shards.py --shards_root <dir with shard_*/> --dst "$DS"` joins the
slices (SAPIEN's Vulkan occasionally dies on a reconfigure, so a crash then costs one slice).

The renderer writes LeRobot v3.0. GR00T's loader reads v2.1, so convert in place (the v3.0
copy is kept as `<name>_v3.0` for other trainers), write the modality config and validate:

```bash
python libs/Isaac-GR00T/scripts/lerobot_conversion/convert_v3_to_v2.py --root $DEX_OUT/datasets --repo-id ${STEM}_rig_vla
python utils/groot_utils/modality_configs.py --env-id $ENV --write-json "$DS"
python scripts/groot/validate_dataset.py --dataset-path "$DS" --env-id $ENV
```

A dataset rendered before the per-episode protocol (one instruction everywhere) is relabelled
in place with `utils/groot_utils/relabel_instructions.py` and checked with
`scripts/dex/ops/verify_labels.py`; no re-render is needed.

## 2. Finetune a subtask checkpoint

Full-parameter finetune of GR00T N1.7-3B (vision and language backbone unfrozen), effective batch
32 as 2 × 16 gradient accumulation, learning rate 4e-5, 12 000 steps, one 80 GB GPU (~64 GB;
`--no_tune_llm --no_tune_visual` trains the action head only in ~20 GB). This is the recipe
behind every number in the paper.

```bash
python train/train_groot.py \
  --dataset_path "$DS" --env_id $ENV --base_model_path /models/GR00T-N1.7-3B \
  --modality_config_path utils/groot_utils/modality_configs.py \
  --output_dir $DEX_OUT/runs/groot_${STEM}_rig --num_gpus 1 --master_port 29500 \
  --global_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 4e-05 \
  --max_steps 12000 --save_steps 1000 --save_total_limit 2
```

`train_groot.py` registers the modality config of `--env_id` (`$GROOT_MODALITY_ENV` overrides it)
and hands over to GR00T's `launch_finetune.py` under `torchrun`. A checkpoint is written every
`--save_steps` (~12 GB of weights plus the optimizer state; `checkpoint-12000` is the final
one); `--resume_from_checkpoint 1` with the same `--output_dir` continues from the newest
checkpoint in that directory, restoring optimizer state, schedule and step count. The
instruction each episode carries is what the model is conditioned on; nothing is passed on the
command line.

## 3. Evaluate a subtask checkpoint

Fifty episodes, seeds 10001–10050, the task's own horizon, action horizon 8, the fixed camera
rig, and an instruction drawn per episode from the subtask's registered phrasings (the seed
decides, so a run is reproducible; `--instruction_sampling canonical` runs every episode on the
canonical phrasing, which is the protocol of the single-instruction numbers in the paper).

```bash
read H ROBOT < <(python -c "from utils.dex_tasks import resolve_dex_task as r; t=r('$ENV'); print(t.max_episode_steps, t.robot_uids)")
CK=$DEX_OUT/runs/groot_${STEM}_rig/checkpoint-12000

python train/train_groot.py --evaluate --model_path "$CK" --env_id $ENV --control_mode pd_joint_pos \
  --robot_uids $ROBOT --camera_size 256 --camera_profile robodex_fixed \
  --instruction_sampling registered --seed 10001 --num_eval_episodes 50 \
  --num_eval_steps $H --max_episode_steps $H --sim_backend physx_cpu --action_horizon 8 \
  --capture_video --eval_output_dir $DEX_OUT/evals/groot/${STEM}_rig/checkpoint-12000
```

The output directory gets `summary.json` (success rate and the full protocol), `episodes.jsonl`
(one line per episode: seed, instruction, success, length, video file) and `videos/`.
`--scene_background <scene>` evaluates the same checkpoint on one of the shared native-scene
backgrounds ([howto_cross_scene_eval.md](howto_cross_scene_eval.md)).

## 4. Long-horizon tasks

Each of the 10 long tasks is its stages in order, and every stage is one of the 30 subtasks
(`utils/dex_long_tasks.py`; the order and the `info` gate of each stage were read from the task
modules and confirmed on the live environments). Nine long tasks run the 58-joint dual-arm
robot while the policies are 29-joint single-arm, so during a long-task rollout the acting
arm's 29 joints are routed into its slots and the idle arm is held at its current pose
(`route_action`); the policy's "wrist" view is the acting arm's camera. `TableCleaningLong-Sharpa-v1` is the
one family whose stages all use the right hand, so it runs the single-arm robot as is.

### 4.1 Family datasets

A family checkpoint trains on all of its subtasks' data at once. Within a family the subtask
datasets share cameras, joint layout and fps and differ only in their instructions, so the
family dataset is their concatenation with episodes, frames and the task table renumbered:

```bash
python utils/groot_utils/merge_family_dataset.py --family juice --out-tag v4      # -> datasets/juice_family_rig_v4_vla
python scripts/dex/ops/verify_family_labels.py --dataset $DEX_OUT/datasets/juice_family_rig_v4_vla
```

`meta/robodex_family.json` records which merged episodes came from which subtask
(`source_ranges`) and whether every source carried per-episode labels; the verifier checks each
source's labels against the instruction table (25/25/25/25 per subtask). The package family is
the exception: `OpenPackage` is dual-arm (58 joints, three views) and `ExtractAndPlaceApple`
single-arm, so `build_package_family_dataset.py` lifts the single-arm episodes into the 58-joint
layout (left arm and hand filled with the dual robot's initial pose, read from the dual-arm
episodes) and keeps the two views both subtasks have, the base camera and the right wrist.

```bash
python utils/groot_utils/build_package_family_dataset.py --out-tag v4   # -> datasets/package_family_rig_v4_vla
```

### 4.2 Family training

The same command as section 2, on the family dataset. `--env_id` names any stage whose joint
layout the family dataset has (it only selects the modality config): a single-arm subtask for the
nine single-arm families, `PackageOpeningOpenPackage-Sharpa-v1` for the package family, which
also needs `GROOT_VIDEO_KEYS=head:base_0_rgb,right_wrist:right_wrist_0_rgb` exported so that
the modality config lists exactly the dataset's two views. The budget is the same 12 000 steps as
one subtask checkpoint: the question the experiment asks is "one model or k models for the same
compute", not "k times the compute".

```bash
python train/train_groot.py --dataset_path $DEX_OUT/datasets/juice_family_rig_v4_vla --env_id JuicePreparationOpenFridge-Sharpa-v1 \
  --output_dir $DEX_OUT/runs/groot_juice_family_rig_v4 ...   # remaining arguments as in section 2
```

### 4.3 Evaluation, version 1: the long task, chained

The long task's own environment, one episode = all stages in order; a stage is passed when its
gate in `info` latches, the episode succeeds only when the last stage passes within the long
task's horizon. Two modes:

- `chain`: one subtask checkpoint per stage, control switches to the next checkpoint when a
  stage passes (only the acting policy is resident on the GPU);
- `family`: one family checkpoint, only the instruction switches.

```bash
unset GROOT_MODALITY_ENV GROOT_VIDEO_KEYS     # each checkpoint's own layout is read from its files
python scripts/dex/eval_long_chain.py --long juice --mode chain --ckpt-root $DEX_OUT/runs \
  --episodes 50 --out $DEX_OUT/evals/long/chain/juice
python scripts/dex/eval_long_chain.py --long juice --mode family \
  --family-ckpt $DEX_OUT/runs/groot_juice_family_rig_v4/checkpoint-12000 \
  --episodes 50 --out $DEX_OUT/evals/long_v4/family/juice
```

Each stage's instruction is drawn per episode from its subtask's registered phrasings
(`--instruction-sampling canonical` for the fixed ones). `summary.json` reports the success
rate and, per stage, how many episodes got past it (`stage_reached_counts`);
`episodes.jsonl` has the stages done, steps per stage and the instructions of every episode.

### 4.4 Evaluation, version 2: per-subtask AND

Each subtask is evaluated alone, in its own environment, from its own initial state, with the
same seeds; a long task passes seed *s* only if every one of its subtasks passed seed *s*. For the
subtask checkpoints this is computed from their ordinary evaluations (section 3). For a family
checkpoint, `eval_family_on_subtask.py` runs it in each subtask environment:

```bash
for SUB in JuicePreparationOpenFridge-Sharpa-v1 JuicePreparationAppleToBlender-Sharpa-v1 JuicePreparationCloseFridge-Sharpa-v1 JuicePreparationPlaceLid-Sharpa-v1 JuicePreparationPressButton-Sharpa-v1; do
  STEM=$(python -c "from utils.dex_tasks import resolve_dex_task as r; print(r('$SUB').short_name)")
  python scripts/dex/eval_family_on_subtask.py --family juice --subtask $SUB \
    --family-ckpt $DEX_OUT/runs/groot_juice_family_rig_v4/checkpoint-12000 \
    --episodes 50 --out $DEX_OUT/evals/long_v2_v4/family/juice/$STEM
done
python scripts/dex/ops/long_v2_table.py --run-tag v4     # the table for all ten long tasks
```

## Results

Successes out of 50 episodes (seeds 10001–10050, the task's own horizon, action horizon 8, the
fixed camera rig). Two rounds were run with the same recipe and differ only in the labels:
**single-instruction** checkpoints (every episode of a task carries its canonical instruction,
evaluated on that instruction; these are the paper's numbers) and **multi-instruction** checkpoints
(each episode carries one of the task's four registered instructions, 25 each, and evaluation
draws one per episode; the checkpoints in `SIPAIR/DexArena`).

### Subtasks

| Task | Env id | Single-instruction | Multi-instruction |
|---|---|---|---|
| OpenCabinet | `CabinetPouringOpenCabinet-Sharpa-v1` | 46 | 12 |
| PickBottle | `CabinetPouringPickBottle-Sharpa-v1` | 39 | 20 |
| UnscrewCap | `CabinetPouringUnscrewCap-Sharpa-v1` | 31 | 23 |
| PourBeads | `CabinetPouringPourBeads-Sharpa-v1` | 39 | 33 |
| OpenDrawer | `DrawerPlugOpenDrawer-Sharpa-v1` | 49 | 50 |
| PickCharger | `DrawerPlugPickCharger-Sharpa-v1` | 39 | 38 |
| InsertPlug | `DrawerPlugInsertPlug-Sharpa-v1` | 3 | 3 |
| PickBolt | `NutBoltPickBolt-Sharpa-v1` | 22 | 10 |
| SeatNut | `NutBoltSeatNut-Sharpa-v1` | 13 | 14 |
| TurnNut | `NutBoltTurnNut-Sharpa-v1` | 50 | 46 |
| OpenLaptop | `OfficeDeskOpenLaptop-Sharpa-v1` | 50 | 50 |
| TypeKeys | `OfficeDeskTypeKeys-Sharpa-v1` | 16 | 5 |
| ClickMouse | `OfficeDeskClickMouse-Sharpa-v1` | 49 | 48 |
| LayFlatPanel | `FurnitureAssemblyLayFlatPanel-Sharpa-v1` | 32 | 30 |
| InsertRail | `FurnitureAssemblyInsertRail-Sharpa-v1` | 8 | 4 |
| FastenScrew | `FurnitureAssemblyFastenScrew-Sharpa-v1` | 42 | 46 |
| AspirateLiquid | `PipetteTransferAspirateLiquid-Sharpa-v1` | 4 | 0 |
| DispenseLiquid | `PipetteTransferDispenseLiquid-Sharpa-v1` | 4 | 0 |
| OpenFridge | `JuicePreparationOpenFridge-Sharpa-v1` | 47 | 39 |
| AppleToBlender | `JuicePreparationAppleToBlender-Sharpa-v1` | 27 | 26 |
| CloseFridge | `JuicePreparationCloseFridge-Sharpa-v1` | 49 | 41 |
| PlaceLid | `JuicePreparationPlaceLid-Sharpa-v1` | 21 | 18 |
| PressButton | `JuicePreparationPressButton-Sharpa-v1` | 32 | 35 |
| OpenPackage | `PackageOpeningOpenPackage-Sharpa-v1` | 32 | 24 |
| ExtractApple | `PackageOpeningExtractApple-Sharpa-v1` | 50 | 45 |
| PickDebris | `TableCleaningPickDebris-Sharpa-v1` | 0 | 1 |
| DumpDustpan | `TableCleaningDumpDustpan-Sharpa-v1` | 41 | 46 |
| OpenLunchBox | `MealPreparationOpenLunchBox-Sharpa-v1` | 50 | 48 |
| PlaceBread | `MealPreparationPlaceBread-Sharpa-v1` | 30 | 32 |
| ScoopTomato | `MealPreparationScoopTomato-Sharpa-v1` | 6 | 8 |
| **Mean** | | **61.4 % (921 / 1500)** | **53.0 % (795 / 1500)** |

The per-episode protocol costs eight points on average, mostly on the contact-heavy tasks
(OpenCabinet −34, PickBottle −19, PickBolt −12, TypeKeys −11); the other twenty-six move by
five or fewer, within the noise of a 50-episode evaluation. Re-evaluating four single-instruction
checkpoints under the per-episode protocol costs them 0–12 episodes, so most of the gap is the
training, not the evaluation: with 12 000 steps and 25 episodes per phrasing the mixed
retrain learns the task less well, while the single-instruction checkpoints already follow the
other phrasings reasonably.

### Long-horizon tasks

Each cell is the number of successful episodes out of 50, followed in parentheses by the number
of episodes that completed each stage (version 1) or passed each subtask on its own (version 2).
Version 1 chains the stages in the long task's environment: `chain` switches between the
subtask checkpoints, `family` runs the one family checkpoint. Version 2 evaluates every subtask
alone and counts a seed as a success only if all subtasks of the task passed it.

| Long task | Stages | v1 chain, single | v1 family, single | v2 subtask ckpts, single | v2 family, single | v1 family, multi | v2 subtask ckpts, multi | v2 family, multi |
|---|---|---|---|---|---|---|---|---|
| MealPreparation (`MealPreparationLong-Sharpa-v1`) | OpenLunchBox, PlaceBread, ScoopTomato | 0 (50/37/0) | 0 (50/7/0) | 3 (50/30/6) | 0 (42/10/0) | 0 (50/2/0) | 4 (48/32/8) | 0 (43/1/0) |
| CabinetAndPouring (`CabinetPouringLong-Sharpa-v1`) | OpenCabinet, PickBottle, UnscrewCap, PourBeads | 0 (41/26/0/0) | 0 (44/33/0/0) | 20 (46/39/31/39) | 1 (41/23/29/17) | 0 (26/11/0/0) | 0 (12/20/23/33) | 2 (40/16/22/31) |
| DrawerAndPlug (`DrawerPlugLong-Sharpa-v1`) | OpenDrawer, PickCharger, InsertPlug | 0 (50/18/0) | 0 (10/2/0) | 3 (49/39/3) | 0 (49/43/0) | 0 (12/2/0) | 1 (50/38/3) | 0 (40/26/0) |
| NutAndBolt (`NutBoltLong-Sharpa-v1`) | PickBolt, SeatNut, TurnNut | 0 (12/0/0) | 0 (0/0/0) | 5 (22/13/50) | 1 (3/8/48) | 0 (0/0/0) | 2 (10/14/46) | 2 (13/10/49) |
| OfficeDesk (`OfficeDeskLong-Sharpa-v1`) | OpenLaptop, TypeKeys, ClickMouse | 0 (50/28/0) | 0 (36/0/0) | 15 (50/16/49) | 0 (48/0/25) | 0 (24/0/0) | 5 (50/5/48) | 0 (46/0/21) |
| FurnitureAssembly (`FurnitureAssemblyLong-Sharpa-v1`) | LayFlatPanel, InsertRail, FastenScrew | 0 (3/0/0) | 0 (4/0/0) | 5 (32/8/42) | 1 (18/4/40) | 0 (1/0/0) | 2 (30/4/46) | 1 (13/2/47) |
| PipetteTransfer (`PipetteTransferLong-Sharpa-v1`) | AspirateLiquid, DispenseLiquid | 0 (2/0) | 0 (0/0) | 0 (4/4) | 0 (3/0) | 0 (0/0) | 0 (0/0) | 0 (2/0) |
| JuicePreparation (`JuicePreparationLong-Sharpa-v1`) | OpenFridge, AppleToBlender, CloseFridge, PlaceLid, PressButton | 0 (39/25/15/0/0) | 0 (8/0/0/0/0) | 7 (47/27/49/21/32) | 0 (41/12/50/2/18) | 0 (14/4/2/0/0) | 4 (39/26/41/18/35) | 0 (43/31/50/3/30) |
| PackageOpening (`PackageOpeningLong-Sharpa-v1`) | OpenPackage, ExtractApple | 0 (28/0) | 0 (2/0) | 32 (32/50) | 0 (3/9) | 0 (0/0) | 22 (24/45) | 2 (4/38) |
| TableCleaning (`TableCleaningLong-Sharpa-v1`) | PickDebris, DumpDustpan | 0 (1/0) | 0 (0/0) | 0 (0/41) | 0 (0/36) | 0 (0/0) | 1 (1/46) | 0 (0/41) |
| **Mean** | | **0.0 %** | **0.0 %** | **18.0 %** | **0.6 %** | **0.0 %** | **8.2 %** | **1.4 %** |

No long task has been completed end to end in the chained setting, in either round and with
either kind of checkpoint, and no episode has completed more than three stages. Each family has
a stage that is at or near zero on its own (TypeKeys, ScoopTomato, DispenseLiquid, PickDebris,
InsertRail, InsertPlug), and later stages start from whatever state the earlier ones left
behind. Under the same 12 000-step budget a family checkpoint is also weaker than the subtask
checkpoints on the hardest subtask of its family (OpenPackage 3 vs 32, TypeKeys 0 vs 16,
InsertPlug 0 vs 3, ScoopTomato 0 vs 6 in the single-instruction round), which is what drives
the version-2 gap; on the opening and turning stages it is close to them.

The multi-instruction checkpoints are in the `SIPAIR/DexArena` dataset repository under
`ckpt/gr00t-n1.7-multi-instruction/<Task>/` (30 subtasks) and
`ckpt/gr00t-n1.7-multi-instruction/long-horizon/<Task>/` (10 long tasks), each a complete
`checkpoint-12000` with a README that maps the task names above to env ids.
