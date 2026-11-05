# DexArena

DexArena is a simulation benchmark for long-horizon dexterous manipulation built on SAPIEN. It ships 10 long-horizon tasks, their 30 stage tasks, four end effectors (Sharpa, Wuji, XHand, Franka gripper), the tools the released demonstrations were exported, validated and rendered with, and the training and evaluation code for the baselines.

## Repository layout

```
DexArena/
├── tasks/                     # the 40 tasks, one file per task, grouped by family (see Tasks)
│   ├── config/                # simulation profile, materials, hands.py (end effectors), task table
│   ├── robots/                # robot agents: Panda arm + Sharpa / Wuji / XHand / gripper
│   ├── scene_builders/        # table, backgrounds and scene library
│   └── assets/                # robot URDFs and meshes (task objects come from the HF assets/)
├── data_gen/
│   ├── recordings/            # episode store, integrity certificates, review and export (README.md)
│   ├── replay/                # validation and release subset tools (README.md)
│   ├── render/                # offline multi-view rendering (README.md)
│   └── render_pixel_demos.py  # renders recorded demonstrations for the pixel-based trainers
├── train/                     # training / evaluation entry points and tyro configs (TRAIN_EVAL.md)
├── algo/                      # policy implementations used by the trainers
├── utils/                     # env construction, datasets, cameras, scene evaluation
├── scripts/                   # GR00T wrappers incl. its long-horizon evaluation; gripper URDF generator; scene-library CLI
├── tests/                     # functional tests of the task table, data contracts and baseline entry points
├── docs/                      # guides: task instructions, GR00T, PI0.5, scene library, cameras
└── libs/Isaac-GR00T/          # Isaac GR00T submodule, pinned; needed by the GR00T baseline only
```

## Install

```
conda create -n ms3 python=3.10
conda activate ms3
python -m pip install datasets "lerobot==0.4.0"
python -m pip install --upgrade mani_skill "gymnasium==0.29.1" tyro swanlab h5py tensorboard coacd
```

LeRobot installs PyTorch (the default wheel is built for CUDA 12.6). Rendering returns images on the GPU, so the wheel has to support your GPU; on RTX 50-series cards (compute capability 12.0) reinstall the CUDA 12.8 build:

```
python -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
```

Task-object meshes are not in git. Download them from the dataset repository into `tasks/assets/` (the `hf` command ships with `huggingface_hub`; the dataset repository is gated, so request access on its page once, which is granted automatically, and run `hf auth login`):

```
hf download SIPAIR/DexArena --repo-type dataset --include "assets/**" --local-dir tasks
```

## Tasks

Every task registers one environment id per end effector: `<Task>-<Hand>-v1` with `Hand` in `Sharpa`, `Wuji`, `XHand`, `Gripper`. `Arm` is the arm the task is performed with (`left`, `right`, or `both`), `Horizon` the registered episode length in control steps at 20 Hz. The list is also available in code as `tasks.config.collection_targets.TASKS`.

```python
import gymnasium as gym
import tasks  # registers every environment id

env = gym.make("DrawerPlugOpenDrawer-Sharpa-v1", obs_mode="state", control_mode="pd_joint_pos")
```

| Family | Task | Arm | Horizon | Env id (`<Task>-<Hand>-v1`, Hand = Sharpa, Wuji, XHand, Gripper) |
|---|---|---|---|---|
| Cabinet and pouring | CabinetPouringLong | both | 3400 | `CabinetPouringLong-<Hand>-v1` |
| Cabinet and pouring | CabinetPouringOpenCabinet | left | 600 | `CabinetPouringOpenCabinet-<Hand>-v1` |
| Cabinet and pouring | CabinetPouringPickBottle | right | 900 | `CabinetPouringPickBottle-<Hand>-v1` |
| Cabinet and pouring | CabinetPouringPourBeads | right | 700 | `CabinetPouringPourBeads-<Hand>-v1` |
| Cabinet and pouring | CabinetPouringUnscrewCap | right | 1400 | `CabinetPouringUnscrewCap-<Hand>-v1` |
| Drawer and plug | DrawerPlugLong | both | 3400 | `DrawerPlugLong-<Hand>-v1` |
| Drawer and plug | DrawerPlugInsertPlug | right | 900 | `DrawerPlugInsertPlug-<Hand>-v1` |
| Drawer and plug | DrawerPlugOpenDrawer | left | 600 | `DrawerPlugOpenDrawer-<Hand>-v1` |
| Drawer and plug | DrawerPlugPickCharger | right | 900 | `DrawerPlugPickCharger-<Hand>-v1` |
| Furniture assembly | FurnitureAssemblyLong | both | 2400 | `FurnitureAssemblyLong-<Hand>-v1` |
| Furniture assembly | FurnitureAssemblyFastenScrew | right | 600 | `FurnitureAssemblyFastenScrew-<Hand>-v1` |
| Furniture assembly | FurnitureAssemblyInsertRail | right | 700 | `FurnitureAssemblyInsertRail-<Hand>-v1` |
| Furniture assembly | FurnitureAssemblyLayFlatPanel | left | 600 | `FurnitureAssemblyLayFlatPanel-<Hand>-v1` |
| Juice preparation | JuicePreparationLong | both | 1800 | `JuicePreparationLong-<Hand>-v1` |
| Juice preparation | JuicePreparationAppleToBlender | right | 700 | `JuicePreparationAppleToBlender-<Hand>-v1` |
| Juice preparation | JuicePreparationCloseFridge | left | 650 | `JuicePreparationCloseFridge-<Hand>-v1` |
| Juice preparation | JuicePreparationOpenFridge | left | 650 | `JuicePreparationOpenFridge-<Hand>-v1` |
| Juice preparation | JuicePreparationPlaceLid | right | 400 | `JuicePreparationPlaceLid-<Hand>-v1` |
| Juice preparation | JuicePreparationPressButton | right | 400 | `JuicePreparationPressButton-<Hand>-v1` |
| Meal preparation | MealPreparationLong | both | 1900 | `MealPreparationLong-<Hand>-v1` |
| Meal preparation | MealPreparationOpenLunchBox | left | 300 | `MealPreparationOpenLunchBox-<Hand>-v1` |
| Meal preparation | MealPreparationPlaceBread | right | 400 | `MealPreparationPlaceBread-<Hand>-v1` |
| Meal preparation | MealPreparationScoopTomato | right | 1200 | `MealPreparationScoopTomato-<Hand>-v1` |
| Nut and bolt | NutBoltLong | both | 2400 | `NutBoltLong-<Hand>-v1` |
| Nut and bolt | NutBoltPickBolt | left | 700 | `NutBoltPickBolt-<Hand>-v1` |
| Nut and bolt | NutBoltSeatNut | right | 800 | `NutBoltSeatNut-<Hand>-v1` |
| Nut and bolt | NutBoltTurnNut | right | 900 | `NutBoltTurnNut-<Hand>-v1` |
| Office desk | OfficeDeskLong | both | 1400 | `OfficeDeskLong-<Hand>-v1` |
| Office desk | OfficeDeskClickMouse | right | 500 | `OfficeDeskClickMouse-<Hand>-v1` |
| Office desk | OfficeDeskOpenLaptop | right | 600 | `OfficeDeskOpenLaptop-<Hand>-v1` |
| Office desk | OfficeDeskTypeKeys | left | 700 | `OfficeDeskTypeKeys-<Hand>-v1` |
| Package opening | PackageOpeningLong | both | 800 | `PackageOpeningLong-<Hand>-v1` |
| Package opening | PackageOpeningExtractApple | right | 700 | `PackageOpeningExtractApple-<Hand>-v1` |
| Package opening | PackageOpeningOpenPackage | both | 500 | `PackageOpeningOpenPackage-<Hand>-v1` |
| Pipette transfer | PipetteTransferLong | both | 1600 | `PipetteTransferLong-<Hand>-v1` |
| Pipette transfer | PipetteTransferAspirateLiquid | right | 800 | `PipetteTransferAspirateLiquid-<Hand>-v1` |
| Pipette transfer | PipetteTransferDispenseLiquid | left | 1100 | `PipetteTransferDispenseLiquid-<Hand>-v1` |
| Table cleaning | TableCleaningLong | right | 1200 | `TableCleaningLong-<Hand>-v1` |
| Table cleaning | TableCleaningDumpDustpan | right | 600 | `TableCleaningDumpDustpan-<Hand>-v1` |
| Table cleaning | TableCleaningPickDebris | right | 900 | `TableCleaningPickDebris-<Hand>-v1` |

## Data

The demonstrations and the baseline checkpoints live in the Hugging Face dataset repository [`SIPAIR/DexArena`](https://huggingface.co/datasets/SIPAIR/DexArena). The released demonstrations were recorded with the Sharpa hand: one LeRobot v3.0 dataset per stage task under `data/sharpa/<env id>/`, 100 episodes each, with multi-view videos; `data_append/sharpa/<env id>/` holds the remaining reviewed episodes of each stage task, state only; `ckpt/` holds the baseline checkpoints. In general `<hand>` is the env id's end effector in lower case (`sharpa`, `wuji`, `xhand`, `gripper`). The dataset columns, the integrity certificate and the raw-recording sidecar are described in [data_gen/recordings/README.md](data_gen/recordings/README.md). Download a dataset into `data/<hand>/<env id>/` from the repository root:

```
hf download SIPAIR/DexArena --repo-type dataset --include "data/sharpa/DrawerPlugOpenDrawer-Sharpa-v1/**" --local-dir .
```

- Dataset layout, review and export, integrity certificate: [data_gen/recordings/README.md](data_gen/recordings/README.md)
- Validate recordings and build the release subset: [data_gen/replay/README.md](data_gen/replay/README.md)
- Render the state-only datasets to multi-view videos: [data_gen/render/README.md](data_gen/render/README.md)

## Train and evaluate

- PPO, DPPO, the offline (BC, FQE, AWR, CQL, IQL, TD3+BC) and offline-to-online (OPT, SAC, RLPD) agents, and training from pixels: [train/TRAIN_EVAL.md](train/TRAIN_EVAL.md)
- ACT, image Diffusion Policy and online IQL fine-tuning: `train/train_ACT.py`, `train/train_DP.py` and `train/train_iql_online.py` (`--help` lists the options; the ACT and Diffusion Policy checkpoints are `ckpt/act` and `ckpt/dp` on the Hub)
- PI0.5: [docs/PI05.md](docs/PI05.md)
- GR00T N1.7: [docs/GROOT_DEXARENA.md](docs/GROOT_DEXARENA.md)
- Cross-scene evaluation of the 30 stage tasks: [docs/howto_cross_scene_eval.md](docs/howto_cross_scene_eval.md)
- Scene randomization, the scene library and the cameras: [docs/howto_randomize_scene.md](docs/howto_randomize_scene.md), [docs/howto_scene_library.md](docs/howto_scene_library.md), [docs/camera_setup/README.md](docs/camera_setup/README.md), [docs/dexarena_base_camera.md](docs/dexarena_base_camera.md)

`tests/` holds functional tests of the task table, the dataset and camera contracts and the baseline entry points. They run on the CPU in under a minute and need neither datasets nor checkpoints (`python -m pip install pytest`, then `python -m pytest tests -q`). They are the regression check for extending the benchmark with a task, an end effector or a baseline; they do not evaluate policies, which is what the guides above are for.

## License

Apache-2.0, see [LICENSE](LICENSE).

## Acknowledgement

Some robot models come from https://github.com/dexsuite/dex-urdf.
