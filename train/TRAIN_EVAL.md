# Train / Eval / Config Guide

A practical guide for the fifteen training workflows in this repo:

1. **PPO** (`train_ppo.py`) — online single-step PPO, optionally demo-augmented (BC loss + demo-init resets).
2. **State BC** (`train_bc.py`) — offline deterministic MLP behavior cloning from LeRobot transitions.
3. **FQE/SARSA** (`train_fqe.py`) — offline evaluation of the demonstration behavior policy; it produces no actor.
4. **AWR** (`train_awr.py`) — one-step policy improvement using a frozen FQE behavior-Q.
5. **CQL** (`train_cql.py`) — conservative offline actor-critic learning.
6. **IQL** (`train_iql.py`) — offline expectile value learning and advantage-weighted policy extraction.
7. **TD3+BC** (`train_td3bc.py`) — deterministic offline TD3 regularized by behavior cloning.
8. **OPT** (`train_opt.py`) — IQL-based offline-to-online training with an online pre-training phase.
9. **SAC** (`train_sac.py`) — online SAC fine-tuning from an IQL/CQL checkpoint.
10. **RLPD** (`train_rlpd.py`) — from-scratch, high-UTD online SAC with symmetric offline/online sampling.
11. **RLPD from pixels** (`train_rlpd_pixels.py`) — the same recipe from two RGB cameras plus proprioception, with a visual encoder trained by the critic loss.
12. **PPO from pixels** (`train_ppo_pixels.py`) — online PPO on the same observation, anchored by a BC loss on rendered demos and demo-state resets.
13. **Diffusion BC** (`train_diffusion.py`) — offline behavior cloning of a diffusion policy from LeRobot demos.
14. **DPPO** (`train_dppo.py`) — PPO fine-tuning of a BC diffusion policy's denoising chain.
15. **PI0.5 OpenPI LoRA** (`train_pi05.py`) — supervised LoRA
    fine-tuning and asynchronous absolute-joint evaluation.

The state-policy algorithms have three explicit checkpoint pipelines:

```text
FQE -> AWR                 # evaluate demo behavior, then extract a policy
IQL -> OPT                 # offline IQL, then OPT online pre-training/fine-tuning
IQL or CQL -> SAC          # offline warm start, then online SAC fine-tuning
```

RLPD deliberately has no offline warm start; it starts from a random policy and
mixes frozen demos into every online update. Diffusion and DPPO form another
two-stage pipeline: Diffusion BC produces the base policy and DPPO fine-tunes
it. Examples use state observations unless stated otherwise. The non-PI0.5
commands assume the `ms3` conda env; PI0.5 uses the separate OpenPI/JAX runtime
documented below. All commands run from the repo root. The document is organized
into three blocks — **[Train](#train)**, **[Eval](#eval)**, **[Config](#config)**
— with PI0.5 called out where its contract differs.

The nine state-policy agents are grouped by training paradigm:

| Group | Module | Algorithms |
|---|---|---|
| Offline RL | `algo/offline_rl/` | BC, FQE, AWR, CQL, IQL, TD3+BC |
| Offline-to-online RL | `algo/offline_to_online_rl/` | OPT, SAC, RLPD |

Network primitives shared by both groups live in `algo/networks.py`; the
env-construction and evaluation helpers used by every `train_*.py` script live in
`utils/train_utils.py`.

## Using these algorithms on your own dataset

Nothing in `algo/` or `utils/train_utils.py` is tied to a particular task. To
train on your own data you need:

1. **A LeRobot dataset** whose `meta/info.json` declares these features:
   `observation.state`, `action`, `next.reward`, `next.done`, `next.truncated`,
   `next.success`. The loader (`utils/offline_replay_buffer.py`) infers `obs_dim`
   and `action_dim` from the dataset — no code change is needed for a different
   observation or action size. `next.success` is the terminal signal; episodes
   that never reach it have their final (unbootstrappable) transition dropped.
2. **A registered ManiSkill environment** whose flattened state observation and
   action space match the dataset exactly, and whose `--control_mode` matches the
   controller the demos were recorded with. Put the task module under `tasks/`;
   `import tasks` registers every module there automatically.
3. **Paths.** Absolute `--demo_data_path` values are used as-is. Relative values
   resolve against `$ROBODEX_DATA_ROOT`, and checkpoints/tensorboard runs against
   `$ROBODEX_STORAGE_ROOT` / `$ROBODEX_RUNS_ROOT`. All three default to this
   repository, so nothing is tied to one machine:

   ```bash
   export ROBODEX_STORAGE_ROOT=/your/scratch
   export ROBODEX_DATA_ROOT=/your/scratch/datasets
   ```

Then point any trainer at it, e.g.
`python train/train_iql.py --env_id <YourEnv-v1> --control_mode <your-controller> --demo_data_path <lerobot-root>`.

> **Hard constraints (read first)**
> - `--control_mode` must match the demos' controller across Diffusion BC **and** DPPO
>   (demos here are `pd_ee_delta_pose`, 7-dim). A mismatch fails on action dim.
> - The state BC/FQE/AWR/CQL/IQL/TD3+BC/OPT/SAC/RLPD workflows likewise require
>   the environment observation/action dimensions and controller to match the
>   LeRobot dataset. Keep reward semantics consistent whenever Q/V targets are
>   trained or transferred.
> - AWR must use an FQE checkpoint trained on the same observation/action
>   contract. OPT expects an IQL checkpoint; SAC accepts an IQL or CQL
>   checkpoint. Use `--checkpoint` only to resume/evaluate the same algorithm,
>   and `--pretrained_checkpoint` for offline-to-online initialization.
> - DPPO must reuse the BC run's exact backbone args (`--backbone`, and for MLP
>   `--mlp_dims/--residual_style/--use_layernorm`), or `load_base_policy` errors
>   on state-dict keys.
> - `--n_obs_steps / --horizon / --n_action_steps` must be identical in BC and DPPO.
> - PI0.5 uses named absolute `pd_joint_pos` state/action channels (29 single-arm
>   or 58 dual-arm), RGB inputs and episode labels from `utils.dex_tasks`.
>   Prepare a matching dataset/controller binding and normalization contract;
>   see [the PI0.5 guide](../docs/PI05.md).

---

## Train

### PPO (`train_ppo.py`)

Online single-step PPO (no diffusion backbone), optionally augmented with a BC
loss and demo-initialized resets. Train the grasping task with BC loss and
demo-initialized resets:
```
python train/train_ppo.py \
  --env_id CabinetPouringOpenCabinet-Sharpa-v1 \
  --use_bc_loss \
  --use_demo_init_states \
  --total_timesteps 50000000
```
For `CabinetPouringOpenCabinet-Sharpa-v1`, pass `--control_mode pd_ee_delta_pose --reward_mode
normalized_dense` and point `--bc_data_path/--demo_init_data_path` at the
filtered demo set.

### Offline RL (`algo/offline_rl/`)

Trained purely from a static LeRobot dataset; an environment is built only
for periodic evaluation rollouts.

#### State BC (`train_bc.py`)

Pure offline behavior cloning of a deterministic MLP policy. It consumes only
demo `(observation, action)` pairs: there is no critic, reward target, or online
collection. `--success_only` filters failed/timeout episodes; it is currently
off by default. The last action dimension can be emphasized with
`--bc_gripper_weight`.

```bash
python train/train_bc.py \
  --env_id CabinetPouringOpenCabinet-Sharpa-v1 --control_mode pd_ee_delta_pose \
  --demo_data_path <lerobot-demo-root> --success_only \
  --total_updates 100000 --batch_size 256 \
  --exp_name ballgrasp_bc --track
```

`--drop_object_pose` removes the seven flattened object-pose coordinates for
the supported ball-grasp state layout. Pass the same flag explicitly during
evaluation so that the reconstructed actor has the checkpoint's input size.

#### FQE/SARSA (`train_fqe.py`)

Fitted-Q Evaluation learns the demonstration behavior policy's Q-function with
the in-dataset SARSA target
`r + gamma * (1 - done) * Q_target(next_observation, next_action)`. It creates
no environment and no policy. During training it compares predicted Q against
the demo Monte-Carlo return-to-go; inspect `q_vs_mc_mse` and `q_vs_mc_corr`
before using the result for AWR.

```bash
python train/train_fqe.py \
  --demo_data_path <lerobot-demo-root> \
  --gamma 0.99 --total_updates 200000 \
  --exp_name ballgrasp_fqe --track
```

The checkpoint contains `qf`, `qf_target`, and the observation statistics.
FQE has no `--evaluate` mode: its validation is the periodic offline MC check.

#### Frozen-Q AWR (`train_awr.py`)

AWR is the policy-improvement half of the FQE/AWR pipeline. It freezes the FQE
Q-network, fits an expectile `V(s)` to Q on dataset actions, and trains the
actor by advantage-weighted regression. It does not update a critic and does
not consume rewards or next states during its own updates.

```bash
python train/train_awr.py \
  --fqe_checkpoint runs/ballgrasp_fqe/final_ckpt.pt \
  --demo_data_path <same-lerobot-demo-root> \
  --env_id CabinetPouringOpenCabinet-Sharpa-v1 --control_mode pd_ee_delta_pose \
  --iql_expectile 0.7 --iql_beta 3.0 \
  --total_updates 200000 --exp_name ballgrasp_awr --track
```

The FQE observation statistics are reused verbatim. The resulting AWR
checkpoint is self-contained: it embeds the frozen Q and does not require
`--fqe_checkpoint` for eval-only loading.

#### CQL (`train_cql.py`)

CQL is an offline SAC-style twin-Q learner. Its critic loss combines the normal
entropy-regularized Bellman loss with a conservative penalty on random and
policy-sampled actions. The actor supports a pure-BC warmup and a persistent BC
anchor; optional counterfactual transitions can be supplied through
`--extra_npz_paths`.

The defaults match the other offline trainers (`CabinetPouringOpenCabinet-Sharpa-v1`,
`pd_ee_delta_pose`); override `--env_id`, `--control_mode` and
`--demo_data_path` together when using another dataset/task:

```bash
python train/train_cql.py \
  --env_id CabinetPouringOpenCabinet-Sharpa-v1 --control_mode pd_ee_delta_pose \
  --demo_data_path <lerobot-demo-root> \
  --cql_alpha 5.0 --cql_num_actions 10 \
  --bc_warmup_steps 10000 --actor_bc_weight 1.0 \
  --total_updates 500000 --exp_name ballgrasp_cql --track
```

`--bc_only` uses the same trainer as a stochastic-actor BC diagnostic. For
normal CQL training leave it disabled. Entropy autotuning is intentionally off
by default in the current task recipe.

#### IQL (`train_iql.py`)

IQL is fully offline. It fits `V(s)` by expectile regression toward the twin-Q
estimate of dataset actions, bootstraps Q with `r + gamma * V(next_state)`, and
extracts the policy with advantage-weighted regression. It therefore avoids
sampling out-of-distribution actions inside the critic target.

```bash
python train/train_iql.py \
  --env_id CabinetPouringOpenCabinet-Sharpa-v1 --control_mode pd_ee_delta_pose \
  --demo_data_path <lerobot-demo-root> --success_only \
  --iql_expectile 0.7 --iql_beta 3.0 \
  --bc_warmup_steps 20000 --actor_bc_weight 1.0 \
  --total_updates 500000 --exp_name ballgrasp_iql --track
```

As in CQL, `--bc_only` is a diagnostic mode. IQL also accepts branch-transition
`.npz` files through `--extra_npz_paths`; `critic_extra_fraction`,
`actor_demo_only`, `value_demo_only`, and `bootstrap_extra_transitions` control
where those counterfactual rows participate.

#### TD3+BC (`train_td3bc.py`)

TD3+BC is a standalone deterministic offline learner with twin critics, target
policy smoothing, delayed actor/target updates, and the actor objective
`-lambda * Q(s, pi(s)) + MSE(pi(s), action_demo)`. Its networks are local to
`algo/offline_rl/td3bc.py`, rather than the stochastic actor/Q definitions shared by
CQL/IQL/SAC.

```bash
python train/train_td3bc.py \
  --env_id CabinetPouringOpenCabinet-Sharpa-v1 --control_mode pd_ee_delta_pose \
  --demo_data_path <lerobot-demo-root> --success_only \
  --policy_freq 2 --policy_noise 0.2 --noise_clip 0.5 --alpha 2.5 \
  --total_updates 500000 --exp_name ballgrasp_td3bc --track
```

Optionally pass `--bc_checkpoint runs/<bc-run>/final_ckpt.pt` to initialize only
the actor from State BC. In contrast, `--checkpoint` resumes a complete TD3+BC
agent including critics and optimizer state; the two flags are mutually
exclusive.

### Offline-to-online RL (`algo/offline_to_online_rl/`)

Collect online transitions while still mixing the offline dataset into every
update. They build a training environment and an online replay buffer.

#### OPT (`train_opt.py`)

OPT is a three-stage IQL-based offline-to-online workflow. Phase A is a separate
offline IQL run. `train_opt.py` then performs Phase B by collecting online data
with the frozen offline policy and meta-training fresh online Q/V functions,
followed by Phase C joint fine-tuning. The actor's AWR advantage interpolates
between offline and online value functions with the scheduled coefficient
`kappa`.

```bash
python train/train_opt.py \
  --pretrained_checkpoint runs/ballgrasp_iql/final_ckpt.pt \
  --demo_data_path <same-lerobot-demo-root> \
  --env_id CabinetPouringOpenCabinet-Sharpa-v1 --control_mode pd_ee_delta_pose \
  --opt_env_steps 25000 --opt_grad_steps 50000 \
  --total_env_steps 1000000 --offline_mix_fraction 0.5 \
  --exp_name ballgrasp_opt --track
```

Use `--checkpoint` to resume OPT itself. The online replay batch can continue
mixing frozen demos, and a decaying demo BC anchor is enabled by the current
defaults to reduce offline-to-online unlearning.

#### SAC offline-to-online fine-tuning (`train_sac.py`)

This is online SAC initialized from a structurally compatible IQL or CQL
checkpoint. It transfers the actor, twin critics, target critics, and
observation statistics; IQL's extra `vf` is discarded. Online batches mix FIFO
environment replay with frozen demos, and the actor has an optional decaying BC
anchor.

```bash
python train/train_sac.py \
  --pretrained_checkpoint runs/ballgrasp_iql/final_ckpt.pt \
  --demo_data_path <same-lerobot-demo-root> \
  --env_id CabinetPouringOpenCabinet-Sharpa-v1 --control_mode pd_ee_delta_pose \
  --total_env_steps 1000000 --updates_per_iter 16 \
  --offline_mix_fraction 0.5 \
  --exp_name ballgrasp_sac --track
```

`--checkpoint` means a full SAC resume and takes precedence over
`--pretrained_checkpoint`. Fixed entropy temperature is the current default;
enable `--autotune_entropy` only as an intentional experiment.

#### RLPD (`train_rlpd.py`)

RLPD starts from scratch: it does not load an offline policy and has no BC
anchor. After a collection warmup it performs high-UTD SAC updates, with each
batch sampled symmetrically from online replay and the frozen offline dataset.
The critic is a LayerNorm Q ensemble and each target uses a random minimum
subset (REDQ style).

```bash
python train/train_rlpd.py \
  --env_id CabinetPouringOpenCabinet-Sharpa-v1 --control_mode pd_ee_delta_pose \
  --demo_data_path <lerobot-demo-root> \
  --num_envs 1 --start_steps 5000 --utd_ratio 20 \
  --offline_mix_fraction 0.5 --total_env_steps 1000000 \
  --exp_name ballgrasp_rlpd --track
```

With `num_envs=1`, `utd_ratio` is exactly the number of critic updates per
collected transition. Increasing `num_envs` without increasing `utd_ratio`
reduces the effective per-transition UTD. For `pd_joint_pos`, random warmup arm
targets are sampled locally around the current qpos using
`--joint_warmup_delta`.

#### Training from pixels (`train_rlpd_pixels.py`, `train_ppo_pixels.py`)

Two trainers consume the same image observation: RLPD from pixels and a
demo-anchored PPO. Both were run on the 30 `sharpa_hand` subtasks with one set
of hyperparameters each (only the episode length varies by task); the commands
below are those configurations.

##### The pixel observation

Every pixel trainer, evaluator and rendered dataset builds its observation from
one `PixelObsConfig` (`utils/pixel_cameras.py`), selected by `--camera-preset`
and `--image-size`:

- `cam_0` is the long task's fixed base camera from `utils/camera_configs.py`
  (`FAMILY_CAMERA_RIGS`) and `cam_1` the calibrated wrist camera on the hand
  flange (`resolve_wrist_camera_mount`). The `sharpa_hand` subtasks were all
  rendered and trained with `--camera-preset cabinet_fixed --image-size 128`.
- `proprio` is `qpos`, `qvel` and `tcp_pose`, plus the previous joint command
  when `--delta-action` is on. Keys the task leaves in its rgb observation that a
  real robot could not read (`door_qpos`, object poses, goal positions, ...) are
  listed in `PRIVILEGED_ENV_OBS_KEYS` in `utils/pixel_observation.py` and never
  reach the policy.
- The config hashes itself (`camera_config_hash`) into every checkpoint and
  dataset. A dataset rendered under a different preset, resolution or shader is
  refused on load rather than mixed in.

The task files are not changed for this. `utils/pixel_env.install_pixel_env`
runs before `gym.make` and swaps the registered task class for a subclass
built from the mixins in `utils/task_adapters/`: the generic part mounts the
wrist camera, adds `tcp_pose` to the rgb observation and, for the CabinetPour
stage tasks, latches success after one step; the per-family mixins carry what
a task needs under `physx_cuda` and `num_envs > 1` (deferred load-time drive
targets, kinematic holds, render-visibility workarounds, per-env layouts). On
CPU with one env every adapter defers to the task, and the state-based envs
are untouched.

##### Rendering the demos

The offline data is an HDF5 file in the `utils/pixel_dataset.py` schema. The
demos were teleoperated without these cameras, so they are re-rendered frame by
frame from the environment states stored in the LeRobot dataset:

```bash
python -m data_gen.render_pixel_demos \
  --demo-path data/sharpa/JuicePreparationOpenFridge-Sharpa-v1 \
  --env-id JuicePreparationOpenFridge-Sharpa-v1 \
  --robot-uids panda_sharpa_hand_left_angled_mobile_base \
  --control-mode pd_joint_pos --sim-backend physx_cuda \
  --camera-preset cabinet_fixed --image-size 128 \
  --max-episode-steps 500 \
  --save-video --out data/pixels/open_fridge_demos_100_128.h5
```

`obs` and `next_obs` of every row are rendered from their own recorded state
(`observation.environment_state` / `next.observation.environment_state`);
nothing is simulated forward, so there is no set-state-vs-execution gap in the
TD target. Notes:

- `--sim-backend physx_cpu` is required for tasks whose recorded frames toggle a
  render body's visibility (e.g. `CabinetPouringPickBottle-Sharpa-v1`): the batched
  GPU renderer refuses that. Otherwise prefer `physx_cuda`.
- A long render can exhaust the Vulkan device. `--skip N --append` continues an
  existing file, so a dataset can be rendered in chunks of a few episodes, one
  process each.
- `--save-video` writes both camera views of episode 0 side by side. Look at it
  before spending training time.
- 100 episodes at 128x128 with two cameras is 0.3-1.5 GB gzipped, depending on
  episode length.

##### RLPD from pixels

The algorithm is `train_rlpd.py` unchanged: symmetric offline/online sampling, a
LayerNorm critic ensemble with a random-subset min target, entropy backup off.
Only the observation differs, and the visual encoder is trained by the critic
loss alone (the actor consumes the detached latent). Prior data is optional:
without `--demo-data-path` the run is online-only and `offline_mix_fraction` is
forced to 0.

```bash
python train/train_rlpd_pixels.py \
  --env-id JuicePreparationOpenFridge-Sharpa-v1 \
  --robot-uids panda_sharpa_hand_left_angled_mobile_base \
  --control-mode pd_joint_pos \
  --demo-data-path data/pixels/open_fridge_demos_100_128.h5 \
  --camera-preset cabinet_fixed --image-size 128 \
  --reward-mode sparse --reward-bias -1 \
  --delta-action --delta-action-scale 0.2 \
  --encoder-arch d4pg --latent-dim-per-cam 64 \
  --encoder-grad-source critic_only --augmentation --augment-pad 4 \
  --max-episode-steps 500 --num-eval-steps 500 \
  --num-envs 1 --utd-ratio 1 --num-qs 5 --num-min-qs 2 --gamma 0.999 \
  --batch-size 256 --buffer-size 200000 --start-steps 5000 \
  --total-env-steps 600000 --eval-freq 20000 --save-freq 20000 \
  --num-eval-envs 1 --num-eval-episodes 20 --eval-reconfiguration-freq 0 \
  --exp-name open_fridge_rlpd_pixels --track
```

- **`--delta-action`** is the residual joint-position control of the
  state-based CabinetPour runs. `DeltaActionWrapper` turns a `[-1, 1]` action
  into `a_t = clip(a_{t-1} + u * scale)`; the previous command is appended to
  `proprio` (images are untouched) and the offline dataset is rewritten into the
  same parameterization at load time (`to_delta_image_dataset`), so both halves
  of a batch mean the same thing. In evaluate mode `--demo-data-path` is still
  needed with this flag: the first command of an episode comes from the demos.
- **`--reward-bias -1`** turns the `{0, 1}` success into a per-step cost, so the
  return prefers finishing early; `--gamma 0.999` keeps that signal alive over
  the longest demos (~580 steps).
- **`--reward-info-key door_open_now`** (CabinetPouringOpenCabinet-Sharpa-v1) replaces the
  task's latched stage flag by the instantaneous predicate from `info`.
- **UTD.** `utd_ratio` gradient updates run per iteration and an iteration
  advances `num_envs` env steps, so gradient steps per env step is
  `utd_ratio / num_envs`.
- **`--num-envs 1`.** Several subtasks only build with one env: the ones that
  need `--sim-backend physx_cpu`, and the refrigerator / box tasks, which refuse
  `num_envs > 1` in their own `__init__`. The eval env is sized separately, so
  those also need `--num-eval-envs 1`.
- **Terminal semantics.** Every demo terminates on success and online
  collection honours `terminated = info["success"]`, so both halves of a batch
  agree that success is absorbing. Only time-limit truncations bootstrap.
- **Memory.** Frames are uint8 in host RAM and `next_obs` is shared by index; a
  2e5 buffer is ~19 GiB at 128x128x3 with two cameras.
- **Encoders.** `--encoder-arch d4pg` is the 4-layer CNN trained from scratch
  (all reported runs). `--encoder-arch resnet10` is a ResNet-10 with spatial
  softmax (GroupNorm, 4 groups; needs `--image-size >= 64`):
  `--freeze-backbone` trains only the head, `--encoder-pretrained-path` loads a
  backbone state dict in this module's parameter names, and
  `--backbone-lr-scale` / `--backbone-warmup-steps` slow the backbone down so a
  loaded initialization survives the noisy early critic loss.
  `losses/backbone_drift` reports how far it has moved from its initial weights.

##### PPO from pixels

Online PPO on the same observation, anchored on demonstrations: under a sparse
`{0, 1}` reward a random policy never sees a return. The demos enter twice,
through two different datasets:

- **BC anchor** on the actor, from the *rendered* HDF5 (`--bc-data-path`; BC
  needs the images), with a pure-BC warmup before the first rollout
  (`--bc-warmup-steps`). The weight decays from `bc_coef` to `bc_min_coef`.
- **Demo-state resets** from the *LeRobot* dataset (`--demo-init-data-path`,
  where the physical `environment_state` lives). A fraction of episodes starts
  from a restored demo state, chosen by a reverse curriculum that begins near
  the end of each demo and walks back to its start. The curriculum clock only
  advances while the recent training success rate is at least
  `--demo-init-advance-success`; the reset probability decays to
  `--demo-init-min-prob`.

Beyond the observation, this differs from `train_ppo.py` in: only time-limit
truncations bootstrap (success is absorbing and worth exactly 1); no reward
scale or bias, so values live in `[0, 1]`; absolute `pd_joint_pos` control with
the actor's tanh spanned by the demo actions' per-joint range
(`--action-scale-source demos`); separate actor and critic encoders, each
trained by its own loss; random shift on the value branch and the BC batch
only; a smaller actor learning rate and a KL budget that stops only the policy
(`--kl-stop-scope policy`); and a rollout kept as uint8 on the training device
(64 envs x 200 steps is 2.4 GiB).

```bash
python train/train_ppo_pixels.py \
  --env-id CabinetPouringOpenCabinet-Sharpa-v1 \
  --robot-uids panda_sharpa_hand_left_angled_mobile_base \
  --control-mode pd_joint_pos --reward-mode sparse \
  --camera-preset cabinet_fixed --image-size 128 \
  --max-episode-steps 400 \
  --num-envs 64 --num-steps 200 --num-minibatches 32 --update-epochs 1 \
  --total-timesteps 5000000 --stop-at-step 1000000 \
  --learning-rate 3e-4 --actor-learning-rate 3e-5 \
  --gamma 0.99 --gae-lambda 0.95 --clip-coef 0.2 \
  --target-kl 0.2 --kl-stop-scope policy --recompute-old-logprob \
  --ent-coef 0.0 --vf-coef 0.5 --max-grad-norm 0.5 --actor-logstd-init -3.5 \
  --action-scale-source demos --action-scale-margin 0.1 \
  --hidden-dims 512 512 512 512 \
  --encoder-arch d4pg --latent-dim-per-cam 128 \
  --encoder-sharing separate_per_head --encoder-grad-source actor_and_critic \
  --augmentation --augment-pad 4 \
  --use-bc-loss --bc-data-path data/pixels/door_open_demos_100_128.h5 \
  --bc-coef 1.0 --bc-min-coef 0.05 --bc-decay-steps 0.5 \
  --bc-batch-size 256 --bc-loss-type mse --bc-val-fraction 0.1 \
  --bc-warmup-steps 20000 --bc-warmup-batch-size 256 \
  --use-demo-init-states \
  --demo-init-data-path data/sharpa/CabinetPouringOpenCabinet-Sharpa-v1 \
  --demo-init-prob 1.0 --demo-init-min-prob 0.1 --demo-init-decay-steps 0.7 \
  --demo-init-curriculum progress \
  --demo-init-curriculum-start-frac 0.2 --demo-init-curriculum-steps 0.5 \
  --demo-init-advance-success 0.5 --demo-init-success-window 200 \
  --eval-reconfiguration-freq 0 \
  --num-eval-envs 25 --num-eval-episodes 50 --num-eval-steps 400 \
  --eval-freq 100000 --save-freq 200000 \
  --exp-name cabinet_pour_door_open_ppo_pixels --track
```

`--stop-at-step` ends the run early while every schedule (BC decay, demo-init
decay, curriculum) stays on the `--total-timesteps` clock. PPO needs the GPU
backend for its 64 collection envs, so the subtasks that only run on
`physx_cpu` cannot be trained with this configuration.

##### Evaluating a pixel checkpoint

Both trainers evaluate with `--evaluate --checkpoint <ckpt>` and the same
environment, camera and encoder flags as training (the checkpoint carries the
`camera_config_hash` and refuses a mismatched preset). RLPD writes
`best_ckpt.pt`, `ckpt_<env_step>.pt` and `final_ckpt.pt`; PPO writes
`ckpt_best.pt`, `ckpt_<step>.pt` and `final_ckpt.pt`. The reported number is
`eval_success_once_mean`: an episode counts as a success if the task ever
reported success during it. PPO's argument check still expects the inputs of a
training run: when evaluating, pass `--bc-warmup-steps 0 --no-use-demo-init-states`
together with `--bc-data-path` (the action scale is read from the demos).

```bash
# RLPD: 50 episodes, one env, no partial reset
python train/train_rlpd_pixels.py --evaluate --checkpoint runs/<run>/best_ckpt.pt \
  <env / camera / encoder / delta-action / --demo-data-path flags as above> \
  --seed 1 --num-eval-envs 1 --num-eval-episodes 50 --num-eval-steps 500 \
  --eval-reconfiguration-freq 0 [--capture-video]

# PPO: 50 envs x one episode
python train/train_ppo_pixels.py --evaluate --checkpoint runs/<run>/ckpt_best.pt \
  <env / camera / encoder flags as above> \
  --seed 1 --num-eval-envs 50 --num-eval-episodes 50 --num-eval-steps 400 \
  --eval-reconfiguration-freq 0 [--capture-video]
```

`--capture-video` renders an extra camera on top of the policy cameras; a few
CabinetPour stages cannot record at all. `--scene-background <id>` evaluates
the checkpoint on one of the native RoboCasa kitchen surfaces
(`utils.dex_scene_eval.BACKGROUND_IDS`, see
[howto_cross_scene_eval.md](../docs/howto_cross_scene_eval.md)) while keeping
the training cameras; it requires `--sim-backend physx_cpu`, so episodes run
serially.

### Diffusion BC (`train_diffusion.py`)

MLP backbone:
```
python train/train_diffusion.py \
  --env_id CabinetPouringOpenCabinet-Sharpa-v1 --control_mode pd_ee_delta_pose \
  --demo_data_path teleop_lerobot/env_CabinetPouringOpenCabinet-Sharpa-v1__filtered_bc \
  --backbone mlp --mlp_dims 1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 \
  --residual_style --use_layernorm \
  --n_obs_steps 2 --horizon 16 --n_action_steps 8 \
  --num_epochs 500 --batch_size 256 --learning_rate 1e-4 \
  --num_eval_envs 16 --num_eval_steps 200 --eval_freq 25 \
  --exp_name bc_mlp_s1 --seed 1
```

U-Net backbone (swap `--backbone unet`; drop the MLP-only flags):
```
python train/train_diffusion.py \
  --env_id CabinetPouringOpenCabinet-Sharpa-v1 --control_mode pd_ee_delta_pose \
  --demo_data_path teleop_lerobot/env_CabinetPouringOpenCabinet-Sharpa-v1__filtered_bc \
  --backbone unet \
  --n_obs_steps 2 --horizon 16 --n_action_steps 8 \
  --num_epochs 500 --batch_size 256 --learning_rate 1e-4 \
  --num_eval_envs 16 --num_eval_steps 200 --eval_freq 25 \
  --exp_name bc_unet_s1 --seed 1
```
Checkpoints land in `runs/<exp_name>/` (`ckpt_<epoch>.pt`, `final_ckpt.pt`),
each storing `{model, normalizer, backbone, ...}`. The U-Net (~64M) fits demos
much better than the MLP (~11M) but trains slower.

### DPPO fine-tune (`train_dppo.py`)

Start from a BC checkpoint via `--base_policy_path`. The normalizer and backbone
are inherited from that checkpoint. The config defaults already encode the
**winning ShadowHandBallGrasp recipe** (sparse-hold + BC anchor + loose clip), so
the command is minimal:
```
python train/train_dppo.py \
  --base_policy_path runs/bc_forearm_unet_s1/final_ckpt.pt --backbone unet
# key defaults: reward_mode=sparse, ft_denoising_steps=5, use_bc_loss=True bc_coef=0.03,
#   clip_ploss_coef=0.03, reward_scale=0.1, clip_vloss_coef=2.0, total_timesteps=5M
```
For an MLP base policy, set `--backbone mlp` and append the **same** MLP flags
used in BC (`--mlp_dims ... --residual_style --use_layernorm`).

> **ShadowHand ball-grasp recipe (do not use dense).** Use **sparse** reward —
> it is HOLD-type (success does not terminate; reward = #steps in success). Dense
> **fails**: its "grasp+hover" plateau is a local optimum (reward↑ while success↓).
> Two things make fine-tuning work: (a) `--use_bc_loss` BC anchor prevents the
> success collapse; (b) `--clip_ploss_coef 0.03` — **not** 0.01: too tight makes
> clipfrac ~90% and freezes the gradient. Sparse-hold returns are large, so pair
> `--reward_scale 0.1` with `--clip_vloss_coef 2.0` to keep the value bounded.
> Result: 64env success_once 0.375 (BC) → ~0.79. See `runs/README_dppo_ballgrasp.md`.

> **Memory:** U-Net DPPO at `--num_envs 128` uses ~26 GB (backprop through
> `ft_denoising_steps` × the backbone). Drop `--num_envs` (e.g. 64) if you OOM.
> MLP DPPO is ~6× cheaper/faster.

### PI0.5 OpenPI LoRA (`train_pi05.py`)

The OpenPI/JAX baseline replaces the legacy LeRobot full-parameter SFT/GRPO
implementation. Fine-tune with `train/train_pi05.py` and evaluate
with the same entry point's `--evaluate` flag. The old SFT, GRPO and separate
evaluation commands are no longer supported; old `final_policy/` packages and
Torch checkpoints are not compatible with this baseline.

See [PI05.md](../docs/PI05.md) for pinned installation,
data preparation, training, checkpoint resume and complete launch examples.
The model runs in a Python 3.11 OpenPI environment. Native ManiSkill evaluation
may stay in its existing environment and launch the private inference child
with `--policy-python`.

---

## Eval

The `--evaluate --checkpoint <path>` convention below applies to PPO and
Diffusion. DPPO has the caveat described in its subsection. PI0.5 uses the
same `train_pi05.py --evaluate --model-path <run>` entry point.

### PI0.5 asynchronous task-level evaluation

```bash
python train/train_pi05.py --evaluate \
  --model-path /path/to/runs/task-lora \
  --output-dir /path/to/eval/task-standard \
  --policy-python /path/to/pi05-venv/bin/python
```

The frozen model predicts 30-step chunks. A new request is issued when fewer
than 24 actions remain; predictions are aligned to the observation control step
and overlapping absolute targets are blended. Reset invalidates stale responses
and refreshes the live robot/controller binding. Native task success predicates
and episode horizons remain unchanged.

Use `--scene-background s1`, `s2` or `s6` in distinct output directories for the
shared background scenes. Eval50 uses seeds 10001-10050; infrastructure errors
are unscored, and `--resume` only continues pending seeds under the same identity.
A final success rate is reported only after all 50 episodes complete.

### PPO

```
python train/train_ppo.py --evaluate \
  --checkpoint runs/<run_name>/final_ckpt.pt \
  --env_id CabinetPouringOpenCabinet-Sharpa-v1
```

### Offline RL

#### State BC

```bash
python train/train_bc.py --evaluate \
  --checkpoint runs/<bc-run>/final_ckpt.pt \
  --env_id CabinetPouringOpenCabinet-Sharpa-v1 --control_mode pd_ee_delta_pose \
  --num_eval_episodes 100
```

Keep `--drop_object_pose` consistent with training when that projection was
used. The checkpoint restores its observation normalization statistics.

#### FQE

FQE does not produce a policy and has no environment eval entry point. Its
supported evaluation is the periodic `q_vs_mc_mse`/`q_vs_mc_corr` comparison
against demonstration return-to-go during `train_fqe.py`.

#### AWR

```bash
python train/train_awr.py --evaluate \
  --checkpoint runs/<awr-run>/final_ckpt.pt \
  --env_id CabinetPouringOpenCabinet-Sharpa-v1 --control_mode pd_ee_delta_pose \
  --num_eval_episodes 100
```

The AWR checkpoint embeds the frozen FQE Q, so `--fqe_checkpoint` is not needed
for evaluation.

#### CQL

```bash
python train/train_cql.py --evaluate \
  --checkpoint runs/<cql-run>/final_ckpt.pt \
  --env_id <training-env-id> --control_mode <training-control-mode> \
  --num_eval_episodes 100
```

Evaluate a CQL checkpoint with the same `--env_id`, `--control_mode`,
horizon and task arguments it was trained with.

#### IQL

```bash
python train/train_iql.py --evaluate \
  --checkpoint runs/<iql-run>/final_ckpt.pt \
  --env_id CabinetPouringOpenCabinet-Sharpa-v1 --control_mode pd_ee_delta_pose \
  --num_eval_episodes 100
```

#### TD3+BC

```bash
python train/train_td3bc.py --evaluate \
  --checkpoint runs/<td3bc-run>/final_ckpt.pt \
  --env_id CabinetPouringOpenCabinet-Sharpa-v1 --control_mode pd_ee_delta_pose \
  --num_eval_episodes 100
```

Evaluation requires a full TD3+BC `--checkpoint`; a State BC checkpoint passed
through `--bc_checkpoint` is only an actor initialization for training.

### Offline-to-online RL

#### OPT

```bash
python train/train_opt.py --evaluate \
  --checkpoint runs/<opt-run>/final_ckpt.pt \
  --env_id CabinetPouringOpenCabinet-Sharpa-v1 --control_mode pd_ee_delta_pose \
  --num_eval_episodes 100
```

Passing `--pretrained_checkpoint` instead evaluates the offline IQL warm-start
through the OPT model before either online phase; use `--checkpoint` for the
fine-tuned OPT result.

#### SAC

```bash
python train/train_sac.py --evaluate \
  --checkpoint runs/<sac-run>/final_ckpt.pt \
  --env_id CabinetPouringOpenCabinet-Sharpa-v1 --control_mode pd_ee_delta_pose \
  --num_eval_episodes 100
```

As with OPT, `--pretrained_checkpoint` can evaluate the transferred offline
actor, while `--checkpoint` evaluates a completed/resumed SAC agent.

#### RLPD

```bash
python train/train_rlpd.py --evaluate \
  --checkpoint runs/<rlpd-run>/final_ckpt.pt \
  --env_id CabinetPouringOpenCabinet-Sharpa-v1 --control_mode pd_ee_delta_pose \
  --num_eval_episodes 100
```

For every state-policy evaluator, the environment, `robot_uids` when relevant,
controller, action bounds, observation layout, reward mode, and episode horizon
should match the training contract. `--capture_video` writes videos through the
same ManiSkill evaluation wrapper used during periodic training evaluation.

### Diffusion BC

```
python train/train_diffusion.py --evaluate \
  --checkpoint runs/bc_unet_s1/final_ckpt.pt \
  --env_id CabinetPouringOpenCabinet-Sharpa-v1 --control_mode pd_ee_delta_pose --reward_mode normalized_dense \
  --backbone unet --n_obs_steps 2 --horizon 16 --n_action_steps 8 \
  --num_eval_envs 64 --num_eval_steps 200 --num_eval_episodes 64 --seed 1
```
This also writes a `runs/<run>/test_videos/0.mp4` grid (real-time `control_freq`
fps). For an MLP checkpoint, mirror the BC backbone flags.

### DPPO

Eval-only metrics via training entrypoint (note: `train_dppo.py --evaluate` loads
the BC base via `--base_policy_path`, **not** the fine-tuned weights):
```
python train/train_dppo.py --evaluate \
  --base_policy_path runs/dppo_unet_dense_s1/ckpt_97.pt --backbone unet \
  --env_id CabinetPouringOpenCabinet-Sharpa-v1 --control_mode pd_ee_delta_pose --reward_mode normalized_dense \
  --n_obs_steps 2 --horizon 16 --n_action_steps 8 --max_episode_steps 200 \
  --num_eval_envs 64 --num_eval_steps 200 --seed 1
```

This repository currently has no `scripts/render_dppo.py`; therefore the guide
does not claim a supported fine-tuned DPPO video command. Add and test a helper
before documenting that workflow.

---

## Config

Each trainer is driven by a `tyro` dataclass under `train/train_configs/`; every
field is a `--flag`. Run `python train/<trainer>.py --help` for the full list.

| Trainer | Config file | Key fields |
|---|---|---|
| `train_ppo.py` | `train_configs/ppo_config.py` | `use_bc_loss`, `use_demo_init_states`, `bc_data_path`/`demo_init_data_path`, `reward_mode`, `total_timesteps` |
| `train_bc.py` | `train_configs/bc_config.py` | `demo_data_path`, `success_only`, `drop_object_pose`, `bc_gripper_weight`, `hidden_dims`, `total_updates` |
| `train_fqe.py` | `train_configs/fqe_config.py` | `demo_data_path`, `gamma`, `tau`, `hidden_dims`, `total_updates`, `mc_eval_freq`/`mc_eval_batch_size` |
| `train_awr.py` | `train_configs/awr_config.py` | required training `fqe_checkpoint`, `demo_data_path`, `iql_expectile`, `iql_beta`, `iql_adv_clip`, `total_updates` |
| `train_cql.py` | `train_configs/cql_config.py` | `demo_data_path`, `extra_npz_paths`, `cql_alpha`/`cql_temperature`/`cql_num_actions`, `bc_warmup_steps`, `actor_bc_weight`, `bc_only`, `total_updates` |
| `train_iql.py` | `train_configs/iql_config.py` | `demo_data_path`, `success_only`, `extra_npz_paths`, `iql_expectile`/`iql_beta`/`iql_adv_clip`, `bc_warmup_steps`, `actor_bc_weight`, `total_updates` |
| `train_td3bc.py` | `train_configs/td3bc_config.py` | `demo_data_path`, `success_only`, optional actor-only `bc_checkpoint`, `policy_freq`, `policy_noise`/`noise_clip`, `alpha`, `total_updates` |
| `train_opt.py` | `train_configs/opt_config.py` | offline-IQL `pretrained_checkpoint`, `opt_env_steps`/`opt_grad_steps`, `meta_lr`, `kappa_start`/`kappa_end`, `offline_mix_fraction`, `total_env_steps` |
| `train_sac.py` | `train_configs/sac_config.py` | IQL/CQL `pretrained_checkpoint`, `start_steps`, `updates_per_iter`, `offline_mix_fraction`, `entropy_alpha`/`autotune_entropy`, decaying BC anchor, `total_env_steps` |
| `train_rlpd.py` | `train_configs/rlpd_config.py` | `demo_data_path`, `robot_uids`, `start_steps`, `joint_warmup_delta`, `utd_ratio`, `offline_mix_fraction`, `num_qs`/`num_min_qs`, `total_env_steps` |
| `train_rlpd_pixels.py` | `train_configs/rlpd_pixels_config.py` | optional `demo_data_path`, `delta_action`/`delta_action_scale`, `reward_info_key`, `camera_preset`/`image_size`, `encoder_arch`/`latent_dim_per_cam`/`encoder_grad_source`, `freeze_backbone`/`update_bn_stats`/`encoder_pretrained_path`, `augmentation`/`augment_pad`, `num_envs`/`utd_ratio`/`buffer_size`, `total_env_steps` |
| `train_ppo_pixels.py` | `train_configs/ppo_pixels_config.py` | `bc_data_path` (rendered HDF5) + `demo_init_data_path` (LeRobot), `camera_preset`/`image_size`, `encoder_arch`/`encoder_sharing`, `actor_learning_rate`, `target_kl`/`kl_stop_scope`, `action_scale_source`, `bc_coef`/`bc_warmup_steps`, `demo_init_*` curriculum, `num_envs`/`num_steps`, `total_timesteps`/`stop_at_step` |
| `train_diffusion.py` | `train_configs/diffusion_config.py` | `backbone`, `mlp_dims`/`down_dims`, `n_obs_steps`/`horizon`/`n_action_steps`, `denoising_steps`, `predict_epsilon`, `obs/action_normalization_mode`, `num_epochs`, `demo_data_path` |
| `train_dppo.py` | `train_configs/dppo_config.py` | `base_policy_path`, `backbone`, `ft_denoising_steps`, `num_envs`/`num_steps`, `actor_lr`/`critic_lr`, `clip_ploss_coef`, `reward_mode`, `max_episode_steps`, `total_timesteps` |
| `train_pi05.py` | `train_configs/pi05_config.py` | training `dataset_path`, `data_contract`, `binding`, `base_model_path`, `base_manifest`; evaluation `evaluate`, `model_path`, `policy_python`, `scene_background`; shared `output_dir`, `resume` |

Notes:
- **State-policy checkpoint semantics:** `checkpoint` resumes/evaluates the same
  agent. `fqe_checkpoint` initializes AWR's frozen Q;
  `pretrained_checkpoint` initializes OPT/SAC from an offline agent; and
  `bc_checkpoint` initializes only the TD3+BC actor.
- **State-policy normalization:** observation mean/std and action bounds live in
  checkpoints. AWR deliberately inherits FQE's statistics, while OPT/SAC copy
  matching offline weights and statistics during warm start.
- **Update units:** BC/FQE/AWR/CQL/IQL/TD3+BC budgets and checkpoint intervals
  are optimizer updates. OPT/SAC/RLPD budgets and log/eval/save intervals are
  environment steps; their gradient work is controlled separately by
  `updates_per_iter` or `utd_ratio`.
- **Backbone parity is mandatory** between a BC run and the DPPO run that loads
  it — the same `--backbone` plus, for MLP, the same `--mlp_dims/--residual_style/--use_layernorm`.
- **Normalization** is computed once in BC and stored in the checkpoint; DPPO
  inherits it (do not re-specify).
- **Logging:** PPO/Diffusion/DPPO expose their existing SwanLab/TensorBoard
  options. PI0.5 currently writes durable `metrics.jsonl`, immutable run
  metadata, resumable optimizer checkpoints and per-seed evaluation records;
  it does not claim SwanLab or
  TensorBoard integration.
- **Reward modes:** `normalized_dense` (shaped, default for RL), `sparse`
  (success-only), set via `--reward_mode`. `return` is only comparable within
  the same reward mode; use `success_*` to compare across modes.

## Vision / RGB observations

The dexterous-hand grasping tasks carry static world cameras in the env — a `base_camera`
(RoboTwin-style head view) and an `external_camera` (observer overview) — plus a
palm-mounted wrist camera on each hand, all 320×240 / 58° FOV (RealSense D435 spec):

- Single right hands (`panda_{shadow,sharpa,wuji}_hand_right`): one `wrist_camera`.
- Dual-arm bases (`panda_dual_{shadow,sharpa,wuji}_hand_trapezoid_base`): `left_wrist_camera`
  + `right_wrist_camera`, one per hand.

PI0.5 evaluation uses the shared fixed 256x256 policy-camera profile and checks
its live named joint/controller binding after reset. Dataset preparation records
the image slots and shapes; arbitrary robot sensor streams are not interchangeable
with that model input contract. See [the camera guide](../docs/camera_setup/README.md).

The default registered envs remain `obs_mode="state"`; request images by passing
`obs_mode="rgb"` at `gym.make` time. Camera streams appear under `obs["sensor_data"][cam]["rgb"]`.

Image recording uses the same interface as the gripper (`qiyan_tasks`): camera RGB is
extracted via `utils/image_observation.py` (`extract_camera_rgb`, aliases) into LeRobot
fields `observation.images.base_0_rgb` / `observation.images.left_wrist_0_rgb`, and
written with `LeRobotEpisodeRecorder.add_episode(..., image_sequences={field: (T,H,W,3) uint8})`
(stored as LeRobot `image` features).
