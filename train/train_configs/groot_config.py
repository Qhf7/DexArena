"""Configuration for the GR00T baseline (NVIDIA Isaac GR00T N1.7-3B).

Mirrors the repo's existing config style (``dppo_config.py`` / ``diffusion_config.py``) with a
single tyro :class:`Args` dataclass. ``train/train_groot.py`` is finetune-first: by default it
**finetunes** GR00T (dispatching GR00T's ``launch_finetune.py`` via ``torchrun``), and with
``--evaluate`` it runs our ManiSkill ``evaluate_policy`` loop where :class:`algo.groot.GrootAgent`
loads the checkpoint **in-process** (unified env — no inference server). Eval field names mirror
``dppo_config`` so the success-rate row is directly comparable to DP / PPO / DPPO.

Task-agnostic: pick the task with ``--env_id``.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import tyro


@dataclass
class Args:
    # Experiment metadata
    exp_name: Optional[str] = None
    """name of this experiment (run name and log directory)"""
    prompt: Optional[str] = None
    """language instruction to evaluate under. For a registered dex task it must be one the
    task registers (utils/dex_tasks.py); default is that task's canonical instruction. Outside
    the table it is free text and falls back to `exp_name`."""
    instruction_sampling: str = "registered"
    """'registered' draws one of the task's instructions per episode (utils/dex_tasks.py),
    'canonical' runs every episode on `prompt`. Numbers measured before 2026-09-18 were
    evaluated on whatever string `exp_name` held; pass 'canonical' for a single fixed one."""
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    gpu_id: Optional[int] = None
    """CUDA device index for the ManiSkill GPU backends (eval side)"""
    evaluate: bool = False
    """default is finetune; pass --evaluate to run the ManiSkill eval loop instead"""

    # Environment identity (drives obs/action dims + camera mounts)
    env_id: str = tyro.MISSING
    control_mode: Optional[str] = None
    reward_mode: Optional[str] = None
    """environment reward mode (e.g. `sparse`); `None` uses the env default"""
    sim_backend: Optional[str] = None
    robot_uids: Optional[str] = None
    """override the env's robot (e.g. `panda_wristcam` to match a wrist-cam trained policy)"""

    # Data
    demo_data_path: Optional[str] = None
    """LeRobot demo dataset root used by utils/groot_utils/replay_to_vla_dataset.py"""

    # Cameras -> GR00T modality (2 views: base/head + wrist).
    base_camera_field: str = "observation.images.base_0_rgb"
    wrist_camera_field: str = "observation.images.left_wrist_0_rgb"

    # ---- Finetune (GR00T's launch_finetune.py, dispatched via torchrun) ----
    base_model_path: str = "/mnt/ceph2/ckpt/GR00T-N1.7-3B"
    """base GR00T-N1.7-3B checkpoint to finetune from"""
    dataset_path: Optional[str] = None
    """LeRobot v2.1 VLA dataset to finetune on (required for finetune)"""
    modality_config_path: Optional[str] = None
    """path to the modality config .py registering the NEW_EMBODIMENT layout"""
    output_dir: Optional[str] = None
    """where finetune writes checkpoints (default: runs/<run_name>)"""
    num_gpus: int = 4
    """GPUs for finetune (torchrun --nproc_per_node)"""
    global_batch_size: int = 4
    dataloader_num_workers: Optional[int] = None
    """dataloader workers. Set 0 to read single-process: torchcodec's video decode has
    been seen to fail under concurrent workers with "no more frames left to decode" on a
    dataset whose parquet row counts and video frame counts all match."""
    resume_from_checkpoint: Optional[str] = None
    """continue a run from its own checkpoint, restoring optimizer state, LR schedule and
    step count. Pointing --base_model_path at a checkpoint only reloads weights."""
    gradient_accumulation_steps: int = 1
    """accumulate this many micro-batches per optimizer step. The effective batch is
    global_batch_size * this; GR00T's default learning rate assumes a large batch, so a
    small per-GPU batch without accumulation makes every gradient mostly noise."""
    learning_rate: Optional[float] = None
    """override GR00T's default LR. Scale it with the effective batch, not the micro-batch."""
    max_steps: int = 10000
    save_steps: int = 1000
    save_total_limit: int = 2
    tune_llm: bool = True
    tune_visual: bool = True
    master_port: int = 29541

    # ---- Eval: GR00T policy loaded in-process (no server) ----
    model_path: Optional[str] = None
    """finetuned GR00T checkpoint dir loaded in-process for --evaluate"""
    embodiment_tag: str = "NEW_EMBODIMENT"
    eval_env_extra: Optional[str] = None
    """JSON dict of extra env kwargs for evaluation, e.g.
    '{"visible_goal_site": true, "goal_site_visual_radius": 0.05}'. Must repeat whatever
    the dataset build passed, or the policy is evaluated on a scene it never saw."""
    camera_size: Optional[int] = None
    """override ALL eval sensor cameras to NxN (match the dataset build for heterogeneous-camera envs)"""
    camera_profile: str = "none"
    """'robodex_fixed' evaluates on the accepted fixed base + wrist views (utils/dex_cameras.py);
    must equal what the dataset was rendered with (meta/robodex_render.json)"""
    scene_background: Optional[str] = None
    """Fixed 30-task scene evaluation background (utils.dex_scene_eval)."""
    action_horizon: int = 8
    """number of actions in the chunk GR00T returns per query (executed receding-horizon)"""
    stub: bool = False
    """smoke-test the rgb env + eval harness with a zero-action agent (no GR00T policy needed)"""

    # Eval (names mirror dppo eval args so the comparison table lines up)
    num_eval_envs: int = 1
    """GR00T eval runs one env at a time (policy returns one action chunk per obs)"""
    num_eval_episodes: int = 50
    num_eval_steps: int = 100
    max_episode_steps: Optional[int] = None
    """env episode truncation; defaults to num_eval_steps when unset"""
    eval_partial_reset: bool = False
    eval_reconfiguration_freq: Optional[int] = 1
    capture_video: bool = False
    eval_output_dir: Optional[str] = None
    """where evaluation writes videos/ (with --capture_video), episodes.jsonl and summary.json;
    default runs/<run_name>/eval_videos for the videos only"""

    # Logging
    track: bool = False
    swanlab_project_name: str = "groot"
    swanlab_entity: Optional[str] = None

    # filled at runtime
    env_max_episode_steps: int = 0


def _apply_overrides(args: Args, overrides: Optional[Dict[str, Any]]) -> Args:
    if overrides:
        for key, value in overrides.items():
            if not hasattr(args, key):
                raise ValueError(f"Unknown arg override: {key}")
            setattr(args, key, value)
    if args.gpu_id is not None and args.gpu_id < 0:
        raise ValueError(f"`gpu_id` must be non-negative, got {args.gpu_id}.")
    return args


def parse_args(overrides: Optional[Dict[str, Any]] = None) -> Args:
    return _apply_overrides(tyro.cli(Args), overrides)
