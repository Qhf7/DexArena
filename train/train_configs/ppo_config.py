from dataclasses import dataclass
from typing import Any, Dict, Optional

import tyro

from utils.paths import resolve_training_paths


@dataclass
class Args:
    # Experiment metadata
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=True`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    gpu_id: Optional[int] = None
    """CUDA device index to use for both PyTorch and ManiSkill GPU backends; defaults to the current CUDA default device"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    swanlab_project_name: str = "ManiSkill"
    """the swanlab's project name"""
    swanlab_entity: Optional[str] = None
    """the entity (team) of swanlab's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""

    # Algorithm specific arguments
    env_id: str = tyro.MISSING
    """the id of the environment"""
    robot_uids: Optional[str] = None
    """robot uid override passed to the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 512
    """the number of parallel environments"""
    num_eval_envs: int = 40
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 300
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 300
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = "pd_joint_pos"
    """the control mode to use for the environment"""
    reward_mode: Optional[str] = None
    """the reward mode to use for the environment (e.g. normalized_dense, sparse); None uses the env default"""
    """the object id to use for the environment (if applicable)"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    anneal_ent_coef: bool = True
    """Toggle entropy coefficient annealing for policy updates"""
    gamma: float = 0.98
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    actor_logstd_init: float = -0.2
    """initial actor log-std (exploration noise on absolute joint targets; e.g. -2.5 -> sigma~0.08 rad)"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.1
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = False
    use_bc_loss: bool = False
    """whether to add BC loss when a BC dataset is available"""
    bc_data_path: Optional[str] = "teleop_lerobot"
    """offline BC dataset path (LeRobot dataset root, parent directory, glob pattern, or legacy .h5 path)"""
    use_demo_init_states: bool = False
    """whether to reset training envs from randomly sampled states in the demonstration dataset"""
    demo_init_data_path: Optional[str] = None
    """demo dataset path used for reset-state sampling; defaults to bc_data_path when unset"""
    demo_init_prob: float = 1.0
    """initial probability of resetting an env from a demo state when demo init is enabled"""
    demo_init_min_prob: float = 0.0
    """minimum probability of demo-state reset after decay"""
    demo_init_decay: bool = True
    """whether to linearly decay the probability of demo-state resets"""
    demo_init_decay_steps: float = 1.0
    """fraction of total_timesteps used for demo-init probability decay"""
    demo_init_curriculum: str = "progress"
    """demo reset-state sampler: 'uniform' or 'progress' (easy-to-hard by demo timestep)"""
    demo_init_curriculum_start_frac: float = 0.3
    """initial fraction of the tail of each demo that is eligible when using progress curriculum"""
    demo_init_curriculum_steps: float = 0.5
    """fraction of total_timesteps used to expand the progress curriculum to the full demo"""
    bc_coef: float = 1.0
    """initial BC loss coefficient"""
    bc_min_coef: float = 0.0
    """minimum BC loss coefficient after decay"""
    bc_decay: bool = True
    """whether to linearly decay BC coefficient"""
    bc_decay_steps: float = 0.5
    """fraction of total_timesteps used for BC coefficient decay (e.g., 1.0 means full training horizon)"""
    bc_batch_size: int = 0
    """BC batch size sampled from dataset each PPO minibatch; <=0 uses minibatch_size"""
    bc_loss_type: str = "mse"
    """BC loss type: 'mse' or 'nll'"""
    bc_warmup_steps: int = 0
    """pure-BC gradient steps to pretrain the actor before PPO starts (0 = disabled)"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def parse_args(overrides: Optional[Dict[str, Any]] = None) -> Args:
    """
    Parse CLI args and optionally override any fields programmatically
    (useful when launching experiments from code rather than the shell).
    """
    args = tyro.cli(Args)
    if overrides:
        for key, value in overrides.items():
            if not hasattr(args, key):
                raise ValueError(f"Unknown arg override: {key}")
            setattr(args, key, value)
    if args.gpu_id is not None and args.gpu_id < 0:
        raise ValueError(f"`gpu_id` must be non-negative, got {args.gpu_id}.")
    resolve_training_paths(args)
    return args
