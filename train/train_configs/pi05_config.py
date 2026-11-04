"""Shared training/evaluation arguments and the PI0.5 LoRA baseline configuration."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DEXJOCO_COMMIT = "8d23b0fab23b17a58c4b55f3942e17013aaf8267"


@dataclass(frozen=True)
class TrainingContract:
    dual_arm: bool = False
    batch_size: int = 32
    action_horizon: int = 30
    warmup_steps: int = 10000
    peak_lr: float = 5e-5
    decay_steps: int = 1000000
    decay_lr: float = 5e-5
    max_token_len: int = 250
    seed: int = 1000
    paligemma_variant: str = "gemma_2b_lora"
    action_expert_variant: str = "gemma_300m_lora"
    paligemma_lora_rank: int = 16
    paligemma_lora_alpha: float = 16.0
    action_expert_lora_rank: int = 32
    action_expert_lora_alpha: float = 32.0

    @property
    def action_dim(self):
        return 58 if self.dual_arm else 32

    @property
    def physical_dim(self):
        return 58 if self.dual_arm else 29

    @property
    def steps(self):
        return 60000 if self.dual_arm else 30000

    def validate(self):
        if (
            self.batch_size,
            self.action_horizon,
            self.warmup_steps,
            self.peak_lr,
            self.decay_steps,
            self.decay_lr,
            self.max_token_len,
        ) != (32, 30, 10000, 5e-5, 1000000, 5e-5, 250):
            raise ValueError("Frozen new-baseline contract changed")
        if (
            self.paligemma_variant,
            self.action_expert_variant,
            self.paligemma_lora_rank,
            self.paligemma_lora_alpha,
            self.action_expert_lora_rank,
            self.action_expert_lora_alpha,
        ) != ("gemma_2b_lora", "gemma_300m_lora", 16, 16.0, 32, 32.0):
            raise ValueError("Pinned DexJoCo LoRA contract changed")


@dataclass
class Args:
    """Fine-tune by default; select --evaluate for standard/background Eval50."""

    output_dir: Path
    evaluate: bool = False
    env_id: str | None = None
    dataset_path: Path | None = None
    data_contract: Path | None = None
    binding: Path | None = None
    base_model_path: Path | None = None
    base_manifest: Path | None = None
    model_path: Path | None = None
    policy_python: str = sys.executable
    scene_background: Literal["s1", "s2", "s6"] | None = None
    accumulation_steps: Literal[1, 2] = 1
    resume: bool = False


def parse_args():
    import tyro

    args = tyro.cli(Args)
    required = (
        ("model_path",)
        if args.evaluate
        else ("dataset_path", "data_contract", "binding", "base_model_path", "base_manifest")
    )
    missing = ["--" + key.replace("_", "-") for key in required if getattr(args, key) is None]
    if missing:
        raise ValueError("Missing arguments: " + ", ".join(missing))
    if not args.evaluate and args.scene_background is not None:
        raise ValueError("--scene-background applies only to --evaluate")
    return args
