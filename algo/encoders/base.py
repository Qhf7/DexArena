"""Encoder interface and configuration shared by every visual encoder.

The agent only sees ``z = encoder(obs)`` and ``encoder.latent_dim``; proprio is
concatenated by the caller.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


ENCODER_ARCHS = ("d4pg", "resnet10")
GRAD_SOURCES = ("critic_only", "actor_and_critic", "frozen", "bc_loss")
SHARING_MODES = ("shared", "separate_per_head")
NORM_LAYERS = ("batch", "group", "none")


@dataclass
class EncoderConfig:
    """Switches for the visual stack. ``grad_source`` and ``sharing`` are read by
    the agent, not by the encoder module."""

    arch: str = "d4pg"
    latent_dim_per_cam: int = 64
    grad_source: str = "critic_only"
    sharing: str = "shared"
    augmentation: bool = True
    augment_pad: int = 4
    freeze_backbone: bool = False
    # BatchNorm backbones only: True keeps the running stats updating, False
    # keeps the checkpoint's.
    update_bn_stats: bool = True
    norm_layer: str = "batch"
    # GroupNorm group count for `norm_layer="group"`; must match the converted
    # weights (4 for the released ResNet-10), a mismatch loads without error.
    norm_groups: int = 4
    # Local path to converted ResNet-10 weights; None = random init.
    pretrained_path: str | None = None
    # LayerNorm + tanh on the latent.
    latent_layer_norm: bool = True

    def validate(self) -> "EncoderConfig":
        if self.arch not in ENCODER_ARCHS:
            raise ValueError(f"`arch` must be one of {ENCODER_ARCHS}, got {self.arch!r}.")
        if self.grad_source not in GRAD_SOURCES:
            raise ValueError(f"`grad_source` must be one of {GRAD_SOURCES}, got {self.grad_source!r}.")
        if self.sharing not in SHARING_MODES:
            raise ValueError(f"`sharing` must be one of {SHARING_MODES}, got {self.sharing!r}.")
        if self.norm_layer not in NORM_LAYERS:
            raise ValueError(f"`norm_layer` must be one of {NORM_LAYERS}, got {self.norm_layer!r}.")
        if self.latent_dim_per_cam <= 0:
            raise ValueError(f"`latent_dim_per_cam` must be positive, got {self.latent_dim_per_cam}.")
        if self.augment_pad < 0:
            raise ValueError(f"`augment_pad` must be non-negative, got {self.augment_pad}.")
        if self.norm_groups <= 0:
            raise ValueError(f"`norm_groups` must be positive, got {self.norm_groups}.")
        return self


class VisualEncoder(nn.Module):
    """obs dict -> (B, latent_dim) float32.

    ``obs[cam_key]`` is ``(B, H, W, 3)`` uint8, already augmented by the caller.
    Subclasses set ``self.latent_dim``.
    """

    latent_dim: int

    def __init__(self, camera_keys: Sequence[str]):
        super().__init__()
        self.camera_keys = tuple(camera_keys)
        self.latent_dim = 0

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def diagnostics(self) -> dict[str, float]:
        """Scalars for logging; empty unless overridden."""
        return {}

    # ---- fine-tuning support -------------------------------------------------

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Split parameters into ``{"backbone": [...], "head": [...]}``; default:
        no backbone, everything is head."""
        return {"backbone": [], "head": [p for p in self.parameters()]}

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Enable/disable gradients through the pre-trained backbone; no-op
        without one."""
        return None

    @contextmanager
    def frozen_norm_stats(self):
        """Encode without updating BatchNorm running statistics (``no_grad`` alone
        does not stop them)."""
        modules = [
            module
            for module in self.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training
        ]
        for module in modules:
            module.eval()
        try:
            yield
        finally:
            for module in modules:
                module.train()


def to_float_chw(images: torch.Tensor) -> torch.Tensor:
    """(B, H, W, 3) uint8 -> (B, 3, H, W) float32 in [0, 1]."""
    if images.dtype == torch.uint8:
        images = images.float().div_(255.0)
    elif images.dtype != torch.float32:
        images = images.float()
    return images.permute(0, 3, 1, 2).contiguous()
