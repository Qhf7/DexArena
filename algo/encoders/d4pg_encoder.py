"""From-scratch D4PG convolutional encoder.

Per camera, on a 128x128x3 input, all convolutions 3x3 stride 2 with VALID
padding::

    conv 32  -> 63x63    ReLU
    conv 64  -> 31x31    ReLU
    conv 128 -> 15x15    ReLU
    conv 256 ->  7x7     ReLU
    flatten  -> 12544
    Dense    -> latent_dim_per_cam
    LayerNorm + tanh

Cameras use separate trunks and their latents are concatenated, so
``latent_dim = num_cameras * latent_dim_per_cam``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from algo.encoders.base import EncoderConfig, VisualEncoder, to_float_chw


D4PG_CHANNELS = (32, 64, 128, 256)


class D4PGTrunk(nn.Module):
    """Single-camera conv trunk + projection to ``latent_dim``."""

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        latent_dim: int,
        *,
        layer_norm: bool = True,
        channels: Sequence[int] = D4PG_CHANNELS,
    ):
        super().__init__()
        height, width, in_channels = image_shape
        layers: list[nn.Module] = []
        current = int(in_channels)
        for out_channels in channels:
            conv = nn.Conv2d(current, int(out_channels), kernel_size=3, stride=2, padding=0)
            nn.init.orthogonal_(conv.weight, gain=np.sqrt(2))
            nn.init.zeros_(conv.bias)
            layers.extend([conv, nn.ReLU()])
            current = int(out_channels)
        self.conv = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, int(in_channels), int(height), int(width))
            self.conv_output_shape = tuple(self.conv(dummy).shape[1:])
        flat_dim = int(np.prod(self.conv_output_shape))

        self.project = nn.Linear(flat_dim, int(latent_dim))
        nn.init.orthogonal_(self.project.weight, gain=1.0)
        nn.init.zeros_(self.project.bias)
        self.norm = nn.LayerNorm(int(latent_dim)) if layer_norm else nn.Identity()
        self.layer_norm = bool(layer_norm)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # uint8 (B, H, W, 3) -> float (B, 3, H, W) centred at 0, as in RLPD/DrQ.
        x = to_float_chw(images) - 0.5
        x = self.conv(x).flatten(1)
        z = self.norm(self.project(x))
        return torch.tanh(z) if self.layer_norm else z


class D4PGEncoder(VisualEncoder):
    def __init__(
        self,
        camera_keys: Sequence[str],
        image_shape: tuple[int, int, int],
        config: EncoderConfig,
    ):
        super().__init__(camera_keys)
        self.trunks = nn.ModuleDict(
            {
                key: D4PGTrunk(
                    image_shape,
                    config.latent_dim_per_cam,
                    layer_norm=config.latent_layer_norm,
                )
                for key in self.camera_keys
            }
        )
        self.latent_dim = len(self.camera_keys) * int(config.latent_dim_per_cam)
        if config.freeze_backbone:
            self.requires_grad_(False)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([self.trunks[key](obs[key]) for key in self.camera_keys], dim=-1)
