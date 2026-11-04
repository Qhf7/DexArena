"""Spatial softmax pooling: a (B, C, H, W) feature map -> C expected keypoint
coordinates in normalized image space."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSoftmax(nn.Module):
    """(B, C, H, W) -> (B, C, 2) keypoints in [-1, 1], flattened by the caller."""

    def __init__(
        self,
        height: int,
        width: int,
        *,
        temperature: float = 1.0,
        learnable_temperature: bool = True,
    ):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.height),
            torch.linspace(-1.0, 1.0, self.width),
            indexing="ij",
        )
        self.register_buffer("pos_x", pos_x.reshape(1, 1, -1))
        self.register_buffer("pos_y", pos_y.reshape(1, 1, -1))
        log_temperature = torch.tensor(float(temperature)).log()
        if learnable_temperature:
            self.log_temperature = nn.Parameter(log_temperature)
        else:
            self.register_buffer("log_temperature", log_temperature)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        if (height, width) != (self.height, self.width):
            raise ValueError(
                f"SpatialSoftmax was built for {self.height}x{self.width} maps, "
                f"got {height}x{width}."
            )
        flat = features.reshape(batch, channels, height * width)
        attention = F.softmax(flat / self.log_temperature.exp().clamp_min(1e-4), dim=-1)
        keypoint_x = (attention * self.pos_x).sum(dim=-1)
        keypoint_y = (attention * self.pos_y).sum(dim=-1)
        return torch.stack([keypoint_x, keypoint_y], dim=-1)  # (B, C, 2)
