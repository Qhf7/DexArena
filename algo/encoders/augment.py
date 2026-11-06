"""Random-shift image augmentation.

Replicate-pad by ``pad`` pixels, then crop
back to the original size at a random integer offset. 
"""

from __future__ import annotations

from typing import Mapping

import torch


def random_shift(images: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Random shift of up to ``pad`` pixels. ``images``: (B, H, W, C), any dtype.
    """
    if pad == 0:
        return images
    if images.ndim != 4:
        raise ValueError(f"Expected (B, H, W, C) images, got shape {tuple(images.shape)}.")
    batch, height, width, _ = images.shape
    device = images.device
    dy = torch.randint(-pad, pad + 1, (batch,), device=device)
    dx = torch.randint(-pad, pad + 1, (batch,), device=device)
    rows = (torch.arange(height, device=device).unsqueeze(0) + dy.unsqueeze(1)).clamp_(
        0, height - 1
    )  # (B, H) — clamping is the replicate padding
    cols = (torch.arange(width, device=device).unsqueeze(0) + dx.unsqueeze(1)).clamp_(
        0, width - 1
    )  # (B, W)
    batch_index = torch.arange(batch, device=device).view(batch, 1, 1)
    return images[batch_index, rows.unsqueeze(2), cols.unsqueeze(1)]


def augment_obs(
    obs: Mapping[str, torch.Tensor], camera_keys: tuple[str, ...], pad: int = 4
) -> dict[str, torch.Tensor]:
    """Apply an *independent* random shift to each camera; pass proprio through.
    """
    augmented = dict(obs)
    for key in camera_keys:
        augmented[key] = random_shift(obs[key], pad)
    return augmented
