"""ResNet-10 + spatial softmax encoder.

    ImageNet mean/std normalization
    ResNet-10 backbone                   -> 4x4x512 for a 128x128 input
    SpatialSoftmax                       -> (512, 2) keypoints -> flatten
    Dense                                -> latent_dim_per_cam

The released SERL / HIL-SERL weights use GroupNorm with 4 groups and JAX
``SAME`` padding (extra pixel bottom/right) in the max pool and block
convolutions; both are reproduced here. The backbone is shared by all cameras,
the spatial softmax and projection are per camera.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from algo.encoders.base import EncoderConfig, VisualEncoder, to_float_chw
from algo.encoders.spatial_softmax import SpatialSoftmax


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _same_padding(size: int, kernel: int, stride: int) -> tuple[int, int]:
    """JAX ``padding="SAME"`` split: the extra pixel goes on the bottom/right."""
    total = max(0, (math.ceil(size / stride) - 1) * stride + kernel - size)
    before = total // 2
    return before, total - before


def _pad_same(x: torch.Tensor, kernel: int, stride: int, value: float = 0.0) -> torch.Tensor:
    top, bottom = _same_padding(int(x.shape[-2]), kernel, stride)
    left, right = _same_padding(int(x.shape[-1]), kernel, stride)
    if (top, bottom, left, right) == (0, 0, 0, 0):
        return x
    return F.pad(x, (left, right, top, bottom), value=value)


class Conv2dSame(nn.Conv2d):
    """Conv with JAX ``SAME`` padding (extra pixel on the bottom/right)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias
        )
        self.kernel = int(kernel_size)
        self.stride_size = int(stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(_pad_same(x, self.kernel, self.stride_size))


class MaxPool2dSame(nn.Module):
    """Max pool with JAX ``SAME`` padding, padding with ``-inf`` as JAX does."""

    def __init__(self, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        self.kernel = int(kernel_size)
        self.stride_size = int(stride)
        self.pool = nn.MaxPool2d(kernel_size=self.kernel, stride=self.stride_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(_pad_same(x, self.kernel, self.stride_size, value=-float("inf")))


def _make_norm(norm_layer: str, channels: int, groups: int = 4) -> nn.Module:
    if norm_layer == "batch":
        return nn.BatchNorm2d(channels)
    if norm_layer == "group":
        # `groups` must match the checkpoint; a mismatch loads without error.
        return nn.GroupNorm(num_groups=min(groups, channels), num_channels=channels)
    if norm_layer == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm_layer {norm_layer!r}.")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_channels: int, out_channels: int, stride: int, norm_layer: str, groups: int = 4
    ):
        super().__init__()
        self.conv1 = Conv2dSame(in_channels, out_channels, 3, stride=stride, bias=False)
        self.bn1 = _make_norm(norm_layer, out_channels, groups)
        self.conv2 = Conv2dSame(out_channels, out_channels, 3, stride=1, bias=False)
        self.bn2 = _make_norm(norm_layer, out_channels, groups)
        self.relu = nn.ReLU(inplace=True)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                Conv2dSame(in_channels, out_channels, 1, stride=stride, bias=False),
                _make_norm(norm_layer, out_channels, groups),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet10(nn.Module):
    """torchvision-style ResNet with one BasicBlock per stage (10 conv layers)."""

    def __init__(self, norm_layer: str = "batch", groups: int = 4):
        super().__init__()
        # The stem pads symmetrically like the reference; only the max pool and
        # the block convolutions use SAME padding.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = _make_norm(norm_layer, 64, groups)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = MaxPool2dSame(3, stride=2)
        self.layer1 = nn.Sequential(BasicBlock(64, 64, 1, norm_layer, groups))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2, norm_layer, groups))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2, norm_layer, groups))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2, norm_layer, groups))
        self.out_channels = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


def load_pretrained_backbone(
    backbone: ResNet10, path: str, *, norm_layer: str = "batch", norm_groups: int = 4
) -> dict[str, int]:
    """Load ResNet-10 backbone weights (this module's parameter names) from a local checkpoint.

    Accepts a plain ``state_dict`` or a checkpoint dict wrapping one and strips
    the usual prefixes. Raises if too few tensors match or if the checkpoint's
    recorded norm configuration disagrees with the encoder's.
    """
    checkpoint = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        expected_layer = checkpoint.get("norm_layer")
        expected_groups = checkpoint.get("norm_groups")
        if expected_layer is not None and expected_layer != norm_layer:
            raise RuntimeError(
                f"{path} was saved for norm_layer={expected_layer!r} but the encoder "
                f"was built with {norm_layer!r}. Pass --encoder_norm_layer "
                f"{expected_layer}; otherwise the normalization layers stay randomly "
                "initialized while the convolutions load."
            )
        if (
            expected_groups is not None
            and norm_layer == "group"
            and int(expected_groups) != int(norm_groups)
        ):
            raise RuntimeError(
                f"{path} was saved for norm_groups={expected_groups} but the encoder "
                f"uses {norm_groups}. The tensors would load and then normalize over a "
                "different partition of the channels, with no error."
            )
        for key in ("state_dict", "model_state_dict", "model", "params"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path} does not contain a state dict (got {type(checkpoint).__name__}).")

    target = backbone.state_dict()
    cleaned: dict[str, torch.Tensor] = {}
    for name, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        for prefix in ("module.", "model.", "backbone.", "resnet.", "encoder."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        if name in target and target[name].shape == value.shape:
            cleaned[name] = value
    result = backbone.load_state_dict(cleaned, strict=False)
    loaded = len(cleaned)
    if loaded < len(target) // 2:
        raise RuntimeError(
            f"Only {loaded}/{len(target)} ResNet-10 tensors matched the checkpoint at "
            f"{path}. Missing example keys: {list(result.missing_keys)[:5]}. Refusing to "
            "train on a mostly-random 'pre-trained' backbone."
        )
    print(
        f"[resnet10] loaded {loaded}/{len(target)} tensors from {path} "
        f"(missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)})"
    )
    return {"loaded": loaded, "total": len(target)}


class ResNet10Encoder(VisualEncoder):
    def __init__(
        self,
        camera_keys: Sequence[str],
        image_shape: tuple[int, int, int],
        config: EncoderConfig,
    ):
        super().__init__(camera_keys)
        self.freeze_backbone = bool(config.freeze_backbone)
        self.update_bn_stats = bool(config.update_bn_stats)

        self.backbone = ResNet10(norm_layer=config.norm_layer, groups=config.norm_groups)
        if config.pretrained_path is not None:
            load_pretrained_backbone(
                self.backbone,
                config.pretrained_path,
                norm_layer=config.norm_layer,
                norm_groups=config.norm_groups,
            )
        else:
            print("[resnet10] no pretrained_path given -> randomly initialized backbone")

        height, width, channels = image_shape
        # Probe in eval(): a BatchNorm backbone in train() would fold the zeros
        # into its running statistics.
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, int(channels), int(height), int(width))
            feature_shape = tuple(self.backbone(dummy).shape[1:])
        self.backbone.train(was_training)
        self.feature_shape = feature_shape  # printed at build time: expect (512, 4, 4)
        feature_channels, feature_h, feature_w = feature_shape
        if feature_h * feature_w == 1:
            # A 1x1 feature map makes SpatialSoftmax constant and the backbone
            # gradient exactly zero; the backbone downsamples by 32.
            raise ValueError(
                f"ResNet-10 on a {height}x{width} input gives a {feature_h}x{feature_w} "
                "feature map, which makes SpatialSoftmax constant. Use image_size >= 64."
            )

        self.pool = nn.ModuleDict(
            {key: SpatialSoftmax(feature_h, feature_w) for key in self.camera_keys}
        )
        self.project = nn.ModuleDict()
        self.norm = nn.ModuleDict()
        for key in self.camera_keys:
            linear = nn.Linear(feature_channels * 2, int(config.latent_dim_per_cam))
            nn.init.orthogonal_(linear.weight, gain=1.0)
            nn.init.zeros_(linear.bias)
            self.project[key] = linear
            self.norm[key] = (
                nn.LayerNorm(int(config.latent_dim_per_cam))
                if config.latent_layer_norm
                else nn.Identity()
            )
        self.latent_layer_norm = bool(config.latent_layer_norm)
        self.latent_dim = len(self.camera_keys) * int(config.latent_dim_per_cam)

        self.register_buffer(
            "image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )
        self._last_keypoints: dict[str, torch.Tensor] = {}

        # `freeze_backbone` is the configured setting; `_backbone_trainable` is
        # the runtime state toggled by the freeze -> unfreeze warmup.
        self._backbone_trainable = not self.freeze_backbone
        self.backbone.requires_grad_(self._backbone_trainable)

    # ---- fine-tuning -----------------------------------------------------------

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Pre-trained backbone vs randomly initialized head (spatial softmax,
        projection, LayerNorm)."""
        backbone = list(self.backbone.parameters())
        backbone_ids = {id(p) for p in backbone}
        head = [p for p in self.parameters() if id(p) not in backbone_ids]
        return {"backbone": backbone, "head": head}

    def set_backbone_trainable(self, trainable: bool) -> None:
        if self.freeze_backbone:
            # A configured-frozen backbone stays frozen.
            return
        self._backbone_trainable = bool(trainable)
        self.backbone.requires_grad_(self._backbone_trainable)
        # Re-apply the BN-statistics policy for the new state.
        self.train(self.training)

    @property
    def backbone_trainable(self) -> bool:
        return bool(self._backbone_trainable)

    def train(self, mode: bool = True) -> "ResNet10Encoder":
        super().train(mode)
        # `requires_grad_(False)` does not freeze BatchNorm statistics; only
        # `.eval()` does.
        if mode and not self.update_bn_stats:
            self.backbone.eval()
        return self

    def _encode_camera(self, key: str, images: torch.Tensor) -> torch.Tensor:
        x = (to_float_chw(images) - self.image_mean) / self.image_std
        if not self._backbone_trainable:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        keypoints = self.pool[key](features)  # (B, C, 2)
        self._last_keypoints[key] = keypoints.detach()
        z = self.norm[key](self.project[key](keypoints.flatten(1)))
        return torch.tanh(z) if self.latent_layer_norm else z

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [self._encode_camera(key, obs[key]) for key in self.camera_keys], dim=-1
        )

    def diagnostics(self) -> dict[str, float]:
        """Keypoint spread per camera; collapse to a constant means the encoder
        stopped tracking."""
        metrics: dict[str, float] = {}
        for key, keypoints in self._last_keypoints.items():
            if keypoints.numel() == 0:
                continue
            metrics[f"keypoint_batch_var/{key}"] = float(
                keypoints.float().var(dim=0).mean().item()
            )
            metrics[f"keypoint_spatial_std/{key}"] = float(
                keypoints.float().std(dim=1).mean().item()
            )
        return metrics
