"""Visual encoders for image-based RL"""

from __future__ import annotations

from typing import Sequence

from algo.encoders.augment import augment_obs, random_shift
from algo.encoders.base import (
    ENCODER_ARCHS,
    GRAD_SOURCES,
    SHARING_MODES,
    EncoderConfig,
    VisualEncoder,
    to_float_chw,
)
from algo.encoders.d4pg_encoder import D4PGEncoder
from algo.encoders.resnet10_encoder import ResNet10Encoder
from algo.encoders.spatial_softmax import SpatialSoftmax


def build_encoder(
    config: EncoderConfig,
    camera_keys: Sequence[str],
    image_shape: tuple[int, int, int],
) -> VisualEncoder:
    """Instantiate the encoder named by ``config.arch``.

    ``image_shape`` is the per-camera (H, W, C) of one stored frame, i.e.
    ``PixelObsConfig.image_shape``.
    """
    config.validate()
    if config.arch == "d4pg":
        encoder = D4PGEncoder(camera_keys, image_shape, config)
    elif config.arch == "resnet10":
        encoder = ResNet10Encoder(camera_keys, image_shape, config)
    else:  # pragma: no cover - validate() already rejected it
        raise ValueError(f"Unknown encoder arch {config.arch!r}.")
    return encoder


__all__ = [
    "ENCODER_ARCHS",
    "GRAD_SOURCES",
    "SHARING_MODES",
    "D4PGEncoder",
    "EncoderConfig",
    "ResNet10Encoder",
    "SpatialSoftmax",
    "VisualEncoder",
    "augment_obs",
    "build_encoder",
    "random_shift",
    "to_float_chw",
]
