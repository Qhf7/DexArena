"""Procedural texture pool. PIL-generated PNGs cached on disk."""
from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image

_IMG_SIZE = 256


def _hsv_color(rng: np.random.Generator) -> tuple[int, int, int]:
    h = rng.uniform(0.0, 1.0)
    s = rng.uniform(0.3, 0.8)
    v = rng.uniform(0.4, 0.9)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def _solid(rng: np.random.Generator) -> Image.Image:
    return Image.new("RGB", (_IMG_SIZE, _IMG_SIZE), _hsv_color(rng))


def _gradient(rng: np.random.Generator) -> Image.Image:
    c1 = np.array(_hsv_color(rng), dtype=np.float32)
    c2 = np.array(_hsv_color(rng), dtype=np.float32)
    direction = rng.choice(["h", "v", "d"])
    if direction == "h":
        t = np.linspace(0.0, 1.0, _IMG_SIZE)[None, :, None]
    elif direction == "v":
        t = np.linspace(0.0, 1.0, _IMG_SIZE)[:, None, None]
    else:
        lin = np.linspace(0.0, 1.0, _IMG_SIZE)
        t = ((lin[:, None] + lin[None, :]) / 2.0)[..., None]
    arr = c1 * (1.0 - t) + c2 * t
    return Image.fromarray(arr.astype(np.uint8))


def _ensure_texture_pool(texture_dir: Path, n: int, seed: int) -> list[Path]:
    texture_dir = Path(texture_dir)
    existing = sorted(texture_dir.glob("*.png"))
    if len(existing) >= n:
        return existing[:n]
    texture_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    half = n // 2
    for i in range(n):
        img = _solid(rng) if i < half else _gradient(rng)
        img.save(texture_dir / f"{i:03d}.png")
    return sorted(texture_dir.glob("*.png"))[:n]
