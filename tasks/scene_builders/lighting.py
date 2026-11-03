"""Per-sub-env lighting randomization for the scene library.

The same axis exists inline in ``randomized_table.py``. Deliberately duplicated
rather than shared: that file sits in the code-fingerprint closure of seven
collection targets (TableClean, the Office family), so importing from here would
have re-versioned their datasets mid-campaign for a refactor. If the procedural
table builder is ever retired, fold it into this one.

Sampling is driven by ``env._episode_rng`` so a seeded reset reproduces the same
lights.

Colour is drawn as a TEMPERATURE, not as three independent channel gains. The
first version scaled R, G and B independently and produced green and magenta
rooms -- hues no light source emits, so a policy would have been asked to
generalise across a nuisance that does not occur. Real indoor light runs from
warm tungsten to cool daylight along the black-body curve, which is one
parameter.

The randomizer owns the directional lights it creates: ``apply()`` removes the
light it added on the previous reset for that sub-env before adding a new one,
because SAPIEN's ``add_directional_light`` appends rather than replaces.
"""
from __future__ import annotations

import numpy as np


def blackbody_rgb(kelvin: float) -> np.ndarray:
    """Approximate linear RGB of a black body at ``kelvin``, normalised to peak 1.

    Tanner Helland's piecewise fit: accurate enough over 1000-40000 K for
    lighting, and needs no colour-science dependency.
    """
    t = np.clip(kelvin, 1000.0, 40000.0) / 100.0
    if t <= 66.0:
        r = 255.0
        g = 99.4708025861 * np.log(t) - 161.1195681661
    else:
        r = 329.698727446 * (t - 60.0) ** -0.1332047592
        g = 288.1221695283 * (t - 60.0) ** -0.0755148492
    if t >= 66.0:
        b = 255.0
    elif t <= 19.0:
        b = 0.0
    else:
        b = 138.5177312231 * np.log(t - 10.0) - 305.0447927307
    rgb = np.clip(np.array([r, g, b], dtype=np.float64), 0.0, 255.0) / 255.0
    return rgb / max(rgb.max(), 1e-6)


class LightingRandomizer:
    """Randomizes ambient color and one directional light per sub-env."""

    def __init__(
        self,
        env,
        ambient_range: tuple[float, float] = (0.1, 0.6),
        temperature_range: tuple[float, float] = (2700.0, 6500.0),
        intensity_range: tuple[float, float] = (0.6, 1.4),
        light_down_bias: float = 0.3,
    ):
        self.env = env
        self.ambient_range = ambient_range
        # 2700 K is a domestic tungsten bulb, 6500 K overcast daylight: the ends
        # of what a room is actually lit by.
        self.temperature_range = temperature_range
        self.intensity_range = intensity_range
        self.light_down_bias = light_down_bias
        self.sampled_ambients: np.ndarray | None = None
        # sub-env index -> directional light component from the previous reset
        self._dir_lights: dict = {}

    def apply(self, env_idx):
        idx_list = env_idx.tolist() if hasattr(env_idx, "tolist") else list(env_idx)
        rng = self.env._episode_rng
        if self.sampled_ambients is None:
            self.sampled_ambients = np.zeros((self.env.num_envs, 3), dtype=np.float32)
        for i in idx_list:
            kelvin = rng.uniform(*self.temperature_range)
            tint = blackbody_rgb(kelvin)
            # One temperature drives ambient and key light together: a room lit by
            # a warm bulb has warm fill, and mismatching them reads as two rooms
            # composited.
            amb = (tint * rng.uniform(*self.ambient_range)).astype(np.float32)
            self.sampled_ambients[i] = amb
            sub = self.env.scene.sub_scenes[i]
            sub.set_ambient_light(amb.tolist())
            self._remove_previous_light(sub, i)
            direction = rng.uniform(-1.0, 1.0, size=3)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            # Force the light to point somewhat downward
            direction[2] = -abs(direction[2]) - self.light_down_bias
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            key = (tint * rng.uniform(*self.intensity_range)).astype(np.float32)
            self._dir_lights[i] = sub.add_directional_light(
                direction=direction.tolist(),
                color=key.tolist(),
                shadow=False,
            )

    def _remove_previous_light(self, sub, i):
        old = self._dir_lights.pop(i, None)
        if old is None:
            return
        try:
            sub.remove_light(old)
        except Exception:
            try:
                sub.remove_entity(old.entity)
            except Exception:
                pass
