"""Per-episode layout randomization, shared by the task families.

Every draw comes from the environment's EPISODE RNG. ManiSkill seeds it from
``env.reset(seed=...)`` (and re-seeds it after a reconfigure), so the
``episode_seed`` a recording stores reproduces the layout bit for bit, on CPU
and GPU alike. Nothing here touches ``torch.rand`` or a module-level RNG.

Conventions
-----------
* Offsets are drawn UNIFORMLY in ``[-half, +half]`` around the family's nominal
  layout, so the nominal layout is the centre of the distribution and a policy
  trained on randomized episodes still sees it.
* Yaw offsets rotate about the object's own vertical axis; a positional offset
  is applied in the WORLD frame (the tabletop is horizontal everywhere).
* Arm joint noise is Gaussian with the std below, on the arm joints only.

Blast radius: this file is imported by the families that randomize through it
(PackageOpening, Pipette, TableClean, Juice, and the Office / Furniture long
tasks). Editing it changes their code fingerprints and nobody else's.
"""
from __future__ import annotations

import numpy as np
import torch

#: std [rad] of the Gaussian added to every arm joint at reset. The families that
#: already randomized (CabinetPour, DrawerPlug, Screwing) show a 0.09 rad spread
#: across seeds, which is what this value produces.
ARM_INIT_QPOS_NOISE_STD = 0.02


def episode_rng(env, i: int = 0) -> np.random.RandomState:
    """The numpy RNG of sub-environment ``i`` for the current episode."""
    return env._batched_episode_rng[i]


LAYOUT_STREAM_SALT = 0x5EED_1A70


def layout_rng(env, i: int = 0, salt: int = LAYOUT_STREAM_SALT) -> np.random.RandomState:
    """A fresh RNG for the LAYOUT of sub-environment ``i`` this episode.

    The stream is keyed on ONE integer drawn from ManiSkill's batched episode RNG
    (plus ``salt``). That RNG is re-seeded by ``reset(seed=...)`` -- so a seeded
    reset reproduces the layout -- and simply advances on ``reset()`` without a
    seed, which is how the teleop loop resets between episodes, so every episode
    gets a new layout. (A first version keyed the stream on the episode SEED
    value; under unseeded resets of an environment with reconfiguration_freq=0
    that value never changes, and the pipette session recorded ten
    episodes with the identical layout.)

    Every environment of a family must make this its FIRST draw from the episode
    RNG after a reset (before arm-joint noise etc.), so that the long task and
    its subtasks build the same layout for the same seed."""
    key = int(env._batched_episode_rng[i].randint(2 ** 31))
    return np.random.RandomState((key * 1_000_003 + salt) % (2 ** 31 - 1))


def uniform_offset(rng, half_xy, yaw_half_deg: float = 0.0) -> tuple[float, float, float]:
    """``(dx, dy, yaw)``: uniform in ``[-half, +half]`` per axis; yaw in radians."""
    hx, hy = float(half_xy[0]), float(half_xy[1])
    dx = float(rng.uniform(-hx, hx)) if hx > 0 else 0.0
    dy = float(rng.uniform(-hy, hy)) if hy > 0 else 0.0
    yaw = float(np.deg2rad(rng.uniform(-yaw_half_deg, yaw_half_deg))) if yaw_half_deg > 0 else 0.0
    return dx, dy, yaw


def yaw_quat_wxyz(yaw: float) -> np.ndarray:
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float64)


def quat_mul_wxyz(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def rotate_xy(dx: float, dy: float, yaw: float) -> tuple[float, float]:
    c, s = np.cos(yaw), np.sin(yaw)
    return float(c * dx - s * dy), float(s * dx + c * dy)


def offset_pose(base_p, base_q_wxyz, dx: float, dy: float, yaw: float):
    """Nominal pose moved by ``(dx, dy)`` and turned by ``yaw`` about its own vertical axis."""
    p = np.asarray(base_p, dtype=np.float64).copy()
    p[0] += dx
    p[1] += dy
    q = quat_mul_wxyz(yaw_quat_wxyz(yaw), np.asarray(base_q_wxyz, dtype=np.float64))
    return p, q


def attached_point(anchor_p, anchor_yaw: float, local_xy_offset) -> np.ndarray:
    """World position of a point that rides on a randomized anchor.

    ``local_xy_offset`` is the point's nominal offset from the anchor in the
    world frame (anchor at its nominal pose); the anchor has since moved to
    ``anchor_p`` and turned by ``anchor_yaw``."""
    ox, oy = rotate_xy(float(local_xy_offset[0]), float(local_xy_offset[1]), anchor_yaw)
    p = np.asarray(anchor_p, dtype=np.float64).copy()
    p[0] += ox
    p[1] += oy
    return p


def arm_qpos_noise(rng, agent, batch_size: int, std: float = ARM_INIT_QPOS_NOISE_STD, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """``(indices, noise)`` for the agent's arm joints (single or dual arm)."""
    active = [j.name for j in agent.robot.get_active_joints()]
    indices = [active.index(n) for n in getattr(agent, "arm_joint_names", []) if n in active]
    index = torch.tensor(indices, dtype=torch.long, device=device)
    if not indices or std <= 0:
        return index, torch.zeros((batch_size, len(indices)), dtype=torch.float32, device=device)
    noise = torch.as_tensor(rng.normal(0.0, std, size=(batch_size, len(indices))), dtype=torch.float32, device=device)
    return index, noise


def scatter_xy(rng, n: int, center_xy, half_span: float, min_separation: float, max_tries: int = 200) -> list[tuple[float, float]]:
    """``n`` points uniform in a square of ``half_span`` around ``center_xy``, at
    least ``min_separation`` apart (rejection sampling; deterministic in ``rng``)."""
    points: list[tuple[float, float]] = []
    cx, cy = float(center_xy[0]), float(center_xy[1])
    for _ in range(max_tries):
        x = cx + float(rng.uniform(-half_span, half_span))
        y = cy + float(rng.uniform(-half_span, half_span))
        if all((x - px) ** 2 + (y - py) ** 2 >= min_separation ** 2 for px, py in points):
            points.append((x, y))
            if len(points) == n:
                return points
    raise RuntimeError(f"could not scatter {n} points {min_separation} m apart in a {2*half_span} m square")
