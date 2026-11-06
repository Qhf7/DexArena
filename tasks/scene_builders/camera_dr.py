"""Camera-pose jitter helper.

Use this from your env's `_default_sensor_configs` (or
`_default_human_render_camera_configs`) to spawn batched, per-sub-env
camera poses:

    from tasks.scene_builders import jitter_pose

    @property
    def _default_sensor_configs(self):
        rng = np.random.default_rng(self._cam_seed)
        pose = jitter_pose(
            base_eye=[0.5, 0.0, 0.4],
            base_target=[0.0, 0.0, 0.1],
            eye_range=0.05,
            target_range=0.02,
            num_envs=self.num_envs,
            rng=rng,
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi/2, 0.01, 100)]

The function is pure — no SAPIEN scene access — so it is testable without
a built env. Internally it composes per-env look_at poses, returning a
batched ManiSkill `Pose` struct.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import Pose


def jitter_pose(
    base_eye: Sequence[float],
    base_target: Sequence[float],
    eye_range: float,
    target_range: float,
    num_envs: int,
    rng: np.random.Generator,
) -> Pose:
    """Per-env look_at pose with eye and target uniformly jittered.

    Returns a batched `mani_skill.utils.structs.pose.Pose` with shape (num_envs,).
    """
    base_eye = np.asarray(base_eye, dtype=np.float32)
    base_target = np.asarray(base_target, dtype=np.float32)
    eyes = base_eye + rng.uniform(-eye_range, eye_range, size=(num_envs, 3)).astype(np.float32)
    targets = base_target + rng.uniform(-target_range, target_range, size=(num_envs, 3)).astype(np.float32)

    qs = np.zeros((num_envs, 4), dtype=np.float32)
    ps = np.zeros((num_envs, 3), dtype=np.float32)
    for i in range(num_envs):
        look = sapien_utils.look_at(eye=eyes[i].tolist(), target=targets[i].tolist())
        ps[i] = look.p
        qs[i] = look.q
    return Pose.create_from_pq(p=torch.from_numpy(ps), q=torch.from_numpy(qs))
