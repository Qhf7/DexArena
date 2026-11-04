"""TableClean stages under num_envs > 1.

The episode-layout helpers emit one pose each, so the task refuses more than
one env. With several envs the reset draws one layout per env from the same
per-env streams and sizes every write to the reset sub-batch; one env runs the
task's own reset. One limit: the bin is static and placed at scene build, so
every env shares one bin offset for the life of the env.
"""

from __future__ import annotations

import numpy as np
import torch

from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.structs.pose import Pose

from ._base import MultiEnvInitMixin
from ._shims import patched


def _all_scenes(builder, scene_idxs=None):
    return builder


class TableCleanMixin(MultiEnvInitMixin):
    def _load_scene(self, options):
        if self.num_envs == 1:
            super()._load_scene(options)
        else:
            import tasks.table_cleaning.table_cleaning_long as module

            build_debris = module.build_debris

            def build_in_every_scene(*args, **kwargs):
                with patched(ActorBuilder, "set_scene_idxs", _all_scenes):
                    return build_debris(*args, **kwargs)

            with patched(module, "build_debris", build_in_every_scene):
                super()._load_scene(options)
        # The orientation each free body was built with: with several envs
        # nothing is rebuilt between episodes, so `pose.q` drifts. Read from the
        # SAPIEN entity (GPU pose buffers do not exist yet).
        self._built_q = {
            id(a): np.asarray(a._objs[0].pose.q, dtype=np.float64) for a in (self.pan, *self.debris)
        }

    def _reset_robot_rows(self, env_ids: list[int]):
        from tasks.table_cleaning.table_cleaning_long import ARM_INIT_QPOS, ROBOT_ROOT, arm_qpos_noise, episode_rng

        b = len(env_ids)
        kf = self.agent.keyframes["rest"].qpos
        qpos = torch.tensor(kf, dtype=torch.float32, device=self.device).view(1, -1).expand(b, -1).clone()
        qpos[:, :7] = torch.tensor(ARM_INIT_QPOS, dtype=torch.float32, device=self.device)
        qpos[:, 7:] = 0.0
        for k, env_i in enumerate(env_ids):
            index, noise = arm_qpos_noise(
                episode_rng(self, env_i), self.agent, 1, self.robot_init_qpos_noise, device=self.device
            )
            qpos[k, index] += noise[0]
        self.agent.reset(qpos)
        self.agent.robot.set_root_pose(ROBOT_ROOT)

    def _write_pose(self, actor, p, q):
        b = len(p)
        t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=self.device)  # noqa: E731
        actor.set_pose(Pose.create_from_pq(t(p), t(q)))
        zero = torch.zeros(b, 3, dtype=torch.float32, device=self.device)
        actor.set_linear_velocity(zero)
        actor.set_angular_velocity(zero)

    def _place_scene_start_batched(self, env_idx, load_pan: bool):
        """``_place_scene_start`` (+ ``_place_in_pan``) with one layout per env."""
        from tasks.table_cleaning.table_cleaning_long import (
            DEBRIS_CENTER,
            DEBRIS_MIN_SEPARATION,
            DEBRIS_SCATTER_HALF_SPAN,
            DEBRIS_SIZE,
            DUSTPAN_POS,
            LAYOUT_STREAM_SALT,
            PAN_RANDOM_XY,
            PAN_RANDOM_YAW_DEG,
            PAN_REST_Z,
            PICK_DEBRIS_SCALE,
            layout_rng,
            quat_mul_wxyz,
            scatter_xy,
            uniform_offset,
            yaw_quat_wxyz,
        )

        ids = [int(i) for i in torch.as_tensor(env_idx).reshape(-1).tolist()]
        b = len(ids)
        rest_z = PICK_DEBRIS_SCALE * DEBRIS_SIZE / 2
        pan_p = np.zeros((b, 3))
        pan_q = np.zeros((b, 4))
        deb_p = np.zeros((len(self.debris), b, 3))
        deb_q = np.zeros((len(self.debris), b, 4))
        rngs = [layout_rng(self, i) for i in ids]  # first episode-RNG draw, per env
        self._reset_robot_rows(ids)
        for k, rng in enumerate(rngs):
            dx, dy, yaw = uniform_offset(rng, PAN_RANDOM_XY, PAN_RANDOM_YAW_DEG)
            pan_p[k] = (DUSTPAN_POS[0] + dx, DUSTPAN_POS[1] + dy, PAN_REST_Z)
            pan_q[k] = quat_mul_wxyz(yaw_quat_wxyz(yaw), self._built_q[id(self.pan)])
            spots = scatter_xy(rng, self.debris_count, DEBRIS_CENTER, DEBRIS_SCATTER_HALF_SPAN, DEBRIS_MIN_SEPARATION)
            for i, (d, (x, y)) in enumerate(zip(self.debris, spots)):
                deb_p[i, k] = (x, y, rest_z + 0.004)
                deb_q[i, k] = quat_mul_wxyz(
                    yaw_quat_wxyz(np.deg2rad(float(rng.uniform(0.0, 360.0)))), self._built_q[id(d)]
                )
        if load_pan:
            f = self._pan_floor
            xs = np.linspace(f["x"][0], f["x"][1], len(self.debris))
            for k, env_i in enumerate(ids):
                rng = layout_rng(self, env_i, salt=LAYOUT_STREAM_SALT + 1)
                w, x, y, z = pan_q[k] / np.linalg.norm(pan_q[k])
                R = np.array(
                    [
                        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                    ]
                )
                for i in range(len(self.debris)):
                    local = np.array(
                        [
                            xs[i],
                            float(rng.uniform(f["y"][0], f["y"][1])) * 0.75,
                            f["z"] + PICK_DEBRIS_SCALE * DEBRIS_SIZE / 2,
                        ]
                    )
                    deb_p[i, k] = pan_p[k] + R @ local
        self._write_pose(self.pan, pan_p, pan_q)
        for i, d in enumerate(self.debris):
            self._write_pose(d, deb_p[i], deb_q[i])
        self._reset_buffers(env_idx)

    def _place_scene_start(self, env_idx, load_pan: bool = False):
        if self.num_envs == 1:
            return super()._place_scene_start(env_idx)
        from tasks.table_cleaning.table_cleaning_long import set_work_table_pose

        self.table_scene.initialize(env_idx)
        set_work_table_pose(self)
        self._place_scene_start_batched(env_idx, load_pan=load_pan)


class DumpDustpanMixin(TableCleanMixin):
    def _initialize_episode(self, env_idx, options):
        if self.num_envs == 1:
            return super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            # The pieces are seated inside the batched reset (one layout per env).
            self._place_scene_start(env_idx, load_pan=self.start_loaded)
            if self.start_loaded:
                self.f_picked[env_idx] = True
