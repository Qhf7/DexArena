"""Agent variants that carry the calibrated Marvin wrist camera (runtime sensor contract).

These two agents exist for the wrist-camera runtime contract (tasks/pick_single_ycb_to_goal_table.py
and tests/test_marvin_wrist_camera_runtime.py). No collection target uses them: the recording
agents keep their own hard-coded wrist camera. They live in their own module so that the camera
configuration (utils/camera_configs.py) is imported by nothing in a collection target's
code-fingerprint closure -- editing a camera never re-versions a dataset.
"""
from __future__ import annotations

import numpy as np

from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

from tasks.robots.panda_sharpa_hand_right import PandaSharpaHandRight, PandaSharpaHandRightAngled
from tasks.robots.panda_wuji_hand_right import PandaWujiHandRight
from utils.camera_configs import SHARPA_WRIST_CAMERA_V1, WUJI_WRIST_CAMERA_V1


@register_agent()
class PandaSharpaHandRightAngledWristCamV1(PandaSharpaHandRightAngled):
    """Angled Sharpa with the immutable Marvin wrist-camera v1 profile."""

    uid = SHARPA_WRIST_CAMERA_V1.robot_uid
    _urdf_path = PandaSharpaHandRight._urdf_path.with_name(
        SHARPA_WRIST_CAMERA_V1.urdf_filename
    )
    urdf_path = str(_urdf_path.resolve())

    wrist_camera_pose = sapien_utils.look_at(
        eye=list(SHARPA_WRIST_CAMERA_V1.eye_in_parent_m),
        target=list(SHARPA_WRIST_CAMERA_V1.target_in_parent_m),
        up=list(SHARPA_WRIST_CAMERA_V1.image_up_in_parent),
    )

    def _after_init(self):
        super()._after_init()
        self.wrist_camera_mount_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(),
            SHARPA_WRIST_CAMERA_V1.mount_link_name,
        )

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                "wrist_camera",
                self.wrist_camera_pose,
                SHARPA_WRIST_CAMERA_V1.width,
                SHARPA_WRIST_CAMERA_V1.height,
                np.deg2rad(SHARPA_WRIST_CAMERA_V1.vertical_fov_deg),
                SHARPA_WRIST_CAMERA_V1.near_m,
                SHARPA_WRIST_CAMERA_V1.far_m,
                mount=self.wrist_camera_mount_link,
            )
        ]


@register_agent()
class PandaWujiHandRightWristCamV1(PandaWujiHandRight):
    """Wuji with the immutable Marvin wrist-camera v1 profile."""

    uid = WUJI_WRIST_CAMERA_V1.robot_uid
    _urdf_path = PandaWujiHandRight._urdf_path.with_name(
        WUJI_WRIST_CAMERA_V1.urdf_filename
    )
    urdf_path = str(_urdf_path.resolve())

    wrist_camera_pose = sapien_utils.look_at(
        eye=list(WUJI_WRIST_CAMERA_V1.eye_in_parent_m),
        target=list(WUJI_WRIST_CAMERA_V1.target_in_parent_m),
        up=list(WUJI_WRIST_CAMERA_V1.image_up_in_parent),
    )

    def _after_init(self):
        super()._after_init()
        self.wrist_camera_mount_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(),
            WUJI_WRIST_CAMERA_V1.mount_link_name,
        )

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                "wrist_camera",
                self.wrist_camera_pose,
                WUJI_WRIST_CAMERA_V1.width,
                WUJI_WRIST_CAMERA_V1.height,
                np.deg2rad(WUJI_WRIST_CAMERA_V1.vertical_fov_deg),
                WUJI_WRIST_CAMERA_V1.near_m,
                WUJI_WRIST_CAMERA_V1.far_m,
                mount=self.wrist_camera_mount_link,
            )
        ]
