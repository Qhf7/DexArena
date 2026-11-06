"""Observation contract for image-based RL.

Every data source must produce exactly this dict::

    {"cam_0": uint8 (B, H, W, 3), "cam_1": uint8 (B, H, W, 3),
     "proprio": float32 (B, D)}

``proprio`` holds only quantities a real robot can read (qpos, qvel, TCP pose);
object state is privileged and never assembled into the observation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from utils.pixel_cameras import PixelObsConfig


# Keys the task environments expose that carry object state.
# `assert_no_privileged_keys` checks an assembled observation against this list;
# `report_env_obs_keys` uses it to show what was dropped from the raw env obs.
PRIVILEGED_ENV_OBS_KEYS: frozenset[str] = frozenset(
    {
        "object_pose",
        "obj_pose",
        "object_pos",
        "obj_tip_dist",
        "tcp_to_object_pos",
        "object_to_goal_pos",
        "obj_to_goal_pos",
        "obj_to_goal_dist",
        "object_vel",
        "obj_vel",
        "object_angular_vel",
        "contact_forces",
        "net_contact_forces",
        "is_grasped",
        "loose_grasp",
        # The task goal (sampled near the object).
        "goal_pos",
        # CabinetPour family.
        "door_qpos",
        "tilt_cos",
        "cap_pos",
        "cap_twist",
        "cap_detached",
        "pour_ratio",
        "num_spilled",
        "bottle_rep",
        "target_pos",
        # State-mode only.
        "door_pose",
        "cabinet_xy",
        "tcp_to_door_pos",
        "stages",
        "source_pose",
        "target_pose",
        "cap_pose",
        "tcp_to_source_pos",
        "tcp_to_cap_pos",
        "source_to_place_pos",
        "source_to_target_pos",
        # Office family.
        "mouse_root_pose",
        "left_button_link_pose",
        "button_top_pos",
        "button_released_top_pos",
        "button_press_axis",
        "button_up_axis",
        "button_long_axis",
        "button_short_axis",
        "button_half_extents",
        "tip_to_button",
        "tip_over_button",
        "tip_long",
        "tip_short",
        "height_above_released",
        "left_qpos",
        "left_qvel",
        "right_qpos",
        "right_qvel",
        "left_press_fraction",
        "right_press_fraction",
        "mouse_spawn_xy",
        "mouse_spawn_yaw",
        "laptop_root_pose",
        "screen_pose",
        "screen_edge_pos",
        "screen_extent_local",
        "tip_to_screen_edge",
        "hinge_qpos",
        "hinge_qvel",
        "open_fraction",
        "opening_angle_rad",
        "target_angle_rad",
        "maintain_angle_rad",
        "target_progress",
        "target_reached_once",
        "maintained",
        "laptop_spawn_xy",
        "laptop_spawn_yaw",
        "keyboard_root_pose",
        "space_key_link_pose",
        "key_top_pos",
        "key_released_top_pos",
        "press_tip_to_key",
        "press_achieved",
        "sequence_completed",
        "sequence_index",
        "sequence_progress",
        "task_success",
        # Shared across the office stages.
        "final_region_center",
        "final_region_dist",
        "final_progress",
        "inside_final_region",
        "hold_progress",
        "hold_completed",
        "release_progress",
        "released_after_hold",
        "task_phase_one_hot",
        # Robot forward kinematics (`task_tip_pos`, `all_tip_positions`, ...)
        # are not privileged and are not listed.
        # DrawerPlug family.
        "drawer_qpos",
        "charger_depth",
        "charger_place_distance",
        "charger_pos",
        "charger_place_pos",
        "socket_pos",
        "drawer_pose",
        "drawer_xy",
        "tcp_to_drawer_pos",
        "charger_pose",
        "socket_pose",
        "tcp_to_charger_pos",
        "charger_to_socket_pos",
        "charger_to_place_pos",
        "charger_lateral",
        "charger_slot_error",
        "charger_yaw_alignment",
        # Remaining sharpa_hand families, enumerated from each `_get_obs_extra`
        # (`left_tcp_pose`, `right_tcp_pose`, `gripper_opening`, `pressing_point`
        # are robot-intrinsic and not listed).
        # Screwing
        "bolt_place_distance",
        "bolt_place_pos",
        "bolt_pos",
        "cum_rotation",
        "nut_horizontal",
        "nut_pos",
        "nut_tilt_cos",
        "nut_z_gap",
        "stage_bolt_placed",
        "stage_nut_placed",
        "stage_turned",
        "threaded_depth",
        # PackageOpening: package_opening
        "apple_pick_completed",
        "apple_pose",
        "box_pose",
        "box_qpos",
        "plate_pose",
        "stage",
        "subtask_progress",
        "target_stage",
        # MealPreparation: meal_preparation
        "ball_done",
        "ball_grasped",
        "ball_left_box",
        "ball_on_plate",
        "ball_process_ok",
        "box_open",
        "bread_done",
        "bread_on_plate",
        "bread_process_ok",
        "ever_ball_left_box",
        "ever_ball_scooped",
        "ever_ball_transport",
        "ever_bread_grasped",
        "ever_bread_manipulated",
        "ever_bread_transport",
        "ever_left_hand_opened_box",
        "ever_right_hand_used_spoon",
        "initial_ball_in_box",
        "open_process_ok",
        "stage_ball_done",
        "stage_box_open",
        "stage_bread_done",
        "stage_id",
        # Furniture: furniture, furniture_panel_lay_flat, furniture_rail_insert_groove
        "axis_align",
        "contact_count",
        "extract_progress",
        "flatness",
        "footprint_inside",
        "grasp_point",
        "groove_axis",
        "height_above_rack",
        "in_place_region",
        "is_aligned",
        "is_flat_on_table",
        "is_lifted",
        "is_out_of_rack",
        "is_released",
        "is_seated",
        "is_static",
        "lateral_error",
        "left_panel_grasped_once",
        "left_tcp_to_panel",
        "left_tcp_to_rail",
        "panel_normal",
        "panel_pose",
        "place_center",
        "rail_axis",
        "rail_local_pos",
        "rail_pose",
        "rail_to_seated",
        "rail_up",
        "right_rail_grasped_once",
        "right_tcp_to_tool",
        "right_tool_engaged_once",
        "right_tool_grasped_once",
        "rotation_progress",
        "screw_angle",
        "screwdriver_pose",
        "seat_depth_error",
        "seated_pose",
        "stage_one_hot",
        "stop_error",
        "tcp_to_grasp",
        "tcp_to_rail_grasp",
        "tool_axis_alignment",
        "tool_tip_distance",
        "up_align",
        "yaw_alignment",
        # Pipette: pipette_transfer
        "cup_pose",
        "first_press",
        "pipette_pose",
        "pipette_returned",
        "pipette_stand_pose",
        "rack_pose",
        "second_press",
        "tube_pose",
        # Furniture: furniture_screw_fasten
        "screw_pose",
        "tcp_to_screw",
        # Juice: juice_open_fridge, juice_press_blender_button, juice_take_apple_to_blender
        "apple_angular_velocity",
        "apple_inside_blender",
        "apple_inside_fridge",
        "apple_linear_velocity",
        "blender_opening_center",
        "blender_xy_distance",
        "button_axial_gap",
        "button_hold_progress",
        "button_lateral_error",
        "button_link_pose",
        "button_press_direction",
        "button_press_progress",
        "button_pressed",
        "button_pressed_qpos",
        "button_qpos",
        "button_qvel",
        "button_rest_qpos",
        "button_surface_position",
        "door_qvel",
        "handle_position",
        "near_button",
        "pressing_point_distance",
        "pressing_point_to_button",
        "settle_progress",
        "tcp_to_handle",
        # Juice: juice_place_blender_lid
        "lid_angular_velocity",
        "lid_grasp_point",
        "lid_linear_velocity",
        "lid_pose",
        "lid_to_seat_position",
        "picked_once",
        "released_once",
        "seat_target_position",
        "seated_once",
        "transported_once",
        # TableClean: table_clean
        "bin_pos",
        "debris_pos",
        "n_in_bin",
        "n_in_pan",
        "pan_pose",
    }
)

PROPRIO_KEY = "proprio"


def _as_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    return torch.as_tensor(np.asarray(value))


def _batched(value: torch.Tensor, expected_ndim: int) -> torch.Tensor:
    """Add a leading batch dim when a single (unbatched) sample is passed in."""
    if value.ndim == expected_ndim:
        return value
    if value.ndim == expected_ndim - 1:
        return value.unsqueeze(0)
    raise ValueError(
        f"Expected a tensor with {expected_ndim} (or {expected_ndim - 1}) dims, "
        f"got shape {tuple(value.shape)}."
    )


def _sensor_rgb(env_obs: Mapping[str, Any], sensor_name: str) -> torch.Tensor:
    sensor_data = env_obs.get("sensor_data")
    if not isinstance(sensor_data, Mapping):
        raise KeyError(
            "Observation has no `sensor_data`; the env must be created with "
            "obs_mode='rgb' (see utils.pixel_env.make_pixel_train_env)."
        )
    if sensor_name not in sensor_data:
        raise KeyError(
            f"Camera {sensor_name!r} is missing from sensor_data "
            f"{sorted(sensor_data)}. Check `PixelObsConfig.cameras`."
        )
    camera = sensor_data[sensor_name]
    rgb = camera["rgb"] if isinstance(camera, Mapping) else camera
    rgb = _batched(_as_tensor(rgb), 4)
    if rgb.shape[-1] != 3:
        raise ValueError(f"{sensor_name} rgb must be (B, H, W, 3), got {tuple(rgb.shape)}.")
    if rgb.dtype != torch.uint8:
        raise TypeError(
            f"{sensor_name} rgb must be uint8 (it is stored raw and normalized in the "
            f"encoder), got {rgb.dtype}."
        )
    return rgb


def _proprio_field(env_obs: Mapping[str, Any], field: str) -> torch.Tensor:
    agent = env_obs.get("agent")
    extra = env_obs.get("extra")
    for container in (agent, extra, env_obs):
        if isinstance(container, Mapping) and field in container:
            return _batched(_as_tensor(container[field]).float(), 2)
    raise KeyError(
        f"Proprio field {field!r} not found in the observation. "
        f"agent keys={sorted(agent) if isinstance(agent, Mapping) else None}, "
        f"extra keys={sorted(extra) if isinstance(extra, Mapping) else None}."
    )


def build_proprio(env_obs: Mapping[str, Any], config: PixelObsConfig) -> torch.Tensor:
    """Assemble the non-privileged proprioceptive vector, in a fixed field order."""
    leaked = set(config.proprio_fields) & PRIVILEGED_ENV_OBS_KEYS
    if leaked:
        raise AssertionError(
            f"`proprio_fields` requests privileged object state {sorted(leaked)} "
            ". Proprio may only carry robot-intrinsic quantities."
        )
    parts = [_proprio_field(env_obs, field) for field in config.proprio_fields]
    return torch.cat(parts, dim=-1).float()


def extract_pixel_obs(
    env_obs: Mapping[str, Any], config: PixelObsConfig
) -> dict[str, torch.Tensor]:
    """Raw ManiSkill rgb observation -> the observation contract.

    Tensors stay on the device the env produced them on; the replay buffer
    moves images to CPU uint8 storage.
    """
    if not isinstance(env_obs, Mapping):
        raise TypeError(
            "Expected a dict observation from an obs_mode='rgb' env, got "
            f"{type(env_obs).__name__}."
        )
    obs: dict[str, torch.Tensor] = {}
    for camera in config.cameras:
        obs[camera.obs_key] = _sensor_rgb(env_obs, camera.sensor_name)
    obs[PROPRIO_KEY] = build_proprio(env_obs, config)
    return obs


# ---- contract assertions (used by verify_pixel_consistency and at train start) ----


def assert_no_privileged_keys(obs: Mapping[str, Any], config: PixelObsConfig) -> None:
    expected = set(config.camera_keys) | {PROPRIO_KEY}
    unexpected = set(obs) - expected
    if unexpected:
        raise AssertionError(
            f"Observation carries keys outside the contract: {sorted(unexpected)}. "
            f"Allowed: {sorted(expected)}."
        )
    leaked = {key for key in obs if key in PRIVILEGED_ENV_OBS_KEYS}
    if leaked:
        raise AssertionError(f"Observation leaks privileged fields: {sorted(leaked)}.")


def assert_obs_contract(
    obs: Mapping[str, Any],
    config: PixelObsConfig,
    *,
    proprio_dim: int | None = None,
    batch_size: int | None = None,
) -> None:
    """Assert keys / shapes / dtypes of one observation dict."""
    assert_no_privileged_keys(obs, config)
    height, width, channels = config.image_shape
    for key in config.camera_keys:
        if key not in obs:
            raise AssertionError(f"Observation is missing camera {key!r}.")
        image = _as_tensor(obs[key])
        if image.ndim != 4 or tuple(image.shape[1:]) != (height, width, channels):
            raise AssertionError(
                f"{key}: expected (B, {height}, {width}, {channels}), got {tuple(image.shape)}."
            )
        if image.dtype != torch.uint8:
            raise AssertionError(f"{key}: expected uint8, got {image.dtype}.")
        if batch_size is not None and image.shape[0] != batch_size:
            raise AssertionError(f"{key}: expected batch {batch_size}, got {image.shape[0]}.")
    proprio = _as_tensor(obs[PROPRIO_KEY])
    if proprio.ndim != 2:
        raise AssertionError(f"proprio: expected (B, D), got {tuple(proprio.shape)}.")
    if proprio.dtype != torch.float32:
        raise AssertionError(f"proprio: expected float32, got {proprio.dtype}.")
    if proprio_dim is not None and proprio.shape[1] != proprio_dim:
        raise AssertionError(
            f"proprio: expected dim {proprio_dim}, got {proprio.shape[1]}."
        )
    if batch_size is not None and proprio.shape[0] != batch_size:
        raise AssertionError(f"proprio: expected batch {batch_size}, got {proprio.shape[0]}.")


def report_env_obs_keys(env_obs: Mapping[str, Any], config: PixelObsConfig) -> str:
    """Human-readable dump of the raw env obs and what the contract keeps/drops."""
    lines = ["raw env observation:"]
    dropped: list[str] = []
    kept = {field: "kept -> proprio" for field in config.proprio_fields}

    def walk(mapping: Mapping[str, Any], prefix: str) -> None:
        for key, value in mapping.items():
            if isinstance(value, Mapping):
                walk(value, f"{prefix}{key}/")
                continue
            tensor = _as_tensor(value)
            mark = ""
            if prefix.startswith(("agent/", "extra/")) or prefix == "":
                if key in kept:
                    mark = f"  [{kept[key]}]"
                elif key in PRIVILEGED_ENV_OBS_KEYS:
                    mark = "  [DROPPED: privileged]"
                    dropped.append(f"{prefix}{key}")
                else:
                    mark = "  [dropped]"
            lines.append(f"  {prefix}{key}: {tuple(tensor.shape)} {tensor.dtype}{mark}")

    walk(env_obs, "")
    lines.append(f"privileged keys dropped: {sorted(dropped) if dropped else 'none present'}")
    return "\n".join(lines)
