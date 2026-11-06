"""Fixed DexArena image-policy cameras for the dexterous-hand tasks.

The published datasets (`SIPAIR/DexArena`, `data/<hand>/<env id>/`) ship videos
rendered with the rig in ``utils/dex_camera_rigs.py`` (``family_rig_v1_z1p08_overhead20``):
ONE world-fixed base camera per task FAMILY (main's rule ``z1p08_overhead20_back40cm_v1`` --
1.08 m high, 20 deg from the tabletop, behind the family's scene centroid along -X, 256x256,
90 deg vertical FOV) plus the calibrated Marvin wrist camera on every arm the subtask uses.
This module installs exactly that rig on the env so our renders and evaluations match the
published videos pixel-for-pixel.

The task modules still register their own historical base camera and the robots their runtime
wrist camera, so the swap happens at env-construction time: the dataset renderer and every
evaluator call :func:`install_camera_profile` before ``gym.make``. Wrong or missing camera
views silently change the observation distribution, so a task with no rig family is refused
(not defaulted). The scene-evaluation factory restores the registration entry afterwards.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

CAMERA_PROFILE_NONE = "none"            # whatever the task module / robot register
CAMERA_PROFILE_FIXED = "robodex_fixed"  # the published per-family rig
CAMERA_PROFILES = (CAMERA_PROFILE_NONE, CAMERA_PROFILE_FIXED)

BASE_CAMERA_NAME = "base_camera"
# Dataset fields our renderer writes, in rig camera order. The single wrist of a right-handed
# subtask also lands in ``left_wrist_0_rgb``: the name is the SLOT, not the side, and keeping it
# means the datasets and modality configs of the already-trained runs stay valid.
WRIST_FIELD_BY_SLOT = ("left_wrist_0_rgb", "right_wrist_0_rgb")
BASE_FIELD = "base_0_rgb"


@dataclass(frozen=True)
class FixedCameraSpec:
    env_id: str
    family: str
    version: str
    base_camera: str
    wrist_cameras: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def _rigs():
    from utils import dex_camera_rigs

    return dex_camera_rigs


def family_of(env_id: str) -> str:
    """Rig family of ``env_id``; raises when the task has no accepted fixed view."""
    rigs = _rigs()
    try:
        return rigs.family_of(env_id)
    except KeyError:
        raise ValueError(
            f"{env_id} has no camera-rig family; add a rule in utils/dex_camera_rigs.py "
            f"(families: {sorted(rigs.FAMILIES)})."
        ) from None


def resolve_fixed_camera_spec(env_id: str) -> FixedCameraSpec:
    """Rig description for ``env_id``. The wrist cameras are only known once the robot is
    built (one per arm present), so they are filled in by :func:`install_fixed_cameras`."""
    rigs = _rigs()
    family = family_of(env_id)
    rigs.base_camera_spec(family)  # raises if the family has no target
    return FixedCameraSpec(
        env_id=env_id, family=family, version=rigs.RIG_VERSION,
        base_camera=BASE_CAMERA_NAME, wrist_cameras=(),
    )


def _hide_agent_sensors(agent) -> None:
    """The robot's own runtime wrist camera must not be rendered (or picked up by the camera
    aliases) once the rig's wrist views are installed. ``_sensor_configs`` is a class property,
    so the instance is moved to a subclass that returns none."""
    if agent is None or getattr(type(agent), "_dexarena_sensors_hidden", False):
        return
    cls = type(agent)
    hidden = type(
        cls.__name__ + "NoRuntimeSensors",
        (cls,),
        {"_sensor_configs": property(lambda self: []), "_dexarena_sensors_hidden": True},
    )
    agent.__class__ = hidden


def install_fixed_cameras(env_id: str) -> FixedCameraSpec:
    """Make ``gym.make(env_id)`` build the published rig instead of the task's own cameras.

    Replaces the registered env class with a subclass whose ``_default_sensor_configs`` is the
    rig and whose agent contributes no sensors. Idempotent. Call after the task modules are
    imported (they register the env) and before the env is created.
    """
    from mani_skill.utils.registration import REGISTERED_ENVS

    rigs = _rigs()
    spec = resolve_fixed_camera_spec(env_id)
    try:
        env_spec = REGISTERED_ENVS[env_id]
    except KeyError:
        raise KeyError(f"{env_id} is not registered; import its task module first") from None
    base_cls = env_spec.cls
    if getattr(base_cls, "_dexarena_rig_family", None) == spec.family:
        return base_cls._dexarena_fixed_camera_spec

    family = spec.family

    class RigCameraEnv(base_cls):
        _dexarena_rig_family = family
        _dexarena_fixed_camera_spec = spec

        @property
        def _default_sensor_configs(self):
            cameras = rigs.rig_cameras(self, family)
            # Record what was actually built (arm count is only known now).
            type(self)._dexarena_fixed_camera_spec = FixedCameraSpec(
                env_id=spec.env_id, family=family, version=spec.version,
                base_camera=BASE_CAMERA_NAME,
                wrist_cameras=tuple(c.uid for c in cameras[1:]),
            )
            return cameras

        def _setup_sensors(self, options: dict):
            _hide_agent_sensors(self.agent)
            super()._setup_sensors(options)

    RigCameraEnv.__name__ = base_cls.__name__ + "RigCameras"
    RigCameraEnv.__qualname__ = RigCameraEnv.__name__
    RigCameraEnv.__module__ = base_cls.__module__
    env_spec.cls = RigCameraEnv
    return spec


def camera_fields(spec: FixedCameraSpec) -> dict[str, str]:
    """Rig camera name -> dataset image field, in rig order (base first, then wrists)."""
    fields = {spec.base_camera: BASE_FIELD}
    for slot, name in enumerate(spec.wrist_cameras):
        fields[name] = WRIST_FIELD_BY_SLOT[slot]
    return fields


def install_camera_profile(env_id: str, profile: str | None) -> dict | None:
    """Apply ``profile`` for ``env_id``; returns the spec dict (for manifests) or None."""
    if profile in (None, "", CAMERA_PROFILE_NONE):
        return None
    if profile != CAMERA_PROFILE_FIXED:
        raise ValueError(f"unknown camera profile {profile!r}; choose from {CAMERA_PROFILES}")
    spec = install_fixed_cameras(env_id)
    print(f"[camera] {env_id}: profile={profile} rig={spec.version} family={spec.family} "
          f"base={spec.base_camera} 256x256 fov90")
    return {"profile": profile, **spec.as_dict()}
