"""GR00T modality configs for all tasks, keyed by env id (one file, was per-task pairs).

Each env maps to two things:
  - MODALITY_JSON[env]: the dataset-side splits written to ``meta/modality.json`` (read by the
    LeRobot loader).
  - the GR00T ``ModalityConfig`` dict registered under ``EmbodimentTag.NEW_EMBODIMENT`` (state/
    action splits, action representation, video keys).

Add a new task = add one entry to MODALITY_JSON and _configs() below — no new files.

Interfaces:
  - register(env_id): register that env's config under NEW_EMBODIMENT (one per process).
  - write_modality_json(env_id, dataset_path): write ``meta/modality.json`` for the dataset.
  - import side-effect: if ``$GROOT_MODALITY_ENV`` is set, auto-register it. This lets GR00T's
    ``launch_finetune.py`` (which imports ``--modality_config_path`` and expects the module to
    self-register) work with this shared file — the finetune sets the env var.
  - CLI: ``python utils/groot_utils/modality_configs.py --env-id <env_id> [--write-json <dataset>]``
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---- dataset-side splits (was <task>_modality.json), keyed by env id ----
MODALITY_JSON = {
}


# ---- dexterous-hand tasks (utils/dex_tasks.py): joint groups in controller order ----
# Single arm: single_arm 7 + hand 22. Dual arm: left_arm 7, right_arm 7, left_hand 22,
# right_hand 22 (58; GR00T N1.7 accepts up to 132 action dims).
def _dex_modality_json(task):
    groups = {name: {"start": start, "end": end} for name, start, end in task.group_slices}
    return {
        "state": dict(groups),
        "action": dict(groups),
        "video": _dex_video_json(task),
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def _dex_camera_fields():
    """(base field, wrist fields) from utils/dex_cameras.py, loaded the same way as the task
    registry: GR00T's launcher imports this file by path, where ``utils`` can resolve to an
    unrelated module (CloseFridge segment 1 died on exactly that)."""
    try:
        from utils.dex_cameras import BASE_FIELD, WRIST_FIELD_BY_SLOT  # noqa: PLC0415
    except ImportError:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "robodex_dex_cameras", REPO_ROOT / "utils" / "dex_cameras.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        BASE_FIELD, WRIST_FIELD_BY_SLOT = module.BASE_FIELD, module.WRIST_FIELD_BY_SLOT
    return BASE_FIELD, WRIST_FIELD_BY_SLOT


def _dex_video_keys(task):
    """GR00T video keys in rig order: the base view, then one wrist view per arm.

    ``$GROOT_VIDEO_KEYS`` ("head:base_0_rgb,right_wrist:right_wrist_0_rgb") overrides the rule
    for datasets that deliberately carry fewer views than the robot has arms -- the
    package_opening family dataset keeps only the base and the right wrist because its
    single-arm subtask has no left wrist to render (ported from the family round's
    worktree so the same family trains and evaluates from this one).
    """
    override = os.environ.get("GROOT_VIDEO_KEYS")
    if override:
        return [tuple(item.split(":", 1)) for item in override.split(",") if item]
    BASE_FIELD, WRIST_FIELD_BY_SLOT = _dex_camera_fields()

    arms = sum(1 for name, _ in task.groups if name.endswith("arm"))
    keys = [("head", BASE_FIELD)]
    if arms == 1:
        keys.append(("wrist", WRIST_FIELD_BY_SLOT[0]))
    else:
        keys += [("left_wrist", WRIST_FIELD_BY_SLOT[0]), ("right_wrist", WRIST_FIELD_BY_SLOT[1])]
    return keys


def _dex_video_json(task):
    return {key: {"original_key": f"observation.images.{field}"} for key, field in _dex_video_keys(task)}


def _dex_tasks():
    """The dex task registry, importable even when GR00T's launcher loads this file by path.

    ``launch_finetune.py`` imports the modality module by its stem with its own directory
    first on sys.path, where ``utils`` can resolve to an unrelated module; so fall back to
    loading the registry straight from its file.
    """
    try:
        from utils.dex_tasks import DEX_TASKS  # noqa: PLC0415
    except ImportError:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "robodex_dex_tasks", REPO_ROOT / "utils" / "dex_tasks.py"
        )
        module = importlib.util.module_from_spec(spec)
        # dataclass processing looks the module up in sys.modules, so register it first.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        DEX_TASKS = module.DEX_TASKS
    return DEX_TASKS


MODALITY_JSON.update({task.env_id: _dex_modality_json(task) for task in _dex_tasks()})


def _configs():
    """Build the per-env GR00T ModalityConfig dicts (lazy import of gr00t types)."""
    from gr00t.data.types import (
        ActionConfig,
        ActionFormat,
        ActionRepresentation,
        ActionType,
        ModalityConfig,
    )

    def _mc_video():
        return ModalityConfig(delta_indices=[0], modality_keys=["head", "wrist"])

    def _mc_state():
        return ModalityConfig(delta_indices=[0], modality_keys=["single_arm", "gripper"])

    def _mc_lang():
        return ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"])

    def _dex_config(task):
        video_keys = [k for k, _ in _dex_video_keys(task)]
        keys = [name for name, _ in task.groups]
        # arm joints as deltas, finger joints as absolute targets (dexjoco convention)
        reps = [ActionRepresentation.RELATIVE if k.endswith("arm") else ActionRepresentation.ABSOLUTE for k in keys]
        return {
            "video": ModalityConfig(delta_indices=[0], modality_keys=video_keys),
            "state": ModalityConfig(delta_indices=[0], modality_keys=list(keys)),
            "action": ModalityConfig(
                delta_indices=list(range(0, 16)),
                modality_keys=list(keys),
                action_configs=[
                    ActionConfig(rep=rep, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT) for rep in reps
                ],
            ),
            "language": _mc_lang(),
        }

    configs = {task.env_id: _dex_config(task) for task in _dex_tasks()}
    configs.update({
    })
    return configs


def register(env_id: str) -> None:
    """Register ``env_id``'s modality config under EmbodimentTag.NEW_EMBODIMENT."""
    from gr00t.configs.data.embodiment_configs import register_modality_config
    from gr00t.data.embodiment_tags import EmbodimentTag

    cfgs = _configs()
    if env_id not in cfgs:
        raise KeyError(f"No GR00T modality config for env '{env_id}'. Known: {sorted(cfgs)}")
    register_modality_config(cfgs[env_id], embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)


def write_modality_json(env_id: str, dataset_path) -> Path:
    """Write ``<dataset_path>/meta/modality.json`` for ``env_id``."""
    if env_id not in MODALITY_JSON:
        raise KeyError(f"No modality.json for env '{env_id}'. Known: {sorted(MODALITY_JSON)}")
    out = Path(dataset_path) / "meta" / "modality.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(MODALITY_JSON[env_id], indent=4) + "\n")
    print(f"[modality] wrote {out} for env {env_id}")
    return out


# Auto-register on import for GR00T's launch_finetune (it imports the module by path and expects
# a self-registering side-effect). The finetune sets $GROOT_MODALITY_ENV to the target env.
_ENV = os.environ.get("GROOT_MODALITY_ENV")
if _ENV:
    register(_ENV)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GR00T modality configs keyed by env id.")
    ap.add_argument("--env-id", required=True, help="env id key (e.g. CabinetPouringOpenCabinet-Sharpa-v1)")
    ap.add_argument("--write-json", metavar="DATASET_PATH", default=None,
                    help="write meta/modality.json into this dataset for --env")
    a = ap.parse_args()
    if a.write_json:
        write_modality_json(a.env_id, a.write_json)
    else:
        register(a.env_id)
        print(f"[modality] registered {a.env_id} under NEW_EMBODIMENT")
