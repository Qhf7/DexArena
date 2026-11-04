"""Named joint/RGB adapters, dataset normalization and preparation for PI0.5 LoRA."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from train.train_configs.pi05_config import TrainingContract
from utils import dex_tasks

DEFAULT_SOURCE = Path(dex_tasks.__file__).resolve()


SOURCE_SHA256 = hashlib.sha256(DEFAULT_SOURCE.read_bytes()).hexdigest()


def load_protocol(source=None):
    if source is not None and Path(source).resolve() != DEFAULT_SOURCE:
        raise ValueError("PI0.5 uses the shared utils.dex_tasks instruction table")
    return dex_tasks


def training_prompt(env_id, episode_index, source=None):
    if type(episode_index) is not int or episode_index < 0:
        raise ValueError("Invalid episode index")
    return load_protocol(source).dataset_episode_instruction(env_id, episode_index)


def evaluation_prompt(env_id, seed, source=None):
    if type(seed) is not int:
        raise ValueError("Invalid evaluation seed")
    return load_protocol(source).episode_instruction(env_id, seed)


def reorder_joints(values, source_names, target_names):
    source_names, target_names = tuple(source_names), tuple(target_names)
    if (
        len(set(source_names)) != len(source_names)
        or len(set(target_names)) != len(target_names)
        or set(source_names) != set(target_names)
    ):
        raise ValueError("Missing/duplicate/extra named joint")
    values = np.asarray(values)
    if values.ndim < 1 or values.shape[-1] != len(source_names):
        raise ValueError("Joint channel count mismatch")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite joint data")
    index = {name: i for i, name in enumerate(source_names)}
    return values[..., [index[name] for name in target_names]].copy()


SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def rgb_uint8(value):
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError("Expected RGB image")
    if image.shape[-1] != 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError("Expected three image channels")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Float RGB must be finite in[0,1]")
        image = np.rint(image * 255).astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError("RGB must be uint8 or float[0,1]")
    return image.copy()


@dataclass(frozen=True)
class JointInputs:
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    controller_names: tuple[str, ...]
    camera_keys: dict[str, str]

    def __call__(self, data):
        n = len(self.controller_names)
        if (
            n not in (29, 58)
            or "base_0_rgb" not in self.camera_keys
            or set(self.camera_keys) - set(SLOTS)
        ):
            raise ValueError("Invalid joint/camera contract")
        state = reorder_joints(data["state"], self.state_names, self.controller_names)
        if state.shape != (n,):
            raise ValueError("Expected one unbatched joint state")
        images = {slot: rgb_uint8(data[key]) for slot, key in self.camera_keys.items()}
        result = {"state": state.astype(np.float32), "image": {}, "image_mask": {}}
        for slot in SLOTS:
            result["image"][slot] = (
                images[slot] if slot in images else np.zeros_like(images["base_0_rgb"])
            )
            result["image_mask"][slot] = np.asarray(slot in images, dtype=bool)
        if not isinstance(data.get("prompt"), str) or not data["prompt"].strip():
            raise ValueError("Official episode prompt required")
        result["prompt"] = data["prompt"]
        if "actions" in data:
            # Keep native physical width for normalization before model padding.
            result["actions"] = reorder_joints(
                data["actions"], self.action_names, self.controller_names
            ).astype(np.float32)
            if result["actions"].shape != (30, n):
                raise ValueError("Expected30-step physical action chunk")
        return result


@dataclass(frozen=True)
class JointOutputs:
    physical_dim: int

    def __call__(self, data):
        actions = np.asarray(data["actions"])
        capacity = 32 if self.physical_dim == 29 else 58
        if (
            self.physical_dim not in (29, 58)
            or actions.shape != (30, capacity)
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Invalid denormalized model action dimensions")
        return {"actions": actions[:, : self.physical_dim].copy()}


def statistics(state, actions, *, state_names, action_names, controller_names):
    values = {
        "state": reorder_joints(state, state_names, controller_names),
        "actions": reorder_joints(actions, action_names, controller_names),
    }
    if (
        any(v.ndim != 2 for v in values.values())
        or values["state"].shape != values["actions"].shape
    ):
        raise ValueError("Expected aligned frame x joint arrays")
    if not len(state):
        raise ValueError("Empty training frames")
    return {
        key: dict(
            mean=v.mean(axis=0, dtype=np.float64),
            std=v.std(axis=0, dtype=np.float64),
            q01=np.quantile(v, 0.01, axis=0),
            q99=np.quantile(v, 0.99, axis=0),
        )
        for key, v in values.items()
    }


def native_stats(stats):
    from openpi.shared.normalize import NormStats

    return {key: NormStats(**value) for key, value in stats.items()}


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class TrainingView:
    def __init__(self, dataset, env_id, joint_inputs, model_transform):
        self.dataset = dataset
        self.env_id = env_id
        self.joint_inputs = joint_inputs
        self.model_transform = model_transform
        self.protocol = load_protocol()
        self.physical_dim = len(joint_inputs.controller_names)
        self.capacity = 32 if self.physical_dim == 29 else 58

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        raw = self.dataset[index]
        episode = array(raw["episode_index"])
        if episode.size != 1 or not np.issubdtype(episode.dtype, np.integer):
            raise ValueError("Expected integer episode identity")
        prompt = self.protocol.dataset_episode_instruction(self.env_id, int(episode.item()))
        if raw.get("task") != prompt:
            raise ValueError("Dataset prompt differs from official episode assignment")
        pad = array(raw["action_is_pad"])
        if pad.shape != (30,) or pad.dtype != np.bool_ or pad[0] or np.any(pad[:-1] & ~pad[1:]):
            raise ValueError("Invalid episode-tail padding mask")
        inputs = {
            "state": array(raw["observation.state"]),
            "actions": array(raw["action"]),
            "prompt": prompt,
        }
        for key in self.joint_inputs.camera_keys.values():
            inputs[key] = array(raw[key])
        physical = self.joint_inputs(inputs)
        transformed = self.model_transform(physical)
        transformed["state"] = np.asarray(transformed["state"], dtype=np.float32)
        actions = np.asarray(transformed.pop("actions"), dtype=np.float32)
        if actions.shape != (30, self.capacity) or not np.isfinite(actions).all():
            raise ValueError("Invalid transformed actions")
        mask = np.zeros(actions.shape, dtype=bool)
        mask[:, : self.physical_dim] = ~pad[:, None]
        return transformed, actions, mask


def native_transforms(norm_stats, capacity, tokenizer):
    from openpi import transforms

    if set(norm_stats) != {"state", "actions"}:
        raise ValueError("Both joint state and action statistics are required")
    # Normalize physical channels BEFORE tokenization and zero padding; no delta transform.
    return transforms.compose(
        [
            transforms.Normalize(norm_stats, use_quantiles=True, strict=True),
            transforms.ResizeImages(224, 224),
            transforms.TokenizePrompt(tokenizer, discrete_state_input=True),
            transforms.PadStatesAndActions(capacity),
        ]
    )


def open_view(root, repo_id, fps):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if fps <= 0:
        raise ValueError("Invalid control-frame rate")
    # Zero offset pairs recorded observation[t] with controller target[t].
    return LeRobotDataset(
        repo_id=repo_id, root=root, delta_timestamps={"action": [i / fps for i in range(30)]}
    )


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 << 20):
            digest.update(block)
    return digest.hexdigest()


def camera_mapping(prepared):
    return {
        key.removeprefix("observation.images.")
        .replace("base_camera", "base_0_rgb")
        .replace("left_wrist_camera", "left_wrist_0_rgb")
        .replace("right_wrist_camera", "right_wrist_0_rgb"): key
        for key in prepared["camera_keys"]
    }


def read_data_contract(path, binding):
    prepared = json.loads(Path(path).read_text())
    if prepared.get("prompt_sha256", SOURCE_SHA256) != SOURCE_SHA256:
        raise ValueError("Prepared data uses a different instruction protocol")
    names = prepared["controller_names"]
    contract = TrainingContract(dual_arm=len(names) == 58)
    contract.validate()
    if len(names) != contract.physical_dim:
        raise ValueError("Expected 29 or 58 physical joint channels")
    for key in ("state_names", "action_names"):
        reorder_joints(np.zeros(len(names)), prepared[key], names)
    if (
        binding.get("status") != "LIVE_ROBOT_CONTROLLER_BINDING_PASS"
        or prepared["task"] not in binding["tasks"]
        or binding["qpos_joint_names"] != prepared["state_names"]
        or binding["action_joint_names"] != names
        or prepared["action_names"] != names
    ):
        raise ValueError("Recorded dataset and controller binding disagree")
    task = load_protocol().resolve_dex_task(prepared["env_id"])
    if task.action_dim != len(names):
        raise ValueError("Dataset width differs from the registered task")
    if prepared["frames"] < 32 or prepared["fps"] <= 0:
        raise ValueError("Dataset needs a full batch and a positive control frequency")
    cameras = camera_mapping(prepared)
    if "base_0_rgb" not in cameras or set(cameras) - {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    }:
        raise ValueError("Unsupported camera mapping")
    if set(prepared["stats"]) != {"state", "actions"}:
        raise ValueError("Both state and action normalization are required")
    for values in prepared["stats"].values():
        if set(values) != {"mean", "std", "q01", "q99"}:
            raise ValueError("Incomplete normalization statistics")
        for value in values.values():
            array = np.asarray(value)
            if array.shape != (len(names),) or not np.isfinite(array).all():
                raise ValueError("Normalization width or numerical value is invalid")
    return prepared, contract


def prepare(dataset_root, binding, *, task, env_id):
    import pyarrow.parquet as pq

    root = Path(dataset_root).resolve()
    info = json.loads((root / "meta/info.json").read_text())
    if (
        binding.get("status") != "LIVE_ROBOT_CONTROLLER_BINDING_PASS"
        or task not in binding["tasks"]
    ):
        raise ValueError("A recorded named robot/controller binding is required")
    blocks, files = [], []
    columns = [
        "observation.state",
        "action",
        "episode_index",
        "frame_index",
        "timestamp",
        "task_index",
    ]
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=columns)
        blocks.append({key: np.asarray(table[key].to_pylist()) for key in columns})
        files.append(dict(path=path.relative_to(root).as_posix(), sha256=sha(path)))
    if not blocks:
        raise ValueError("No dataset shards found")
    data = {key: np.concatenate([block[key] for block in blocks]) for key in columns}
    if len(data["action"]) != info["total_frames"]:
        raise ValueError("Dataset frame count mismatch")
    # LeRobot shards can be split at any row. Explicit episode/frame order is the
    # source of temporal alignment, independent of shard filename ordering.
    order = np.lexsort((data["frame_index"], data["episode_index"]))
    data = {key: value[order] for key, value in data.items()}
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pandas()
    labels = {
        int(row["task_index"]): str(row["task"] if "task" in tasks.columns else index)
        for index, row in tasks.iterrows()
    }
    episodes = np.unique(data["episode_index"])
    if not np.array_equal(episodes, np.arange(info["total_episodes"])):
        raise ValueError("Missing or noncontiguous episode ids")
    for episode in episodes:
        selected = np.flatnonzero(data["episode_index"] == episode)
        if not np.array_equal(data["frame_index"][selected], np.arange(len(selected))):
            raise ValueError("Invalid episode frame indexing")
        if len(selected) > 1 and not np.allclose(
            np.diff(data["timestamp"][selected]), 1 / info["fps"], atol=1e-4
        ):
            raise ValueError("Dataset frame spacing differs from the control frequency")
        indices = np.unique(data["task_index"][selected])
        if len(indices) != 1 or labels[int(indices[0])] != training_prompt(env_id, int(episode)):
            raise ValueError("Dataset episode label differs from utils.dex_tasks")
    qnames, anames = binding["qpos_joint_names"], binding["action_joint_names"]
    stats = statistics(
        data["observation.state"],
        data["action"],
        state_names=qnames,
        action_names=anames,
        controller_names=anames,
    )
    cameras = {}
    for key, feature in info["features"].items():
        if key.startswith("observation.images."):
            shape = feature["shape"]
            if len(shape) != 3 or shape[-1] != 3:
                raise ValueError("Expected HWC camera metadata")
            cameras[key] = shape
    return dict(
        task=task,
        env_id=env_id,
        frames=info["total_frames"],
        episodes=info["total_episodes"],
        fps=info["fps"],
        state_names=qnames,
        action_names=anames,
        controller_names=anames,
        camera_keys=cameras,
        stats={k: {n: v.tolist() for n, v in values.items()} for k, values in stats.items()},
        files=files,
        prompt_sha256=SOURCE_SHA256,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binding = json.loads(args.binding.read_text())
    prepared = prepare(args.dataset_root, binding, task=args.task, env_id=args.env_id)
    with args.output.open("x") as stream:
        json.dump(prepared, stream, indent=2)


if __name__ == "__main__":
    main()
