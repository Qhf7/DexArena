"""Strict inference-only Base + trainable parameter overlay (not a resume state)."""

import hashlib
import json
from pathlib import Path

import numpy as np

from train.train_configs.pi05_config import DEXJOCO_COMMIT, TrainingContract
from utils.pi05_utils.checkpoint import sha


def flatten(tree, prefix=""):
    result = {}
    for key, value in tree.items():
        if not isinstance(key, str) or not key or "/" in key:
            raise ValueError("Invalid model parameter key")
        path = prefix + "/" + key if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, path))
        else:
            result[path] = value
    return result


def array_sha(value):
    # BF16 does not implement Python's buffer format; hash its raw byte view.
    value = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(value.view(np.uint8).reshape(-1))).hexdigest()


def package_path(root, name):
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError("Package path must be relative")
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Missing or escaping package path: " + name)
    return path


def read_compact_config(package):
    """Check the published certificate, file hashes and pinned model contract."""
    root = Path(package).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "EXACT_ALL_PARAMETER_BYTES_RECONSTRUCTION_PASS":
        raise ValueError("Compact package lacks exact reconstruction certificate")
    records = manifest.get("files", [])
    names = [r["path"] for r in records]
    if len(set(names)) != len(names) or not {"config.json", "ckpt/trainable.npz"} <= set(names):
        raise ValueError("Incomplete or duplicate compact file inventory")
    for record in records:
        path = package_path(root, record["path"])
        if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError("Compact file checksum mismatch: " + record["path"])
    config = json.loads((root / "config.json").read_text())
    if (
        config.get("schema_version") != 1
        or config.get("method") != "pi05"
        or config.get("format") != "pi05_trainable_npz_v1"
        or config.get("inference_only") is not True
        or config.get("optimizer_state_included") is not False
        or config.get("checkpoint") != "ckpt/trainable.npz"
        or config.get("source_commits", {}).get("dexjoco") != DEXJOCO_COMMIT
    ):
        raise ValueError("Unsupported compact model format/source")
    if type(config["training"].get("dual_arm")) is not bool:
        raise ValueError("Invalid dual-arm contract")
    contract = TrainingContract(**config["training"])
    contract.validate()
    if config["optimizer_updates"] != contract.steps:
        raise ValueError("Compact checkpoint is not the final training update")
    archive_record = next(r for r in records if r["path"] == config["checkpoint"])
    if archive_record["sha256"] != config["trainable_sha256"]:
        raise ValueError("Trainable archive digest differs from certificate")
    if (
        manifest["identity_sha256"] != config["identity_sha256"]
        or manifest["base_manifest_sha256"] != config["base_manifest_sha256"]
    ):
        raise ValueError("Compact certificate identity mismatch")
    prepared = config["data_contract"]
    n = contract.physical_dim
    if prepared["env_id"] != config["env_id"] or prepared["task"] != config["task"]:
        raise ValueError("Compact tasks/data identity mismatch")
    for key in ("state_names", "action_names", "controller_names"):
        names = prepared[key]
        if len(names) != n or any(not isinstance(x, str) or not x for x in names) or len(set(names)) != n:
            raise ValueError("Compact named joint dimension mismatch")
    if set(prepared["stats"]) != {"state", "actions"}:
        raise ValueError("Incomplete compact normalization")
    for kind in ("state", "actions"):
        if set(prepared["stats"][kind]) != {"mean", "std", "q01", "q99"}:
            raise ValueError("Incomplete compact normalization")
        for key in ("mean", "std", "q01", "q99"):
            values = np.asarray(prepared["stats"][kind][key])
            if values.shape != (n,) or not np.isfinite(values).all():
                raise ValueError("Compact normalization dimensions/numerics mismatch")
    return config


def restore_base(base):
    from algo.pi05 import restore_host_payload

    tree = restore_host_payload(Path(base) / "params")
    return tree["params"] if set(tree) == {"params"} else tree


def load_compact_params(package, base, expected_params):
    """Overlay ALL trained arrays, preserving dtype; reject partial/extra trees."""
    root = Path(package).resolve()
    config = read_compact_config(root)
    # Same exact base-file admission as resumable loading, with embedded manifest.
    base = Path(base).resolve()
    records = config["base_manifest"]
    if not records:
        raise ValueError("Missing pinned base manifest")
    expected_files = set()
    for record in records:
        name = record["name"].removeprefix("checkpoints/pi05_base/")
        path = (base / name).resolve()
        if not path.is_relative_to(base / "params") or name in expected_files:
            raise ValueError("Invalid or duplicate base file path")
        expected_files.add(name)
        if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError("Pinned base file checksum mismatch: " + name)
    actual_files = {p.relative_to(base).as_posix() for p in (base / "params").rglob("*") if p.is_file()}
    if actual_files != expected_files:
        raise ValueError("Pinned base inventory mismatch")
    expected = flatten(expected_params)
    rows = config["parameter_manifest"]
    paths = [r["path"] for r in rows]
    if not expected or len(set(paths)) != len(paths) or set(paths) != set(expected):
        raise ValueError("Compact parameter tree differs from the model")
    if any(type(r["trainable"]) is not bool for r in rows):
        raise ValueError("Invalid trainable parameter flag")
    pretrained = flatten(restore_base(base))
    params = {}
    with np.load(package_path(root, config["checkpoint"]), allow_pickle=False) as archive:
        wanted = {r["path"] for r in rows if r["trainable"]}
        if len(archive.files) != len(wanted) or set(archive.files) != wanted:
            raise ValueError("Missing, duplicate or unexpected trainable arrays")
        for record in rows:
            path = record["path"]
            reference = expected[path]
            if record["shape"] != list(reference.shape) or record["dtype"] != str(reference.dtype):
                raise ValueError("Manifest shape/dtype differs from model: " + path)
            if record["trainable"]:
                value = archive[path]
            else:
                if path not in pretrained:
                    raise ValueError("Missing frozen base parameter: " + path)
                value = np.asarray(pretrained[path]).astype(reference.dtype, copy=False)
            if (
                value.shape != reference.shape
                or value.dtype != reference.dtype
                or not np.isfinite(value).all()
                or array_sha(value) != record["sha256"]
            ):
                raise ValueError("Reconstructed parameter mismatch: " + path)
            node = params
            parts = path.split("/")
            for key in parts[:-1]:
                node = node.setdefault(key, {})
            node[parts[-1]] = value
    return params
