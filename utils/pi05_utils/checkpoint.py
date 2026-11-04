"""Run metadata, resumable checkpoints and per-seed evaluation results."""

import dataclasses
import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np

from train.train_configs.pi05_config import TrainingContract
from utils.pi05_utils.data import SOURCE_SHA256, read_data_contract, sha


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class Checkpoints:
    def __init__(self, root, identity, *, resume=False):
        self.root = Path(root).resolve()
        self.identity = identity
        self.identity_hash = hashlib.sha256(canonical(identity).encode()).hexdigest()
        contract = self.root / "identity.json"
        if resume:
            if json.loads(contract.read_text()) != identity:
                raise ValueError("Resume identity mismatch")
        else:
            self.root.mkdir(parents=True, exist_ok=False)
            with contract.open("x") as f:
                f.write(canonical(identity))

    def save(self, payload):
        import orbax.checkpoint as ocp

        step = int(np.asarray(payload["step"]))
        if step < 0:
            raise ValueError("Negative update")
        directory = self.root / f"update-{step:08d}"
        if directory.exists():
            raise ValueError("Checkpoint update already exists; never overwrite")
        # A killed save must not obstruct replay from the previous checkpoint.
        # Interrupted staging directories remain preserved but are never selected.
        staging = self.root / f"staging-{step:08d}-{uuid.uuid4().hex}"
        with ocp.PyTreeCheckpointer() as checkpointer:
            checkpointer.save(str(staging), payload)
        with (staging / "complete.json").open("x") as f:
            json.dump({"optimizer_updates": step, "identity_sha256": self.identity_hash}, f)
            f.flush()
            os.fsync(f.fileno())
        # Same filesystem atomic publication, while the caller holds the output lock.
        staging.rename(directory)
        return directory

    def latest(self):
        candidates = []
        for p in self.root.glob("update-*/complete.json"):
            record = json.loads(p.read_text())
            step = record["optimizer_updates"]
            if (
                record["identity_sha256"] != self.identity_hash
                or p.parent.name != f"update-{step:08d}"
            ):
                raise ValueError("Checkpoint completion identity mismatch")
            candidates.append((step, p.parent))
        if not candidates:
            raise ValueError("No complete resumable checkpoint")
        return max(candidates)[1]

    def restore(self, target):
        import orbax.checkpoint as ocp

        directory = self.latest()
        with ocp.PyTreeCheckpointer() as checkpointer:
            restored = checkpointer.restore(str(directory), item=target)
        if int(np.asarray(restored["step"])) != int(directory.name.removeprefix("update-")):
            raise ValueError("Saved optimizer update disagrees with directory")
        return restored


def pack_state(state, train_rng):
    import jax

    return {
        "params": state.params.to_pure_dict(),
        "opt_state": state.opt_state,
        "step": np.asarray(state.step),
        "train_rng": np.asarray(jax.random.key_data(train_rng)),
    }


def unpack_state(state, payload):
    import flax.nnx as nnx
    import jax

    # Preserve the target graph/variable metadata, replacing only array values.
    params = nnx.State(state.params)
    params.replace_by_pure_dict(payload["params"])
    return dataclasses.replace(
        state, params=params, opt_state=payload["opt_state"], step=payload["step"]
    ), jax.random.wrap_key_data(payload["train_rng"])


def inspect_final(output, expected_identity, expected_data):
    root = Path(output).resolve()
    identity = json.loads((root / "identity.json").read_text())
    if identity != expected_identity or identity.get("baseline") != BASELINE:
        raise ValueError("Unexpected final run identity")
    contract = TrainingContract(**identity["config"])
    contract.validate()
    if (
        identity.get("config_sha256")
        != hashlib.sha256(canonical(dataclasses.asdict(contract)).encode()).hexdigest()
    ):
        raise ValueError("Config identity mismatch")
    if json.loads((root / "data-contract.json").read_text()) != expected_data:
        raise ValueError("Final normalization/data binding mismatch")
    checkpoints = root / "checkpoints"
    if json.loads((checkpoints / "identity.json").read_text()) != identity:
        raise ValueError("Checkpoint run identity mismatch")
    final = json.loads((root / "FINAL.json").read_text())
    if final["optimizer_updates"] != contract.steps:
        raise ValueError("Final update count incomplete")
    directory = Path(final["checkpoint"])
    if not directory.is_absolute():
        directory = root / directory
    directory = directory.resolve()
    if directory != checkpoints / f"update-{contract.steps:08d}" or not directory.is_dir():
        raise ValueError("Final checkpoint must be exact published update in this run")
    complete = json.loads((directory / "complete.json").read_text())
    digest = hashlib.sha256(canonical(identity).encode()).hexdigest()
    if complete != {"optimizer_updates": contract.steps, "identity_sha256": digest}:
        raise ValueError("Final checkpoint completion identity mismatch")
    return dict(
        directory=directory,
        identity_sha256=digest,
        optimizer_updates=contract.steps,
        status="METADATA_ONLY_REQUIRES_RESTORE",
    )


def verify_restored(payload, expected_params, steps):
    if set(payload) != {"params", "opt_state", "step", "train_rng"}:
        raise ValueError("Incomplete full training checkpoint payload")
    if int(np.asarray(payload["step"])) != steps or np.asarray(payload["train_rng"]).shape != (2,):
        raise ValueError("Restored update/RNG mismatch")

    def walk(actual, expected, path="params"):
        if isinstance(expected, dict):
            if not isinstance(actual, dict) or set(actual) != set(expected):
                raise ValueError("Restored parameter keys mismatch:" + path)
            for key in expected:
                walk(actual[key], expected[key], path + "/" + key)
        else:
            values = np.asarray(actual)
            if (
                values.shape != expected.shape
                or values.dtype != expected.dtype
                or not np.isfinite(values).all()
            ):
                raise ValueError("Restored parameter shape/dtype/numerics mismatch:" + path)

    if not expected_params:
        raise ValueError("Empty model tree")
    walk(payload["params"], expected_params)
    return "RESTORED_MODEL_PARAMETERS_VERIFIED"


class EvalJournal:
    def __init__(self, root, identity, *, resume=False):
        self.root = Path(root)
        self.identity = identity
        self.digest = hashlib.sha256(canonical(identity).encode()).hexdigest()
        if resume:
            if json.loads((self.root / "identity.json").read_text()) != identity:
                raise ValueError("Evaluation identity mismatch")
        else:
            self.root.mkdir(parents=True, exist_ok=False)
            (self.root / "attempts").mkdir()
            (self.root / "completed").mkdir()
            with (self.root / "identity.json").open("x") as f:
                f.write(canonical(identity))

    def completed(self):
        results = {}
        for path in (self.root / "completed").glob("seed-*.json"):
            value = json.loads(path.read_text())
            seed = value["seed"]
            if (
                type(seed) is not int
                or seed not in range(10001, 10051)
                or path.name != f"seed-{seed}.json"
                or value["identity_sha256"] != self.digest
                or value["status"] != "completed"
                or type(value["success"]) is not bool
            ):
                raise ValueError("Invalid published episode")
            results[seed] = value
        return results

    def pending(self):
        completed = self.completed()
        return [s for s in range(10001, 10051) if s not in completed]

    def record(self, result):
        seed = result["seed"]
        status = result["status"]
        if type(seed) is not int or seed not in range(10001, 10051):
            raise ValueError("Invalid seed")
        if status not in ("completed", "infrastructure_error"):
            raise ValueError("Invalid outcome class")
        if (
            status == "completed"
            and type(result.get("success")) is not bool
            or status == "infrastructure_error"
            and result.get("success") is not None
        ):
            raise ValueError("Infrastructure failure must remain unscored")
        value = dict(result, identity_sha256=self.digest)
        attempt = self.root / "attempts" / f"{seed}-{uuid.uuid4().hex}.json"
        with attempt.open("x") as f:
            json.dump(value, f)
            f.flush()
            os.fsync(f.fileno())
        if status == "completed":
            os.link(attempt, self.root / "completed" / f"seed-{seed}.json")
        return attempt

    def summary(self):
        values = self.completed()
        n = len(values)
        success = sum(v["success"] for v in values.values())
        return dict(
            completed_episodes=n,
            successes=success,
            status="complete" if n == 50 else "incomplete",
            success_rate=success / 50 if n == 50 else None,
        )


BASELINE = "dexarena-pi05-openpi-lora-v1"


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def source_identity():
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "utils/pi05_utils").glob("*.py")) + [
        root / "algo/pi05.py",
        root / "train/train_pi05.py",
        root / "train/train_configs/pi05_config.py",
    ]
    return digest({p.relative_to(root).as_posix(): sha(p) for p in sorted(paths)})


def verify_base(root, manifest):
    root = Path(root).resolve()
    records = json.loads(Path(manifest).read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Base manifest must list checkpoint file names, sizes and SHA256 values")
    expected = set()
    for record in records:
        name = record["name"].removeprefix("checkpoints/pi05_base/")
        path = (root / name).resolve()
        if not path.is_relative_to(root / "params") or name in expected:
            raise ValueError("Invalid or duplicate pretrained parameter path")
        expected.add(name)
        if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError("Pretrained checkpoint differs from its manifest: " + name)
    actual = {p.relative_to(root).as_posix() for p in (root / "params").rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("Pretrained checkpoint file inventory differs from its manifest")


def read_run(root):
    root = Path(root).resolve()
    identity = json.loads((root / "identity.json").read_text())
    prepared = json.loads((root / "data-contract.json").read_text())
    binding = json.loads((root / "controller-binding.json").read_text())
    if identity.get("baseline") != BASELINE:
        raise ValueError("Expected an OpenPI LoRA run from this entry point")
    if (
        identity.get("prompt_sha256") != SOURCE_SHA256
        or identity.get("dataset_sha256") != digest(prepared)
        or identity.get("binding_sha256") != digest(binding)
    ):
        raise ValueError("Run data, binding or instruction protocol changed")
    read_data_contract(root / "data-contract.json", binding)
    return identity, prepared, binding
