"""PI0.5 LoRA optimizer updates, sampling and the supervised training loop."""

import dataclasses
import functools
import json
import time
import uuid
from pathlib import Path

import numpy as np

from algo.pi05 import NativeWeightLoader, masked_flow_loss, model_config
from train.train_configs.pi05_config import TrainingContract
from utils.pi05_utils.checkpoint import pack_state, unpack_state


def training_config(*, name, dual_arm, data_factory, pretrained_params, output_root, experiment):
    from openpi.training.config import TrainConfig
    from openpi.training.optimizer import AdamW, CosineDecaySchedule

    c = TrainingContract(dual_arm=dual_arm)
    model = model_config(dual_arm)
    return TrainConfig(
        name=name,
        project_name="dexarena-lora-v1",
        exp_name=experiment,
        model=model,
        data=data_factory,
        weight_loader=NativeWeightLoader(pretrained_params),
        freeze_filter=model.get_freeze_filter(),
        batch_size=c.batch_size,
        num_train_steps=c.steps,
        seed=c.seed,
        lr_schedule=CosineDecaySchedule(
            warmup_steps=c.warmup_steps,
            peak_lr=c.peak_lr,
            decay_steps=c.decay_steps,
            decay_lr=c.decay_lr,
        ),
        optimizer=AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        save_interval=10000,
        keep_period=None,
        wandb_enabled=False,
        overwrite=False,
        resume=False,
        checkpoint_base_dir=str(output_root),
        num_workers=4,
        fsdp_devices=1,
    )


def initialize(config, init_rng):
    """Use real initialized tails for58D, not abstract shape placeholders."""
    import flax.nnx as nnx
    import jax.numpy as jnp
    from openpi.shared import nnx_utils
    from openpi.training import optimizer, utils

    model = config.model.create(init_rng)
    graph, params = nnx.split(model)
    loaded = config.weight_loader.load(params.to_pure_dict())
    params.replace_by_pure_dict(loaded)
    params = nnx_utils.state_map(
        params, config.freeze_filter, lambda p: p.replace(jnp.asarray(p.value, dtype=jnp.bfloat16))
    )
    tx = optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    return utils.TrainState(
        step=0,
        params=params,
        model_def=graph,
        tx=tx,
        opt_state=tx.init(params.filter(config.trainable_filter)),
        ema_decay=None,
        ema_params=None,
    )


def train_step(config, rng, state, batch):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax

    model = nnx.merge(state.model_def, state.params)
    model.train()
    observation, actions, mask = batch

    def loss_fn(model):
        return jnp.mean(
            masked_flow_loss(
                model, jax.random.fold_in(rng, state.step), observation, actions, mask, train=True
            )
        )

    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, config.trainable_filter))(
        model
    )
    params = state.params.filter(config.trainable_filter)
    updates, opt_state = state.tx.update(grads, state.opt_state, params)
    nnx.update(model, optax.apply_updates(params, updates))
    return dataclasses.replace(
        state, step=state.step + 1, params=nnx.state(model), opt_state=opt_state
    ), {"loss": loss, "grad_norm": optax.global_norm(grads)}


def parameter_inventory(state, config):
    import flax.traverse_util as traverse

    all_leaves = traverse.flatten_dict(state.params.to_pure_dict(), sep="/")
    trainable = traverse.flatten_dict(
        state.params.filter(config.trainable_filter).to_pure_dict(), sep="/"
    )
    rows = [
        dict(path=p, shape=list(x.shape), dtype=str(x.dtype), trainable=p in trainable)
        for p, x in all_leaves.items()
    ]
    lora = [r for r in rows if "lora" in r["path"]]
    if not lora or not all(r["trainable"] for r in lora):
        raise ValueError("Expected trainable LoRA leaves")
    return rows


def train_step_accum(config, rng, state, batch):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax

    from algo.pi05 import masked_flow_loss

    observation, actions, mask = batch
    if actions.shape[0] != 32 or config.batch_size != 32:
        raise ValueError("Accumulation must retain global batch32")
    microbatches = jax.tree.map(lambda x: x.reshape((2, 16) + x.shape[1:]), batch)
    params = state.params.filter(config.trainable_filter)
    zeros = jax.tree.map(jnp.zeros_like, params)
    step_rng = jax.random.fold_in(rng, state.step)

    def micro_step(carry, inputs):
        loss_sum, grad_sum = carry
        index, (obs, act, valid) = inputs
        model = nnx.merge(state.model_def, state.params)
        model.train()

        def loss_fn(model):
            return jnp.mean(
                masked_flow_loss(
                    model, jax.random.fold_in(step_rng, index), obs, act, valid, train=True
                )
            )

        loss, grads = nnx.value_and_grad(
            loss_fn, argnums=nnx.DiffState(0, config.trainable_filter)
        )(model)
        grads = jax.tree.map(lambda total, value: total + value * 0.5, grad_sum, grads)
        return (loss_sum + loss * 0.5, grads), None

    (loss, grads), _ = jax.lax.scan(
        micro_step, (jnp.asarray(0.0), zeros), (jnp.arange(2), microbatches)
    )
    updates, opt_state = state.tx.update(grads, state.opt_state, params)
    model = nnx.merge(state.model_def, state.params)
    nnx.update(model, optax.apply_updates(params, updates))
    return dataclasses.replace(
        state, step=state.step + 1, params=nnx.state(model), opt_state=opt_state
    ), {"loss": loss, "grad_norm": optax.global_norm(grads)}


def batch_indices(dataset_size, update_index, *, seed=1000, batch_size=32):
    if (
        type(dataset_size) is not int
        or dataset_size < batch_size
        or type(update_index) is not int
        or update_index < 0
        or batch_size != 32
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError("Invalid fixed-batch sampling contract")
    batches = dataset_size // batch_size
    epoch, offset = divmod(update_index, batches)
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    order = rng.permutation(dataset_size)
    return order[offset * batch_size : (offset + 1) * batch_size]


def train(config, state, rng, view, checkpoints, output, *, resume=False, accumulation_steps=1):
    import jax
    from openpi.models.model import Observation

    contract = TrainingContract(dual_arm=config.model.action_dim == 58)
    contract.validate()
    if (
        config.num_train_steps != contract.steps
        or config.batch_size != 32
        or config.model.action_horizon != 30
    ):
        raise ValueError("Formal training contract mismatch")
    if resume:
        state, rng = unpack_state(state, checkpoints.restore(pack_state(state, rng)))
    start = int(state.step)
    if not 0 <= start <= contract.steps:
        raise ValueError("Invalid restored update count")
    if accumulation_steps not in (1, 2):
        raise ValueError("Unsupported accumulation")
    implementation = train_step
    if accumulation_steps == 2:
        implementation = train_step_accum
    update = jax.jit(functools.partial(implementation, config), donate_argnums=(1,))
    output = Path(output)
    # One exclusive immutable record per completed update; no shared overwrite.
    # A restart may replay updates after the last checkpoint; retain previous
    # attempt logs without conflicting with repeated update numbers.
    progress = output / "progress" / ("attempt-" + uuid.uuid4().hex)
    progress.mkdir(parents=True, exist_ok=False)
    for step in range(start, contract.steps):
        begin = time.monotonic()
        indices = batch_indices(len(view), step, seed=contract.seed)
        samples = [view[int(index)] for index in indices]
        observations, actions, mask = jax.tree.map(lambda *values: np.stack(values), *samples)
        if actions.shape != (32, 30, contract.action_dim) or mask.shape != actions.shape:
            raise ValueError("Incorrect full batch or mask shape")
        if not np.all(mask.any(axis=(-2, -1))) or not np.isfinite(actions).all():
            raise ValueError("Invalid numerical batch")
        batch = (Observation.from_dict(observations), actions, mask)
        state, metrics = update(rng, state, batch)
        jax.block_until_ready(state)
        metrics = {k: float(v) for k, v in metrics.items()}
        if int(state.step) != step + 1 or not all(np.isfinite(v) for v in metrics.values()):
            raise ValueError("Nonfinite update or counter mismatch; no completion claim")
        record = {
            "optimizer_updates": step + 1,
            "wall_seconds": time.monotonic() - begin,
            **metrics,
        }
        if (step + 1) % 20 == 0 or step == start:
            with (progress / f"update-{step + 1:08d}.json").open("x") as f:
                json.dump(record, f)
            print(json.dumps(record), flush=True)
        if (step + 1) % 10000 == 0 or step + 1 == contract.steps:
            checkpoints.save(pack_state(state, rng))
    # Only after final synchronous save completed. Resume-at-final is also safe.
    if (
        int(state.step) == contract.steps
        and int(checkpoints.latest().name.removeprefix("update-")) == contract.steps
    ):
        marker = output / "FINAL.json"
        if not marker.exists():
            with marker.open("x") as f:
                json.dump(
                    {
                        "optimizer_updates": contract.steps,
                        "checkpoint": checkpoints.latest().relative_to(output.resolve()).as_posix(),
                    },
                    f,
                )
    return state


def finetune(args):
    """Run supervised LoRA fine-tuning or resume the same run."""
    import os

    from filelock import FileLock

    from train.train_configs.pi05_config import DEXJOCO_COMMIT
    from utils.pi05_utils.checkpoint import (
        BASELINE,
        digest,
        source_identity,
        verify_base,
    )
    from utils.pi05_utils.data import (
        SOURCE_SHA256,
        camera_mapping,
        read_data_contract,
        sha,
    )

    os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
    output = args.output_dir.resolve()
    binding = json.loads(args.binding.read_text())
    prepared, contract = read_data_contract(args.data_contract, binding)
    if args.env_id and args.env_id != prepared["env_id"]:
        raise ValueError("--env-id differs from the dataset contract")
    info = json.loads((args.dataset_path / "meta/info.json").read_text())
    if info["total_frames"] != prepared["frames"] or info["fps"] != prepared["fps"]:
        raise ValueError("Runtime dataset census differs from the data contract")
    # Paths are relative to the dataset root so the input can move between hosts.
    dataset_files = {
        str(path.relative_to(args.dataset_path)).replace("\\", "/"): sha(path)
        for folder in ("data", "meta")
        for path in sorted((args.dataset_path / folder).rglob("*"))
        if path.is_file()
    }
    if not any(name.startswith("data/") for name in dataset_files):
        raise ValueError("No dataset shards found")
    expected_shards = {}
    for item in prepared["files"]:
        name = item["path"].replace("\\", "/")
        if "/data/" in name:
            name = "data/" + name.split("/data/", 1)[1]
        if not name.startswith("data/") or name in expected_shards:
            raise ValueError("Invalid prepared dataset shard path")
        expected_shards[name] = item["sha256"]
    if expected_shards != {
        name: value for name, value in dataset_files.items() if name.startswith("data/")
    }:
        raise ValueError("Training data differs from the normalization source")
    verify_base(args.base_model_path, args.base_manifest)
    identity = dict(
        baseline=BASELINE,
        task=prepared["task"],
        env_id=prepared["env_id"],
        code_sha256=source_identity(),
        entry_sha256=sha(__file__),
        dataset_sha256=digest(prepared),
        dataset_files_sha256=digest(dataset_files),
        binding_sha256=digest(binding),
        prompt_sha256=SOURCE_SHA256,
        base_checkpoint_sha256=sha(args.base_manifest),
        dexjoco_commit=DEXJOCO_COMMIT,
        config=dataclasses.asdict(contract),
        config_sha256=digest(dataclasses.asdict(contract)),
        seed=contract.seed,
        batch_size=32,
        sampler_version="fullbatch-update-addressed-v1",
        execution=dict(
            global_batch=32,
            accumulation_steps=args.accumulation_steps,
            microbatch=32 // args.accumulation_steps,
        ),
    )
    with FileLock(str(output) + ".lock", timeout=0):
        import jax
        import numpy as np
        from openpi.models.tokenizer import PaligemmaTokenizer

        from utils.pi05_utils.checkpoint import Checkpoints
        from utils.pi05_utils.data import (
            JointInputs,
            TrainingView,
            native_stats,
            native_transforms,
            open_view,
        )

        if args.resume:
            if not output.is_dir():
                raise ValueError("Resume output is missing")
        else:
            output.mkdir(exist_ok=False)
        checkpoints = Checkpoints(output / "checkpoints", identity, resume=args.resume)
        if not args.resume:
            for name, value in (
                ("identity.json", identity),
                ("data-contract.json", prepared),
                ("controller-binding.json", binding),
                ("dataset-files.json", dataset_files),
            ):
                with (output / name).open("x") as stream:
                    json.dump(value, stream, indent=2)
        config = training_config(
            name=prepared["task"],
            dual_arm=contract.dual_arm,
            data_factory=None,
            pretrained_params=str(args.base_model_path.resolve() / "params"),
            output_root=str(output),
            experiment=output.name,
        )
        inputs = JointInputs(
            tuple(prepared["state_names"]),
            tuple(prepared["action_names"]),
            tuple(prepared["controller_names"]),
            camera_mapping(prepared),
        )
        stats = native_stats(
            {
                key: {name: np.asarray(value) for name, value in values.items()}
                for key, values in prepared["stats"].items()
            }
        )
        view = TrainingView(
            open_view(args.dataset_path.resolve(), "local/" + prepared["task"], prepared["fps"]),
            prepared["env_id"],
            inputs,
            native_transforms(stats, contract.action_dim, PaligemmaTokenizer(250)),
        )
        if len(view) != prepared["frames"]:
            raise ValueError("Runtime dataset frame count changed")
        rng, init_rng = jax.random.split(jax.random.key(contract.seed))
        state = initialize(config, init_rng)
        jax.block_until_ready(state)
        inventory = parameter_inventory(state, config)
        if not args.resume:
            with (output / "parameter-inventory.json").open("x") as stream:
                json.dump(inventory, stream, indent=2)
        train(
            config,
            state,
            rng,
            view,
            checkpoints,
            output,
            resume=args.resume,
            accumulation_steps=args.accumulation_steps,
        )
