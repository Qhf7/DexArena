"""OpenPI PI0.5 model, masked flow loss and absolute-joint policy inference.

The flow loss is adapted from the pinned DexJoCo/OpenPI source; see
utils/pi05_utils/NOTICE and LICENSE.openpi for attribution."""

from dataclasses import dataclass

import numpy as np

from train.train_configs.pi05_config import TrainingContract
from utils.pi05_utils.checkpoint import inspect_final, verify_restored
from utils.pi05_utils.data import JointInputs, JointOutputs, native_stats


def expand_projection(path, pretrained, initialized):
    if type(initialized).__name__ == "ShapeDtypeStruct":
        raise ValueError("Loader requires actual initialized arrays, not eval_shape placeholders")
    old = np.asarray(pretrained)
    # Same-shape leaves are completely replaced. Inspect metadata only: copying
    # initialized GPU values to host here is unnecessary and can stall loading.
    if old.shape == tuple(initialized.shape):
        # Reuse the restored CPU array when dtype already agrees. The loader
        # never mutates it; avoid another multi-GB host allocation/copy.
        return old.astype(initialized.dtype, copy=False)
    new = np.array(initialized, copy=True)
    axes = {"action_in_proj/kernel": 0, "action_out_proj/kernel": 1, "action_out_proj/bias": 0}
    if path not in axes:
        raise ValueError(f"Unapproved pretrained shape mismatch: {path}: {old.shape}->{new.shape}")
    axis = axes[path]
    if (
        old.ndim != new.ndim
        or old.ndim <= axis
        or old.shape[axis] != 32
        or new.shape[axis] != 58
        or any(a != b for i, (a, b) in enumerate(zip(old.shape, new.shape)) if i != axis)
    ):
        raise ValueError(f"Expected only32->58 action capacity expansion: {path}")
    slices = [slice(None)] * new.ndim
    slices[axis] = slice(0, 32)
    new[tuple(slices)] = old
    return new


@dataclass(frozen=True)
class NativeWeightLoader:
    params_path: str

    def load(self, params):
        # Lazy imports: CPU contract tests do not need to install model libraries.
        import flax.traverse_util
        from openpi.models import model
        from openpi.shared import download

        source = model.restore_params(
            download.maybe_download(self.params_path), restore_type=np.ndarray
        )
        reference = flax.traverse_util.flatten_dict(params, sep="/")
        pretrained = flax.traverse_util.flatten_dict(source, sep="/")
        result = {}
        for path, initial in reference.items():
            if path in pretrained:
                result[path] = expand_projection(path, pretrained[path], initial)
            elif "lora" in path:
                result[path] = initial
            else:
                raise ValueError(f"Missing non-LoRA pretrained leaf: {path}")
        extra = set(pretrained) - set(reference)
        if extra:
            raise ValueError(f"Unexpected pretrained leaves: {sorted(extra)[:10]}")
        return flax.traverse_util.unflatten_dict(result, sep="/")


def model_config(dual_arm=False):
    from openpi.models.pi0_config import Pi0Config

    c = TrainingContract(dual_arm=dual_arm)
    c.validate()
    return Pi0Config(
        pi05=True,
        action_dim=c.action_dim,
        action_horizon=c.action_horizon,
        max_token_len=c.max_token_len,
        paligemma_variant=c.paligemma_variant,
        action_expert_variant=c.action_expert_variant,
    )


def masked_flow_loss(model, rng, observation, actions, mask, *, train=True):
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_api
    from openpi.models.pi0 import make_attn_mask

    if actions.shape != mask.shape:
        raise ValueError("Mask must match batch x horizon x model capacity")
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = model_api.preprocess_observation(preprocess_rng, observation, train=train)
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
    u_t = noise - actions
    prefix, prefix_mask, prefix_ar = model.embed_prefix(observation)
    suffix, suffix_mask, suffix_ar, cond = model.embed_suffix(observation, x_t, time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    attention = make_attn_mask(input_mask, jnp.concatenate([prefix_ar, suffix_ar], axis=0))
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm(
        [prefix, suffix], mask=attention, positions=positions, adarms_cond=[None, cond]
    )
    velocity = model.action_out_proj(suffix_out[:, -model.action_horizon :])
    error = jnp.square(velocity - u_t)
    # Return one equally weighted loss per sampled training state. All-false
    # samples are rejected by the host-side dataset validator, not silently used.
    count = jnp.sum(mask, axis=(-2, -1))
    return jnp.sum(jnp.where(mask, error, 0), axis=(-2, -1)) / count


def inference_transforms(stats, capacity, tokenizer):
    from openpi import transforms

    # Observation-only input: strict training normalization also expects actions;
    # inference must require state stats without inventing a target action.
    inputs = transforms.compose(
        [
            transforms.Normalize({"state": stats["state"]}, use_quantiles=True, strict=True),
            transforms.ResizeImages(224, 224),
            transforms.TokenizePrompt(tokenizer, discrete_state_input=True),
            transforms.PadStatesAndActions(capacity),
        ]
    )
    outputs = transforms.Unnormalize({"actions": stats["actions"]}, use_quantiles=True)
    return inputs, outputs


class NativeJointInference:
    def __init__(self, model, prepared, *, live_state_names, camera_keys, denoising_steps=10):
        import jax
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.shared import nnx_utils

        self.jax = jax
        self.n = len(prepared["controller_names"])
        self.capacity = 32 if self.n == 29 else 58
        if self.n not in (29, 58) or tuple(live_state_names) != tuple(prepared["state_names"]):
            raise ValueError("Inference state order differs from verified robot/data binding")
        if model.action_dim != self.capacity or model.action_horizon != 30 or denoising_steps != 10:
            raise ValueError(
                "Explicit native model inference dimensions/sampling contract mismatch"
            )
        self.inputs = JointInputs(
            tuple(live_state_names),
            tuple(prepared["action_names"]),
            tuple(prepared["controller_names"]),
            camera_keys,
        )
        stats = native_stats(
            {k: {n: np.asarray(v) for n, v in d.items()} for k, d in prepared["stats"].items()}
        )
        self.pre, self.post = inference_transforms(stats, self.capacity, PaligemmaTokenizer(250))
        self.trim = JointOutputs(self.n)
        model.eval()
        self.sample = nnx_utils.module_jit(model.sample_actions)
        self.denoising_steps = denoising_steps

    def __call__(self, observation, prompt, seed, request):
        import jax.numpy as jnp
        from openpi.models.model import Observation

        data = self.pre(self.inputs(dict(observation, prompt=prompt)))
        batch = self.jax.tree.map(lambda x: jnp.asarray(x)[None, ...], data)
        rng = self.jax.random.key(seed)
        rng = self.jax.random.fold_in(rng, request.generation)
        rng = self.jax.random.fold_in(rng, request.serial)
        actions = self.sample(rng, Observation.from_dict(batch), num_steps=self.denoising_steps)
        physical = self.post({"actions": np.asarray(actions[0])})
        return self.trim(physical)["actions"]


def restore_host_payload(directory):
    import jax
    import orbax.checkpoint as ocp

    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(str(directory))
        restore_args = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), metadata)
        return checkpointer.restore(
            str(directory), args=ocp.args.PyTreeRestore(restore_args=restore_args)
        )


def expected_model_tree(dual_arm):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from openpi.shared import nnx_utils

    config = model_config(dual_arm)
    abstract = nnx.eval_shape(config.create, jax.random.key(0))
    _, state = nnx.split(abstract)
    # Match training's frozen BF16 leaves without materializing random weights.
    state = nnx_utils.state_map(
        state,
        config.get_freeze_filter(),
        lambda p: p.replace(jax.ShapeDtypeStruct(p.value.shape, jnp.bfloat16)),
    )
    return config, state.to_pure_dict()


def load_final_model(output, expected_identity, expected_data, *, device):
    import jax

    admission = inspect_final(output, expected_identity, expected_data)
    config, expected = expected_model_tree(expected_identity["config"]["dual_arm"])
    payload = restore_host_payload(admission["directory"])
    verify_restored(payload, expected, admission["optimizer_updates"])
    # Read ALL optimizer arrays as well: a model-only partial checkpoint must not
    # masquerade as the complete resumable training artifact we promised.
    optimizer_leaves = jax.tree.leaves(payload["opt_state"])
    if not optimizer_leaves or not all(np.isfinite(np.asarray(x)).all() for x in optimizer_leaves):
        raise ValueError("Missing or nonfinite restored optimizer state")
    params = jax.device_put(payload["params"], device)
    jax.block_until_ready(params)
    del payload
    # Never silently intersect away unexpected parameters; exact tree checked above.
    model = config.load(params, remove_extra_params=False)
    model.eval()
    return model, dict(
        identity_sha256=admission["identity_sha256"],
        optimizer_updates=admission["optimizer_updates"],
        status="FINAL_FULL_RESTORE_AND_MODEL_BINDING_VERIFIED",
    )


def load_compact_model(package, base, *, device):
    """Load a published inference-only package, not a resumable optimizer state."""
    import jax

    from utils.pi05_utils.compact import load_compact_params, read_compact_config

    metadata = read_compact_config(package)
    config, expected = expected_model_tree(metadata["training"]["dual_arm"])
    params = load_compact_params(package, base, expected)
    params = jax.device_put(params, device)
    jax.block_until_ready(params)
    model = config.load(params, remove_extra_params=False)
    model.eval()
    return model, metadata["data_contract"], dict(
        status="COMPACT_BASE_OVERLAY_AND_MODEL_BINDING_VERIFIED",
        identity_sha256=metadata["identity_sha256"],
        optimizer_updates=metadata["optimizer_updates"],
        trainable_sha256=metadata["trainable_sha256"],
        inference_only=True,
    )
