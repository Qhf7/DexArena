"""Scoped patches and GPU-reset helpers shared by the task adapters."""

from __future__ import annotations

import contextlib

import torch

_MISSING = object()


@contextlib.contextmanager
def patched(target, name: str, value):
    """Replace attribute ``name`` of ``target`` (class, module or instance) for the block."""
    old = target.__dict__.get(name, _MISSING) if hasattr(target, "__dict__") else _MISSING
    setattr(target, name, value)
    try:
        yield
    finally:
        if old is _MISSING:
            delattr(target, name)
        else:
            setattr(target, name, old)


@contextlib.contextmanager
def reset_mask(scene, mask: torch.Tensor):
    """Run the block with ``scene._reset_mask`` replaced (the GPU setters index by it)."""
    prev = scene._reset_mask
    scene._reset_mask = mask
    try:
        yield
    finally:
        scene._reset_mask = prev


def full_reset_mask(scene):
    return reset_mask(scene, torch.ones_like(scene._reset_mask))


def rows_reset_mask(scene, env_idx):
    mask = torch.zeros_like(scene._reset_mask)
    mask[torch.as_tensor(env_idx, device=mask.device)] = True
    return reset_mask(scene, mask)


def partial_gpu_reset(env) -> bool:
    """True while a GPU reset covers only some of the sub-envs."""
    if not env.gpu_sim_enabled:
        return False
    mask = env.scene._reset_mask
    return int(mask.sum()) < env.num_envs


@contextlib.contextmanager
def slice_full_writes(env, *structs, methods=("set_qpos", "set_qvel")):
    """Make ``set_qpos``/``set_qvel`` on ``structs`` accept full ``(num_envs, ...)``
    tensors during a partial GPU reset by slicing them to the reset rows.

    The GPU setters write exactly the rows of ``scene._reset_mask``; code written
    for the CPU path builds full-size tensors and writes them whole.
    """
    if not partial_gpu_reset(env):
        yield
        return
    mask = env.scene._reset_mask
    stack = contextlib.ExitStack()

    def make(struct, method):
        orig = getattr(struct, method)

        def call(value):
            value = torch.as_tensor(value)
            if value.ndim >= 1 and value.shape[0] == env.num_envs:
                value = value[mask]
            return orig(value)

        return call

    with stack:
        for struct in structs:
            if struct is None:
                continue
            for method in methods:
                stack.enter_context(patched(struct, method, make(struct, method)))
        yield


def scatter_rows(env, cache, env_idx, value: torch.Tensor) -> torch.Tensor:
    """Write ``value`` (one row per env in ``env_idx``) into a full-size copy of ``cache``."""
    if cache is None or cache.shape[0] != env.num_envs:
        cache = value[:1].expand(env.num_envs, *value.shape[1:]).clone()
    else:
        cache = cache.clone()
    rows = cache[env_idx]
    cache[env_idx] = value.reshape(rows.shape).to(cache.dtype)
    return cache
