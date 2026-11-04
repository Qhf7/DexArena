"""Restoring a single-env recording into a multi-env scene."""

from __future__ import annotations


def drop_per_scene_entries(env, state: dict) -> dict:
    """Keep only the merged actors/articulations of ``state``.

    Scenes built per sub-scene and merged register both the merged ``package``
    and the per-scene ``package_<i>``; a single-env recording carries both.
    With N envs a per-scene entry lives in one sub-scene and cannot take an
    N-row state ("[64, 7] vs [1, 7]"). The merged entry holds the same data.
    """
    if env.num_envs == 1:
        return state
    state = dict(state)
    registry = env.scene.state_dict_registry
    for category in ("actors", "articulations"):
        items = state.get(category)
        if not isinstance(items, dict):
            continue
        known = getattr(registry, category)
        state[category] = {
            name: value
            for name, value in items.items()
            if name not in known or len(known[name]._scene_idxs) == env.num_envs
        }
    return state


class MergedStateDictMixin:
    def set_state_dict(self, state: dict, env_idx=None):
        return super().set_state_dict(drop_per_scene_entries(self, state), env_idx)
