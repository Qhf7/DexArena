"""Declarative task-state serialisation for the furniture family.

Every other family hand-packs `torch.cat([...])` and hand-unpacks by positional
index. That is exactly the pattern that produced this family's defect: furniture
had no serialisation at all, and the families that did have it drifted (a field
written but never read back, a field written only under one build flag). Adding
a fourth and fifth hand-rolled packer here would repeat the mistake.

So the fields are DECLARED once per environment, and pack/unpack are generated
from that declaration. Adding a state tensor without adding its declaration is
still possible -- nothing can stop that -- but it is caught by
`check_state_restore_completeness.py`, and there is no index arithmetic left to
get wrong.

Width is the version: an older recording is narrower, and its missing trailing
fields keep the values `_initialize_episode` produced. Same contract the other
families use, so old data keeps restoring.
"""
from __future__ import annotations

import torch
from mani_skill.utils import common


class TaskStateField:
    """One serialised task attribute: name, element count, restore dtype."""

    __slots__ = ("name", "width", "dtype")

    def __init__(self, name: str, width: int = 1, dtype: torch.dtype = torch.float32):
        self.name = name
        self.width = int(width)
        self.dtype = dtype


class DeclarativeTaskStateMixin:
    """Generates get_state_dict/set_state_dict from ``TASK_STATE_FIELDS``.

    A subclass sets ``TASK_STATE_KEY`` and ``TASK_STATE_FIELDS``. Fields must
    only ever be APPENDED, never reordered or removed, or older recordings would
    silently restore into the wrong attributes.
    """

    TASK_STATE_KEY: str = ""
    TASK_STATE_FIELDS: tuple[TaskStateField, ...] = ()

    def _task_state_tensor(self) -> torch.Tensor:
        columns = []
        for field in self.TASK_STATE_FIELDS:
            value = getattr(self, field.name)
            value = value.reshape(self.num_envs, -1).to(torch.float32)
            if value.shape[-1] != field.width:
                raise RuntimeError(
                    f"{self.TASK_STATE_KEY}: field {field.name!r} declares width "
                    f"{field.width} but holds {value.shape[-1]}"
                )
            columns.append(value)
        return torch.cat(columns, dim=-1)

    def get_state_dict(self):
        state = super().get_state_dict()
        if self.TASK_STATE_KEY and self.TASK_STATE_FIELDS:
            state[self.TASK_STATE_KEY] = self._task_state_tensor()
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        state = dict(state)
        task_state = state.pop(self.TASK_STATE_KEY, None) if self.TASK_STATE_KEY else None
        if env_idx is None:
            restore_idx = torch.arange(self.num_envs, device=self.device)
            super().set_state_dict(state)
        else:
            restore_idx = torch.as_tensor(
                env_idx, dtype=torch.long, device=self.device
            ).flatten()
            super().set_state_dict(state, restore_idx)
        if task_state is None:
            return
        if isinstance(task_state, dict):
            # A flattened LeRobot environment state restores a top-level tensor
            # as {"value": tensor}; raw_states.pkl keeps the tensor directly.
            task_state = task_state.get("value")
        if task_state is None:
            return
        task_state = common.to_tensor(task_state, device=self.device).reshape(
            len(restore_idx), -1
        )
        full = sum(f.width for f in self.TASK_STATE_FIELDS)
        if task_state.shape[-1] > full:
            raise ValueError(
                f"{self.TASK_STATE_KEY} has {task_state.shape[-1]} fields, "
                f"this build declares {full}"
            )
        offset = 0
        for field in self.TASK_STATE_FIELDS:
            end = offset + field.width
            if end > task_state.shape[-1]:
                # Recorded before this field existed: leave the reset value.
                break
            chunk = task_state[:, offset:end]
            target = getattr(self, field.name)
            if field.dtype == torch.bool:
                value = chunk.reshape(target[restore_idx].shape).bool()
            else:
                value = chunk.reshape(target[restore_idx].shape).to(field.dtype)
            target[restore_idx] = value
            offset = end
