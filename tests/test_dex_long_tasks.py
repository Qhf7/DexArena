"""The long-task registry (utils/dex_long_tasks.py) and the action routing it defines.

Every stage of a long task is a registered subtask, every long task has a distinct env id and
family name, and the arm that acts in a stage is one the dual-arm layout can route to. The
routing itself is pure numpy and is checked without a simulator.
"""
from __future__ import annotations

import numpy as np
import pytest

from utils import dex_long_tasks as L
from utils.dex_tasks import DEX_TASKS, resolve_dex_task


def test_every_stage_is_a_registered_subtask_and_every_subtask_is_a_stage() -> None:
    stages = [s.subtask for t in L.LONG_TASKS for s in t.stages]
    assert len(stages) == len(set(stages)) == 30
    assert set(stages) == {t.env_id for t in DEX_TASKS}


def test_long_tasks_are_distinct_and_resolvable_by_env_id_or_family() -> None:
    assert len(L.LONG_TASKS) == 10
    assert len({t.env_id for t in L.LONG_TASKS}) == 10
    assert len({t.family for t in L.LONG_TASKS}) == 10
    for t in L.LONG_TASKS:
        assert L.resolve_long_task(t.env_id) is t
        assert L.resolve_long_task(t.family) is t
    with pytest.raises(KeyError):
        L.resolve_long_task("not-a-long-task")


def test_stage_arms_and_action_dims_match_the_robot() -> None:
    for t in L.LONG_TASKS:
        assert t.action_dim == (L.DUAL_ACTION_DIM if t.is_dual else L.SINGLE_ACTION_DIM)
        for s in t.stages:
            assert s.arm in (L.LEFT, L.RIGHT, L.DUAL)
            if not t.is_dual:
                assert s.arm == L.RIGHT, "a single-arm long task cannot route to another arm"
            if s.arm == L.DUAL:
                assert resolve_dex_task(s.subtask).action_dim == L.DUAL_ACTION_DIM
            else:
                assert resolve_dex_task(s.subtask).action_dim == L.SINGLE_ACTION_DIM


def test_family_name_is_the_prefix_the_subtask_datasets_use() -> None:
    # merge_family_dataset.py finds a family's subtasks by the first token of their short names.
    for t in L.LONG_TASKS:
        for s in t.stages:
            assert resolve_dex_task(s.subtask).short_name.split("_")[0] == t.family


def test_route_action_places_a_single_arm_action_into_its_slots_and_holds_the_rest() -> None:
    held = np.arange(L.DUAL_ACTION_DIM, dtype=np.float32)
    action = -np.ones(L.SINGLE_ACTION_DIM, dtype=np.float32)
    for arm in (L.LEFT, L.RIGHT):
        out = L.route_action(action, arm, held, dual=True)
        slots = L.DUAL_SLOTS[arm]
        assert out.shape == (L.DUAL_ACTION_DIM,)
        assert np.all(out[slots["arm"][0]:slots["arm"][1]] == -1)
        assert np.all(out[slots["hand"][0]:slots["hand"][1]] == -1)
        untouched = np.ones(L.DUAL_ACTION_DIM, dtype=bool)
        untouched[slots["arm"][0]:slots["arm"][1]] = False
        untouched[slots["hand"][0]:slots["hand"][1]] = False
        assert np.all(out[untouched] == held[untouched])
    # a dual stage passes the 58-dim action through; a single-arm robot needs no routing
    full = np.full(L.DUAL_ACTION_DIM, 2.0, dtype=np.float32)
    assert np.all(L.route_action(full, L.DUAL, held, dual=True) == full)
    assert np.all(L.route_action(action, L.RIGHT, held[:29], dual=False) == action)
    with pytest.raises(ValueError):
        L.route_action(action, L.DUAL, held, dual=True)
    with pytest.raises(ValueError):
        L.route_action(full, L.RIGHT, held, dual=True)


def test_stage_reached_reads_both_gate_kinds_and_refuses_a_missing_key() -> None:
    flag = L.LongStage("CabinetPouringOpenCabinet-Sharpa-v1", L.LEFT, "stage_door_open")
    assert L.stage_reached({"stage_door_open": np.array([True])}, flag)
    assert not L.stage_reached({"stage_door_open": np.array([False])}, flag)
    counter = L.LongStage("JuicePreparationOpenFridge-Sharpa-v1", L.LEFT, "stage", 2, "stage")
    assert L.stage_reached({"stage": np.array([2])}, counter)
    assert L.stage_reached({"stage": np.array([3])}, counter)
    assert not L.stage_reached({"stage": np.array([1])}, counter)
    with pytest.raises(KeyError):
        L.stage_reached({}, counter)
