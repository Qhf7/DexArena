"""The shared dex tasks/instruction table (utils/dex_tasks.py).

Imports nothing but the table itself, so it runs in any environment -- the point is to keep
the one table every branch derives its language from honest, not to exercise a policy.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from utils.dex_tasks import (
    DEX_INSTRUCTIONS_VERSION,
    DEX_TASKS,
    DEX_TASKS_BY_ENV,
    assign_dataset_instructions,
    dataset_episode_instruction,
    dex_instruction,
    dex_instructions,
    draw_instruction,
    episode_instruction,
    instruction_table,
    is_dex_instruction,
    resolve_dex_task,
    sample_dex_instruction,
    validate_dex_instruction,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# The instruction each released dataset is labelled with. Rewording one means re-rendering
# that dataset and retraining its policy, so changing this table must be a deliberate edit
# here and not a drive-by edit of a string in the registry.
CANONICAL_INSTRUCTIONS = {
    "CabinetPouringOpenCabinet-Sharpa-v1": "open the cabinet door",
    "CabinetPouringPickBottle-Sharpa-v1": "take the bottle out of the cabinet and put it on the table",
    "CabinetPouringUnscrewCap-Sharpa-v1": "unscrew the bottle cap and put it down on the table",
    "CabinetPouringPourBeads-Sharpa-v1": "pour the contents of the bottle into the bowl",
    "OfficeDeskClickMouse-Sharpa-v1": "click the left mouse button",
    "OfficeDeskOpenLaptop-Sharpa-v1": "open the laptop lid",
    "OfficeDeskTypeKeys-Sharpa-v1": "type P A I R on the keyboard and press enter",
    "PackageOpeningOpenPackage-Sharpa-v1": "open the package",
    "NutBoltTurnNut-Sharpa-v1": "turn the nut on the bolt counterclockwise",
    "TableCleaningDumpDustpan-Sharpa-v1": "empty the dustpan into the trash bin and put it down",
    "MealPreparationOpenLunchBox-Sharpa-v1": "open the box lid",
    "MealPreparationPlaceBread-Sharpa-v1": "pick up the bread and put it on the plate",
    "NutBoltPickBolt-Sharpa-v1": "pick up the bolt and stand it upright on the table",
    "NutBoltSeatNut-Sharpa-v1": "pick up the nut and put it on the bolt plate",
    "FurnitureAssemblyLayFlatPanel-Sharpa-v1": "lay the panel flat on the table",
    "FurnitureAssemblyInsertRail-Sharpa-v1": "insert the rail into the groove",
    "PipetteTransferAspirateLiquid-Sharpa-v1": "aspirate liquid with the pipette and put it back",
    "JuicePreparationOpenFridge-Sharpa-v1": "open the refrigerator door",
    "JuicePreparationCloseFridge-Sharpa-v1": "close the refrigerator door",
    "JuicePreparationAppleToBlender-Sharpa-v1": "take the apple and put it into the blender",
    "JuicePreparationPressButton-Sharpa-v1": "press the blender button",
    "FurnitureAssemblyFastenScrew-Sharpa-v1": "turn the wing screw a quarter turn to fasten the rail",
    "JuicePreparationPlaceLid-Sharpa-v1": "put the lid on the blender",
    "PackageOpeningExtractApple-Sharpa-v1":
        "take the apple out of the package and put it on the table",
    "TableCleaningPickDebris-Sharpa-v1": "pick up the debris and put it in the dustpan",
    "MealPreparationScoopTomato-Sharpa-v1":
        "scoop the tomato with the spoon and put it on the plate",
    "PipetteTransferDispenseLiquid-Sharpa-v1": "dispense the liquid from the pipette into the well",
    "DrawerPlugOpenDrawer-Sharpa-v1": "open the middle drawer",
    "DrawerPlugPickCharger-Sharpa-v1": "take the charger out of the drawer and put it on the table",
    "DrawerPlugInsertPlug-Sharpa-v1": "plug the charger into the socket",
}

ENV_IDS = sorted(DEX_TASKS_BY_ENV)


def test_registry_covers_exactly_the_released_tasks() -> None:
    assert ENV_IDS == sorted(CANONICAL_INSTRUCTIONS)
    assert len(DEX_TASKS) == len(DEX_TASKS_BY_ENV), "duplicate env id in DEX_TASKS"


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_canonical_instruction_is_the_released_label(env_id: str) -> None:
    task = resolve_dex_task(env_id)
    assert task.instructions[0] == CANONICAL_INSTRUCTIONS[env_id]
    assert task.prompt == task.instructions[0]
    assert dex_instruction(env_id) == task.prompt


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_instruction_list_is_usable_as_a_paraphrase_set(env_id: str) -> None:
    instructions = dex_instructions(env_id)
    assert len(instructions) >= 2, "a one-entry list cannot vary the wording at evaluation"
    assert len(set(instructions)) == len(instructions)
    for text in instructions:
        assert text == text.strip() and text
        assert not text.endswith("."), f"{env_id}: instructions are written without a period"


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_task_module_exists_and_registers_the_env(env_id: str) -> None:
    task = resolve_dex_task(env_id)
    module = REPO_ROOT / task.task_module
    assert module.is_file(), f"{env_id}: {task.task_module} is missing"
    # The id is composed by the registration decorator, so check the registry, not the text.
    import tasks  # noqa: F401
    from mani_skill.utils.registration import REGISTERED_ENVS
    assert REGISTERED_ENVS[env_id].cls.__module__ == task.task_module[:-3].replace("/", "."), (
        f"{env_id} is not registered by {task.task_module}")


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_episode_instruction_is_reproducible_and_registered(env_id: str) -> None:
    instructions = set(dex_instructions(env_id))
    for seed in (0, 1, 10_001, 2**31 - 1):
        drawn = episode_instruction(env_id, seed)
        assert drawn in instructions
        assert drawn == episode_instruction(env_id, seed), "the draw must not depend on call order"


def test_episode_instruction_varies_across_a_seed_schedule() -> None:
    # The evaluation protocol's own schedule: 100 episodes from seed 10001.
    drawn = {episode_instruction("NutBoltTurnNut-Sharpa-v1", seed) for seed in range(10_001, 10_101)}
    assert drawn == set(dex_instructions("NutBoltTurnNut-Sharpa-v1"))


def test_draw_instruction_works_for_any_list() -> None:
    # Profiles outside the dex table (the Franka/YCB lineage) reuse the same draw.
    options = ("a", "b", "c")
    assert draw_instruction("Env-v1", 3, options) in options
    assert draw_instruction("Env-v1", 3, options) == draw_instruction("Env-v1", 3, options)
    assert draw_instruction("Env-v1", 3, ("only",)) == "only"
    assert {draw_instruction("Env-v1", seed, options) for seed in range(50)} == set(options)
    with pytest.raises(ValueError, match="No instructions"):
        draw_instruction("Env-v1", 3, ())


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_a_hundred_episode_dataset_carries_every_phrasing(env_id: str) -> None:
    """The released datasets are 100 episodes; each phrasing must get its fair share."""
    assignment = assign_dataset_instructions(env_id, range(100))
    counts = Counter(assignment.values())
    assert set(counts) == set(dex_instructions(env_id))
    assert set(counts.values()) == {100 // len(dex_instructions(env_id))}


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_a_render_shard_can_label_itself(env_id: str) -> None:
    """The renderer runs 10-episode shards in parallel processes and aggregates afterwards,
    so a shard's labels have to match the whole dataset's without any coordination."""
    whole = assign_dataset_instructions(env_id, range(100))
    for start in range(0, 100, 10):
        shard = assign_dataset_instructions(env_id, range(start, start + 10))
        assert all(whole[index] == text for index, text in shard.items())


# Fingerprint of the full recording-side assignment: every task, 100 episodes each.
# It is pinned because the whole point of this table is that two people who render the same
# task independently get the same labels. Any change here -- a reworded instruction, a new
# phrasing, a different assignment rule -- means datasets built before and after no longer
# agree, so it has to be a deliberate edit of this constant (and a DEX_INSTRUCTIONS_VERSION
# bump). Verified identical on CPython 3.9/macOS and 3.10/Linux, 2026-09-20.
ASSIGNMENT_SHA256 = "b6204ecb7f230721188eb6fbe17cb0fc91e5bc2dcfa97311c36be011a464b505"


def test_the_assignment_is_the_same_for_everyone() -> None:
    # Keyed by instruction_key, not env id: the pin survives env id renames because the
    # assignment is hashed from the key.
    payload = {
        resolve_dex_task(env_id).instruction_key: [
            dataset_episode_instruction(env_id, index) for index in range(100)
        ]
        for env_id in ENV_IDS
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    assert digest == ASSIGNMENT_SHA256


def test_episode_zero_is_not_the_canonical_everywhere() -> None:
    # A fixed round-robin would give every task's episode 0 the same list position, which
    # would make instructions[0] systematically over-represented in any subset of episode 0s.
    positions = {
        dex_instructions(env_id).index(dataset_episode_instruction(env_id, 0))
        for env_id in ENV_IDS
    }
    assert len(positions) > 1


def test_dataset_labels_and_evaluation_draws_are_separate() -> None:
    """Recording labels are balanced round-robin over the episode index; evaluation draws
    are a hash of the environment seed. Tying the two together would make the test-time
    wording a function of the training label, which is exactly what must not happen."""
    env_id = "NutBoltTurnNut-Sharpa-v1"
    period = len(dex_instructions(env_id))
    labels = [dataset_episode_instruction(env_id, index) for index in range(40)]
    assert labels == [labels[index % period] for index in range(40)], "labels are periodic"
    draws = [episode_instruction(env_id, seed) for seed in range(40)]
    assert draws != [draws[seed % period] for seed in range(40)], "draws are not periodic"


def test_seeded_sampling_is_reproducible() -> None:
    first = [sample_dex_instruction("DrawerPlugOpenDrawer-Sharpa-v1", seed=7) for _ in range(5)]
    assert len(set(first)) == 1, "the same seed must give the same instruction"
    assert first[0] in dex_instructions("DrawerPlugOpenDrawer-Sharpa-v1")


def test_validate_accepts_registered_and_rejects_everything_else() -> None:
    env_id = "JuicePreparationOpenFridge-Sharpa-v1"
    for text in dex_instructions(env_id):
        assert validate_dex_instruction(env_id, text) == text
        assert is_dex_instruction(env_id, text)
    assert not is_dex_instruction(env_id, env_id)
    with pytest.raises(ValueError, match="not a registered instruction"):
        validate_dex_instruction(env_id, "open the door")  # a plausible ad-hoc rewording
    with pytest.raises(ValueError, match="not a registered instruction"):
        validate_dex_instruction(env_id, dex_instruction("JuicePreparationCloseFridge-Sharpa-v1"))


def test_unknown_env_points_at_the_table() -> None:
    with pytest.raises(KeyError, match="utils/dex_tasks.py"):
        resolve_dex_task("NoSuchTask-v1")


def test_instruction_table_exports_as_json() -> None:
    table = instruction_table()
    assert table["version"] == DEX_INSTRUCTIONS_VERSION
    assert sorted(table["tasks"]) == ENV_IDS
    restored = json.loads(json.dumps(table, ensure_ascii=False))
    for env_id, entry in restored["tasks"].items():
        assert entry["instructions"] == list(dex_instructions(env_id))
        assert entry["canonical"] == entry["instructions"][0]


def test_instructions_are_not_shared_between_tasks() -> None:
    # Two tasks sharing a string would make them indistinguishable to a multi-task policy;
    # the fridge pair is the near miss this guards.
    seen: dict[str, str] = {}
    for task in DEX_TASKS:
        for text in task.instructions:
            assert text not in seen, f"{task.env_id} and {seen[text]} share {text!r}"
            seen[text] = task.env_id
