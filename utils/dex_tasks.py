"""Registry of the dexterous-hand (Sharpa) tasks the VLA baselines train on.

One table shared by every branch, so an env id maps to one instruction set, one robot and
one proprio layout everywhere: the dataset renderer, the GR00T modality configs and the
PI0.5 task profiles all derive from it instead of spelling the strings out again.

Language instructions
---------------------
Each task carries a **list** of instructions, all describing the same success condition in
different words. The protocol around that list:

  * ``instructions[0]`` is the canonical one. It is the label the first released datasets
    carry, so it must not be reworded without relabelling and retraining the task.
  * Every recorded EPISODE is assigned one instruction from the list, so a 100-episode
    dataset carries all of them (``dataset_episode_instruction``). Training then conditions
    on a different phrasing from episode to episode instead of on one fixed string.
  * Evaluation draws an instruction from the same list per episode, so a policy is scored on
    following the task rather than on matching one string. ``episode_instruction`` makes
    that draw reproducible from the episode's environment seed.
  * Both rules hash the task's ``instruction_key``, not its env id. The key is fixed when
    the task's list is published and does not follow env id renames, so a rename leaves
    every dataset label and every evaluation draw where it was.

The published demonstration datasets carry no language field of their own, so this table is
the only place the instructions exist: read them from here rather than writing new ones, or
two runs of the same task stop being comparable.

Embodiment
----------
Layout shared by every entry: ``pd_joint_pos`` on a Panda arm + 22-DoF Sharpa hand, so the
env action is 29 absolute joint targets in controller order (arm 7, then hand 22) and the
VLA state is the same 29 joints read back from the simulator in that order.

The teleop demonstrations were recorded in ``obs_mode="state"`` (no images) and live in
LeRobot v3.0 form under ``DEX_DEMO_ROOT``; the renderer re-opens the env in ``obs_mode="rgb"``
and restores each recorded state to render the camera views. Since the 2026-09-11 data release
(100 replay-successful episodes per task) the views are the published per-family rig
(utils/dex_camera_rigs.py via utils/dex_cameras.py), not the task's historical camera.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Sequence


DEX_DEMO_ROOT = "/mnt/ceph2/DexArena/data"

SHARPA_ARM_DOF = 7
SHARPA_HAND_DOF = 22
SHARPA_ACTION_DIM = SHARPA_ARM_DOF + SHARPA_HAND_DOF  # 29

# Bumped whenever an instruction list changes, so a run can record which table it used.
# 1: 2026-09-18, the first list-valued table (canonical strings unchanged from the single
#    prompts the 2026-09-11/15/16/17 datasets were rendered with).
DEX_INSTRUCTIONS_VERSION = 1


@dataclass(frozen=True)
class DexTask:
    env_id: str
    # Interchangeable phrasings of the same goal; [0] is canonical (see the module docstring).
    instructions: tuple[str, ...]
    robot_uids: str
    # Task module under tasks/ that registers the env (joins evaluation fingerprints).
    task_module: str
    # Env horizon the task registers with; evaluation defaults to it.
    max_episode_steps: int
    # Key the per-episode label cycle and the evaluation draw are hashed with. Fixed when
    # the task's instruction list was published; it stays when the env id is renamed, so
    # datasets labelled under an earlier id keep their labels and a seed keeps its draw.
    instruction_key: str
    control_mode: str = "pd_joint_pos"
    arm_dof: int = SHARPA_ARM_DOF
    hand_dof: int = SHARPA_HAND_DOF
    # Joint groups in controller order, (name, dof). None = the single-arm default
    # (single_arm arm_dof, hand hand_dof). A dual-arm robot lists both arms then both hands,
    # which is the order its pd_joint_pos controller concatenates (left_arm, right_arm,
    # left_hand, right_hand). Group names ending in "arm" get RELATIVE actions in GR00T,
    # "hand" groups ABSOLUTE.
    layout: tuple[tuple[str, int], ...] | None = None

    def __post_init__(self) -> None:
        if not self.instructions:
            raise ValueError(f"{self.env_id}: needs at least one instruction.")
        for text in self.instructions:
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{self.env_id}: instructions must be non-empty strings.")
        if len(set(self.instructions)) != len(self.instructions):
            raise ValueError(f"{self.env_id}: instructions must be unique.")
        if not isinstance(self.instruction_key, str) or not self.instruction_key.strip():
            raise ValueError(f"{self.env_id}: instruction_key must be a non-empty string.")

    @property
    def prompt(self) -> str:
        """The canonical instruction: what the released datasets are labelled with."""
        return self.instructions[0]

    def instruction(self, index: int = 0) -> str:
        try:
            return self.instructions[index]
        except IndexError:
            raise IndexError(
                f"{self.env_id} has {len(self.instructions)} instructions "
                f"(0..{len(self.instructions) - 1}); got index {index}."
            ) from None

    def sample_instruction(self, rng: random.Random | None = None) -> str:
        """Draw one instruction uniformly; pass a seeded ``Random`` to make it reproducible."""
        return (rng or random).choice(self.instructions)

    @property
    def groups(self) -> tuple[tuple[str, int], ...]:
        return self.layout or (("single_arm", self.arm_dof), ("hand", self.hand_dof))

    @property
    def group_slices(self) -> tuple[tuple[str, int, int], ...]:
        out, start = [], 0
        for name, dof in self.groups:
            out.append((name, start, start + dof)); start += dof
        return tuple(out)

    @property
    def is_dual_arm(self) -> bool:
        return sum(1 for name, _ in self.groups if name.endswith("arm")) > 1

    @property
    def action_dim(self) -> int:
        return sum(dof for _, dof in self.groups)

    @property
    def state_dim(self) -> int:
        return sum(dof for _, dof in self.groups)

    @property
    def demo_dir_name(self) -> str:
        return self.env_id

    @property
    def hand_dir(self) -> str:
        """The directory level a dataset sits under: the env id's hand token in lower case, the
        rule `data_gen/teleop/dataset_layout.py` applies to every exported dataset."""
        return self.env_id.split("-")[-2].lower()

    @property
    def demo_root(self) -> str:
        """The demo dataset: ``<DEX_DEMO_ROOT>/<hand>/<env id>``."""
        return f"{DEX_DEMO_ROOT}/{self.hand_dir}/{self.env_id}"

    @property
    def family(self) -> str:
        """Family package name: ``tasks/cabinet_pouring/...`` -> ``cabinet_pouring``."""
        return self.task_module.split("/")[1]

    @property
    def short_name(self) -> str:
        """Filesystem-friendly stem: the task module's file name without ``.py``."""
        return self.task_module.rsplit("/", 1)[-1].removesuffix(".py")


_RIGHT = "panda_sharpa_hand_right_angled_mobile_base"
_LEFT = "panda_sharpa_hand_left_angled_mobile_base"
_DUAL = "panda_dual_sharpa_hand_angled_mobile_base"
DUAL_SHARPA_LAYOUT = (
    ("left_arm", SHARPA_ARM_DOF), ("right_arm", SHARPA_ARM_DOF),
    ("left_hand", SHARPA_HAND_DOF), ("right_hand", SHARPA_HAND_DOF),
)  # 58, verified against the dual controller's joint order on 2026-09-14

DEX_TASKS: tuple[DexTask, ...] = (
    DexTask(
        env_id="CabinetPouringOpenCabinet-Sharpa-v1",
        instruction_key="CabinetPourDoorOpen-v1",
        instructions=(
            "open the cabinet door",
            "swing the cabinet door open",
            "pull the cabinet door open",
            "open the door of the cabinet",
        ),
        robot_uids=_LEFT,
        task_module="tasks/cabinet_pouring/cabinet_pouring_open_cabinet.py",
        max_episode_steps=600,
    ),
    DexTask(
        env_id="CabinetPouringPickBottle-Sharpa-v1",
        instruction_key="CabinetPourBottlePickPlace-v1",
        instructions=(
            "take the bottle out of the cabinet and put it on the table",
            "remove the bottle from the cabinet and stand it on the table",
            "pick the bottle off the cabinet shelf and set it upright on the table",
            "get the bottle out of the cabinet and place it on the tabletop",
        ),
        robot_uids=_RIGHT,
        task_module="tasks/cabinet_pouring/cabinet_pouring_pick_bottle.py",
        max_episode_steps=900,
    ),
    DexTask(
        env_id="CabinetPouringUnscrewCap-Sharpa-v1",
        instruction_key="CabinetPourCapRemovePlace-v1",
        instructions=(
            "unscrew the bottle cap and put it down on the table",
            "twist the cap off the bottle and set it on the table",
            "remove the bottle cap and place it on the tabletop",
            "take the cap off the bottle and put it down on the table",
        ),
        robot_uids=_RIGHT,
        task_module="tasks/cabinet_pouring/cabinet_pouring_unscrew_cap.py",
        max_episode_steps=1400,
    ),
    DexTask(
        env_id="CabinetPouringPourBeads-Sharpa-v1",
        instruction_key="CabinetPourPour-v1",
        instructions=(
            "pour the contents of the bottle into the bowl",
            "tip the bottle up and pour it into the bowl",
            "empty the bottle into the bowl",
            "pour the liquid from the bottle into the bowl",
        ),
        robot_uids=_RIGHT,
        task_module="tasks/cabinet_pouring/cabinet_pouring_pour_beads.py",
        max_episode_steps=700,
    ),
    DexTask(
        env_id="OfficeDeskClickMouse-Sharpa-v1",
        instruction_key="SharpaAngledClickMouse-v1",
        instructions=(
            "click the left mouse button",
            "press the left button on the mouse",
            "left-click the mouse",
            "click the mouse with its left button",
        ),
        robot_uids=_RIGHT,
        task_module="tasks/office_desk/office_desk_click_mouse.py",
        max_episode_steps=500,
    ),
    DexTask(
        env_id="OfficeDeskOpenLaptop-Sharpa-v1",
        instruction_key="SharpaAngledOpenLaptop-v1",
        instructions=(
            "open the laptop lid",
            "lift the laptop screen open",
            "open the laptop",
            "raise the lid of the laptop",
        ),
        robot_uids=_RIGHT,
        task_module="tasks/office_desk/office_desk_open_laptop.py",
        max_episode_steps=600,
    ),
    DexTask(
        env_id="OfficeDeskTypeKeys-Sharpa-v1",
        instruction_key="SharpaAngledTypeKeyboardAllKeys-v1",
        instructions=(
            "type P A I R on the keyboard and press enter",
            "type the letters P A I R and then hit enter",
            "press P, A, I, R in order on the keyboard, then press enter",
            "spell out P A I R on the keyboard and finish with the enter key",
        ),
        robot_uids=_LEFT,
        task_module="tasks/office_desk/office_desk_type_keys.py",
        max_episode_steps=700,
    ),
    # Dual-arm package opening (2026-09-11 release, 100 episodes, action 58). GR00T N1.7's
    # action head takes up to 132 dims, so only the modality layout changes; PI0.5 (32-dim
    # native) needs the DexJoCo-style action-projection expansion first and skips this task.
    DexTask(
        env_id="PackageOpeningOpenPackage-Sharpa-v1",
        instruction_key="SharpaPackageOpeningOpenPackage-v1",
        instructions=(
            "open the package",
            "open the package with both hands",
            "fold both lids of the package open",
            "open the box flaps with both hands",
        ),
        robot_uids=_DUAL,
        task_module="tasks/package_opening/package_opening_open_package.py",
        max_episode_steps=500,
        layout=DUAL_SHARPA_LAYOUT,
    ),
    # Remaining single-arm tasks of the 2026-09-11 release.
    DexTask(
        env_id="NutBoltTurnNut-Sharpa-v1",
        instruction_key="ScrewingTurn-v1",
        instructions=(
            "turn the nut on the bolt counterclockwise",
            "unscrew the nut counterclockwise on the bolt",
            "rotate the nut counterclockwise on the bolt",
            "twist the nut counterclockwise",
        ),
        robot_uids=_RIGHT,
        task_module="tasks/nut_bolt/nut_bolt_turn_nut.py",
        max_episode_steps=900,
    ),
    DexTask(
        env_id="TableCleaningDumpDustpan-Sharpa-v1",
        instruction_key="TableCleanDumpDustpan-v1",
        instructions=(
            "empty the dustpan into the trash bin and put it down",
            "pick up the dustpan, tip the debris into the trash bin, and set it back down",
            "dump what is in the dustpan into the bin and put the dustpan away",
            "tip the dustpan out over the trash bin and put the dustpan down",
        ),
        robot_uids=_RIGHT,
        task_module="tasks/table_cleaning/table_cleaning_dump_dustpan.py",
        max_episode_steps=600,
    ),
    DexTask(
        env_id="MealPreparationOpenLunchBox-Sharpa-v1",
        instruction_key="AngledMobileBaseSharpaOpenCloseBox-v1",
        instructions=(
            "open the box lid",
            "lift the lid of the box open",
            "open the hinged box",
            "swing the box lid open",
        ),
        robot_uids=_LEFT,
        task_module="tasks/meal_preparation/meal_preparation_open_lunch_box.py",
        max_episode_steps=300,
    ),
    # ---- 2026-09-15 release: ten more single-arm subtasks (100 episodes each) ----
    DexTask(
        env_id="MealPreparationPlaceBread-Sharpa-v1",
        instruction_key="AngledMobileBaseSharpaPickAndPlaceBread-v1",
        instructions=(
            "pick up the bread and put it on the plate",
            "place the bread on the plate",
            "grab the bread and set it down on the plate",
            "move the bread onto the plate",
        ),
        robot_uids=_RIGHT,
        task_module="tasks/meal_preparation/meal_preparation_place_bread.py",
        max_episode_steps=400,
    ),
    DexTask(
        env_id="NutBoltPickBolt-Sharpa-v1",
        instruction_key="ScrewingBoltPickPlace-v1",
        instructions=(
            "pick up the bolt and stand it upright on the table",
            "place the bolt upright in its spot on the table",
            "grab the bolt and set it standing on the table",
            "stand the bolt up on the table",
        ),
        robot_uids=_LEFT,
        task_module="tasks/nut_bolt/nut_bolt_pick_bolt.py",
        max_episode_steps=700,
    ),
    DexTask(
        env_id="NutBoltSeatNut-Sharpa-v1",
        instruction_key="ScrewingNutPickPlace-v1",
        instructions=(
            "pick up the nut and put it on the bolt plate",
            "place the nut onto the bolt plate",
            "grab the nut and set it on the plate with the bolt",
            "move the nut onto the bolt plate",
        ),
        robot_uids=_RIGHT,
        task_module="tasks/nut_bolt/nut_bolt_seat_nut.py",
        max_episode_steps=800,
    ),
    DexTask(
        env_id="FurnitureAssemblyLayFlatPanel-Sharpa-v1",
        instruction_key="SharpaAngledMobilePanelLayFlatLeft-v1",
        instructions=(
            "lay the panel flat on the table",
            "pull the panel out of the rack and lay it flat on the table",
            "take the panel out of its slot and set it flat with the groove facing up",
            "lay the wooden panel down flat with its groove facing up",
        ),
        robot_uids=_LEFT,
        task_module="tasks/furniture_assembly/furniture_assembly_lay_flat_panel.py",
        max_episode_steps=600,
    ),
    DexTask(
        env_id="FurnitureAssemblyInsertRail-Sharpa-v1",
        instruction_key="SharpaAngledMobileRailInsert-v1",
        instructions=(
            "insert the rail into the groove",
            "pick up the rail and seat it in the panel's groove",
            "fit the metal rail into the groove of the panel",
            "put the rail into the groove and push it home",
        ),
        robot_uids=_RIGHT,
        task_module="tasks/furniture_assembly/furniture_assembly_insert_rail.py",
        max_episode_steps=700,
    ),
    DexTask(
        env_id="PipetteTransferAspirateLiquid-Sharpa-v1",
        instruction_key="SharpaAngledMobilePipetteAspirate-v1",
        instructions=(
            "aspirate liquid with the pipette and put it back",
            "pick up the pipette, draw liquid from the cup, and return it to the stand",
            "use the pipette to draw up liquid from the cup, then park it back on its stand",
            "draw liquid into the pipette and put the pipette back where it was",
        ),
        robot_uids=_RIGHT,
        task_module="tasks/pipette_transfer/pipette_transfer_aspirate_liquid.py",
        max_episode_steps=800,
    ),
    DexTask(
        env_id="JuicePreparationOpenFridge-Sharpa-v1",
        instruction_key="SharpaOpenFridge-v1",
        instructions=(
            "open the refrigerator door",
            "open the fridge",
            "pull the refrigerator door open",
            "swing the fridge door open",
        ),
        robot_uids=_LEFT,
        task_module="tasks/juice_preparation/juice_preparation_open_fridge.py",
        max_episode_steps=650,
    ),
    DexTask(
        env_id="JuicePreparationCloseFridge-Sharpa-v1",
        instruction_key="SharpaCloseFridge-v1",
        instructions=(
            "close the refrigerator door",
            "close the fridge",
            "push the refrigerator door shut",
            "swing the fridge door closed",
        ),
        robot_uids=_LEFT,
        task_module="tasks/juice_preparation/juice_preparation_close_fridge.py",
        max_episode_steps=650,
    ),
    DexTask(
        env_id="JuicePreparationAppleToBlender-Sharpa-v1",
        instruction_key="SharpaTakeAppleToBlender-v1",
        instructions=(
            "take the apple and put it into the blender",
            "pick the apple out of the fridge and drop it into the blender",
            "move the apple from the refrigerator into the blender",
            "put the apple in the blender",
        ),
        robot_uids=_RIGHT,
        task_module="tasks/juice_preparation/juice_preparation_apple_to_blender.py",
        max_episode_steps=700,
    ),
    DexTask(
        env_id="JuicePreparationPressButton-Sharpa-v1",
        instruction_key="SharpaMobilePressBlenderButton-v1",
        instructions=(
            "press the blender button",
            "push the power button on the blender",
            "turn the blender on with its button",
            "press the button on the blender",
        ),
        robot_uids=_RIGHT,
        task_module="tasks/juice_preparation/juice_preparation_press_button.py",
        max_episode_steps=400,
    ),
    # ---- 2026-09-16 release: four more single-arm subtasks ----
    DexTask(
        env_id="FurnitureAssemblyFastenScrew-Sharpa-v1",
        instruction_key="SharpaAngledMobileScrewFasten-v1",
        instructions=(
            "turn the wing screw a quarter turn to fasten the rail",
            "tighten the rail's wing screw by a quarter turn",
            "turn the wing screw clockwise to the stop",
            "fasten the rail by turning the wing screw 90 degrees",
        ),
        robot_uids=_RIGHT,
        task_module="tasks/furniture_assembly/furniture_assembly_fasten_screw.py",
        # 600 is what the task module registers. This table said 900 until 2026-09-20, so the
        # first evaluation gave the policy 50% more time than the task allows; the shared
        # scene-evaluation factory refused the mismatch, which is how it surfaced. Every other
        # task in this table passed that same check.
        max_episode_steps=600,
    ),
    DexTask(
        env_id="JuicePreparationPlaceLid-Sharpa-v1",
        instruction_key="SharpaMobilePlaceBlenderLid-v1",
        instructions=(
            "put the lid on the blender",
            "place the blender lid on top of the jar",
            "cover the blender with its lid",
            "pick up the lid and set it on the blender",
        ),
        robot_uids=_RIGHT,
        task_module="tasks/juice_preparation/juice_preparation_place_lid.py",
        max_episode_steps=400,
    ),
    DexTask(
        env_id="PackageOpeningExtractApple-Sharpa-v1",
        instruction_key="SharpaPackageOpeningExtractAndPlaceApple-v1",
        instructions=(
            "take the apple out of the package and put it on the table",
            "lift the apple out of the box and set it on the table",
            "remove the apple from the package and place it on the tabletop",
            "get the apple out of the package and put it down on the table",
        ),
        robot_uids=_RIGHT,
        task_module="tasks/package_opening/package_opening_extract_apple.py",
        max_episode_steps=700,
    ),
    DexTask(
        env_id="TableCleaningPickDebris-Sharpa-v1",
        instruction_key="TableCleanPickDebris-v1",
        instructions=(
            "pick up the debris and put it in the dustpan",
            "pick the pieces off the table and drop them in the dustpan",
            "clear the debris from the table into the dustpan",
            "put the scattered pieces into the dustpan by hand",
        ),
        robot_uids=_RIGHT,
        task_module="tasks/table_cleaning/table_cleaning_pick_debris.py",
        max_episode_steps=900,
    ),
    # ---- 2026-09-17 release: the last three subtasks (data reached 100 episodes) ----
    DexTask(
        env_id="MealPreparationScoopTomato-Sharpa-v1",
        instruction_key="AngledMobileBaseSharpaScoopTomato-v1",
        instructions=(
            "scoop the tomato with the spoon and put it on the plate",
            "pick up the spoon, scoop up the tomato, and tip it onto the plate",
            "use the spoon to lift the tomato out of the box and onto the plate",
            "spoon the tomato onto the plate",
        ),
        robot_uids=_RIGHT,
        task_module="tasks/meal_preparation/meal_preparation_scoop_tomato.py",
        max_episode_steps=1200,
    ),
    # The canonical string says "well"; the scene dispenses into the test tube standing in the
    # rack (tasks/pipette_transfer/pipette_transfer_dispense_liquid.py). It is kept as-is because the released dataset carries it
    # -- rewording it means re-rendering and retraining -- and the paraphrases name the tube.
    DexTask(
        env_id="PipetteTransferDispenseLiquid-Sharpa-v1",
        instruction_key="SharpaAngledMobilePipetteDispense-v1",
        instructions=(
            "dispense the liquid from the pipette into the well",
            "dispense the liquid into the test tube and put the pipette back",
            "take the pipette, empty it into the tube in the rack, and return it to the stand",
            "squeeze the liquid out of the pipette into the tube, then park the pipette",
        ),
        robot_uids=_LEFT,
        task_module="tasks/pipette_transfer/pipette_transfer_dispense_liquid.py",
        max_episode_steps=1100,
    ),
    # DrawerPlug family. All three reached 100 replay-successful episodes: Open in the first
    # release, PickPlace on 2026-09-16, Insert on 2026-09-17.
    DexTask(
        env_id="DrawerPlugOpenDrawer-Sharpa-v1",
        instruction_key="DrawerPlugOpen-v1",
        instructions=(
            "open the middle drawer",
            "pull the middle drawer open",
            "slide the middle drawer out",
            "open the drawer in the middle",
        ),
        robot_uids=_LEFT,
        task_module="tasks/drawer_plug/drawer_plug_open_drawer.py",
        max_episode_steps=600,
    ),
    DexTask(
        env_id="DrawerPlugPickCharger-Sharpa-v1",
        instruction_key="DrawerPlugPickPlace-v1",
        instructions=(
            "take the charger out of the drawer and put it on the table",
            "lift the charger out of the drawer and set it on the table",
            "remove the charger from the drawer and place it on the tabletop",
            "get the charger from the drawer and put it down on the table",
        ),
        robot_uids=_RIGHT,
        task_module="tasks/drawer_plug/drawer_plug_pick_charger.py",
        max_episode_steps=900,
    ),
    DexTask(
        env_id="DrawerPlugInsertPlug-Sharpa-v1",
        instruction_key="DrawerPlugInsert-v1",
        instructions=(
            "plug the charger into the socket",
            "insert the charger into the socket",
            "push the charger's plug into the socket",
            "plug the charger in",
        ),
        robot_uids=_RIGHT,
        task_module="tasks/drawer_plug/drawer_plug_insert_plug.py",
        max_episode_steps=900,
    ),
)

DEX_TASKS_BY_ENV: dict[str, DexTask] = {task.env_id: task for task in DEX_TASKS}


def resolve_dex_task(env_id: str) -> DexTask:
    try:
        return DEX_TASKS_BY_ENV[env_id]
    except KeyError:
        raise KeyError(
            f"No dexterous-hand task registered for env {env_id!r}. "
            f"Known: {sorted(DEX_TASKS_BY_ENV)}. Add a DexTask entry in utils/dex_tasks.py."
        ) from None


# ---- language instructions -------------------------------------------------------------

def dex_instructions(env_id: str) -> tuple[str, ...]:
    """Every instruction registered for a task, canonical one first."""
    return resolve_dex_task(env_id).instructions


def dex_instruction(env_id: str, index: int = 0) -> str:
    """One instruction by position; index 0 is the canonical one the datasets carry."""
    return resolve_dex_task(env_id).instruction(index)


def sample_dex_instruction(
    env_id: str, *, seed: int | None = None, rng: random.Random | None = None
) -> str:
    """Draw an instruction uniformly at random. Pass ``seed`` or ``rng`` for reproducibility."""
    if rng is None:
        rng = random.Random(seed) if seed is not None else random
    return rng.choice(resolve_dex_task(env_id).instructions)


def draw_instruction(key: str, seed: int, instructions: Sequence[str]) -> str:
    """Pick one instruction from ``(key, seed)`` alone.

    Derived by SHA-256 rather than from a stateful RNG, so the same seed yields the same
    instruction in any process, in any order, however many draws came before it. That is what
    lets an evaluation vary the wording across episodes and still be reproducible.
    """
    instructions = tuple(instructions)
    if not instructions:
        raise ValueError(f"No instructions to draw from for {key!r}.")
    digest = hashlib.sha256(f"{key}:{int(seed)}".encode()).digest()
    return instructions[int.from_bytes(digest[:8], "big") % len(instructions)]


def episode_instruction(env_id: str, episode_seed: int) -> str:
    """The instruction one evaluation episode of a registered dex task runs under.

    Keyed by the task's ``instruction_key``, so a given seed draws the same instruction
    before and after an env id rename.
    """
    task = resolve_dex_task(env_id)
    return draw_instruction(task.instruction_key, episode_seed, task.instructions)


def _instruction_order(env_id: str) -> tuple[int, ...]:
    """A per-task permutation of the instruction positions.

    Ordered by SHA-256 rather than by a seeded ``random.shuffle``: the shuffle agrees across
    the Python versions in use today, but that is an implementation detail of CPython's RNG,
    not a guarantee. This assignment gets baked into datasets that several teams have to
    reproduce independently, so it is derived the same way the evaluation draw is.

    Keyed by the task's ``instruction_key`` rather than its env id, so renaming the env
    does not reshuffle which episode carries which phrasing.
    """
    task = resolve_dex_task(env_id)
    positions = range(len(task.instructions))
    return tuple(sorted(positions, key=lambda position: hashlib.sha256(
        f"{task.instruction_key}:{DEX_INSTRUCTIONS_VERSION}:order:{position}".encode()).digest()))


def dataset_episode_instruction(env_id: str, episode_index: int) -> str:
    """The instruction one RECORDED episode is labelled with.

    Balanced round-robin over a per-task permutation of the list, rather than an independent
    draw per episode, for two reasons:

      * 100 episodes then carry exactly 25 of each of four phrasings. An i.i.d. draw leaves
        the split to chance (21/24/27/28 on the first task tried), and the rarest phrasing is
        the one the policy learns worst.
      * It depends on the episode index alone, so the renderer's parallel 10-episode shards
        each compute their own labels and still agree once aggregated.

    The permutation is keyed by the task, so the same episode index does not get position 0
    in every task.
    """
    task = resolve_dex_task(env_id)
    order = _instruction_order(env_id)
    return task.instructions[order[int(episode_index) % len(order)]]


def assign_dataset_instructions(env_id: str, episode_indices: Sequence[int]) -> dict[int, str]:
    """``dataset_episode_instruction`` over many episodes -> {episode_index: instruction}."""
    return {int(index): dataset_episode_instruction(env_id, index) for index in episode_indices}


def is_dex_instruction(env_id: str, text: str) -> bool:
    return text in resolve_dex_task(env_id).instructions


def validate_dex_instruction(env_id: str, text: str) -> str:
    """Return ``text`` if the task registers it; otherwise say what is allowed.

    An unregistered instruction is the failure this table exists to prevent: it trains or
    evaluates on a string nobody else uses, and nothing else in the pipeline would notice.
    """
    task = resolve_dex_task(env_id)
    if text not in task.instructions:
        listed = "\n  ".join(repr(candidate) for candidate in task.instructions)
        raise ValueError(
            f"{text!r} is not a registered instruction for {env_id!r}. Use one of:\n  {listed}\n"
            "Add it to utils/dex_tasks.py if the task really should accept it."
        )
    return text


def instruction_table() -> dict:
    """JSON-ready export of the table, for tooling that cannot import this module."""
    return {
        "schema": "robodex.dex_task_instructions",
        "version": DEX_INSTRUCTIONS_VERSION,
        "tasks": {
            task.env_id: {
                "instructions": list(task.instructions),
                "instruction_key": task.instruction_key,
                "canonical": task.prompt,
                "robot_uids": task.robot_uids,
                "task_module": task.task_module,
                "max_episode_steps": task.max_episode_steps,
            }
            for task in DEX_TASKS
        },
    }


if __name__ == "__main__":
    import sys

    if "--json" in sys.argv[1:]:
        import json

        print(json.dumps(instruction_table(), indent=2, ensure_ascii=False))
    else:
        for task in DEX_TASKS:
            print(f"{task.env_id:42s} {task.robot_uids:44s} dims={task.state_dim}/{task.action_dim} "
                  f"steps={task.max_episode_steps} groups={[n for n, _ in task.groups]} "
                  f"instructions={len(task.instructions)}")
            for index, text in enumerate(task.instructions):
                print(f"{'':42s} [{index}] {text}")
