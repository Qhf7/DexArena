"""Runtime adapters that make the task classes trainable by the pixel agents.

``adapt(env_id, base_cls, ...)`` builds a subclass of the task class from the
mixins registered for it; the task files themselves stay as every other
algorithm sees them. The generic part (``PixelObsMixin``) is the observation
contract; the per-family mixins carry what the tasks need under ``physx_cuda``
and ``num_envs > 1``, with the CPU single-env path deferring to the task.
"""

from __future__ import annotations

from ._base import NumEnvsTrampoline, PixelObsMixin

# Task class name -> mixins, most derived task class first.
ADAPTERS: dict[str, tuple[str, ...]] = {
    "CabinetPouringPickBottleEnv": ("cabinet_pour", "BottlePickPlaceMixin"),
    "CabinetPouringUnscrewCapEnv": ("cabinet_pour", "CapRemovePlaceMixin"),
    "CabinetPouringLongEnv": ("cabinet_pour", "CabinetPourStageMixin"),
    "FurnitureAssemblyInsertRailEnv": ("furniture", "RailInsertGrooveMixin"),
    "FurnitureAssemblyFastenScrewEnv": ("furniture", "ScrewFastenMixin"),
    "JuicePreparationCloseFridgeEnv": ("juice", "CloseFridgeMixin"),
    "JuicePreparationOpenFridgeEnv": ("juice", "OpenFridgeMixin"),
    "JuicePreparationPlaceLidEnv": ("juice", "PlaceBlenderLidMixin"),
    "JuicePreparationPressButtonEnv": ("juice", "PressBlenderButtonMixin"),
    "JuicePreparationAppleToBlenderEnv": ("juice", "TakeAppleToBlenderMixin"),
    "MealPreparationBase": ("meal_preparation", "MealPreparationMixin"),
    "OfficeDeskTypeKeysEnv": ("office", "KeyboardTypeMixin"),
    "OfficeDeskClickMouseEnv": ("office", "ClickMouseMixin"),
    "PackageOpeningLongEnv": ("package_opening", "PackageOpeningMixin"),
    "PipetteTransferLongEnv": ("pipette", "PipetteTransferMixin"),
    "NutBoltPickBoltEnv": ("screwing", "ScrewingBoltPickPlaceMixin"),
    "NutBoltSeatNutEnv": ("screwing", "ScrewingNutPickPlaceMixin"),
    "NutBoltTurnNutEnv": ("screwing", "ScrewingTurnMixin"),
    "TableCleaningDumpDustpanEnv": ("table_clean", "DumpDustpanMixin"),
    "TableCleanBase": ("table_clean", "TableCleanMixin"),
}


def family_mixin(base_cls) -> type | None:
    """The registered mixin of the most derived task class in ``base_cls``'s MRO."""
    import importlib

    for cls in base_cls.__mro__:
        entry = ADAPTERS.get(cls.__name__)
        if entry is not None:
            module = importlib.import_module(f"{__name__}.{entry[0]}")
            return getattr(module, entry[1])
    return None


def adapt(env_id: str, base_cls, *, reward_info_key: str | None = None) -> type:
    mixins = [PixelObsMixin]
    family = family_mixin(base_cls)
    if family is not None:
        mixins.append(family)
    bases = (*mixins, base_cls)
    if any(getattr(m, "_pixel_needs_trampoline", False) for m in mixins):
        bases += (NumEnvsTrampoline,)
    return type(
        base_cls.__name__ + "Pixels",
        bases,
        {
            "_pixel_base": base_cls,
            "_pixel_env_id": env_id,
            "_pixel_reward_info_key": reward_info_key,
            "__module__": __name__,
        },
    )
