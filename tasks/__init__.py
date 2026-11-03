"""Import every task module so that all environment ids are registered."""
from importlib import import_module
from pkgutil import iter_modules

FAMILIES: tuple[str, ...] = (
    "cabinet_pouring",
    "drawer_plug",
    "furniture_assembly",
    "juice_preparation",
    "meal_preparation",
    "nut_bolt",
    "office_desk",
    "package_opening",
    "pipette_transfer",
    "table_cleaning",
)


def _import_task_modules() -> list[str]:
    names = []
    for family in FAMILIES:
        package = import_module(f"{__name__}.{family}")
        for info in iter_modules(package.__path__):
            if info.name.startswith("_"):
                continue
            import_module(f"{package.__name__}.{info.name}")
            names.append(f"{family}.{info.name}")
    return names


TASK_MODULES: tuple[str, ...] = tuple(_import_task_modules())
