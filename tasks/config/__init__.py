"""Single source of truth for locked simulation parameters.

- ``sim_profiles``: environment-layer SimConfig (solver / offsets / flags).
- ``object_materials``: task-object contact friction (physical pair-mu table).
"""

from tasks.config.sim_profiles import make_sim_config  # noqa: F401
from tasks.config.object_materials import (  # noqa: F401
    PAD_ENGINE_MU,
    REAL_PAIR_MU,
    object_friction,
    object_material,
)
