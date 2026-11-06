"""Common Panda agent bases with frozen non-hand structural materials."""

from mani_skill.agents.base_agent import BaseAgent

from tasks.config.contact_materials import apply_panda_structure_contact_material


class PandaStructureContactMixin:
    """Apply project-wide Franka/mount/base materials after URDF loading."""

    def _apply_frozen_contact_materials(self):
        return apply_panda_structure_contact_material(self.robot)

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        self.frozen_contact_material_counts = (
            self._apply_frozen_contact_materials()
        )


class PandaAgent(PandaStructureContactMixin, BaseAgent):
    """Base class for repository-defined Panda agents.

    New Panda agents must inherit this instead of ManiSkill ``BaseAgent``.
    """
