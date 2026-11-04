"""FurnitureAssembly rail insert."""

from ._base import GpuLoadWritesMixin


class RailInsertGrooveMixin(GpuLoadWritesMixin):
    """The screw joint's velocity target is written at load (``build_slider_rail``);
    under physx_cuda that write is deferred past GPU init."""


class ScrewFastenMixin(GpuLoadWritesMixin):
    """``build_wing_screw_rail`` calls ``apply_screw_thread_resistance``, which
    writes the screw joint's velocity target at load; deferred the same way."""
