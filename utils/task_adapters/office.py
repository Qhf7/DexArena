"""Office keyboard / mouse: the key and button drive targets are written at load."""

from ._base import GpuLoadWritesMixin


class KeyboardTypeMixin(GpuLoadWritesMixin):
    pass


class ClickMouseMixin(GpuLoadWritesMixin):
    pass
