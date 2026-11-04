# DexArena base camera

Use `build_dexarena_base_camera` when a task wants to opt in to the shared
DexArena base camera schema.

```python
from utils.camera_configs import build_dexarena_base_camera

@property
def _default_sensor_configs(self):
    return [build_dexarena_base_camera()]
```

Task owners can keep the same schema while adjusting the view:

```python
return [
    build_dexarena_base_camera(
        eye=[0.55, -0.55, 0.45],
        target=[0.04, 0.0, 0.08],
    )
]
```
