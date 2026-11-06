# Deliberately does NOT re-export the scene library. Collection-target modules
# import this package (`from tasks.scene_builders import TableSceneBuilder`), and
# the code fingerprint walks every import in a module's AST, so anything named
# here lands in those environments' closures and re-versions their datasets.
# Import the library from its own module instead:
#
#     from tasks.scene_builders.scene_library import FullSceneBuilder
from tasks.scene_builders.table import TableSceneBuilder
from tasks.scene_builders.randomized_table import RandomizedTableSceneBuilder
from tasks.scene_builders.camera_dr import jitter_pose

__all__ = ["TableSceneBuilder", "RandomizedTableSceneBuilder", "jitter_pose"]
