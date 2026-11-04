#!/usr/bin/env bash
# Scene-library entry point. The tools themselves live in tasks/scene_builders/.
#
#   ./scripts/scene_library.sh surfaces <scene> [--min-area 0.15]
#   ./scripts/scene_library.sh fetch    <objaverse-uid> [--endpoint https://hf-mirror.com]
#   ./scripts/scene_library.sh fetch-labutopia [--source /path/to/lab_003] [--out DIR]
#   ./scripts/scene_library.sh labutopia-preview [--env-id TASK] [--out DIR]
#
#   surfaces   measure the flat work surfaces inside a single-mesh scene     (tasks.scene_builders.find_glb_surfaces)
#   fetch      download an Objaverse mesh into the GLB scene directory       (tasks.scene_builders.fetch_objaverse_scene)
#
# Everything after the mode is passed through untouched. Set PYTHON to pick an
# interpreter. labutopia-preview needs a render device, so on the cluster it belongs in a
# GPU job -- see docs/howto_scene_library.md.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

# MS_ASSET_DIR is the .maniskill root and ManiSkill appends `data` itself, so the
# GLB directory sits one level further down. Deriving it here means a job script
# that sets the asset root does not also have to know that layout.
if [[ -n "${MS_ASSET_DIR:-}" && -z "${ROBODEX_SCENE_GLB_DIR:-}" ]]; then
  export ROBODEX_SCENE_GLB_DIR="$MS_ASSET_DIR/data/scene_datasets/glb_scenes"
fi

MODE="${1:-}"
[[ $# -gt 0 ]] && shift
case "$MODE" in
  surfaces)     MODULE="tasks.scene_builders.find_glb_surfaces" ;;
  labutopia-preview) MODULE="tasks.scene_builders.render_labutopia" ;;
  fetch)        MODULE="tasks.scene_builders.fetch_objaverse_scene" ;;
  fetch-labutopia)
    exec "$PYTHON" "$REPO_ROOT/tasks/scene_builders/fetch_labutopia_scene.py" "$@" ;;
  -h|--help|"") sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 0 ;;
  *)            echo "unknown mode '$MODE' (surfaces|fetch|fetch-labutopia|labutopia-preview)" >&2; exit 2 ;;
esac

cd "$REPO_ROOT"
exec "$PYTHON" -m "$MODULE" "$@"
