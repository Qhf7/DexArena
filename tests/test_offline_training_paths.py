import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from utils.offline_replay_buffer import load_lerobot_replay_dataset
from utils.paths import (
    DATA_ROOT,
    RUNS_ROOT,
    STORAGE_ROOT,
    demo_path,
    ensure_parent_dir,
    resolve_training_paths,
    runs_path,
)
from utils.train_utils import eval_video_output_dir


def _write_failed_episode(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    features = {
        "observation.state": {"dtype": "float32", "shape": [1]},
        "action": {"dtype": "float32", "shape": [1]},
        "next.reward": {"dtype": "float32", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
        "next.truncated": {"dtype": "bool", "shape": [1]},
        "next.success": {"dtype": "bool", "shape": [1]},
    }
    (root / "meta" / "info.json").write_text(json.dumps({"features": features}))
    table = pa.table(
        {
            "episode_index": [0, 0, 0],
            "frame_index": [0, 1, 2],
            "observation.state": [[0.0], [1.0], [2.0]],
            "action": [[10.0], [11.0], [12.0]],
            "next.reward": [[1.0], [2.0], [3.0]],
            "next.done": [[False], [False], [True]],
            "next.truncated": [[False], [False], [True]],
            "next.success": [[False], [False], [False]],
        }
    )
    pq.write_table(table, root / "data" / "episode.parquet")


def test_failed_episode_closes_retained_replay_tail(tmp_path: Path) -> None:
    dataset_root = tmp_path / "demo"
    _write_failed_episode(dataset_root)

    replay = load_lerobot_replay_dataset(str(dataset_root), gamma=0.5)

    assert len(replay) == 2
    torch.testing.assert_close(replay.dones[:, 0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(replay.next_observations[:, 0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(replay.next_actions[:, 0], torch.tensor([11.0, 12.0]))
    torch.testing.assert_close(replay.returns_to_go[:, 0], torch.tensor([2.0, 2.0]))

    sparse_replay = load_lerobot_replay_dataset(
        str(dataset_root), gamma=0.5, sparse_reward=True
    )
    torch.testing.assert_close(sparse_replay.dones[:, 0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(sparse_replay.returns_to_go[:, 0], torch.zeros(2))


def test_data_and_run_paths_use_their_configured_roots(tmp_path: Path) -> None:
    assert demo_path("teleop_lerobot") == str(DATA_ROOT / "teleop_lerobot")
    assert runs_path("ppo", "final_ckpt.pt") == str(
        RUNS_ROOT / "ppo" / "final_ckpt.pt"
    )
    output = Path(ensure_parent_dir(tmp_path / "run" / "checkpoint.pt"))
    assert output.parent.is_dir()

    args = SimpleNamespace(
        evaluate=True,
        checkpoint=None,
        pretrained_checkpoint="/checkpoints/iql/final.pt",
    )
    assert eval_video_output_dir(args, "eval") == "/checkpoints/iql/test_videos"

    args.pretrained_checkpoint = None
    assert eval_video_output_dir(args, "eval") == str(RUNS_ROOT / "eval" / "videos")


def test_legacy_trainer_paths_use_shared_roots() -> None:
    args = SimpleNamespace(
        demo_data_path="teleop_lerobot",
        bc_data_path="teleop_lerobot",
        demo_init_data_path=None,
        checkpoint="runs/ppo/final_ckpt.pt",
        base_policy_path="runs/diffusion/final_ckpt.pt",
    )

    resolve_training_paths(args)

    assert args.demo_data_path == str(DATA_ROOT / "teleop_lerobot")
    assert args.bc_data_path == str(DATA_ROOT / "teleop_lerobot")
    assert args.checkpoint == str(STORAGE_ROOT / "runs/ppo/final_ckpt.pt")
    assert args.base_policy_path == str(
        STORAGE_ROOT / "runs/diffusion/final_ckpt.pt"
    )
