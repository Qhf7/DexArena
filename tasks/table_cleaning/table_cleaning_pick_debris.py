"""Subtask 1 of ``TableCleaningLong-Sharpa-v1``: pick the debris up by hand and put it in the dustpan.

No tool. The operator picks each 20 mm piece off the table and drops it into
the dustpan. The dustpan sits where the long task places it and the debris
spawns over exactly the same spread, so a policy trained here sees the long
task's stage-1 reset unchanged. The trash bin is in the scene (shared with
``TableCleaningLongEnv``) but nothing here touches it.

Success is the long task's first milestone: at least ``pick_target`` pieces
resting in the dustpan with NO piece still held, latched for ``HOLD_STEPS``
consecutive control steps. The "hand off" term is what makes a piece placed
and released count and a piece merely carried over the pan not count.

Everything else -- scene, sensors, robot start pose, grasp and containment
predicates -- is inherited from ``TableCleanBase`` in ``tasks/table_cleaning/table_cleaning_long.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch

from tasks.config.hands import register_task

from tasks.table_cleaning.table_cleaning_long import PICK_DEBRIS_COUNT, TableCleanBase

ENV_ID = "TableCleaningPickDebris-Sharpa-v1"
# 700 was below the longest recorded human episode (783 steps; p95 661);
# randomized debris positions add reach. The budget only binds training / evaluation.
MAX_EPISODE_STEPS = 900          # 45 s for three hand-picks
STAGE_NAMES = ("pick_debris", "done")


@register_task(
    "TableCleaningPickDebris",
    side="right",
    max_episode_steps=MAX_EPISODE_STEPS,
)
class TableCleaningPickDebrisEnv(TableCleanBase):
    """pick each piece by hand -> drop it in the dustpan."""

    MAX_REWARD = 10.0
    STAGE_ONE_HOT_CLASSES = 2

    def __init__(self, *args, pick_target=None,
                 debris_count=PICK_DEBRIS_COUNT, **kwargs):
        # (user): every piece has to be in the pan, not count-1.
        self._pick_target = (max(1, int(debris_count))
                             if pick_target is None else int(pick_target))
        super().__init__(*args, debris_count=debris_count, **kwargs)

    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            self._place_scene_start(env_idx)

    def evaluate(self):
        n_pan, picked_now = self._update_pick(self._pick_target)
        n_bin = self._debris_in_bin()
        self.stage = self.f_picked.long()
        self.success_latched |= self._latch(self.f_picked)
        return {
            "success": self.success_latched,
            "n_in_pan": n_pan, "n_in_bin": n_bin,
            "stage": self.stage,
            "picked": self.f_picked,
            "debris_held": self._any_debris_grasped(),
        }

    def compute_dense_reward(self, obs, action, info):
        r = torch.zeros(self.num_envs, device=self.device)
        n = max(1, self.debris_count)
        nearest = torch.stack([self._tcp_to(d) for d in self.debris], dim=0).min(dim=0).values
        r = r + (1.0 - torch.tanh(5.0 * nearest)) * (~self.f_picked).float()
        r = r + 6.0 * (info["n_in_pan"] / n)
        r = r + 3.0 * self.f_picked.float()
        return torch.where(info["success"],
                           torch.full_like(r, self.MAX_REWARD), r)
