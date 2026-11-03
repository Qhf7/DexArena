"""Subtask 2 of ``TableCleaningLong-Sharpa-v1``: empty the loaded dustpan into the bin.

The dustpan is the tool: pick it up by its handle with the debris already
aboard, carry it over the trash bin and tip it out, then put the dustpan down
at its stow spot. This resets from the state subtask 1 ends in -- debris in the
pan, hand empty -- so it trains without subtask 1 being solved first. Pass
``start_loaded=False`` to reset with the debris loose on the table instead (for
replaying a picked end-state supplied externally).

Success (user): every piece resting on the bin floor for
``DUMP_HOLD_STEPS`` (10) consecutive control steps, latched. Grasping, pouring
and setting the pan down are milestones for the HUD and the dense reward only.

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

from tasks.table_cleaning.table_cleaning_long import LIFT_H, PAN_STOW_XY, PICK_DEBRIS_COUNT, TableCleanBase

ENV_ID = "TableCleaningDumpDustpan-Sharpa-v1"
MAX_EPISODE_STEPS = 600          # 30 s: lift, carry, pour, set down
STAGE_NAMES = ("grasp_pan", "pour", "stow_pan", "done")


@register_task(
    "TableCleaningDumpDustpan",
    side="right",
    max_episode_steps=MAX_EPISODE_STEPS,
)
class TableCleaningDumpDustpanEnv(TableCleanBase):
    """pick up the loaded dustpan -> pour into the bin -> put the dustpan down."""

    MAX_REWARD = 10.0
    STAGE_ONE_HOT_CLASSES = 4

    def __init__(self, *args, start_loaded=True, dump_target=None,
                 debris_count=PICK_DEBRIS_COUNT, **kwargs):
        self.start_loaded = bool(start_loaded)
        self._dump_target = (max(1, int(debris_count) - 1)
                             if dump_target is None else int(dump_target))
        super().__init__(*args, debris_count=debris_count, **kwargs)

    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            self._place_scene_start(env_idx)
            if self.start_loaded:
                self._place_in_pan(len(env_idx), env_idx)
                # Stage 1 is the precondition here, so it starts satisfied --
                # the same latch the long task carries into its stage 2.
                self.f_picked[env_idx] = True

    def evaluate(self):
        n_pan, n_bin, grasped, lift = self._update_dump(
            self._dump_target, gate=self.f_picked)
        self.stage = (self.f_grasp_pan.long() + self.f_poured.long()
                      + self.f_stow_pan.long())
        n_floor, all_in = self._dump_success(gate=self.f_picked)
        return {
            "success": self.success_latched,
            "n_in_pan": n_pan, "n_in_bin": n_bin, "n_on_bin_floor": n_floor,
            "all_in_bin": all_in,
            "pan_grasped": grasped, "pan_lift": lift,
            "pan_tilt": self._tilt(self.pan),
            "stage": self.stage,
            "grasp_pan": self.f_grasp_pan,
            "poured": self.f_poured,
            "stow_pan": self.f_stow_pan,
        }

    def compute_dense_reward(self, obs, action, info):
        r = torch.zeros(self.num_envs, device=self.device)
        reach = 1.0 - torch.tanh(5.0 * self._tcp_to(self.pan))
        r = r + reach
        r = r + 1.5 * info["pan_grasped"].float()
        r = r + 1.0 * torch.clamp(info["pan_lift"] / LIFT_H, 0.0, 1.0) \
            * info["pan_grasped"].float()
        keep = info["n_in_pan"] / max(1, self.debris_count)
        r = r + 1.0 * keep * self.f_grasp_pan.float()
        bin_xy = self.bin.pose.p[:, :2]
        d_bin = torch.linalg.norm(self.pan.pose.p[:, :2] - bin_xy, dim=1)
        r = r + 1.5 * (1.0 - torch.tanh(3.0 * d_bin)) * self.f_grasp_pan.float()
        r = r + 3.0 * (info["n_in_bin"] / max(1, self.debris_count))
        stow = torch.tensor(PAN_STOW_XY, dtype=torch.float32,
                            device=self.device).view(1, 2)
        d_stow = torch.linalg.norm(self.pan.pose.p[:, :2] - stow, dim=1)
        r = r + 1.0 * (1.0 - torch.tanh(3.0 * d_stow)) * self.f_poured.float()
        return torch.where(info["success"],
                           torch.full_like(r, self.MAX_REWARD), r)
