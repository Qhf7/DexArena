"""Screwing stages under GPU PhysX.

GPU PhysX cannot lock motion axes after load, so a body a stage holds for the
whole episode is built kinematic instead (``gpu_build_held``); a hold that only
serves a later stage is skipped (``gpu_unheld_ok``) and the stage's latched flag
becomes its success. State-dict layout is unchanged; CPU never reads either.
"""

from __future__ import annotations

import torch

from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.structs.pose import Pose

from ._shims import patched


def _build_kinematic(name_: str):
    orig = ActorBuilder.build

    def build(builder, name=None):
        if name == name_:
            builder.set_physx_body_type("kinematic")
        return orig(builder, name)

    return patched(ActorBuilder, "build", build)


class ScrewingMixin:
    # Subset of {"bolt", "nut"}: held from reset to the end of every episode.
    gpu_build_held: tuple = ()
    # Subset of {"bolt", "nut"}: bodies whose hold only serves a later stage.
    gpu_unheld_ok: tuple = ()

    def _gpu_build_holds(self, name: str) -> bool:
        return bool(self.gpu_sim_enabled) and name in self.gpu_build_held

    def _gpu_skips_latch(self, name: str) -> bool:
        return bool(self.gpu_sim_enabled) and name in self.gpu_unheld_ok

    def _load_bolt(self):
        if not self._gpu_build_holds("bolt"):
            return super()._load_bolt()
        with _build_kinematic("bolt"):
            super()._load_bolt()

    def _load_nut(self):
        if not self._gpu_build_holds("nut"):
            return super()._load_nut()
        with _build_kinematic("nut"):
            super()._load_nut()

    def _place_bolt(self, env_idx: torch.Tensor, b: int, already_placed=False):
        super()._place_bolt(env_idx, b, already_placed)
        if already_placed and self._gpu_build_holds("bolt"):
            # CPU parks the bolt on its lock pose through the pose write in
            # _refresh_hold_state; the GPU branch writes nothing, so the
            # kinematic bolt goes there directly.
            with torch.device(self.device):
                self.bolt.set_pose(Pose.create(self._bolt_lock_pose[env_idx]))

    def _engage_newly_placed_nuts(self, mask: torch.Tensor):
        # The swap into the revolute twin prepares the TURN stage; a stage that
        # lists "nut" in gpu_unheld_ok ends at this transition.
        if self._gpu_skips_latch("nut"):
            return
        super()._engage_newly_placed_nuts(mask)

    def _lock_newly_placed_bolts(self, mask: torch.Tensor):
        # The lock holds the bolt for the NUT stages; a stage that lists "bolt"
        # in gpu_unheld_ok ends at this transition.
        if self._gpu_skips_latch("bolt"):
            return
        super()._lock_newly_placed_bolts(mask)

    def _sync_nut_visibility(self):
        # A shown<->hidden flip after the batched render system exists raises,
        # and rgb on GPU builds it. The parked representation sits at +5 m,
        # where a 3 cm nut is under one pixel of a 128 px policy camera.
        if self.gpu_sim_enabled:
            return
        super()._sync_nut_visibility()

    def _refresh_hold_state(self, write_pose: bool = True):
        if not self.gpu_sim_enabled:
            return super()._refresh_hold_state(write_pose)
        if not hasattr(self, "_bolt_locked"):
            return
        # Nothing to write on GPU: a held body was built kinematic and a
        # tolerated one stays dynamic. Only check that every set mask is one
        # the stage declared.
        unsupported = []
        for name, mask in (
            ("bolt", self._bolt_locked if self.has_bolt else None),
            ("nut", self._nut_engaged if self.has_nut else None),
        ):
            if mask is None:
                continue
            if name in self.gpu_build_held:
                if not bool(mask.all()):
                    raise RuntimeError(
                        f"{type(self).__name__} built `{name}` kinematic for GPU sim "
                        "(gpu_build_held) but its hold mask is not set in every env; "
                        "a kinematic body cannot be freed."
                    )
            elif name not in self.gpu_unheld_ok and bool(mask.any()):
                unsupported.append(name)
        if unsupported:
            raise RuntimeError(
                f"GPU sim cannot hold {unsupported} for {type(self).__name__}: the hold "
                "uses the PhysX kinematic flag, which GPU PhysX cannot change after "
                "scene load. Run the task on CPU (num_envs=1)."
            )


class ScrewingBoltPickPlaceMixin(ScrewingMixin):
    # Success IS the placement; the lock it triggers only serves the nut stages.
    gpu_unheld_ok = ("bolt",)

    def evaluate(self):
        info = super().evaluate()
        if self._gpu_skips_latch("bolt"):
            # Without the lock a restored post-success state can fail "released"
            # again while the stage flag keeps paying reward; latch the success.
            success = info["bolt_placed"] | self._stage_bolt_placed
            info["success_condition"] = success
            info["success"] = success
        return info


class ScrewingNutPickPlaceMixin(ScrewingMixin):
    # The bolt is the jig: locked from reset. Success IS the nut engagement;
    # the swap into the revolute twin it triggers only serves the turn stage.
    gpu_build_held = ("bolt",)
    gpu_unheld_ok = ("nut",)

    def evaluate(self):
        info = super().evaluate()
        if self._gpu_skips_latch("nut"):
            success = info["nut_ready"] | self._stage_nut_placed
            info["success_condition"] = success
            info["success"] = success
        return info


class ScrewingTurnMixin(ScrewingMixin):
    # Both holds are preconditions of this stage, set at reset and never
    # released: the bolt is the jig, and the free nut is parked at +5 m while
    # the revolute twin plays the nut.
    gpu_build_held = ("bolt", "nut")
