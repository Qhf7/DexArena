"""GR00T baseline — eval-side agent (NVIDIA Isaac GR00T N1.7-3B).

GR00T *training* is done by GR00T's own fine-tune CLI (see ``scripts/groot/``). This module
is the **evaluation bridge**: :class:`GrootAgent` loads a finetuned GR00T policy in-process
(:class:`gr00t.policy.gr00t_policy.Gr00tPolicy`) — the unified env has ``gr00t`` and ManiSkill
in one venv, so no inference server/socket is needed. It turns a ManiSkill ``obs_mode="rgb"``
observation into GR00T's nested observation dict, calls ``get_action``, and exposes the
returned action chunk to our ``evaluate_policy`` loop (``train/train_groot.py``).

Contract used by the eval loop (mirrors how dppo/diffusion agents are driven):
    agent.eval()
    agent.reset_rollout()                 # clears any per-episode state
    chunk = agent.predict_action_chunk(obs)   # torch.Tensor (1, action_horizon, action_dim)
    agent.logger                          # utils.logger.Logger for scalar logging

GR00T's wire format (verified at pinned commit ab88b50; see the integration-plan API section):
    observation = {
        "video":    {"<cam>":   uint8   (B, T, H, W, 3)},   # keys = video modality_keys
        "state":    {"<state>": float32 (B, T, D)},          # keys = state modality_keys
        "language": {"<lang>":  [[str]]},                    # (B, 1)
    }
    action, info = policy.get_action(observation)
    action = {"<action_key>": float32 (B, T, D)}             # physical units; take [:, t, :]

We run GR00T eval with a single env (B = 1); ``extract_camera_rgb`` is single-env. Multi-env
batching is left as future work.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from utils.image_observation import (
    BASE_CAMERA_ALIASES,
    LEFT_WRIST_CAMERA_ALIASES,
    RIGHT_WRIST_CAMERA_ALIASES,
    WRIST_CAMERA_ALIASES,
    extract_camera_rgb,
    extract_franka_proprio,
)
from utils.logger import Logger, SummaryWriter


# Default mapping from a GR00T video modality key -> ManiSkill camera aliases.
# Keys mirror the SO100 convention ("head"/"wrist"); reconcile with meta/modality.json.
DEFAULT_VIDEO_KEY_ALIASES = {
    "head": BASE_CAMERA_ALIASES,
    "base": BASE_CAMERA_ALIASES,
    "front": BASE_CAMERA_ALIASES,
    # A single-arm task has one wrist view and calls it "wrist"; a dual-arm task has one per
    # arm and must resolve each SIDE to that arm's camera. Mapping both sides to the generic
    # tuple would silently feed the same image to both keys (2026-09-17: OpenPackage's eval
    # asserted on the missing key instead, which is how this was caught).
    "wrist": WRIST_CAMERA_ALIASES,
    "left_wrist": LEFT_WRIST_CAMERA_ALIASES,
    "right_wrist": RIGHT_WRIST_CAMERA_ALIASES,
}

# Default state split for the Franka gripper embodiment: 7 arm joints + 1 gripper width.
# (name, start, end) into the flat proprio vector. Override per-embodiment from modality.json.
GRIPPER_STATE_LAYOUT: Tuple[Tuple[str, int, int], ...] = (
    ("single_arm", 0, 7),
    ("gripper", 7, 8),
)
GRIPPER_ACTION_KEYS: Tuple[str, ...] = ("single_arm", "gripper")


def _build_logger(args, run_name: str) -> Logger:
    """Construct the repo's Logger exactly like dppo/diffusion agents do."""
    writer = SummaryWriter(f"runs/{run_name}")
    return Logger(log_swanlab=getattr(args, "track", False), tensorboard=writer)


def _ensure_local_backbone(model_path) -> None:
    """Token-free load: repoint the checkpoint's baked-in backbone name at
    ``$GROOT_BACKBONE_PATH`` when set. The finetune bakes the base model's gated backbone name
    (``nvidia/Cosmos-Reason2-2B``) into every checkpoint, so an in-process load would otherwise
    hit the gated HF API. Atomic writes; each file is a no-op if it already resolves to a local dir.

    Two files carry that name and **both** matter:

    - ``config.json`` -> ``model_name``, which loads the model weights;
    - ``processor_config.json`` -> ``processor_kwargs.model_name``, which loads the
      *tokenizer* and image processor.

    Patching only the first is what makes an otherwise fully-local evaluation stall on
    ``HEAD https://huggingface.co/nvidia/Cosmos-Reason2-2B/resolve/main/tokenizer_config.json``
    even though that file is sitting in the local backbone directory.
    """
    import json
    import os
    import shutil
    import tempfile

    bb = os.environ.get("GROOT_BACKBONE_PATH")
    if not bb:
        return

    # (filename, path to the model-name field within that file)
    targets = (
        ("config.json", ("model_name",)),
        ("processor_config.json", ("processor_kwargs", "model_name")),
    )
    for name, keys in targets:
        cfg = os.path.join(str(model_path), name)
        if not os.path.isfile(cfg):
            continue
        with open(cfg) as f:
            c = json.load(f)
        node = c
        for k in keys[:-1]:
            node = node.get(k) if isinstance(node, dict) else None
            if node is None:
                break
        if not isinstance(node, dict):
            continue
        cur = node.get(keys[-1])
        if cur == bb or os.path.isdir(str(cur or "")):
            continue
        node[keys[-1]] = bb
        d = os.path.dirname(cfg) or "."
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(c, f, indent=2, ensure_ascii=False)
            f.write("\n")
        shutil.copymode(cfg, tmp)
        os.replace(tmp, cfg)
        print(f"[groot] set {name} {'.'.join(keys)} -> {bb}")


class _BaseEvalAgent(nn.Module):
    """Shared eval-agent plumbing (logger + chunk contract) for GR00T and the stub."""

    def __init__(self, args, *, action_dim: int, run_name: str, device):
        super().__init__()
        self.args = args
        self.device = device
        self.action_dim = action_dim
        self.action_horizon = int(getattr(args, "action_horizon", 8))
        self.logger = _build_logger(args, run_name)

    def reset_rollout(self, num_envs=None, done_indices=None):  # noqa: D401 - hook
        """Clear per-episode state. Default: nothing to clear."""

    @torch.no_grad()
    def predict_action_chunk(self, obs) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


class ZeroActionAgent(_BaseEvalAgent):
    """Stub agent that emits zero actions.

    Used to smoke-test the rgb env + ``evaluate_policy`` harness end-to-end before the
    GR00T server is wired (Phase 1, Task 1.3). Under delta control modes a zero action is a
    no-op, so success rate is ~0 — exactly what we want to prove the plumbing, not the policy.
    """

    @torch.no_grad()
    def predict_action_chunk(self, obs) -> torch.Tensor:
        return torch.zeros((1, self.action_horizon, self.action_dim), dtype=torch.float32, device=self.device)


class GrootAgent(_BaseEvalAgent):
    """Loads a finetuned GR00T policy in-process and drives it from ManiSkill observations.

    Constructs a :class:`gr00t.policy.gr00t_policy.Gr00tPolicy` from ``--model_path`` (a
    finetuned checkpoint dir). ``get_action(obs) -> (action, info)`` is called directly; the
    modality/state/action layout is taken from our adapter config so no server is involved.

    Args:
        args: parsed :class:`train.train_configs.groot_config.Args`.
        action_dim: flat env action dimension (sum of action-key widths).
        run_name: run name for the logger / video dir.
        device: torch device for the returned action tensors.
        prompt: language instruction sent to GR00T each step.
        video_key_aliases: GR00T video key -> ManiSkill camera aliases. Defaults cover
            head/base/front + wrist; override to match meta/modality.json.
        state_layout: ordered (name, start, end) splits of the flat proprio vector into the
            GR00T state modality keys. Defaults to the Franka gripper layout.
        action_keys: ordered GR00T action keys; the flat env action is their concatenation
            in this order. Defaults to the Franka gripper keys.
        state_fn: callable(obs) -> flat float32 state vector. Defaults to the Franka 8-dim
            proprio (7 arm joints + gripper width); pass raw-qpos for dexterous hands.
    """

    def __init__(
        self,
        args,
        *,
        action_dim: int,
        run_name: str,
        device,
        prompt: Optional[str] = None,
        video_keys: Optional[Sequence[str]] = None,
        video_key_aliases: Optional[dict] = None,
        state_layout: Optional[Sequence[Tuple[str, int, int]]] = None,
        action_keys: Optional[Sequence[str]] = None,
        language_key: str = "annotation.human.task_description",
        video_T: int = 1,
        state_T: int = 1,
        state_fn: Optional[Callable] = None,
    ):
        super().__init__(args, action_dim=action_dim, run_name=run_name, device=device)
        self.prompt = prompt or getattr(args, "exp_name", None) or "task"
        self._video_key_aliases = dict(video_key_aliases or DEFAULT_VIDEO_KEY_ALIASES)
        self._state_layout = tuple(state_layout or GRIPPER_STATE_LAYOUT)
        self._action_keys = tuple(action_keys or GRIPPER_ACTION_KEYS)
        self._state_fn = state_fn or (lambda obs: extract_franka_proprio(obs))

        # Load the GR00T policy IN-PROCESS. With the unified env (gr00t + ManiSkill in one
        # venv) there is no socket server; Gr00tPolicy.get_action(obs)->(action, info) has the
        # same contract the old ZeroMQ client exposed, so the rest of this class is unchanged.
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        model_path = getattr(args, "model_path", None) or getattr(args, "checkpoint", None)
        if not model_path:
            raise ValueError(
                "GrootAgent needs a finetuned checkpoint dir via --model_path "
                "(e.g. runs/groot_bottlecap_ft/checkpoint-10000)."
            )
        _ensure_local_backbone(model_path)  # keep in-process load token-free (was serve.sh's job)
        self.policy = Gr00tPolicy(
            embodiment_tag=getattr(args, "embodiment_tag", "new_embodiment"),
            model_path=str(model_path),
            device=str(device),
        )
        # Task-adaptive: if action keys weren't given, take them from the checkpoint's action
        # modality (gripper tasks -> ["single_arm","gripper"]; ee-control -> ["eef","gripper"]).
        if action_keys is None:
            mc = getattr(self.policy, "modality_configs", None)
            if mc and mc.get("action") is not None and getattr(mc["action"], "modality_keys", None):
                self._action_keys = tuple(mc["action"].modality_keys)
        # Same for the video keys: a single-arm checkpoint wants ["head", "wrist"], a dual-arm
        # one ["head", "left_wrist", "right_wrist"]. Hard-coding the single-arm pair made
        # OpenPackage's evaluation fail GR00T's own check_observation (2026-09-17).
        if video_keys is None:
            mc = getattr(self.policy, "modality_configs", None)
            if mc and mc.get("video") is not None and getattr(mc["video"], "modality_keys", None):
                video_keys = list(mc["video"].modality_keys)
        self._video_keys = list(video_keys or ["head", "wrist"])
        self._state_keys = [name for name, _, _ in self._state_layout]
        self._language_keys = [language_key]
        self._video_T = int(video_T)
        self._state_T = int(state_T)

        self._validate_layout()

    def _validate_layout(self):
        layout_names = [name for name, _, _ in self._state_layout]
        if set(layout_names) != set(self._state_keys):
            raise ValueError(
                "state_layout names {} do not match server state modality keys {}".format(
                    layout_names, self._state_keys
                )
            )
        for name in self._video_keys:
            if name not in self._video_key_aliases:
                raise ValueError(
                    f"No camera aliases configured for GR00T video key '{name}'. "
                    f"Add it to video_key_aliases (have: {sorted(self._video_key_aliases)})."
                )

    def set_instruction(self, prompt: str):
        """Swap the language instruction sent to GR00T (evaluation varies it per episode)."""
        if not prompt:
            raise ValueError("GR00T needs a non-empty language instruction.")
        self.prompt = prompt

    def reset_rollout(self, num_envs=None, done_indices=None):
        try:
            self.policy.reset()
        except Exception:  # reset is optional; ignore if the server doesn't implement it
            pass

    def _build_observation(self, obs) -> dict:
        # Video: (1, T, H, W, 3) uint8 per GR00T video key. T frames = current frame tiled.
        video = {}
        for name in self._video_keys:
            rgb = extract_camera_rgb(obs, self._video_key_aliases[name])  # (H, W, 3) uint8
            video[name] = np.repeat(rgb[None, None], self._video_T, axis=1).astype(np.uint8)

        # State: split the flat proprio vector into the GR00T state keys, (1, T, D) float32.
        flat_state = np.asarray(self._state_fn(obs), dtype=np.float32).reshape(-1)
        state = {}
        for name, start, end in self._state_layout:
            seg = flat_state[start:end][None, None]  # (1, 1, D)
            state[name] = np.repeat(seg, self._state_T, axis=1).astype(np.float32)

        # Language: one prompt per (B=1, T=1).
        language = {key: [[self.prompt]] for key in self._language_keys}

        return {"video": video, "state": state, "language": language}

    @torch.no_grad()
    def predict_action_chunk(self, obs) -> torch.Tensor:
        groot_obs = self._build_observation(obs)
        action_dict, _ = self.policy.get_action(groot_obs)
        # Concatenate the action keys in our configured order -> (T, action_dim).
        segments = [np.asarray(action_dict[key], dtype=np.float32)[0] for key in self._action_keys]
        chunk = np.concatenate(segments, axis=-1)  # (T, action_dim)
        chunk = chunk[: self.action_horizon]
        return torch.as_tensor(chunk, dtype=torch.float32, device=self.device).unsqueeze(0)


# ---------------------------------------------------------------------------------
# Dexterous-hand long-horizon evaluation: one agent per checkpoint, laid out for its arm
# (scripts/dex/eval_long_chain.py, scripts/dex/eval_family_on_subtask.py)
# ---------------------------------------------------------------------------------


def dex_state_fn(arm: str, env):
    """The policy's proprio: its own arm's 29 joints out of the long robot's joint vector."""
    from utils.dex_long_tasks import DUAL, DUAL_SLOTS, SINGLE_ACTION_DIM
    from utils.image_observation import extract_controller_qpos

    def fn(obs, _env=env):
        q = np.asarray(extract_controller_qpos(obs, env=_env), dtype=np.float32).reshape(-1)
        if arm == DUAL or q.size == SINGLE_ACTION_DIM:
            return q
        s = DUAL_SLOTS[arm]
        return np.concatenate([q[s["arm"][0]:s["arm"][1]], q[s["hand"][0]:s["hand"][1]]])
    return fn


def dex_video_aliases(arm: str):
    """Point the policy's generic 'wrist' key at the acting arm's camera."""
    aliases = dict(DEFAULT_VIDEO_KEY_ALIASES)
    if arm in ("left", "right"):
        aliases["wrist"] = LEFT_WRIST_CAMERA_ALIASES if arm == "left" else RIGHT_WRIST_CAMERA_ALIASES
    return aliases


def build_dex_agent(model_path: str, modality_env: str, arm: str, env, device, action_horizon: int,
                    run_name: str, action_dim: int) -> "GrootAgent":
    """A GrootAgent for one checkpoint, laid out for the arm it will drive.

    Gr00tPolicy takes the layout from the checkpoint's own saved processor
    (processor.get_modality_configs()), not from the process-wide registry, so several
    checkpoints with different layouts load side by side without registering anything.
    (Registering per checkpoint trips GR00T's "already registered" assert on the second one.)
    """
    from types import SimpleNamespace
    from utils.groot_utils.modality_configs import MODALITY_JSON

    state = MODALITY_JSON[modality_env]["state"]
    state_layout = tuple((k, v["start"], v["end"]) for k, v in state.items())
    args = SimpleNamespace(model_path=model_path, embodiment_tag="NEW_EMBODIMENT",
                           action_horizon=action_horizon, track=False, exp_name=run_name, seed=0)
    agent = GrootAgent(args, action_dim=action_dim, run_name=run_name, device=device,
                       prompt="task", video_key_aliases=dex_video_aliases(arm),
                       state_layout=state_layout, state_fn=dex_state_fn(arm, env)).to(device)
    agent.eval()
    return agent
