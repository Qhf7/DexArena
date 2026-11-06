"""Native ManiSkill scenes and asynchronous PI0.5 Eval50 orchestration."""

import json
import os
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np

from utils.pi05_utils.checkpoint import (
    EvalJournal,
    digest,
    read_run,
    source_identity,
)
from utils.pi05_utils.data import evaluation_prompt, reorder_joints, sha
from utils.pi05_utils.runtime import AsyncController, PipeInference, Request


def host_array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def native_bool(value, name):
    array = host_array(value)
    if array.shape not in ((), (1,)) or array.dtype != np.dtype(bool):
        raise ValueError("Expected one native boolean: " + name)
    return bool(array.reshape(-1)[0])


def native_flags(info, terminated, truncated):
    if "success" not in info:
        raise ValueError("Native success predicate missing")
    return (
        native_bool(info["success"], "success"),
        native_bool(terminated, "terminated"),
        native_bool(truncated, "truncated"),
    )


class NativeJointAdapter:
    def __init__(self, agent, prepared, binding, *, tensor_factory=None, agent_provider=None):
        self.agent = agent
        self.prepared = prepared
        self.binding = binding
        self.agent_provider = agent_provider
        self._robot = agent.robot
        self._controller = agent.controller
        self.rebind_count = 0
        self.qnames = [j.name for j in agent.robot.get_active_joints()]
        controllers = agent.controller.controllers
        self.anames = [j.name for c in controllers.values() for j in c.joints]
        self.n = len(self.qnames)
        if (
            binding.get("status") != "LIVE_ROBOT_CONTROLLER_BINDING_PASS"
            or prepared["task"] not in binding["tasks"]
            or agent.uid != binding["uid"]
            or self.n not in (29, 58)
            or len(set(self.qnames)) != self.n
            or len(set(self.anames)) != self.n
            or set(self.qnames) != set(self.anames)
            or self.qnames != binding["qpos_joint_names"]
            or self.anames != binding["action_joint_names"]
            or self.qnames != prepared["state_names"]
            or self.anames != prepared["controller_names"]
            or self.anames != prepared["action_names"]
        ):
            raise ValueError("Live articulation/controller differs from independent named binding")
        lows = []
        highs = []
        if set(controllers) != set(binding["groups"]):
            raise ValueError("Controller groups changed")
        for name, c in controllers.items():
            expected = binding["groups"][name]
            if (
                [j.name for j in c.joints] != expected["joint_names"]
                or type(c).__name__ != expected["controller"]
                or c.config.normalize_action
                or getattr(c.config, "use_delta", False)
                or getattr(c.config, "use_target", False)
            ):
                raise ValueError("Controller must remain raw absolute joint targets")
            low = host_array(c.action_space.low)
            high = host_array(c.action_space.high)
            if low.shape != (len(c.joints),) or high.shape != low.shape:
                raise ValueError("Controller joint/action dimensions differ")
            lows.append(low)
            highs.append(high)
        self.lower = np.concatenate(lows)
        self.upper = np.concatenate(highs)
        if (
            not np.isfinite(self.lower).all()
            or not np.isfinite(self.upper).all()
            or np.any(self.lower > self.upper)
        ):
            raise ValueError("Finite absolute-joint action limits required")
        cameras = prepared["camera_keys"]
        self.camera_keys = {
            k.removeprefix("observation.images.")
            .replace("base_camera", "base_0_rgb")
            .replace("left_wrist_camera", "left_wrist_0_rgb")
            .replace("right_wrist_camera", "right_wrist_0_rgb"): k
            for k in cameras
        }
        if "base_0_rgb" not in self.camera_keys or set(self.camera_keys) - {
            "base_0_rgb",
            "left_wrist_0_rgb",
            "right_wrist_0_rgb",
        }:
            raise ValueError("Unsupported training camera roles")
        self.tensor_factory = tensor_factory

    def _refresh_live_binding(self):
        if self.agent_provider is None:
            return
        live = self.agent_provider()
        if live is self.agent and live.robot is self._robot and live.controller is self._controller:
            return
        # Validate the replacement fully before publishing it. Existing async
        # queues retain bounds, so a reset must not silently change those bounds.
        checked = NativeJointAdapter(
            live, self.prepared, self.binding, tensor_factory=self.tensor_factory
        )
        if not np.array_equal(self.lower, checked.lower) or not np.array_equal(
            self.upper, checked.upper
        ):
            raise ValueError("Reset changed absolute-joint action limits")
        self.agent = live
        self._robot = live.robot
        self._controller = live.controller
        self.rebind_count += 1

    def state(self):
        self._refresh_live_binding()
        value = host_array(self.agent.robot.get_qpos())
        if value.shape != (1, self.n) or not np.isfinite(value).all():
            raise ValueError("Expected one full live articulation qpos")
        return value[0].astype(np.float32, copy=True)

    def observation(self, observation):
        result = {"state": self.state()}
        sensors = observation["sensor_data"]
        expected = {k.removeprefix("observation.images.") for k in self.prepared["camera_keys"]}
        if set(sensors) != expected:
            raise ValueError("Live camera keys differ from training")
        for key, shape in self.prepared["camera_keys"].items():
            value = host_array(sensors[key.removeprefix("observation.images.")]["rgb"])
            if value.dtype != np.uint8 or list(value.shape) != [1, *shape]:
                raise ValueError("Live RGB must match training batched HWC uint8")
            result[key] = value[0].copy()
        return result

    def initial_target(self, observation):
        # Holding current joint angles is not holding a qpos-order action vector.
        return reorder_joints(self.state(), self.qnames, self.anames)

    def action(self, values):
        self._refresh_live_binding()
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (self.n,) or not np.isfinite(values).all():
            raise ValueError("Expected controller-order finite absolute joint targets")
        # Queue already accounts for limit clipping; no second representation map.
        if self.tensor_factory is not None:
            return self.tensor_factory(values.copy()[None, :])
        import torch

        return torch.as_tensor(
            values[None, :], dtype=torch.float32, device=self.agent.robot.get_qpos().device
        )


def run_episode(
    *,
    env,
    env_id,
    seed,
    max_episode_steps,
    control_hz,
    joint_count,
    lower,
    upper,
    initial_target,
    observation_adapter,
    action_adapter,
    infer,
    native_flags,
    prompt_source=None,
    controller_factory=AsyncController,
    monotonic=time.monotonic,
    sleep=time.sleep,
):
    if type(seed) is not int or seed not in range(10001, 10051):
        raise ValueError("Official Eval50 seed required")
    if type(max_episode_steps) is not int or max_episode_steps <= 0 or not 0 < control_hz <= 1000:
        raise ValueError("Explicit native episode horizon and control frequency required")
    prompt = evaluation_prompt(env_id, seed, **({"source": prompt_source} if prompt_source else {}))
    result = dict(
        seed=seed,
        env_id=env_id,
        prompt=prompt,
        status="infrastructure_error",
        success=None,
        steps=0,
        error=None,
    )
    controller = None
    try:
        observation, _ = env.reset(seed=seed)
        # inference receives immutable per-control-step snapshots and request time.
        controller = controller_factory(
            joint_count,
            lower,
            upper,
            initial_target(observation),
            lambda obs, request: infer(obs, prompt, seed, request),
        )
        begin = monotonic()
        for step in range(max_episode_steps):
            deadline = begin + step / control_hz
            sleep(max(0.0, deadline - monotonic()))
            action = controller.tick(observation_adapter(observation))
            observation, _, terminated, truncated, info = env.step(action_adapter(action))
            result["steps"] = step + 1
            success, done, timeout = native_flags(info, terminated, truncated)
            if any(type(v) is not bool for v in (success, done, timeout)):
                raise ValueError("Native flag adapter must return three actual booleans")
            if success or done or timeout:
                result.update(
                    status="completed",
                    success=success,
                    error=None,
                    terminated=done,
                    truncated=timeout,
                )
                break
        else:
            result.update(
                status="completed", success=False, error=None, terminated=False, truncated=True
            )
    except Exception as exc:
        result.update(
            status="infrastructure_error", success=None, error=f"{type(exc).__name__}: {exc}"
        )
    finally:
        if controller is not None:
            result.update(
                starvation_steps=controller.queue.starvation_steps,
                clipped_values=controller.queue.clipped_values,
            )
            controller.close()
    return result


def evaluate50(
    *,
    env,
    adapter,
    prepared,
    infer,
    journal,
    final_certificate,
    scene_certificate,
    prompt_source=None,
    max_infra_attempts=2,
    episode_runner=run_episode,
):
    if max_infra_attempts not in (1, 2):
        raise ValueError("Bounded infrastructure retry required")
    if (
        final_certificate.get("status") != "FINAL_FULL_RESTORE_AND_MODEL_BINDING_VERIFIED"
        or final_certificate.get("optimizer_updates") != (60000 if adapter.n == 58 else 30000)
        or journal.identity.get("checkpoint_identity_sha256")
        != final_certificate["identity_sha256"]
    ):
        raise ValueError("Verified final checkpoint must match evaluation identity")
    if (
        scene_certificate.get("status") != "NATIVE_SCENE_CAMERA_CONTROLLER_ACCEPTED"
        or scene_certificate.get("task") != prepared["task"]
        or scene_certificate.get("env_id") != prepared["env_id"]
        or journal.identity.get("scene_identity_sha256") != scene_certificate.get("identity_sha256")
        or not scene_certificate.get("identity_sha256")
        or scene_certificate.get("control_hz") != prepared["fps"]
    ):
        raise ValueError("Verified scene/time/camera binding required")
    limit = scene_certificate["max_episode_steps"]
    hz = scene_certificate["control_hz"]
    if type(limit) is not int or limit <= 0:
        raise ValueError("Native episode limit required")
    pending = journal.pending()
    if not pending:
        return journal.summary()
    # Compile inference before any scored episode, then run_episode resets again.
    # Warmup never advances native physics and its action is not queued/executed.
    seed = pending[0]
    observation, _ = env.reset(seed=seed)
    prompt = evaluation_prompt(
        prepared["env_id"], seed, **({"source": prompt_source} if prompt_source else {})
    )
    infer(adapter.observation(observation), prompt, seed, Request(0, 1, 0))
    for seed in pending:
        for _ in range(max_infra_attempts):
            result = episode_runner(
                env=env,
                env_id=prepared["env_id"],
                seed=seed,
                max_episode_steps=limit,
                control_hz=hz,
                joint_count=adapter.n,
                lower=adapter.lower,
                upper=adapter.upper,
                initial_target=adapter.initial_target,
                observation_adapter=adapter.observation,
                action_adapter=adapter.action,
                infer=infer,
                native_flags=native_flags,
                prompt_source=prompt_source,
            )
            if result["seed"] != seed or result["env_id"] != prepared["env_id"]:
                raise ValueError("Episode result identity mismatch")
            journal.record(result)
            if result["status"] == "completed":
                break
        else:
            # Stop at persistent infrastructure errors; all attempts preserved,
            # same seed remains pending and must not count as policy failure.
            return dict(
                journal.summary(),
                blocked_seed=seed,
                blocked_reason="bounded_infrastructure_retries_exhausted",
            )
    return journal.summary()


BACKGROUNDS = {"s1": "robocasa_l3_s1", "s2": "robocasa_l3_s2", "s6": "robocasa_l3_s6"}


@contextmanager
def make_scene(prepared, binding, background):
    import gymnasium as gym
    from mani_skill.utils.registration import REGISTERED_ENVS

    import tasks  # noqa: F401
    from utils.dex_cameras import install_camera_profile
    from utils.dex_scene_eval import make_scene_eval_env, registered_task_settings

    env_id = prepared["env_id"]
    settings = registered_task_settings(env_id)
    kwargs = dict(
        num_envs=1,
        reconfiguration_freq=1,
        obs_mode="rgb",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        render_backend="sapien_cuda",
        **settings,
    )
    if background:
        env = make_scene_eval_env(env_id, scene_background=BACKGROUNDS[background], **kwargs)
    else:
        original = REGISTERED_ENVS[env_id].cls
        try:
            install_camera_profile(env_id, "robodex_fixed")
            env = gym.make(env_id, sensor_configs=dict(width=256, height=256), **kwargs)
        finally:
            REGISTERED_ENVS[env_id].cls = original
    try:
        observation, _ = env.reset(seed=10001)
        adapter = NativeJointAdapter(
            env.unwrapped.agent, prepared, binding, agent_provider=lambda: env.unwrapped.agent
        )
        adapter.observation(observation)
        hz = env.unwrapped.control_freq
        if hz != prepared["fps"]:
            raise ValueError("Native control rate differs from training data")
        from utils.dex_camera_rigs import describe, family_of

        scene = dict(
            task=prepared["task"],
            env_id=env_id,
            control_hz=hz,
            max_episode_steps=settings["max_episode_steps"],
            robot_uid=env.unwrapped.agent.uid,
            camera_rig=describe(family_of(env_id), env.unwrapped),
            background=getattr(env.unwrapped, "eval_background_manifest", None),
        )
        yield env, adapter, scene
    finally:
        env.close()


def evaluate(args):
    """Evaluate a final policy on the standard task or a shared background."""
    from filelock import FileLock

    REPO_ROOT = Path(__file__).resolve().parents[2]
    trained, prepared, binding = read_run(args.model_path)
    if args.env_id and args.env_id != prepared["env_id"]:
        raise ValueError("--env-id differs from the trained policy")
    from utils.pi05_utils.checkpoint import inspect_final

    final = inspect_final(args.model_path, trained, prepared)
    runtime_files = {
        path.relative_to(REPO_ROOT).as_posix(): sha(path)
        for directory in (REPO_ROOT / "tasks", REPO_ROOT / "utils")
        for path in sorted(directory.rglob("*.py"))
    }
    identity = dict(
        baseline=trained["baseline"],
        task=prepared["task"],
        env_id=prepared["env_id"],
        checkpoint_identity_sha256=final["identity_sha256"],
        evaluator_sha256=source_identity(),
        entry_sha256=sha(__file__),
        runtime_sha256=digest(runtime_files),
        prompt_sha256=trained["prompt_sha256"],
        background=args.scene_background,
        seeds=list(range(10001, 10051)),
    )
    output = args.output_dir.resolve()
    with FileLock(str(output) + ".lock", timeout=0):
        if args.resume:
            if not output.is_dir():
                raise ValueError("Resume output is missing")
        else:
            output.mkdir(exist_ok=False)
        with ExitStack() as stack:
            stderr = stack.enter_context((output / "policy-worker.log").open("a"))
            environment = dict(os.environ)
            environment["PYTHONPATH"] = os.pathsep.join(
                [str(REPO_ROOT)]
                + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
            )
            worker = PipeInference(
                [
                    args.policy_python,
                    "-m",
                    "utils.pi05_utils.runtime",
                    "--run",
                    str(args.model_path.resolve()),
                ],
                stderr=stderr,
                environment=environment,
            )
            stack.callback(worker.close)
            # Model allocation precedes simulator CUDA initialization.
            env, adapter, scene = stack.enter_context(
                make_scene(prepared, binding, args.scene_background)
            )
            scene_identity = digest(dict(scene=scene, runtime_sha256=identity["runtime_sha256"]))
            evaluation_identity = dict(identity, scene_identity_sha256=scene_identity)
            journal = EvalJournal(output / "episodes", evaluation_identity, resume=args.resume)
            if not args.resume:
                with (output / "scene.json").open("x") as stream:
                    json.dump(scene, stream, indent=2)
            certificate = dict(
                scene,
                identity_sha256=scene_identity,
                status="NATIVE_SCENE_CAMERA_CONTROLLER_ACCEPTED",
            )
            summary = evaluate50(
                env=env,
                adapter=adapter,
                prepared=prepared,
                infer=worker,
                journal=journal,
                final_certificate=worker.certificate,
                scene_certificate=certificate,
            )
            print(json.dumps(summary), flush=True)
