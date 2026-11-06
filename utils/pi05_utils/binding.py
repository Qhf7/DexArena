"""Write the robot/controller binding that PI0.5 data preparation, fine-tuning
and evaluation check the live robot against.

The binding names the robot (``uid``), the ordered joints of the articulation
(``qpos_joint_names``), the ordered joints the controller groups drive
(``action_joint_names``) and, per controller group, its joint names and
controller class. It is read from the registered task environment, which is
the same robot and controller the released demonstrations were recorded with.

    python -m utils.pi05_utils.binding \\
        --env-id CabinetPouringOpenCabinet-Sharpa-v1 \\
        --output /path/to/opencabinet-binding.json

``--task`` adds task names to the binding's ``tasks`` list on top of the
registered short name of ``--env-id`` (a data contract prepared under another
task name, such as a published package, must find its name there).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

STATUS = "LIVE_ROBOT_CONTROLLER_BINDING_PASS"


def live_binding(env_id: str, extra_tasks=()) -> dict:
    """Build the binding dict from a freshly constructed task environment."""
    import gymnasium as gym

    import tasks  # noqa: F401
    from utils.dex_scene_eval import registered_task_settings
    from utils.dex_tasks import resolve_dex_task

    settings = registered_task_settings(env_id)
    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="state",
        sim_backend="physx_cpu",
        **settings,
    )
    try:
        env.reset(seed=0)
        agent = env.unwrapped.agent
        controllers = agent.controller.controllers
        qpos_joint_names = [joint.name for joint in agent.robot.get_active_joints()]
        groups = {
            name: dict(
                joint_names=[joint.name for joint in controller.joints],
                controller=type(controller).__name__,
            )
            for name, controller in controllers.items()
        }
        action_joint_names = [name for group in groups.values() for name in group["joint_names"]]
        if set(qpos_joint_names) != set(action_joint_names) or len(set(qpos_joint_names)) != len(
            qpos_joint_names
        ):
            raise ValueError("Controller groups do not cover the articulation's active joints exactly once")
        for name, controller in controllers.items():
            config = controller.config
            if (
                getattr(config, "normalize_action", False)
                or getattr(config, "use_delta", False)
                or getattr(config, "use_target", False)
            ):
                raise ValueError(f"Controller group {name!r} is not raw absolute joint targets")
        task_names = [resolve_dex_task(env_id).short_name, *extra_tasks]
        return dict(
            status=STATUS,
            env_id=env_id,
            uid=agent.uid,
            tasks=list(dict.fromkeys(task_names)),
            control_mode=settings.get("control_mode"),
            qpos_joint_names=qpos_joint_names,
            action_joint_names=action_joint_names,
            groups=groups,
        )
    finally:
        env.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--task", action="append", default=[], help="additional task name accepted by this binding")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    binding = live_binding(args.env_id, args.task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(binding, indent=2) + "\n")
    print(
        f"[binding] {args.output}: uid={binding['uid']} joints={len(binding['qpos_joint_names'])} "
        f"groups={list(binding['groups'])} tasks={binding['tasks']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
