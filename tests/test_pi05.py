"""CPU regression coverage for the sole PI0.5 baseline (no model/GPU required)."""

import ast
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from algo.pi05 import expand_projection
from train import train_pi05 as entry
from train.train_configs.pi05_config import TrainingContract, parse_args
from utils.dex_tasks import dataset_episode_instruction, episode_instruction
from utils.pi05_utils.checkpoint import EvalJournal, source_identity
from utils.pi05_utils.data import (
    JointOutputs,
    evaluation_prompt,
    reorder_joints,
    training_prompt,
)
from utils.pi05_utils.eval import NativeJointAdapter, native_flags
from utils.pi05_utils.runtime import AsyncJointQueue, read_frame, write_frame


@pytest.mark.parametrize("dual,physical,capacity,steps", [(False,29,32,30000),(True,58,58,60000)])
def test_training_contract(dual, physical, capacity, steps):
    contract = TrainingContract(dual_arm=dual)
    contract.validate()
    assert (contract.physical_dim, contract.action_dim, contract.steps) == (physical,capacity,steps)
    with pytest.raises(ValueError, match="contract changed"):
        TrainingContract(batch_size=16).validate()


@pytest.mark.parametrize("evaluate", [False, True])
def test_single_entry_dispatch(monkeypatch, evaluate):
    from utils.pi05_utils import eval as evaluation, training

    args = SimpleNamespace(evaluate=evaluate)
    calls = []
    monkeypatch.setattr(entry, "parse_args", lambda: args)
    monkeypatch.setattr(training, "finetune", lambda actual: calls.append(("train",actual)))
    monkeypatch.setattr(evaluation, "evaluate", lambda actual: calls.append(("eval",actual)))
    entry.main()
    assert calls == [("eval" if evaluate else "train",args)]


def test_help_needs_no_openpi_or_maniskill_runtime():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable,"train/train_pi05.py","--help"],
        cwd=root,capture_output=True,text=True,
    )
    assert result.returncode == 0,result.stderr
    assert "--evaluate" in result.stdout and "--policy-python" in result.stdout


@pytest.mark.parametrize("arguments", [[],["--evaluate"]])
def test_required_mode_specific_paths(monkeypatch, tmp_path, arguments):
    monkeypatch.setattr(sys,"argv",["pi05","--output-dir",str(tmp_path),*arguments])
    with pytest.raises(ValueError,match="Missing arguments"):
        parse_args()


def test_evaluation_cli_selects_the_new_model_run(monkeypatch, tmp_path):
    monkeypatch.setattr(sys,"argv",["pi05","--output-dir",str(tmp_path),"--evaluate",
                                   "--model-path",str(tmp_path/"model"),"--scene-background","s2"])
    args = parse_args()
    assert args.evaluate and args.scene_background == "s2"
    assert args.model_path == tmp_path/"model" and args.dataset_path is None


def test_prompts_share_the_official_instruction_protocol():
    env = "OfficeDeskClickMouse-Sharpa-v1"
    assert training_prompt(env,3) == dataset_episode_instruction(env,3)
    assert evaluation_prompt(env,10001) == episode_instruction(env,10001)


def test_named_joint_order_and_validation():
    np.testing.assert_array_equal(reorder_joints([1,2], ["arm","hand"], ["hand","arm"]), [2,1])
    with pytest.raises(ValueError, match="named joint"):
        reorder_joints([1,2], ["arm","arm"], ["hand","arm"])
    with pytest.raises(ValueError, match="Nonfinite"):
        reorder_joints([np.nan,2], ["arm","hand"], ["arm","hand"])


@pytest.mark.parametrize("physical,capacity", [(29,32),(58,58)])
def test_physical_actions_strip_only_model_padding(physical, capacity):
    output = JointOutputs(physical)({"actions":np.ones((30,capacity))})
    assert output["actions"].shape == (30,physical)
    with pytest.raises(ValueError, match="dimensions"):
        JointOutputs(physical)({"actions":np.ones((29,capacity))})


def queue():
    return AsyncJointQueue(29, -np.ones(29)*10, np.ones(29)*10, np.zeros(29))


def test_replan_threshold_and_observation_time_alignment():
    q = queue()
    first = q.request()
    assert q.accept(first,np.ones((30,29)))
    for _ in range(6):q.take()
    assert q.remaining == 24 and q.request() is None
    q.take()
    request = q.request()
    assert request.observation_step == 7
    for _ in range(3):q.take()
    assert q.accept(request,np.full((30,29),2.0))
    assert min(q.queue) == 10 and max(q.queue) == 36
    assert 1 < q.take()[0] < 2  # Absolute-target overlap blending.


def test_reset_discards_old_predictions():
    q = queue()
    old = q.request()
    q.reset(np.full(29,3.0))
    assert not q.accept(old,np.ones((30,29)))
    np.testing.assert_array_equal(q.take(),np.full(29,3.0))
    assert q.starvation_steps == 1


def test_expired_chunks_are_not_shifted_to_arrival_time():
    q = queue()
    request = q.request()
    for _ in range(31):q.take()
    assert not q.accept(request,np.ones((30,29)))
    assert not q.queue and q.pending is None


def test_ipc_roundtrip():
    stream = BytesIO()
    original = {"state":np.arange(29,dtype=np.float32),"rgb":np.zeros((2,2,3),dtype=np.uint8)}
    write_frame(stream, original)
    stream.seek(0)
    restored = read_frame(stream)
    for k in original:np.testing.assert_array_equal(original[k],restored[k])


def test_native_success_is_not_an_integer_stage_flag():
    assert native_flags({"success":np.array([True])},False,True) == (True,False,True)
    with pytest.raises(ValueError, match="boolean"):
        native_flags({"success":1},False,True)


def test_adapter_rebinds_the_live_robot_after_reset():
    names = [f"joint_{i}" for i in range(29)]
    joints = [SimpleNamespace(name=name) for name in names]

    class AbsoluteController:
        def __init__(self):
            self.joints = joints
            self.config = SimpleNamespace(normalize_action=False,use_delta=False,use_target=False)
            self.action_space = SimpleNamespace(low=-np.ones(29),high=np.ones(29))

    def agent(value):
        robot = SimpleNamespace(get_active_joints=lambda:joints,get_qpos=lambda:np.full((1,29),value))
        return SimpleNamespace(uid="test",robot=robot,
                               controller=SimpleNamespace(controllers={"all":AbsoluteController()}))

    first, replacement = agent(0.0), agent(0.5)
    live = [first]
    prepared = dict(task="test",state_names=names,action_names=names,controller_names=names,
                    camera_keys={"observation.images.base_0_rgb":[2,2,3]})
    binding = dict(status="LIVE_ROBOT_CONTROLLER_BINDING_PASS",tasks=["test"],uid="test",
                   qpos_joint_names=names,action_joint_names=names,
                   groups={"all":{"joint_names":names,"controller":"AbsoluteController"}})
    adapter = NativeJointAdapter(first,prepared,binding,agent_provider=lambda:live[0])
    np.testing.assert_array_equal(adapter.state(),np.zeros(29))
    live[0] = replacement
    np.testing.assert_array_equal(adapter.state(),np.full(29,0.5))
    assert adapter.agent is replacement and adapter.rebind_count == 1
    replacement.controller.controllers["all"].config.use_delta = True
    live[0] = agent(0.2)
    live[0].controller.controllers["all"].config.use_delta = True
    with pytest.raises(ValueError,match="absolute joint targets"):
        adapter.state()


def test_eval50_does_not_score_infrastructure_or_partial_runs(tmp_path):
    journal = EvalJournal(tmp_path/"eval", {"model":"test","scene":"s1"})
    journal.record({"seed":10001,"status":"infrastructure_error","success":None})
    assert len(journal.pending()) == 50 and journal.summary()["success_rate"] is None
    for seed in range(10001,10051):
        journal.record({"seed":seed,"status":"completed","success":seed%2==0})
    assert journal.summary() == {"completed_episodes":50,"successes":25,"status":"complete","success_rate":0.5}
    resumed = EvalJournal(tmp_path/"eval",journal.identity,resume=True)
    assert resumed.pending() == []
    with pytest.raises(ValueError,match="identity"):
        EvalJournal(tmp_path/"eval",{"model":"changed"},resume=True)
    with pytest.raises(FileExistsError):
        journal.record({"seed":10001,"status":"completed","success":True})


def test_dual_arm_projection_preserves_pretrained_channels():
    old = np.ones((32,4),dtype=np.float32)
    initialized = np.full((58,4),2,dtype=np.float32)
    result = expand_projection("action_in_proj/kernel",old,initialized)
    np.testing.assert_array_equal(result[:32],old)
    np.testing.assert_array_equal(result[32:],initialized[32:])
    with pytest.raises(ValueError,match="Unapproved"):
        expand_projection("other/kernel",old,initialized)


def test_no_legacy_pi05_implementation_or_imports_remain():
    root = Path(__file__).resolve().parents[1]
    removed = ["algo/pi05_grpo.py","train/train_pi05_sft.py","train/train_pi05_grpo.py",
               "train/eval_pi05.py","utils/pi05_utils/pi05_artifacts.py"]
    assert all(not (root/p).exists() for p in removed)
    forbidden = ("algo.pi05_grpo","utils.pi05_utils.pi05_artifacts","utils.pi05_utils.pi05_eval_artifacts",
                 "utils.pi05_utils.pi05_policy_artifact","utils.pi05_utils.pi05_sft_artifacts",
                 "train.eval_pi05","train.train_pi05_sft","train.train_pi05_grpo",
                 "train.train_configs.pi05_sft_config","train.train_configs.pi05_grpo_config","train.train_configs.pi05_eval_config")
    for directory in ["algo","train","utils","tests"]:
        for path in (root/directory).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
                imports = [alias.name for alias in node.names] if isinstance(node,ast.Import) else [node.module or ""] if isinstance(node,ast.ImportFrom) else []
                assert not any(name==prefix or name.startswith(prefix+".") for name in imports for prefix in forbidden),path
    assert len(source_identity()) == 64
