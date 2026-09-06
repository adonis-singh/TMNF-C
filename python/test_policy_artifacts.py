"""Evaluated weights survive later training and load in the public evaluator."""
import hashlib
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import pytest
import torch

from tmnf_rl.agents.ppo import Agent
from tmnf_rl.agents.td3 import OffPolicyAgent
from tmnf_rl.config import TrainConfig
from tmnf_rl.evaluate import load_policy
from tmnf_rl.policy_artifacts import retain_policy


@pytest.mark.parametrize("algorithm", ["ppo", "td3"])
@pytest.mark.parametrize("encoder_version", [1, 2, 3])
def test_retained_policy_loads_exact_evaluated_weights(tmp_path, algorithm, encoder_version):
    config = TrainConfig(algorithm=algorithm, action_space="analog", hidden_size=8,
                         td3_critic_width=8, td3_bins=11, encoder_version=encoder_version)
    agent = (Agent("mlp", 8, "analog", encoder_version=encoder_version) if algorithm == "ppo" else
             OffPolicyAgent(8, 8, 11, -10, 20, 0.2, 0.1, encoder_version=encoder_version))
    expected = {k: v.clone() for k, v in agent.state_dict().items()}
    payload = {"agent": agent.state_dict(), "config": config.to_dict(),
               "counters": {"update": 37}, "physics_sha256": "physics",
               "summary": {"return_support": [-10, 20]} if algorithm == "td3" else {}}
    rng = torch.get_rng_state().clone()
    reference = retain_policy(tmp_path, payload)
    assert torch.equal(rng, torch.get_rng_state())
    assert retain_policy(tmp_path, payload) == reference
    with torch.no_grad():
        for parameter in agent.parameters():
            parameter.add_(5)
    path = tmp_path / reference["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"]
    loaded, _, meta = load_policy(path, torch.device("cpu"))
    assert meta["checkpoint_update"] == 37
    assert all(torch.equal(loaded.state_dict()[k], v) for k, v in expected.items())
    newer = retain_policy(tmp_path, payload)
    assert newer["path"] != reference["path"]
    assert path.is_file()


def test_retention_refuses_corruption_and_cleans_failed_write(tmp_path, monkeypatch):
    payload = {"agent": {"weight": torch.ones(2)}, "counters": {"update": 1}}
    original = retain_policy(tmp_path, payload)
    path = tmp_path / original["path"]
    path.write_bytes(b"damaged")
    with pytest.raises(RuntimeError, match="corrupt"):
        retain_policy(tmp_path, payload)
    assert path.read_bytes() == b"damaged"
    payload["counters"]["update"] = 2
    def fail_sync(fd):
        raise OSError("disk failure")
    monkeypatch.setattr(os, "fsync", fail_sync)
    with pytest.raises(OSError, match="disk failure"):
        retain_policy(tmp_path, payload)
    assert list(path.parent.glob("*.pt")) == [path]
    assert not list(path.parent.glob("*.tmp"))
