"""Versioned surface identity, checkpoint loading and bounded feature layout."""
from pathlib import Path
import re
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import pytest
import torch

from tmnf_rl import encoder
from tmnf_rl.agents.ppo import Agent
from tmnf_rl.agents.td3 import OffPolicyAgent
from tmnf_rl.config import TrainConfig, ConfigError, config_from_dict
from tmnf_rl.evaluate import load_policy


def observations():
    obs = torch.zeros(35, 81)
    obs[:, 6] = 1
    obs[:, 29:33] = torch.arange(-2, 33).reshape(-1, 1)
    return obs


def test_all_native_materials_are_distinct_with_explicit_unknown():
    header = (Path(__file__).resolve().parents[1] / 'src/surface_material.h').read_text()
    native_count = int(re.search(r'TMNF_SURFACE_MATERIAL_COUNT\s*=\s*(\d+)', header)[1])
    assert native_count == encoder.SURFACE_MATERIAL_COUNT
    obs = observations()
    ids = encoder.encode(obs, 2)['material'][:, 0]
    assert ids[2:2 + native_count].tolist() == list(range(native_count))
    assert ids[:2].tolist() == [native_count] * 2
    assert ids[2 + native_count:].tolist() == [native_count] * 2
    assert encoder.feature_names(1) == encoder.FEATURE_NAMES
    assert len(encoder.feature_names(2)) == encoder.flat_features(2) == 249
    full = encoder.encode_flat(obs, 2)
    assert full.shape == (35, 249)
    assert torch.unique(full[2:2 + native_count], dim=0).shape[0] == native_count
    assert encoder.encode(obs)['material'][19:24, 0].tolist() == [17] * 5


@pytest.mark.parametrize('version', [1, 2])
@pytest.mark.parametrize('kind', ['mlp', 'transformer_s', 'td3'])
def test_public_loader_preserves_version_and_policy_outputs(tmp_path, monkeypatch, version, kind):
    # Exercise both transformer embeddings and inference without a compiler
    # startup in a CPU unit test; compiled graph checks are a separate GPU gate.
    monkeypatch.setattr(torch, 'compile', lambda module: module)
    config = TrainConfig(encoder_version=version, algorithm='td3' if kind == 'td3' else 'ppo',
                         arch='mlp' if kind == 'td3' else kind, hidden_size=8,
                         action_space='analog', td3_critic_width=8, td3_bins=11)
    model = (OffPolicyAgent(8, 8, 11, -10, 20, .2, .1, version) if kind == 'td3' else
             Agent(kind, 8, 'analog', encoder_version=version))
    model.eval()
    obs = observations()
    with torch.no_grad(): expected = model.get_deterministic_action(obs)
    path = tmp_path / 'policy.pt'
    stored_config = config.to_dict()
    if version == 1:
        del stored_config['encoder_version']  # Historical policies default to v1.
    torch.save({'agent': model.state_dict(), 'config': stored_config,
                'summary': {'return_support': [-10, 20]}}, path)
    loaded, loaded_config, _ = load_policy(path, torch.device('cpu'))
    assert loaded_config['encoder_version'] == version
    with torch.no_grad(): assert torch.equal(loaded.get_deterministic_action(obs), expected)
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())
    # Both actor and critic paths support backward on the expanded features.
    if kind == 'td3':
        steer = model.actor(obs)
        loss = model.critics[0](obs, steer).square().mean()
    else:
        hidden = model.hidden(obs)
        loss = model.value(hidden, obs).square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_unknown_version_rejected():
    with pytest.raises(ConfigError, match='encoder_version'):
        config_from_dict({'encoder_version': 99})
    with pytest.raises(ValueError, match='version'):
        encoder.FlatEncoder(version=99)


def test_resume_rejects_encoder_switch_before_loading_weights(monkeypatch, tmp_path):
    from tmnf_rl.agents.ppo import PPOTrainer, CHECKPOINT_FORMAT, CHECKPOINT_VERSION
    trainer = object.__new__(PPOTrainer)
    trainer.config = TrainConfig(encoder_version=2)
    trainer.physics_sha256 = 'unchanged'
    trainer.run_id = 'run'
    legacy = TrainConfig().to_dict()
    legacy.pop('encoder_version')
    state = {'format': CHECKPOINT_FORMAT, 'version': CHECKPOINT_VERSION,
             'physics_sha256': 'unchanged', 'run_id': 'run', 'config': legacy}
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: state)
    with pytest.raises(RuntimeError, match='encoder_version'):
        trainer.setup_resume(tmp_path / 'legacy.pt')
