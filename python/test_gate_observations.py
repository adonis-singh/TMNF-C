"""Optional gate observations preserve legacy physics and terminal ownership."""
import numpy as np
import pytest
import torch
from tmnf_rl.env import TmnfVectorEnv
from tmnf_rl.spaces import observation_tensor

@pytest.mark.parametrize('mode', ['discrete', 'analog'])
def test_gate_step_reset_restore(mode):
    env = TmnfVectorEnv(3, thread_count=2, action_space=mode,
                        gate_observations=True, horizon_ticks=800)
    control = TmnfVectorEnv(3, thread_count=2, action_space=mode,
                            horizon_ticks=800)
    try:
        obs, _ = env.reset()
        control.reset()
        assert obs['gates'].shape == (3, 8, 19)
        assert env.observation_space.contains(obs)
        original = obs['gates'].copy()
        snapshots = env.capture()
        endings = 0
        for _ in range(220):
            actions = (np.full(3, 4, dtype=np.int32) if mode == 'discrete' else
                       dict(steer=np.zeros(3, dtype=np.float32),
                            gas=np.ones(3, dtype=np.uint8),
                            brake=np.zeros(3, dtype=np.uint8)))
            env.step(actions)
            control.step(actions)
            np.testing.assert_array_equal(env.raw_results, control.raw_results)
            np.testing.assert_array_equal(
                observation_tensor(obs, torch.device('cpu')).numpy(),
                env.policy_observations,
            )
            assert np.isfinite(obs['gates']).all()
            assert np.all(env._final_gate_buffer[~env.final_observation_valid] == 0)
            endings += int(env.final_observation_valid.sum())
            for i in np.flatnonzero(env.final_observation_valid):
                np.testing.assert_array_equal(env._final_obs_objects[i]['gates'],
                                              env.final_observations['gates'][i])
        assert endings > 0
        env.restore([0, 1, 2], snapshots)
        np.testing.assert_array_equal(obs['gates'], original)
        assert np.all(env._final_gate_buffer == 0)
        env.reset()
        np.testing.assert_array_equal(obs['gates'], original)
    finally:
        env.close()
        control.close()
