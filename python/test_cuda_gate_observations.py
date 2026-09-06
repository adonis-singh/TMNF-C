"""Device gate buffers match CPU through steps and indexed restores."""
from pathlib import Path
import numpy as np
import pytest
import torch
from tmnf_rl.env import TmnfVectorEnv
from tmnf_rl.cuda_env import TmnfCudaVectorEnv

@pytest.mark.parametrize('mode', ['discrete', 'analog'])
def test_device_gates(mode):
    root = Path(__file__).resolve().parents[1]
    cpu = TmnfVectorEnv(33, thread_count=3, action_space=mode,
                        action_repeat=5, horizon_ticks=800, gate_observations=True)
    gpu = None
    try:
        gpu = TmnfCudaVectorEnv(33, action_space=mode, action_repeat=5,
            max_race_ticks=cpu.max_race_ticks, horizon_ticks=cpu.horizon_ticks,
            gate_observations=True, library_path=root/'build/libtmnf_cuda.so',
            device=torch.device('cuda:0'))
        cpu.reset(); gpu.reset()
        def compare():
            np.testing.assert_array_equal(cpu.policy_observations, gpu.device_observations.cpu().numpy())
            np.testing.assert_array_equal(cpu.observations['gates'], gpu.device_gates.cpu().numpy())
            np.testing.assert_array_equal(cpu.observations['gate_counts'], gpu.device_gate_counts.cpu().numpy())
        compare()
        endings = 0
        for step in range(200):
            actions = ((np.arange(33)+step)%12 if mode == 'discrete' else
                dict(steer=((np.arange(33)+step)%3-1).astype(np.float32)*.5,
                     gas=np.ones(33,dtype=np.uint8),brake=np.zeros(33,dtype=np.uint8)))
            cpu.step(actions); gpu.step(actions)
            compare()
            np.testing.assert_array_equal(cpu.final_observations['gates'], gpu.device_final_gates.cpu().numpy())
            np.testing.assert_array_equal(cpu.final_observations['gate_counts'], gpu.device_final_gate_counts.cpu().numpy())
            np.testing.assert_array_equal(cpu.policy_final_observations, gpu.device_final_observations.cpu().numpy())
            np.testing.assert_array_equal(cpu.raw_results, gpu.device_results.cpu().numpy())
            endings += int(cpu.final_observation_valid.sum())
            if step % 39 == 0:
                snaps = cpu.capture([0, 3, 32])
                cpu.restore([32, 0, 3], snaps); gpu.restore([32, 0, 3], snaps)
                compare()
                np.testing.assert_array_equal(cpu.raw_results, gpu.device_results.cpu().numpy())
                np.testing.assert_array_equal(cpu.policy_final_observations, gpu.device_final_observations.cpu().numpy())
                np.testing.assert_array_equal(cpu.final_observation_valid, gpu.final_observation_valid)
        assert endings > 0
    finally:
        if gpu is not None: gpu.close()
        cpu.close()


@pytest.mark.parametrize('gates', [False, True])
def test_restore_clears_only_selected_completed_transition(gates):
    root = Path(__file__).resolve().parents[1]
    cpu = TmnfVectorEnv(3, action_space='analog', action_repeat=5,
                       horizon_ticks=800, gate_observations=gates)
    gpu = None
    try:
        gpu = TmnfCudaVectorEnv(3, action_space='analog', action_repeat=5,
            max_race_ticks=cpu.max_race_ticks, horizon_ticks=800,
            gate_observations=gates, library_path=root/'build/libtmnf_cuda.so',
            device=torch.device('cuda:0'))
        cpu.reset(); gpu.reset()
        snapshot = cpu.capture([0])
        actions = dict(steer=np.zeros(3, dtype=np.float32),
                       gas=np.zeros(3, dtype=np.uint8), brake=np.zeros(3, dtype=np.uint8))
        for _ in range(160):
            cpu.step(actions); gpu.step(actions)
            if cpu.final_observation_valid.all():
                break
        assert cpu.final_observation_valid.all()
        cpu.restore([1], snapshot); gpu.restore([1], snapshot)
        np.testing.assert_array_equal(cpu.final_observation_valid, [True, False, True])
        np.testing.assert_array_equal(cpu.final_observation_valid, gpu.final_observation_valid)
        np.testing.assert_array_equal(cpu.raw_results, gpu.device_results.cpu().numpy())
        np.testing.assert_array_equal(cpu.policy_observations, gpu.device_observations.cpu().numpy())
        np.testing.assert_array_equal(cpu.policy_final_observations, gpu.device_final_observations.cpu().numpy())
        np.testing.assert_array_equal(cpu.policy_transitions, gpu.device_transitions.cpu().numpy())
        cpu.reset(); gpu.reset()
        cpu.restore([1], snapshot); gpu.restore([1], snapshot)
        np.testing.assert_array_equal(cpu.final_observation_valid, gpu.final_observation_valid)
        assert not gpu.final_observation_valid.any()
        np.testing.assert_array_equal(cpu.policy_observations, gpu.device_observations.cpu().numpy())
        np.testing.assert_array_equal(cpu.policy_final_observations, gpu.device_final_observations.cpu().numpy())
    finally:
        if gpu is not None:
            gpu.close()
        cpu.close()
