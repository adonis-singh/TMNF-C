"""Training uses the CPU evaluator's resolved, independent learning budget."""
from pathlib import Path

import numpy as np
import torch

from tmnf_rl.agents.ppo import make_env, make_train_env
from tmnf_rl.config import config_from_dict


def test_cuda_training_inherits_resolved_budget():
    root = Path(__file__).resolve().parents[1]
    config = config_from_dict(dict(track='coast-a1', num_envs=33, thread_count=2,
        env_device='cuda', budget_reference_speed=25.0, action_repeat=5))
    cpu = make_env(config, 33, root/'build/libtmnf_physics.so')
    gpu = None
    try:
        gpu = make_train_env(config, root/'build/libtmnf_cuda.so', cpu, torch.device('cuda:0'))
        assert cpu.max_race_ticks == cpu.horizon_ticks == 3600
        assert gpu.max_race_ticks == gpu.horizon_ticks == 3600
        cpu.reset(); gpu.reset()
        for step in range(120):
            actions = (np.arange(33)+step//5) % 12
            cpu.step(actions); gpu.step(actions)
            np.testing.assert_array_equal(cpu.raw_results, gpu.device_results.cpu().numpy())
            np.testing.assert_array_equal(cpu.capture(), gpu.capture())
            if step % 31 == 0:
                snapshot = cpu.capture([0, 32])
                cpu.restore([32, 1], snapshot); gpu.restore([32, 1], snapshot)
    finally:
        if gpu is not None:
            gpu.close()
        cpu.close()
