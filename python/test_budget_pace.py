"""A learning budget must not implicitly change progress reward scaling."""
from contextlib import ExitStack
import numpy as np
import pytest

from tmnf_rl.config import ConfigError, config_from_dict
from tmnf_rl.env import TmnfVectorEnv


@pytest.mark.parametrize('track', ['a01', 'a04', 'd01', 'coast-a1', 'rally-a1',
                                   'snow-a1', 'desert-a1', 'island-a1', 'bay-a1'])
def test_budget_pace_keeps_reward_and_transition_semantics(track):
    with ExitStack() as stack:
        def create(**kwargs):
            env = TmnfVectorEnv(3, thread_count=2, track=track, action_repeat=5, **kwargs)
            stack.callback(env.close)
            return env
        derived = create(budget_reference_speed=25.0)
        ticks = int(np.ceil(2.0 * derived.route_length * derived.lap_count / 25.0 / .01))
        assert derived.max_race_ticks == derived.horizon_ticks == ticks
        explicit = create(max_race_ticks=ticks, horizon_ticks=ticks)
        derived.reset(); explicit.reset()
        rng = np.random.default_rng(98)
        for step in range(80):
            actions = rng.integers(0, 12, 3)
            derived.step(actions); explicit.step(actions)
            np.testing.assert_array_equal(derived.raw_results, explicit.raw_results)
            np.testing.assert_array_equal(derived.capture(), explicit.capture())
            if step == 39:
                snapshots = derived.capture([0, 2])
                derived.restore([2, 1], snapshots)
                explicit.restore([2, 1], snapshots)


@pytest.mark.parametrize('explicit_key', ['max_race_ticks', 'horizon_ticks'])
def test_explicit_budget_overrides_only_its_own_automatic_budget(explicit_key):
    env = TmnfVectorEnv(1, track='coast-a1', thread_count=1,
                        budget_reference_speed=25.0, **{explicit_key: 2400})
    try:
        assert getattr(env, explicit_key) == 2400
        other = 'horizon_ticks' if explicit_key == 'max_race_ticks' else 'max_race_ticks'
        assert getattr(env, other) == 3600
    finally:
        env.close()


@pytest.mark.parametrize('speed', [-1, -1e-100, float('nan'), float('inf'), 1e100, 1e-100])
def test_invalid_budget_pace_rejected_before_native_creation(speed):
    with pytest.raises(ValueError, match='budget_reference_speed'):
        TmnfVectorEnv(1, budget_reference_speed=speed)


def test_config_default_and_budget_override():
    assert config_from_dict({}).budget_reference_speed == 0
    assert config_from_dict({'budget_reference_speed': 25}).budget_reference_speed == 25
    for value in [-1, float('nan'), float('inf')]:
        with pytest.raises(ConfigError, match='budget_reference_speed'):
            config_from_dict({'budget_reference_speed': value})
