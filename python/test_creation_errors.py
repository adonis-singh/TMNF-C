"""Bad user budgets must not abort the Python process or poison the next env."""
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tmnf_rl.env import TmnfVectorEnv

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = Path(os.environ.get('TMNF_TEST_LIBRARY', ROOT/'build/libtmnf_physics.so'))


def test_impossible_long_race_budget_is_recoverable_in_subprocess():
    code = '''
from tmnf_rl.env import TmnfVectorEnv
import sys
for name in ['max_race_ticks', 'horizon_ticks']:
    try:
        TmnfVectorEnv(1, track='rally-e', thread_count=1,
                      library_path=sys.argv[1], **{name: 1})
    except ValueError as error:
        assert name in str(error) and 'minimum' in str(error), error
    else:
        raise AssertionError('impossible budget accepted')
e = TmnfVectorEnv(1, track='rally-e', thread_count=1, library_path=sys.argv[1])
assert e.max_race_ticks > 1
e.close()
print('recoverable')
'''
    result = subprocess.run([sys.executable, '-Werror', '-c', code, str(LIBRARY)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr
    assert result.stdout.strip() == 'recoverable'


@pytest.mark.parametrize('field', ['max_race_ticks', 'horizon_ticks', 'num_envs',
                                   'thread_count', 'action_repeat',
                                   'off_track_grace_ticks', 'stuck_grace_ticks'])
@pytest.mark.parametrize('value', [-1, 2**32, 1.5])
def test_uint_creation_arguments_do_not_wrap_or_truncate(field, value):
    args = dict(num_envs=1, thread_count=1, library_path=LIBRARY)
    args[field] = value
    with pytest.raises(ValueError, match=field):
        TmnfVectorEnv(**args)


@pytest.mark.parametrize('field', ['reference_speed', 'budget_reference_speed'])
def test_overflowing_derived_budget_is_recoverable(field):
    with pytest.raises(ValueError, match='overflowing race budget'):
        TmnfVectorEnv(1, thread_count=1, library_path=LIBRARY, **{field: 1e-30})


def test_budget_boundary_and_valid_transition_compatibility():
    current = TmnfVectorEnv(3, track='rally-e', thread_count=2, library_path=LIBRARY)
    reference = TmnfVectorEnv(3, track='rally-e', thread_count=1,
                             library_path=ROOT/'build/libtmnf_physics.so')
    try:
        minimum = max(1, math.ceil(current.route_length*current.lap_count/277.77777099609375/.01))
        boundary = TmnfVectorEnv(1, track='rally-e', thread_count=1,
                                max_race_ticks=minimum, horizon_ticks=minimum,
                                library_path=LIBRARY)
        boundary.close()
        with pytest.raises(ValueError, match='max_race_ticks'):
            TmnfVectorEnv(1, track='rally-e', thread_count=1,
                          max_race_ticks=minimum-1, library_path=LIBRARY)
        recovered = TmnfVectorEnv(1, track='rally-e', thread_count=1, library_path=LIBRARY)
        recovered.close()
        current.reset(); reference.reset()
        rng = np.random.default_rng(727)
        for _ in range(100):
            actions = rng.integers(0, 12, 3)
            current.step(actions); reference.step(actions)
            np.testing.assert_array_equal(current.raw_results, reference.raw_results)
            assert current.capture() == reference.capture()
        assert not current._lib.tmnf_env_creation_error()
    finally:
        current.close(); reference.close()
