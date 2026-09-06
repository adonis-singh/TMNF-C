from __future__ import annotations

import ctypes
import os
import json
import struct
import subprocess
import threading
import urllib.request
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gymnasium as gym
import numpy as np
import pytest
import torch

from tmnf_rl.agents.ppo import Agent
from tmnf_rl.encoder import encode_flat
from tmnf_rl.snapshot_starts import (
    POOL_BIN_METERS,
    PREFAILURE_EXCLUSION_TICKS,
    PoolState,
    SnapshotPool,
    Trajectory,
    trajectory_score,
)
from tmnf_rl.env import (
    OBSERVATION_SIZE,
    OBSERVATION_VERSION,
    POLICY_OBSERVATION_WIDTH,
    STEP_RESULT_SIZE,
    TERMINATION_NAMES,
    TmnfVectorEnv,
    _Observation,
    _StepResult,
)
from tmnf_rl.spaces import environment_action
from tmnf_rl.spectate import RING_CAPACITY, SpectateBuffer, SpectateServer


PROJECT_ROOT = Path(os.environ["TMNF_PROJECT_ROOT"])
REFERENCE_EXECUTABLE = Path(os.environ["TMNF_REFERENCE_EXECUTABLE"])


def _copy_observation(observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in observation.items()}


def test_observation_layout_version_offsets_and_sizes() -> None:
    assert OBSERVATION_VERSION == 2
    assert POLICY_OBSERVATION_WIDTH == 81
    assert ctypes.sizeof(_Observation) == OBSERVATION_SIZE == 324
    assert _Observation.gear.offset == 136
    assert _Observation.input_steer.offset == 140
    assert _Observation.turbo_active.offset == 184
    assert _Observation.centerline_lookahead.offset == 196
    assert ctypes.sizeof(_StepResult) == STEP_RESULT_SIZE == 688
    assert _StepResult.final_observation.offset == 324
    assert _StepResult.reward.offset == 648
    assert _StepResult.episode_id.offset == 672
    assert TERMINATION_NAMES == {
        1: "finish",
        2: "timeout",
        3: "off_track",
        4: "stuck",
        5: "fell",
        6: "restart",
    }


def _states(progresses: list[float], tick0: int = 0) -> list[PoolState]:
    return [
        PoolState(blob=bytes([index]), progress=progress, capture_tick=tick0 + 100 * index, score=0.0)
        for index, progress in enumerate(progresses)
    ]


@pytest.mark.parametrize("exclusion,expected", [(200, [199, 200]),
                                               (50, [199, 200, 349, 350]),
                                               (1, [199, 200, 349, 350, 351, 399])])
def test_snapshot_pool_recovery_window_and_checkpoint(exclusion, expected):
    pool = SnapshotPool(route_length=1000, rng=np.random.default_rng(3),
                        prefailure_exclusion_ticks=exclusion)
    states = [PoolState(bytes([i]), i * 25.0, tick, 0.0)
              for i, tick in enumerate([199, 200, 349, 350, 351, 399, 400])]
    pool.add_trajectory(Trajectory(states, 100), finished=False, episode_ticks=400)
    assert [s.capture_tick for s in pool.candidates[0].states] == expected
    saved = pool.state_dict()
    restored = SnapshotPool(route_length=1000, rng=np.random.default_rng(3),
                            prefailure_exclusion_ticks=exclusion)
    restored.load_state_dict(saved)
    pool.refresh()
    restored.refresh()
    assert [pool.choose() for _ in range(20)] == [restored.choose() for _ in range(20)]
    mismatch = SnapshotPool(route_length=1000, rng=np.random.default_rng(3),
                           prefailure_exclusion_ticks=exclusion + 1)
    with pytest.raises(ValueError, match="differs"):
        mismatch.load_state_dict(saved)
    assert not mismatch.states and not mismatch.candidates


def test_snapshot_pool_legacy_checkpoint_keeps_original_exclusion():
    pool = SnapshotPool(route_length=1000, rng=np.random.default_rng(3))
    legacy = {"states": [], "candidates": [], "refreshes": 2}
    pool.load_state_dict(legacy)
    assert pool.prefailure_exclusion_ticks == 200 and pool.refreshes == 2
    with pytest.raises(ValueError, match="positive"):
        SnapshotPool(route_length=1000, rng=np.random.default_rng(3),
                     prefailure_exclusion_ticks=0)


def test_snapshot_pool_keeps_one_state_per_bin_from_the_best_trajectory() -> None:
    pool = SnapshotPool(route_length=200.0, rng=np.random.default_rng(3))
    # A failure at 90 m: its captures inside the last 200 ticks are dropped.
    failed = Trajectory(states=_states([5.0, 30.0, 55.0, 80.0]))
    failed.score = trajectory_score(
        final_progress=90.0, finished=False, episode_ticks=400, max_race_ticks=1_000, reference_speed=50.0
    )
    pool.add_trajectory(failed, finished=False, episode_ticks=400)
    assert [state.progress for state in pool.candidates[0].states] == [5.0, 30.0, 55.0]
    assert pool.candidates[0].states[-1].capture_tick == 400 - PREFAILURE_EXCLUSION_TICKS
    # A finish ranks above every failure and a faster finish above a slower one.
    slow = trajectory_score(final_progress=200.0, finished=True, episode_ticks=900, max_race_ticks=1_000, reference_speed=50.0)
    fast = trajectory_score(final_progress=200.0, finished=True, episode_ticks=600, max_race_ticks=1_000, reference_speed=50.0)
    assert failed.score < slow < fast
    slow_trajectory = Trajectory(states=_states([2.0, 25.0, 45.0, 70.0, 150.0]), score=slow)
    fast_trajectory = Trajectory(states=_states([3.0, 33.0, 110.0]), score=fast)
    pool.add_trajectory(slow_trajectory, finished=True, episode_ticks=900)
    pool.add_trajectory(fast_trajectory, finished=True, episode_ticks=600)
    pool.refresh()
    by_bin = {pool.bin_of(state.progress): state for state in pool.states}
    assert len(pool.states) == len(by_bin) == 6
    # Bin 0 and 1 come from the fast lap, bins 2 and 3 from the slow one
    # (the fast lap has no capture there), bin 5 from the fast, bin 7 slow.
    assert by_bin[0].progress == 3.0 and by_bin[1].progress == 33.0
    assert by_bin[2].progress == 45.0 and by_bin[3].progress == 70.0
    assert by_bin[5].progress == 110.0 and by_bin[7].progress == 150.0
    assert pool.candidates == [] and pool.refreshes == 1
    assert POOL_BIN_METERS == 20.0
    # A later, worse trajectory never displaces a pooled state.
    worse = Trajectory(states=_states([1.0, 31.0]), score=10.0)
    pool.add_trajectory(worse, finished=False, episode_ticks=10_000)
    pool.refresh()
    assert {pool.bin_of(s.progress): s.progress for s in pool.states}[0] == 3.0
    # Choices come from the pool rng only.
    choices = {pool.choose()[:2] for _ in range(64)}
    assert {source for source, _ in choices} == {"own"}
    assert {index for _, index in choices} <= set(range(6)) and len(choices) > 1
    # An empty pool offers nothing: the reset falls back to the grid start.
    assert SnapshotPool(route_length=200.0, rng=np.random.default_rng(5)).choose() is None
    restored = SnapshotPool(route_length=200.0, rng=np.random.default_rng(3))
    restored.load_state_dict(pool.state_dict())
    assert [(s.blob, s.progress, s.capture_tick, s.score) for s in restored.states] == [
        (s.blob, s.progress, s.capture_tick, s.score) for s in pool.states
    ]


def test_restored_snapshot_start_carries_its_race_clock_and_progress() -> None:
    """A state captured at tick T restored into another environment reports
    elapsed_fraction T / max_race_ticks and the capture's progress, and the
    race clock continues from T (the observation and the reward see the same
    clock the source saw)."""
    env = TmnfVectorEnv(2, root=PROJECT_ROOT, thread_count=1, action_repeat=5, max_race_ticks=2_000)
    try:
        env.reset()
        for _ in range(40):
            env.step(np.array([4, 1], dtype=np.int32))
        race = env.observations["race"].copy()
        tick = int(np.rint(race[0, 8] * env.max_race_ticks))
        assert tick == 200 and race[0, 4] > 20.0 and race[1, 4] < race[0, 4]
        blob = env.capture([0])[0]
        source_raw = env.raw_observations[0].tobytes()
        env.reset()
        assert env.observations["race"][1, 8] == 0.0
        env.restore([1], [blob])
        restored = env.observations["race"][1]
        assert int(np.rint(restored[8] * env.max_race_ticks)) == tick
        assert restored[4] == race[0, 4] and restored[7] == race[0, 7]
        assert env.raw_observations[1].tobytes() == source_raw
        assert env.restore_count == 1
        env.step(np.array([4, 1], dtype=np.int32))
        assert int(np.rint(env.observations["race"][1, 8] * env.max_race_ticks)) == tick + 5
        assert int(np.rint(env.observations["race"][0, 8] * env.max_race_ticks)) == 5
    finally:
        env.close()


def test_bindings_match_c_reference(tmp_path: Path) -> None:
    reference_path = tmp_path / "python-reference.bin"
    subprocess.run(
        [
            REFERENCE_EXECUTABLE,
            PROJECT_ROOT / "oracle/tracks/A01-Race.tmnftrack",
            PROJECT_ROOT / "oracle/vehicles/A01-Stadium.tmnfvehicle",
            PROJECT_ROOT / "oracle/routes/A01-Race.tmnfroute",
            reference_path,
        ],
        check=True,
    )
    reference = reference_path.read_bytes()
    environment_count, step_count, observation_size, result_size = struct.unpack_from(
        "<4I", reference
    )
    offset = struct.calcsize("<4I")

    env = TmnfVectorEnv(
        environment_count,
        root=PROJECT_ROOT,
        thread_count=1,
        action_repeat=2,
        off_track_grace_ticks=50,
        stuck_grace_ticks=80,
    )
    try:
        # tests/python_reference.c uses the default config: both budgets derived
        # per track (A01: 2 x 2,209.7 m at 50 m/s = 8,839 ticks), no clamp.
        assert env.max_race_ticks == 8_839
        assert env.horizon_ticks == 8_839
        assert observation_size == env.observation_size
        assert result_size == env.result_size
        observations, _ = env.reset()
        np.testing.assert_array_equal(
            observations["turbo"],
            np.zeros((environment_count, 3), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            env.policy_observations[:, 78:],
            observations["turbo"],
        )
        reset_bytes = environment_count * observation_size
        assert env.raw_observations.tobytes() == reference[offset : offset + reset_bytes]
        offset += reset_bytes

        for step in range(step_count):
            actions = np.array(
                [(step * 5 + index * 7) % 12 for index in range(environment_count)],
                dtype=np.int64,
            )
            env.step(actions)
            result_bytes = environment_count * result_size
            assert env.raw_results.tobytes() == reference[offset : offset + result_bytes]
            offset += result_bytes
        assert offset == len(reference)

        assert env.metadata["autoreset_mode"] is gym.vector.AutoresetMode.SAME_STEP
        for array in observations.values():
            assert np.shares_memory(array, env.raw_results)
        assert np.shares_memory(env.rewards, env.raw_results)
        assert np.shares_memory(env.terminations, env.raw_results)
        assert np.shares_memory(env.truncations, env.raw_results)
    finally:
        env.close()


def test_python_determinism_across_thread_counts() -> None:
    kwargs = {
        "num_envs": 8,
        "root": PROJECT_ROOT,
        "action_repeat": 3,
        "max_race_ticks": 1_000,
        "off_track_grace_ticks": 50,
        "stuck_grace_ticks": 80,
    }
    serial = TmnfVectorEnv(thread_count=1, **kwargs)
    parallel = TmnfVectorEnv(thread_count=8, **kwargs)
    try:
        # horizon_ticks omitted: derived per track (8,839), independent of the
        # explicit 1,000-tick race timeout; the timeout ends episodes first.
        assert serial.horizon_ticks == parallel.horizon_ticks == 8_839
        assert serial.max_race_ticks == parallel.max_race_ticks == 1_000
        serial.reset()
        parallel.reset()
        assert serial.raw_observations.tobytes() == parallel.raw_observations.tobytes()
        for step in range(300):
            actions = np.array(
                [(step * 11 + index * 5) % 12 for index in range(8)],
                dtype=np.int32,
            )
            serial.step(actions)
            parallel.step(actions)
            assert serial.raw_observations.tobytes() == parallel.raw_observations.tobytes()
            assert serial.rewards.tobytes() == parallel.rewards.tobytes()
            assert serial.terminations.tobytes() == parallel.terminations.tobytes()
            assert serial.truncations.tobytes() == parallel.truncations.tobytes()
    finally:
        parallel.close()
        serial.close()


def test_analog_action_space_quantization_and_repeat_one() -> None:
    env = TmnfVectorEnv(
        7,
        root=PROJECT_ROOT,
        thread_count=3,
        action_repeat=1,
        action_space="analog",
        discount_per_tick=0.9999,
    )
    try:
        assert isinstance(env.single_action_space, gym.spaces.Dict)
        assert isinstance(
            env.single_action_space["steer"], gym.spaces.Box
        )
        env.reset()
        scale = np.float32(65_536.0)
        steer = np.array(
            [
                -1.0,
                -0.5,
                -1.4 / 65_536.0,
                0.0,
                1.4 / 65_536.0,
                0.5,
                1.0,
            ],
            dtype=np.float32,
        )
        env.step(
            {
                "steer": steer,
                "gas": np.array([0, 1, 0, 1, 1, 0, 1], dtype=np.int8),
                "brake": np.array(
                    [1, 0, 1, 0, 1, 1, 0], dtype=np.int8
                ),
            }
        )
        expected_integer = np.array(
            [-65_536, -32_768, -1, 0, 1, 32_768, 65_536],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            env.observations["race"][:, 0],
            expected_integer / scale,
        )
        np.testing.assert_array_equal(
            env.executed_ticks, np.ones(7, dtype=np.uint32)
        )
        np.testing.assert_array_equal(
            env.transition_discounts,
            np.full(7, np.float32(0.9999), dtype=np.float32),
        )
    finally:
        env.close()


def test_analog_policy_distribution_matches_environment_space() -> None:
    torch.manual_seed(7)
    agent = Agent("mlp", 32, action_space="analog")
    observation = torch.zeros((16, POLICY_OBSERVATION_WIDTH))
    action, log_probability, entropy, value = agent.get_action_and_value(
        observation
    )
    assert action.shape == (16, 3)
    assert log_probability.shape == (16,)
    assert entropy.shape == (16,)
    assert value.shape == (16,)
    assert torch.isfinite(log_probability).all()
    assert torch.isfinite(entropy).all()
    # Column 0 is the pre-tanh steer sample (unbounded); the env receives tanh of it.
    assert torch.all((action[:, 1:] == 0.0) | (action[:, 1:] == 1.0))
    converted = environment_action(action, "analog")
    assert isinstance(converted, dict)
    assert converted["steer"].dtype == np.float32
    assert np.all((converted["steer"] > -1.0) & (converted["steer"] < 1.0))
    np.testing.assert_array_equal(converted["steer"], torch.tanh(action[:, 0]).numpy())
    assert converted["gas"].dtype == np.int8
    assert converted["brake"].dtype == np.int8
    _, replay_log_probability, _, replay_value = agent.get_action_and_value(
        observation, action
    )
    torch.testing.assert_close(replay_log_probability, log_probability, rtol=0, atol=0)
    torch.testing.assert_close(replay_value, value, rtol=0, atol=0)


def test_analog_action_repeat_telescopes_exactly() -> None:
    env = TmnfVectorEnv(
        1,
        root=PROJECT_ROOT,
        thread_count=1,
        action_space="analog",
        action_repeat=3,
        max_race_ticks=2_000,
        horizon_ticks=1_500,
        discount_per_tick=0.9,
    )
    action = {
        "steer": np.array([0.375], dtype=np.float32),
        "gas": np.array([1], dtype=np.int8),
        "brake": np.array([0], dtype=np.int8),
    }
    try:
        env.reset()
        for _ in range(11):
            env.step(action)
        snapshot = env.capture([0])[0]
        env.step(action)
        repeated_observation = env.raw_observations[0].tobytes()
        repeated_reward = env.rewards[0].copy()
        repeated_discount = env.transition_discounts[0].copy()

        env.restore([0], [snapshot])
        env.action_repeat = 1
        combined_reward = np.float32(0.0)
        combined_discount = np.float32(1.0)
        for _ in range(3):
            env.step(action)
            combined_reward = np.float32(
                combined_reward + combined_discount * env.rewards[0]
            )
            combined_discount = np.float32(
                combined_discount * env.transition_discounts[0]
            )
        assert env.raw_observations[0].tobytes() == repeated_observation
        assert repeated_reward.tobytes() == combined_reward.tobytes()
        assert repeated_discount.tobytes() == combined_discount.tobytes()
    finally:
        env.close()


def test_python_analog_determinism_and_snapshot_restore() -> None:
    kwargs = {
        "num_envs": 8,
        "root": PROJECT_ROOT,
        "action_space": "analog",
        "action_repeat": 3,
        "max_race_ticks": 1_000,
        "off_track_grace_ticks": 50,
        "stuck_grace_ticks": 80,
    }
    serial = TmnfVectorEnv(thread_count=1, **kwargs)
    parallel = TmnfVectorEnv(thread_count=8, **kwargs)
    try:
        serial.reset()
        parallel.reset()
        for step in range(100):
            integer_steer = np.array(
                [
                    ((step * 997 + index * 8191) % 131_073) - 65_536
                    for index in range(8)
                ],
                dtype=np.float32,
            )
            actions = {
                "steer": integer_steer / np.float32(65_536.0),
                "gas": np.array(
                    [(step + index) % 2 for index in range(8)],
                    dtype=np.int8,
                ),
                "brake": np.array(
                    [(step // 3 + index) % 2 for index in range(8)],
                    dtype=np.int8,
                ),
            }
            serial.step(actions)
            parallel.step(actions)
            assert serial.raw_results.tobytes() == parallel.raw_results.tobytes()

        snapshot = serial.capture([3])[0]
        expected_results: list[bytes] = []
        for step in range(100, 117):
            integer_steer = np.array(
                [
                    ((step * 997 + index * 8191) % 131_073) - 65_536
                    for index in range(8)
                ],
                dtype=np.float32,
            )
            actions = {
                "steer": integer_steer / np.float32(65_536.0),
                "gas": np.array(
                    [(step + index) % 2 for index in range(8)],
                    dtype=np.int8,
                ),
                "brake": np.array(
                    [(step // 3 + index) % 2 for index in range(8)],
                    dtype=np.int8,
                ),
            }
            serial.step(actions)
            expected_results.append(serial.raw_results[3].tobytes())
        serial.restore([3], [snapshot])
        for expected_index, step in enumerate(range(100, 117)):
            integer_steer = np.array(
                [
                    ((step * 997 + index * 8191) % 131_073) - 65_536
                    for index in range(8)
                ],
                dtype=np.float32,
            )
            actions = {
                "steer": integer_steer / np.float32(65_536.0),
                "gas": np.array(
                    [(step + index) % 2 for index in range(8)],
                    dtype=np.int8,
                ),
                "brake": np.array(
                    [(step // 3 + index) % 2 for index in range(8)],
                    dtype=np.int8,
                ),
            }
            serial.step(actions)
            assert (
                serial.raw_results[3].tobytes()
                == expected_results[expected_index]
            )
    finally:
        parallel.close()
        serial.close()


def test_snapshot_restore_continuation_is_byte_identical() -> None:
    env = TmnfVectorEnv(
        3,
        root=PROJECT_ROOT,
        thread_count=3,
        action_repeat=3,
        max_race_ticks=2_000,
        horizon_ticks=1_500,
    )
    try:
        env.reset()
        for step in range(11):
            env.step(
                np.array(
                    [(step * 7 + index * 3) % 12 for index in range(3)],
                    dtype=np.int32,
                )
            )

        snapshot = env.capture([1])[0]
        assert isinstance(snapshot, bytes)
        assert len(snapshot) == env.snapshot_size
        actions = [
            np.array([(step * 5 + index * 7) % 12 for index in range(3)])
            for step in range(17)
        ]
        expected = []
        for action in actions:
            env.step(action)
            expected.append(env.raw_results[1].tobytes())

        env.restore([1], [snapshot])
        for action, expected_result in zip(actions, expected, strict=True):
            env.step(action)
            assert env.raw_results[1].tobytes() == expected_result
    finally:
        env.close()


# A01 needs 796 ticks at the game's speed cap; the env aborts on any horizon
# below that, so truncation tests idle the car (action 1 = no input) for one
# repeat-800 step with the stuck rule disabled.
A01_TRUNCATION_HORIZON = 800

# Per-step rewards (float32 bits) of an idle A01 car at discount 0.9 per tick,
# action repeat 5, for the 19 whole steps before the progress-only stuck rule
# (grace 100) ends the episode at tick 100. Recorded on the physics at commit
# d09dc00 and re-pinned on 4 September 2026 when the A01 route was regenerated
# with the current generator (float32 arc accumulation; positions moved by
# up to 4 mm); these ticks are plain -0.01 + 0.9 phi' - phi and did not
# change with the terminal potential. The ending step (ticks 96-100) is
# derived from the reward identity in the test.
EXPECTED_IDLE_REWARD_BITS = [
    1099986041, 1099986043, 1099986034, 1099986015, 1099986011, 1099986010,
    1099986010, 1099986012, 1099986010, 1099986010, 1099986010, 1099986010,
    1099986008, 1099986010, 1099986010, 1099986008, 1099986010, 1099986010,
    1099986012,
]
# race[7] is remaining_distance, race[4] unwrapped_progress (env.py layout).
RACE_REMAINING = 7
RACE_PROGRESS = 4
REFERENCE_SPEED = 50.0


def _potential(race: np.ndarray) -> float:
    return -float(race[RACE_REMAINING]) / REFERENCE_SPEED


def test_same_step_autoreset_invariant() -> None:
    env = TmnfVectorEnv(
        1,
        root=PROJECT_ROOT,
        thread_count=1,
        action_repeat=A01_TRUNCATION_HORIZON,
        max_race_ticks=1_000,
        horizon_ticks=A01_TRUNCATION_HORIZON,
        stuck_grace_ticks=100_000,
    )
    try:
        reset_observation, _ = env.reset()
        expected_reset = _copy_observation(reset_observation)
        observation, _, terminated, truncated, info = env.step(
            np.array([1], dtype=np.int32)
        )
        assert not terminated[0]
        assert truncated[0]
        assert info["_final_obs"][0]
        for name in observation:
            np.testing.assert_array_equal(observation[name][0], expected_reset[name][0])
        assert not np.array_equal(
            env.raw_observations[0], env.raw_final_observations[0]
        )
        assert env.final_observations["race"][0, 8] > 0.0
        assert env.observations["race"][0, 8] == 0.0
    finally:
        env.close()


def test_respawn_action_is_optional_and_restarts_before_a_checkpoint() -> None:
    """respawn_action=False keeps the twelve-action set; with it, action a + 12
    is action a plus the press, a press before any respawnable checkpoint ends
    the episode as a restart (analysis/respawn.md), and the analog space
    gains a binary respawn key."""
    plain = TmnfVectorEnv(1, root=PROJECT_ROOT, thread_count=1)
    try:
        assert plain.single_action_space == gym.spaces.Discrete(12)
        assert plain.action_count == 12
    finally:
        plain.close()
    env = TmnfVectorEnv(2, root=PROJECT_ROOT, thread_count=1, respawn_action=True)
    try:
        assert env.single_action_space == gym.spaces.Discrete(24)
        env.reset()
        _, _, terminated, _, info = env.step(np.array([4, 4 + 12]))
        assert terminated.tolist() == [False, True]
        assert TERMINATION_NAMES[int(info["termination_reason"][1])] == "restart"
    finally:
        env.close()
    analog = TmnfVectorEnv(
        1, root=PROJECT_ROOT, thread_count=1, action_space="analog", respawn_action=True
    )
    try:
        assert set(analog.single_action_space.spaces) == {"steer", "gas", "brake", "respawn"}
        analog.reset()
        action = {
            "steer": np.zeros(1, dtype=np.float32),
            "gas": np.ones(1, dtype=np.int8),
            "brake": np.zeros(1, dtype=np.int8),
            "respawn": np.zeros(1, dtype=np.int8),
        }
        _, _, terminated, _, _ = analog.step(action)
        assert not terminated[0]
        action["respawn"][0] = 1
        _, _, terminated, _, info = analog.step(action)
        assert terminated[0]
        assert TERMINATION_NAMES[int(info["termination_reason"][0])] == "restart"
    finally:
        analog.close()


def test_nonzero_reset_seed_is_rejected() -> None:
    env = TmnfVectorEnv(1, root=PROJECT_ROOT, thread_count=1)
    try:
        with pytest.raises(
            ValueError,
            match="seed must be zero until randomized reset exists",
        ):
            env.reset(seed=1)
        env.reset(seed=0)
    finally:
        env.close()


def test_truncated_final_observation_encodes_like_the_next_observation() -> None:
    """The truncation bootstrap value is computed from the final observation
    of the truncated env; it must be the very observation an untruncated env
    reports next, through the same fixed-scale encoder."""
    common = {
        "num_envs": 1,
        "root": PROJECT_ROOT,
        "thread_count": 1,
        "action_repeat": A01_TRUNCATION_HORIZON,
        "max_race_ticks": 1_000,
        "stuck_grace_ticks": 100_000,
    }
    ordinary = TmnfVectorEnv(horizon_ticks=A01_TRUNCATION_HORIZON + 1, **common)
    truncated = TmnfVectorEnv(horizon_ticks=A01_TRUNCATION_HORIZON, **common)
    try:
        ordinary.reset()
        truncated.reset()
        np.testing.assert_array_equal(
            ordinary.policy_observations,
            truncated.policy_observations,
        )
        action = np.array([1], dtype=np.int32)
        _, _, ordinary_terminated, ordinary_truncated, _ = ordinary.step(action)
        ordinary_next = torch.from_numpy(ordinary.policy_observations.copy())
        _, _, final_terminated, final_truncated, _ = truncated.step(action)
        truncated_final = torch.from_numpy(truncated.policy_final_observations.copy())

        assert not ordinary_terminated[0]
        assert not ordinary_truncated[0]
        assert not final_terminated[0]
        assert final_truncated[0]
        torch.testing.assert_close(ordinary_next, truncated_final, rtol=0, atol=0)
        torch.testing.assert_close(
            encode_flat(ordinary_next), encode_flat(truncated_final), rtol=0, atol=0
        )
    finally:
        truncated.close()
        ordinary.close()


def test_completed_return_is_exact_published_reward_sum() -> None:
    gamma = 0.9
    max_race_ticks = 1_000
    env = TmnfVectorEnv(
        1,
        root=PROJECT_ROOT,
        thread_count=1,
        action_repeat=5,
        max_race_ticks=max_race_ticks,
        horizon_ticks=A01_TRUNCATION_HORIZON,
        off_track_grace_ticks=100,
        stuck_grace_ticks=100,
        discount_per_tick=gamma,
    )
    try:
        observation, _ = env.reset()
        initial_potential = _potential(observation["race"][0])
        potential_before_last = initial_potential
        reward_tensor = []
        discounted_return = 0.0
        weight = 1.0
        completed_return = None
        while completed_return is None:
            # Idle car: the progress-only stuck rule ends the episode after
            # 100 ticks without 1 mm of route progress (tick 100, step 20;
            # the spawned car's settling does not move its projection).
            observation, reward, terminated, truncated, info = env.step(
                np.array([1], dtype=np.int32)
            )
            reward_tensor.append(reward.copy())
            discounted_return += weight * float(reward[0])
            weight *= float(info["transition_discount"][0])
            if terminated[0] or truncated[0]:
                assert terminated[0] and not truncated[0]
                assert info["termination_reason"][0] == 4
                assert info["completed_episode_ticks"][0] == 100
                assert info["executed_ticks"][0] == 5
                completed_return = info["completed_episode_return"].copy()
                terminal_potential = _potential(env.final_observations["race"][0])
            else:
                potential_before_last = _potential(observation["race"][0])

        rewards = np.stack(reward_tensor)
        assert rewards.shape == (20, 1)
        expected_reward_bits = np.array(EXPECTED_IDLE_REWARD_BITS, dtype=np.uint32)
        np.testing.assert_array_equal(
            rewards[:19].view(np.uint32).reshape(-1), expected_reward_bits
        )
        # Ending step, ticks 96..100 with tick 100 the STUCK tick: the four
        # ordinary ticks cost -0.01 each, tick 100 carries the discounted cost
        # of the 901 remaining budget ticks, and the shaping telescopes across
        # the step to gamma^5 gamma^(Tmax - 100) phi(s_100) - phi(s_95):
        # phi(s_100) is kept and deferred to the end of the budget.
        ticks = 5
        lump = -0.01 * (1.0 - gamma ** (max_race_ticks - 99)) / (1.0 - gamma)
        expected_last = (
            -0.01 * (1.0 - gamma ** (ticks - 1)) / (1.0 - gamma)
            + gamma ** (ticks - 1) * lump
            + gamma ** (ticks + max_race_ticks - 100) * terminal_potential
            - potential_before_last
        )
        assert abs(float(rewards[19, 0]) - expected_last) < 5e-5, (
            rewards[19, 0],
            expected_last,
        )
        # Whole episode: G = gamma^Tmax phi(s_T) - phi(s_0) - 0.01 (1 - gamma^Tmax) / (1 - gamma).
        expected_return = (
            gamma**max_race_ticks * terminal_potential
            - initial_potential
            - 0.01 * (1.0 - gamma**max_race_ticks) / (1.0 - gamma)
        )
        assert abs(discounted_return - expected_return) < 1e-3, (
            discounted_return,
            expected_return,
        )
        # The published completed return is the float32 sum of the raw step
        # rewards, not the discounted return.
        published_sum = np.float32(0.0)
        for reward in rewards[:, 0]:
            published_sum = np.float32(published_sum + reward)
        assert completed_return is not None
        assert completed_return[0].view(np.uint32) == published_sum.view(np.uint32)
    finally:
        env.close()


def _discounted_episode(
    env: TmnfVectorEnv, actions: np.ndarray | None, hold: int
) -> dict[str, float]:
    """One episode per environment at action repeat 1; returns per-env
    discounted returns, initial/terminal potentials, ticks and reasons.
    ``actions`` is a per-tick schedule for env 0 (the others hold ``hold``)."""
    observation, _ = env.reset()
    count = env.num_envs
    initial = np.array([_potential(observation["race"][i]) for i in range(count)])
    returns = np.zeros(count)
    weight = np.ones(count)
    terminal = np.zeros(count)
    ticks = np.zeros(count, dtype=np.int64)
    reasons = np.zeros(count, dtype=np.int64)
    done = np.zeros(count, dtype=bool)
    tick = 0
    while not done.all():
        action = np.full(count, hold, dtype=np.int32)
        if actions is not None:
            action[0] = actions[tick] if tick < len(actions) else hold
        _, reward, terminated, truncated, info = env.step(action)
        live = ~done
        returns[live] += weight[live] * reward[live].astype(np.float64)
        weight[live] *= info["transition_discount"][live]
        ended = live & (terminated | truncated)
        assert not np.any(live & truncated), "identity episode truncated"
        for i in np.flatnonzero(ended):
            terminal[i] = _potential(env.final_observations["race"][i])
            ticks[i] = tick + 1
            reasons[i] = int(info["termination_reason"][i])
        done |= ended
        tick += 1
    return {
        "return": returns,
        "initial": initial,
        "terminal": terminal,
        "ticks": ticks,
        "reason": reasons,
    }


def test_finish_beats_every_failure_and_progress_orders_failures() -> None:
    """Return identities of src/vec_env.c tick_reward on A01 with one shared
    race budget: env 0 drives the committed policy lap to the finish, env 1
    coasts to the timeout, env 2 holds full gas into a wall and times out.

    finish  = -phi(s_0) - 0.01 (1 - gamma^T) / (1 - gamma)
    failure = gamma^Tmax phi(s_T) - phi(s_0) - 0.01 (1 - gamma^Tmax) / (1 - gamma)

    so finish > gas > coast, and gas - coast = gamma^Tmax (phi_gas - phi_coast).
    Before this identity the terminal potential was zeroed on failure and every
    non-finishing episode had the same return (docs/RL_PLATFORM.md F33)."""
    from tmnf_rl.inputs import decode_discrete_schedule

    schedule = decode_discrete_schedule(
        (PROJECT_ROOT / "oracle" / "results" / "policy_lap_inputs.bin").read_bytes(),
        2700,
    )
    gamma = 0.9999
    max_race_ticks = 2527 + 1000
    env = TmnfVectorEnv(
        3,
        root=PROJECT_ROOT,
        thread_count=3,
        action_repeat=1,
        max_race_ticks=max_race_ticks,
        horizon_ticks=max_race_ticks + 1,
        off_track_grace_ticks=1_000_000,
        stuck_grace_ticks=1_000_000,
        discount_per_tick=gamma,
    )
    try:
        # Env 2 holds full gas; env 1 coasts; env 0 follows the lap. The hold
        # action applies to envs 1 and 2, so run them as two episodes sets.
        lap = _discounted_episode(env, np.asarray(schedule.actions), hold=1)
        gas = _discounted_episode(env, None, hold=4)
    finally:
        env.close()
    finish_return, coast_return = lap["return"][0], lap["return"][1]
    gas_return = gas["return"][2]
    time_constant = 0.01 * (1.0 - gamma**max_race_ticks) / (1.0 - gamma)

    assert lap["reason"][0] == 1 and lap["ticks"][0] == 2527
    assert lap["terminal"][0] == 0.0
    finish_identity = -lap["initial"][0] - 0.01 * (1.0 - gamma ** 2527) / (1.0 - gamma)
    assert abs(finish_return - finish_identity) < 4e-6 * 2527 + 1e-4

    for outcome, index in ((lap, 1), (gas, 2)):
        assert outcome["reason"][index] == 2 and outcome["ticks"][index] == max_race_ticks
        identity = (
            gamma**max_race_ticks * outcome["terminal"][index]
            - outcome["initial"][index]
            - time_constant
        )
        assert abs(outcome["return"][index] - identity) < 4e-6 * max_race_ticks + 1e-4

    # 117 m of progress for the gas car, none for the coasting one.
    progress_gap = REFERENCE_SPEED * (gas["terminal"][2] - lap["terminal"][1])
    assert progress_gap > 50.0
    assert finish_return > gas_return + 1.0 > coast_return + 1.5
    assert abs(
        (gas_return - coast_return) - gamma**max_race_ticks * progress_gap / REFERENCE_SPEED
    ) < 4e-6 * 2 * max_race_ticks + 1e-4


def test_step_rejects_concurrent_close() -> None:
    env = TmnfVectorEnv(
        1,
        root=PROJECT_ROOT,
        thread_count=1,
        action_repeat=1,
    )
    native_step = env._lib.tmnf_env_step_discrete
    entered_step = threading.Event()
    release_step = threading.Event()
    step_errors: list[BaseException] = []

    def blocking_step(handle: int, action_repeat: int) -> None:
        entered_step.set()
        if not release_step.wait(timeout=5.0):
            raise TimeoutError("concurrent close test did not release step")
        native_step(handle, action_repeat)

    def run_step() -> None:
        try:
            env.step(np.array([1], dtype=np.int32))
        except BaseException as error:
            step_errors.append(error)

    env._lib.tmnf_env_step_discrete = blocking_step
    worker = threading.Thread(target=run_step)
    try:
        env.reset()
        worker.start()
        assert entered_step.wait(timeout=5.0)
        with pytest.raises(
            RuntimeError, match="close overlaps active step operation"
        ):
            env.close()
        release_step.set()
        worker.join(timeout=5.0)
        assert not worker.is_alive()
        assert step_errors == []
    finally:
        release_step.set()
        worker.join(timeout=5.0)
        env._lib.tmnf_env_step_discrete = native_step
        env.close()


def test_a10_zero_checkpoint_environment_steps() -> None:
    env = TmnfVectorEnv(
        2,
        root=PROJECT_ROOT,
        track="a10",
        thread_count=2,
        action_repeat=3,
    )
    try:
        observations, _ = env.reset()
        assert env.track_id == "a10"
        assert env.track_name == "A10-Acrobatic"
        assert env.checkpoint_count == 0
        assert env.route_length > 400.0
        env.step(np.array([4, 4], dtype=np.int32))
        assert np.isfinite(observations["vehicle"]).all()
    finally:
        env.close()


def test_spectate_ring_and_http_protocol() -> None:
    count = 6
    observations = {
        "vehicle": np.zeros((count, 34), dtype=np.float32),
        "gear": np.ones(count, dtype=np.int32),
        "race": np.zeros((count, 11), dtype=np.float32),
    }
    final_observations = {
        name: value.copy() for name, value in observations.items()
    }
    observations["vehicle"][:, 3] = np.float32(0.70710677)
    observations["vehicle"][:, 5] = np.float32(0.70710677)
    observations["vehicle"][:, 7] = np.arange(count, dtype=np.float32)
    observations["vehicle"][:, 21:25] = 1.0
    observations["vehicle"][:, 33] = 7_500.0
    observations["race"][:, 4] = np.arange(count, dtype=np.float32) * 10.0
    observations["race"][:, 8] = 0.25
    final_observations["vehicle"][:] = observations["vehicle"]
    final_observations["race"][:] = observations["race"]

    buffer = SpectateBuffer(
        num_envs=count,
        stream_envs=4,
        action_repeat=5,
        max_race_ticks=100,
        initial_meta={
            "streamId": "test-stream",
            "trackId": "a10",
            "trackName": "A10-Acrobatic",
            "scene": "scenes/a10_turbo.json",
        },
    )
    ended = np.zeros(count, dtype=np.bool_)
    ended[5] = True
    reasons = np.zeros(count, dtype=np.int32)
    reasons[5] = 4
    episodes = np.arange(count, dtype=np.uint64) + 100
    for _ in range(RING_CAPACITY + 2):
        buffer.capture(
            observations=observations,
            final_observations=final_observations,
            ended=ended,
            termination_reasons=reasons,
            episode_ids=episodes,
            progress=observations["race"][:, 4],
        )

    payload = buffer.frames_since(-1)
    assert payload["dropped"] == 1794
    assert len(payload["frames"]) == 256
    followed = payload["frames"][-1]["envs"][0]
    assert followed["envId"] == 5
    assert followed["speedMps"] == 5.0
    assert followed["reset"]
    assert followed["terminationReason"] == "stuck"
    assert followed["wheels"][0]["steer"] is None

    server = SpectateServer(0, buffer)
    server.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.port}/spectate/meta"
        ) as response:
            meta = json.load(response)
            assert response.headers["Access-Control-Allow-Origin"] == "*"
        assert meta["protocolVersion"] == 1
        assert meta["trackId"] == "a10"
        assert meta["envIndices"][0] == 5
    finally:
        server.close()
