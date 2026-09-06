"""Full-start deterministic evaluation and exact schedule replay.

Every lap time reported here comes from an episode that started at the
official spawn in a dedicated environment that has never been restored from a
snapshot, and whose Python-counted ticks equal the native episode tick count.
Both invariants are asserted, not assumed.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from tmnf_rl.env import FINISH_REASON, TERMINATION_NAMES, TmnfVectorEnv
from tmnf_rl.exploration import PedalHold
from tmnf_rl.inputs import InputSchedule
from tmnf_rl.spaces import environment_action


class EvaluationIntegrityError(RuntimeError):
    pass


@dataclass
class EvaluationResult:
    # "greedy": argmax actions from the official start; every episode is the
    # same trajectory, so `finished` is the whole result and `finish_rate` is
    # 0 or 1 by construction (F21/F29). "sampled": actions drawn from the
    # policy distribution with a fixed seed; the rate is a real rate.
    mode: str
    finished: bool | None
    episodes: int
    finishes: int
    finish_rate: float
    distance_mean: float
    distance_p90: float
    best_distance: float
    median_lap_ms: float | None
    best_lap_ms: int | None
    physics_steps: int
    decision_steps: int
    evaluation_seconds: float
    physics_steps_per_second: float
    termination_reason_counts: dict[str, int]
    lap_times_ms: list[int] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)
    best_schedule: InputSchedule | None = field(default=None, repr=False)

    def row(self) -> dict[str, Any]:
        """The evaluations.csv columns: eval_fullstart/* for the greedy
        trajectory (with `finished` instead of a rate), eval_sampled/* for the
        sampled policy."""
        reasons = json.dumps(
            {
                str(reason): count
                for reason, count in sorted(
                    (int(key), value)
                    for key, value in self.termination_reason_counts.items()
                )
            },
            sort_keys=True,
        )
        if self.mode == "greedy":
            return {
                "eval_fullstart/episodes": self.episodes,
                "eval_fullstart/finished": int(bool(self.finished)),
                "eval_fullstart/distance_mean": self.distance_mean,
                "eval_fullstart/best_distance": self.best_distance,
                "eval_fullstart/median_lap_ms": self.median_lap_ms,
                "eval_fullstart/best_lap_ms": self.best_lap_ms,
                "eval_fullstart/physics_steps": self.physics_steps,
                "eval_fullstart/decision_steps": self.decision_steps,
                "eval_fullstart/evaluation_seconds": self.evaluation_seconds,
                "eval_fullstart/physics_steps_per_second": self.physics_steps_per_second,
                "termination_reason_counts": reasons,
            }
        return {
            "eval_sampled/episodes": self.episodes,
            "eval_sampled/finishes": self.finishes,
            "eval_sampled/finish_rate": self.finish_rate,
            "eval_sampled/distance_mean": self.distance_mean,
            "eval_sampled/distance_p90": self.distance_p90,
            "eval_sampled/best_distance": self.best_distance,
            "eval_sampled/median_lap_ms": self.median_lap_ms,
            "eval_sampled/best_lap_ms": self.best_lap_ms,
            "eval_sampled/physics_steps": self.physics_steps,
            "eval_sampled/decision_steps": self.decision_steps,
            "eval_sampled/evaluation_seconds": self.evaluation_seconds,
            "eval_sampled/physics_steps_per_second": self.physics_steps_per_second,
            "eval_sampled/termination_reason_counts": reasons,
        }

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("best_schedule")
        payload["termination_reason_names"] = {
            TERMINATION_NAMES.get(int(key), str(key)): value
            for key, value in self.termination_reason_counts.items()
        }
        if self.best_schedule is not None:
            payload["best_schedule_ticks"] = self.best_schedule.tick_count
            payload["best_schedule_sha256"] = self.best_schedule.sha256()
        return payload


@torch.no_grad()
def evaluate_full_start(
    agent: Any,
    env: TmnfVectorEnv,
    episode_count: int,
    device: torch.device,
    *,
    sampled: bool = False,
    sample_seed: int = 0,
    pedal_hold_decisions: int = 1,
) -> EvaluationResult:
    """Evaluate from the official start on a never-restored environment.

    Greedy (default): argmax actions; all episodes are one trajectory, and the
    result asserts that (a divergence would be a physics nondeterminism).
    Sampled: actions drawn from the policy with `sample_seed`; the Torch CPU
    and CUDA RNG states are saved and restored around the evaluation so a
    wall-clock-scheduled evaluation never touches the training RNG stream.
    """
    if env.restore_count != 0:
        raise EvaluationIntegrityError(
            "evaluation environment has been restored from a snapshot "
            f"{env.restore_count} time(s); full-start laps require a "
            "never-restored environment"
        )
    if not sampled:
        return _evaluate(agent, env, episode_count, device, sampled=False, pedal_hold_decisions=pedal_hold_decisions)
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device)
    try:
        torch.manual_seed(sample_seed)
        return _evaluate(agent, env, episode_count, device, sampled=True, pedal_hold_decisions=pedal_hold_decisions)
    finally:
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state, device)


def _evaluate(
    agent: Any,
    env: TmnfVectorEnv,
    episode_count: int,
    device: torch.device,
    *,
    sampled: bool,
    pedal_hold_decisions: int,
) -> EvaluationResult:
    observations, _ = env.reset()
    race = observations["race"]
    if np.any(race[:, 8] != 0.0) or np.any(np.abs(race[:, 4]) > 1.0):
        raise EvaluationIntegrityError(
            "reset did not place every environment at the official start"
        )
    completed = 0
    finishes = 0
    lap_times: list[int] = []
    distances: list[float] = []
    decision_steps = 0
    physics_steps = 0
    reason_counts: dict[int, int] = {}
    best_lap_ms: int | None = None
    best_schedule: InputSchedule | None = None
    mode = env.action_space_mode
    decision_actions: list[np.ndarray] = []
    decision_ticks: list[np.ndarray] = []
    episode_start_decision = np.zeros(env.num_envs, dtype=np.int64)
    episode_ticks = np.zeros(env.num_envs, dtype=np.int64)
    # Pinned staging: the upload is then asynchronous and the action download
    # below is the one stream sync per step (see PPOTrainer.pinned_observations).
    pinned = torch.empty(env.policy_observations.shape, dtype=torch.float32, pin_memory=True)
    # Pedal hold (config.pedal_hold_decisions): the policy was trained with
    # gas and brake held between decisions, so it drives that way here too.
    pedal_hold = PedalHold(env.num_envs, pedal_hold_decisions, device) if pedal_hold_decisions > 1 else None
    evaluation_start = time.perf_counter()
    while completed < episode_count:
        observation = pinned.copy_(torch.from_numpy(env.policy_observations)).to(device, non_blocking=True)
        if sampled:
            action = agent.get_action_and_value(observation)[0]
        else:
            action = agent.get_deterministic_action(observation)
        if pedal_hold is not None:
            action = pedal_hold.apply(action, pedal_hold.mask())
        # An agent with its own action coordinates (agents/td3.py) converts
        # them itself; the PPO heads use the shared conversion.
        convert = getattr(agent, "environment_action", None)
        action_array = convert(action, mode) if convert is not None else environment_action(action, mode)
        _, _, terminated, truncated, info = env.step(action_array)
        executed = env.executed_ticks.copy()
        if mode == "discrete":
            decision_actions.append(np.asarray(action_array, dtype=np.int16).copy())
        else:
            decision_actions.append(
                np.stack(
                    (
                        np.asarray(action_array["steer"], dtype=np.float32),
                        np.asarray(action_array["gas"], dtype=np.float32),
                        np.asarray(action_array["brake"], dtype=np.float32),
                    ),
                    axis=1,
                )
            )
        decision_ticks.append(executed)
        decision_index = len(decision_actions)
        decision_steps += env.num_envs
        physics_steps += int(executed.sum())
        episode_ticks += executed
        ended = terminated | truncated
        if pedal_hold is not None:
            pedal_hold.advance(ended)
        for index in np.flatnonzero(ended):
            index = int(index)
            native_ticks = int(info["completed_episode_ticks"][index])
            if native_ticks != int(episode_ticks[index]):
                raise EvaluationIntegrityError(
                    f"env {index} finished an episode of {native_ticks} native "
                    f"ticks but only {int(episode_ticks[index])} were stepped "
                    "here; the episode did not start at the official spawn"
                )
            if completed < episode_count:
                reason = int(info["termination_reason"][index])
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                distances.append(
                    float(
                        np.clip(
                            env.final_observations["race"][index, 4],
                            0.0,
                            env.route_length * env.lap_count,
                        )
                    )
                )
                if reason == FINISH_REASON:
                    finishes += 1
                    lap_ms = int(info["race_time_ms"][index])
                    if lap_ms != native_ticks * 10:
                        raise EvaluationIntegrityError(
                            f"race time {lap_ms} ms does not match "
                            f"{native_ticks} ticks"
                        )
                    lap_times.append(lap_ms)
                    if best_lap_ms is None or lap_ms < best_lap_ms:
                        best_lap_ms = lap_ms
                        start = int(episode_start_decision[index])
                        best_schedule = InputSchedule.from_decisions(
                            mode,
                            [
                                decision_actions[step][index]
                                for step in range(start, decision_index)
                            ],
                            [
                                int(decision_ticks[step][index])
                                for step in range(start, decision_index)
                            ],
                        )
                        if best_schedule.tick_count != native_ticks:
                            raise EvaluationIntegrityError(
                                "recorded schedule length does not match the lap"
                            )
                completed += 1
            episode_start_decision[index] = decision_index
            episode_ticks[index] = 0
    torch.cuda.synchronize()
    evaluation_seconds = time.perf_counter() - evaluation_start
    finished: bool | None = None
    if not sampled:
        # Every greedy episode starts at the same state under the same argmax
        # policy on deterministic physics, so they are one trajectory. Anything
        # else means the environment or the policy is not deterministic.
        if finishes not in (0, completed) or len(set(distances)) != 1:
            raise EvaluationIntegrityError(
                f"greedy trajectories diverged: {finishes}/{completed} finished, "
                f"{len(set(distances))} distinct distances"
            )
        finished = finishes == completed
    return EvaluationResult(
        mode="sampled" if sampled else "greedy",
        finished=finished,
        episodes=completed,
        finishes=finishes,
        finish_rate=finishes / completed,
        distance_mean=float(np.mean(distances)),
        distance_p90=float(np.percentile(distances, 90.0)),
        best_distance=max(distances),
        median_lap_ms=float(np.median(lap_times)) if lap_times else None,
        best_lap_ms=best_lap_ms,
        physics_steps=physics_steps,
        decision_steps=decision_steps,
        evaluation_seconds=evaluation_seconds,
        physics_steps_per_second=physics_steps / evaluation_seconds,
        termination_reason_counts={
            str(key): value for key, value in reason_counts.items()
        },
        lap_times_ms=lap_times,
        distances=distances,
        best_schedule=best_schedule,
    )


def replay_schedule(
    schedule: InputSchedule,
    *,
    track: str,
    library_path: Any,
    max_race_ticks: int,
    horizon_ticks: int,
    off_track_grace_ticks: int,
    stuck_grace_ticks: int,
    stuck_progress_epsilon: float,
    discount_per_tick: float,
    root: Any = None,
) -> dict[str, Any]:
    """Replay a schedule tick by tick and return exact checkpoint splits."""
    env = TmnfVectorEnv(
        1,
        track=track,
        thread_count=1,
        action_repeat=1,
        action_space=schedule.mode,
        max_race_ticks=max_race_ticks,
        horizon_ticks=horizon_ticks,
        off_track_grace_ticks=off_track_grace_ticks,
        stuck_grace_ticks=stuck_grace_ticks,
        stuck_progress_epsilon=stuck_progress_epsilon,
        discount_per_tick=discount_per_tick,
        library_path=library_path,
        root=root,
    )
    try:
        observations, _ = env.reset()
        previous_fraction = float(observations["race"][0, 9])
        splits_ms: list[int] = []
        finish_ms: int | None = None
        reason: int | None = None
        for tick in range(schedule.tick_count):
            if schedule.mode == "discrete":
                action = np.array([int(schedule.actions[tick])], dtype=np.int64)
            else:
                steer, gas, brake = schedule.actions[tick]
                action = {
                    "steer": np.array([steer], dtype=np.float32),
                    "gas": np.array([int(gas)], dtype=np.int8),
                    "brake": np.array([int(brake)], dtype=np.int8),
                }
            observations, _, terminated, truncated, info = env.step(action)
            ended = bool(terminated[0] or truncated[0])
            race = env.final_observations["race"] if ended else observations["race"]
            fraction = float(race[0, 9])
            if fraction > previous_fraction:
                splits_ms.append((tick + 1) * 10)
            previous_fraction = fraction
            if ended:
                reason = int(info["termination_reason"][0])
                if reason == FINISH_REASON:
                    finish_ms = int(info["race_time_ms"][0])
                if tick != schedule.tick_count - 1:
                    raise EvaluationIntegrityError(
                        f"schedule ended at tick {tick + 1} of {schedule.tick_count}"
                    )
        return {
            "ticks": schedule.tick_count,
            "finish_ms": finish_ms,
            "termination_reason": reason,
            "termination_name": TERMINATION_NAMES.get(reason) if reason else None,
            "checkpoint_splits_ms": splits_ms,
            "checkpoint_count": env.checkpoint_count,
        }
    finally:
        env.close()


__all__ = [
    "EvaluationIntegrityError",
    "EvaluationResult",
    "evaluate_full_start",
    "replay_schedule",
]
