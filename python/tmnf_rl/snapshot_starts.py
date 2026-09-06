"""Snapshot starts: a pool of route states from the agent's own best
trajectories. Nothing else ever enters the pool.

During training a fraction ``snapshot_start_fraction`` of episode resets
restores one of these states instead of the grid start. The pool holds at
most one state per ``POOL_BIN_METERS`` of route progress, taken from the
best-scoring trajectory that captured a state in that bin, and is rebuilt
every ``REFRESH_UPDATES`` updates from the trajectories completed since the
last refresh plus the previous pool (so a bin never gets worse).

Every snapshot is a native ``TmnfEnvSnapshot`` (pointer-free, cross-env
restorable), which carries the race clock, so the restored observation's
elapsed fraction and race time are those of the captured tick by
construction. Evaluation environments never see the pool
(``evaluate_full_start`` refuses a restored environment, F9).

Deterministic: every choice comes from ``trainer_rng`` and the ranking from
recorded floats, so two same-seed runs and a resumed run reproduce bit for
bit (the pool, the candidates and the in-flight captures are checkpointed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


OFFICIAL_START = "official"
SNAPSHOT_START = "snapshot"
POOL_BIN_METERS = 20.0
REFRESH_UPDATES = 10
CANDIDATE_LIMIT = 64
OWN_SOURCE = "own"
# Captures closer than this to a failure are not start states: the car is
# already leaving the road. Finished trajectories keep every capture.
PREFAILURE_EXCLUSION_TICKS = 200


@dataclass
class PoolState:
    blob: bytes
    progress: float
    capture_tick: int
    score: float

    def meta(self) -> dict[str, Any]:
        return {"progress": self.progress, "capture_tick": self.capture_tick, "score": self.score}


@dataclass
class Trajectory:
    """Captures of one episode, scored when it ends."""

    states: list[PoolState] = field(default_factory=list)
    score: float = 0.0


def trajectory_score(
    *, final_progress: float, finished: bool, episode_ticks: int, max_race_ticks: int, reference_speed: float
) -> float:
    """Metres of route: progress, plus the budget a finish left unused at the
    reference speed so faster laps rank above slower ones and every finish
    above every failure."""
    if finished:
        return final_progress + (max_race_ticks - episode_ticks) * reference_speed / 100.0
    return final_progress


class SnapshotPool:
    def __init__(self, *, route_length: float, rng: np.random.Generator,
                 prefailure_exclusion_ticks: int = PREFAILURE_EXCLUSION_TICKS) -> None:
        if route_length <= 0.0:
            raise ValueError("route_length must be positive")
        if prefailure_exclusion_ticks < 1:
            raise ValueError("prefailure_exclusion_ticks must be positive")
        self.prefailure_exclusion_ticks = prefailure_exclusion_ticks
        self.route_length = route_length
        self.rng = rng
        self.states: list[PoolState] = []
        self.candidates: list[Trajectory] = []
        self.refreshes = 0

    def bin_of(self, progress: float) -> int:
        clipped = min(max(progress, 0.0), np.nextafter(self.route_length, 0.0))
        return int(clipped // POOL_BIN_METERS)

    def add_trajectory(
        self,
        trajectory: Trajectory,
        *,
        finished: bool,
        episode_ticks: int,
    ) -> None:
        if finished:
            usable = trajectory.states
        else:
            cutoff = episode_ticks - self.prefailure_exclusion_ticks
            usable = [state for state in trajectory.states if state.capture_tick <= cutoff]
        if not usable:
            return
        for state in usable:
            state.score = trajectory.score
        self.candidates.append(Trajectory(states=usable, score=trajectory.score))
        if len(self.candidates) > CANDIDATE_LIMIT:
            # Stable: ties keep the earlier trajectory.
            self.candidates.sort(key=lambda item: -item.score)
            del self.candidates[CANDIDATE_LIMIT:]

    def refresh(self) -> None:
        """One state per bin from the best trajectory covering it."""
        best: dict[int, PoolState] = {state_bin(self, state): state for state in self.states}
        for trajectory in sorted(self.candidates, key=lambda item: -item.score):
            for state in trajectory.states:
                index = self.bin_of(state.progress)
                current = best.get(index)
                if current is None or state.score > current.score:
                    best[index] = state
        self.states = [best[index] for index in sorted(best)]
        self.candidates = []
        self.refreshes += 1

    def choose(self) -> tuple[str, int, PoolState] | None:
        """(source, index, state) from the agent's own pool; one rng draw."""
        if not self.states:
            return None
        index = int(self.rng.integers(len(self.states)))
        return OWN_SOURCE, index, self.states[index]

    def metrics(self) -> dict[str, float | int]:
        return {
            "snapshot_pool_states": len(self.states),
            "snapshot_pool_candidates": len(self.candidates),
            "snapshot_pool_refreshes": self.refreshes,
            "snapshot_pool_max_progress": max((state.progress for state in self.states), default=0.0),
            "snapshot_pool_best_score": max((state.score for state in self.states), default=0.0),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "prefailure_exclusion_ticks": self.prefailure_exclusion_ticks,
            "states": [(state.blob, state.meta()) for state in self.states],
            "candidates": [
                (trajectory.score, [(state.blob, state.meta()) for state in trajectory.states])
                for trajectory in self.candidates
            ],
            "refreshes": self.refreshes,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("prefailure_exclusion_ticks", PREFAILURE_EXCLUSION_TICKS) != self.prefailure_exclusion_ticks:
            raise ValueError("snapshot pool recovery exclusion differs from checkpoint")
        self.states = [PoolState(bytes(blob), **meta) for blob, meta in state["states"]]
        self.candidates = [
            Trajectory(states=[PoolState(bytes(blob), **meta) for blob, meta in states], score=score)
            for score, states in state["candidates"]
        ]
        self.refreshes = int(state["refreshes"])


def state_bin(pool: SnapshotPool, state: PoolState) -> int:
    return pool.bin_of(state.progress)


__all__ = [
    "CANDIDATE_LIMIT",
    "OFFICIAL_START",
    "OWN_SOURCE",
    "POOL_BIN_METERS",
    "PREFAILURE_EXCLUSION_TICKS",
    "PoolState",
    "REFRESH_UPDATES",
    "SNAPSHOT_START",
    "SnapshotPool",
    "Trajectory",
    "trajectory_score",
]
