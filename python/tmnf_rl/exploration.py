"""Exploration bonuses and temporally extended pedal actions.

``LandingNovelty``: a count-based bonus on a physical descriptor of each
landing, (20 m progress bin, airborne decisions, 2 m/s landing speed bin).
A landing after at least ``MIN_AIR_DECISIONS`` airborne decisions adds
``coef / sqrt(n)`` to that decision's reward, ``n`` the visits of the cell
so far in the run. Landings the policy has not produced before pay; the
usual landing stops paying after a few hundred visits.

``PedalHold``: gas and brake are decided every ``period`` decisions of an
episode and held between; steer is decided every decision. The policy's
pedal log-probability and entropy are masked out on held decisions
(``Agent.get_action_and_value(pedal_mask=...)``), so a held decision
carries only the steer term.

Both are deterministic given the rollout (counts and phases are plain
state and are checkpointed), so same-seed runs reproduce bit for bit.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

# Flat policy observation columns (src/vec_env.c flatten_observation).
LINEAR_SPEED = slice(7, 10)
WHEEL_CONTACT = slice(21, 25)
UNWRAPPED_PROGRESS = 39
NOVELTY_PROGRESS_BIN_METERS = 20.0
NOVELTY_SPEED_BIN = 2.0
MIN_AIR_DECISIONS = 2


class LandingNovelty:
    def __init__(self, num_envs: int, coef: float) -> None:
        if coef <= 0.0:
            raise ValueError("novelty coefficient must be positive")
        self.coef = coef
        self.counts: dict[tuple[int, int, int], int] = {}
        self.air_decisions = np.zeros(num_envs, dtype=np.int32)
        self.bonus_total = 0.0
        self.landings = 0

    def step(self, observation: np.ndarray, ended: np.ndarray) -> np.ndarray:
        """Bonus per environment for the decision that produced
        ``observation`` (the flat batch after the step). ``ended`` marks
        environments whose episode ended on this step: their observation is a
        new episode's first, so no landing is scored and the counter resets."""
        airborne = observation[:, WHEEL_CONTACT].max(axis=1) <= 0.0
        landed = (~airborne) & (self.air_decisions >= MIN_AIR_DECISIONS) & ~ended
        bonus = np.zeros(observation.shape[0], dtype=np.float32)
        for index in np.flatnonzero(landed):
            speed = float(np.linalg.norm(observation[index, LINEAR_SPEED]))
            cell = (
                int(observation[index, UNWRAPPED_PROGRESS] // NOVELTY_PROGRESS_BIN_METERS),
                int(self.air_decisions[index]),
                int(speed // NOVELTY_SPEED_BIN),
            )
            count = self.counts.get(cell, 0) + 1
            self.counts[cell] = count
            bonus[index] = self.coef / math.sqrt(count)
            self.landings += 1
        self.bonus_total += float(bonus.sum())
        self.air_decisions = np.where(airborne & ~ended, self.air_decisions + 1, 0).astype(np.int32)
        return bonus

    def metrics(self) -> dict[str, float | int]:
        return {
            "novelty_cells": len(self.counts),
            "novelty_landings": self.landings,
            "novelty_bonus_total": self.bonus_total,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "counts": [(list(cell), count) for cell, count in self.counts.items()],
            "air_decisions": self.air_decisions.copy(),
            "bonus_total": self.bonus_total,
            "landings": self.landings,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.counts = {tuple(int(v) for v in cell): int(count) for cell, count in state["counts"]}
        self.air_decisions = np.asarray(state["air_decisions"], dtype=np.int32)
        self.bonus_total = float(state["bonus_total"])
        self.landings = int(state["landings"])


class PedalHold:
    def __init__(self, num_envs: int, period: int, device: torch.device) -> None:
        if period < 2:
            raise ValueError("pedal hold period must be at least 2 decisions")
        self.period = period
        self.phase = np.zeros(num_envs, dtype=np.int32)
        self.held = torch.zeros((num_envs, 2), device=device)
        self.device = device

    def mask(self) -> torch.Tensor:
        """True where this decision decides the pedals."""
        return torch.as_tensor(self.phase == 0, device=self.device)

    def apply(self, action: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Replace the pedals of held decisions by the held values and record
        the decided ones; returns the action the environment executes."""
        applied = action.clone()
        applied[:, 1:] = torch.where(mask.unsqueeze(1), action[:, 1:], self.held)
        self.held = applied[:, 1:].clone()
        return applied

    def advance(self, ended: np.ndarray) -> None:
        self.phase = np.where(ended, 0, (self.phase + 1) % self.period).astype(np.int32)

    def state_dict(self) -> dict[str, Any]:
        return {"phase": self.phase.copy(), "held": self.held.cpu()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.phase = np.asarray(state["phase"], dtype=np.int32)
        self.held = torch.as_tensor(state["held"], device=self.device)


def entropy_weights(observation: torch.Tensor, window: tuple[float, float], boost: float) -> torch.Tensor:
    """Per-row entropy weight 1 + boost inside the progress window, 1 outside."""
    progress = observation[:, UNWRAPPED_PROGRESS]
    inside = (progress >= window[0]) & (progress < window[1])
    return 1.0 + boost * inside.to(observation.dtype)


def parse_window(text: str) -> tuple[float, float]:
    low, high = (float(part) for part in text.split(","))
    if not low < high:
        raise ValueError("entropy boost window must be 'low,high' metres with low < high")
    return low, high


@torch.no_grad()
def ema_update(average: torch.nn.Module, live: torch.nn.Module, decay: float) -> None:
    for target, source in zip(average.parameters(), live.parameters(), strict=True):
        target.mul_(decay).add_(source, alpha=1.0 - decay)


FINISH_REASON = 1  # TMNF_TERMINATION_FINISH (tmnf_rl.env)
RACE_TICK_MS = 10


def finish_time_rewards(
    ended: np.ndarray, termination_reasons: np.ndarray, race_time_ms: np.ndarray,
    max_race_ticks: int,
) -> np.ndarray:
    """Finish-time-only reward of one decision: 0 for every environment except
    those whose episode just ended with a finish, which receive the unused race
    budget in seconds, (max_race_ticks - lap ticks) / 100. Failures pay 0, so
    every finish outranks every failure and 10 ms of lap time is 0.01."""
    rewards = np.zeros(ended.shape[0], dtype=np.float32)
    finished = ended & (np.asarray(termination_reasons) == FINISH_REASON)
    lap_ticks = np.asarray(race_time_ms, dtype=np.float64)[finished] / RACE_TICK_MS
    rewards[finished] = (max_race_ticks - lap_ticks) / 100.0
    return rewards


__all__ = [
    "LandingNovelty",
    "MIN_AIR_DECISIONS",
    "NOVELTY_PROGRESS_BIN_METERS",
    "NOVELTY_SPEED_BIN",
    "PedalHold",
    "UNWRAPPED_PROGRESS",
    "WHEEL_CONTACT",
    "ema_update",
    "entropy_weights",
    "finish_time_rewards",
    "parse_window",
]
