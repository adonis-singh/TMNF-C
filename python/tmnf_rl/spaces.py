"""Observation flattening and action conversion between Torch and the env."""

from __future__ import annotations

import numpy as np
import torch

from tmnf_rl.env import POLICY_OBSERVATION_WIDTH, TmnfVectorEnv


OBSERVATION_WIDTH = POLICY_OBSERVATION_WIDTH
LEGACY_OBSERVATION_WIDTH = 78
ACTION_SPACES = ("discrete", "analog")


def observation_tensor(
    observation: dict[str, np.ndarray], device: torch.device
) -> torch.Tensor:
    vehicle = torch.as_tensor(observation["vehicle"], device=device)
    gear = torch.as_tensor(observation["gear"], device=device).to(torch.float32)
    race = torch.as_tensor(observation["race"], device=device)
    centerline = torch.as_tensor(observation["centerline"], device=device)
    turbo = torch.as_tensor(observation["turbo"], device=device)
    flattened = torch.cat(
        (
            vehicle,
            gear.unsqueeze(1),
            race,
            centerline.flatten(start_dim=1),
            turbo,
        ),
        dim=1,
    )
    if "gates" in observation:
        gates = torch.as_tensor(observation["gates"], device=device)
        counts = torch.as_tensor(observation["gate_counts"], device=device)
        flattened = torch.cat(
            (flattened, gates.flatten(start_dim=1), counts.to(torch.float32)), dim=1
        )
    return flattened


def environment_action(
    action: torch.Tensor, action_space: str
) -> np.ndarray | dict[str, np.ndarray]:
    if action_space == "discrete":
        return action.detach().cpu().numpy()
    if action_space != "analog":
        raise ValueError("action_space must be discrete or analog")
    # Column 0 is the pre-tanh steer (what the policy stores and scores);
    # the env takes the squashed value in [-1, 1].
    values = action.detach()
    values = torch.cat((torch.tanh(values[:, :1]), values[:, 1:]), dim=1).cpu().numpy()
    return {
        "steer": values[:, 0],
        "gas": values[:, 1].astype(np.int8),
        "brake": values[:, 2].astype(np.int8),
    }


def device_environment_action(action: torch.Tensor, action_space: str) -> torch.Tensor:
    """``environment_action`` for the CUDA env: the same conversion (tanh on
    the pre-tanh steer column) as a device tensor, no host round trip."""
    if action_space == "discrete":
        return action.detach()
    if action_space != "analog":
        raise ValueError("action_space must be discrete or analog")
    values = action.detach()
    return torch.cat((torch.tanh(values[:, :1]), values[:, 1:]), dim=1)


def forward_action(env: TmnfVectorEnv) -> np.ndarray | dict[str, np.ndarray]:
    if env.action_space_mode == "discrete":
        return np.full(env.num_envs, 4, dtype=np.int32)
    return {
        "steer": np.zeros(env.num_envs, dtype=np.float32),
        "gas": np.ones(env.num_envs, dtype=np.int8),
        "brake": np.zeros(env.num_envs, dtype=np.int8),
    }


__all__ = [
    "ACTION_SPACES",
    "LEGACY_OBSERVATION_WIDTH",
    "OBSERVATION_WIDTH",
    "device_environment_action",
    "environment_action",
    "forward_action",
    "observation_tensor",
]
