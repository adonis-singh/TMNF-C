"""Car pose out of a TmnfVectorEnv, in the fly head frame.

A TmnfEnvSnapshot (src/vec_env.h) starts with a u32 version followed by the
live CHmsStateDyna (src/hms_state.h): quaternion (16 B), 3x3 rotation (36 B,
row-major, world = R @ local), position (12 B), linear velocity (12 B). The
car's local axes are x = left, y = up, z = forward (measured: the rotation's
third column aligns with the velocity of a car driving straight, the second
with world up). The head frame used by the eye is x forward, y left, z up.
"""

from __future__ import annotations

import numpy as np

from tmnf_rl.env import TmnfVectorEnv

_POSE = slice(4, 80)


def car_poses(env: TmnfVectorEnv, indices: list[int] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(positions (N,3), head_to_world (N,3,3), velocities (N,3)) for the given envs."""
    indices = list(range(env.num_envs)) if indices is None else indices
    blobs = env.capture(indices)
    positions = np.empty((len(blobs), 3), np.float32)
    rotations = np.empty((len(blobs), 3, 3), np.float32)
    velocities = np.empty((len(blobs), 3), np.float32)
    for i, blob in enumerate(blobs):
        raw = np.frombuffer(bytes(blob[_POSE]), np.float32)
        rot = raw[4:13].reshape(3, 3)
        positions[i] = raw[13:16]
        velocities[i] = raw[16:19]
        rotations[i] = head_to_world(rot)
    return positions, rotations, velocities


def head_to_world(car_rotation: np.ndarray) -> np.ndarray:
    """Columns: fly forward = car local z, fly left = car local x, fly up = car local y."""
    return np.stack((car_rotation[:, 2], car_rotation[:, 0], car_rotation[:, 1]), axis=1)


__all__ = ["car_poses", "head_to_world"]
