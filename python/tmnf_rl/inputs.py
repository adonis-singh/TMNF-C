"""Per-tick input schedules in the exact TMNFRaceInputs binary format.

The format is the one consumed by ``tests/replay_tick.c`` (``input_file``
mode), ``tools/export_viewer_scene.c`` and TMInterface replays: one 72-byte
``TMNFRaceInputs`` record per 10 ms physics tick, timestamped
``(tick + 1) * 10`` ms.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TICK_MS = 10
RECORD = struct.Struct("<IIiIIiIIfIIiIIiIIf")
RECORD_SIZE = RECORD.size
STEER_QUANTUM = 65536.0

if RECORD_SIZE != 0x48:
    raise RuntimeError(f"TMNFRaceInputs encoder is {RECORD_SIZE} bytes")


def quantize_steer(steer: float) -> float:
    """Mirror TmnfVecEnv_QuantizeAnalogSteer and analog_input exactly."""
    value = np.float32(steer)
    if not np.isfinite(value) or value < -1.0 or value > 1.0:
        raise ValueError("analog steer is outside [-1, 1]")
    scaled = np.float32(value * np.float32(STEER_QUANTUM))
    # roundf: half away from zero.
    rounded = np.sign(scaled) * np.floor(np.abs(scaled) + np.float32(0.5))
    return float(np.float32(-np.float32(rounded) / np.float32(STEER_QUANTUM)))


def encode_discrete(action: int | None, tick: int) -> bytes:
    timestamp = (tick + 1) * TICK_MS
    if action is None:
        steer_left = steer_right = accelerate = brake = 0
    else:
        if not 0 <= action < 12:
            raise ValueError(f"discrete action {action} is out of range")
        longitudinal = action // 3
        steering = action % 3
        steer_left = int(steering == 0)
        steer_right = int(steering == 2)
        accelerate = int(longitudinal in (1, 3))
        brake = int(longitudinal in (2, 3))
    return RECORD.pack(
        timestamp, 0, steer_left,
        timestamp, 0, steer_right,
        0, 0, 0.0,
        timestamp, 0, accelerate,
        timestamp, 0, brake,
        0, 0, 0.0,
    )


def encode_analog(steer: float, gas: int, brake: int, tick: int) -> bytes:
    timestamp = (tick + 1) * TICK_MS
    if gas not in (0, 1) or brake not in (0, 1):
        raise ValueError("gas and brake must be binary")
    return RECORD.pack(
        0, 0, 0,
        0, 0, 0,
        timestamp, 0, quantize_steer(steer),
        timestamp, 0, int(gas),
        timestamp, 0, int(brake),
        0, 0, 0.0,
    )


@dataclass(frozen=True)
class InputSchedule:
    """One control record per executed physics tick."""

    mode: str
    actions: np.ndarray  # discrete: int16 [ticks]; analog: float32 [ticks, 3]

    def __post_init__(self) -> None:
        if self.mode not in ("discrete", "analog"):
            raise ValueError("schedule mode must be discrete or analog")
        if self.mode == "discrete" and self.actions.ndim != 1:
            raise ValueError("discrete schedule must be a vector of actions")
        if self.mode == "analog" and self.actions.shape[1:] != (3,):
            raise ValueError("analog schedule must be [ticks, 3]")
        if self.actions.shape[0] == 0:
            raise ValueError("schedule has no ticks")

    @property
    def tick_count(self) -> int:
        return int(self.actions.shape[0])

    @staticmethod
    def from_decisions(
        mode: str,
        decisions: list[np.ndarray | int | tuple[float, int, int]],
        executed_ticks: list[int],
    ) -> "InputSchedule":
        if len(decisions) != len(executed_ticks):
            raise ValueError("decision and tick lists differ in length")
        rows: list = []
        for decision, ticks in zip(decisions, executed_ticks, strict=True):
            if ticks < 1:
                raise ValueError("every decision must execute at least one tick")
            rows.extend([decision] * int(ticks))
        if mode == "discrete":
            actions = np.asarray(rows, dtype=np.int16)
        else:
            actions = np.asarray(rows, dtype=np.float32).reshape(-1, 3)
        return InputSchedule(mode, actions)

    def to_bytes(self, total_ticks: int | None = None) -> bytes:
        total = self.tick_count if total_ticks is None else int(total_ticks)
        if total < self.tick_count:
            raise ValueError(
                f"requested {total} schedule ticks for a "
                f"{self.tick_count}-tick schedule"
            )
        out = bytearray()
        if self.mode == "discrete":
            for tick, action in enumerate(self.actions.tolist()):
                out.extend(encode_discrete(int(action), tick))
        else:
            for tick, (steer, gas, brake) in enumerate(self.actions.tolist()):
                out.extend(encode_analog(steer, int(gas), int(brake), tick))
        for tick in range(self.tick_count, total):
            out.extend(encode_discrete(None, tick))
        return bytes(out)

    def write(self, path: Path, total_ticks: int | None = None) -> str:
        payload = self.to_bytes(total_ticks)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
        return hashlib.sha256(payload).hexdigest()

    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()


def decode_discrete_schedule(
    payload: bytes, tick_count: int | None = None
) -> InputSchedule:
    """Inverse of ``InputSchedule.to_bytes`` for pure discrete schedules.

    ``tick_count`` is the number of driven ticks; without it, trailing
    released-input records are treated as padding (a lap that ends with the
    coast/neutral action would then be shortened). Fails if the bytes are not
    exactly what ``to_bytes`` would produce for the decoded actions.
    """
    if len(payload) % RECORD_SIZE != 0:
        raise ValueError("payload is not a whole number of TMNFRaceInputs records")
    words = np.frombuffer(payload, dtype=np.dtype("<u4")).reshape(-1, RECORD_SIZE // 4)
    steer_left = words[:, 2].astype(np.int32)
    steer_right = words[:, 5].astype(np.int32)
    accelerate = words[:, 11].astype(np.int32)
    brake = words[:, 14].astype(np.int32)
    steering = np.where(steer_left == 1, 0, np.where(steer_right == 1, 2, 1))
    actions = ((accelerate + 2 * brake) * 3 + steering).astype(np.int16)
    if tick_count is None:
        active = np.flatnonzero((steer_left | steer_right | accelerate | brake) != 0)
        if active.size == 0:
            raise ValueError("schedule has no active inputs")
        tick_count = int(active[-1]) + 1
    if not 0 < tick_count <= len(actions):
        raise ValueError(f"tick_count {tick_count} is outside the payload")
    schedule = InputSchedule("discrete", actions[:tick_count])
    if schedule.to_bytes(len(actions)) != payload:
        raise ValueError("payload is not a pure discrete-action schedule")
    return schedule


def decode_schedule(payload: bytes) -> InputSchedule:
    """Decode a TMNFRaceInputs file into the per-tick controls the car saw.

    Each tick's steer is resolved the way the game's input mapper does it
    (``vehicle.c``): the analog word wins when its timestamp is newer than
    both digital steer timestamps (or equal to them with no digital press
    and |value| > 0.01), otherwise left / right / none. Files with an analog
    steer on any tick decode to an analog schedule (digital ticks become
    -1 / 0 / +1), all-digital files to a discrete one. Analog gas is
    refused (never used by the exporters or the captured replays). Every
    record is kept, including the exporters' released-input padding: a lap
    may end on a coasting tick, so a replay must stop at the finish, not at
    the last active input. The file stores the game's sign-flipped steer;
    the returned steer is the value the env receives.
    """
    if len(payload) % RECORD_SIZE:
        raise ValueError("payload is not a whole number of TMNFRaceInputs records")
    words = np.frombuffer(payload, dtype="<u4").reshape(-1, RECORD_SIZE // 4)
    floats = np.frombuffer(payload, dtype="<f4").reshape(-1, RECORD_SIZE // 4)
    if (words[:, 15] != 0).any():
        raise ValueError("analog gas schedules are not supported")
    if not (words[:, 2] | words[:, 5] | words[:, 11] | words[:, 14] | words[:, 6]).any():
        raise ValueError("schedule has no active inputs")
    ticks = words.shape[0]
    left, right, analog_time = words[:ticks, 2], words[:ticks, 5], words[:ticks, 6]
    analog_value = floats[:ticks, 8]
    latest = np.maximum(words[:ticks, 0], words[:ticks, 3])
    analog_wins = (analog_time > latest) | (
        (analog_time == latest) & (left == 0) & (right == 0) & (np.abs(analog_value) > 0.01)
    )
    if not analog_wins.any():
        return decode_discrete_schedule(payload, ticks)
    digital = np.where(left != 0, -1.0, np.where(right != 0, 1.0, 0.0))
    steer = np.where(analog_wins, -analog_value.astype(np.float64), digital)
    actions = np.stack(
        [steer, words[:ticks, 11].astype(np.float64), words[:ticks, 14].astype(np.float64)], axis=1
    ).astype(np.float32)
    return InputSchedule("analog", actions)


__all__ = [
    "InputSchedule",
    "decode_discrete_schedule",
    "decode_schedule",
    "RECORD_SIZE",
    "TICK_MS",
    "encode_analog",
    "encode_discrete",
    "quantize_steer",
]
