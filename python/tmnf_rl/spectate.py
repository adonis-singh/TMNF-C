"""Read-only live telemetry ring and localhost HTTP server for PPO training."""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np


PROTOCOL_VERSION = 1
RING_CAPACITY = 2048
MAX_RESPONSE_FRAMES = 256
ROTATION_DECISIONS = 600
TERMINATION_NAMES = {
    1: "finish",
    2: "timeout",
    3: "off_track",
    4: "stuck",
    5: "fell",
}


class SpectateBuffer:
    """Single-producer ring copied from already-published environment views."""

    def __init__(
        self,
        *,
        num_envs: int,
        stream_envs: int,
        action_repeat: int,
        max_race_ticks: int,
        initial_meta: dict[str, Any],
    ) -> None:
        self.stream_count = min(stream_envs, num_envs)
        self.action_repeat = action_repeat
        self.max_race_ticks = max_race_ticks
        self._num_envs = num_envs
        self._sequence = np.full(RING_CAPACITY, -1, dtype=np.int64)
        self._env_ids = np.zeros(
            (RING_CAPACITY, self.stream_count), dtype=np.int32
        )
        self._episode_ids = np.zeros_like(self._env_ids, dtype=np.uint64)
        self._ticks = np.zeros_like(self._env_ids, dtype=np.int32)
        self._gear = np.zeros_like(self._env_ids, dtype=np.int32)
        self._reason = np.zeros_like(self._env_ids, dtype=np.int8)
        self._vehicle = np.zeros(
            (RING_CAPACITY, self.stream_count, 34), dtype=np.float32
        )
        self._race = np.zeros(
            (RING_CAPACITY, self.stream_count, 11), dtype=np.float32
        )
        self._selected = np.arange(self.stream_count, dtype=np.int32)
        self._latest_sequence = -1
        self._meta = dict(initial_meta)

    @property
    def latest_sequence(self) -> int:
        return self._latest_sequence

    @property
    def selected_envs(self) -> list[int]:
        return [int(index) for index in self._selected]

    def update_meta(self, values: dict[str, Any]) -> None:
        updated = dict(self._meta)
        updated.update(values)
        self._meta = updated

    def meta(self) -> dict[str, Any]:
        return {
            **self._meta,
            "protocolVersion": PROTOCOL_VERSION,
            "envIndices": self.selected_envs,
            "tickRateHz": 100,
            "decisionIntervalMs": self.action_repeat * 10,
            "ringCapacity": RING_CAPACITY,
            "latestSequence": self._latest_sequence,
        }

    def _rotate(self, progress: np.ndarray, sequence: int) -> None:
        best = int(np.argmax(progress))
        selected = [best]
        candidate = (sequence // ROTATION_DECISIONS * 3) % self._num_envs
        while len(selected) < self.stream_count:
            index = int(candidate % self._num_envs)
            if index not in selected:
                selected.append(index)
            candidate += 1
        self._selected[:] = selected

    def capture(
        self,
        *,
        observations: dict[str, np.ndarray],
        final_observations: dict[str, np.ndarray],
        ended: np.ndarray,
        termination_reasons: np.ndarray,
        episode_ids: np.ndarray,
        progress: np.ndarray,
    ) -> None:
        sequence = self._latest_sequence + 1
        if sequence % ROTATION_DECISIONS == 0:
            self._rotate(progress, sequence)

        slot = sequence % RING_CAPACITY
        self._sequence[slot] = -1
        for stream_index, env_index_value in enumerate(self._selected):
            env_index = int(env_index_value)
            source = final_observations if bool(ended[env_index]) else observations
            self._env_ids[slot, stream_index] = env_index
            self._episode_ids[slot, stream_index] = episode_ids[env_index]
            self._vehicle[slot, stream_index] = source["vehicle"][env_index]
            self._race[slot, stream_index] = source["race"][env_index]
            self._gear[slot, stream_index] = source["gear"][env_index]
            self._ticks[slot, stream_index] = int(
                np.rint(
                    source["race"][env_index, 8] * self.max_race_ticks
                )
            )
            self._reason[slot, stream_index] = (
                int(termination_reasons[env_index])
                if bool(ended[env_index])
                else 0
            )
        self._sequence[slot] = sequence
        self._latest_sequence = sequence

    def frames_since(self, since: int) -> dict[str, Any]:
        newest = self._latest_sequence
        if newest < 0:
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "oldestSequence": 0,
                "latestSequence": -1,
                "dropped": 0,
                "frames": [],
            }

        oldest = max(0, newest - RING_CAPACITY + 1)
        requested = since + 1
        dropped = max(0, oldest - requested)
        start = max(requested, oldest, newest - MAX_RESPONSE_FRAMES + 1)
        dropped += max(0, start - max(requested, oldest))
        frames = []
        for sequence in range(start, newest + 1):
            slot = sequence % RING_CAPACITY
            if int(self._sequence[slot]) != sequence:
                continue
            envs = [
                self._serialize_env(slot, stream_index)
                for stream_index in range(self.stream_count)
            ]
            if int(self._sequence[slot]) != sequence:
                continue
            frames.append({"sequence": sequence, "envs": envs})
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "oldestSequence": oldest,
            "latestSequence": newest,
            "dropped": dropped,
            "frames": frames,
        }

    def _serialize_env(self, slot: int, stream_index: int) -> dict[str, Any]:
        vehicle = self._vehicle[slot, stream_index]
        race = self._race[slot, stream_index]
        reason = int(self._reason[slot, stream_index])
        wheels = [
            {
                "contact": bool(vehicle[21 + wheel] >= 0.5),
                "sliding": bool(vehicle[25 + wheel] >= 0.5),
                "steer": None,
                "damper": float(vehicle[17 + wheel]),
                "speed": float(vehicle[13 + wheel]),
            }
            for wheel in range(4)
        ]
        speed = math.sqrt(
            float(vehicle[7]) ** 2
            + float(vehicle[8]) ** 2
            + float(vehicle[9]) ** 2
        )
        return {
            "envId": int(self._env_ids[slot, stream_index]),
            "episodeId": int(self._episode_ids[slot, stream_index]),
            "tick": int(self._ticks[slot, stream_index]),
            "raceTimeMs": int(self._ticks[slot, stream_index]) * 10,
            "position": [float(value) for value in vehicle[0:3]],
            "quaternion": [float(value) for value in vehicle[3:7]],
            "speedMps": speed,
            "rpm": float(vehicle[33]),
            "gear": int(self._gear[slot, stream_index]),
            "inputSteer": float(race[0]),
            "distance": float(race[4]),
            "wheels": wheels,
            "reset": reason != 0,
            "terminationReason": TERMINATION_NAMES.get(reason),
        }


class _SpectateHttpServer(HTTPServer):
    buffer: SpectateBuffer


class _SpectateHandler(BaseHTTPRequestHandler):
    server: _SpectateHttpServer

    def do_GET(self) -> None:
        request = urlsplit(self.path)
        try:
            if request.path == "/spectate/meta":
                self._reply(self.server.buffer.meta())
                return
            if request.path == "/spectate/frames":
                values = parse_qs(request.query)
                raw_since = values.get("since", ["-1"])
                if len(raw_since) != 1:
                    raise ValueError("since must appear once")
                self._reply(self.server.buffer.frames_since(int(raw_since[0])))
                return
            self.send_error(404)
        except (TypeError, ValueError) as error:
            self.send_error(400, str(error))

    def _reply(self, payload: dict[str, Any]) -> None:
        body = json.dumps(
            payload, separators=(",", ":"), allow_nan=False
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        del format, args


class SpectateServer:
    def __init__(self, port: int, buffer: SpectateBuffer) -> None:
        server = _SpectateHttpServer(("127.0.0.1", port), _SpectateHandler)
        server.buffer = buffer
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            name="tmnf-spectate-http",
            daemon=True,
        )

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()


__all__ = [
    "PROTOCOL_VERSION",
    "RING_CAPACITY",
    "SpectateBuffer",
    "SpectateServer",
]
