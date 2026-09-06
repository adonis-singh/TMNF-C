#!/usr/bin/env python3
"""Generate the hand-scripted technique-regime input schedules."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct


INPUT_STRUCT = struct.Struct("<IIiIIiIIfIIiIIiIIf")
Segment = tuple[int, bool, bool, int]
ControlEvent = tuple[int, str, bool]

A10_AUTHOR_EVENTS: tuple[ControlEvent, ...] = (
    (10, "accelerate", True),
    (648, "left", True),
    (684, "left", False),
    (856, "left", True),
    (938, "left", False),
    (1325, "left", True),
    (1481, "left", False),
    (1675, "left", True),
    (1715, "left", False),
    (2259, "left", True),
    (2298, "left", False),
    (2481, "left", True),
    (2559, "left", False),
    (3590, "right", True),
    (3672, "right", False),
    (4071, "right", True),
    (4228, "right", False),
    (4308, "right", True),
    (4426, "right", False),
    (4809, "right", True),
    (4953, "right", False),
    (5071, "right", True),
    (5168, "right", False),
    (5677, "left", True),
    (5731, "left", False),
    (6700, "accelerate", False),
    (8589, "accelerate", True),
)


SCHEDULES: dict[str, tuple[Segment, ...]] = {
    "a10_mixed": (
        (180, True, False, 0),
        (140, True, False, -1),
        (140, True, False, 1),
        (80, True, True, -1),
        (80, True, True, 1),
        (120, False, True, 0),
        (180, True, False, 0),
        (160, True, False, 1),
        (120, False, False, 0),
    ),
    "a10_wall_contact": (
        (220, True, False, 0),
        (320, True, False, 1),
        (80, True, True, 1),
        (220, True, False, -1),
        (80, True, True, -1),
        (320, True, False, 1),
        (160, False, False, 0),
    ),
    "wall_contact": (
        (180, True, False, 0),
        (90, True, False, -1),
        (50, True, False, -1),
        (20, True, False, 0),
        (50, True, False, -1),
        (20, True, False, 1),
        (50, True, False, -1),
        (20, True, False, 0),
        (50, True, False, -1),
        (300, True, False, 1),
        (50, True, False, -1),
        (180, False, False, 0),
    ),
    "speedslide": (
        (220, True, False, 0),
        (25, True, False, -1),
        (5, True, False, 0),
        (25, True, False, 1),
        (5, True, False, 0),
        (25, True, False, -1),
        (5, True, False, 0),
        (25, True, False, 1),
        (5, True, False, 0),
        (25, True, False, -1),
        (5, True, False, 0),
        (25, True, False, 1),
        (5, True, False, 0),
        (25, True, False, -1),
        (5, True, False, 0),
        (25, True, False, 1),
        (5, True, False, 0),
        (25, True, False, -1),
        (5, True, False, 0),
        (25, True, False, 1),
        (5, True, False, 0),
        (300, True, False, 0),
    ),
    "neoslide": (
        (50, True, False, 0),
        (18, True, False, -1),
        (6, True, False, 0),
        (18, True, True, -1),
        (35, True, False, 0),
        (22, True, False, 1),
        (9, True, False, 0),
        (22, True, True, 1),
        (40, True, False, 0),
        (26, True, False, -1),
        (12, True, False, 0),
        (26, True, True, -1),
        (40, True, False, 0),
        (30, True, False, 1),
        (15, True, False, 0),
        (30, True, True, 1),
        (200, True, False, 0),
    ),
    "gas_brake": (
        (150, True, False, 0),
        (100, True, True, -1),
        (50, True, True, 0),
        (100, True, True, 1),
        (50, True, True, 0),
        (100, True, True, -1),
        (50, True, True, 0),
        (100, True, True, 1),
        (200, False, False, 0),
    ),
    "airborne_landing": (
        (700, True, False, 0),
        (300, False, False, 0),
    ),
    "reverse_recovery": (
        (150, True, False, 0),
        (350, False, True, 0),
        (30, False, False, 0),
        (500, True, False, 0),
        (200, False, False, 0),
    ),
}


def encode_schedule(segments: tuple[Segment, ...]) -> bytes:
    output = bytearray()
    tick = 0
    for count, accelerate, brake, steer in segments:
        for _ in range(count):
            timestamp = (tick + 1) * 10
            output.extend(
                INPUT_STRUCT.pack(
                    timestamp,
                    0,
                    int(steer < 0),
                    timestamp,
                    0,
                    int(steer > 0),
                    0,
                    0,
                    0.0,
                    timestamp,
                    0,
                    int(accelerate),
                    timestamp,
                    0,
                    int(brake),
                    0,
                    0,
                    0.0,
                )
            )
            tick += 1
    return bytes(output)


def encode_events(events: tuple[ControlEvent, ...], tick_count: int) -> bytes:
    state = {"accelerate": False, "brake": False, "left": False, "right": False}
    event_index = 0
    output = bytearray()
    for tick in range(tick_count):
        timestamp = (tick + 1) * 10
        while event_index < len(events) and events[event_index][0] <= timestamp:
            _, name, enabled = events[event_index]
            state[name] = enabled
            event_index += 1
        steer = -1 if state["left"] else 1 if state["right"] else 0
        output.extend(
            encode_schedule(
                ((1, state["accelerate"], state["brake"], steer),)
            )
        )
        struct.pack_into("<I", output, len(output) - INPUT_STRUCT.size, timestamp)
        struct.pack_into("<I", output, len(output) - INPUT_STRUCT.size + 12, timestamp)
        struct.pack_into("<I", output, len(output) - INPUT_STRUCT.size + 36, timestamp)
        struct.pack_into("<I", output, len(output) - INPUT_STRUCT.size + 48, timestamp)
    return bytes(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("schedule", choices=sorted((*SCHEDULES, "a10_author")))
    parser.add_argument("output", type=Path)
    parser.add_argument("--ticks", type=int)
    args = parser.parse_args()
    data = (
        encode_events(A10_AUTHOR_EVENTS, 1000)
        if args.schedule == "a10_author"
        else encode_schedule(SCHEDULES[args.schedule])
    )
    if args.ticks is not None:
        if args.ticks <= 0 or args.ticks * INPUT_STRUCT.size > len(data):
            raise ValueError("requested tick prefix lies outside the schedule")
        data = data[: args.ticks * INPUT_STRUCT.size]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    print(f"{args.schedule}: {len(data) // INPUT_STRUCT.size} ticks -> {args.output}")


if __name__ == "__main__":
    main()
