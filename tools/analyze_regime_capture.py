#!/usr/bin/env python3
"""Report technique-regime coverage in a 1,668-byte tick capture."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct
from typing import Callable


RECORD_SIZE = 1668
SCENE_OFFSET = 176
WHEELS_OFFSET = 356
WHEEL_SIZE = 328


def i32(record: memoryview, offset: int) -> int:
    return struct.unpack_from("<i", record, offset)[0]


def f32(record: memoryview, offset: int) -> float:
    return struct.unpack_from("<f", record, offset)[0]


def longest_run(values: list[bool]) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    start: int | None = None
    for index, value in enumerate((*values, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            candidate = (start, index)
            if best is None or candidate[1] - candidate[0] > best[1] - best[0]:
                best = candidate
            start = None
    return best


def format_run(run: tuple[int, int] | None) -> str:
    if run is None:
        return "none"
    start, end = run
    return f"{start + 1}-{end} ({end - start} ticks, {(end - start) * 10} ms)"


def collect(records: list[memoryview], predicate: Callable[[memoryview], bool]) -> list[bool]:
    return [predicate(record) for record in records]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument(
        "--controls",
        type=int,
        nargs=2,
        metavar=("FIRST_TICK", "LAST_TICK"),
    )
    parser.add_argument(
        "--wheels",
        type=int,
        nargs=2,
        metavar=("FIRST_TICK", "LAST_TICK"),
    )
    args = parser.parse_args()

    data = args.capture.read_bytes()
    if len(data) == 0 or len(data) % RECORD_SIZE != 0:
        raise ValueError("capture is not a whole number of 1,668-byte records")
    records = [
        memoryview(data)[offset : offset + RECORD_SIZE]
        for offset in range(0, len(data), RECORD_SIZE)
    ]

    gas_brake = collect(
        records,
        lambda record: f32(record, SCENE_OFFSET) == 1.0
        and f32(record, SCENE_OFFSET + 4) == 1.0,
    )
    global_sliding = collect(
        records, lambda record: i32(record, SCENE_OFFSET + 92) != 0
    )
    wheel_sliding = collect(
        records,
        lambda record: any(
            i32(record, WHEELS_OFFSET + wheel * WHEEL_SIZE + 280) != 0
            for wheel in range(4)
        ),
    )
    side_contact = collect(
        records, lambda record: i32(record, SCENE_OFFSET + 52) != 0
    )
    reverse = collect(
        records, lambda record: i32(record, SCENE_OFFSET + 44) != 0
    )
    airborne = collect(
        records,
        lambda record: all(
            i32(record, WHEELS_OFFSET + wheel * WHEEL_SIZE + 272) == 0
            for wheel in range(4)
        ),
    )

    speeds = [
        math.sqrt(
            f32(record, 68) ** 2 + f32(record, 72) ** 2 + f32(record, 76) ** 2
        )
        for record in records
    ]
    max_speed_tick = max(range(len(records)), key=speeds.__getitem__)
    gears = [i32(record, SCENE_OFFSET + 48) for record in records]
    side_steers = [
        f32(record, SCENE_OFFSET + 8)
        for record, active in zip(records, side_contact, strict=True)
        if active
    ]
    side_steer_changes = sum(
        left != right for left, right in zip(side_steers, side_steers[1:])
    )

    landings = []
    air_start: int | None = None
    for tick, active in enumerate((*airborne, False)):
        if active and air_start is None:
            air_start = tick
        elif not active and air_start is not None:
            if tick < len(records):
                record = records[tick]
                contact_mask = sum(
                    (i32(record, WHEELS_OFFSET + wheel * WHEEL_SIZE + 272) != 0)
                    << wheel
                    for wheel in range(4)
                )
                front_tick = next(
                    (
                        probe
                        for probe in range(tick, min(tick + 20, len(records)))
                        if any(
                            i32(
                                records[probe],
                                WHEELS_OFFSET + wheel * WHEEL_SIZE + 272,
                            )
                            != 0
                            for wheel in (0, 1)
                        )
                    ),
                    None,
                )
                rear_tick = next(
                    (
                        probe
                        for probe in range(tick, min(tick + 20, len(records)))
                        if any(
                            i32(
                                records[probe],
                                WHEELS_OFFSET + wheel * WHEEL_SIZE + 272,
                            )
                            != 0
                            for wheel in (2, 3)
                        )
                    ),
                    None,
                )
                landings.append(
                    (
                        air_start,
                        tick,
                        f32(record, 72),
                        contact_mask,
                        f32(record, 40),
                        f32(record, 36),
                        front_tick,
                        rear_tick,
                    )
                )
            air_start = None

    print(f"capture: {args.capture}")
    print(f"ticks: {len(records)}")
    print(f"duration: {len(records) * 10} ms")
    print(f"global sliding ticks: {sum(global_sliding)}")
    print(f"global sliding longest: {format_run(longest_run(global_sliding))}")
    print(f"wheel sliding ticks: {sum(wheel_sliding)}")
    print(f"wheel sliding longest: {format_run(longest_run(wheel_sliding))}")
    print(f"side contact ticks: {sum(side_contact)}")
    print(f"side contact longest: {format_run(longest_run(side_contact))}")
    print(f"side-contact steering values: {sorted(set(side_steers))}")
    print(f"side-contact steering changes: {side_steer_changes}")
    print(f"gas+brake ticks: {sum(gas_brake)}")
    print(f"gas+brake longest: {format_run(longest_run(gas_brake))}")
    print(f"reverse-latch ticks: {sum(reverse)}")
    print(f"reverse-latch longest: {format_run(longest_run(reverse))}")
    print(f"gear range: {min(gears)}..{max(gears)}")
    print(f"airborne ticks: {sum(airborne)}")
    print(f"airborne longest: {format_run(longest_run(airborne))}")
    print(f"landing count: {len(landings)}")
    for index, (
        start,
        end,
        vertical_speed,
        contact_mask,
        forward_y,
        up_y,
        front_tick,
        rear_tick,
    ) in enumerate(landings, 1):
        axle_spread = (
            "unresolved"
            if front_tick is None or rear_tick is None
            else f"{abs(front_tick - rear_tick)} ticks"
        )
        print(
            f"landing {index}: air={start + 1}-{end} "
            f"({end - start} ticks), vertical_speed={vertical_speed:.9g}, "
            f"first_contact_mask=0x{contact_mask:x}, "
            f"forward_y={forward_y:.9g}, up_y={up_y:.9g}, "
            f"axle_spread={axle_spread}"
        )
    print(
        f"max linear speed: {speeds[max_speed_tick]:.9g} "
        f"at tick {max_speed_tick + 1}"
    )
    final = records[-1]
    print(
        "final position: "
        f"({f32(final, 56):.9g}, {f32(final, 60):.9g}, {f32(final, 64):.9g})"
    )
    if args.controls is not None:
        first, last = args.controls
        if first < 1 or last < first or last > len(records):
            raise ValueError("control range lies outside the capture")
        run_start = first
        current = None
        for tick in range(first, last + 2):
            controls = (
                None
                if tick == last + 1
                else (
                    int(f32(records[tick - 1], SCENE_OFFSET)),
                    int(f32(records[tick - 1], SCENE_OFFSET + 4)),
                    int(f32(records[tick - 1], SCENE_OFFSET + 8)),
                )
            )
            if current is None:
                current = controls
            elif controls != current:
                print(
                    f"controls {run_start}-{tick - 1}: "
                    f"gas={current[0]} brake={current[1]} steer={current[2]}"
                )
                run_start = tick
                current = controls
    if args.wheels is not None:
        first, last = args.wheels
        if first < 1 or last < first or last > len(records):
            raise ValueError("wheel range lies outside the capture")
        for tick in range(first, last + 1):
            record = records[tick - 1]
            contacts = [
                i32(record, WHEELS_OFFSET + wheel * WHEEL_SIZE + 272)
                for wheel in range(4)
            ]
            materials = [
                i32(record, WHEELS_OFFSET + wheel * WHEEL_SIZE + 276)
                & 0xFFFF
                for wheel in range(4)
            ]
            sliding = [
                i32(record, WHEELS_OFFSET + wheel * WHEEL_SIZE + 280)
                for wheel in range(4)
            ]
            print(
                f"wheels {tick}: contact={contacts} "
                f"material={materials} sliding={sliding}"
            )


if __name__ == "__main__":
    main()
