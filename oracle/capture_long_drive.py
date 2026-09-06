#!/usr/bin/env python3
"""Capture a deterministic 30-second A01 drive without an early crash."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from determinism_test import (
    DEFAULT_MAP,
    FIELDS,
    TICK_MS,
    encode_state,
    map_command,
    wait_for_race_start,
)
from tmi_client import MessageType, TMInterface


TICK_COUNT = 3000
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "a01_long_drive.bin"


def scheduled_input(tick: int) -> dict[str, bool]:
    return {
        "accelerate": tick < 250,
        "brake": 250 <= tick < 300,
        "left": False,
        "right": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--map", default=DEFAULT_MAP)
    args = parser.parse_args()

    interface = TMInterface(args.port, timeout=60)
    connect_message = interface.read_message_type()
    if connect_message != MessageType.SC_ON_CONNECT_SYNC:
        raise RuntimeError(f"expected connect sync, got {connect_message.name}")

    interface.set_timeout(300_000)
    interface.set_on_step_period(TICK_MS)
    interface.set_speed(1.0)
    interface.execute_command("set countdown_speed 5")
    interface.execute_command("set autorewind false")
    interface.execute_command("set use_valseed false")
    interface.execute_command(map_command(args.map))
    interface.respond(connect_message)

    blocked_message, _ = wait_for_race_start(interface)
    output = bytearray()
    field_sizes: tuple[int, ...] | None = None
    final_state = None

    for tick in range(TICK_COUNT):
        interface.set_input_state(**scheduled_input(tick))
        interface.respond(blocked_message)
        while True:
            blocked_message = interface.read_message_type()
            if blocked_message == MessageType.SC_RUN_STEP_SYNC:
                race_time = interface.read_run_step()
                break
            if blocked_message in (
                MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC,
                MessageType.SC_LAP_COUNT_CHANGED_SYNC,
            ):
                interface.read_count_change()
                interface.respond(blocked_message)
                continue
            raise RuntimeError(f"unexpected TMInterface message {blocked_message.name}")

        expected_time = (tick + 1) * TICK_MS
        if race_time != expected_time:
            raise RuntimeError(f"expected race time {expected_time}, got {race_time}")
        final_state = interface.get_simulation_state()
        record, current_sizes = encode_state(final_state)
        if field_sizes is None:
            field_sizes = current_sizes
        elif current_sizes != field_sizes:
            raise RuntimeError(
                f"state field sizes changed from {field_sizes} to {current_sizes}"
            )
        if len(record) != 1668:
            raise RuntimeError(f"expected 1668-byte record, got {len(record)}")
        output.extend(record)

    interface.set_input_state(left=False, right=False, accelerate=False, brake=False)
    interface.respond(blocked_message)
    interface.close()

    if field_sizes is None or final_state is None:
        raise RuntimeError("captured no states")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    print(f"ticks: {TICK_COUNT}")
    print(f"duration: {TICK_COUNT * TICK_MS} ms")
    print(f"bytes per tick: {sum(field_sizes)}")
    print(f"bytes: {len(output)}")
    print(f"final position: {tuple(final_state.dyna.current_state.position)}")
    print(f"final linear speed: {tuple(final_state.dyna.current_state.linear_speed)}")
    print("fields: " + ", ".join(field.name for field in FIELDS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
