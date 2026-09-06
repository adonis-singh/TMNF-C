#!/usr/bin/env python3
"""Play a TMInterface input script in TMNF and capture every tick.

The script is loaded with TMInterface's own `load` command, so analog steering
events from pad replays are applied by TMInterface itself. The client never
calls set_input_state during the run. The capture uses the same 1,668-byte
record layout as capture_policy_lap.py. Checkpoint and lap events are logged
with their race time so the env pass-through can compare trigger timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys

from determinism_test import (
    FIELDS,
    TICK_MS,
    encode_state,
    map_command,
    wait_for_race_start,
)
from tmi_client import MessageType, TMInterface
from game_paths import script_directory


RECORD_SIZE = 1668


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--map", required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--ticks", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument(
        "--after-finish-ticks", type=int, default=20,
        help="ticks captured after the game finish before stopping",
    )
    args = parser.parse_args()
    if args.ticks <= 0:
        raise ValueError("tick count must be positive")
    if not args.script.is_file():
        raise FileNotFoundError(args.script)
    scripts_dir = script_directory()
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_name = args.script.name
    if any(character in script_name for character in ' "\\/'):
        raise ValueError("script name must be a plain file name")
    shutil.copyfile(args.script, scripts_dir / script_name)

    interface = TMInterface(args.port, timeout=60)
    connected = False
    events: list[dict[str, object]] = []
    finish_time_ms: int | None = None
    output = bytearray()
    field_sizes: tuple[int, ...] | None = None
    captured_ticks = 0
    try:
        connect_message = interface.read_message_type()
        if connect_message != MessageType.SC_ON_CONNECT_SYNC:
            raise RuntimeError(
                f"expected connect sync, got {connect_message.name}"
            )
        connected = True
        interface.set_timeout(300_000)
        interface.set_on_step_period(TICK_MS)
        interface.set_speed(1.0)
        interface.execute_command("set countdown_speed 5")
        interface.execute_command("set autorewind false")
        interface.execute_command("set use_valseed false")
        interface.execute_command(f"load {script_name}")
        interface.execute_command(map_command(args.map))
        interface.respond(connect_message)

        blocked_message, start_state = wait_for_race_start(interface)
        simulation_time_offset = int(start_state.time)
        finish_tick: int | None = None

        for tick in range(args.ticks):
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
                    current, target = interface.read_count_change()
                    event_state = interface.get_simulation_state()
                    event = {
                        "tick": tick,
                        "type": blocked_message.name,
                        "current": current,
                        "target": target,
                        "race_time": int(event_state.race_time),
                        "position": [
                            float(value)
                            for value in event_state.dyna.current_state.position
                        ],
                    }
                    events.append(event)
                    print(f"event: {json.dumps(event)}", flush=True)
                    if (
                        blocked_message
                        == MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC
                        and current == target
                    ):
                        if finish_time_ms is not None:
                            raise RuntimeError("received a second finish")
                        if not interface.race_finished():
                            raise RuntimeError(
                                "final checkpoint did not set RaceFinished"
                            )
                        finish_time_ms = int(event_state.race_time)
                        finish_tick = tick
                    if finish_time_ms is not None:
                        interface.prevent_simulation_finish()
                    interface.respond(blocked_message)
                    continue
                raise RuntimeError(
                    f"unexpected TMInterface message {blocked_message.name}"
                )

            expected_time = (tick + 1) * TICK_MS
            if finish_time_ms is None and race_time != expected_time:
                raise RuntimeError(
                    f"expected race time {expected_time}, got {race_time}"
                )
            state = interface.get_simulation_state()
            simulation_time = int(state.time) - simulation_time_offset
            if simulation_time != expected_time:
                raise RuntimeError(
                    f"expected simulation time {expected_time}, "
                    f"got {simulation_time}"
                )
            record, current_sizes = encode_state(state)
            record = struct.pack("<i", simulation_time) + record[4:]
            if field_sizes is None:
                field_sizes = current_sizes
            elif current_sizes != field_sizes:
                raise RuntimeError("state field sizes changed")
            if len(record) != RECORD_SIZE:
                raise RuntimeError(f"expected {RECORD_SIZE}-byte record")
            output.extend(record)
            captured_ticks = tick + 1
            if (
                finish_tick is not None
                and tick >= finish_tick + args.after_finish_ticks
            ):
                break

        interface.execute_command("unload")
        interface.respond(blocked_message)
    finally:
        if connected:
            interface.close()

    if field_sizes is None:
        raise RuntimeError("captured no states")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    summary = {
        "map": args.map,
        "script": script_name,
        "ticks": captured_ticks,
        "finish_time_ms": finish_time_ms,
        "events": events,
        "capture_sha256": hashlib.sha256(output).hexdigest(),
    }
    args.events.parent.mkdir(parents=True, exist_ok=True)
    args.events.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"ticks: {captured_ticks}")
    print(f"bytes: {len(output)}")
    if finish_time_ms is None:
        print("game finish: no")
    else:
        print(f"game finish: yes, race time {finish_time_ms} ms")
    print("fields: " + ", ".join(field.name for field in FIELDS))
    print(f"capture sha256: {summary['capture_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
