#!/usr/bin/env python3
"""Play the exported PPO input schedule into TMNF and capture every tick.

Digital records (left/right/gas/brake with timestamps on words 0, 3, 9, 12)
are sent with TMInterface's set_input_state before each tick. Analog records
(steer_analog_time on word 6, `-steer_integer / 65536` on word 8) cannot be
sent that way: Python_Link.as only carries the four digital keys. They are
converted to a TMInterface input script (`<time> steer <integer>` for the
16.16 steer value, `<from>-<to> press up|down` spans for gas and brake) and
loaded with TMInterface's own `load`, the path that produced every analog
capture under oracle/results/wr/ and oracle/results/regimes/a01_respawn.bin.
Tick i of the schedule is script time i*10 ms (tools/wr_replay.py).

The schedule must be an analog prefix followed by a digital suffix (the
exporter's released padding), so the two sources never compete for a tick.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import struct
import sys

from capture_replay_run import script_directory
from determinism_test import (
    DEFAULT_MAP,
    FIELDS,
    TICK_MS,
    encode_state,
    map_command,
    wait_for_race_start,
)
from tmi_client import MessageType, TMInterface


INPUT_STRUCT = struct.Struct("<IIiIIiIIfIIiIIiIIf")
DEFAULT_INPUTS = Path(__file__).resolve().parent / "results" / "policy_lap_inputs.bin"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "policy_lap.bin"
POSITION_TICKS = frozenset((500, 1000, 1500, 2000, 2500))


def read_inputs(path: Path) -> tuple[list[dict[str, bool] | None], str]:
    """Return the per-tick digital controls (None on analog ticks) and the
    TMInterface script that carries the analog ticks ("" when none)."""
    data = path.read_bytes()
    if len(data) == 0 or len(data) % INPUT_STRUCT.size != 0:
        raise ValueError(
            f"{path} size {len(data)} is not a whole number of "
            f"{INPUT_STRUCT.size}-byte inputs"
        )
    result: list[dict[str, bool] | None] = []
    analog: list[tuple[int, int, int]] = []  # (steer_integer, gas, brake)
    for tick, values in enumerate(INPUT_STRUCT.iter_unpack(data)):
        timestamp = (tick + 1) * TICK_MS
        if any(values[index] != 0 for index in (1, 4, 7, 10, 13, 15, 16)):
            raise ValueError(f"input tick {tick} has nonzero reserved state")
        if values[17] != 0.0:
            raise ValueError(f"input tick {tick} has analog gas")
        if values[9] != timestamp or values[12] != timestamp:
            raise ValueError(f"input tick {tick} has a noncanonical timestamp")
        if values[11] not in (0, 1) or values[14] not in (0, 1):
            raise ValueError(f"input tick {tick} has non-binary gas or brake")
        if values[6] == timestamp:
            if values[0] != 0 or values[3] != 0 or values[2] != 0 or values[5] != 0:
                raise ValueError(f"input tick {tick} mixes analog and digital steer")
            if result and result[-1] is not None:
                raise ValueError(f"input tick {tick} is analog after a digital tick")
            scaled = -values[8] * 65536.0
            steer_integer = int(scaled)
            if float(steer_integer) != scaled or abs(steer_integer) > 65536:
                raise ValueError(
                    f"input tick {tick} steer {values[8]!r} is not a 16.16 value"
                )
            analog.append((steer_integer, values[11], values[14]))
            result.append(None)
            continue
        if values[0] != timestamp or values[3] != timestamp or values[6] != 0:
            raise ValueError(f"input tick {tick} has a noncanonical timestamp")
        if values[8] != 0.0:
            raise ValueError(f"input tick {tick} has analog state")
        result.append(
            {
                "left": bool(values[2]),
                "right": bool(values[5]),
                "accelerate": bool(values[11]),
                "brake": bool(values[14]),
            }
        )
    return result, build_script(analog)


def build_script(analog: list[tuple[int, int, int]]) -> str:
    lines: list[str] = []
    previous_steer: int | None = None
    spans = {"up": None, "down": None}
    for tick, (steer, gas, brake) in enumerate(analog):
        if steer != previous_steer:
            lines.append(f"{tick * TICK_MS} steer {steer}")
            previous_steer = steer
        for key, pressed in (("up", gas), ("down", brake)):
            if pressed and spans[key] is None:
                spans[key] = tick
            elif not pressed and spans[key] is not None:
                lines.append(f"{spans[key] * TICK_MS}-{tick * TICK_MS} press {key}")
                spans[key] = None
    for key in ("up", "down"):
        if spans[key] is not None:
            lines.append(f"{spans[key] * TICK_MS}-{len(analog) * TICK_MS} press {key}")
    return "".join(line + "\n" for line in lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--map", default=DEFAULT_MAP)
    args = parser.parse_args()
    schedule, script = read_inputs(args.inputs)
    position_ticks = frozenset(
        tick for tick in POSITION_TICKS if tick <= len(schedule)
    )
    script_name: str | None = None
    if script:
        script_name = f"policy_lap_{hashlib.sha256(script.encode()).hexdigest()[:12]}.txt"
        scripts_dir = script_directory()
        scripts_dir.mkdir(parents=True, exist_ok=True)
        (scripts_dir / script_name).write_text(script)
        script_copy = args.output.with_suffix(".script.txt")
        script_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(scripts_dir / script_name, script_copy)
        analog_ticks = sum(1 for inputs in schedule if inputs is None)
        print(
            f"analog ticks: {analog_ticks}, script {script_name}: "
            f"{len(script.splitlines())} lines, copy {script_copy}",
            flush=True,
        )

    interface = TMInterface(args.port, timeout=60)
    connected = False
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
        if script_name is not None:
            interface.execute_command(f"load {script_name}")
        interface.execute_command(map_command(args.map))
        interface.respond(connect_message)

        blocked_message, start_state = wait_for_race_start(interface)
        simulation_time_offset = int(start_state.time)
        output = bytearray()
        field_sizes: tuple[int, ...] | None = None
        positions: dict[int, tuple[float, float, float]] = {}
        finish_time_ms: int | None = None
        finish_checkpoint: tuple[int, int] | None = None

        for tick, inputs in enumerate(schedule):
            if inputs is not None:
                interface.set_input_state(**inputs)
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
                    print(
                        f"count event while advancing tick {tick}: "
                        f"{blocked_message.name} {current}/{target}",
                        flush=True,
                    )
                    if (
                        blocked_message
                        == MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC
                        and current == target
                    ):
                        if finish_time_ms is not None:
                            raise RuntimeError("received a second finish checkpoint")
                        finish_state = interface.get_simulation_state()
                        if not interface.race_finished():
                            raise RuntimeError(
                                "final checkpoint did not set RaceFinished"
                            )
                        finish_time_ms = int(finish_state.race_time)
                        finish_checkpoint = (current, target)
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
                raise RuntimeError(
                    f"state field sizes changed from {field_sizes} "
                    f"to {current_sizes}"
                )
            if len(record) != 1668:
                raise RuntimeError(
                    f"expected 1668-byte record, got {len(record)}"
                )
            output.extend(record)
            tick_number = tick + 1
            if tick_number in position_ticks:
                positions[tick_number] = tuple(
                    float(value) for value in state.dyna.current_state.position
                )

        interface.set_input_state(
            left=False, right=False, accelerate=False, brake=False
        )
        if script_name is not None:
            interface.execute_command("unload")
        interface.respond(blocked_message)
    finally:
        if connected:
            interface.close()

    if field_sizes is None:
        raise RuntimeError("captured no states")
    if positions.keys() != position_ticks:
        raise RuntimeError(f"missing position samples: {positions.keys()}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    print(f"ticks: {len(schedule)}")
    print(f"duration: {len(schedule) * TICK_MS} ms")
    print(f"bytes per tick: {sum(field_sizes)}")
    print(f"bytes: {len(output)}")
    if finish_time_ms is None or finish_checkpoint is None:
        print("game finish: no")
    else:
        print(
            f"game finish: yes, race time {finish_time_ms} ms, "
            f"checkpoint {finish_checkpoint[0]}/{finish_checkpoint[1]}"
        )
    for tick in sorted(positions):
        print(f"game position tick {tick}: {positions[tick]}")
    print("fields: " + ", ".join(field.name for field in FIELDS))
    print(f"capture sha256: {hashlib.sha256(output).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
