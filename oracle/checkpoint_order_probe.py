#!/usr/bin/env python3
"""Probe the game's checkpoint rule on A01-Race by visiting triggers out of order.

The car is teleported (TMInterface rewind with the rigid-body state replaced by
a record of the A01 world-record capture a few hundred ms before a trigger) and
then driven at full throttle through the trigger while the game's
SC_CHECKPOINT_COUNT_CHANGED_SYNC events are recorded. Phases:

  finish_first   finish with 0/2 checkpoints    -> no event expected
  cp1_first      the ghost's second checkpoint  -> 1/3 expected
  cp1_again      the same checkpoint again      -> no event expected
  cp0_second     the ghost's first checkpoint   -> 2/3 expected
  finish_last    the finish                     -> 3/3 and RaceFinished

Run under oracle/launch_game.sh via tools/wr_replay.py's game harness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys

import numpy as np

from determinism_test import TICK_MS, map_command, wait_for_race_start
from tmi_client import MessageType, TMInterface

RECORD_SIZE = 1668
DYNA_FIELDS = (
    ("quat", 4),
    ("rotation", 9),
    ("position", 3),
    ("linear_speed", 3),
    ("add_linear_speed", 3),
    ("angular_speed", 3),
    ("force", 3),
    ("torque", 3),
    ("inverse_inertia_tensor", 9),
    ("not_tweaked_linear_speed", 3),
)

PHASES = (
    # name, WR capture tick to teleport to, expected (current, target) or None
    ("finish_first", 2320, None),
    ("cp1_first", 1920, (1, 3)),
    ("cp1_again", 1920, None),
    ("cp0_second", 1520, (2, 3)),
    ("finish_last", 2320, (3, 3)),
)
DRIVE_TICKS = 150


def dyna_record(reference: bytes, tick: int) -> dict[str, np.ndarray]:
    record = reference[tick * RECORD_SIZE:(tick + 1) * RECORD_SIZE]
    if len(record) != RECORD_SIZE:
        raise ValueError(f"reference has no tick {tick}")
    offset = 4
    fields = {}
    for name, count in DYNA_FIELDS:
        values = np.frombuffer(record, dtype="<f4", count=count, offset=offset)
        offset += 4 * count
        fields[name] = values.reshape(3, 3) if count == 9 else values
    return fields


def teleport(interface: TMInterface, fields: dict[str, np.ndarray]) -> None:
    state = interface.get_simulation_state()
    for dyna_state in (
        state.dyna.previous_state,
        state.dyna.current_state,
        state.dyna.temp_state,
    ):
        for name, _ in DYNA_FIELDS:
            setattr(dyna_state, name, fields[name])
    interface.rewind_to_state(state)


def drive(
    interface: TMInterface, blocked: MessageType, ticks: int
) -> tuple[MessageType, list[dict[str, object]]]:
    events: list[dict[str, object]] = []
    for _ in range(ticks):
        interface.set_input_state(left=False, right=False, accelerate=True, brake=False)
        interface.respond(blocked)
        while True:
            blocked = interface.read_message_type()
            if blocked == MessageType.SC_RUN_STEP_SYNC:
                interface.read_run_step()
                break
            if blocked in (
                MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC,
                MessageType.SC_LAP_COUNT_CHANGED_SYNC,
            ):
                current, target = interface.read_count_change()
                state = interface.get_simulation_state()
                events.append({
                    "type": blocked.name,
                    "current": current,
                    "target": target,
                    "race_time": int(state.race_time),
                    "position": [float(v) for v in state.dyna.current_state.position],
                    "race_finished": bool(interface.race_finished()),
                })
                if current == target:
                    interface.prevent_simulation_finish()
                interface.respond(blocked)
                continue
            raise RuntimeError(f"unexpected TMInterface message {blocked.name}")
    return blocked, events


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--map", default="A01-Race")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = args.reference.read_bytes()

    interface = TMInterface(args.port, timeout=60)
    connect = interface.read_message_type()
    if connect != MessageType.SC_ON_CONNECT_SYNC:
        raise RuntimeError(f"expected connect sync, got {connect.name}")
    interface.set_timeout(300_000)
    interface.set_on_step_period(TICK_MS)
    interface.set_speed(1.0)
    interface.execute_command("set countdown_speed 5")
    interface.execute_command("set autorewind false")
    interface.execute_command("set use_valseed false")
    interface.execute_command(map_command(args.map))
    interface.respond(connect)

    blocked, _ = wait_for_race_start(interface)
    results = []
    ok = True
    try:
        for name, tick, expected in PHASES:
            teleport(interface, dyna_record(reference, tick))
            blocked, events = drive(interface, blocked, DRIVE_TICKS)
            checkpoint_events = [
                e for e in events if e["type"] == "SC_CHECKPOINT_COUNT_CHANGED_SYNC"
            ]
            observed = (
                (checkpoint_events[0]["current"], checkpoint_events[0]["target"])
                if checkpoint_events else None
            )
            passed = observed == expected and len(checkpoint_events) <= 1
            if expected == (3, 3):
                passed = passed and checkpoint_events[0]["race_finished"]
            ok = ok and passed
            result = {
                "phase": name,
                "teleport_tick": tick,
                "expected": list(expected) if expected else None,
                "events": events,
                "passed": passed,
            }
            results.append(result)
            print(f"phase: {json.dumps(result)}", flush=True)
        interface.set_input_state(left=False, right=False, accelerate=False, brake=False)
        interface.respond(blocked)
    finally:
        interface.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"map": args.map, "reference": args.reference.name, "phases": results, "passed": ok},
        indent=1,
    ) + "\n")
    print(f"checkpoint order probe: {'passed' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
