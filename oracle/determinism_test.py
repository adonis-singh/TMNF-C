#!/usr/bin/env python3
"""Empirically test TMNF physics determinism through TMInterface 2."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import struct
import sys
from typing import Callable

import numpy as np
from tminterface.structs import SimStateData

from tmi_client import MessageType, TMInterface


TICK_MS = 10
DEFAULT_TICKS = 1000
DEFAULT_MAP = "A01-Race"


def map_command(track_name: str) -> str:
    if not track_name or any(character in track_name for character in '"\\/'):
        raise ValueError("track name must be a plain Official Maps stem")
    return f'map "Official Maps/{track_name}.Challenge.Gbx"'


@dataclass(frozen=True)
class StateField:
    name: str
    encode: Callable[[SimStateData], bytes]


def f32(value: object) -> bytes:
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.asarray(value, dtype="<f4").tobytes(order="C")


def i32(value: object) -> bytes:
    return struct.pack("<i", int(value))


def encode_scene_mobil(state: SimStateData) -> bytes:
    scene = state.scene_mobil
    engine = scene.engine
    return b"".join(
        (
            f32((scene.input_gas, scene.input_brake, scene.input_steer, scene.max_linear_speed)),
            i32(scene.gearbox_state),
            i32(scene.block_flags),
            f32(
                (
                    engine.max_rpm,
                    engine.braking_factor,
                    engine.clamped_rpm,
                    engine.actual_rpm,
                    engine.slide_factor,
                )
            ),
            i32(engine.rear_gear),
            i32(engine.gear),
            i32(scene.has_any_lateral_contact),
            i32(scene.last_has_any_lateral_contact_time),
            i32(scene.water_forces_applied),
            f32((scene.turning_rate, scene.turbo_boost_factor)),
            i32(scene.last_turbo_type_change_time),
            i32(scene.last_turbo_time),
            i32(scene.turbo_type),
            f32(scene.roulette_value),
            i32(scene.is_freewheeling),
            i32(scene.is_sliding),
            i32(scene.wheel_contact_absorb_counter),
            i32(scene.burnout_state),
            f32(scene.current_local_speed),
            f32(scene.total_central_force_added),
            i32(scene.is_rubber_ball),
            f32(scene.saved_state),
        )
    )


def encode_wheel(state: SimStateData, index: int) -> bytes:
    wheel = state.simulation_wheels[index]
    surface = wheel.surface_handler
    realtime = wheel.real_time_state
    return b"".join(
        (
            i32(wheel.steerable),
            i32(wheel.field_8),
            f32(surface.unknown),
            f32(surface.rotation),
            f32(surface.position),
            f32(wheel.field_112),
            i32(wheel.field_160),
            i32(wheel.field_164),
            f32(wheel.offset_from_vehicle),
            f32((realtime.damper_absorb, realtime.field_4, realtime.field_8)),
            f32(realtime.field_12),
            f32(realtime.field_48),
            f32(realtime.field_84),
            f32(realtime.field_108),
            i32(realtime.has_ground_contact),
            i32(realtime.contact_material_id),
            i32(realtime.is_sliding),
            f32(realtime.relative_rotz_axis),
            i32(realtime.nb_ground_contacts),
            f32(realtime.field_144),
            i32(wheel.field_348),
            f32(wheel.contact_relative_local_distance),
        )
    )


FIELDS = (
    StateField("race_time", lambda state: i32(state.race_time)),
    StateField("dyna.quat", lambda state: f32(state.dyna.current_state.quat)),
    StateField("dyna.rotation", lambda state: f32(state.dyna.current_state.rotation)),
    StateField("dyna.position", lambda state: f32(state.dyna.current_state.position)),
    StateField("dyna.linear_speed", lambda state: f32(state.dyna.current_state.linear_speed)),
    StateField("dyna.add_linear_speed", lambda state: f32(state.dyna.current_state.add_linear_speed)),
    StateField("dyna.angular_speed", lambda state: f32(state.dyna.current_state.angular_speed)),
    StateField("dyna.force", lambda state: f32(state.dyna.current_state.force)),
    StateField("dyna.torque", lambda state: f32(state.dyna.current_state.torque)),
    StateField(
        "dyna.inverse_inertia_tensor",
        lambda state: f32(state.dyna.current_state.inverse_inertia_tensor),
    ),
    StateField(
        "dyna.not_tweaked_linear_speed",
        lambda state: f32(state.dyna.current_state.not_tweaked_linear_speed),
    ),
    StateField("scene_mobil.physics", encode_scene_mobil),
    StateField("wheel[0].physics", lambda state: encode_wheel(state, 0)),
    StateField("wheel[1].physics", lambda state: encode_wheel(state, 1)),
    StateField("wheel[2].physics", lambda state: encode_wheel(state, 2)),
    StateField("wheel[3].physics", lambda state: encode_wheel(state, 3)),
)


def scheduled_input(tick: int) -> dict[str, bool]:
    return {
        "accelerate": tick < 800 or tick >= 900,
        "brake": 600 <= tick < 650 or 800 <= tick < 900,
        "left": 100 <= tick < 300,
        "right": 300 <= tick < 500,
    }


def encode_state(state: SimStateData) -> tuple[bytes, tuple[int, ...]]:
    chunks = tuple(field.encode(state) for field in FIELDS)
    return b"".join(chunks), tuple(len(chunk) for chunk in chunks)


def read_next_run_step(interface: TMInterface) -> tuple[MessageType, int]:
    while True:
        message_type = interface.read_message_type()
        if message_type == MessageType.SC_RUN_STEP_SYNC:
            return message_type, interface.read_run_step()
        if message_type in (
            MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC,
            MessageType.SC_LAP_COUNT_CHANGED_SYNC,
        ):
            interface.read_count_change()
            interface.respond(message_type)
            continue
        raise RuntimeError(f"unexpected TMInterface message {message_type.name}")


def wait_for_race_start(interface: TMInterface) -> tuple[MessageType, SimStateData]:
    saw_countdown = False
    while True:
        message_type, race_time = read_next_run_step(interface)
        saw_countdown |= race_time < 0
        if saw_countdown and race_time == 0:
            return message_type, interface.get_simulation_state()
        interface.respond(message_type)


def capture_run(
    interface: TMInterface,
    blocked_message: MessageType,
    ticks: int,
    field_sizes: tuple[int, ...],
) -> tuple[bytes, MessageType]:
    output = bytearray()
    for tick in range(ticks):
        interface.set_input_state(**scheduled_input(tick))
        interface.respond(blocked_message)
        blocked_message, race_time = read_next_run_step(interface)
        expected_time = (tick + 1) * TICK_MS
        if race_time != expected_time:
            raise RuntimeError(f"expected race time {expected_time}, got {race_time}")
        record, current_sizes = encode_state(interface.get_simulation_state())
        if current_sizes != field_sizes:
            raise RuntimeError(f"state field sizes changed from {field_sizes} to {current_sizes}")
        output.extend(record)
    return bytes(output), blocked_message


def first_divergence(
    expected: bytes,
    actual: bytes,
    ticks: int,
    field_sizes: tuple[int, ...],
) -> str | None:
    if expected == actual:
        return None
    record_size = sum(field_sizes)
    expected_size = ticks * record_size
    if len(expected) != expected_size or len(actual) != expected_size:
        return f"log size differs: expected {len(expected)} bytes, got {len(actual)} bytes"

    byte_index = next(index for index, pair in enumerate(zip(expected, actual)) if pair[0] != pair[1])
    tick = byte_index // record_size
    field_offset = byte_index % record_size
    offset = 0
    for field, size in zip(FIELDS, field_sizes):
        if field_offset < offset + size:
            local_offset = field_offset - offset
            start = tick * record_size + offset
            context_start = max(0, local_offset - 8)
            context_end = min(size, local_offset + 9)
            return (
                f"tick {tick + 1} (race time {(tick + 1) * TICK_MS} ms), "
                f"field {field.name}, byte {local_offset}: "
                f"{expected[start + context_start:start + context_end].hex()} != "
                f"{actual[start + context_start:start + context_end].hex()}"
            )
        offset += size
    raise AssertionError("field offset outside encoded record")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument("--map", default=DEFAULT_MAP)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--save-cross-process-reference", type=Path)
    parser.add_argument("--cross-process-reference", type=Path)
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

    blocked_message, reference_state = wait_for_race_start(interface)
    reference_record, field_sizes = encode_state(reference_state)

    run1, blocked_message = capture_run(interface, blocked_message, args.ticks, field_sizes)

    interface.rewind_to_state(reference_state)
    rewound_record, rewound_sizes = encode_state(interface.get_simulation_state())
    if rewound_sizes != field_sizes:
        raise RuntimeError(f"rewound state field sizes changed from {field_sizes} to {rewound_sizes}")
    rewind_divergence = first_divergence(reference_record, rewound_record, 1, field_sizes)

    run2, blocked_message = capture_run(interface, blocked_message, args.ticks, field_sizes)
    interface.set_input_state(left=False, right=False, accelerate=False, brake=False)
    interface.respond(blocked_message)
    interface.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run1.bin").write_bytes(run1)
    (args.output_dir / "run2.bin").write_bytes(run2)

    divergence = first_divergence(run1, run2, args.ticks, field_sizes)
    print(f"ticks: {args.ticks}")
    print(f"bytes per tick: {sum(field_sizes)}")
    print("fields: " + ", ".join(field.name for field in FIELDS))
    print(f"immediate rewind byte-identical: {'yes' if rewind_divergence is None else 'no'}")
    if rewind_divergence is not None:
        print(f"immediate rewind first divergence: {rewind_divergence}")
    print(f"same-process byte-identical: {'yes' if divergence is None else 'no'}")
    if divergence is not None:
        print(f"first divergence: {divergence}")

    if args.save_cross_process_reference is not None:
        args.save_cross_process_reference.parent.mkdir(parents=True, exist_ok=True)
        args.save_cross_process_reference.write_bytes(run1)
        print(f"cross-process reference: {args.save_cross_process_reference}")

    cross_process_divergence = None
    if args.cross_process_reference is not None:
        cross_process_divergence = first_divergence(
            args.cross_process_reference.read_bytes(),
            run1,
            args.ticks,
            field_sizes,
        )
        print(f"cross-process byte-identical: {'yes' if cross_process_divergence is None else 'no'}")
        if cross_process_divergence is not None:
            print(f"cross-process first divergence: {cross_process_divergence}")

    return int(divergence is not None or cross_process_divergence is not None)


if __name__ == "__main__":
    sys.exit(main())
