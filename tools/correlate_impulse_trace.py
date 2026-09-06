#!/usr/bin/env python3
"""Correlate a WheelAbsorbContact record with its nested impulse call."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct


def read_trace(path: Path) -> list[tuple[int, dict[int, bytes], dict[int, bytes]]]:
    data = memoryview(path.read_bytes())
    if data[:8] != b"TMNFTRC1":
        raise ValueError(f"{path} has bad trace magic")
    count = struct.unpack_from("<I", data, 12)[0]
    offset = 16
    records = []
    for _ in range(count):
        sequence, input_count, output_count = struct.unpack_from(
            "<IHH", data, offset
        )
        offset += 8
        groups = []
        for buffer_count in (input_count, output_count):
            buffers = {}
            for _ in range(buffer_count):
                tag, _, size = struct.unpack_from("<III", data, offset)
                offset += 12
                buffers[tag] = bytes(data[offset : offset + size])
                offset += size
            groups.append(buffers)
        records.append((sequence, groups[0], groups[1]))
    if offset != len(data):
        raise ValueError(f"{path} has trailing trace bytes")
    return records


def f32x3(data: bytes, offset: int = 0) -> tuple[float, float, float]:
    return struct.unpack_from("<fff", data, offset)


def bits3(data: bytes, offset: int = 0) -> tuple[str, str, str]:
    return tuple(
        f"{value:08x}" for value in struct.unpack_from("<III", data, offset)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel_trace", type=Path)
    parser.add_argument("impulse_trace", type=Path)
    parser.add_argument("wheel_sequence", type=int)
    args = parser.parse_args()

    wheel_records = read_trace(args.wheel_trace)
    impulse_records = read_trace(args.impulse_trace)
    wheel = next(
        record for record in wheel_records if record[0] == args.wheel_sequence
    )
    wheel_state = wheel[1][16]
    matches = [
        record
        for record in impulse_records
        if record[1][3][0x34:0x4c] == wheel_state[0x34:0x4c]
    ]
    if len(matches) != 1:
        raise RuntimeError(f"found {len(matches)} nested impulse calls")
    impulse = matches[0]
    contact = wheel[1][2]
    impulse_state = impulse[1][3]
    mismatch = next(
        (
            offset
            for offset, values in enumerate(zip(wheel_state, impulse_state))
            if values[0] != values[1]
        ),
        None,
    )
    print(f"wheel sequence: {wheel[0]}")
    print(f"impulse sequence: {impulse[0]}")
    print(f"first dyna-input mismatch: {mismatch}")
    print(f"contact normal: {f32x3(contact, 0x0c)}")
    print(f"contact position: {f32x3(contact, 0x18)}")
    print(f"contact relative speed: {f32x3(contact, 0x24)}")
    print(f"contact replacement: {f32x3(contact, 0x30)}")
    print(f"impulse input: {f32x3(impulse[1][1])}")
    print(f"impulse input bits: {bits3(impulse[1][1])}")
    print(f"impulse point: {f32x3(impulse[1][2])}")
    print(f"dyna linear input bits: {bits3(wheel_state, 0x40)}")
    print(f"dyna linear output bits: {bits3(wheel[2][35], 0x40)}")


if __name__ == "__main__":
    main()
