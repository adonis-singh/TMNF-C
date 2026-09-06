#!/usr/bin/env python3
"""Compare requested policy controls with TMInterface's captured controls.

Each 72-byte TMNFRaceInputs record is resolved with the game's 0x004FE500
source-priority rule (tools/wr_replay.py resolve_record) into the gas, brake
and steer floats the game stores at scene offset 176, and compared bit for
bit with the capture. Digital records resolve to 0/1 and -1/0/1; analog
records resolve to `-steer_analog`, so a centred analog stick is -0.0 and a
16.16 steer integer v is `-(-v / 65536)`. Every mismatch is printed with the
requested and captured words, including the 16.16 integer view of the steer,
so a systematic analog quantization difference shows up as such.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wr_replay import bits, resolve_record  # noqa: E402


RECORD_SIZE = 1668
SCENE_OFFSET = 176
INPUT_STRUCT = struct.Struct("<IIiIIiIIfIIiIIiIIf")


def steer_integer(steer_word: int) -> float:
    return -struct.unpack("<f", struct.pack("<I", steer_word))[0] * 65536.0


def describe(words: tuple[int, int, int]) -> str:
    gas, brake, steer = words
    return (
        f"gas={gas:#010x} brake={brake:#010x} steer={steer:#010x} "
        f"(16.16 {steer_integer(steer):+.1f})"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        type=Path,
        default=root / "oracle" / "results" / "policy_lap_inputs.bin",
    )
    parser.add_argument(
        "--capture",
        type=Path,
        default=root / "oracle" / "results" / "policy_lap.bin",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    requested = args.inputs.read_bytes()
    capture = args.capture.read_bytes()
    if len(requested) % INPUT_STRUCT.size != 0:
        raise ValueError("input schedule has a partial TMNFRaceInputs record")
    tick_count = len(requested) // INPUT_STRUCT.size
    if len(capture) != tick_count * RECORD_SIZE:
        raise ValueError("capture tick count differs from input schedule")

    effective = bytearray(requested)
    mismatches = []
    analog_ticks = 0
    for tick, requested_values in enumerate(INPUT_STRUCT.iter_unpack(requested)):
        expected = tuple(bits(value) for value in resolve_record(requested_values))
        accepted = struct.unpack_from("<III", capture, tick * RECORD_SIZE + SCENE_OFFSET)
        gas, brake, steer = struct.unpack("<fff", struct.pack("<III", *accepted))
        if gas not in (0.0, 1.0) or brake not in (0.0, 1.0):
            raise ValueError(f"tick {tick} has a non-binary longitudinal input")
        if accepted != expected:
            mismatches.append((tick, expected, accepted))
        values = list(requested_values)
        values[11] = int(gas)
        values[14] = int(brake)
        if requested_values[6] != 0:
            analog_ticks += 1
            # The record carries -steer; -(-0.0) keeps the packet's +0.0.
            values[8] = -steer
        elif steer in (-1.0, 0.0, 1.0):
            values[2] = int(steer == -1.0)
            values[5] = int(steer == 1.0)
        else:
            # The game kept an analog steer on a tick the schedule requested
            # digitally (released keys after an analog run generate no digital
            # steer event). The effective record becomes an analog record.
            timestamp = (tick + 1) * 10
            values[0:9] = [0, 0, 0, 0, 0, 0, timestamp, 0, -steer]
        INPUT_STRUCT.pack_into(effective, tick * INPUT_STRUCT.size, *values)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(effective)
    print(f"ticks: {tick_count}")
    print(f"analog ticks: {analog_ticks}")
    print(f"input mismatches: {len(mismatches)}")
    for tick, expected, accepted in mismatches:
        print(f"tick {tick}: requested {describe(expected)}")
        print(f"{' ' * len(f'tick {tick}: ')}captured  {describe(accepted)}")
    print(f"effective schedule sha256: {hashlib.sha256(effective).hexdigest()}")


if __name__ == "__main__":
    main()
