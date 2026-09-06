#!/usr/bin/env python3
"""Copy never-written game memory from the native capture before a byte compare.

Every word touched here is justified in analysis/capture_normalization.md.
The rule is the same for all of them: a word is normalized only while no game
code can have written it since CSceneVehicleCar::VehicleReset, so its content
is heap leftover of the capturing process and not a physical quantity. Once
the game's only writer has run, the word is compared byte for byte.

Car words (CSceneVehicleCar offsets; record offsets 248, 252, 260):
  +0x5f8 turbo start tick, +0x5fc turbo end tick: written only by
  CSceneVehicleCar::EnableTurbo (0x007BCF90), which also sets +0x600 turbo
  type to a nonzero value in the same call. Normalized only on ticks before
  the first tick with a nonzero turbo type, and only while both captures still
  hold their tick-0 value.
  +0x608 roulette value: written only by EnableTurbo for turbo type 2.
  Normalized only before the first tick with turbo type 2, same value rule.
  +0x5f4 turbo factor and +0x60c freewheel flag are zeroed by VehicleReset
  (0x007C03D7, 0x007C05F5) and are never normalized.

Wheel words (SSimulationWheel offsets; record offsets 356 + wheel * 328 + 276
and + 284):
  +0x128 contact material: every game store is a 16-bit store (WheelReset
  0x007BD38A, ComputeForces 0x007C7903, WheelAbsorbContact 0x007C1295) and
  every load is a 16-bit load. The upper half is never written by anything and
  is always replaced by the native upper half; the low 16 bits are compared.
  +0x130..+0x13b surface position in item space: written by WheelAbsorbContact
  (0x007C12D0..0x007C12E1) only for contacts that carry a surface; not zeroed
  by WheelReset. Normalized as one 12-byte vector only while both captures
  still hold their tick-0 value.
"""

from __future__ import annotations

import argparse
from pathlib import Path


RECORD_SIZE = 1668
CAR = 176
TURBO_START = CAR + 72      # car +0x5f8
TURBO_END = CAR + 76        # car +0x5fc
TURBO_TYPE = CAR + 80       # car +0x600
ROULETTE_VALUE = CAR + 84   # car +0x608
ROULETTE_TURBO_TYPE = 2


def wheel_offset(wheel: int, offset: int) -> int:
    return 356 + wheel * 328 + offset


WHEEL_MATERIAL = tuple(wheel_offset(wheel, 276) for wheel in range(4))       # +0x128
WHEEL_SURFACE_POSITION = tuple(wheel_offset(wheel, 284) for wheel in range(4))  # +0x130
SURFACE_POSITION_SIZE = 12


def word(data: bytes | bytearray, record: int, offset: int, size: int = 4) -> bytes:
    start = record * RECORD_SIZE + offset
    return bytes(data[start:start + size])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    original = args.game.read_bytes()
    game = bytearray(original)
    native = args.native.read_bytes()
    if len(game) == 0 or len(game) % RECORD_SIZE != 0:
        raise ValueError("game capture is not a whole number of records")
    if len(native) != len(game):
        raise ValueError("native capture size differs from game capture")
    ticks = len(game) // RECORD_SIZE

    counts = {
        "turbo_start": 0, "turbo_end": 0, "roulette_value": 0,
        "material_high16": 0, "surface_position": 0,
    }
    pristine = {
        TURBO_START: True, TURBO_END: True, ROULETTE_VALUE: True,
        **{offset: True for offset in WHEEL_SURFACE_POSITION},
    }
    turbo_seen = False
    roulette_seen = False

    def still_pristine(record: int, offset: int, size: int) -> bool:
        return (
            pristine[offset]
            and word(original, record, offset, size) == word(original, 0, offset, size)
            and word(native, record, offset, size) == word(native, 0, offset, size)
        )

    for record in range(ticks):
        base = record * RECORD_SIZE
        turbo_type = int.from_bytes(word(game, record, TURBO_TYPE), "little")
        turbo_seen = turbo_seen or turbo_type != 0
        roulette_seen = roulette_seen or turbo_type == ROULETTE_TURBO_TYPE

        for offset, name, blocked in (
            (TURBO_START, "turbo_start", turbo_seen),
            (TURBO_END, "turbo_end", turbo_seen),
            (ROULETTE_VALUE, "roulette_value", roulette_seen),
        ):
            pristine[offset] = still_pristine(record, offset, 4) and not blocked
            if pristine[offset]:
                target = slice(base + offset, base + offset + 4)
                if game[target] != native[target]:
                    game[target] = native[target]
                    counts[name] += 1

        for offset in WHEEL_MATERIAL:
            target = slice(base + offset, base + offset + 4)
            game_value = int.from_bytes(game[target], "little")
            native_value = int.from_bytes(native[target], "little")
            normalized = (game_value & 0xFFFF) | (native_value & 0xFFFF0000)
            if normalized != game_value:
                game[target] = normalized.to_bytes(4, "little")
                counts["material_high16"] += 1

        for offset in WHEEL_SURFACE_POSITION:
            pristine[offset] = still_pristine(record, offset, SURFACE_POSITION_SIZE)
            if pristine[offset]:
                target = slice(base + offset, base + offset + SURFACE_POSITION_SIZE)
                if game[target] != native[target]:
                    game[target] = native[target]
                    counts["surface_position"] += SURFACE_POSITION_SIZE // 4

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(game)
    print(f"ticks: {ticks}")
    print(f"normalized words: {sum(counts.values())}")
    for name, count in counts.items():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
