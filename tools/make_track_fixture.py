#!/usr/bin/env python3
"""Create the smallest valid TMNFTRK1 snapshot for loader smoke tests."""

from __future__ import annotations

import hashlib
import pathlib
import struct
import sys


HEADER_SIZE = 0x150
EXE_SHA = bytes.fromhex(
    "3847cf9f20bfc63914450060ed528c121"
    "04f743d96ad23d6e76abd178de8c84f")
TRACK_SHA = bytes(range(32))


def align8(value: int) -> int:
    return (value + 7) & ~7


def main() -> None:
    output = pathlib.Path(sys.argv[1])

    vertices = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 0, 1)
    face = struct.pack("<3fI3IHH", 0, 1, 0, 0, 0, 1, 2, 0, 0)
    node = struct.pack("<I6fI", 1, 0.5, 0, 0.5, 0.5, 0.1, 0.5, 0)
    material_ids = b"\0"
    material_data = b"".join(
        struct.pack("<2f", 1.0, 0.5) for _ in range(31))
    pair = struct.pack("<5I", 0, 1, 0, 0, 0)

    sections_data = [
        None,  # entry, after surface offset is known
        None,  # surface, after mesh/material offsets are known
        None,  # mesh, after array offsets are known
        vertices,
        face,
        node,
        material_ids,
        material_data,
        pair,
        None,  # static corpus location of corpus id 1, after the iso is built
        # Stadium water owner: 32 m cells, 32x32, level 8.0, floor -992.0.
        struct.pack("<4fIIIffI", 32.0, 32.0, 0.0, 0.0, 32, 32, 0, 8.0, -992.0, 0),
        bytes(32 * 32),
    ]
    strides = [0x60, 0x18, 0x38, 0x0C, 0x20, 0x20, 1, 8, 0x14, 0x30, 0x28, 1]
    counts = [1, 1, 1, 3, 1, 1, 1, 31, 1, 1, 1, 32 * 32]

    # Reserve fixed-size sections first; their content references later arrays.
    sizes = ([0x60, 0x18, 0x38] + [len(x) for x in sections_data[3:9]] + [0x30]
             + [len(x) for x in sections_data[10:]])
    offsets: list[int] = []
    cursor = HEADER_SIZE
    for size in sizes:
        cursor = align8(cursor)
        offsets.append(cursor)
        cursor += size

    identity_iso = (
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
    )
    box = (0.5, 0.0, 0.5, 0.5, 0.1, 0.5)
    sections_data[9] = struct.pack("<12f", *identity_iso)
    sections_data[0] = struct.pack(
        "<I6f12fIQII",
        1, *box, *identity_iso, 0x80, offsets[1], 1, 1)
    sections_data[1] = struct.pack(
        "<QQII", offsets[2], offsets[6], 1, 0)
    sections_data[2] = struct.pack(
        "<IHBBIIQIIQIIQ",
        0, 0, 7, 0,
        3, 0, offsets[3],
        1, 0, offsets[4],
        1, 0, offsets[5])

    image = bytearray(cursor)
    for offset, data in zip(offsets, sections_data, strict=True):
        image[offset:offset + len(data)] = data

    struct.pack_into(
        "<8sIIIIQ", image, 0,
        b"TMNFTRK1", 3, 0x12345678, HEADER_SIZE, 12, len(image))
    image[32:64] = EXE_SHA
    image[64:96] = TRACK_SHA
    for i, (offset, count, stride) in enumerate(
            zip(offsets, counts, strides, strict=True)):
        struct.pack_into("<QII", image, 96 + i * 16, offset, count, stride)
    image[288:320] = hashlib.sha256(image[HEADER_SIZE:]).digest()
    output.write_bytes(image)
    print(f"{output}: {len(image)} bytes, track_sha256={TRACK_SHA.hex()}")


if __name__ == "__main__":
    main()
