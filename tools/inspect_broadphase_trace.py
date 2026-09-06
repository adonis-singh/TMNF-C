#!/usr/bin/env python3
"""Print broadphase graph counts for captured trace records."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct


def read_buffer(data: memoryview, offset: int) -> tuple[int, bytes, int]:
    tag, _, size = struct.unpack_from("<III", data, offset)
    offset += 12
    return tag, bytes(data[offset : offset + size]), offset + size


def graph_counts(graph: bytes) -> tuple[int, int, int, int, tuple[float, float, float] | None]:
    if graph[:8] != b"TMNFBP01":
        raise ValueError("bad broadphase graph magic")
    kind = struct.unpack_from("<I", graph, 12)[0]
    state_count = struct.unpack_from("<I", graph, 60)[0]
    tree_count = struct.unpack_from("<I", graph, 92)[0]
    buffer_count = struct.unpack_from("<I", graph, 128)[0]
    collision_count = struct.unpack_from("<I", graph, 132)[0]
    states_offset = struct.unpack_from("<I", graph, 156)[0]
    position = (
        struct.unpack_from("<fff", graph, states_offset + 4 + 0x34)
        if state_count != 0
        else None
    )
    return kind, tree_count, buffer_count, collision_count, position


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    data = memoryview(args.trace.read_bytes())
    if data[:8] != b"TMNFTRC1":
        raise ValueError("bad trace magic")
    count = struct.unpack_from("<I", data, 12)[0]
    offset = 16
    for _ in range(count):
        sequence, input_count, output_count = struct.unpack_from(
            "<IHH", data, offset
        )
        offset += 8
        inputs = {}
        outputs = {}
        for buffers, buffer_count in (
            (inputs, input_count),
            (outputs, output_count),
        ):
            for _ in range(buffer_count):
                tag, payload, offset = read_buffer(data, offset)
                buffers[tag] = payload
        print(
            f"sequence={sequence} input={graph_counts(inputs[0])} "
            f"output={graph_counts(outputs[32])}"
        )


if __name__ == "__main__":
    main()
