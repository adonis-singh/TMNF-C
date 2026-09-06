#!/usr/bin/env python3
"""Extract the first fail-fast Model6 input graph as a vehicle snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct


TRACE_MAGIC = b"TMNFTRC1"
MODEL6_MAGIC = b"TMNFM6G1"
# CSceneVehicleCar::ComputeForcesModel{3,4,5,6}: one graph format, selected
# by tuning +0x354 in 0x007C69E0.
MODEL_VAS = {
    0x007FA770: "Model3",
    0x007FB5F0: "Model4",
    0x007FC170: "Model5",
    0x007C3E80: "Model6",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    data = args.trace.read_bytes()
    require(len(data) >= 36, "truncated Model6 trace")
    magic, va, count = struct.unpack_from("<8sII", data)
    require(magic == TRACE_MAGIC, "bad trace magic")
    require(va in MODEL_VAS, f"trace function 0x{va:08X} is not a ComputeForces model")
    require(count > 0, f"empty {MODEL_VAS[va]} trace")
    seq, input_count, output_count = struct.unpack_from("<IHH", data, 16)
    require(seq == 0 and input_count == 1 and output_count == 1,
            "unexpected first-record framing")
    tag, address, size = struct.unpack_from("<III", data, 24)
    require(tag == 0 and address == 0, "unexpected Model6 input tag")
    require(size >= 16 and 36 + size <= len(data), "truncated Model6 graph")
    graph = data[36:36 + size]
    graph_magic, version, total_size, phase = struct.unpack_from("<8sIII", graph)
    require(graph_magic == MODEL6_MAGIC, "bad Model6 graph magic")
    require(version == 1, "unsupported Model6 graph version")
    require(total_size == len(graph), "Model6 graph size mismatch")
    require(phase == 0, "vehicle snapshot is not an input graph")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(graph)
    print(f"{args.output}: {len(graph)} bytes from {MODEL_VAS[va]} (0x{va:08X})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
