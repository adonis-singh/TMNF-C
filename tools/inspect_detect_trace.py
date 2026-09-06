#!/usr/bin/env python3
"""Print one graph-complete collision-detection trace record."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct


def read_buffer(data: memoryview, offset: int) -> tuple[int, bytes, int]:
    tag, _, size = struct.unpack_from("<III", data, offset)
    offset += 12
    return tag, bytes(data[offset : offset + size]), offset + size


def header(graph: bytes) -> dict[str, int]:
    if graph[:8] != b"TMNFDET1":
        raise ValueError("bad detect graph magic")
    names = (
        "version",
        "kind",
        "total_size",
        "return_value",
        "located_count",
        "surface_count",
        "iso_count",
        "plug_count",
        "buffer_count",
        "vertex_count",
        "face_count",
        "node_count",
        "material_id_count",
        "located_offset",
        "surfaces_offset",
        "isos_offset",
        "plugs_offset",
        "buffers_offset",
        "vertices_offset",
        "faces_offset",
        "nodes_offset",
        "material_ids_offset",
        "collision_records_offset",
        "collision_record_count",
        "boxes_offset",
        "box_count",
        "comparator_va",
    )
    values = struct.unpack_from("<27I", graph, 8)
    return dict(zip(names, values, strict=True))


def print_graph(label: str, graph: bytes) -> None:
    info = header(graph)
    print(
        f"{label}: return={info['return_value']} "
        f"buffers={info['buffer_count']} "
        f"collisions={info['collision_record_count']} "
        f"surfaces={info['surface_count']} faces={info['face_count']} "
        f"nodes={info['node_count']}"
    )
    for surface in range(info["surface_count"]):
        offset = info["surfaces_offset"] + surface * 0x38
        surface_id, material, kind = struct.unpack_from("<IHB", graph, offset)
        shape = struct.unpack_from("<fff", graph, offset + 8)
        print(
            f"  surface={surface_id} type={kind} material={material} "
            f"shape=({shape[0]:.9g},{shape[1]:.9g},{shape[2]:.9g})"
        )
    for buffer in range(info["buffer_count"]):
        offset = info["buffers_offset"] + buffer * 0x18
        buffer_id, count, capacity, record_index, active, has_active = (
            struct.unpack_from("<IIIIII", graph, offset)
        )
        print(
            f"  buffer={buffer_id} count={count}/{capacity} "
            f"record_index={record_index} active={active}/{has_active}"
        )
    for collision in range(info["collision_record_count"]):
        offset = info["collision_records_offset"] + collision * 0x4C
        separation = struct.unpack_from("<III", graph, offset + 0x10)
        normal = struct.unpack_from("<III", graph, offset + 0x1C)
        position = struct.unpack_from("<III", graph, offset + 0x28)
        material1, material2, flags = struct.unpack_from(
            "<HHI", graph, offset + 0x34
        )
        face_normal = struct.unpack_from("<III", graph, offset + 0x3C)
        print(
            f"  collision={collision} "
            f"separation={separation[0]:08x},{separation[1]:08x},{separation[2]:08x} "
            f"normal={normal[0]:08x},{normal[1]:08x},{normal[2]:08x} "
            f"position={position[0]:08x},{position[1]:08x},{position[2]:08x} "
            f"material={material1}/{material2} flags={flags} "
            f"face={face_normal[0]:08x},{face_normal[1]:08x},{face_normal[2]:08x}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("sequence", type=int)
    args = parser.parse_args()
    data = memoryview(args.trace.read_bytes())
    if data[:8] != b"TMNFTRC1":
        raise ValueError("bad trace magic")
    count = struct.unpack_from("<I", data, 12)[0]
    offset = 16
    emitted_before = 0
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
        if sequence == args.sequence:
            print(f"sequence={sequence} prior_output_collisions={emitted_before}")
            print_graph("input", inputs[0])
            print_graph("output", outputs[32])
            return
        emitted_before += header(outputs[32])["collision_record_count"]
    raise RuntimeError("sequence not found")


if __name__ == "__main__":
    main()
