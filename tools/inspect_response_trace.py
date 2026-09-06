#!/usr/bin/env python3
"""Print collision-response callback records for one simulation tick."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct


def read_buffer(data: memoryview, offset: int) -> tuple[int, bytes, int]:
    tag, _, size = struct.unpack_from("<III", data, offset)
    offset += 12
    return tag, bytes(data[offset : offset + size]), offset + size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("tick", type=int)
    args = parser.parse_args()

    data = memoryview(args.trace.read_bytes())
    if data[:8] != b"TMNFTRC1":
        raise ValueError("bad trace magic")
    count = struct.unpack_from("<I", data, 12)[0]
    offset = 16
    matched = 0
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
        if 16 not in inputs or struct.unpack("<I", inputs[16])[0] != args.tick:
            continue
        matched += 1
        graph = inputs[0]
        if graph[:8] != b"TMNFRSP1":
            raise ValueError("bad response graph magic")
        (
            _,
            kind,
            total_size,
            collision_count,
            body_count,
            material_count,
            tree_count,
            contact_count,
            event_count,
            replacement_count,
            collisions_offset,
            bodies_offset,
            materials_offset,
            contacts_offset,
            events_offset,
            replacements_offset,
            surface_materials_offset,
        ) = struct.unpack_from("<17I", graph, 8)
        print(
            f"sequence={sequence} kind={kind} size={total_size} "
            f"collisions={collision_count} events={event_count} "
            f"bodies={body_count} trees={tree_count} "
            f"materials={material_count} contacts={contact_count} "
            f"replacements={replacement_count}"
        )
        for event in range(event_count):
            event_offset = events_offset + event * 0xA8
            target, item_body = struct.unpack_from("<II", graph, event_offset)
            contact = event_offset + 8
            tree = struct.unpack_from("<I", graph, contact + 8)[0]
            surface = struct.unpack_from("<H", graph, contact + 12)[0]
            other_tree = struct.unpack_from("<I", graph, contact + 0x44)[0]
            other_surface = struct.unpack_from("<H", graph, contact + 0x48)[0]
            normal = struct.unpack_from("<fff", graph, contact + 16)
            print(
                f"  event={event} target=0x{target:08x} item={item_body} "
                f"tree={tree}/{other_tree} material={surface}/{other_surface} "
                f"normal=({normal[0]:.9g},{normal[1]:.9g},{normal[2]:.9g})"
            )
        expected = outputs[32]
        expected_body_count = struct.unpack_from("<I", expected, 24)[0]
        expected_replacement_count = struct.unpack_from("<I", expected, 44)[0]
        expected_bodies_offset = struct.unpack_from("<I", expected, 52)[0]
        expected_replacements_offset = struct.unpack_from("<I", expected, 68)[0]
        for body in range(expected_body_count):
            body_offset = expected_bodies_offset + body * 0x158
            if struct.unpack_from("<I", expected, body_offset + 64)[0] == 0:
                continue
            replacement_index, replacement_count = struct.unpack_from(
                "<II", expected, body_offset + 84
            )
            print(
                f"  output body={body + 1} "
                f"replacements={replacement_count}/{expected_replacement_count}"
            )
            for replacement in range(replacement_count):
                vector_offset = expected_replacements_offset + (
                    replacement_index + replacement
                ) * 12
                vector = struct.unpack_from("<III", expected, vector_offset)
                print(
                    f"    replacement={replacement} "
                    f"{vector[0]:08x},{vector[1]:08x},{vector[2]:08x}"
                )
        _ = (
            collisions_offset,
            bodies_offset,
            materials_offset,
            contacts_offset,
            replacements_offset,
            surface_materials_offset,
            outputs,
        )
    if matched == 0:
        raise RuntimeError("tick has no response records")


if __name__ == "__main__":
    main()
