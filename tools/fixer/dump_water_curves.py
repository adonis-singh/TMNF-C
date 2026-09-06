#!/usr/bin/env python3
"""Dump CFuncKeysReal curves at tuning +0x210/+0x214 (water impulse) for every
tuning key of the live CSceneVehicleTunings container.  usage: OUT_JSON"""
from __future__ import annotations

import json
from pathlib import Path
import struct
import sys

ROOT = Path("/home/adityas/Projects/TMNF-C")
sys.path.insert(0, str(ROOT / "oracle"))
import dump_vehicle_snapshot as dvs  # noqa: E402

dvs.PREFIX = ROOT / "oracle/wineprefix_fixer"
dvs.READER_EXE = Path("/home/adityas/fixer-tools/build/vehicle_memory_reader.exe")


def u32(b, o=0):
    return struct.unpack_from("<I", b, o)[0]


def main():
    out = Path(sys.argv[1])
    reader = dvs.MemoryReader()
    result = []
    for container in reader.scan(struct.pack("<I", dvs.TUNINGS_VTABLE)):
        head = reader.read(container, 0x28)
        count = u32(head, 0x14)
        data = u32(head, 0x18)
        key = u32(head, 0x24)
        if u32(head, 0x20) != dvs.TUNING_CLASS_ID or count == 0 or count > 64:
            continue
        addresses = struct.unpack(f"<{count}I", reader.read(data, count * 4))
        entry = {"container": container, "active_key": key, "tunings": []}
        for k, address in enumerate(addresses):
            tuning = reader.read(address, dvs.TUNING_SIZE)
            if u32(tuning, 0) != dvs.TUNING_VTABLE:
                break
            row = {"key": k, "scalars": {f"{o:#x}": f"{u32(tuning, o):08x}" for o in (0x204, 0x208, 0x20C, 0x21C, 0x220)}}
            for offset in (0x210, 0x214, 0x218):
                curve = u32(tuning, offset)
                if curve == 0:
                    row[f"{offset:#x}"] = None
                    continue
                raw = reader.read(curve, 0x2C)
                n = u32(raw, 0x14)
                pos = reader.read(u32(raw, 0x18), n * 4)
                vals = reader.read(u32(raw, 0x24), u32(raw, 0x20) * 4)
                row[f"{offset:#x}"] = {
                    "count": n, "value_count": u32(raw, 0x20),
                    "interpolation": struct.unpack_from("<i", raw, 0x28)[0],
                    "positions_hex": [f"{u32(pos, 4 * i):08x}" for i in range(n)],
                    "values_hex": [f"{u32(vals, 4 * i):08x}" for i in range(u32(raw, 0x20))],
                    "positions": list(struct.unpack(f"<{n}f", pos)),
                    "values": list(struct.unpack(f"<{u32(raw, 0x20)}f", vals)),
                }
            entry["tunings"].append(row)
        result.append(entry)
        print(json.dumps({"container": hex(container), "active_key": key, "count": count}))
        for row in entry["tunings"]:
            if row["key"] == key:
                print(json.dumps(row))
    out.write_text(json.dumps(result, indent=1))
    reader.close()


main()
