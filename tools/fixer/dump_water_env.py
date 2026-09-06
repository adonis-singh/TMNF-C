#!/usr/bin/env python3
"""Dump the live water environment reached by 0x007C2910 ApplyWaterForces.

item = car+0x28; corpus0 = *(*(item+0x38)); zone = *(corpus0+0x14);
env = zone->vtable[+0xa8](zone) -- the slot target is printed so it can be
disassembled; ENV_FIELD_OFFSET (optional argv[2]) gives the field the slot
returns so the grid at env+0x154 and levels at env+0x178/+0x17c are dumped.

usage: dump_water_env.py OUT_DIR [env_field_offset_hex]
"""
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


def f32(b, o=0):
    return struct.unpack_from("<f", b, o)[0]


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    env_field = int(sys.argv[2], 16) if len(sys.argv) > 2 else None
    reader = dvs.MemoryReader()
    cars = []
    for container in reader.scan(struct.pack("<I", dvs.TUNINGS_VTABLE)):
        head = reader.read(container, 0x28)
        if u32(head, 0x20) != dvs.TUNING_CLASS_ID:
            continue
        for pointer in reader.scan(struct.pack("<I", container)):
            if pointer < 0x64:
                continue
            car_address = pointer - 0x64
            car = reader.read(car_address, dvs.CAR_SIZE)
            if u32(car, 0) != dvs.VEHICLE_CAR_VTABLE or u32(car, 0x2E8) != 4:
                continue
            cars.append((car_address, car))
    print(f"found {len(cars)} cars: {[hex(a) for a, _ in cars]}")
    for car_address, car in cars:
        item = u32(car, 0x28)
        if item == 0:
            print(f"car {car_address:#x}: no item")
            continue
        try:
            item_raw = reader.read(item, 0x40)
        except Exception as error:
            print(f"car {car_address:#x}: item unreadable ({error})")
            reader = dvs.MemoryReader()
            continue
        count = u32(item_raw, 0x34)
        data = u32(item_raw, 0x38)
        corpus0 = u32(reader.read(data, 4))
        zone = u32(reader.read(corpus0 + 0x14, 4))
        vtable = u32(reader.read(zone, 4))
        slot = u32(reader.read(vtable + 0xA8, 4))
        info = {"car": car_address, "item": item, "corpus_count": count, "corpus0": corpus0,
                "zone": zone, "zone_vtable": vtable, "slot_a8": slot}
        print(json.dumps({k: hex(v) for k, v in info.items()}))
        zone_raw = reader.read(zone, 0x400)
        (out / f"zone_{car_address:08x}.bin").write_bytes(zone_raw)
        if env_field is None:
            continue
        env = u32(reader.read(zone + env_field, 4))
        info["env"] = env
        raw = reader.read(env, 0x190)
        (out / f"env_{car_address:08x}.bin").write_bytes(raw)
        count = u32(raw, 0x154 + 0x1c)
        data = u32(raw, 0x154 + 0x20)
        if count and data:
            cells = reader.read(data, count)
            (out / f"env_{car_address:08x}_cells.bin").write_bytes(cells)
            info["cells_nonzero"] = sum(1 for c in cells if c)
            info["cell_values"] = sorted(set(cells))
        grid = {
            "cell_x": f"{u32(raw, 0x154):08x}", "cell_z": f"{u32(raw, 0x158):08x}",
            "origin_x": f"{u32(raw, 0x15c):08x}", "origin_z": f"{u32(raw, 0x160):08x}",
            "width": u32(raw, 0x164), "height": u32(raw, 0x168),
            "default_cell": raw[0x16c], "flag_16c": raw[0x16c],
            "array_16d_16f": raw[0x16d:0x170].hex(),
            "buffer_170": f"{u32(raw, 0x170):08x}", "buffer_174": f"{u32(raw, 0x174):08x}",
            "level_178": f"{u32(raw, 0x178):08x}", "level_17c": f"{u32(raw, 0x17c):08x}",
            "raw_140_190": raw[0x140:0x190].hex(),
        }
        info["grid"] = grid
        print(json.dumps(grid))
        (out / f"env_{car_address:08x}.json").write_text(json.dumps(info, indent=1))
    reader.close()


main()
