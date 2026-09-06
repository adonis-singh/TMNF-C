#!/usr/bin/env python3
"""Dump the live vehicle ground-material table with fake-contact descriptors.

Walks car+0x68 (material manager: count +0x14, ptrs +0x18) and car+0x6c/0x70
(physical material id -> material index). For every material object reads
values[4] at +0x14, the bump mask object at +0x24, periods +0x28/+0x2c, impulse
scale +0x30, limit +0x34, and the mask image ([mask+0x48]: width +0x18,
height +0x1c, format +0x24, data +0x28), exactly the fields read by
0x007C3C00 CSceneVehicleCar::CreateFakeContacts.

usage: dump_fake_contact_materials.py OUT_DIR   (game must be running in the fixer prefix)
"""
from __future__ import annotations

import hashlib
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
    print(f"found {len(cars)} vehicle cars: {[hex(a) for a, _ in cars]}")
    base = out
    for car_address, car in cars:
        out = base / f"car_{car_address:08x}"
        out.mkdir(parents=True, exist_ok=True)
        dump_car(reader, out, car_address, car)
    reader.close()


def dump_car(reader, out, car_address, car):
    print(f"car 0x{car_address:08x}")
    manager = u32(car, 0x68)
    id_count = u32(car, 0x6C)
    ids_ptr = u32(car, 0x70)
    ids = struct.unpack(f"<{id_count}I", reader.read(ids_ptr, id_count * 4))
    mcount = u32(reader.read(manager + 0x14, 4))
    mptrs = struct.unpack(f"<{mcount}I", reader.read(u32(reader.read(manager + 0x18, 4)), mcount * 4))
    print(f"manager 0x{manager:08x} materials={mcount} ids={id_count}")
    result = {"car": car_address, "ground_ids": list(ids), "materials": []}
    for index, address in enumerate(mptrs):
        raw = reader.read(address, 0x38)
        entry = {
            "index": index,
            "address": address,
            "values": [f32(raw, 0x14 + 4 * i) for i in range(4)],
            "values_hex": [f"{u32(raw, 0x14 + 4 * i):08x}" for i in range(4)],
            "mask": u32(raw, 0x24),
            "period_x": f"{u32(raw, 0x28):08x}",
            "period_z": f"{u32(raw, 0x2C):08x}",
            "impulse_scale": f"{u32(raw, 0x30):08x}",
            "impulse_limit": f"{u32(raw, 0x34):08x}",
            "physical_ids": [pid for pid, mi in enumerate(ids) if mi == index],
        }
        if entry["mask"]:
            mask = reader.read(entry["mask"], 0x50)
            image = u32(mask, 0x48)
            entry["image"] = image
            if image:
                img = reader.read(image, 0x30)
                width, height, fmt, data = u32(img, 0x18), u32(img, 0x1C), u32(img, 0x24), u32(img, 0x28)
                bpp = (fmt >> 2) & 7
                entry.update(width=width, height=height, format=f"{fmt:08x}", bpp=bpp, data=data)
                if data and width and height and bpp:
                    pixels = reader.read(data, width * height * bpp)
                    channel0 = bytes(pixels[i * bpp] for i in range(width * height))
                    same = all(pixels[i * bpp:(i + 1) * bpp] == pixels[i * bpp:i * bpp + 1] * bpp
                               for i in range(width * height))
                    name = out / f"material_{index}_mask.bin"
                    name.write_bytes(channel0)
                    (out / f"material_{index}_mask_raw.bin").write_bytes(pixels)
                    entry["mask_sha256"] = hashlib.sha256(channel0).hexdigest()
                    entry["channels_equal"] = same
        result["materials"].append(entry)
        print(json.dumps(entry))
    (out / "materials.json").write_text(json.dumps(result, indent=1))


main()
