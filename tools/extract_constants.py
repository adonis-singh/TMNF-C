#!/usr/bin/env python3
"""Extract .rdata FP constants referenced by the physics decompilation.

Reads each DAT_ address as float32 and float64 from TmForever.exe and emits a
CSV. Width (dword/qword load) must be confirmed against the disassembly.
"""
import struct
import sys
import pathlib

EXE = pathlib.Path(sys.argv[1])
REFS = pathlib.Path(sys.argv[2])
OUT = pathlib.Path(sys.argv[3])

data = EXE.read_bytes()
pe = struct.unpack_from("<I", data, 0x3C)[0]
num_sections = struct.unpack_from("<H", data, pe + 6)[0]
opt_size = struct.unpack_from("<H", data, pe + 20)[0]
sec_off = pe + 24 + opt_size

sections = []
for i in range(num_sections):
    off = sec_off + i * 40
    name = data[off:off + 8].rstrip(b"\x00").decode("latin1")
    vsize, vaddr, rawsize, rawptr = struct.unpack_from("<IIII", data, off + 8)
    sections.append((name, vaddr, vsize, rawptr, rawsize))

IMAGE_BASE = 0x400000

def va_to_off(va):
    rva = va - IMAGE_BASE
    for name, vaddr, vsize, rawptr, rawsize in sections:
        if vaddr <= rva < vaddr + max(vsize, rawsize):
            return rawptr + (rva - vaddr)
    raise ValueError(f"VA {va:#x} not in any section")

rows = []
for line in REFS.read_text().split():
    va = int(line, 16)
    off = va_to_off(va)
    f32 = struct.unpack_from("<f", data, off)[0]
    f64 = struct.unpack_from("<d", data, off)[0]
    raw = data[off:off + 8].hex()
    rows.append((va, f32, f64, raw))

rows.sort()
with OUT.open("w") as w:
    w.write("va,float32,float64,raw_le_8bytes\n")
    for va, f32, f64, raw in rows:
        w.write(f"0x{va:08X},{f32!r},{f64!r},{raw}\n")
print(f"wrote {len(rows)} constants to {OUT}")
