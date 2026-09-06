#!/usr/bin/env python3
"""Dump one record of a trace file as float arrays for debugging."""
import struct, sys

path, want_seq = sys.argv[1], int(sys.argv[2])
d = open(path, "rb").read()
assert d[:8] == b"TMNFTRC1"
va, count = struct.unpack_from("<II", d, 8)
off = 16
sequences = []
for _ in range(count):
    seq, n_in, n_out = struct.unpack_from("<IHH", d, off)
    sequences.append(seq)
    off += 8
    bufs = []
    for _ in range(n_in + n_out):
        tag, addr, ln = struct.unpack_from("<III", d, off)
        off += 12
        raw = d[off:off+ln]
        off += ln
        bufs.append((tag, addr, ln, raw))
    if seq == want_seq:
        print(f"seq={seq} va=0x{va:08X}")
        for tag, addr, ln, raw in bufs:
            floats = struct.unpack_from(f"<{ln//4}f", raw) if ln % 4 == 0 else None
            print(f"  tag={tag} addr=0x{addr:08X} len={ln}")
            if floats:
                print("   ", [f"{x:.9g}" for x in floats])
        break
else:
    print(f"record count={count} sequences={sequences}")
