#!/usr/bin/env python3
"""Compare a v1 and a v2 track snapshot: all payload sections except the new
corpus table must be identical once the 16-byte header growth is removed from
the relative pointers. usage: check_v2.py OLD NEW"""
import struct
import sys

def header(d):
    magic, ver, endian, hsize, nsec, fsize = struct.unpack_from("<8sIIIIQ", d, 0)
    secs = [struct.unpack_from("<QII", d, 64 + 32 + i * 16) for i in range(nsec)]
    return ver, hsize, nsec, secs

def section(d, sec):
    off, count, stride = sec
    return d[off:off + count * stride]

old = open(sys.argv[1], "rb").read()
new = open(sys.argv[2], "rb").read()
ov, oh, on, osecs = header(old)
nv, nh, nn, nsecs = header(new)
assert (ov, oh, on) == (1, 0x120, 9), (ov, oh, on)
assert (nv, nh, nn) == (2, 0x130, 10), (nv, nh, nn)
names = ["entries", "surfaces", "meshes", "vertices", "faces", "nodes", "material_ids", "material_data", "pairs"]
delta = nsecs[0][0] - osecs[0][0]
ok = True
for i, name in enumerate(names):
    a, b = section(old, osecs[i]), section(new, nsecs[i])
    if osecs[i][1:] != nsecs[i][1:]:
        print(f"{name}: count/stride differ {osecs[i][1:]} vs {nsecs[i][1:]}"); ok = False; continue
    if name == "entries":
        for k in range(osecs[i][1]):
            ea, eb = a[k * 0x60:(k + 1) * 0x60], b[k * 0x60:(k + 1) * 0x60]
            ra = struct.unpack_from("<Q", ea, 0x50)[0]; rb = struct.unpack_from("<Q", eb, 0x50)[0]
            if ea[:0x50] != eb[:0x50] or ea[0x58:] != eb[0x58:] or rb - ra != nsecs[1][0] - osecs[1][0]:
                print(f"entries: entry {k} differs"); ok = False; break
    elif name == "surfaces":
        for k in range(osecs[i][1]):
            sa, sb = a[k * 0x18:(k + 1) * 0x18], b[k * 0x18:(k + 1) * 0x18]
            ma, ia, ca = struct.unpack_from("<QQI", sa, 0); mb, ib, cb = struct.unpack_from("<QQI", sb, 0)
            if ca != cb or mb - ma != nsecs[2][0] - osecs[2][0] or ib - ia != nsecs[6][0] - osecs[6][0]:
                print(f"surfaces: surface {k} differs"); ok = False; break
    elif name == "meshes":
        for k in range(osecs[i][1]):
            sa, sb = a[k * 0x38:(k + 1) * 0x38], b[k * 0x38:(k + 1) * 0x38]
            if sa[:0x10] != sb[:0x10] or sa[0x18:0x20] != sb[0x18:0x20] or sa[0x28:0x30] != sb[0x28:0x30]:
                print(f"meshes: mesh {k} header differs"); ok = False; break
            for fo, si in ((0x10, 3), (0x20, 4), (0x30, 5)):
                ra = struct.unpack_from("<Q", sa, fo)[0]; rb = struct.unpack_from("<Q", sb, fo)[0]
                if rb - ra != nsecs[si][0] - osecs[si][0]:
                    print(f"meshes: mesh {k} rel {fo:#x} differs"); ok = False; break
    else:
        if a != b:
            print(f"{name}: bytes differ"); ok = False
corp = nsecs[9]
print(("OK" if ok else "MISMATCH"), f"corpora={corp[1]} stride={corp[2]}")
sys.exit(0 if ok else 1)
