"""float32 geometry with the game's exact operation order.

TmForever runs the x87 FPU with PC=24, so every add/mul rounds to float32.
Computing each operation in double and rounding to float32 is bit-identical
(53 >= 2*24+2, so double rounding is innocuous). Isos are 12-tuples laid out
like GmIso4 (row-major 3x3 then translation); boxes are 6-tuples (center,
half extents) like GmBoxAligned.
"""
from __future__ import annotations

import struct

_F = struct.Struct("<f")
_ISO = struct.Struct("<12f")
_BOX = struct.Struct("<6f")


def f32(x: float) -> float:
    return _F.unpack(_F.pack(x))[0]


def mul(a: float, b: float) -> float:
    return f32(a * b)


def add(a: float, b: float) -> float:
    return f32(a + b)


def sub(a: float, b: float) -> float:
    return f32(a - b)


IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


def iso_from_bytes(raw: bytes) -> tuple:
    return _ISO.unpack(raw)


def iso_to_bytes(iso: tuple) -> bytes:
    return _ISO.pack(*iso)


def box_from_bytes(raw: bytes) -> tuple:
    return _BOX.unpack(raw)


def box_to_bytes(box: tuple) -> bytes:
    return _BOX.pack(*box)


def iso_set_mult(a: tuple, b: tuple) -> tuple:
    """GmIso4::SetMult (0x008E2750): result applies a first, then b.

    Addition order follows the compiled code exactly; products commute."""
    a0, a1, a2, a3, a4, a5, a6, a7, a8, at0, at1, at2 = a
    b0, b1, b2, b3, b4, b5, b6, b7, b8, bt0, bt1, bt2 = b
    return (
        add(add(mul(a6, b2), mul(a3, b1)), mul(a0, b0)),
        add(add(mul(a7, b2), mul(a4, b1)), mul(a1, b0)),
        add(add(mul(a8, b2), mul(a5, b1)), mul(b0, a2)),
        add(add(mul(b5, a6), mul(b3, a0)), mul(b4, a3)),
        add(add(mul(b5, a7), mul(a1, b3)), mul(b4, a4)),
        add(add(mul(b5, a8), mul(b3, a2)), mul(b4, a5)),
        add(add(mul(b7, a3), mul(b6, a0)), mul(b8, a6)),
        add(add(mul(b8, a7), mul(a4, b7)), mul(b6, a1)),
        add(add(mul(b8, a8), mul(b6, a2)), mul(b7, a5)),
        add(add(add(mul(at2, b2), mul(at0, b0)), mul(at1, b1)), bt0),
        add(add(add(mul(b5, at2), mul(b3, at0)), mul(b4, at1)), bt1),
        add(add(add(mul(b8, at2), mul(at0, b6)), mul(b7, at1)), bt2),
    )


def box_set_mult(box: tuple, iso: tuple) -> tuple:
    """GmBoxAligned::SetMult (0x008E5230), same order as src/collision.c."""
    cx, cy, cz, hx, hy, hz = box
    m0, m1, m2, m3, m4, m5, m6, m7, m8, t0, t1, t2 = iso
    return (
        add(add(add(mul(cx, m0), mul(m1, cy)), mul(m2, cz)), t0),
        add(add(add(mul(m4, cy), mul(cx, m3)), mul(m5, cz)), t1),
        add(add(add(mul(m7, cy), mul(m6, cx)), mul(m8, cz)), t2),
        add(add(mul(abs(m1), hy), mul(abs(m0), hx)), mul(abs(m2), hz)),
        add(add(mul(abs(m4), hy), mul(abs(m3), hx)), mul(abs(m5), hz)),
        add(add(mul(abs(m7), hy), mul(abs(m6), hx)), mul(abs(m8), hz)),
    )


def box_set_min_max(lo: tuple, hi: tuple) -> tuple:
    """GmBoxAligned::SetMinMax (0x0052C600)."""
    return (
        mul(add(lo[0], hi[0]), 0.5),
        mul(add(lo[1], hi[1]), 0.5),
        mul(add(lo[2], hi[2]), 0.5),
        mul(sub(hi[0], lo[0]), 0.5),
        mul(sub(hi[1], lo[1]), 0.5),
        mul(sub(hi[2], lo[2]), 0.5),
    )


def box_union(box: tuple, other: tuple) -> tuple:
    """GmBoxAligned::Union (0x008E5610)."""
    if box[3] < 0.0:
        box = other
    if not (0.0 <= other[3]):
        return box
    cx, cy, cz, hx, hy, hz = box
    ox, oy, oz, px, py, pz = other
    lo = [sub(cx, hx), sub(cy, hy), sub(cz, hz)]
    hi = [add(cx, hx), add(hy, cy), add(hz, cz)]
    olo = (sub(ox, px), sub(oy, py), sub(oz, pz))
    ohi = (add(px, ox), add(py, oy), add(pz, oz))
    for i in range(3):
        if olo[i] < lo[i]:
            lo[i] = olo[i]
        if hi[i] < ohi[i]:
            hi[i] = ohi[i]
    return box_set_min_max(tuple(lo), tuple(hi))


_QUARTER = (1.0, 0.0, -1.0, 0.0)


def rotate_quarter_y(q: int) -> tuple:
    """GmMat3::SetRotateQuarterY (0x008E1EB0). Note m[6] = -table[q-1] yields
    -0.0 for even quarters; that sign bit reaches the snapshot."""
    c = _QUARTER[q & 3]
    s = _QUARTER[(q - 1) & 3]
    return (c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c)


def mobil_loc(coord: tuple, direction: int, size: tuple, square: float, height: float) -> tuple:
    """CGameCtnBlock::GetMobilLoc (0x0060ACB0): block coordinate, cardinal
    direction and block size (in units) to the mobil GmIso4. Rotation pivots
    on the block's far corner so the footprint stays inside the coord cell."""
    x, y, z = coord
    tx = mul(float(x), square)
    ty = mul(float(y), height)
    tz = mul(float(z), square)
    sx, _, sz = size
    if direction == 0:
        rot = rotate_quarter_y(0)
    elif direction == 1:
        tx = add(mul(float(sz), square), tx)
        rot = rotate_quarter_y(3)
    elif direction == 2:
        tx = add(mul(float(sx), square), tx)
        tz = add(mul(float(sz), square), tz)
        rot = rotate_quarter_y(2)
    elif direction == 3:
        tz = add(mul(float(sx), square), tz)
        rot = rotate_quarter_y(1)
    else:
        raise ValueError(f"bad direction {direction}")
    return rot + (tx, ty, tz)
