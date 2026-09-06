#!/usr/bin/env python3
"""Refresh stale trigger transforms from a fresh game dump, without promotion.

Matches challenge block identity rather than process-dependent enumeration.
Preserves canonical checkpoint indices and the reference line. Regenerate
the centerline and validate race timings before promoting the output.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

from generate_route_centerline import layout_route, route_sections


def align_selected_finish(original: bytes, captured: bytes) -> tuple[bytes, list[int]]:
    """Project a full game dump onto the existing set of finish choices.

    This never chooses a new finish or silently adds an alternative.
    Call refresh afterwards to validate every other field and identity.
    """
    before, after = route_sections(original), route_sections(captured)
    a, n, s = before[3]
    b, m, t = after[3]
    if n < 1 or s != 144 or t != 144 or m < n:
        raise ValueError('selected-finish projection requires captured coverage of original finishes')
    if struct.unpack_from('<I', original, before[0][0]+8)[0] != n or \
            struct.unpack_from('<I', captured, after[0][0]+8)[0] != m:
        raise ValueError('finish metadata count differs from section')
    selected = [struct.unpack_from('<II', original, a+i*s+4) for i in range(n)]
    if len(set(selected)) != n:
        raise ValueError('duplicate original finish identity')
    rows = [captured[b+i*t:b+(i+1)*t] for i in range(m)]
    keys = [struct.unpack_from('<II', row, 4) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate captured finish identity')
    if any(key not in keys for key in selected):
        raise ValueError('selected finish is absent from capture')
    if m == n:
        return captured, []
    payloads = [bytearray(captured[o:o+c*w]) for o, c, w in after]
    payloads[3] = bytearray().join(rows[keys.index(key)] for key in selected)
    for i in range(n):
        struct.pack_into('<I', payloads[3], i*t, i)
    struct.pack_into('<I', payloads[0], 8, n)
    version = struct.unpack_from('<I', captured, 8)[0]
    projected = layout_route(captured, payloads, tuple(x[2] for x in after), version)
    return projected, [k[0] for k in keys if k not in selected]


def refresh(original: bytes, captured: bytes) -> tuple[bytes, list[dict]]:
    before, after = route_sections(original), route_sections(captured)
    if original[32:96] != captured[32:96]:
        raise ValueError('game executable or challenge hash differs')
    a, b = before[0][0], after[0][0]
    if original[a:a+28] != captured[b:b+28]:
        raise ValueError('race metadata differs')
    a, n, s = before[1]
    b, m, t = after[1]
    if (n, s) != (m, t) or original[a:a+n*s] != captured[b:b+m*t]:
        raise ValueError('start state differs')
    result = bytearray(original)
    changes = []
    for section in (2, 3):
        a, n, s = before[section]
        b, m, t = after[section]
        if (n, s) != (m, t) or s != 144:
            raise ValueError('trigger section framing differs')
        def records(data, offset, count, stride):
            rows = [data[offset+i*stride:offset+(i+1)*stride] for i in range(count)]
            keys = [struct.unpack_from('<II', row, 4) for row in rows]
            if len(set(keys)) != len(keys):
                raise ValueError('duplicate challenge block identity')
            return rows, keys
        old_rows, old_keys = records(original, a, n, s)
        new_rows, new_keys = records(captured, b, m, t)
        if set(old_keys) != set(new_keys):
            raise ValueError('trigger challenge block identities differ')
        lookup = dict(zip(new_keys, new_rows))
        for index, (old, key) in enumerate(zip(old_rows, old_keys)):
            new = lookup[key]
            if not all(math.isfinite(x) for x in struct.unpack_from('<12f', new, 40)):
                raise ValueError(f'trigger {key} has a nonfinite transform')
            # Runtime enumeration can differ. No other field besides the
            # transform may change in this deliberately constrained repair.
            if old[4:40] != new[4:40] or old[88:] != new[88:]:
                raise ValueError(f'trigger {key} has changes outside its transform')
            if old[40:88] != new[40:88]:
                result[a+index*s+40:a+index*s+88] = new[40:88]
                changes.append(dict(section=section, index=index, block_index=key[0],
                                    old_transform=list(struct.unpack_from('<12f', old, 40)),
                                    new_transform=list(struct.unpack_from('<12f', new, 40))))
    result[176:208] = hashlib.sha256(result[224:]).digest()
    return bytes(result), changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--captured', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--match-selected-finish', action='store_true',
                        help='match the original selected finish by block identity; report omitted alternatives')
    args = parser.parse_args()
    paths = [p.resolve() for p in (args.original, args.captured, args.output, args.report)]
    if len(set(paths)) != len(paths):
        parser.error('input, output and report paths must be distinct')
    if args.output.exists() or args.report.exists():
        parser.error('output and report must not already exist')
    old, new = args.original.read_bytes(), args.captured.read_bytes()
    aligned, omitted = align_selected_finish(old, new) if args.match_selected_finish else (new, [])
    result, changes = refresh(old, aligned)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(result)
    report = dict(original_sha256=hashlib.sha256(old).hexdigest(),
                  captured_sha256=hashlib.sha256(new).hexdigest(),
                  output_sha256=hashlib.sha256(result).hexdigest(), changes=changes,
                  omitted_finish_blocks=omitted,
                  centerline_regeneration_required=bool(changes))
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
