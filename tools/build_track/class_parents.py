#!/usr/bin/env python3
"""Extract the game's class hierarchy from TmForever.exe.

Every engine class has a `MwIsKindOf(classId)` virtual that is compiled as a
chain of `cmp eax, <own id>` / `cmp eax, <parent id>` / ... / `cmp eax,
0x01001000` (CMwNod). Reading those immediates gives the exact parent of each
class. The pak cipher is re-keyed with the parent class id of every node the
game instantiates while reading, so GbxDump needs this table (see GameRekey).

Usage: class_parents.py TmForever.exe tmnf_symbols.tsv OUT.txt
"""
import re
import struct
import sys

IMAGE_BASE = 0x400000
CMWNOD = 0x01001000


def main():
    exe_path, symbols_path, out_path = sys.argv[1:4]
    exe = open(exe_path, 'rb').read()
    rows = []
    for line in open(symbols_path):
        m = re.match(r'0x([0-9A-F]+)\t\?MwIsKindOf@(\w+)@@UBEHK@Z', line)
        if not m:
            continue
        va = int(m.group(1), 16)
        name = m.group(2)
        code = exe[va - IMAGE_BASE:va - IMAGE_BASE + 96]
        ids = []
        i = 0
        while i < len(code):
            if code[i] == 0x3D:
                value = struct.unpack_from('<I', code, i + 1)[0]
                ids.append(value)
                i += 5
                if value == CMWNOD:
                    break
            else:
                i += 1
        if len(ids) < 2 or ids[-1] != CMWNOD:
            # UI card classes tail-jump into the parent's MwIsKindOf; none of
            # them ever appears inside a pak-encrypted asset.
            continue
        rows.append((ids[0], ids[1], name))
    rows.sort()
    with open(out_path, 'w') as out:
        out.write('# class parent name (from MwIsKindOf chains in TmForever.exe)\n')
        for class_id, parent, name in rows:
            out.write(f'{class_id:08X} {parent:08X} {name}\n')
    print(f'{len(rows)} classes')


if __name__ == '__main__':
    main()
