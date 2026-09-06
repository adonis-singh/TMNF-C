#!/usr/bin/env python3
"""Fail closed on game-data files in the public Git index."""
from pathlib import Path
import hashlib
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_DIRS = ('local/', 'artifacts/', 'research/', 'analysis/', 'third_party/',
                'viewer/assets/', 'viewer/scenes/', 'oracle/tracks/',
                'oracle/vehicles/', 'oracle/routes/', 'oracle/results/',
                'oracle/traces/', 'oracle/replays/', 'oracle/ghosts/', 'oracle/wineprefix')
BLOCKED_EXTENSIONS = {'.pak', '.gbx', '.exe', '.dll', '.asi', '.bin', '.pt', '.pth',
                      '.tga', '.dds', '.png', '.jpg', '.gltf', '.glb',
                      '.tmnftrack', '.tmnfvehicle', '.tmnfroute'}
MASK_HASH = '6dac49cc6f6b02b3c9684a987dd32d507b9dda63f463665745d24113c4c01e70'


def audit():
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    failures = []
    for name in filter(None, names):
        path = ROOT / name
        if name.startswith(BLOCKED_DIRS) or Path(name).suffix.lower() in BLOCKED_EXTENSIONS:
            failures.append(name + ': excluded data path or extension')
            continue
        if path.is_symlink():
            failures.append(name + ': symlink requires explicit review')
            continue
        data = path.read_bytes()
        if data.startswith((b'TMNFTRK1', b'TMNFM6G1', b'TMNFROU1', b'MZ', b'GBX')):
            failures.append(name + ': game or executable format')
        # The bundled font files have an explicit redistribution license.
        if name.startswith('viewer/fonts/') and path.suffix == '.woff2':
            continue
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError:
            failures.append(name + ': unexpected binary file')
            continue
        if path.suffix in ('.h', '.c', '.cu'):
            for array in re.findall(r'=\s*\{([^{}]+)\}', text, re.S):
                tokens = re.findall(r'\b(?:0x[0-9a-fA-F]+|[0-9]+)\b', array)
                if len(tokens) == 16384:
                    values = [int(x, 16 if x.startswith('0x') else 10) for x in tokens]
                    if all(0 <= x <= 255 for x in values) and hashlib.sha256(bytes(values)).hexdigest() == MASK_HASH:
                        failures.append(name + ': embedded game image')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'Public source inventory passed ({len([x for x in names if x])} tracked files).')


if __name__ == '__main__':
    audit()
