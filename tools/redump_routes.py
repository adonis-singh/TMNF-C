#!/usr/bin/env python3
"""Upgrade committed route snapshots to version 3 (respawn locations).

For every track named on the command line (or every registered track with
--all) the game is started on the given lane with the route tracer, which
dumps a fresh version-1 route whose start and trigger records carry
CGameCtnBlock::GetSpawnLoc of their blocks (oracle/tracer/route_dump.c). The
spawn fields are then merged into the committed route by block index: the
metadata, the dense centerline, the checkpoint order and every pre-existing
field stay byte-identical, only the records widen.

    tools/redump_routes.py E04-Obstacle A01-Race
    tools/redump_routes.py --all --wineprefix oracle/wineprefix_lane1 \
        --port 8491 --display 91
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import onboard_track as ob  # noqa: E402
from generate_route_centerline import layout_route  # noqa: E402

ROOT = ob.ROOT
HEADER = 0xE0
OLD_START, OLD_TRIGGER = 0xE4, 0x58
NEW_START, NEW_TRIGGER = 0x114, 0x90
SPAWN = 0x30


def fail(message: str) -> None:
    raise RuntimeError(message)


def sections(data: bytes, version: int, start: int, trigger: int,
             reference: int) -> list[tuple[int, int, int]]:
    magic, actual, endian, header_size, count, size = struct.unpack_from(
        "<8sIIIIQ", data, 0)
    if (magic != b"TMNFROU1" or actual != version or endian != 0x12345678
            or header_size != HEADER or count != 5 or size != len(data)):
        fail(f"route header is not version {version}")
    if hashlib.sha256(data[HEADER:]).digest() != data[176:208]:
        fail("route payload digest mismatch")
    result = [struct.unpack_from("<QII", data, 96 + i * 16) for i in range(5)]
    for (offset, n, stride), expected in zip(
            result, (0x40, start, trigger, trigger, reference), strict=True):
        if stride != expected or offset + n * stride > len(data):
            fail("route section framing mismatch")
    return result


def records(data: bytes, section: tuple[int, int, int]) -> list[bytes]:
    offset, count, stride = section
    return [data[offset + i * stride:offset + (i + 1) * stride]
            for i in range(count)]


def widen(old: bytes, dump_by_block: dict[int, bytes], kind: str) -> bytes:
    block = struct.unpack_from("<I", old, 4)[0]
    new = dump_by_block.get(block)
    if new is None:
        fail(f"{kind} block {block} is missing from the route dump")
    # race_index may differ (the generator reorders checkpoints to the ghost's
    # crossing order); everything else the tracer wrote must be identical.
    if old[4:OLD_TRIGGER] != new[4:OLD_TRIGGER]:
        fail(f"{kind} block {block} differs between the committed route and the dump")
    return old + new[OLD_TRIGGER:NEW_TRIGGER]


def merge(committed: bytes, dump: bytes) -> bytes:
    old = sections(committed, 2, OLD_START, OLD_TRIGGER, 0x18)
    new = sections(dump, 1, NEW_START, NEW_TRIGGER, 0x10)
    old_meta = struct.unpack_from("<7I", committed, old[0][0])
    new_meta = struct.unpack_from("<7I", dump, new[0][0])
    # lap count, checkpoint count, total, limit, flags; the finish count may
    # differ because the generator keeps one of several finish blocks.
    for index in (0, 1, 4, 5, 6):
        if old_meta[index] != new_meta[index]:
            fail("route metadata differs between the committed route and the dump")
    old_start = records(committed, old[1])[0]
    new_start = records(dump, new[1])[0]
    if old_start != new_start[:OLD_START]:
        fail("route start differs between the committed route and the dump")
    start = old_start + new_start[OLD_START:NEW_START]
    dump_checkpoints = {struct.unpack_from("<I", r, 4)[0]: r
                        for r in records(dump, new[2])}
    dump_finishes = {struct.unpack_from("<I", r, 4)[0]: r
                     for r in records(dump, new[3])}
    checkpoints = [widen(r, dump_checkpoints, "checkpoint")
                   for r in records(committed, old[2])]
    finishes = [widen(r, dump_finishes, "finish")
                for r in records(committed, old[3])]
    payloads = [
        bytearray(committed[old[0][0]:old[0][0] + 0x40]),
        bytearray(start),
        bytearray(b"".join(checkpoints)),
        bytearray(b"".join(finishes)),
        bytearray(committed[old[4][0]:old[4][0] + old[4][1] * 0x18]),
    ]
    return layout_route(
        committed, payloads, (0x40, NEW_START, NEW_TRIGGER, NEW_TRIGGER, 0x18), 3)


def dump_route(track_name: str, staging: Path) -> Path:
    challenge = ob.challenge_for(track_name)
    ob.DOCUMENT_TRACKS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(challenge, ob.DOCUMENT_TRACKS / challenge.name)
    route_dir = staging / "routes"
    route_dir.mkdir(parents=True, exist_ok=True)
    dump = route_dir / f"{track_name}.tmnfroute"
    dump.unlink(missing_ok=True)
    ob.run_game(
        track_name,
        [ob.PYTHON, ROOT / "oracle/determinism_test.py",
         "--port", str(ob.PORT), "--map", track_name, "--ticks", "10",
         "--output-dir", staging / "probe"],
        staging / f"{track_name}.log",
        {"TMNF_ROUTE_DIR": str(route_dir),
         "TMNF_TRACK_DIR": str(staging / "tracks")},
    )
    if not dump.is_file():
        fail(f"route tracer wrote no dump for {track_name}")
    ob.verify_bound_snapshot(dump, b"TMNFROU1", bytes.fromhex(ob.file_sha256(challenge)), 64)
    return dump


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tracks", nargs="*")
    parser.add_argument("--all", action="store_true",
                        help="every track in oracle/tracks/manifest.txt and "
                             "oracle/results/wr/manifest.txt")
    parser.add_argument("--wineprefix", type=Path, default=ob.PREFIX)
    parser.add_argument("--port", type=int, default=ob.PORT)
    parser.add_argument("--display", default=ob.XVFB_DISPLAY)
    parser.add_argument("--cpu-set", default=ob.CPU_SET)
    parser.add_argument("--merge-only", action="store_true",
                        help="reuse dumps already in the staging directory")
    args = parser.parse_args()
    ob.set_lane(args.wineprefix, args.port, args.display, args.cpu_set)
    names = list(args.tracks)
    if args.all:
        seen = set(names)
        for manifest in (ob.MANIFEST, ROOT / "oracle/results/wr/manifest.txt"):
            for line in manifest.read_text().splitlines():
                if line and not line.startswith("#"):
                    name = line.split("|")[1]
                    if name not in seen:
                        seen.add(name)
                        names.append(name)
    if not names:
        parser.error("no tracks given")
    staging = ROOT / "build/redump_routes"
    staging.mkdir(parents=True, exist_ok=True)
    if not args.merge_only:
        ob.install_tracer("route")
    failures = []
    for name in names:
        route = ROOT / f"oracle/routes/{name}.tmnfroute"
        committed = route.read_bytes()
        if struct.unpack_from("<I", committed, 8)[0] == 3:
            print(f"{name}: already version 3", flush=True)
            continue
        try:
            dump = (staging / "routes" / f"{name}.tmnfroute" if args.merge_only
                    else dump_route(name, staging))
            merged = merge(committed, dump.read_bytes())
        except (RuntimeError, OSError) as error:
            print(f"{name}: FAILED {error}", flush=True)
            failures.append(name)
            continue
        route.write_bytes(merged)
        print(f"{name}: wrote version 3 ({len(committed)} -> {len(merged)} bytes)",
              flush=True)
    if not args.merge_only:
        ob.disable_tracer()
    if failures:
        print("failed: " + " ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"redump_routes: {error}", file=sys.stderr)
        raise SystemExit(1)
