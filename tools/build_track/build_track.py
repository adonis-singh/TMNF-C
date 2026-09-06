#!/usr/bin/env python3
"""Build a TMNFTRK v3 collision snapshot from a .Challenge.Gbx without the game.

    build_track.py TRACK.Challenge.Gbx -o TRACK.tmnftrack [--compare ORACLE]

The build replays the game's block placement (CGameCtnChallenge::
InitChallengeData), scene population and static collision tree construction
with float32 arithmetic in the game's operation order, so the payload is
byte-identical to an in-game snapshot up to heap noise in inner BVH nodes
(see trkfile.canonicalize). --compare prints the section-by-section diff
against an in-game snapshot, canonicalizing both sides.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import struct
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bvh  # noqa: E402
import fp  # noqa: E402
import trkfile  # noqa: E402
from assets import (Assets, BuildError, DEFAULT_PACKS, cache_dir_for,  # noqa: E402
                    ensure_cache, gbxdump)
from scene import Challenge  # noqa: E402

# GmSurfMesh vtable in TmForever.exe; every static surface is a mesh.
MESH_VTABLE = 0x00BBE17C

# Runtime tables that live in the exe's .bss and are identical for every track
# (captured from the process by oracle/tracer/track_dump.c):
#  * 31 GmMaterial rows {friction, restitution} at VA 0x00D6EEC0,
#  * the car group's SColPair records against the static group.
MATERIAL_DATA = bytes.fromhex(
    "0000803f0000003f0000803f0000003f0000803f0000003f00000000000000000000803f0000003f"
    "0000803f0000003f0000803f0000003f0000803f0000003f0000803f0000003f0000000000000000"
    "00000000000000bf00000000000000000000803f0000003f0000803f0000003f0000803f0000003f"
    "0000803f0000003f0000803f0000003f0000803f0000003f0000803f0000003f0000803f0000003f"
    "0000803f0000003f0000803f0000003f0000803f0000003f0000803f3333733f0000803fcdcc4c3f"
    "0000803fcdcc4c3f0000803f0000003f0000803f0000003f0000803f0000003f0000803f0000003f"
    "0000803f0000003f")
COLLISION_PAIRS = [bytes.fromhex(h) for h in (
    "0300000001000000000000000000000001000000",
    "0300000003000000010000000100000001000000",
    "0300000004000000010000000100000001000000")]

# Stadium's CHmsZone water owner: 32 m cells over the 32x32 block grid, water
# surface at 8 m, floor at -992 m, no cell set by default. The set cells are
# exactly the blocks carrying Water (13) faces (every live Stadium dump agrees).
WATER_CELL = 32.0
WATER_GRID = 32
WATER_LEVEL = 8.0
WATER_FLOOR = -992.0
WATER_MATERIAL = 13


def water_map(track: trkfile.Track) -> trkfile.Water:
    cells = bytearray(WATER_GRID * WATER_GRID)
    for entry in track.entries:
        if not entry.active:
            continue
        surface = track.surfaces[entry.surface]
        # Most blocks cannot contain water. Avoid decoding every triangle of
        # every placement when none of its material slots is water.
        if WATER_MATERIAL not in surface.materials:
            continue
        mesh = track.meshes[surface.mesh]
        iso = struct.unpack("<12f", entry.iso)
        for f in range(len(mesh.faces) // 32):
            v0, v1, v2, mi = struct.unpack_from("<3IH", mesh.faces, f * 32 + 0x10)
            if surface.materials[mi] != WATER_MATERIAL:
                continue
            cx = cz = 0.0
            for v in (v0, v1, v2):
                x, y, z = struct.unpack_from("<3f", mesh.vertices, v * 12)
                wy = iso[3] * x + iso[4] * y + iso[5] * z + iso[10]
                if abs(wy - WATER_LEVEL) > 1e-4:
                    raise BuildError(f"water face at y={wy}, not {WATER_LEVEL}")
                cx += (iso[0] * x + iso[1] * y + iso[2] * z + iso[9]) / 3
                cz += (iso[6] * x + iso[7] * y + iso[8] * z + iso[11]) / 3
            fx, fz = int(cx // WATER_CELL), int(cz // WATER_CELL)
            if not (0 <= fx < WATER_GRID and 0 <= fz < WATER_GRID):
                raise BuildError(f"water face outside the grid at ({cx}, {cz})")
            cells[fz * WATER_GRID + fx] = 1
    return trkfile.Water(WATER_CELL, WATER_CELL, 0.0, 0.0, WATER_GRID, WATER_GRID, 0,
                         WATER_LEVEL, WATER_FLOOR, bytes(cells))


def challenge_json(assets: Assets, path: str) -> dict:
    """Dump the challenge through GbxDump, cached by file hash."""
    import json
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    out = os.path.join(assets.json_root, "challenges", digest + ".json")
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        gbxdump("dump", path, out)
    with open(out) as fh:
        return json.load(fh)["node"]


def build(assets: Assets, challenge_path: str, env: str | None = None,
          quality: str = "low") -> trkfile.Track:
    doc = challenge_json(assets, challenge_path)
    if env is not None and doc["collection"] != env:
        raise BuildError(f"challenge environment is {doc['collection']}, not {env}")
    ch = Challenge(assets, doc, quality)
    ch.init()
    corpora = ch.corpora()
    cells = bvh.flatten(corpora)
    if not cells:
        raise BuildError("no static surfaces")
    nodes = bvh.build(cells)

    track = trkfile.Track()
    track.track_sha256 = hashlib.sha256(open(challenge_path, "rb").read()).digest()
    track.material_data = MATERIAL_DATA
    track.collision_pairs = list(COLLISION_PAIRS)
    mesh_index: dict[int, int] = {}
    tree_ids: dict[int, int] = {}
    corpus_ids: dict[int, int] = {}
    for node in nodes:
        cell = node.cell
        if cell is None:
            track.entries.append(trkfile.Entry(
                node.skip, fp.box_to_bytes(node.box), b"\0" * 0x30, 0, 0, 0, 0))
            continue
        surface = cell.surface
        key = surface.mesh_key
        if key not in mesh_index:
            mesh_index[key] = len(track.meshes)
            track.meshes.append(trkfile.Mesh(
                MESH_VTABLE, 0, 7, 0, surface.vertices, surface.faces, surface.nodes))
        surface_index = len(track.surfaces)
        track.surfaces.append(trkfile.Surface(mesh_index[key], surface.materials))
        tree_key = id(cell.tree)
        if tree_key not in tree_ids:
            tree_ids[tree_key] = len(tree_ids) + 1
        if cell.corpus not in corpus_ids:
            corpus_ids[cell.corpus] = len(corpus_ids) + 1
            track.corpus_isos.append(fp.iso_to_bytes(corpora[cell.corpus][1]))
        track.entries.append(trkfile.Entry(
            node.skip, fp.box_to_bytes(node.box), fp.iso_to_bytes(cell.iso),
            cell.tree.flags, surface_index, tree_ids[tree_key], corpus_ids[cell.corpus]))
    if doc["collection"] != "Stadium":
        raise BuildError(f"water map constants are Stadium's, challenge is {doc['collection']}")
    track.water = water_map(track)
    return track


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("challenge", help=".Challenge.Gbx to build")
    ap.add_argument("-o", "--output", help="snapshot to write")
    ap.add_argument("--compare", metavar="SNAPSHOT", help="in-game snapshot to diff against")
    ap.add_argument("--packs", default=DEFAULT_PACKS,
                    help="game Packs directory (<Env>.pak, Game.pak, Resource.pak)")
    ap.add_argument("--cache", help="asset cache directory (default: per install under third_party/build_track_cache)")
    ap.add_argument("--env", help="require the challenge to use this environment (collection)")
    ap.add_argument("--quality", choices=("low", "high"), default="low",
                    help="display quality the game ran with (drives the warp decorator's flag bits)")
    ap.add_argument("--limit", type=int, default=20, help="rows shown per differing section")
    args = ap.parse_args(argv)

    cache = args.cache or cache_dir_for(args.packs)
    ensure_cache(cache, args.packs)
    assets = Assets(args.packs, cache)
    started = time.monotonic()
    try:
        track = build(assets, args.challenge, args.env, args.quality)
    except BuildError as error:
        print(f"build_track: {args.challenge}: {error}", file=sys.stderr)
        return 1
    elapsed = time.monotonic() - started
    digest = hashlib.sha256(track.payload()).hexdigest()
    print(f"built {os.path.basename(args.challenge)}: {len(track.entries)} entries, "
          f"{len(track.surfaces)} surfaces, {len(track.meshes)} meshes, "
          f"{len(track.corpus_isos)} corpora, payload sha256 {digest[:16]}, {elapsed:.2f}s")
    if args.output:
        trkfile.save(track, args.output)
    if args.compare:
        oracle = trkfile.load(args.compare)
        raw_identical = trkfile.build_image(track)[trkfile.HEADER_SIZE:] == \
            trkfile.build_image(oracle)[trkfile.HEADER_SIZE:]
        lines = trkfile.compare(trkfile.canonicalize(track), trkfile.canonicalize(oracle),
                                limit=args.limit)
        print(f"raw payload identical: {'yes' if raw_identical else 'no'}")
        if lines:
            print("canonical payload identical: no")
            print("\n".join(lines))
            return 2
        print("canonical payload identical: yes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
