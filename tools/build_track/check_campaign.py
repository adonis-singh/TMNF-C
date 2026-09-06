#!/usr/bin/env python3
"""Build every manifest track offline and hold it against the in-game snapshot.

    check_campaign.py --out-dir BUILD/tracks [--replay-tick BUILD/tests/replay_tick]

A track passes when its payload is byte-identical to oracle/tracks/<name>
.tmnftrack, or identical after trkfile.canonicalize (heap noise in inactive
BVH slots), or, failing both, when every non-pending reference replay in the
manifest is byte-exact on the built snapshot (physics equivalence). The exit
status is non-zero when any track fails all three.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trkfile  # noqa: E402
from track_env import parse_track  # noqa: E402
from assets import (Assets, BuildError, DEFAULT_PACKS, DOTNET_INSTALL_HINT, cache_complete,  # noqa: E402
                    cache_dir_for, ensure_cache, find_dotnet)
from build_track import build  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ctest SKIP_RETURN_CODE for build_track_campaign: the prerequisites (game
# install, dotnet for the first cache build) are missing on this machine.
SKIP = 77


def read_manifest(path: str, oracle_dir: str):
    """Nations manifest rows, then every other in-game snapshot in oracle_dir
    (A01 is the golden track and has its own tests; it must still build
    identically). United rows are skipped: the builder reads Stadium paks and
    the United challenges live in another prefix."""
    seen = set()
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        track_id, name, sha, replays = line.split("|")
        seen.add(name)
        if parse_track(name).campaign != "nations":
            continue
        yield track_id, name, sha, [r for r in replays.split(",") if r and not r.startswith("pending:")]
    for snapshot in sorted(glob.glob(os.path.join(oracle_dir, "*.tmnftrack"))):
        name = os.path.basename(snapshot)[:-len(".tmnftrack")]
        if name not in seen:
            yield name.split("-")[0].lower(), name, "", []


def find_challenge(root: str, name: str) -> str:
    hits = glob.glob(os.path.join(root, "**", f"{name}.Challenge.Gbx"), recursive=True)
    if len(hits) != 1:
        raise BuildError(f"{name}: expected one challenge under {root}, found {len(hits)}")
    return hits[0]


def replay_ok(replay_tick: str, snapshot: str, track_id: str, sha: str, replay: str) -> bool:
    vehicle = os.path.join(REPO, "oracle", "vehicles", f"{track_id.upper()}-Stadium.tmnfvehicle")
    reference = os.path.join(REPO, "oracle", "results", f"{track_id}_{replay}.bin")
    inputs = os.path.join(REPO, "oracle", "results", f"{track_id}_{replay}_inputs.bin")
    proc = subprocess.run([replay_tick, snapshot, vehicle, reference, "input_file", inputs, sha],
                          capture_output=True, text=True)
    return proc.returncode == 0 and "byte-exact" in proc.stdout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", default=os.path.join(REPO, "oracle", "tracks", "manifest.txt"))
    ap.add_argument("--oracle-dir", default=os.path.join(REPO, "oracle", "tracks"))
    ap.add_argument("--packs", default=DEFAULT_PACKS)
    ap.add_argument("--challenges", help="directory searched recursively for <name>.Challenge.Gbx "
                    "(default: <packs>/../GameData/Tracks/Campaigns)")
    ap.add_argument("--out-dir", required=True, help="where the built snapshots are written")
    ap.add_argument("--replay-tick", help="tests/replay_tick binary for the physics-equivalence fallback")
    ap.add_argument("--only", nargs="*", help="track ids to check (default: all)")
    args = ap.parse_args(argv)

    challenges = args.challenges or os.path.join(os.path.dirname(args.packs), "GameData", "Tracks", "Campaigns")
    cache = cache_dir_for(args.packs)
    if not os.path.isdir(args.packs) or not os.path.isdir(challenges):
        print(f"SKIP: game install not found ({args.packs}, {challenges})")
        return SKIP
    if not cache_complete(cache) and find_dotnet() is None:
        print(f"SKIP: asset cache {cache} is missing and dotnet is not available to build it; {DOTNET_INSTALL_HINT}")
        return SKIP
    ensure_cache(cache, args.packs)
    assets = Assets(args.packs, cache)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    failed = 0
    for track_id, name, sha, replays in read_manifest(args.manifest, args.oracle_dir):
        if args.only and track_id not in args.only:
            continue
        started = time.monotonic()
        oracle_path = os.path.join(args.oracle_dir, f"{name}.tmnftrack")
        out_path = os.path.join(args.out_dir, f"{name}.tmnftrack")
        try:
            track = build(assets, find_challenge(challenges, name))
        except BuildError as error:
            rows.append((track_id, name, "error", "-", "-", "FAIL", f"{error}"))
            failed += 1
            continue
        trkfile.save(track, out_path)
        oracle = trkfile.load(oracle_path)
        raw = trkfile.build_image(track)[trkfile.HEADER_SIZE:] == trkfile.build_image(oracle)[trkfile.HEADER_SIZE:]
        canonical = raw or not trkfile.compare(trkfile.canonicalize(track), trkfile.canonicalize(oracle), limit=1)
        replay = "-"
        verdict = "identical" if raw else "canonical" if canonical else "FAIL"
        if not canonical and args.replay_tick and replays:
            results = [replay_ok(args.replay_tick, out_path, track_id, sha, r) for r in replays]
            replay = f"{sum(results)}/{len(results)}"
            if all(results):
                verdict = "replay"
        if verdict == "FAIL":
            failed += 1
        rows.append((track_id, name, "yes" if raw else "no", "yes" if canonical else "no", replay, verdict,
                     f"{time.monotonic() - started:.1f}s"))
        print(f"{track_id} {name}: raw={rows[-1][2]} canonical={rows[-1][3]} replays={replay} -> {verdict}",
              flush=True)

    print()
    print(f"{'id':<4} {'track':<18} {'raw':<4} {'canon':<6} {'replays':<8} verdict")
    for row in rows:
        print(f"{row[0]:<4} {row[1]:<18} {row[2]:<4} {row[3]:<6} {row[4]:<8} {row[5]} ({row[6]})")
    counts = {v: sum(1 for r in rows if r[5] == v) for v in ("identical", "canonical", "replay", "FAIL")}
    print(f"\n{len(rows)} tracks: {counts['identical']} identical, {counts['canonical']} canonical, "
          f"{counts['replay']} replay-equivalent, {counts['FAIL']} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
