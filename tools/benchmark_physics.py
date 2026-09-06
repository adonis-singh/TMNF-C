#!/usr/bin/env python3
"""Interleave engine-only benchmarks against saved and current shared libraries.

PYTHONPATH=python taskset -c 2 build/venv/bin/python tools/benchmark_physics.py \
    --libraries build/before.so build/libtmnf_physics.so --output build/physics.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess

from tmnf_rl.tracks import project_root, track_spec

LAPS = {"a01": ("policy_lap_inputs.bin", 2527),
        "a04": ("policy_lap_a04_5900_inputs.bin", 590),
        "b05": ("policy_lap_b05_25750_inputs.bin", 2575),
        "c03": ("policy_lap_c03_12660_inputs.bin", 1266)}
DEFAULT_TRACKS = [*LAPS, "desert-a1", "rally-a1", "snow-a1", "island-a1",
                  "coast-a1", "bay-a1", "stadium-a1"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libraries", type=Path, nargs="+", required=True)
    parser.add_argument("--binary", type=Path, default=root / "build/bench_physics")
    parser.add_argument("--tracks", nargs="+", default=DEFAULT_TRACKS)
    parser.add_argument("--worlds", type=int, default=128)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.worlds, args.trials) < 1:
        parser.error("worlds and trials must be positive")
    libraries = [p.resolve() for p in args.libraries]
    report = dict(format="tmnf-physics-benchmark-v1", platform=platform.platform(),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), worlds=args.worlds,
                  trials=args.trials, binary_sha256=sha(args.binary),
                  libraries={str(p): sha(p) for p in libraries}, cases=[], rows=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for track in args.tracks:
        spec = track_spec(track)
        track_path, vehicle = spec.track_path(root), spec.vehicle_path(root)
        filename, ticks = LAPS.get(track, (f"{track}_mixed_inputs.bin", 1200))
        inputs = root / "oracle/results" / filename
        report["cases"].append(dict(track=track, ticks=ticks,
            fixtures={str(p.relative_to(root)): sha(p) for p in [track_path, vehicle, inputs]}))
        states = set()
        for trial in range(args.trials):
            order = libraries if trial % 2 == 0 else libraries[::-1]
            for library in order:
                command = [str(args.binary), str(library), str(track_path), str(vehicle),
                           str(inputs), str(args.worlds), "1", str(ticks)]
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                row = json.loads(result.stdout)
                row.update(track=track, library=str(library), trial=trial)
                states.add(row["state_fnv1a64"])
                report["rows"].append(row)
                print(json.dumps(row), flush=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
        if len(states) != 1:
            raise RuntimeError(f"{track}: final engine states differ: {states}")
    summaries = []
    for track in args.tracks:
        for library in libraries:
            times = [r["seconds"] for r in report["rows"]
                     if r["track"] == track and r["library"] == str(library)]
            summaries.append(dict(track=track, library=str(library),
                median_seconds=statistics.median(times), mean_seconds=statistics.mean(times),
                stdev_seconds=statistics.stdev(times) if len(times) > 1 else 0.0))
    report.update(passed=True, summaries=summaries)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
