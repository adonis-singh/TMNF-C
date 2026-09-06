"""90-second PPO learning smoke test (see analysis/training_bisect.md).

Runs the start-only seed-1 configuration as a registered run on the GPU named
by ``CUDA_VISIBLE_DEVICES`` with the reserved CPU set, then requires the trailing 100-episode distance
mean to exceed 500 m, improve by at least 400 m from the first window and stay
at least half of the run's peak.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from tmnf_rl.tracks import project_root
from tmnf_rl.utils import CPU_AFFINITY, read_json


HEALTH_ARGS = [
    "--seed", "1",
    "--track", "a01",
    "--num-envs", "256",
    "--num-steps", "128",
    "--action-repeat", "5",
    "--duration-minutes", "1.5",
    "--max-updates", "0",
    "--learning-rate", "0.00025",
    "--discount-per-tick", "0.9999",
    "--gae-lambda", "0.95",
    "--num-minibatches", "4",
    "--update-epochs", "4",
    "--clip-coef", "0.2",
    "--ent-coef", "0.01",
    "--vf-coef", "0.5",
    "--max-grad-norm", "0.5",
    "--hidden-size", "256",
    "--max-race-ticks", "12000",
    "--horizon-ticks", "0",
    "--off-track-grace-ticks", "100",
    "--stuck-grace-ticks", "500",
    "--no-staggered-phases",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tmnf_rl.health_check")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", default="build/runs")
    parser.add_argument("--physics-library", default="build/libtmnf_physics.so")
    parser.add_argument(
        "--thread-count", type=int, default=28,
        help="env threads; results do not depend on it (determinism across thread "
        "counts is tested), throughput and therefore the update count in 90 s do",
    )
    args = parser.parse_args(argv)

    root = project_root()
    os.sched_setaffinity(0, CPU_AFFINITY)
    command = [
        sys.executable,
        "-m",
        "tmnf_rl.train",
        *HEALTH_ARGS,
        "--thread-count",
        str(args.thread_count),
        "--physics-library",
        str(Path(args.physics_library).resolve()),
        "--runs-root",
        args.runs_root,
        "--run-id",
        args.run_id,
    ]
    print(
        json.dumps(
            {"command": shlex.join(command), "cpu_affinity": sorted(CPU_AFFINITY)},
            sort_keys=True,
        ),
        flush=True,
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "python")
    subprocess.run(command, cwd=root, env=environment, check=True)

    runs_root = Path(args.runs_root)
    run_dir = (root / runs_root if not runs_root.is_absolute() else runs_root) / args.run_id
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    distances = [
        float(row["train/fullstart_distance_mean_100"])
        for row in rows
        if row["train/fullstart_distance_mean_100"]
    ]
    initial_distance = distances[0]
    final_distance = distances[-1]
    peak_distance = max(distances)
    if final_distance < 500.0:
        raise AssertionError(f"final distance mean collapsed to {final_distance:.3f} m")
    if final_distance - initial_distance < 400.0:
        raise AssertionError(
            "distance mean failed to improve by 400 m: "
            f"{initial_distance:.3f} -> {final_distance:.3f} m"
        )
    if final_distance < 0.5 * peak_distance:
        raise AssertionError(
            f"distance mean collapsed from {peak_distance:.3f} to {final_distance:.3f} m"
        )

    summary = read_json(run_dir / "summary.json")
    print(
        json.dumps(
            {
                "event": "health_check_passed",
                "run_id": args.run_id,
                "updates": summary["updates"],
                "initial_distance_mean_100": initial_distance,
                "peak_distance_mean_100": peak_distance,
                "final_distance_mean_100": final_distance,
                "finishes": summary["finishes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])
