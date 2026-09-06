"""Multi-seed evaluation protocol: K seeds x T minutes, fixed wall-clock checks.

``python -m tmnf_rl.protocol --name a01_ppo --seeds 1,2,3 --minutes 20``
trains one registered run per seed (sequentially, one GPU), then aggregates
``evaluations.csv`` at every scheduled minute into median and IQR across seeds
and writes a small JSON summary meant to be committed.

Rules baked in:

* every run in a protocol must report the same physics and code hash;
* laps are only aggregated over seeds that finished; the finishing-seed count
  is always reported next to the lap statistics;
* comparisons are by wall-clock minute, never by environment steps.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from tmnf_rl.config import config_from_dict, load_config_file
from tmnf_rl.registry import RunRegistry
from tmnf_rl.tracks import project_root
from tmnf_rl.utils import utc_now_iso, write_json_atomic


def _quartiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "q1": None, "q3": None, "min": None, "max": None, "n": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "q1": float(np.percentile(array, 25.0)),
        "q3": float(np.percentile(array, 75.0)),
        "min": float(array.min()),
        "max": float(array.max()),
        "n": int(array.size),
    }


def aggregate(
    evaluations: dict[int, list[dict[str, str]]], expected_minutes: list[float]
) -> list[dict[str, Any]]:
    """Aggregate per-seed evaluations.csv rows at each scheduled minute.

    The greedy trajectory contributes `greedy_finished_seeds` and its lap
    statistics; the sampled evaluation contributes the finish rate and
    distance quartiles (F29: a rate over one trajectory is not a rate).
    """
    table: list[dict[str, Any]] = []
    for minute in expected_minutes:
        greedy_finished: list[int] = []
        greedy_laps: list[float] = []
        greedy_distances: list[float] = []
        finish_rates: list[float] = []
        median_laps: list[float] = []
        best_laps: list[float] = []
        distances: list[float] = []
        seeds_present: list[int] = []
        for seed, rows in sorted(evaluations.items()):
            matches = [row for row in rows if float(row["scheduled_minutes"]) == minute]
            if len(matches) != 1:
                raise ValueError(
                    f"seed {seed} has {len(matches)} evaluations at minute {minute:g}, "
                    "expected exactly one"
                )
            row = matches[0]
            seeds_present.append(seed)
            greedy_finished.append(int(row["eval_fullstart/finished"]))
            greedy_distances.append(float(row["eval_fullstart/distance_mean"]))
            if row["eval_fullstart/best_lap_ms"]:
                greedy_laps.append(float(row["eval_fullstart/best_lap_ms"]))
            finish_rates.append(float(row["eval_sampled/finish_rate"]))
            distances.append(float(row["eval_sampled/distance_mean"]))
            if row["eval_sampled/median_lap_ms"]:
                median_laps.append(float(row["eval_sampled/median_lap_ms"]))
                best_laps.append(float(row["eval_sampled/best_lap_ms"]))
        table.append(
            {
                "minute": minute,
                "seeds": seeds_present,
                "greedy_finished_seeds": sum(greedy_finished),
                "greedy_lap_ms": _quartiles(greedy_laps),
                "greedy_distance_mean": _quartiles(greedy_distances),
                "finishing_seeds": len(median_laps),
                "finish_rate": _quartiles(finish_rates),
                "distance_mean": _quartiles(distances),
                "median_lap_ms": _quartiles(median_laps),
                "best_lap_ms": _quartiles(best_laps),
            }
        )
    return table


def read_evaluations(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_protocol(
    *,
    name: str,
    seeds: list[int],
    minutes: float,
    base: dict[str, Any],
    registry: RunRegistry,
    root: Path,
) -> dict[str, Any]:
    run_ids: dict[int, str] = {}
    for seed in seeds:
        run_id = f"{name}_s{seed}"
        config = config_from_dict(
            {
                **base,
                "seed": seed,
                "duration_minutes": minutes,
                "max_updates": 0,
                "run_id": run_id,
                "runs_root": str(registry.root),
                "spectate": base.get("spectate", 0),
            }
        )
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(config.to_dict(), handle)
            config_path = Path(handle.name)
        command = [sys.executable, "-m", "tmnf_rl.train", "--config", str(config_path)]
        print(json.dumps({"event": "protocol_run", "seed": seed, "run_id": run_id}), flush=True)
        completed = subprocess.run(command, cwd=root, check=False)
        config_path.unlink(missing_ok=True)
        if completed.returncode != 0:
            raise RuntimeError(f"seed {seed} trainer exited with {completed.returncode}")
        run_ids[seed] = run_id
    return summarize(name=name, run_ids=run_ids, minutes=minutes, registry=registry)


def summarize(
    *, name: str, run_ids: dict[int, str], minutes: float, registry: RunRegistry
) -> dict[str, Any]:
    payloads = {seed: registry.open_run(run_id).payload for seed, run_id in run_ids.items()}
    physics = {payload["physics_sha256"] for payload in payloads.values()}
    code = {payload["code_sha256"] for payload in payloads.values()}
    if len(physics) != 1 or len(code) != 1:
        raise RuntimeError(
            f"protocol runs disagree on physics {sorted(physics)} or code {sorted(code)}"
        )
    # A run resumed across a code change has rows its code_sha256 did not
    # produce; it cannot stand in a protocol that claims one code hash.
    changed = {
        seed: payload["code_changes"]
        for seed, payload in payloads.items()
        if payload.get("code_changes")
    }
    if changed:
        raise RuntimeError(
            "protocol runs were resumed across python/tmnf_rl changes and are not "
            f"reproducible from one code_sha256: {changed}"
        )
    statuses = {seed: payload["status"] for seed, payload in payloads.items()}
    if any(status != "finished" for status in statuses.values()):
        raise RuntimeError(f"protocol runs are not all finished: {statuses}")
    interval = float(next(iter(payloads.values()))["args"]["eval_interval_minutes"])
    expected = [round(k * interval, 6) for k in range(int(np.floor(minutes / interval)) + 1)]
    evaluations = {
        seed: read_evaluations(registry.run_dir(run_id) / "evaluations.csv")
        for seed, run_id in run_ids.items()
    }
    return {
        "format": "tmnf-rl-protocol",
        "version": 1,
        "name": name,
        "generated_at": utc_now_iso(),
        "seeds": sorted(run_ids),
        "minutes": minutes,
        "eval_interval_minutes": interval,
        "physics_sha256": next(iter(physics)),
        "code_sha256": next(iter(code)),
        "git_commit": sorted({payload["git_commit"] for payload in payloads.values()}),
        "runs": {str(seed): run_id for seed, run_id in sorted(run_ids.items())},
        "per_seed": {
            str(seed): {
                "updates": payload["summary"].get("updates"),
                "physics_steps": payload["summary"].get("physics_steps"),
                "physics_steps_per_second": payload["summary"].get("physics_steps_per_second"),
                "train_finishes": payload["summary"].get("finishes"),
                "train_best_lap_ms": payload["summary"].get("best_lap_ms"),
                "eval_best_lap_ms": payload["summary"].get("eval_best_lap_ms"),
            }
            for seed, payload in sorted(payloads.items())
        },
        "checkpoints": aggregate(evaluations, expected),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(prog="tmnf_rl.protocol")
    parser.add_argument("--name", required=True)
    parser.add_argument("--seeds", required=True, help="comma-separated, e.g. 1,2,3")
    parser.add_argument("--minutes", type=float, required=True)
    parser.add_argument("--config", type=Path, default=None, help="base TrainConfig JSON")
    parser.add_argument("--track", default=None)
    parser.add_argument("--eval-interval-minutes", type=float, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="aggregate existing runs <name>_s<seed> without training",
    )
    args = parser.parse_args(argv)
    seeds = sorted({int(value) for value in args.seeds.split(",") if value.strip()})
    if not seeds:
        raise ValueError("at least one seed is required")
    if args.minutes <= 0:
        raise ValueError("minutes must be positive")
    root = project_root()
    registry = RunRegistry(args.runs_root or (root / "build" / "runs"))
    base: dict[str, Any] = {}
    if args.config is not None:
        base.update(load_config_file(args.config))
    if args.track is not None:
        base["track"] = args.track
    if args.eval_interval_minutes is not None:
        base["eval_interval_minutes"] = args.eval_interval_minutes
    config_from_dict({**base, "duration_minutes": args.minutes})  # fail fast on typos

    if args.summarize_only:
        report = summarize(
            name=args.name,
            run_ids={seed: f"{args.name}_s{seed}" for seed in seeds},
            minutes=args.minutes,
            registry=registry,
        )
    else:
        report = run_protocol(
            name=args.name,
            seeds=seeds,
            minutes=args.minutes,
            base=base,
            registry=registry,
            root=root,
        )
    output = args.output or (root / "artifacts" / "protocols" / f"{args.name}.json")
    write_json_atomic(output, report)
    print(json.dumps({"event": "protocol", "output": str(output), **report}, sort_keys=True), flush=True)
    return report


if __name__ == "__main__":
    main(sys.argv[1:])
