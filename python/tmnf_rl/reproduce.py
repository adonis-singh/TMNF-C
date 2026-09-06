"""Re-run a registered run and assert identical metric rows.

``python -m tmnf_rl.reproduce RUN_ID [--updates N]`` starts a fresh trainer
process with the original run's exact config (only the stopping criteria are
changed to ``max_updates=N``), then compares the first N rows of
``metrics.csv`` column by column, ignoring wall-clock columns. It refuses to
run if the physics library or the tmnf_rl source hash differs from the
original run, because a match under different code is not a reproduction and a
mismatch under different code proves nothing about the seed.
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

from tmnf_rl.utils import read_json, require_single_gpu_env, sha256_file, write_json_atomic

require_single_gpu_env()

import torch  # noqa: E402

from tmnf_rl import provenance  # noqa: E402
from tmnf_rl.agents.ppo import TIMING_COLUMNS, select_device  # noqa: E402
from tmnf_rl.config import config_from_dict  # noqa: E402
from tmnf_rl.registry import RunRegistry  # noqa: E402
from tmnf_rl.tracks import project_root  # noqa: E402


def read_rows(path: Path, limit: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit]


def compare_rows(
    original: list[dict[str, str]],
    replica: list[dict[str, str]],
    ignore: frozenset[str] = TIMING_COLUMNS,
) -> dict[str, Any]:
    compared = min(len(original), len(replica))
    mismatches: list[dict[str, Any]] = []
    columns = [name for name in original[0] if name not in ignore] if original else []
    for index in range(compared):
        for column in columns:
            if original[index].get(column) != replica[index].get(column):
                mismatches.append(
                    {
                        "update": index + 1,
                        "column": column,
                        "original": original[index].get(column),
                        "replica": replica[index].get(column),
                    }
                )
    return {
        "compared_updates": compared,
        "compared_columns": len(columns),
        "ignored_columns": sorted(ignore),
        "matched": compared > 0 and not mismatches,
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:10],
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(prog="tmnf_rl.reproduce")
    parser.add_argument("run_id")
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--replica-id", default=None, help="id for the replica run")
    parser.add_argument(
        "--ignore-code-hash",
        action="store_true",
        help="compare metrics even though python/tmnf_rl changed (recorded in the report)",
    )
    args = parser.parse_args(argv)
    if args.updates <= 0:
        raise ValueError("updates must be positive")

    root = project_root()
    registry = RunRegistry(args.runs_root or (root / "build" / "runs"))
    original = registry.open_run(args.run_id)
    original_args = dict(original.payload["args"])
    if original.payload["status"] == "running":
        raise RuntimeError(f"run {args.run_id!r} is still running")
    code_changes = original.payload.get("code_changes") or []
    if code_changes:
        first_change = min(int(change["from_update"]) for change in code_changes)
        if args.updates > first_change:
            raise RuntimeError(
                f"run {args.run_id!r} was resumed across a python/tmnf_rl change at "
                f"update {first_change} ({code_changes}); only its first {first_change} "
                "updates were produced by its code_sha256, so at most "
                f"--updates {first_change} can be reproduced from it (F25)"
            )

    physics_library = Path(original_args["physics_library"])
    if not physics_library.is_absolute():
        physics_library = root / physics_library
    physics_now = sha256_file(physics_library)
    if physics_now != original.payload["physics_sha256"]:
        raise RuntimeError(
            "physics library differs from the original run "
            f"({original.payload['physics_sha256']} -> {physics_now}); rebuild "
            "or pin the same libtmnf_physics.so before reproducing"
        )
    # Pin the numerics the same way the trainer does, then compare the pinned
    # state and the torch/CUDA/cuDNN/GPU identity with the original's record;
    # a mismatch is named, not discovered as an unexplained metric diff (F24).
    select_device()
    current_provenance = provenance.collect(
        physics_library, seed=int(original_args["seed"]), root=root, torch_module=torch
    )
    numerics_diff = provenance.numerics_differences(original.payload["provenance"], current_provenance)
    if numerics_diff:
        names = ", ".join(
            f"{item['field']} {item['recorded']!r} -> {item['current']!r}" for item in numerics_diff
        )
        raise RuntimeError(f"CUDA numerics or stack differ from the original run: {names}")
    code_now = current_provenance["code_sha256"]
    code_matched = code_now == original.payload["code_sha256"]
    if not code_matched and not args.ignore_code_hash:
        raise RuntimeError(
            "python/tmnf_rl differs from the original run "
            f"({original.payload['code_sha256']} -> {code_now}); pass "
            "--ignore-code-hash to compare anyway (the report records it)"
        )

    original_rows = read_rows(original.run_dir / "metrics.csv", args.updates)
    if len(original_rows) < args.updates:
        raise RuntimeError(
            f"original run has only {len(original_rows)} metric rows, "
            f"cannot compare {args.updates}"
        )

    replica_id = args.replica_id or f"{args.run_id}_repro{args.updates}"
    if replica_id == args.run_id:
        raise ValueError("replica id must differ from the original run id")
    replica_config = config_from_dict(
        {
            **original_args,
            "max_updates": args.updates,
            "duration_minutes": 0.0,
            "spectate": 0,
            "run_id": replica_id,
            "runs_root": str(registry.root),
        }
    )
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(replica_config.to_dict(), handle)
        config_path = Path(handle.name)
    command = [sys.executable, "-m", "tmnf_rl.train", "--config", str(config_path)]
    print(json.dumps({"event": "reproduce_start", "command": command}), flush=True)
    completed = subprocess.run(command, cwd=root, check=False)
    config_path.unlink(missing_ok=True)
    if completed.returncode != 0:
        raise RuntimeError(f"replica trainer exited with {completed.returncode}")

    replica = registry.open_run(replica_id)
    replica_rows = read_rows(replica.run_dir / "metrics.csv", args.updates)
    comparison = compare_rows(original_rows, replica_rows)
    replica_numerics_diff = provenance.numerics_differences(
        original.payload["provenance"], replica.payload["provenance"]
    )
    if replica_numerics_diff:
        raise RuntimeError(f"replica recorded different numerics than the original: {replica_numerics_diff}")
    report = {
        "event": "reproduce",
        "original_run_id": args.run_id,
        "replica_run_id": replica_id,
        "updates": args.updates,
        "command_line": [sys.executable, *sys.argv],
        "ignore_code_hash": args.ignore_code_hash,
        "physics_sha256": physics_now,
        "code_sha256_original": original.payload["code_sha256"],
        "code_sha256_now": code_now,
        "code_matched": code_matched,
        "original_code_changes": code_changes,
        "numerics": {field: replica.payload["provenance"].get(field) for field in provenance.NUMERICS_FIELDS},
        "numerics_matched": True,
        **comparison,
    }
    write_json_atomic(replica.run_dir / "reproduce.json", report)
    replica.update(
        reproduces={
            "run_id": args.run_id,
            "matched": comparison["matched"],
            "compared_updates": comparison["compared_updates"],
            "code_matched": code_matched,
        }
    )
    registry.rebuild_index()
    print(json.dumps(report, sort_keys=True), flush=True)
    if not comparison["matched"]:
        raise SystemExit(
            f"reproduction FAILED: {comparison['mismatch_count']} mismatching cells "
            f"in the first {comparison['compared_updates']} updates"
        )
    return report


if __name__ == "__main__":
    main(sys.argv[1:])
