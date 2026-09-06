#!/usr/bin/env python3
"""Verify the retained release evidence without a game installation.

Checks every recorded asset hash, replays the captured game states and race
finishes, then evaluates each retained policy from the official spawn.
Use --replays-only for a CPU-only verification without PyTorch/CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/release/manifest.json")
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    parser.add_argument("--output", type=Path, default=ROOT / "build/release_verify")
    parser.add_argument("--replays-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.build = args.build.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(format="tmnf-release-verification-v1", passed=False,
                  physics_sha256=hashlib.sha256((args.build / "libtmnf_physics.so").read_bytes()).hexdigest(),
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  replays=[], policies=[], policy_evaluation_skipped=args.replays_only)

    def run(label: str, command: list[str]) -> str:
        environment = {**os.environ, "PYTHONPATH": str(ROOT / "python")}
        result = subprocess.run(command, cwd=ROOT, env=environment,
                                capture_output=True, text=True, timeout=300)
        (args.output / f"{label}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{label} failed; see {args.output / (label + '.log')}")
        return result.stdout

    try:
        for relative, expected in manifest["files"].items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"asset hash mismatch: {relative}")
        for lap in manifest["laps"]:
            if lap["lap_ms"] > lap["author_ms"]:
                raise RuntimeError(f"{lap['id']} exceeds author time")
            track, vehicle, route, inputs, capture = [str(ROOT / lap[k]) for k in
                                                    ("track", "vehicle", "route", "inputs", "capture")]
            name = lap["id"]
            output = run(name + "_ticks", [str(args.build / "tests/replay_tick"),
                         "--route", route, track, vehicle, capture, "input_file", inputs,
                         lap["challenge_sha256"]])
            matched = re.search(r"full tick: (\d+)/(\d+) ticks byte-exact", output)
            if not matched or int(matched[1]) != lap["capture_ticks"] or int(matched[2]) != lap["capture_ticks"]:
                raise RuntimeError(f"{name}: incomplete game replay")
            run(name + "_finish", [str(args.build / "tests/wr_passthrough"), track,
                vehicle, route, inputs, lap["challenge_sha256"], "--expect-finish-ms", str(lap["lap_ms"])])
            run(name + "_env", [str(args.build / "tests/vec_env_matches_replay"), track,
                vehicle, route, inputs, lap["challenge_sha256"], str(lap["lap_ms"])])
            report["replays"].append(dict(id=name, lap_ms=lap["lap_ms"], ticks=int(matched[1]), passed=True))
            print(f"{name}: game capture, finish and env continuation verified", flush=True)
        if not args.replays_only:
            for policy in manifest["policies"]:
                identifier = policy.get("id", policy["track_id"])
                output = args.output / f"{identifier}_evaluation.json"
                run(identifier + "_policy", [sys.executable, "-m", "tmnf_rl.evaluate",
                    str(ROOT / policy["path"]), "--physics-library", str(args.build / "libtmnf_physics.so"),
                    "--envs", str(policy.get("evaluation_envs", 32)), "--episodes", "256", "--sample-seed", "0",
                    "--export-scene", str(args.output / f"{identifier}.json"), "--output", str(output)])
                evaluated = json.loads(output.read_text())
                lap = evaluated["best_lap"]
                if (not lap or lap["lap_ms"] != policy["lap_ms"] or not lap["matches_world"]
                        or lap["lap_ms"] > policy["author_ms"]):
                    raise RuntimeError(f"{policy['track_id']}: policy lap changed or exceeds author")
                if (policy.get("schedule_sha256") is not None
                        and lap["schedule_sha256"] != policy["schedule_sha256"]):
                    raise RuntimeError(f"{policy['track_id']}: greedy input schedule changed")
                if evaluated["sampled"]["finish_rate"] < policy["minimum_sampled_finish_rate"]:
                    raise RuntimeError(f"{policy['track_id']}: sampled finish rate regressed")
                report["policies"].append(dict(id=identifier, track=policy["track_id"], passed=True,
                    lap_ms=lap["lap_ms"], sampled_finishes=evaluated["sampled"]["finishes"],
                    sampled_episodes=evaluated["sampled"]["episodes"],
                    physics_matches_training=evaluated["physics_matches_training"]))
                print(f"{policy['track_id']}: retained policy verified at {lap['lap_ms']} ms", flush=True)
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        print(str(error), file=sys.stderr)
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
