#!/usr/bin/env python3
"""Seeded public-binding stress test, isolated per track so native crashes count.

Run with PYTHONPATH=python build/venv/bin/python tools/stress_rl_env.py.
Compares serial and threaded CPU environments byte for byte, including
cross-slot snapshot restores, rewind continuation, autoresets, respawns and
invalid actions. CUDA coverage is provided separately by cuda_lockstep.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from tmnf_rl.env import TERMINATION_NAMES, TmnfVectorEnv
from tmnf_rl.tracks import project_root, track_catalogue, track_spec


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def worker(args: argparse.Namespace, track: str) -> dict:
    start = time.monotonic()
    results = []
    for mode in ("discrete", "analog"):
        rng = np.random.default_rng(args.seed)
        options = dict(track=track, action_repeat=args.repeat,
                       action_space=mode, respawn_action=True,
                       root=args.root,
                       library_path=args.library)
        serial = TmnfVectorEnv(args.envs, thread_count=1,
                              **{**options, "library_path": args.reference_library or args.library})
        parallel = TmnfVectorEnv(args.envs, thread_count=args.threads, **options)
        ticks = 0
        reasons = {name: 0 for name in TERMINATION_NAMES.values()}
        restores = rewinds = 0
        try:
            serial.reset()
            parallel.reset()
            check(serial.capture() == parallel.capture(), "spawn snapshots differ")
            actions = None
            for decision in range(args.decisions):
                # Hold correlated actions long enough to accelerate and make
                # contact, alternating with noise and exact analog endpoints.
                if decision % 12 == 0:
                    gas = (rng.random(args.envs) < 0.8).astype(np.int32)
                    brake = (rng.random(args.envs) < 0.2).astype(np.int32)
                    steer = rng.integers(-1, 2, args.envs)
                    if mode == "discrete":
                        actions = (gas + 2 * brake) * 3 + steer + 1
                    else:
                        analog = rng.uniform(-1, 1, args.envs).astype(np.float32)
                        analog[:3] = [-1, 0, 1]
                        actions = dict(steer=analog, gas=gas, brake=brake,
                                       respawn=np.zeros(args.envs, dtype=np.int32))
                if mode == "discrete":
                    actions = actions % 12
                    if decision % 113 == 112:
                        actions = actions + 12
                else:
                    actions["respawn"][:] = int(decision % 113 == 112)
                for env in (serial, parallel):
                    env.step(actions)
                    for name, array in env.observations.items():
                        check(bool(np.isfinite(array).all()), f"nonfinite {name}")
                    done = env.terminations | env.truncations
                    for name, array in env.final_observations.items():
                        check(bool(np.isfinite(array[done]).all()), f"nonfinite final {name}")
                    check(bool(np.isfinite(env.rewards).all()), "nonfinite reward")
                    check(bool(np.isfinite(env.completed_episode_returns).all()), "nonfinite return")
                    check(bool(np.all((env.transition_discounts >= 0) &
                                      (env.transition_discounts <= 1))), "invalid discount")
                    check(bool(np.all((env.executed_ticks > 0) &
                                      (env.executed_ticks <= args.repeat))), "invalid tick count")
                    check(bool(np.array_equal(done, env.termination_reasons != 0)), "reason/done mismatch")
                    finishes = env.termination_reasons == 1
                    check(bool(np.all(env.race_times_ms[finishes] ==
                                      env.completed_episode_ticks[finishes] * 10)), "finish time mismatch")
                check(bytes(serial._result_buffer) == bytes(parallel._result_buffer),
                      f"step results differ: {track}/{mode}/{decision}")
                ticks += int(serial.executed_ticks.sum())
                for reason, name in TERMINATION_NAMES.items():
                    reasons[name] += int((serial.termination_reasons == reason).sum())
                if decision % 64 == 63:
                    a, b = serial.capture(), parallel.capture()
                    check(a == b, f"snapshots differ: {track}/{mode}/{decision}")
                    indices = np.arange(args.envs, dtype=np.uint32)[::-1].copy()
                    serial.restore(indices, b)
                    parallel.restore(indices, a)
                    check(serial.capture() == parallel.capture(), "cross-slot restore differs")
                    restores += 1
                if decision % 127 == 126:
                    saved = serial.capture()
                    serial.step(actions)
                    expected_result = bytes(serial._result_buffer)
                    expected_state = serial.capture()
                    serial.restore(np.arange(args.envs), saved)
                    serial.step(actions)
                    check(bytes(serial._result_buffer) == expected_result, "rewind result differs")
                    check(serial.capture() == expected_state, "rewind continuation differs")
                    serial.restore(np.arange(args.envs), saved)
                    rewinds += 1
            before = serial.capture()
            invalid = (np.full(args.envs, 24) if mode == "discrete" else
                       {**actions, "steer": np.full(args.envs, np.nan, dtype=np.float32)})
            try:
                serial.step(invalid)
            except ValueError:
                pass
            else:
                raise RuntimeError("invalid action accepted")
            check(serial.capture() == before, "rejected action mutated state")
            results.append(dict(mode=mode, compared_ticks=ticks, terminations=reasons,
                                cross_slot_restores=restores, rewind_checks=rewinds,
                                horizon_ticks=serial.horizon_ticks))
        finally:
            serial.close()
            parallel.close()
        serial.close()
        try:
            serial.step(actions)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("closed environment accepted step")
    root = args.root
    spec = track_spec(track, root)
    return dict(track=track, passed=True, seconds=time.monotonic() - start,
                fixtures={p.name: digest(p) for p in (
                    spec.track_path(root), spec.vehicle_path(root), spec.route_path(root))},
                modes=results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", default="all", help="all or comma-separated ids")
    parser.add_argument("--root", type=Path, default=project_root(),
                        help="fixture root for isolated candidate validation")
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--decisions", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--library", type=Path, default=project_root() / "build/libtmnf_physics.so")
    parser.add_argument("--reference-library", type=Path,
                        help="compare the candidate with a previous library as well as thread counts")
    parser.add_argument("--output", type=Path, default=Path("build/rl_stress/report.json"))
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.envs < 3 or min(args.decisions, args.repeat, args.threads, args.timeout) <= 0:
        parser.error("envs must be >= 3 and other sizes/timeouts positive")
    if args.worker:
        print(json.dumps(worker(args, args.worker)), flush=True)
        return 0
    tracks = sorted(track_catalogue(args.root)) if args.tracks == "all" else args.tracks.split(",")
    for track in tracks:
        track_spec(track, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(format="tmnf-rl-stress-v1", seed=args.seed, envs=args.envs,
                  decisions=args.decisions, action_repeat=args.repeat, threads=args.threads,
                  physics_sha256=digest(args.library),
                  reference_physics_sha256=digest(args.reference_library or args.library),
                  harness_sha256=digest(Path(__file__)),
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  cases=[], passed=False)
    start = time.monotonic()
    for track in tracks:
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", track,
                   "--envs", str(args.envs), "--decisions", str(args.decisions),
                   "--repeat", str(args.repeat), "--threads", str(args.threads),
                   "--root", str(args.root.resolve()),
                   "--seed", str(args.seed), "--library", str(args.library.resolve())]
        if args.reference_library:
            command += ["--reference-library", str(args.reference_library.resolve())]
        try:
            run = subprocess.run(command, text=True, capture_output=True, timeout=args.timeout)
            log = args.output.parent / f"{track}.log"
            log.write_text(run.stdout + run.stderr)
            case = (json.loads(run.stdout.splitlines()[-1]) if run.returncode == 0 else
                    dict(track=track, passed=False, returncode=run.returncode, log=str(log)))
        except subprocess.TimeoutExpired:
            case = dict(track=track, passed=False, error="timeout")
        report["cases"].append(case)
        report["seconds"] = time.monotonic() - start
        report["passed"] = len(report["cases"]) == len(tracks) and all(c["passed"] for c in report["cases"])
        report["compared_ticks"] = sum(m["compared_ticks"] for c in report["cases"] for m in c.get("modes", []))
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)
        print(f"{len(report['cases'])}/{len(tracks)} {track}: {'PASS' if case['passed'] else 'FAIL'}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
