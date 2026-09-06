#!/usr/bin/env python3
"""Interleaved full-binding benchmarks with identical driving actions/state.

PYTHONPATH=python build/venv/bin/python tools/benchmark_env.py \
    --libraries build/before.so build/libtmnf_physics.so --output build/env-bench.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np

from tmnf_rl.env import TmnfVectorEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libraries", type=Path, nargs="+", required=True)
    parser.add_argument("--tracks", nargs="+", default=["a01", "a04", "b05", "c03"])
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--decisions", type=int, default=1000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.envs, args.threads, args.decisions, args.repeat, args.trials) < 1:
        parser.error("benchmark sizes must be positive")
    rows = []
    for track in args.tracks:
        environments = []
        try:
            for library in args.libraries:
                started = time.perf_counter()
                env = TmnfVectorEnv(args.envs, track=track, library_path=library,
                                    thread_count=args.threads, action_repeat=args.repeat)
                environments.append((library, env, time.perf_counter() - started))
            rng = np.random.default_rng(20260905)
            actions = np.empty((args.decisions, args.envs), dtype=np.int64)
            for step in range(0, args.decisions, 20):
                gas = (rng.random(args.envs) < 0.9).astype(np.int64)
                steer = rng.integers(-1, 2, args.envs)
                actions[step:step + 20] = gas * 3 + steer + 1
            for trial in range(args.trials):
                # Alternate order to avoid giving one library all the cold runs.
                order = environments if trial % 2 == 0 else environments[::-1]
                states = []
                for library, env, setup in order:
                    env.reset()
                    for action in actions[:100]:
                        env.step(action)
                    env.reset()
                    ticks = 0
                    speed_sum = 0.0
                    started = time.perf_counter()
                    for action in actions:
                        env.step(action)
                        ticks += int(env.executed_ticks.sum())
                        speed_sum += float(np.linalg.norm(env.observations["vehicle"][:, 7:10], axis=1).sum())
                    seconds = time.perf_counter() - started
                    digest = hashlib.sha256(b"".join(env.capture())).hexdigest()
                    states.append((ticks, digest))
                    row = dict(track=track, library=str(library), trial=trial,
                               setup_seconds=setup, seconds=seconds, ticks=ticks,
                               ticks_per_second=ticks / seconds,
                               mean_speed_mps=speed_sum / (args.decisions * args.envs),
                               final_state_sha256=digest)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                if len(set(states)) != 1:
                    raise RuntimeError(f"{track}: libraries produced different state/ticks")
        finally:
            for _, env, _ in environments:
                env.close()
    summaries = []
    for track in args.tracks:
        for library in args.libraries:
            selected = [r for r in rows if r["track"] == track and r["library"] == str(library)]
            rates = [r["ticks_per_second"] for r in selected]
            summaries.append(dict(track=track, library=str(library),
                                  mean_ticks_per_second=statistics.mean(rates),
                                  stdev_ticks_per_second=statistics.stdev(rates) if len(rates) > 1 else 0.0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(settings=vars(args), rows=rows, summaries=summaries),
                                     indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
