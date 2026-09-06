#!/usr/bin/env python3
"""Interleave native CUDA environment libraries with resident action buffers.

Times native environment steps (physics, race rules, observations), excluding
the learner and per-step host downloads. Set CUDA_VISIBLE_DEVICES to one GPU.
The reported tick count is requested ticks; terminal steps can end earlier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from tmnf_rl.cuda_env import TmnfCudaVectorEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libraries", type=Path, nargs="+", required=True)
    parser.add_argument("--tracks", nargs="+", default=["a04", "rally-a1", "snow-a1", "bay-a1"])
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--decisions", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.envs, args.decisions, args.repeat, args.trials) < 1:
        parser.error("sizes must be positive")
    if torch.cuda.device_count() != 1:
        parser.error("set CUDA_VISIBLE_DEVICES to exactly one GPU")
    device = torch.device("cuda:0")
    libraries = [p.resolve() for p in args.libraries]
    report = dict(format="tmnf-cuda-environment-benchmark-v1", passed=False,
        gpu=torch.cuda.get_device_name(device), visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        torch=torch.__version__, settings=vars(args),
        libraries={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in libraries}, rows=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260905)
    actions = np.empty((args.decisions, args.envs), dtype=np.uint8)
    for i in range(0, args.decisions, 12):
        gas = (rng.random(args.envs) < 0.8).astype(np.uint8)
        brake = (rng.random(args.envs) < 0.1).astype(np.uint8)
        steer = rng.integers(0, 3, args.envs, dtype=np.uint8)
        actions[i:i + 12] = (gas + 2 * brake) * 3 + steer
    resident = torch.from_numpy(actions).to(device)
    pointers = [resident[i].data_ptr() for i in range(args.decisions)]
    for track in args.tracks:
        states = set()
        for trial in range(args.trials):
            for library in libraries if trial % 2 == 0 else libraries[::-1]:
                env = TmnfCudaVectorEnv(args.envs, track=track, action_repeat=args.repeat,
                    action_space="discrete", max_race_ticks=12000, horizon_ticks=12000,
                    library_path=library, device=device)
                try:
                    step = env._lib.tmnf_cuda_env_step_discrete
                    for pointer in pointers[:min(40, len(pointers))]:
                        step(env._handle, pointer, args.repeat)
                    torch.cuda.synchronize()
                    env.reset()
                    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    started = time.perf_counter()
                    start.record()
                    for pointer in pointers:
                        step(env._handle, pointer, args.repeat)
                    stop.record()
                    stop.synchronize()
                    wall = time.perf_counter() - started
                    digest = hashlib.sha256(b"".join(env.capture())).hexdigest()
                    states.add(digest)
                    row = dict(track=track, library=str(library), trial=trial,
                        wall_seconds=wall, gpu_seconds=start.elapsed_time(stop) / 1000,
                        requested_ticks=args.envs * args.decisions * args.repeat,
                        final_state_sha256=digest)
                    report["rows"].append(row)
                    print(json.dumps(row), flush=True)
                    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
                finally:
                    env.close()
        if len(states) != 1:
            raise RuntimeError(f"{track}: final snapshots differ: {states}")
    report["summaries"] = [dict(track=track, library=str(library),
        median_gpu_seconds=statistics.median(r["gpu_seconds"] for r in report["rows"]
            if r["track"] == track and r["library"] == str(library)))
        for track in args.tracks for library in libraries]
    report["passed"] = True
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
