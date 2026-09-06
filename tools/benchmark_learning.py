#!/usr/bin/env python3
"""Measure exact encoder/GAE launch optimizations on one selected GPU.

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python build/venv/bin/python \
    tools/benchmark_learning.py --output build/learning-bench.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

from tmnf_rl.advantages import AdvantageEstimator, generalized_advantage
from tmnf_rl.agents.ppo import pin_cuda_numerics
from tmnf_rl.encoder import FlatEncoder, encode_flat
from tmnf_rl.utils import require_single_gpu_env


def measure(function, iterations: int) -> list[float]:
    for _ in range(3):
        function()
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            function()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - started) / iterations)
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    require_single_gpu_env()
    pin_cuda_numerics()
    torch.manual_seed(20260905)
    rows = []
    for count in [32, 256, 4096]:
        obs = torch.randn((count, 81), device="cuda")
        obs[:, 3:7] = torch.nn.functional.normalize(obs[:, 3:7], dim=1)
        graph = FlatEncoder()
        assert torch.equal(encode_flat(obs), graph(obs))
        eager_times = measure(lambda: encode_flat(obs), args.iterations)
        graph_times = measure(lambda: graph(obs), args.iterations)
        rows.append(dict(stage="encoder", batch=count, eager_seconds=eager_times,
                         graph_seconds=graph_times, bit_exact=True))
    shape = (128, 256)
    buffers = [torch.randn(shape, device="cuda"), torch.rand(shape, device="cuda"),
               torch.rand(shape, device="cuda") < 0.01, torch.rand(shape, device="cuda") < 0.01,
               torch.randn(shape, device="cuda"), torch.randn(shape, device="cuda"),
               torch.randn(256, device="cuda")]
    graph = AdvantageEstimator()
    eager = lambda: generalized_advantage(*buffers, 0.95)
    replay = lambda: graph(*buffers, gae_lambda=0.95)
    assert all(torch.equal(a, b) for a, b in zip(eager(), replay()))
    rows.append(dict(stage="gae", batch=256, steps=128,
                     eager_seconds=measure(eager, max(1, args.iterations // 10)),
                     graph_seconds=measure(replay, max(1, args.iterations // 10)), bit_exact=True))
    for row in rows:
        row["median_speedup"] = statistics.median(row["eager_seconds"]) / statistics.median(row["graph_seconds"])
    report = dict(gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                  cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"], rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
