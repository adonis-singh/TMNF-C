#!/usr/bin/env python3
"""Export one deterministic policy lap as exact per-tick TMNFRaceInputs.

Thin wrapper over ``python -m tmnf_rl.evaluate --envs 1 --episodes 1
--write-inputs OUT --inputs-ticks N``. The default output is the file replayed
by the ``policy_lap_replay`` test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tmnf_rl.evaluate import main as evaluate_main  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, help="run id or policy.pt")
    parser.add_argument(
        "--output", type=Path, default=root / "oracle" / "results" / "policy_lap_inputs.bin"
    )
    parser.add_argument("--ticks", type=int, default=2700)
    args = parser.parse_args()
    report = evaluate_main(
        [
            args.policy,
            "--envs", "1",
            "--episodes", "1",
            "--write-inputs", str(args.output),
            "--inputs-ticks", str(args.ticks),
        ]
    )
    if report["best_lap"] is None:
        raise SystemExit("policy did not finish; no schedule written")


if __name__ == "__main__":
    main()
