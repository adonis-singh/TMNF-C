#!/usr/bin/env python3
"""Capture one targeted function trace while driving a registered input file."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import onboard_track as onboarding  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("track_name")
    parser.add_argument("--target", required=True, help="function VA, e.g. 0x007BD700")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-start", type=int, default=0,
                        help="skip this many calls before recording")
    parser.add_argument("--mode", choices=("trace", "detect"), default="trace",
                        help="tracer build: trace (vehicle targets) or detect "
                             "(collision detection targets)")
    parser.add_argument("--wineprefix", type=Path, default=onboarding.PREFIX)
    parser.add_argument("--port", type=int, default=onboarding.PORT)
    parser.add_argument("--display", default=onboarding.XVFB_DISPLAY)
    parser.add_argument("--cpu-set", default=onboarding.CPU_SET)
    args = parser.parse_args()
    onboarding.set_lane(args.wineprefix, args.port, args.display, args.cpu_set)

    target = int(args.target, 0)
    if target == 0x004FE500:
        onboarding.fail("0x004FE500 must never be hooked during capture")
    if not args.inputs.is_file():
        onboarding.fail(f"missing inputs {args.inputs}")
    if args.output.exists():
        onboarding.fail(f"refusing to overwrite {args.output}")

    track = onboarding.parse_track(args.track_name)
    args.track_name = track.name
    challenge = onboarding.challenge_for(track)
    official_maps = onboarding.layout().official_maps
    official_maps.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(challenge, official_maps / challenge.name)

    staging = onboarding.ROOT / f"build/function-trace/{args.track_name}/{target:08X}"
    if staging.exists():
        shutil.rmtree(staging)
    traces = staging / "traces"
    traces.mkdir(parents=True)
    capture = staging / "capture.bin"

    onboarding.install_tracer(args.mode)
    try:
        onboarding.run_game(
            args.track_name,
            [onboarding.PYTHON, onboarding.ROOT / "oracle/capture_policy_lap.py",
             "--port", str(onboarding.PORT), "--map", args.track_name,
             "--inputs", args.inputs, "--output", capture],
            staging / "capture.log",
            {
                "TMNF_TRACE_HOST_DIR": str(traces),
                "TMNF_TRACE_TARGETS": f"0x{target:08X}",
                "TMNF_TRACE_RECORD_START": str(args.record_start),
            },
        )
    finally:
        onboarding.disable_tracer()

    produced = [
        path for path in traces.glob("*.bin")
        if path.name.startswith(f"{target:08X}_")
    ]
    if len(produced) != 1:
        onboarding.fail(f"expected one 0x{target:08X} trace, found {len(produced)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(produced[0], args.output)
    print(f"captured {produced[0].name} -> {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, onboarding.subprocess.CalledProcessError,
            onboarding.subprocess.TimeoutExpired) as error:
        print(f"capture_function_trace: {error}", file=sys.stderr)
        raise SystemExit(1)
