#!/usr/bin/env python3
"""Fixer-lane copy of tools/capture_function_trace.py bound to wineprefix_fixer.

usage: capture_fixer.py TRACK_NAME --inputs FILE --output FILE [--target VA ...]
       capture_fixer.py TRACK_NAME --inputs FILE --output FILE --no-tracer
Writes the tick capture to --output; targeted traces land in
/home/adityas/fixer-tools/traces/<stem>/.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path("/home/adityas/Projects/TMNF-C")
TOOLS = Path("/home/adityas/fixer-tools")
PREFIX = ROOT / "oracle/wineprefix_fixer"
GAME_DIR = PREFIX / "drive_c/TmNationsForever"
CAMPAIGN_DIR = GAME_DIR / "GameData/Tracks/Campaigns/Nations"
DOCUMENT_TRACKS = (
    PREFIX / "drive_c/users/adityas/TMNFDocuments/TmForever/Tracks/Challenges/Official Maps"
)
TRACER_PLUGIN = GAME_DIR / "TMNFTracer.asi"
PYTHON = ROOT / "third_party/venv/bin/python"
CPU_SET = "9,25"
PORT = 8489


def fail(message: str) -> None:
    raise RuntimeError(message)


def taskset(command):
    return ["taskset", "-c", CPU_SET, *[str(c) for c in command]]


def env() -> dict[str, str]:
    e = os.environ.copy()
    e["CUDA_VISIBLE_DEVICES"] = ""
    e["WINEPREFIX"] = str(PREFIX)
    e["WINEDEBUG"] = "-all"
    e["PYTHONPATH"] = str(ROOT / "oracle")
    return e


def build_tracer(mode: str) -> Path:
    out = TOOLS / "tracer-build" / mode
    out.mkdir(parents=True, exist_ok=True)
    tr = Path(os.environ.get("FIXER_TRACER_SRC", str(ROOT / "oracle/tracer")))
    if mode in ("trace", "detect"):
        targets = tr / ("targets.c" if mode == "trace" else "detect_targets.c")
        sources = [tr / "tracer.c", targets, tr / "broadphase_capture.c",
                   tr / "detect_capture.c", tr / "response_capture.c",
                   tr / "model6_capture.c"]
    elif mode == "track":
        sources = [tr / "track_dump.c"]
    else:
        sources = [tr / "route_dump.c"]
    asi = out / "TMNFTracer.asi"
    subprocess.run(
        ["i686-w64-mingw32-gcc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
         "-shared", "-static-libgcc", "-Wl,--kill-at", *map(str, sources), "-o", str(asi)],
        check=True)
    return asi


def wait_for_port(process: subprocess.Popen, port: int) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            fail(f"game launcher exited with code {process.returncode}")
        result = subprocess.run(["ss", "-ltnH"], check=True, stdout=subprocess.PIPE, text=True)
        if any(line.split()[3].endswith(f":{port}")
               for line in result.stdout.splitlines() if len(line.split()) >= 4):
            time.sleep(1)
            return
        time.sleep(1)
    fail(f"game did not listen on port {port}")


def stop_game(process: subprocess.Popen, environment: dict[str, str]) -> None:
    subprocess.run(["wineserver", "-k"], env=environment, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def run_game(track_name: str, client: list, log_path: Path, extra: dict[str, str] | None,
             post: str = "") -> None:
    environment = env()
    if extra:
        environment.update(extra)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        launcher = subprocess.Popen(
            taskset([TOOLS / "launch_game_fixer.sh", str(PORT), track_name]),
            cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        try:
            wait_for_port(launcher, PORT)
            subprocess.run(taskset(client), cwd=ROOT, env=environment, check=True,
                           timeout=900, stdout=log, stderr=subprocess.STDOUT)
            if post:
                subprocess.run(post, shell=True, cwd=ROOT, env=environment, check=True,
                               timeout=900)
        finally:
            stop_game(launcher, environment)


def challenge_for(track_name: str) -> Path:
    matches = list(CAMPAIGN_DIR.glob(f"*/{track_name}.Challenge.Gbx"))
    if len(matches) != 1:
        fail(f"found {len(matches)} campaign challenges named {track_name}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("track_name")
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", default="trace")
    parser.add_argument("--tag", default="")
    parser.add_argument("--post", default="", help="shell command run after capture while the game is up")
    parser.add_argument("--probe", action="store_true", help="run determinism_test --ticks 10 instead of a capture")
    args = parser.parse_args()

    targets = [int(t, 0) for t in args.target]
    if 0x004FE500 in targets:
        fail("0x004FE500 must never be hooked during capture")
    if not args.probe and not args.inputs.is_file():
        fail(f"missing inputs {args.inputs}")
    if not args.probe and args.output.exists():
        fail(f"refusing to overwrite {args.output}")

    challenge = challenge_for(args.track_name)
    DOCUMENT_TRACKS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(challenge, DOCUMENT_TRACKS / challenge.name)

    stem = args.output.stem + (f"_{args.tag}" if args.tag else "")
    traces = TOOLS / "traces" / stem
    if traces.exists():
        shutil.rmtree(traces)
    traces.mkdir(parents=True)

    extra = {"TMNF_TRACE_HOST_DIR": str(traces)}
    if targets or args.mode != "trace":
        asi = build_tracer(args.mode)
        shutil.copyfile(asi, TRACER_PLUGIN)
        if targets:
            extra["TMNF_TRACE_TARGETS"] = ",".join(f"0x{t:08X}" for t in targets)
    else:
        TRACER_PLUGIN.unlink(missing_ok=True)
    if args.probe:
        client = [PYTHON, ROOT / "oracle/determinism_test.py",
                  "--port", str(PORT), "--map", args.track_name, "--ticks", "10",
                  "--output-dir", traces / "probe"]
    else:
        client = [PYTHON, ROOT / "oracle/capture_policy_lap.py",
                  "--port", str(PORT), "--map", args.track_name,
                  "--inputs", args.inputs, "--output", args.output]
    try:
        run_game(args.track_name, client, traces / "capture.log", extra, args.post)
    finally:
        TRACER_PLUGIN.unlink(missing_ok=True)
    for p in sorted(traces.glob("*.bin")):
        print(f"trace {p} {p.stat().st_size} bytes")
    if not args.probe:
        print(f"capture {args.output} {args.output.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f"capture_fixer: {error}", file=sys.stderr)
        raise SystemExit(1)
