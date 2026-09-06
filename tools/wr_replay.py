#!/usr/bin/env python3
"""World-record replay pass-through pipeline.

Subcommands:

  extract REPLAY --script OUT.txt --meta OUT.json
      Read the ghost from a TMNF .Replay.Gbx and write its inputs as a
      TMInterface input script (the gbxtools generate_input_file.py
      conversion) plus ghost metadata.

  schedule --script S.txt --ticks N --output OUT.bin
      Convert a TMInterface input script to per-tick 72-byte TMNFRaceInputs
      records using the game's own source-priority rule.

  verify --schedule S.bin --capture C.bin
      Compare the schedule's resolved controls against the gas/brake/steer
      words TMInterface captured. Exit 1 on any mismatch.

  run TRACK_NAME REPLAY --tag NAME
      Full pipeline: extract, play the script in the game under Xvfb via
      TMInterface `load`, verify accepted inputs, normalize the capture, and
      replay it natively. Writes oracle/results/wr/<track_id>_<tag>.bin and
      the matching *_inputs.bin plus *_events.json.

Tick i of a schedule corresponds to TMInterface race time i*10 ms. An event at
time t applies from tick t/10 on. These conventions were measured against the
A01 record capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PREFIX = Path(os.environ.get("TMNF_WINEPREFIX", ROOT / "oracle/wineprefix"))
CAMPAIGN_DIR = PREFIX / "drive_c/TmNationsForever/GameData/Tracks/Campaigns/Nations"
PYTHON = ROOT / "third_party/venv/bin/python"
REPLAY_TICK = ROOT / "build/tests/replay_tick"
CPU_SET = os.environ.get("TMNF_CPU_SET", "14,30")
PORT = int(os.environ.get("TMNF_PORT", "8488"))
XVFB_DISPLAY = os.environ.get("TMNF_XVFB_DISPLAY", "98")
TICK_MS = 10
RECORD_SIZE = 1668
SCENE_OFFSET = 176
INPUT_STRUCT = struct.Struct("<IIiIIiIIfIIiIIiIIf")
RESULTS_DIR = ROOT / "oracle/results/wr"


def fail(message: str) -> None:
    raise RuntimeError(message)


# --- extraction (gbxtools generate_input_file.py logic) ---------------------

def _event_time(event) -> int:
    if event.event_name == "Respawn":
        t = int(event.time / 10) * 10
        if event.time % 10 == 0:
            t -= 10
        return t
    return int(event.time / 10) * 10 - 10


def _find_event_end(entries, target, from_index):
    for i in range(from_index, len(entries)):
        if entries[i].event_name == target.event_name:
            return entries[i]
    return None


def _should_skip(event) -> bool:
    if event.event_name in ("AccelerateReal", "BrakeReal"):
        return event.flags != 1
    if event.event_name == "Steer":
        return False
    if event.event_name.startswith("_Fake"):
        return True
    return event.enabled == 0


def _analog_value(event) -> int:
    value = ((event.flags << 16) | event.enabled) & 0xFFFFFFFF
    value = (value << 8) & 0xFFFFFFFF
    if value & 0x80000000:
        value -= 1 << 32
    value >>= 8
    return -value


def extract(replay: Path) -> tuple[str, dict[str, object]]:
    from pygbx import Gbx, GbxType  # type: ignore[import-not-found]

    g = Gbx(str(replay))
    ghosts = g.get_classes_by_ids([GbxType.CTN_GHOST, GbxType.CTN_GHOST_OLD])
    if len(ghosts) != 1:
        fail(f"{replay} contains {len(ghosts)} ghosts; expected exactly one")
    ghost = ghosts[0]
    entries = ghost.control_entries
    is_iface = any(
        e.time % 10 == 5 and e.event_name == "_FakeIsRaceRunning" for e in entries
    )
    invert_axis = any(e.event_name == "_FakeDontInverseAxis" for e in entries)
    if is_iface:
        for e in entries:
            e.time -= 0xFFFF

    lines: list[str] = []
    for i, event in enumerate(entries):
        if _should_skip(event):
            continue
        is_unbound = False
        to_event = _find_event_end(entries, event, i + 1)
        if to_event is not None:
            to = _event_time(to_event)
        else:
            to = ghost.race_time
            if to == 4294967295:
                to = -1
                is_unbound = True
        start = _event_time(event)
        if start < 0:
            if to < 0 and not is_unbound:
                continue
            start = 0
        start = int(start / 10) * 10
        to = int(to / 10) * 10
        name = event.event_name
        if name in ("Accelerate", "AccelerateReal"):
            key = "up"
        elif name == "SteerLeft":
            key = "left"
        elif name == "SteerRight":
            key = "right"
        elif name in ("Brake", "BrakeReal"):
            key = "down"
        elif name == "Respawn":
            key = "enter"
        elif name == "Steer":
            axis = _analog_value(event)
            if invert_axis:
                axis = -axis
            lines.append(f"{start} steer {axis}")
            continue
        elif name == "Gas":
            axis = _analog_value(event)
            if invert_axis:
                axis = -axis
            lines.append(f"{start} gas {axis}")
            continue
        elif name == "Horn":
            continue
        else:
            fail(f"unhandled ghost event {name}")
        if is_unbound:
            lines.append(f"{start} press {key}")
        else:
            lines.append(f"{start}-{to} press {key}")

    meta = {
        "replay": str(replay),
        "replay_sha256": hashlib.sha256(replay.read_bytes()).hexdigest(),
        "race_time_ms": int(ghost.race_time),
        "cp_times_ms": [int(t) for t in ghost.cp_times],
        "game_version": getattr(ghost, "game_version", None),
        "login": ghost.login,
        "event_counts": {
            name: sum(1 for e in entries if e.event_name == name)
            for name in sorted({e.event_name for e in entries})
        },
    }
    return "\n".join(lines) + "\n", meta


# --- schedule ---------------------------------------------------------------

def parse_script(text: str):
    """Return (digital_events, analog_events, respawn_ticks).

    digital_events: list of (tick, key, pressed)
    analog_events: list of (tick, value)
    respawn_ticks: ticks on which `press enter` starts
    """
    digital: list[tuple[int, str, bool]] = []
    analog: list[tuple[int, int]] = []
    respawn: set[int] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3:
            fail(f"unparsable script line: {raw}")
        span, action, argument = parts
        if action == "press":
            if argument not in ("up", "down", "left", "right", "enter"):
                fail(f"unsupported key in script: {raw}")
            if "-" in span:
                a, b = span.split("-", 1)
                start, end = int(a), int(b)
            else:
                start, end = int(span), None
            if start % TICK_MS != 0 or (end is not None and end % TICK_MS != 0):
                fail(f"script time is not tick aligned: {raw}")
            if argument == "enter":
                # The race layer acts on the press edge only
                # (CTrackManiaRace::OnInputEvent -> SmallRespawn).
                respawn.add(start // TICK_MS)
                continue
            digital.append((start // TICK_MS, argument, True))
            if end is not None:
                digital.append((end // TICK_MS, argument, False))
        elif action == "steer":
            t = int(span)
            if t % TICK_MS != 0:
                fail(f"script time is not tick aligned: {raw}")
            value = int(argument)
            if value < -65536 or value > 65536:
                fail(f"analog steer out of range: {raw}")
            analog.append((t // TICK_MS, value))
        else:
            fail(f"unsupported script action (analog gas?): {raw}")
    digital.sort(key=lambda e: e[0])
    analog.sort(key=lambda e: e[0])
    return digital, analog, respawn


def build_schedule(text: str, ticks: int, freeze_tick: int | None = None) -> bytes:
    """Per-tick records. From freeze_tick on, controls hold their previous
    values: the game stops applying input events once the race is finished,
    so a capture that runs past the finish keeps the finish-tick controls."""
    digital, analog, respawn = parse_script(text)
    di = ai = 0
    left = right = up = down = False
    steer_digital_tick = -1
    steer_analog_tick = -1
    analog_value = 0
    out = bytearray()
    for tick in range(ticks):
        if freeze_tick is not None and tick >= freeze_tick:
            previous = out[-INPUT_STRUCT.size:]
            values = list(INPUT_STRUCT.unpack(previous))
            timestamp = (tick + 1) * TICK_MS
            for index in (0, 3, 6, 9, 12):
                if values[index] != 0:
                    values[index] = timestamp
            values[16] = 0
            out.extend(INPUT_STRUCT.pack(*values))
            continue
        while di < len(digital) and digital[di][0] <= tick:
            _, key, pressed = digital[di]
            if key == "up":
                up = pressed
            elif key == "down":
                down = pressed
            elif key == "left":
                left = pressed
                steer_digital_tick = tick
            else:
                right = pressed
                steer_digital_tick = tick
            di += 1
        while ai < len(analog) and analog[ai][0] <= tick:
            analog_value = analog[ai][1]
            steer_analog_tick = tick
            ai += 1
        timestamp = (tick + 1) * TICK_MS
        # Word 16 (TMNFRaceInputs.respawn, +0x40) is never read by the
        # 0x004FE500 control mapper; the race layer consumes it.
        respawn_flag = int(tick in respawn)
        # Game rule (0x004FE500): the newer source wins; on an equal
        # timestamp digital wins unless both digital keys are released and
        # the analog magnitude exceeds 0.01.
        use_analog = steer_analog_tick > steer_digital_tick or (
            steer_analog_tick == steer_digital_tick
            and steer_analog_tick >= 0
            and not left and not right
            and abs(analog_value) / 65536.0 > 0.01
        )
        if use_analog:
            # TMInterface writes (float)(-v) / 65536: v == 0 gives +0.0 and the
            # game's negation yields the -0.0 seen in captures.
            record = INPUT_STRUCT.pack(
                0, 0, 0,
                0, 0, 0,
                timestamp, 0, float(-analog_value) / 65536.0,
                timestamp, 0, int(up),
                timestamp, 0, int(down),
                0, respawn_flag, 0.0,
            )
        else:
            record = INPUT_STRUCT.pack(
                timestamp, 0, int(left),
                timestamp, 0, int(right),
                0, 0, 0.0,
                timestamp, 0, int(up),
                timestamp, 0, int(down),
                0, respawn_flag, 0.0,
            )
        out.extend(record)
    return bytes(out)


def resolve_record(values: tuple) -> tuple[float, float, float]:
    """Mirror 0x004FE500 for a canonical schedule record."""
    left_t, _, left, right_t, _, right, an_t, _, an = values[:9]
    up_t, _, up, down_t, _, down, _, _, _ = values[9:]
    gas = 1.0 if up else 0.0
    brake = 1.0 if down else 0.0
    latest = max(left_t, right_t)
    event = an_t
    if event == latest and not left and not right and abs(an) > 0.01:
        event = latest + 1
    if event <= latest:
        steer = -1.0 if left else (1.0 if right else 0.0)
    else:
        steer = -float(struct.unpack("<f", struct.pack("<f", an))[0])
    return gas, brake, steer


def bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def verify(schedule: bytes, capture: bytes) -> list[tuple[int, tuple, tuple]]:
    if len(schedule) % INPUT_STRUCT.size != 0:
        fail("schedule has a partial record")
    ticks = len(schedule) // INPUT_STRUCT.size
    if len(capture) != ticks * RECORD_SIZE:
        fail(
            f"capture has {len(capture) // RECORD_SIZE} ticks, "
            f"schedule has {ticks}"
        )
    mismatches = []
    for tick, values in enumerate(INPUT_STRUCT.iter_unpack(schedule)):
        expected = tuple(
            bits(v) for v in resolve_record(values)
        )
        actual = struct.unpack_from(
            "<III", capture, tick * RECORD_SIZE + SCENE_OFFSET
        )
        if expected != actual:
            mismatches.append((tick, expected, actual))
    return mismatches


# --- game orchestration -----------------------------------------------------

def set_lane(prefix: Path, port: int, display: str, cpu_set: str) -> None:
    global PREFIX, CAMPAIGN_DIR, PORT, XVFB_DISPLAY, CPU_SET
    PREFIX = Path(prefix).resolve()
    if not (PREFIX / "drive_c/TmNationsForever/TmForever.exe").is_file():
        fail(f"missing Wine prefix with TmForever.exe: {PREFIX}")
    CAMPAIGN_DIR = PREFIX / "drive_c/TmNationsForever/GameData/Tracks/Campaigns/Nations"
    PORT = int(port)
    XVFB_DISPLAY = str(display)
    CPU_SET = cpu_set


def clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["WINEPREFIX"] = str(PREFIX)
    environment["TMNF_WINEPREFIX"] = str(PREFIX)
    environment["TMNF_XVFB_DISPLAY"] = XVFB_DISPLAY
    environment["WINEDEBUG"] = "-all"
    environment["PYTHONPATH"] = str(ROOT / "oracle")
    return environment


def taskset(command: list) -> list[str]:
    return ["taskset", "-c", CPU_SET, *[str(c) for c in command]]


def run(command: list, *, env=None, timeout=None, stdout=None) -> None:
    print("+ " + " ".join(str(c) for c in command), flush=True)
    subprocess.run(
        [str(c) for c in command], cwd=ROOT, env=env, check=True,
        timeout=timeout, stdout=stdout,
        stderr=subprocess.STDOUT if stdout is not None else None,
    )


def wait_for_port(process: subprocess.Popen, port: int) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            fail(f"game launcher exited with code {process.returncode}")
        result = subprocess.run(
            ["ss", "-ltnH"], check=True, stdout=subprocess.PIPE, text=True
        )
        if any(
            line.split()[3].endswith(f":{port}")
            for line in result.stdout.splitlines()
            if len(line.split()) >= 4
        ):
            time.sleep(2)
            return
        time.sleep(1)
    fail(f"game did not listen on port {port}")


def stop_game(process: subprocess.Popen, environment: dict[str, str]) -> None:
    subprocess.run(
        ["wineserver", "-k"], cwd=ROOT, env=environment, check=False,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def run_game(track_name: str, client: list, log_path: Path) -> None:
    environment = clean_environment()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        launcher = subprocess.Popen(
            taskset([ROOT / "oracle/launch_game.sh", str(PORT), track_name]),
            cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for_port(launcher, PORT)
            run(taskset(client), env=environment, timeout=900, stdout=log)
        finally:
            stop_game(launcher, environment)
    # Xvfb from launch_game.sh exits with wineserver; make sure.
    subprocess.run(["pkill", "-f", f"Xvfb :{XVFB_DISPLAY}"], check=False)


def challenge_for(track_name: str) -> Path:
    matches = list(CAMPAIGN_DIR.glob(f"*/{track_name}.Challenge.Gbx"))
    if len(matches) != 1:
        fail(f"found {len(matches)} campaign challenges named {track_name}")
    return matches[0]


def track_id_for(track_name: str) -> str:
    return track_name.split("-", 1)[0].lower()


def command_run(args: argparse.Namespace) -> int:
    track_name: str = args.track_name
    track_id = track_id_for(track_name)
    tag: str = args.tag
    staging = ROOT / f"build/wr/{track_id}_{tag}"
    staging.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    script_text, meta = extract(args.replay)
    script_path = staging / f"{track_id}_{tag}.txt"
    script_path.write_text(script_text)
    (staging / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))
    ghost_time = meta["race_time_ms"]
    ticks = ghost_time // TICK_MS + args.after_finish_ticks + 1

    raw_capture = staging / "capture_raw.bin"
    events_path = RESULTS_DIR / f"{track_id}_{tag}_events.json"
    if args.reuse_capture:
        if not raw_capture.is_file() or not events_path.is_file():
            fail(f"no capture to reuse under {staging}")
        print(f"reusing {raw_capture}")
    else:
        run_game(
            track_name,
            [PYTHON, ROOT / "oracle/capture_replay_run.py",
             "--port", str(PORT), "--map", track_name,
             "--script", script_path, "--ticks", str(ticks),
             "--after-finish-ticks", str(args.after_finish_ticks),
             "--output", raw_capture, "--events", events_path],
            staging / "game.log",
        )
    events = json.loads(events_path.read_text())
    game_finish = events["finish_time_ms"]
    captured_ticks = events["ticks"]
    print(f"ghost time {ghost_time} ms, game finish {game_finish}")
    reproduced = game_finish == ghost_time
    if not reproduced:
        print(
            f"REPLAY DOES NOT REPRODUCE: ghost {ghost_time} ms, "
            f"game {game_finish}", flush=True,
        )
        events["reproduced"] = False
        events["ghost"] = meta
        events_path.write_text(json.dumps(events, indent=1) + "\n")
        return 3

    schedule = build_schedule(
        script_text, captured_ticks, freeze_tick=ghost_time // TICK_MS)
    capture = raw_capture.read_bytes()
    mismatches = verify(schedule, capture)
    print(f"accepted-input mismatches: {len(mismatches)}")
    for tick, expected, actual in mismatches[:20]:
        print(f"  tick {tick}: expected {[hex(v) for v in expected]} "
              f"captured {[hex(v) for v in actual]}")
    if mismatches:
        fail("game applied different controls than the extracted schedule")

    inputs_path = RESULTS_DIR / f"{track_id}_{tag}_inputs.bin"
    capture_path = RESULTS_DIR / f"{track_id}_{tag}.bin"
    if (inputs_path.exists() or capture_path.exists()) and not args.force:
        fail(f"{capture_path} exists; pass --force to replace")
    inputs_path.write_bytes(schedule)

    challenge = challenge_for(track_name)
    sha256 = hashlib.sha256(challenge.read_bytes()).hexdigest()
    track = ROOT / f"oracle/tracks/{track_name}.tmnftrack"
    vehicle = ROOT / f"oracle/vehicles/{track_id.upper()}-Stadium.tmnfvehicle"
    route = ROOT / f"oracle/routes/{track_name}.tmnfroute"
    native = staging / "native.bin"
    run(taskset([REPLAY_TICK, "--route", route, "--native-capture", track,
                 vehicle, inputs_path, native, sha256]),
        env=clean_environment(), timeout=600)
    run([sys.executable, ROOT / "tools/normalize_tick_capture.py",
         "--game", raw_capture, "--native", native, "--output", capture_path])
    result = subprocess.run(
        taskset([REPLAY_TICK, "--route", route, track, vehicle, capture_path,
                 "input_file", inputs_path, sha256]),
        cwd=ROOT, env=clean_environment(), timeout=600,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    print(result.stdout)
    events["reproduced"] = True
    events["ghost"] = meta
    events["track_sha256"] = sha256
    events["inputs_sha256"] = hashlib.sha256(schedule).hexdigest()
    events["normalized_capture_sha256"] = hashlib.sha256(
        capture_path.read_bytes()).hexdigest()
    events["native_replay_output"] = result.stdout
    events["native_replay_exit"] = result.returncode
    events_path.write_text(json.dumps(events, indent=1) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract")
    p.add_argument("replay", type=Path)
    p.add_argument("--script", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)

    p = sub.add_parser("schedule")
    p.add_argument("--script", type=Path, required=True)
    p.add_argument("--ticks", type=int, required=True)
    p.add_argument("--freeze-tick", type=int)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("verify")
    p.add_argument("--schedule", type=Path, required=True)
    p.add_argument("--capture", type=Path, required=True)

    p = sub.add_parser("run")
    p.add_argument("track_name")
    p.add_argument("replay", type=Path)
    p.add_argument("--tag", default="wr")
    p.add_argument("--after-finish-ticks", type=int, default=20)
    p.add_argument("--force", action="store_true")
    p.add_argument("--reuse-capture", action="store_true",
                   help="skip the game; redo verification, normalization and "
                        "the native replay from build/wr/<id>_<tag>/capture_raw.bin")
    p.add_argument("--wineprefix", type=Path, default=PREFIX)
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--display", default=XVFB_DISPLAY)
    p.add_argument("--cpu-set", default=CPU_SET)

    args = parser.parse_args()
    if args.command == "run":
        set_lane(args.wineprefix, args.port, args.display, args.cpu_set)
    if args.command == "extract":
        text, meta = extract(args.replay)
        args.script.write_text(text)
        args.meta.write_text(json.dumps(meta, indent=1) + "\n")
        print(json.dumps(meta, indent=1))
        return 0
    if args.command == "schedule":
        args.output.write_bytes(build_schedule(
            args.script.read_text(), args.ticks, args.freeze_tick))
        return 0
    if args.command == "verify":
        mismatches = verify(
            args.schedule.read_bytes(), args.capture.read_bytes())
        print(f"mismatches: {len(mismatches)}")
        for tick, expected, actual in mismatches[:50]:
            print(f"  tick {tick}: expected {[hex(v) for v in expected]} "
                  f"captured {[hex(v) for v in actual]}")
        return 1 if mismatches else 0
    return command_run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as error:
        print(f"wr_replay: {error}", file=sys.stderr)
        raise SystemExit(1)
