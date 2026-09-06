#!/usr/bin/env python3
"""Capture, validate, register, and export one campaign track.

Tracks are addressed as in tools/track_env.py: a Nations stem (A08-Endurance,
environment stadium) or a United Race stem (DesertA1, also desert/A1). The
lane's Wine prefix must hold the install that ships the track's campaign.

Lane parameters (Wine prefix, TMInterface port, Xvfb display, CPU set) come
from the environment or the command line so several lanes can run at once:

    TMNF_WINEPREFIX  oracle/wineprefix
    TMNF_PORT        8488
    TMNF_XVFB_DISPLAY 98
    TMNF_CPU_SET     14,30

Everything that touches shared state (native build tree, replay binaries,
viewer assets, track manifest) runs under file locks in build/onboarding/locks.
The native code is built from the committed HEAD in a private git worktree so
that uncommitted work in the main tree never affects the oracle comparison.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_env import (  # noqa: E402
    GameLayout, Track, game_layout, parse_track, track_from_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
PREFIX = Path(os.environ.get("TMNF_WINEPREFIX", ROOT / "oracle/wineprefix"))
PORT = int(os.environ.get("TMNF_PORT", "8488"))
XVFB_DISPLAY = os.environ.get("TMNF_XVFB_DISPLAY", "98")
CPU_SET = os.environ.get("TMNF_CPU_SET", "14,30")
LAYOUT: GameLayout | None = None  # bound by set_lane
# CSceneVehicleCar::ComputeForcesModel{3,4,5,6}; the tracer hooks all four and
# the track's tuning (+0x354) decides which one fires.
COMPUTE_FORCES_VAS = (0x007FA770, 0x007FB5F0, 0x007FC170, 0x007C3E80)
MANIFEST = ROOT / "oracle/tracks/manifest.txt"
PENDING_DIR = ROOT / "oracle/results/pending"
ASSET_MANIFEST = ROOT / "viewer/assets/game/manifest.json"
PYTHON = ROOT / "third_party/venv/bin/python"
BUILD_PYTHON = ROOT / "build/venv/bin/python"
ONBOARDING = ROOT / "build/onboarding"
TREE = ONBOARDING / "tree"
BUILD = ONBOARDING / "build"
LOCKS = ONBOARDING / "locks"
REPLAY_TICK = BUILD / "tests/replay_tick"
EXPORT_SCENE = BUILD / "export_viewer_scene"
RECORD_SIZE = 1668
INPUT_STRUCT = struct.Struct("<IIiIIiIIfIIiIIiIIf")
MAIN_PREFIX = ROOT / "oracle/wineprefix"

MIXED_SEGMENTS = (
    (180, True, False, 0),
    (140, True, False, -1),
    (140, True, False, 1),
    (80, True, True, -1),
    (80, True, True, 1),
    (120, False, True, 0),
    (180, True, False, 0),
    (160, True, False, 1),
    (120, False, False, 0),
)
WALL_SEGMENTS = (
    (220, True, False, 0),
    (320, True, False, 1),
    (80, True, True, 1),
    (220, True, False, -1),
    (80, True, True, -1),
    (320, True, False, 1),
    (160, False, False, 0),
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def set_lane(prefix: Path, port: int, display: str, cpu_set: str) -> None:
    """Bind this process to one game lane. Fails if the prefix is unusable."""
    global PREFIX, PORT, XVFB_DISPLAY, CPU_SET, LAYOUT
    LAYOUT = game_layout(Path(prefix))
    PREFIX = LAYOUT.prefix
    PORT = int(port)
    XVFB_DISPLAY = str(display)
    CPU_SET = cpu_set


def layout() -> GameLayout:
    if LAYOUT is None:
        fail("set_lane was not called")
    return LAYOUT


def run(
    command: list[str | Path],
    *,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    stdout: object | None = None,
) -> None:
    text = " ".join(str(value) for value in command)
    print(f"+ {text}", flush=True)
    subprocess.run(
        [str(value) for value in command],
        cwd=ROOT,
        env=env,
        check=True,
        timeout=timeout,
        stdout=stdout,
        stderr=subprocess.STDOUT if stdout is not None else None,
    )


def capture(
    command: list[str | Path],
    *,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    text = " ".join(str(value) for value in command)
    print(f"+ {text}", flush=True)
    result = subprocess.run(
        [str(value) for value in command],
        cwd=ROOT,
        env=env,
        check=False,
        timeout=timeout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    sys.stdout.write(result.stdout)
    sys.stdout.write(result.stderr)
    sys.stdout.flush()
    return result


def taskset(command: list[str | Path]) -> list[str | Path]:
    return ["taskset", "-c", CPU_SET, *command]


def clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["WINEPREFIX"] = str(PREFIX)
    environment["TMNF_WINEPREFIX"] = str(PREFIX)
    environment["TMNF_XVFB_DISPLAY"] = XVFB_DISPLAY
    environment["WINEDEBUG"] = "-all"
    environment["PYTHONPATH"] = str(ROOT / "oracle")
    return environment


@contextlib.contextmanager
def locked(name: str):
    """Exclusive inter-process lock shared by every onboarding lane."""
    LOCKS.mkdir(parents=True, exist_ok=True)
    path = LOCKS / f"{name}.lock"
    with path.open("w") as handle:
        started = time.monotonic()
        fcntl.flock(handle, fcntl.LOCK_EX)
        waited = time.monotonic() - started
        if waited > 1:
            print(f"acquired {name} lock after {waited:.0f}s", flush=True)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def wait_for_port(process: subprocess.Popen[bytes], port: int) -> None:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if process.poll() is not None:
            fail(f"game launcher exited with code {process.returncode}")
        result = subprocess.run(
            ["ss", "-ltnH"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        if any(
            line.split()[3].endswith(f":{port}")
            for line in result.stdout.splitlines()
            if len(line.split()) >= 4
        ):
            time.sleep(1)
            return
        time.sleep(1)
    fail(f"game did not listen on port {port}")


def stop_game(process: subprocess.Popen[bytes], environment: dict[str, str]) -> None:
    subprocess.run(
        ["wineserver", "-k"],
        cwd=ROOT,
        env=environment,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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


def run_game(
    track_name: str,
    client: list[str | Path],
    log_path: Path,
    extra_environment: dict[str, str] | None = None,
) -> None:
    environment = clean_environment()
    if extra_environment is not None:
        environment.update(extra_environment)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        launcher = subprocess.Popen(
            taskset([ROOT / "oracle/launch_game.sh", str(PORT), track_name]),
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for_port(launcher, PORT)
            run(taskset(client), env=environment, timeout=600, stdout=log)
        finally:
            stop_game(launcher, environment)


def install_tracer(mode: str) -> None:
    # The tracer build directory and the ASI loader download are shared.
    with locked("tracer"):
        run(taskset(["bash", ROOT / "oracle/tracer/install.sh", mode]),
            env=clean_environment(), timeout=300)


def disable_tracer() -> None:
    (layout().game_dir / "TMNFTracer.asi").unlink(missing_ok=True)


def challenge_for(track: Track) -> Path:
    return track.challenge(layout().game_dir)


def environment_base_vehicle(track: Track) -> Path | None:
    """The environment's lowest-coded vehicle snapshot (A01-Stadium for
    Stadium): the identity source for dump_vehicle_snapshot.py. None for the
    environment's first track, which is captured in bootstrap mode."""
    candidates = sorted(
        path for path in (ROOT / "oracle/vehicles").glob(
            f"*-{track.environment_name}.tmnfvehicle")
        if path != track.vehicle
    )
    return candidates[0] if candidates else None


def replay_for(challenge: Path) -> Path | None:
    """The route ghost: a committed oracle/ghosts/<Track>.<source>.Replay.Gbx
    when the official replay is unusable, otherwise the official replay."""
    stem = challenge.name.split(".Challenge.", 1)[0]
    committed = [
        path for path in (ROOT / "oracle/ghosts").glob(f"{stem}.*.Replay.*")
        if path.suffix.lower() == ".gbx"
    ]
    if len(committed) > 1:
        fail(f"found multiple committed ghosts for {stem}")
    if committed:
        return committed[0]
    matches = [
        path for path in challenge.parent.glob(f"{stem}.Replay.*")
        if path.suffix.lower() == ".gbx"
    ]
    if len(matches) > 1:
        fail(f"found multiple official replays for {challenge}")
    return matches[0] if matches else None


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_bound_snapshot(
    path: Path, magic: bytes, track_hash: bytes, hash_offset: int
) -> None:
    data = path.read_bytes()
    if len(data) < hash_offset + 32 or data[:8] != magic:
        fail(f"{path} has an invalid snapshot header")
    if data[hash_offset:hash_offset + 32] != track_hash:
        fail(f"{path} is bound to the wrong challenge")


def write_inputs(path: Path, segments: tuple[tuple[int, bool, bool, int], ...]) -> int:
    output = bytearray()
    tick = 0
    for count, accelerate, brake, steer in segments:
        for _ in range(count):
            timestamp = (tick + 1) * 10
            output.extend(INPUT_STRUCT.pack(
                timestamp, 0, int(steer < 0),
                timestamp, 0, int(steer > 0),
                0, 0, 0.0,
                timestamp, 0, int(accelerate),
                timestamp, 0, int(brake),
                0, 0, 0.0,
            ))
            tick += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(output)
    return tick


def manifest_records() -> list[tuple[str, str, str, str]]:
    records = []
    for line in (MANIFEST.read_text().splitlines() if MANIFEST.exists() else []):
        if not line or line.startswith("#"):
            continue
        fields = line.split("|")
        if len(fields) != 4:
            fail(f"invalid track manifest line: {line}")
        records.append(tuple(fields))  # type: ignore[arg-type]
    return records


def write_manifest(records: list[tuple[str, str, str, str]]) -> None:
    text = (
        "# id|track name|challenge SHA-256|comma-separated replay names\n"
        "# a replay name prefixed with pending: is a registered, known-diverging\n"
        "# reference stored under oracle/results/pending/<id>/\n"
        + "\n".join("|".join(record) for record in records)
        + "\n"
    )
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_suffix(".tmp")
    temporary.write_text(text)
    temporary.replace(MANIFEST)


def replay_state(replay_text: str) -> dict[str, bool]:
    """Map replay name -> True when the replay is exact, False when pending."""
    state: dict[str, bool] = {}
    for name in replay_text.split(","):
        if name.startswith("pending:"):
            state[name[len("pending:"):]] = False
        else:
            state[name] = True
    return state


def replay_text(state: dict[str, bool]) -> str:
    return ",".join(
        name if exact else f"pending:{name}" for name, exact in state.items()
    )


def register_track(
    track_id: str, track_name: str, sha256: str, exact: dict[str, bool]
) -> None:
    records = manifest_records()
    output = []
    found = False
    for record in records:
        if record[0] == track_id or record[1] == track_name:
            if record[:3] != (track_id, track_name, sha256):
                fail(f"manifest identity collision for {track_name}")
            state = replay_state(record[3])
            state.update(exact)
            output.append((track_id, track_name, sha256, replay_text(state)))
            found = True
        else:
            output.append(record)
    if not found:
        output.append((track_id, track_name, sha256, replay_text(exact)))
    write_manifest(output)


def reference_paths(track_id: str, name: str, exact: bool) -> tuple[Path, Path]:
    directory = ROOT / "oracle/results"
    if not exact:
        directory = PENDING_DIR / track_id
    return (
        directory / f"{track_id}_{name}.bin",
        directory / f"{track_id}_{name}_inputs.bin",
    )


def registered_outputs(record: tuple[str, str, str, str]) -> list[Path]:
    track_id, track_name, _, text = record
    track = track_from_manifest(track_id, track_name)
    paths = [track.snapshot, track.vehicle, track.route]
    for name, exact in replay_state(text).items():
        paths.extend(reference_paths(track_id, name, exact))
    return paths


def git(*arguments: str, cwd: Path = ROOT) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=cwd, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return result.stdout.strip()


SHIMS = ONBOARDING / "shims.json"


def sync_tree() -> str:
    """Check the committed HEAD out into the private build worktree.

    Harness sources under tools/ and tests/ that HEAD's CMake references but
    other agents keep uncommitted are copied in as shims; src/ is always the
    committed physics.
    """
    head = git("rev-parse", "HEAD")
    shims = json.loads(SHIMS.read_text()) if SHIMS.exists() else []
    for relative in shims:
        (TREE / relative).unlink(missing_ok=True)
    if not (TREE / ".git").exists():
        ONBOARDING.mkdir(parents=True, exist_ok=True)
        if TREE.exists():
            shutil.rmtree(TREE)
        git("worktree", "prune")
        git("worktree", "add", "--detach", str(TREE), head)
    elif git("rev-parse", "HEAD", cwd=TREE) != head:
        git("checkout", "--detach", head, cwd=TREE)
    shims = []
    for directory in ("tools", "tests"):
        for source in (ROOT / directory).glob("*.[ch]"):
            target = TREE / directory / source.name
            if not target.exists():
                shutil.copyfile(source, target)
                shims.append(f"{directory}/{source.name}")
    SHIMS.write_text(json.dumps(shims) + "\n")
    if shims:
        print(f"shimmed uncommitted harness sources: {', '.join(shims)}", flush=True)
    link = TREE / "oracle/wineprefix"
    if not link.exists() and not link.is_symlink():
        # Only TmForever.exe is read from here (inverse_trig_native test).
        link.symlink_to(MAIN_PREFIX)
    # tests/CMakeLists.txt requires the tools venv under third_party/.
    link = TREE / "third_party"
    if not link.is_symlink():
        link.symlink_to(ROOT / "third_party")
    # The Python tests resolve <root>/build/... relative to their source tree.
    (TREE / "build").mkdir(exist_ok=True)
    for name in ("libtmnf_physics.so", "export_viewer_scene", "venv", "runs"):
        link = TREE / "build" / name
        if not link.is_symlink():
            link.symlink_to(ROOT / "build" / name)
    return head


def configure_and_build(all_targets: bool = False) -> None:
    """Build replay_tick and export_viewer_scene from HEAD. Caller holds lock."""
    sync_tree()
    run(taskset([
        "cmake", "-S", TREE, "-B", BUILD, "-DCMAKE_BUILD_TYPE=Debug",
        f"-DPython3_EXECUTABLE={BUILD_PYTHON}",
        "-DTMNF_CUDA=OFF",
        f"-DTMNF_GAME_MASK={ROOT / 'local/game-mask.bin'}",
    ]), env=clean_environment(), timeout=300)
    targets = [] if all_targets else [
        "--target", "replay_tick", "export_viewer_scene"]
    run(taskset(["cmake", "--build", BUILD, *targets, "-j2"]),
        env=clean_environment(), timeout=900)


class ReplayResult:
    def __init__(self, name: str, matched: int, total: int, detail: dict) -> None:
        self.name = name
        self.matched = matched
        self.total = total
        self.detail = detail

    @property
    def exact(self) -> bool:
        return self.matched == self.total

    def as_dict(self) -> dict:
        return {
            "name": self.name, "matched": self.matched, "total": self.total,
            "exact": self.exact, **self.detail,
        }


DIVERGENCE_RE = re.compile(
    r"first divergence: tick (\d+) \(race time (\d+) ms\), field ([^,]+), "
    r"byte (\d+), word (\d+)\n\s*expected: (0x[0-9a-f]+) (\S+)\n"
    r"\s*actual:\s+(0x[0-9a-f]+) (\S+)"
)
MISMATCH_RE = re.compile(r"mismatch (\S+)\s+\+(\d+)\s+([0-9a-f]{8}) != ([0-9a-f]{8})")
SUMMARY_RE = re.compile(r"full tick: (\d+)/(\d+) ticks byte-exact")

SUSPECTS = (
    ("scene_mobil.physics", "CSceneVehicleCar engine/steer state (EngineIntegrate, ComputeForcesModel6)"),
    ("wheel[", "wheel contact/integration (WheelIntegrate, contact materials)"),
    ("dyna.position", "rigid-body step after collision response (MergeAndAddToCollisions, ComputeSynthetizedReplacement)"),
    ("dyna.linear_speed", "force integration or collision response"),
    ("dyna.add_linear_speed", "collision response added velocity"),
    ("dyna.angular_speed", "torque integration or collision response"),
    ("dyna.quat", "rotation integration"),
    ("dyna.rotation", "rotation integration"),
    ("dyna.force", "ComputeForcesModel6 force accumulation"),
    ("dyna.torque", "ComputeForcesModel6 torque accumulation"),
    ("dyna", "rigid-body integration"),
    ("race_time", "race timer"),
)


def suspect_for(field: str, collisions: int) -> str:
    for prefix, label in SUSPECTS:
        if field.startswith(prefix):
            if collisions > 0:
                return f"{label}; {collisions} collision record(s) at the divergent tick"
            return f"{label}; no collisions at the divergent tick"
    return "unknown"


def normalize_and_replay(
    name: str,
    track: Path,
    vehicle: Path,
    inputs: Path,
    reference: Path,
    track_sha256: str,
    staging: Path,
) -> ReplayResult:
    """Normalize non-physical words, then compare the native replay.

    Returns the byte-exact prefix. Any failure other than a plain divergence
    (track cannot load, framing error, crash) is raised as a tooling error.
    """
    native = staging / f"{reference.stem}_native.bin"
    normalized = staging / f"{reference.stem}_normalized.bin"
    run(taskset([
        REPLAY_TICK, "--native-capture", track, vehicle, inputs, native,
        track_sha256,
    ]), env=clean_environment(), timeout=300)
    run([
        sys.executable, ROOT / "tools/normalize_tick_capture.py",
        "--game", reference, "--native", native, "--output", normalized,
    ])
    normalized.replace(reference)
    environment = clean_environment()
    environment["TMNF_REPLAY_VERBOSE"] = "1"
    result = capture(taskset([
        REPLAY_TICK, track, vehicle, reference, "input_file", inputs,
        track_sha256,
    ]), env=environment, timeout=300)
    summary = SUMMARY_RE.search(result.stdout)
    if summary is None:
        fail(f"{name} replay did not run to completion: {result.stderr.strip()[-2000:]}")
    matched, total = int(summary.group(1)), int(summary.group(2))
    if result.returncode == 0 and matched == total:
        return ReplayResult(name, matched, total, {})
    divergence = DIVERGENCE_RE.search(result.stderr)
    if divergence is None:
        fail(f"{name} replay failed without a divergence report")
    tick = int(divergence.group(1))
    mismatches = [
        f"{field} +{offset} {expected}!={actual}"
        for field, offset, expected, actual in MISMATCH_RE.findall(result.stderr)
    ]
    environment["TMNF_REPLAY_COLLISION_TICK"] = str(tick + 1)
    collisions_run = capture(taskset([
        REPLAY_TICK, track, vehicle, reference, "input_file", inputs,
        track_sha256,
    ]), env=environment, timeout=300)
    collision_match = re.search(
        rf"tick {tick + 1} collisions: (\d+)", collisions_run.stdout)
    collisions = int(collision_match.group(1)) if collision_match else 0
    field = divergence.group(3)
    data = reference.read_bytes()
    materials = []
    for wheel in range(4):
        contact, encoded, _ = struct.unpack_from(
            "<3I", data, tick * RECORD_SIZE + 356 + wheel * 328 + 272)
        materials.append(f"{encoded & 0xFFFF}" if contact else "air")
    touched = sorted({
        struct.unpack_from("<3I", data, t * RECORD_SIZE + 356 + w * 328 + 272)[1] & 0xFFFF
        for t in range(tick + 1) for w in range(4)
        if struct.unpack_from("<I", data, t * RECORD_SIZE + 356 + w * 328 + 272)[0]
    })
    return ReplayResult(name, matched, total, {
        "wheel_materials_at_tick": materials,
        "materials_touched_before": touched,
        "tick": tick,
        "race_time_ms": int(divergence.group(2)),
        "field": field,
        "byte": int(divergence.group(4)),
        "word": int(divergence.group(5)),
        "expected": f"{divergence.group(6)} {divergence.group(7)}",
        "actual": f"{divergence.group(8)} {divergence.group(9)}",
        "mismatches": mismatches,
        "collisions_at_tick": collisions,
        "suspect": suspect_for(field, collisions),
    })


def verify_inputs(inputs: Path, reference: Path, staging: Path) -> bool:
    effective = staging / f"{inputs.stem}_effective.bin"
    run([
        sys.executable, ROOT / "tools/extract_effective_policy_inputs.py",
        "--inputs", inputs, "--capture", reference, "--output", effective,
    ])
    return inputs.read_bytes() == effective.read_bytes()


def track_stats(path: Path) -> dict[str, int]:
    data = path.read_bytes()
    sections = [
        struct.unpack_from("<QII", data, 96 + index * 16)
        for index in range(9)
    ]
    return {
        "entries": sections[0][1], "surfaces": sections[1][1],
        "meshes": sections[2][1], "vertices": sections[3][1],
        "faces": sections[4][1],
    }


def format_track_stats(stats: dict[str, int]) -> str:
    return " ".join(f"{key}={value}" for key, value in stats.items())


def route_stats(path: Path) -> dict[str, float | int]:
    data = path.read_bytes()
    sections = [
        struct.unpack_from("<QII", data, 96 + index * 16)
        for index in range(5)
    ]
    point_offset, point_count, point_stride = sections[4]
    if point_count < 2 or point_stride != 24:
        fail("route has no dense centerline")
    length = struct.unpack_from(
        "<f", data, point_offset + (point_count - 1) * point_stride + 12
    )[0]
    widths = [
        struct.unpack_from("<f", data, point_offset + index * point_stride + 16)[0]
        for index in range(point_count)
    ]
    return {
        "points": point_count, "length_m": round(length, 3),
        "half_width_min_m": round(min(widths), 3),
        "half_width_max_m": round(max(widths), 3),
        "laps": struct.unpack_from("<I", data, sections[0][0])[0],
        "checkpoints": sections[2][1],
    }


def format_route_stats(stats: dict) -> str:
    return (
        f"points={stats['points']} length={stats['length_m']:.3f}m "
        f"half_width={stats['half_width_min_m']:.3f}.."
        f"{stats['half_width_max_m']:.3f}m"
    )


def route_start_waypoint(path: Path) -> int:
    data = path.read_bytes()
    start_offset, start_count, start_stride = struct.unpack_from("<QII", data, 112)
    if start_count != 1 or start_stride != 0x114:
        fail("route start section has invalid framing")
    return struct.unpack_from("<I", data, start_offset + 224)[0]


def asset_variant_count() -> int:
    if not ASSET_MANIFEST.is_file():
        return 0
    return len(json.loads(ASSET_MANIFEST.read_text())["assets"])


def run_assets(current: tuple[Track, Path, Path]) -> None:
    """Viewer block assets. The viewer renders Stadium only, so the asset
    manifest covers Stadium tracks and other environments skip this step."""
    track, challenge, scene = current
    if track.environment != "stadium":
        print(f"viewer assets: skipped, {track.environment_name} is not viewable", flush=True)
        return
    records = manifest_records()
    known = {record[0] for record in records}
    arguments: list[str | Path] = [
        sys.executable,
        ROOT / "tools/extract_game_assets/extract_game_assets.py",
        "--game-dir", layout().game_dir,
    ]
    for record in records:
        identifier, name, _, _ = record
        registered = track_from_manifest(identifier, name)
        if registered.environment != "stadium":
            continue
        source = challenge_for(registered)
        source_scene = ROOT / f"viewer/scenes/{identifier}_mixed.json"
        if not source_scene.is_file():
            continue
        arguments.extend(("--track", f"{identifier}={source}"))
        arguments.extend(("--scene-source", f"{identifier}={source_scene}"))
    if track.id not in known:
        arguments.extend(("--track", f"{track.id}={challenge}"))
        arguments.extend(("--scene-source", f"{track.id}={scene}"))
    run(taskset(arguments), env=clean_environment(), timeout=1800)


def write_result(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def idempotent_verify(record: tuple[str, str, str, str], result_path: Path | None) -> None:
    missing = [path for path in registered_outputs(record) if not path.is_file()]
    if missing:
        fail("registered track is incomplete: " + ", ".join(map(str, missing)))
    track_id, track_name, sha256, text = record
    registered = track_from_manifest(track_id, track_name)
    track, vehicle, route = registered.snapshot, registered.vehicle, registered.route
    replays = []
    with locked("shared"):
        configure_and_build()
        for name, exact in replay_state(text).items():
            reference, inputs = reference_paths(track_id, name, exact)
            environment = clean_environment()
            environment["TMNF_REPLAY_VERBOSE"] = "1"
            result = capture(taskset([
                REPLAY_TICK, track, vehicle, reference, "input_file", inputs,
                sha256,
            ]), env=environment, timeout=300)
            summary = SUMMARY_RE.search(result.stdout)
            if summary is None:
                fail(f"{name} replay did not run: {result.stderr.strip()[-2000:]}")
            matched, total = int(summary.group(1)), int(summary.group(2))
            if exact and matched != total:
                fail(f"registered exact replay {track_id}_{name} now diverges at {matched}/{total}")
            replays.append({"name": name, "matched": matched, "total": total,
                            "exact": matched == total, "registered_exact": exact})
    print(f"{track_name} already onboarded; references left byte-identical")
    print(format_track_stats(track_stats(track)))
    print(format_route_stats(route_stats(route)))
    for replay in replays:
        print(f"{replay['name']}: {replay['matched']}/{replay['total']}")
    write_result(result_path, {
        "track_id": track_id, "track_name": track_name, "status": "verified",
        "challenge_sha256": sha256, "track": track_stats(track),
        "route": route_stats(route), "replays": replays,
    })


def capture_reference(
    track_name: str, name: str, inputs: Path, reference: Path, staging: Path
) -> None:
    """Capture one reference; redo once if the game altered any control."""
    for attempt in (1, 2):
        run_game(
            track_name,
            [PYTHON, ROOT / "oracle/capture_policy_lap.py",
             "--port", str(PORT), "--map", track_name,
             "--inputs", inputs, "--output", reference],
            staging / f"{name}{'' if attempt == 1 else '-retry'}.log",
        )
        ticks = len(inputs.read_bytes()) // INPUT_STRUCT.size
        if ticks < 1200 or reference.stat().st_size != ticks * RECORD_SIZE:
            fail(f"{name} reference has invalid tick framing")
        if verify_inputs(inputs, reference, staging):
            return
        print(f"{name} capture contains rejected or altered controls; "
              f"discarding attempt {attempt}", flush=True)
        reference.unlink()
    fail(f"{name} capture rejected or altered controls twice")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("track_name")
    parser.add_argument("--visuals", action="store_true",
                        help="also extract local Stadium viewer models and textures")
    parser.add_argument("--force", action="store_true",
                        help="discard existing outputs and recapture")
    parser.add_argument("--reuse-captures", action="store_true",
                        help="skip the game phase; redo replay, route, scene, "
                             "assets and registration from existing captures")
    parser.add_argument("--wineprefix", type=Path, default=PREFIX)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--display", default=XVFB_DISPLAY)
    parser.add_argument("--cpu-set", default=CPU_SET)
    parser.add_argument("--result", type=Path,
                        help="write a JSON summary of the outcome here")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved lane, paths and captures; run nothing")
    args = parser.parse_args()
    set_lane(args.wineprefix, args.port, args.display, args.cpu_set)
    current = parse_track(args.track_name)
    args.track_name = current.name
    track_id = current.id
    if args.dry_run:
        print_plan(current)
        return 0
    registered = [
        record for record in manifest_records()
        if record[0] == track_id or record[1] == args.track_name
    ]
    if registered and not (args.force or args.reuse_captures):
        idempotent_verify(registered[0], args.result)
        return 0

    challenge = challenge_for(current)
    challenge_sha256 = file_sha256(challenge)
    challenge_hash = bytes.fromhex(challenge_sha256)
    track = current.snapshot
    route = current.route
    vehicle = current.vehicle
    mixed_inputs = current.result("mixed", "_inputs")
    mixed_capture = current.result("mixed")
    wall_inputs = current.result("wall_contact", "_inputs")
    wall_capture = current.result("wall_contact")
    scene = ROOT / f"viewer/scenes/{track_id}_mixed.json"
    scene.parent.mkdir(parents=True, exist_ok=True)
    outputs = (
        track, route, vehicle, mixed_inputs, mixed_capture,
        wall_inputs, wall_capture, scene, PENDING_DIR / track_id,
    )
    started = time.monotonic()
    staging = ROOT / f"build/onboard/{track_id}"
    if args.reuse_captures:
        for name, inputs, reference in (
            ("mixed", mixed_inputs, mixed_capture),
            ("wall_contact", wall_inputs, wall_capture),
        ):
            pending_reference, pending_inputs = reference_paths(track_id, name, False)
            if pending_reference.exists():
                pending_reference.replace(reference)
                pending_inputs.replace(inputs)
        captured = (track, route, vehicle, mixed_inputs, mixed_capture,
                    wall_inputs, wall_capture)
        missing = [path for path in captured if not path.is_file()]
        if missing:
            fail("cannot reuse captures, missing: " + ", ".join(map(str, missing)))
        verify_bound_snapshot(track, b"TMNFTRK1", challenge_hash, 64)
        verify_bound_snapshot(route, b"TMNFROU1", challenge_hash, 64)
        verify_bound_snapshot(vehicle, b"TMNFM6G1", challenge_hash, 288)
        staging.mkdir(parents=True, exist_ok=True)
        for inputs, reference in ((mixed_inputs, mixed_capture), (wall_inputs, wall_capture)):
            if not verify_inputs(inputs, reference, staging):
                fail(f"{reference} contains rejected or altered controls")
        scene.unlink(missing_ok=True)
        if (PENDING_DIR / track_id).exists():
            shutil.rmtree(PENDING_DIR / track_id)
        print("reusing existing captures", flush=True)
    else:
        existing = [path for path in outputs if path.exists()]
        if existing and not args.force:
            fail("refusing to overwrite partial onboarding: " + ", ".join(map(str, existing)))
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        capture_all(current, challenge, challenge_hash,
                    staging, track, route, vehicle,
                    mixed_inputs, mixed_capture, wall_inputs, wall_capture)
    capture_seconds = time.monotonic() - started

    with locked("shared"):
        head = sync_tree()
        configure_and_build()
        replays = [
            normalize_and_replay(
                "mixed", track, vehicle, mixed_inputs, mixed_capture,
                challenge_sha256, staging),
            normalize_and_replay(
                "wall_contact", track, vehicle, wall_inputs, wall_capture,
                challenge_sha256, staging),
        ]
        for replay in replays:
            print(f"{replay.name}: {replay.matched}/{replay.total} byte-exact", flush=True)

        print("generating dense route centerline", flush=True)
        route_command: list[str | Path] = [
            PYTHON, ROOT / "tools/generate_route_centerline.py",
            "--track", track, "--route", route,
            "--run1", mixed_capture, "--long-drive", wall_capture,
            "--validation-drive", mixed_capture,
        ]
        ghost = replay_for(challenge)
        if ghost is not None:
            # Multi-lap ghosts are cut to their first lap by the generator;
            # replays without a usable ghost fall back to the road grid.
            route_command.extend(("--ghost-replay", ghost, "--ghost-optional"))
        run(taskset(route_command), env=clean_environment(), timeout=600)
        run(taskset([*route_command, "--check"]), env=clean_environment(), timeout=600)

        print("exporting viewer scene and missing assets", flush=True)
        run(taskset([
            EXPORT_SCENE, track, route, vehicle, mixed_inputs,
            challenge_sha256, track_id, scene,
        ]), env=clean_environment(), timeout=600)
        variants_before = asset_variant_count()
        if args.visuals:
            run_assets((current, challenge, scene))
        variants_after = asset_variant_count()

        exact = {replay.name: replay.exact for replay in replays}
        for replay, inputs, reference in (
            (replays[0], mixed_inputs, mixed_capture),
            (replays[1], wall_inputs, wall_capture),
        ):
            if replay.exact:
                continue
            pending_reference, pending_inputs = reference_paths(
                track_id, replay.name, False)
            pending_reference.parent.mkdir(parents=True, exist_ok=True)
            reference.replace(pending_reference)
            inputs.replace(pending_inputs)
        register_track(track_id, args.track_name, challenge_sha256, exact)

    elapsed = time.monotonic() - started
    status = "exact" if all(exact.values()) else "pending-divergence"
    print(f"onboarded {args.track_name} in {elapsed:.1f}s ({status})")
    print(f"challenge sha256={challenge_sha256}")
    print(format_track_stats(track_stats(track)))
    print(format_route_stats(route_stats(route)))
    print("captures: " + " ".join(
        f"{replay.name}={replay.matched}/{replay.total}" for replay in replays))
    write_result(args.result, {
        "track_id": track_id, "track_name": args.track_name, "status": status,
        "challenge_sha256": challenge_sha256, "head": head,
        "seconds": round(elapsed, 1), "capture_seconds": round(capture_seconds, 1),
        "track": track_stats(track), "route": route_stats(route),
        "new_block_variants": variants_after - variants_before,
        "replays": [replay.as_dict() for replay in replays],
        "files": [
            str(path.relative_to(ROOT)) for path in (
                track, route, vehicle, scene, MANIFEST,
                *reference_paths(track_id, "mixed", exact["mixed"]),
                *reference_paths(track_id, "wall_contact", exact["wall_contact"]),
            )
        ],
    })
    return 0


def print_plan(current: Track) -> None:
    lane = layout()
    base = environment_base_vehicle(current)
    challenge = challenge_for(current)
    lines = [
        f"track: {current.name} (campaign {current.campaign}, environment "
        f"{current.environment}/{current.environment_name}, code {current.code}, id {current.id})",
        f"lane: prefix {lane.prefix} ({lane.flavor}), port {PORT}, display :{XVFB_DISPLAY}, "
        f"cpus {CPU_SET}",
        f"game: {lane.game_dir}",
        f"challenge: {challenge}",
        f"installed as: {lane.official_maps / challenge.name}",
        f"track snapshot: {current.snapshot.relative_to(ROOT)}",
        f"route snapshot: {current.route.relative_to(ROOT)}",
        f"vehicle snapshot: {current.vehicle.relative_to(ROOT)}",
        "vehicle identity: " + (
            f"base {base.relative_to(ROOT)}" if base is not None
            else f"bootstrap (first {current.environment_name} vehicle)"),
        "vehicle trace targets: " + ",".join(f"0x{va:08X}" for va in COMPUTE_FORCES_VAS),
        "references: " + ", ".join(
            str(current.result(name).relative_to(ROOT)) for name in ("mixed", "wall_contact")),
        f"manifest line: {current.id}|{current.name}|{file_sha256(challenge)}|mixed,wall_contact",
        f"ctest names: {current.id}_mixed_replay, {current.id}_wall_contact_replay",
        "viewer assets: " + ("extract" if current.environment == "stadium" else "skipped (Stadium-only viewer)"),
    ]
    print("\n".join(lines))


def capture_all(
    current: Track, challenge: Path, challenge_hash: bytes,
    staging: Path, track: Path, route: Path, vehicle: Path,
    mixed_inputs: Path, mixed_capture: Path, wall_inputs: Path, wall_capture: Path,
) -> None:
    """Run every game-side capture for one track into its output paths."""
    track_name = current.name
    official_maps = layout().official_maps
    official_maps.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(challenge, official_maps / challenge.name)

    write_inputs(mixed_inputs, MIXED_SEGMENTS)
    write_inputs(wall_inputs, WALL_SEGMENTS)

    print("capturing static track snapshot", flush=True)
    install_tracer("track")
    run_game(
        track_name,
        [PYTHON, ROOT / "oracle/determinism_test.py",
         "--port", str(PORT), "--map", track_name, "--ticks", "10",
         "--output-dir", staging / "track-probe"],
        staging / "track.log",
    )
    verify_bound_snapshot(track, b"TMNFTRK1", challenge_hash, 64)

    print("capturing race route snapshot", flush=True)
    install_tracer("route")
    run_game(
        track_name,
        [PYTHON, ROOT / "oracle/determinism_test.py",
         "--port", str(PORT), "--map", track_name, "--ticks", "10",
         "--output-dir", staging / "route-probe"],
        staging / "route.log",
    )
    verify_bound_snapshot(route, b"TMNFROU1", challenge_hash, 64)

    print("capturing track-bound vehicle graph", flush=True)
    traces = staging / "vehicle-traces"
    traces.mkdir()
    install_tracer("trace")
    run_game(
        track_name,
        [PYTHON, ROOT / "oracle/determinism_test.py",
         "--port", str(PORT), "--map", track_name, "--ticks", "10",
         "--output-dir", staging / "vehicle-probe"],
        staging / "vehicle-phase.log",
        {
            "TMNF_TRACE_HOST_DIR": str(traces),
            "TMNF_TRACE_TARGETS": ",".join(f"0x{va:08X}" for va in COMPUTE_FORCES_VAS),
        },
    )
    # The tracer opens one file per hooked target; only the model the tuning
    # selects records anything, the other three stay at the 16-byte header.
    model_traces = [
        path for path in sorted(traces.glob("*_CSceneVehicleCar_ComputeForcesModel?.bin"))
        if struct.unpack_from("<I", path.read_bytes(), 12)[0] != 0
    ]
    if len(model_traces) != 1:
        fail(f"expected one recorded ComputeForces model trace, found {len(model_traces)}")
    print(f"vehicle model: {model_traces[0].stem.split('ComputeForces', 1)[1]}", flush=True)
    phase = staging / f"{vehicle.stem}-phase.tmnfvehicle"
    run([
        PYTHON, ROOT / "oracle/tracer/extract_model6_vehicle.py",
        model_traces[0], phase,
    ])
    disable_tracer()
    dump_arguments: list[str | Path] = [
        PYTHON, ROOT / "oracle/dump_vehicle_snapshot.py",
        "--port", str(PORT), "--map", track_name,
        "--phase-input", phase, "--output", vehicle,
        "--report", staging / "vehicle.md",
    ]
    base = environment_base_vehicle(current)
    if base is None:
        print(f"first {current.environment_name} vehicle: bootstrapping identities", flush=True)
    else:
        dump_arguments.extend(("--input", base))
    run_game(track_name, dump_arguments, staging / "vehicle.log")
    verify_bound_snapshot(vehicle, b"TMNFM6G1", challenge_hash, 288)

    print("capturing mixed and wall-heavy references", flush=True)
    capture_reference(track_name, "mixed", mixed_inputs, mixed_capture, staging)
    capture_reference(track_name, "wall_contact", wall_inputs, wall_capture, staging)



if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f"onboard_track: {error}", file=sys.stderr)
        raise SystemExit(1)
