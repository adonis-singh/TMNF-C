#!/usr/bin/env python3
"""Validate a TMNF .Replay.Gbx against the bit-exact engine.

    validate_replay.py REPLAY [--ghost N] [--json OUT.json]

Decodes the replay (pygbx), resolves its embedded challenge to a registered
track snapshot by challenge UID, re-simulates the ghost's inputs with the
engine and compares the recorded finish time, checkpoint times and the sampled
ghost trajectory against the simulation. Prints one summary line and exits
0 (valid), 1 (invalid or scripted) or 2 (tooling error: undecodable replay,
unregistered track, unsupported inputs). Respawns are simulated
(analysis/respawn.md). See docs/VALIDATOR.md.

    validate_replay.py --rebuild-uid-table [--campaign-dir DIR]

Regenerates oracle/tracks/challenge_uids.txt from the installed campaign
challenges for every track registered in oracle/tracks/manifest.txt and
oracle/results/wr/manifest.txt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wr_replay  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
TICK_MS = 10
# Ghost samples are the game's float32 rigid-body position at the sample
# time; the engine reproduces them exactly (docs/VALIDATOR.md, corpus
# table), so anything beyond a millimetre is a different trajectory.
TRAJECTORY_TOLERANCE_M = 0.001
# Ticks simulated past the recorded finish before giving up on a finish.
FINISH_MARGIN_TICKS = 100
UID_TABLE = ROOT / "oracle/tracks/challenge_uids.txt"
TRACK_MANIFEST = ROOT / "oracle/tracks/manifest.txt"
WR_MANIFEST = ROOT / "oracle/results/wr/manifest.txt"
DEFAULT_SIMULATOR = ROOT / "build/tests/validate_replay_sim"
DEFAULT_CAMPAIGN_DIR = (
    ROOT / "oracle/wineprefix/drive_c/TmNationsForever/GameData/Tracks/Campaigns/Nations"
)
# TMInterface saves scripted runs with its input clock offset by 0xFFFF ms,
# which leaves the _FakeIsRaceRunning marker on a x5 millisecond. The same
# marker is what ForeverValidator's NormalizeScriptedInputClock detects.
SCRIPTED_CLOCK_OFFSET_MS = 0xFFFF


class ValidationError(RuntimeError):
    """Tooling error: the replay cannot be judged. Exit 2."""


def fail(message: str) -> None:
    raise ValidationError(message)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- replay decoding --------------------------------------------------------

class Replay:
    def __init__(self, path: Path) -> None:
        from pygbx import Gbx, GbxType  # type: ignore[import-not-found]

        self.path = path
        self.sha256 = sha256_file(path)
        try:
            self.gbx = Gbx(str(path))
        except Exception as error:  # pygbx raises plain exceptions
            fail(f"{path} is not a decodable GBX file: {error}")
        records = self.gbx.get_classes_by_ids(
            [GbxType.REPLAY_RECORD, GbxType.REPLAY_RECORD_OLD])
        if len(records) != 1:
            fail(f"{path} contains {len(records)} replay records; expected one")
        record = records[0]
        if record.track is None:
            fail(f"{path} has no embedded challenge")
        challenge = record.track.get_class_by_id(GbxType.CHALLENGE)
        if challenge is None or not challenge.map_uid:
            fail(f"{path}: embedded challenge has no UID")
        self.challenge_uid: str = challenge.map_uid
        self.challenge_name: str = challenge.map_name
        self.environment: str = challenge.environment
        self.nickname = record.nickname
        self.driver_login = record.driver_login
        self.ghosts = self.gbx.get_classes_by_ids(
            [GbxType.CTN_GHOST, GbxType.CTN_GHOST_OLD])
        if not self.ghosts:
            fail(f"{path} contains no ghost")


def is_scripted(ghost) -> bool:
    return any(
        entry.time % 10 == 5 and entry.event_name == "_FakeIsRaceRunning"
        for entry in ghost.control_entries
    )


def ghost_script(ghost) -> tuple[str, dict[str, object]]:
    """The ghost's inputs as a TMInterface script (per-ghost form of
    wr_replay.extract, whose conventions were measured against the game)."""
    entries = ghost.control_entries
    if not entries:
        fail("ghost carries no input events")
    scripted = is_scripted(ghost)
    invert_axis = any(e.event_name == "_FakeDontInverseAxis" for e in entries)
    if scripted:
        for entry in entries:
            entry.time -= SCRIPTED_CLOCK_OFFSET_MS

    lines: list[str] = []
    for index, event in enumerate(entries):
        if wr_replay._should_skip(event):
            continue
        is_unbound = False
        to_event = wr_replay._find_event_end(entries, event, index + 1)
        if to_event is not None:
            to = wr_replay._event_time(to_event)
        else:
            to = ghost.race_time
            if to == 4294967295:
                to = -1
                is_unbound = True
        start = wr_replay._event_time(event)
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
            # The press edge respawns the car at the last checkpoint
            # (analysis/respawn.md); wr_replay.build_schedule sets
            # TMNFRaceInputs.respawn on the start tick.
            key = "enter"
        elif name == "Steer":
            axis = wr_replay._analog_value(event)
            if invert_axis:
                axis = -axis
            lines.append(f"{start} steer {axis}")
            continue
        elif name == "Gas":
            fail("analog gas inputs are not supported by the schedule builder")
        elif name == "Horn":
            continue
        else:
            fail(f"unhandled ghost event {name}")
        if is_unbound:
            lines.append(f"{start} press {key}")
        else:
            lines.append(f"{start}-{to} press {key}")

    counts = {
        name: sum(1 for e in entries if e.event_name == name)
        for name in sorted({e.event_name for e in entries})
    }
    return "\n".join(lines) + "\n", {
        "events": counts,
        "scripted": scripted,
        "analog_steer": "Steer" in counts,
        "digital_steer": "SteerLeft" in counts or "SteerRight" in counts,
        "respawn_presses": sum(
            1 for e in entries if e.event_name == "Respawn" and e.enabled != 0),
    }


# --- track registry ---------------------------------------------------------

class Track:
    def __init__(self, track_id: str, name: str, sha256: str) -> None:
        self.id = track_id
        self.name = name
        self.sha256 = sha256
        self.snapshot = ROOT / f"oracle/tracks/{name}.tmnftrack"
        self.vehicle = ROOT / f"oracle/vehicles/{track_id.upper()}-Stadium.tmnfvehicle"
        self.route = ROOT / f"oracle/routes/{name}.tmnfroute"
        missing = [p for p in (self.snapshot, self.vehicle, self.route) if not p.is_file()]
        if missing:
            fail(
                f"track {name} is registered but incomplete, missing "
                + ", ".join(str(p.relative_to(ROOT)) for p in missing)
                + f"; rerun tools/onboard_track.py {name}")


def uid_table() -> dict[str, Track]:
    if not UID_TABLE.is_file():
        fail(f"{UID_TABLE.relative_to(ROOT)} is missing; run "
             "tools/validate_replay.py --rebuild-uid-table")
    table: dict[str, Track] = {}
    for line in UID_TABLE.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("|")
        if len(fields) != 4:
            fail(f"invalid challenge UID table line: {line}")
        uid, track_id, name, sha256 = fields
        if uid in table:
            fail(f"duplicate challenge UID {uid}")
        table[uid] = Track(track_id, name, sha256)
    return table


def resolve_track(replay: Replay) -> Track:
    if replay.environment != "Stadium":
        fail(f"{replay.challenge_name} is a {replay.environment} challenge; "
             "only Stadium (TMNF) is supported")
    table = uid_table()
    track = table.get(replay.challenge_uid)
    if track is None:
        fail(
            f"challenge {replay.challenge_name} (UID {replay.challenge_uid}) "
            "is not a registered track; onboard it with "
            f"tools/onboard_track.py {replay.challenge_name} and rerun "
            "tools/validate_replay.py --rebuild-uid-table")
    return track


def registered_tracks() -> list[tuple[str, str, str]]:
    records: dict[str, tuple[str, str, str]] = {}
    for manifest in (TRACK_MANIFEST, WR_MANIFEST):
        for line in manifest.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            track_id, name, sha256 = line.split("|")[:3]
            known = records.get(track_id)
            if known is not None and known != (track_id, name, sha256):
                fail(f"manifests disagree on track {track_id}")
            records[track_id] = (track_id, name, sha256)
    return sorted(records.values())


def rebuild_uid_table(campaign_dir: Path) -> int:
    from pygbx import Gbx, GbxType  # type: ignore[import-not-found]

    rows = []
    for track_id, name, sha256 in registered_tracks():
        matches = list(campaign_dir.glob(f"*/{name}.Challenge.Gbx"))
        if len(matches) != 1:
            fail(f"found {len(matches)} campaign challenges named {name}")
        challenge = matches[0]
        actual = sha256_file(challenge)
        if actual != sha256:
            fail(f"{challenge} has SHA-256 {actual}, manifest says {sha256}")
        uid = Gbx(str(challenge)).get_class_by_id(GbxType.CHALLENGE).map_uid
        if not uid:
            fail(f"{challenge} has no UID")
        rows.append((uid, track_id, name, sha256))
    if len({row[0] for row in rows}) != len(rows):
        fail("two registered tracks share a challenge UID")
    UID_TABLE.write_text(
        "# challenge UID|id|track name|challenge SHA-256\n"
        "# generated by tools/validate_replay.py --rebuild-uid-table from the\n"
        "# tracks registered in oracle/tracks/manifest.txt and\n"
        "# oracle/results/wr/manifest.txt\n"
        + "\n".join("|".join(row) for row in rows) + "\n")
    print(f"wrote {UID_TABLE.relative_to(ROOT)}: {len(rows)} tracks")
    return 0


# --- simulation -------------------------------------------------------------

def simulate(
    simulator: Path, track: Track, schedule: bytes, work_dir: Path
) -> tuple[list[dict[str, int]], list[tuple[float, float, float]]]:
    if not simulator.is_file():
        fail(f"simulator {simulator} is missing; build target validate_replay_sim")
    inputs_path = work_dir / "inputs.bin"
    trace_path = work_dir / "trace.bin"
    inputs_path.write_bytes(schedule)
    result = subprocess.run(
        [str(simulator), str(track.snapshot), str(track.vehicle),
         str(track.route), str(inputs_path), track.sha256, str(trace_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    if result.returncode != 0:
        fail(f"simulator failed ({result.returncode}): {result.stderr.strip()}")
    events = []
    ticks = None
    for line in result.stdout.splitlines():
        kind, *fields = line.split()
        if kind == "ticks":
            ticks = int(fields[0])
            continue
        event = {"type": kind}
        for field in fields:
            key, value = field.split("=")
            event[key] = int(value)
        events.append(event)
    if ticks is None:
        fail("simulator printed no tick count")
    trace_bytes = trace_path.read_bytes()
    if len(trace_bytes) != (ticks + 1) * 12:
        fail("simulator trace has invalid framing")
    trace = list(struct.iter_unpack("<3f", trace_bytes))
    return events, trace


def simulated_checkpoint_times(events: list[dict[str, int]]) -> list[int]:
    """Race times of the checkpoint, lap and finish crossings in tick order,
    which is what the game stores in the ghost's checkpoint list."""
    ticks = sorted({
        event["tick"] for event in events
        if event["type"] in ("checkpoint", "lap", "finish")
    })
    return [tick * TICK_MS for tick in ticks]


# --- verdict ----------------------------------------------------------------

def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True,
        stdout=subprocess.PIPE, text=True,
    ).stdout.strip()


def engine_identity(simulator: Path) -> dict[str, object]:
    return {
        "engine_commit": git_output("rev-parse", "HEAD"),
        "engine_dirty": bool(git_output("status", "--porcelain", "--", "src")),
        "physics_sha256": sha256_file(simulator),
    }


def validate(
    replay: Replay, ghost_index: int, simulator: Path, work_dir: Path,
    decode_s: float,
) -> dict[str, object]:
    started = time.monotonic()
    if ghost_index < 0 or ghost_index >= len(replay.ghosts):
        fail(f"ghost index {ghost_index} out of range; replay has "
             f"{len(replay.ghosts)} ghost(s)")
    ghost = replay.ghosts[ghost_index]
    track = resolve_track(replay)
    recorded_ms = int(ghost.race_time)
    if recorded_ms == 0xFFFFFFFF or recorded_ms <= 0:
        fail("ghost has no recorded race time")
    sample_period = int(ghost.sample_period or 0)
    if sample_period <= 0 or sample_period % TICK_MS != 0:
        fail(f"ghost sample period {sample_period} ms is not a tick multiple")
    if not ghost.records:
        fail("ghost carries no trajectory samples")
    cp_times = [int(t) for t in ghost.cp_times]

    script, input_info = ghost_script(ghost)
    ticks = recorded_ms // TICK_MS + FINISH_MARGIN_TICKS
    schedule = wr_replay.build_schedule(
        script, ticks, freeze_tick=recorded_ms // TICK_MS)
    scheduled = time.monotonic()
    events, trace = simulate(simulator, track, schedule, work_dir)
    simulated = time.monotonic()

    finish = [e for e in events if e["type"] == "finish"]
    simulated_ms = finish[0]["race_time_ms"] if finish else None
    simulated_cps = simulated_checkpoint_times(events)
    checkpoints = []
    for index in range(max(len(cp_times), len(simulated_cps))):
        recorded = cp_times[index] if index < len(cp_times) else None
        actual = simulated_cps[index] if index < len(simulated_cps) else None
        checkpoints.append({
            "index": index, "recorded_ms": recorded, "simulated_ms": actual,
            "match": recorded is not None and recorded == actual,
        })
    checkpoints_match = (
        len(cp_times) == len(simulated_cps)
        and all(entry["match"] for entry in checkpoints))

    # Sample i is the game's position labelled race time i * sample_period.
    # The game labels the state after k integration steps with race time
    # (k + 1) * 10 ms (oracle captures, tests/replay_tick.c), and the trace
    # holds the position entering step t + 1 at index t (after t steps and
    # after a respawn applied by that step: the game samples the respawned
    # car under the press's race time), so the sample lives at
    # trace[T / 10 - 1]; sample 0 is the stationary spawn, trace[0]. Samples
    # after the recorded finish are not compared: the game stops applying
    # inputs at the finish and the driver's keys after it are not part of
    # the race.
    samples_expected = recorded_ms // sample_period + 1
    if samples_expected > len(ghost.records):
        fail(f"ghost has {len(ghost.records)} samples, race time needs "
             f"{samples_expected}")
    samples_compared = 0
    first_divergence_tick = None
    max_deviation = 0.0
    max_deviation_tick = 0
    for index in range(samples_expected):
        tick = index * sample_period // TICK_MS
        trace_index = max(tick - 1, 0)
        if trace_index >= len(trace):
            if first_divergence_tick is None:
                first_divergence_tick = tick
            break
        sample = ghost.records[index].position
        x, y, z = trace[trace_index]
        deviation = math.sqrt(
            (x - sample.x) ** 2 + (y - sample.y) ** 2 + (z - sample.z) ** 2)
        samples_compared += 1
        if deviation > max_deviation:
            max_deviation = deviation
            max_deviation_tick = tick
        if deviation > TRAJECTORY_TOLERANCE_M and first_divergence_tick is None:
            first_divergence_tick = tick
    trajectory_match = (
        first_divergence_tick is None and samples_compared == samples_expected)

    reproduced = (
        simulated_ms == recorded_ms and checkpoints_match and trajectory_match)
    scripted = bool(input_info["scripted"])
    verdict: dict[str, object] = {
        "replay": str(replay.path),
        "replay_sha256": replay.sha256,
        "ghost_index": ghost_index,
        "ghost_count": len(replay.ghosts),
        "challenge": {
            "uid": replay.challenge_uid, "name": replay.challenge_name,
            "environment": replay.environment,
        },
        "track": {"id": track.id, "name": track.name},
        "login": ghost.login,
        "game_version": ghost.game_version,
        "valid": reproduced,
        "status": ("scripted" if scripted else
                   "valid" if reproduced else "invalid"),
        "recorded_ms": recorded_ms,
        "simulated_ms": simulated_ms,
        "checkpoints": checkpoints,
        "checkpoints_match": checkpoints_match,
        "first_divergence_tick": first_divergence_tick,
        "first_divergence_ms": (
            None if first_divergence_tick is None
            else first_divergence_tick * TICK_MS),
        "max_trajectory_deviation_m": max_deviation,
        "max_trajectory_deviation_tick": max_deviation_tick,
        "trajectory": {
            "sample_period_ms": sample_period,
            "samples_expected": samples_expected,
            "samples_compared": samples_compared,
            "tolerance_m": TRAJECTORY_TOLERANCE_M,
            "match": trajectory_match,
        },
        "respawns": int(ghost.num_respawns),
        "respawn_ticks": [e["tick"] for e in events if e["type"] == "respawn"],
        "scripted": scripted,
        "inputs": {
            **input_info,
            "ticks_scheduled": ticks,
            "schedule_sha256": hashlib.sha256(schedule).hexdigest(),
        },
        "simulated_ticks": len(trace) - 1,
        **engine_identity(simulator),
        "track_sha256": track.sha256,
        "vehicle_sha256": sha256_file(track.vehicle),
        "route_sha256": sha256_file(track.route),
        "timing_s": {
            "decode": round(decode_s, 4),
            "schedule": round(scheduled - started, 4),
            "simulate": round(simulated - scheduled, 4),
            "compare": round(time.monotonic() - simulated, 4),
        },
    }
    return verdict


def summary_line(verdict: dict[str, object], total_s: float) -> str:
    status = str(verdict["status"]).upper()
    checkpoints = verdict["checkpoints"]
    assert isinstance(checkpoints, list)
    matched = sum(1 for entry in checkpoints if entry["match"])
    text = (
        f"{status} {Path(str(verdict['replay'])).name}"
        f" ghost={verdict['ghost_index']}/{verdict['ghost_count']}"
        f" track={verdict['track']['name']}"  # type: ignore[index]
        f" recorded={verdict['recorded_ms']}ms"
        f" simulated={verdict['simulated_ms']}ms"
        f" checkpoints={matched}/{len(checkpoints)}"
        f" max_dev={verdict['max_trajectory_deviation_m']:.4f}m"
    )
    if verdict["first_divergence_tick"] is not None:
        text += f" first_divergence_tick={verdict['first_divergence_tick']}"
    if verdict["respawns"]:
        text += f" respawns={verdict['respawns']}"
    return text + f" {total_s:.3f}s"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("replay", type=Path, nargs="?")
    parser.add_argument("--ghost", type=int,
                        help="ghost index in a multi-ghost replay")
    parser.add_argument("--json", type=Path, help="write the verdict here")
    parser.add_argument("--simulator", type=Path, default=DEFAULT_SIMULATOR)
    parser.add_argument("--rebuild-uid-table", action="store_true")
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN_DIR)
    args = parser.parse_args()
    if args.rebuild_uid_table:
        return rebuild_uid_table(args.campaign_dir)
    if args.replay is None:
        parser.error("REPLAY is required")

    started = time.monotonic()
    replay = Replay(args.replay)
    decode_s = time.monotonic() - started
    ghost_index = args.ghost
    if ghost_index is None:
        if len(replay.ghosts) != 1:
            fail(f"{args.replay} contains {len(replay.ghosts)} ghosts; "
                 "pass --ghost N")
        ghost_index = 0
    with tempfile.TemporaryDirectory(prefix="validate_replay_") as work:
        verdict = validate(
            replay, ghost_index, args.simulator, Path(work), decode_s)
    total = time.monotonic() - started
    verdict["wall_time_s"] = round(total, 4)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(verdict, indent=1) + "\n")
    print(summary_line(verdict, total))
    return 0 if verdict["valid"] and not verdict["scripted"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValidationError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"validate_replay: {error}", file=sys.stderr)
        raise SystemExit(2)
