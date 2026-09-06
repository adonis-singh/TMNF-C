#!/usr/bin/env python3
"""Corpus gate for tools/validate_replay.py (ctest validate_replay_corpus).

Validates every replay listed in oracle/replays/manifest.txt against its
expected verdict, requires every committed .Replay.Gbx under oracle/replays
and oracle/ghosts to be listed, fabricates two invalid replays from the
corpus (one input tick edited in a world-record ghost; two ghosts spliced)
and requires the validator to reject both at the first ghost sample the
tampered inputs move by more than the trajectory tolerance. Prints the
corpus table with wall times. Exit 0 when every expectation holds.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import validate_replay  # noqa: E402
import wr_replay  # noqa: E402

VALIDATOR = ROOT / "tools/validate_replay.py"
MANIFEST = ROOT / "oracle/replays/manifest.txt"
REPLAY_DIRS = (ROOT / "oracle/replays", ROOT / "oracle/ghosts")
CONTROL_ENTRY_SIZE = 9
CONTROL_TIME_BIAS = 100000


def fail(message: str) -> None:
    raise RuntimeError(message)


# --- corpus -----------------------------------------------------------------

class Expectation:
    def __init__(self, line: str) -> None:
        fields = line.split("|")
        if len(fields) != 6:
            fail(f"invalid corpus manifest line: {line}")
        path, ghost, status, recorded, divergence, self.source = fields
        self.path = ROOT / "oracle" / path
        self.ghost = int(ghost)
        if status not in ("valid", "invalid", "unsupported"):
            fail(f"unknown expected status {status}")
        self.status = status
        self.recorded_ms = int(recorded)
        self.first_divergence_tick = None if divergence == "-" else int(divergence)


def read_manifest() -> list[Expectation]:
    expectations = [
        Expectation(line) for line in MANIFEST.read_text().splitlines()
        if line and not line.startswith("#")
    ]
    listed = {expectation.path for expectation in expectations}
    committed = {
        path for directory in REPLAY_DIRS
        for path in directory.glob("*.Replay.Gbx")
    }
    unlisted = sorted(committed - listed)
    if unlisted:
        fail("replays without a corpus expectation: "
             + ", ".join(str(p.relative_to(ROOT)) for p in unlisted))
    missing = sorted(listed - committed)
    if missing:
        fail("corpus manifest lists missing replays: "
             + ", ".join(str(p.relative_to(ROOT)) for p in missing))
    return expectations


class Run:
    def __init__(self, replay: Path, ghost: int, simulator: Path, json_path: Path) -> None:
        started = time.monotonic()
        result = subprocess.run(
            [sys.executable, str(VALIDATOR), str(replay), "--ghost", str(ghost),
             "--json", str(json_path), "--simulator", str(simulator)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
        )
        self.wall_s = time.monotonic() - started
        self.exit_code = result.returncode
        self.summary = result.stdout.strip()
        self.error = result.stderr.strip()
        self.verdict = json.loads(json_path.read_text()) if json_path.is_file() else None
        if self.exit_code not in (0, 1, 2):
            fail(f"validator crashed on {replay}: {self.error}")
        if self.exit_code == 2 and self.verdict is not None:
            fail(f"validator wrote a verdict but exited 2 on {replay}")
        if self.exit_code != 2 and self.verdict is None:
            fail(f"validator exited {self.exit_code} without a verdict on {replay}")

    @property
    def status(self) -> str:
        return "unsupported" if self.verdict is None else str(self.verdict["status"])


def check_expectation(expectation: Expectation, run: Run) -> list[str]:
    problems = []
    if run.status != expectation.status:
        problems.append(f"status {run.status}, expected {expectation.status}")
    expected_exit = {"valid": 0, "invalid": 1, "unsupported": 2}[expectation.status]
    if run.exit_code != expected_exit:
        problems.append(f"exit {run.exit_code}, expected {expected_exit}")
    if run.verdict is None:
        return problems
    verdict = run.verdict
    if verdict["recorded_ms"] != expectation.recorded_ms:
        problems.append(f"recorded {verdict['recorded_ms']} ms, "
                        f"expected {expectation.recorded_ms}")
    if verdict["first_divergence_tick"] != expectation.first_divergence_tick:
        problems.append(f"first divergence tick {verdict['first_divergence_tick']}, "
                        f"expected {expectation.first_divergence_tick}")
    if expectation.status == "valid":
        if verdict["simulated_ms"] != expectation.recorded_ms:
            problems.append(f"simulated {verdict['simulated_ms']} ms")
        if verdict["max_trajectory_deviation_m"] != 0.0:
            problems.append(f"max deviation {verdict['max_trajectory_deviation_m']} m")
        if not verdict["checkpoints_match"]:
            problems.append("checkpoints differ")
        if verdict["scripted"]:
            problems.append("flagged scripted")
        if verdict["trajectory"]["samples_compared"] != verdict["trajectory"]["samples_expected"]:
            problems.append("not every sample compared")
    return problems


# --- fabricated invalid replays ---------------------------------------------

class GhostBytes:
    """A replay's decompressed body plus the byte range of its first ghost's
    control-entry array (9 bytes per entry: u32 time + 100000, u8 control
    name index, u16 enabled, u16 flags)."""

    def __init__(self, path: Path) -> None:
        from pygbx import Gbx, GbxType  # type: ignore[import-not-found]

        self.path = path
        self.raw = path.read_bytes()
        gbx = Gbx(str(path))
        self.size_offset = gbx.positions["data_size"].pos
        self.body = bytes(gbx.data)
        ghosts = gbx.get_classes_by_ids([GbxType.CTN_GHOST, GbxType.CTN_GHOST_OLD])
        if len(ghosts) != 1:
            fail(f"{path} has {len(ghosts)} ghosts; tampering needs one")
        self.ghost = ghosts[0]
        self.entries = self.ghost.control_entries
        self.race_time = int(self.ghost.race_time)
        self.entries_offset = self._locate()
        self.name_index = {}
        for index, entry in enumerate(self.entries):
            raw_index = self.body[self.entries_offset + index * CONTROL_ENTRY_SIZE + 4]
            known = self.name_index.setdefault(entry.event_name, raw_index)
            if known != raw_index:
                fail("control name index is not stable")

    def _locate(self) -> int:
        entries = self.entries
        if len(entries) < 3:
            fail("ghost has too few control entries to locate")
        hits = []
        for offset in range(len(self.body) - 3 * CONTROL_ENTRY_SIZE):
            if all(
                struct.unpack_from("<I", self.body, offset + k * CONTROL_ENTRY_SIZE)[0]
                == entries[k].time + CONTROL_TIME_BIAS
                for k in range(3)
            ) and struct.unpack_from("<HH", self.body, offset + 5) == (
                entries[0].enabled, entries[0].flags
            ):
                hits.append(offset)
        if len(hits) != 1:
            fail(f"control entries located {len(hits)} times in {self.path.name}")
        count = struct.unpack_from("<I", self.body, hits[0] - 8)[0]
        if count != len(entries):
            fail("control entry count does not match the located array")
        return hits[0]

    def entry_bytes(self, index: int) -> bytes:
        start = self.entries_offset + index * CONTROL_ENTRY_SIZE
        return self.body[start:start + CONTROL_ENTRY_SIZE]

    def write_with_entries(self, entries: list[bytes], output: Path) -> None:
        import lzo  # type: ignore[import-not-found]

        start = self.entries_offset - 8
        end = self.entries_offset + len(self.entries) * CONTROL_ENTRY_SIZE
        block = (struct.pack("<I", len(entries))
                 + self.body[self.entries_offset - 4:self.entries_offset]
                 + b"".join(entries))
        body = self.body[:start] + block + self.body[end:]
        compressed = lzo.compress(body, 9, False)
        output.write_bytes(
            self.raw[:self.size_offset]
            + struct.pack("<II", len(body), len(compressed))
            + compressed)


def steer_word(value: int) -> tuple[int, int]:
    """(enabled, flags) storing analog steer `value` the way the ghost
    recorder does: the negated value in a 24-bit two's complement word."""
    word = (-value) & 0xFFFFFF
    return word & 0xFFFF, word >> 16


def fabricate_edit(source: GhostBytes, edit_ms: int, output: Path) -> int:
    """Negate the analog steer value of the Steer event at `edit_ms`, which
    must be followed by another Steer event 10 ms later so that exactly one
    schedule tick changes. Returns the schedule tick that changed."""
    times = [e.time for e in source.entries]
    candidates = [
        index for index, entry in enumerate(source.entries)
        if entry.event_name == "Steer" and entry.time == edit_ms
        and edit_ms + 10 in times
        and source.entries[times.index(edit_ms + 10)].event_name == "Steer"
    ]
    if len(candidates) != 1:
        fail(f"no single-tick Steer event at {edit_ms} ms in {source.path.name}")
    index = candidates[0]
    entry = source.entries[index]
    value = wr_replay._analog_value(entry)
    if abs(value) < 8192:
        fail(f"steer at {edit_ms} ms is too small ({value}) to make a visible edit")
    enabled, flags = steer_word(-value)
    original = source.entry_bytes(index)
    edited = original[:5] + struct.pack("<HH", enabled, flags)
    entries = [source.entry_bytes(k) for k in range(len(source.entries))]
    entries[index] = edited
    source.write_with_entries(entries, output)
    # wr_replay._event_time: an analog event at t applies from tick t/10 - 1.
    return wr_replay._event_time(entry) // validate_replay.TICK_MS


def fabricate_splice(
    first: GhostBytes, second: GhostBytes, splice_ms: int, output: Path
) -> None:
    """Keep the first ghost (samples, times, header) but replace its inputs
    from `splice_ms` on with the second ghost's inputs."""
    if first.ghost.uid == second.ghost.uid:
        fail("splice needs two different ghosts")
    entries = [
        first.entry_bytes(k) for k, e in enumerate(first.entries)
        if e.time < splice_ms
    ]
    for k, entry in enumerate(second.entries):
        if entry.time < splice_ms or entry.event_name.startswith("_Fake"):
            continue
        if entry.event_name not in first.name_index:
            fail(f"second ghost uses control {entry.event_name} unknown to the first")
        raw = bytearray(second.entry_bytes(k))
        raw[4] = first.name_index[entry.event_name]
        entries.append(bytes(raw))
    # Keep the first ghost's own terminal markers so the timeline framing
    # (_FakeFinishLine at the recorded race time) stays that of ghost one.
    entries.extend(
        first.entry_bytes(k) for k, e in enumerate(first.entries)
        if e.time >= splice_ms and e.event_name.startswith("_Fake")
    )
    first.write_with_entries(entries, output)


def fabricate_scripted(source: GhostBytes, output: Path) -> None:
    """Re-time every control entry by TMInterface's input clock offset, which
    is how a scripted run's ghost stores its inputs (the _FakeIsRaceRunning
    marker lands on a x5 ms time)."""
    entries = []
    for k, entry in enumerate(source.entries):
        raw = bytearray(source.entry_bytes(k))
        struct.pack_into(
            "<I", raw, 0,
            entry.time + CONTROL_TIME_BIAS + validate_replay.SCRIPTED_CLOCK_OFFSET_MS)
        entries.append(bytes(raw))
    source.write_with_entries(entries, output)


def schedule_for(path: Path, simulator: Path, work: Path):
    """The validator's own decode -> schedule -> simulate path for a replay,
    used to compute the ground-truth divergence of a tampered file against
    the untampered ghost samples."""
    replay = validate_replay.Replay(path)
    track = validate_replay.resolve_track(replay)
    ghost = replay.ghosts[0]
    script, _ = validate_replay.ghost_script(ghost)
    ticks = int(ghost.race_time) // validate_replay.TICK_MS + validate_replay.FINISH_MARGIN_TICKS
    schedule = wr_replay.build_schedule(
        script, ticks, freeze_tick=int(ghost.race_time) // validate_replay.TICK_MS)
    _, trace = validate_replay.simulate(simulator, track, schedule, work)
    return schedule, trace


def first_schedule_difference(a: bytes, b: bytes) -> int:
    size = wr_replay.INPUT_STRUCT.size
    for tick in range(min(len(a), len(b)) // size):
        if a[tick * size:(tick + 1) * size] != b[tick * size:(tick + 1) * size]:
            return tick
    fail("schedules are identical")
    return -1


def first_sample_divergence(
    original: GhostBytes, trace: list[tuple[float, float, float]]
) -> int:
    """First ghost sample (as a race tick) where the tampered simulation
    leaves the untampered ghost's samples by more than the tolerance."""
    period = int(original.ghost.sample_period)
    for index in range(original.race_time // period + 1):
        tick = index * period // validate_replay.TICK_MS
        trace_index = max(tick - 1, 0)
        if trace_index >= len(trace):
            return tick
        sample = original.ghost.records[index].position
        if math.dist(trace[trace_index], (sample.x, sample.y, sample.z)) \
                > validate_replay.TRAJECTORY_TOLERANCE_M:
            return tick
    fail("tampered trajectory never leaves the ghost samples")
    return -1


# --- report -----------------------------------------------------------------

def table(rows: list[list[str]]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for index, row in enumerate(rows):
        lines.append("| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |")
        if index == 0:
            lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    return "\n".join(lines)


def row_for(name: str, run: Run, expectation: str, problems: list[str]) -> list[str]:
    verdict = run.verdict
    if verdict is None:
        return [name, "-", "-", "-", "unsupported", "-", "-", f"{run.wall_s:.3f}",
                "ok" if not problems else "; ".join(problems),
                run.error.replace("validate_replay: ", "")]
    checkpoints = verdict["checkpoints"]
    matched = sum(1 for c in checkpoints if c["match"])
    divergence = verdict["first_divergence_tick"]
    return [
        name,
        str(verdict["track"]["name"]),
        str(verdict["recorded_ms"]),
        str(verdict["simulated_ms"]),
        str(verdict["status"]),
        f"{matched}/{len(checkpoints)}",
        "-" if divergence is None else f"{divergence} ({divergence * 10} ms)",
        f"{run.wall_s:.3f}",
        "ok" if not problems else "; ".join(problems),
        f"max_dev={verdict['max_trajectory_deviation_m']:.4f} m, "
        f"decode={verdict['timing_s']['decode']:.3f} s, "
        f"sim={verdict['timing_s']['simulate']:.3f} s"
        + (f", expected {expectation}" if problems else ""),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", type=Path,
                        default=validate_replay.DEFAULT_SIMULATOR)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--report", type=Path, help="write the table here too")
    args = parser.parse_args()
    if not args.simulator.is_file():
        fail(f"simulator {args.simulator} is missing")
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="validate_replay_corpus_"))
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    failures = 0
    rows = [["replay", "track", "recorded ms", "simulated ms", "verdict",
             "checkpoints", "first divergence", "wall s", "check", "detail"]]

    for expectation in read_manifest():
        name = expectation.path.name
        run = Run(expectation.path, expectation.ghost, args.simulator,
                  work / f"{name}.json")
        problems = check_expectation(expectation, run)
        failures += bool(problems)
        rows.append(row_for(name, run, expectation.status, problems))

    # Invalid replay 1: one input tick edited in the A01 world record.
    a01 = GhostBytes(ROOT / "oracle/replays/A01-Race.tmx-12847157.Replay.Gbx")
    edited_path = work / "A01-Race.tmx-12847157.edited-tick.Replay.Gbx"
    edit_tick = fabricate_edit(a01, edit_ms=9100, output=edited_path)
    original_schedule, original_trace = schedule_for(a01.path, args.simulator, work)
    edited_schedule, edited_trace = schedule_for(edited_path, args.simulator, work)
    changed = [
        tick for tick in range(len(original_schedule) // wr_replay.INPUT_STRUCT.size)
        if original_schedule[tick * 72:(tick + 1) * 72] != edited_schedule[tick * 72:(tick + 1) * 72]
    ]
    if changed != [edit_tick]:
        fail(f"edit changed schedule ticks {changed}, expected [{edit_tick}]")
    expected_divergence = first_sample_divergence(a01, edited_trace)
    run = Run(edited_path, 0, args.simulator, work / f"{edited_path.name}.json")
    problems = []
    if run.status != "invalid" or run.exit_code != 1:
        problems.append(f"status {run.status} exit {run.exit_code}, expected invalid/1")
    if run.verdict is not None:
        if run.verdict["first_divergence_tick"] != expected_divergence:
            problems.append(f"first divergence {run.verdict['first_divergence_tick']}, "
                            f"expected {expected_divergence}")
        if run.verdict["first_divergence_tick"] is not None \
                and run.verdict["first_divergence_tick"] <= edit_tick:
            problems.append("divergence reported before the edited tick")
    failures += bool(problems)
    rows.append(row_for(
        f"{edited_path.name} (steer negated at schedule tick {edit_tick})",
        run, f"invalid, first divergence {expected_divergence}", problems))

    # Invalid replay 2: the A01 #1 ghost with the A01 #2 ghost's inputs from
    # 12 s on. The samples, checkpoint times and race time stay ghost one's.
    second = GhostBytes(ROOT / "oracle/replays/A01-Race.tmx-12699771.Replay.Gbx")
    spliced_path = work / "A01-Race.tmx-12847157+12699771.spliced.Replay.Gbx"
    fabricate_splice(a01, second, splice_ms=12000, output=spliced_path)
    spliced_schedule, spliced_trace = schedule_for(spliced_path, args.simulator, work)
    splice_tick = first_schedule_difference(original_schedule, spliced_schedule)
    expected_divergence = first_sample_divergence(a01, spliced_trace)
    run = Run(spliced_path, 0, args.simulator, work / f"{spliced_path.name}.json")
    problems = []
    if run.status != "invalid" or run.exit_code != 1:
        problems.append(f"status {run.status} exit {run.exit_code}, expected invalid/1")
    if run.verdict is not None:
        if run.verdict["first_divergence_tick"] != expected_divergence:
            problems.append(f"first divergence {run.verdict['first_divergence_tick']}, "
                            f"expected {expected_divergence}")
        if run.verdict["first_divergence_tick"] is not None \
                and run.verdict["first_divergence_tick"] <= splice_tick:
            problems.append("divergence reported before the splice")
        if run.verdict["simulated_ms"] == run.verdict["recorded_ms"] \
                and run.verdict["checkpoints_match"]:
            problems.append("spliced inputs reproduced the recorded times")
    failures += bool(problems)
    rows.append(row_for(
        f"{spliced_path.name} (inputs differ from schedule tick {splice_tick})",
        run, f"invalid, first divergence {expected_divergence}", problems))

    # Scripted marker: the A01 record re-timed the way TMInterface saves a
    # scripted run. The physics still reproduce; the verdict is "scripted".
    scripted_path = work / "A01-Race.tmx-12847157.scripted-clock.Replay.Gbx"
    fabricate_scripted(a01, scripted_path)
    run = Run(scripted_path, 0, args.simulator, work / f"{scripted_path.name}.json")
    problems = []
    if run.status != "scripted" or run.exit_code != 1:
        problems.append(f"status {run.status} exit {run.exit_code}, expected scripted/1")
    if run.verdict is not None and not (
        run.verdict["valid"] and run.verdict["scripted"]
        and run.verdict["simulated_ms"] == a01.race_time
        and run.verdict["first_divergence_tick"] is None
    ):
        problems.append("scripted replay did not reproduce")
    failures += bool(problems)
    rows.append(row_for(
        f"{scripted_path.name} (all input times + 0xFFFF ms)",
        run, "scripted, reproduced", problems))

    report = table(rows)
    print(report)
    print(f"\ncorpus: {len(rows) - 1} runs, {failures} failed expectation(s); "
          f"work dir {work}")
    if args.report is not None:
        args.report.write_text(report + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"validate_replay_corpus: {error}", file=sys.stderr)
        raise SystemExit(2)
