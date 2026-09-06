#!/usr/bin/env python3
"""Report static port and golden-trace coverage by original function VA."""

from __future__ import annotations

import csv
import pathlib
import re
import struct
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
VA_RE = re.compile(r"\b0x([0-9A-Fa-f]{8})(?![0-9A-Fa-f])")


def load_inventory(path: pathlib.Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        int(row["va"], 16): row
        for row in rows
        if not row["module"].startswith("LIBCMT:")
    }


def source_vas(suffixes: tuple[str, ...] = (".c", ".h")) -> dict[int, list[str]]:
    """VAs mentioned in sources.

    A VA appearing only in a header is usually a declaration comment, which
    means transcribed-and-declared, not implemented. Callers distinguish the
    two by passing suffixes.
    """
    found: dict[int, list[str]] = {}
    paths: list[pathlib.Path] = []
    for suffix in suffixes:
        paths += sorted((ROOT / "src").glob(f"*{suffix}"))
    for path in paths:
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            for match in VA_RE.finditer(line):
                va = int(match.group(1), 16)
                found.setdefault(va, []).append(
                    f"{path.relative_to(ROOT)}:{line_no}")
    return found


def trace_counts() -> tuple[dict[int, int], dict[int, int]]:
    """Per-VA record count and captured payload bytes.

    Record count alone is misleading: seven broad-phase traces hold records
    with zero buffers, so they are call-count logs that can never validate
    anything. Payload bytes is what makes a trace usable.
    """
    counts: dict[int, int] = {}
    payload: dict[int, int] = {}
    for path in sorted((ROOT / "oracle" / "traces").glob("*.bin")):
        data = path.read_bytes()
        if len(data) < 16:
            raise ValueError(f"short trace header: {path}")
        magic, va, count = struct.unpack_from("<8sII", data)
        if magic != b"TMNFTRC1":
            raise ValueError(f"bad trace magic: {path}")
        counts[va] = count
        off, total = 16, 0
        for _ in range(count):
            _seq, n_in, n_out = struct.unpack_from("<IHH", data, off)
            off += 8
            for _ in range(n_in + n_out):
                _tag, _addr, length = struct.unpack_from("<III", data, off)
                off += 12 + length
                total += length
        payload[va] = total
    return counts, payload


def row_name(row: dict[str, str]) -> str:
    return row.get("demangled_name") or row.get("name") or "<unnamed>"


# Third field is the VA a bare "N/M" result belongs to, for harnesses that
# print a label instead of an address.
HARNESSES = (
    ("replay_golden", ("oracle/traces",), None),
    ("replay_vehicle", ("oracle/traces",), None),
    ("replay_collision_response", ("oracle/traces",), None),
    ("replay_collision_detect", ("oracle/traces",), None),
    ("replay_broadphase", ("oracle/traces",), None),
    ("replay_model6",
     ("oracle/traces/007C3E80_CSceneVehicleCar_ComputeForcesModel6.bin",),
     0x007C3E80),
)

LABELLED_RESULT = re.compile(r"^\w+:\s*(\d+)\s*/\s*(\d+)\b")

# Fail closed: a VA is validated only when a harness reports records compared
# with zero mismatches. Mentioning a VA in a test proves nothing, and a harness
# that reports 0/0 or pass=N/A has verified nothing.
RESULT_PATTERNS = (
    re.compile(r"^OK\s+0x([0-9A-Fa-f]{8})\b"),
    re.compile(r"^0x([0-9A-Fa-f]{8}):\s+\d+ records, (\d+) matched, (\d+) fail"),
    re.compile(r"^([0-9A-Fa-f]{8}):\s*(\d+)\s*/\s*(\d+)\s*$"),
    # Broad-phase harness: "VA Name: pass=N/M replayable=...". A pass=N/A line
    # has no digits here and so correctly fails to match.
    re.compile(r"^([0-9A-Fa-f]{8})\b.*\bpass=(\d+)\s*/\s*(\d+)"),
)


def harness_vas() -> tuple[set[int], list[str]]:
    """VAs a harness actually verified, by running the harnesses.

    Returns the validated set plus any harnesses that could not be run, so the
    report can say so instead of silently under-reporting.
    """
    validated: set[int] = set()
    problems: list[str] = []
    for name, args, implicit_va in HARNESSES:
        binary = ROOT / "build" / "tests" / name
        if not binary.exists():
            problems.append(f"{name}: not built, its coverage is unknown")
            continue
        proc = subprocess.run(
            [str(binary), *args], cwd=ROOT, capture_output=True, text=True)
        for line in (proc.stdout + proc.stderr).splitlines():
            line = line.strip()
            if implicit_va is not None:
                labelled = LABELLED_RESULT.match(line)
                if labelled:
                    good, total = (int(g) for g in labelled.groups())
                    if good > 0 and good == total:
                        validated.add(implicit_va)
                    continue
            for pattern in RESULT_PATTERNS:
                match = pattern.match(line)
                if not match:
                    continue
                groups = match.groups()
                va = int(groups[0], 16)
                if len(groups) == 1:
                    validated.add(va)
                else:
                    good, other = int(groups[1]), int(groups[2])
                    # "N matched, M failed" and "N/M passed" both require a
                    # positive count and no shortfall.
                    if good > 0 and (other == 0 or other == good):
                        validated.add(va)
                break
    return validated, problems


def main() -> None:
    required = load_inventory(ROOT / "analysis" / "function_inventory.csv")
    required.update(load_inventory(ROOT / "analysis" / "vehicle_inventory.csv"))
    implemented = source_vas((".c",))
    declared = source_vas((".h",))
    traces, payload = trace_counts()
    replayed, harness_problems = harness_vas()

    req = set(required)
    impl = req & set(implemented)
    decl_only = (req & set(declared)) - impl
    with_trace = {va for va in req & set(traces) if traces[va] > 0}
    empty_trace = sorted(va for va in with_trace if payload.get(va, 0) == 0)
    usable_trace = with_trace - set(empty_trace)
    checked = with_trace & replayed
    captured_unchecked = sorted(usable_trace - replayed)
    missing = sorted(req - impl)

    out = ROOT / "analysis" / "port_coverage.md"
    with out.open("w") as f:
        f.write("# TMNF native port coverage\n\n")
        f.write("Generated by `tools/port_coverage.py`. These are three "
                "different things and are reported separately on purpose:\n"
                "transcription is not implementation, a trace file is not "
                "validation, and only a replayed trace is evidence.\n\n")
        f.write(f"- Required non-CRT functions: **{len(req)}**\n")
        f.write(f"- Implemented (VA appears in a `src/*.c`): "
                f"**{len(impl)}**\n")
        f.write(f"- Declared only in a header (not implemented): "
                f"**{len(decl_only)}**\n")
        f.write(f"- Have a trace file: **{len(with_trace)}**, of which "
                f"**{len(empty_trace)}** carry zero payload bytes and can "
                f"never validate anything\n")
        f.write(f"- **Verified against the game by a replay harness: "
                f"{len(checked)}**\n")
        f.write(f"- Have a usable trace that is not replayed: "
                f"**{len(captured_unchecked)}**\n")
        f.write(f"- Not implemented: **{len(missing)}**\n\n")
        f.write("Verified means a harness reported records compared with zero "
                "mismatches, measured by running the harnesses, not by "
                "grepping for the VA. A harness reporting `0/0` or `pass=N/A` "
                "counts as unverified.\n\n")
        for problem in harness_problems:
            f.write(f"> Warning: {problem}\n\n")

        f.write("## Traces that are call-count logs only\n\n")
        f.write("Records exist but hold no input or output buffers, so the "
                "call can neither be reconstructed nor compared. These need "
                "recapture with a deep graph spec.\n\n")
        for va in empty_trace:
            f.write(f"- `0x{va:08X}` {row_name(required[va])} "
                    f"— {traces[va]} empty records\n")

        f.write("\n## Usable trace, not replayed\n\n")
        for va in captured_unchecked:
            f.write(f"- `0x{va:08X}` {row_name(required[va])} "
                    f"— {traces[va]} records, {payload[va]} bytes unchecked\n")

        f.write("\n## Implemented without any golden trace\n\n")
        for va in sorted(impl - with_trace):
            f.write(f"- `0x{va:08X}` {row_name(required[va])}\n")

        f.write("\n## Declared but not implemented\n\n")
        for va in sorted(decl_only):
            f.write(f"- `0x{va:08X}` {row_name(required[va])}\n")

        f.write("\n## Not implemented\n\n")
        for va in missing:
            row = required[va]
            f.write(f"- `0x{va:08X}` {row_name(row)} — `{row['module']}`\n")

    print(f"required {len(req)} | implemented {len(impl)} | "
          f"traced {len(with_trace)} ({len(empty_trace)} empty) | "
          f"verified {len(checked)} | usable-unchecked "
          f"{len(captured_unchecked)}")
    for problem in harness_problems:
        print(f"warning: {problem}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
