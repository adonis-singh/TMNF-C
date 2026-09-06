#!/usr/bin/env python3
"""Drive tools/onboard_track.py over a list of campaign tracks in one lane.

    run      onboard every listed track, committing each result as it lands
    status   regenerate analysis/campaign_status.md from recorded outcomes
    note     append a "what broke" entry to the status report
    promote  re-run pending replays against HEAD and promote exact ones

Lanes share the repository. Commits are built with a private index and
`git commit-tree`, so unrelated staged or unstaged changes are excluded
and no trailer hook runs. Every git write is serialized on the `git` lock and
every build/asset write on the `shared` lock (see onboard_track.locked).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import onboard_track as onboarding  # noqa: E402
from track_env import ENVIRONMENTS, parse_track, united_tracks  # noqa: E402

ROOT = onboarding.ROOT
ONBOARDING = onboarding.ONBOARDING
LOGS = ONBOARDING
RESULTS = ONBOARDING / "results"
OUTCOMES = ONBOARDING / "outcomes"
NOTES = ONBOARDING / "notes.json"
SUITE = ONBOARDING / "suite.json"
SRC_TREE = ONBOARDING / "src_tree.txt"
STATUS_MD = ROOT / "analysis/campaign_status.md"
QUEUE_MD = ROOT / "analysis/onboarding_queue.md"
ASSET_DIR = ROOT / "viewer/assets/game"

SERIES = {"A": "White", "B": "Green", "C": "Blue", "D": "Red", "E": "Black"}
# Onboarded before this campaign pass; stats are read from their files.
PRE_EXISTING = ("A01-Race", "A08-Endurance", "A10-Acrobatic", "B04-Acrobatic",
                "C03-Acrobatic", "E01-Obstacle")

QUEUE_HEADER = (
    "# Onboarding divergence queue\n\n"
    "Captured references whose native replay is not byte-exact. Each row is a\n"
    "registered `pending:` replay in `oracle/tracks/manifest.txt` with its\n"
    "files under `oracle/results/pending/<id>/`.\n"
    "`tools/onboard_campaign_lane.py promote` re-runs every pending\n"
    "replay after a `src/` commit lands and moves exact ones back to\n"
    "`oracle/results/`.\n\n"
    "| Track | Capture | Exact prefix | First divergent field / word | "
    "Expected | Actual | Tick (race ms) | First suspect | Status |\n"
    "|---|---|---|---|---|---|---|---|---|\n"
)


NATIONS_LIST = ROOT / "oracle/tracks/nations_campaign.txt"


def campaign_tracks() -> list[str]:
    """The 65 Nations stems, from the committed list (the game install is only
    present on onboarding lanes)."""
    stems = [line.strip() for line in NATIONS_LIST.read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    if len(stems) != 65:
        onboarding.fail(f"expected 65 Nations challenges in {NATIONS_LIST}, found {len(stems)}")
    return stems


def united_campaign() -> dict[str, list[str]]:
    """Environment -> the 21 United Race stems (A1..D5, E)."""
    return {
        environment: [track.name for track in united_tracks(environment)]
        for environment in ENVIRONMENTS
    }


def known_tracks() -> set[str]:
    known = set(campaign_tracks())
    for stems in united_campaign().values():
        known.update(stems)
    return known


def series_of(track_name: str) -> str:
    return SERIES[track_name[0]]


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")


# --- git -------------------------------------------------------------------


def git(*arguments: str, check: bool = True, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments], cwd=cwd, check=check,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def git_retry(*arguments: str) -> None:
    """Run a git command that takes the shared index lock, retrying on it."""
    deadline = time.monotonic() + 600
    while True:
        result = git(*arguments, check=False)
        if result.returncode == 0:
            return
        if "index.lock" in result.stderr and time.monotonic() < deadline:
            time.sleep(2)
            continue
        onboarding.fail(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")


def pull_rebase_if_clean() -> None:
    """Rebase on origin/main when it moved and nothing else is dirty."""
    fetch = git("fetch", "origin", check=False)
    if fetch.returncode != 0:
        print(f"fetch failed, continuing locally: {fetch.stderr.strip()}", flush=True)
        return
    behind = git("rev-list", "--count", "HEAD..origin/main").stdout.strip()
    if behind == "0":
        return
    dirty = git("status", "--porcelain", "--untracked-files=no").stdout.strip()
    if dirty:
        print(f"origin/main is {behind} ahead but the tree has "
              "modifications; committing locally without rebase", flush=True)
        return
    result = git("rebase", "origin/main", check=False)
    if result.returncode != 0:
        git("rebase", "--abort", check=False)
        print(f"rebase onto origin/main failed, continuing locally: {result.stderr.strip()}",
              flush=True)


def commit_paths(message: str, paths: list[Path]) -> str:
    """Commit exactly these paths on top of HEAD using a private index."""
    relative = []
    for path in paths:
        if path.exists() or git("ls-files", "--error-unmatch", str(path), check=False).returncode == 0:
            relative.append(str(path.relative_to(ROOT)))
    index = ONBOARDING / f"index-{os.getpid()}"
    environment = os.environ.copy()
    environment["GIT_INDEX_FILE"] = str(index)
    for _attempt in range(20):
        parent = git("rev-parse", "HEAD").stdout.strip()
        if index.exists():
            index.unlink()
        subprocess.run(["git", "read-tree", parent], cwd=ROOT, check=True, env=environment)
        # -f: oracle/.gitignore ignores results/ while the references are tracked.
        subprocess.run(["git", "add", "-A", "-f", "--", *relative], cwd=ROOT, check=True,
                       env=environment)
        tree = subprocess.run(["git", "write-tree"], cwd=ROOT, check=True, env=environment,
                              stdout=subprocess.PIPE, text=True).stdout.strip()
        if tree == git("rev-parse", f"{parent}^{{tree}}").stdout.strip():
            index.unlink(missing_ok=True)
            print("nothing to commit", flush=True)
            return parent
        commit = subprocess.run(
            ["git", "commit-tree", tree, "-p", parent, "-m", message],
            cwd=ROOT, check=True, stdout=subprocess.PIPE, text=True,
            env={**os.environ, "GIT_INDEX_FILE": str(index)},
        ).stdout.strip()
        moved = git("update-ref", "-m", message, "HEAD", commit, parent, check=False)
        index.unlink(missing_ok=True)
        if moved.returncode == 0:
            # Refresh the shared index so the committed paths show clean.
            git_retry("add", "-A", "-f", "--", *relative)
            print(f"committed {commit[:7]} {message}", flush=True)
            return commit
        print("HEAD moved during commit; retrying", flush=True)
        time.sleep(1)
    onboarding.fail("could not commit: HEAD kept moving")
    return ""


# --- outcomes and status ---------------------------------------------------


def load_outcomes() -> dict[str, dict]:
    outcomes = {}
    if OUTCOMES.exists():
        for path in OUTCOMES.glob("*.json"):
            outcome = json.loads(path.read_text())
            outcomes[outcome["track_name"]] = outcome
    return outcomes


def save_outcome(outcome: dict) -> None:
    OUTCOMES.mkdir(parents=True, exist_ok=True)
    path = OUTCOMES / f"{outcome['track_name']}.json"
    path.write_text(json.dumps(outcome, indent=2) + "\n")


def load_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def registered_outcome(track_name: str) -> dict | None:
    """Reconstruct an outcome for a track that is already in the manifest."""
    records = [r for r in onboarding.manifest_records() if r[1] == track_name]
    if not records and track_name != "A01-Race":
        return None
    parsed = parse_track(track_name)
    track_id = parsed.id
    track = parsed.snapshot
    route = parsed.route
    if not track.is_file() or not route.is_file():
        return None
    replays = []
    if records:
        for name, exact in onboarding.replay_state(records[0][3]).items():
            reference, _ = onboarding.reference_paths(track_id, name, exact)
            total = reference.stat().st_size // onboarding.RECORD_SIZE
            replays.append({"name": name, "matched": total if exact else None,
                            "total": total, "exact": exact})
    else:
        for name, path in (("run1", "run1.bin"), ("long_drive", "a01_long_drive.bin"),
                           ("policy_lap", "policy_lap.bin")):
            reference = ROOT / "oracle/results" / path
            total = reference.stat().st_size // onboarding.RECORD_SIZE
            replays.append({"name": name, "matched": total, "total": total, "exact": True})
    status = "exact" if all(r["exact"] for r in replays) else "pending-divergence"
    return {
        "track_name": track_name, "track_id": track_id, "status": status,
        "track": onboarding.track_stats(track), "route": onboarding.route_stats(route),
        "replays": replays, "lane": "-", "minutes": None, "new_block_variants": None,
    }


def replay_cell(outcome: dict, name: str) -> str:
    for replay in outcome.get("replays", []):
        if replay["name"] == name:
            if replay.get("matched") is None:
                return f"pending/{replay['total']}"
            return f"{replay['matched']}/{replay['total']}"
    return "-"


def status_row(name: str, group: str, outcome: dict | None) -> tuple[str, str, int]:
    """One table row plus the outcome status and its byte-exact tick total."""
    if outcome is None:
        return (f"| {name} | {group} | not-started | - | - | - | - | - | - | - |",
                "not-started", 0)
    status = outcome["status"]
    exact_ticks = sum(
        replay["matched"] for replay in outcome.get("replays", [])
        if replay.get("matched") is not None
    )
    track = outcome.get("track") or {}
    route = outcome.get("route") or {}
    mixed = replay_cell(outcome, "mixed") if name != "A01-Race" else "run1 1000/1000"
    wall = replay_cell(outcome, "wall_contact") if name != "A01-Race" else "long_drive/policy_lap"
    entries = f"{track['entries']}/{track['surfaces']}" if track else "-"
    length = f"{route['length_m']:.1f}" if route else "-"
    variants = outcome.get("new_block_variants")
    minutes = outcome.get("minutes")
    detail = ""
    if status == "tooling-blocked":
        detail = " " + outcome.get("error", "").replace("|", "/")[:160]
    row = (
        f"| {name} | {group} | {status}{detail} | {mixed} | {wall} | {entries} | "
        f"{length} | {'-' if variants is None else variants} | {outcome.get('lane', '-')} | "
        f"{'-' if minutes is None else f'{minutes:.1f}'} |"
    )
    return row, status, exact_ticks


TABLE_HEADER = (
    "| Track | Series | Status | Mixed | Wall | Entries/Surfaces | Centerline m | "
    "New block variants | Lane | Minutes |\n"
    "|---|---|---|---|---|---|---|---|---|---|\n"
)


def united_section(outcomes: dict[str, dict]) -> str:
    text = (
        "## United Race campaign\n\n"
        "Seven environments x 21 tracks from\n"
        "`GameData/Tracks/Campaigns/United/Race/<Env>/` (names read from the\n"
        "installed United 2.11.26; UIDs are recorded in the track snapshots when\n"
        "captured). Ids are `<env>-<code>`, vehicles `<CODE>-<Env>.tmnfvehicle`.\n"
        "Lane: `tools/onboard_campaign_lane.py run --tracks DesertA1,... "
        "--wineprefix oracle/wineprefix_united --port 8495 --display 95`.\n\n"
    )
    for environment, stems in united_campaign().items():
        directory, collection = ENVIRONMENTS[environment]
        rows = []
        exact = 0
        for name in stems:
            row, status, _ = status_row(name, directory, outcomes.get(name))
            rows.append(row)
            exact += status == "exact"
        text += (
            f"### {directory} (collection `{collection}`): {exact}/21 byte-exact\n\n"
            + TABLE_HEADER + "\n".join(rows) + "\n\n"
        )
    return text


def write_status() -> None:
    tracks = campaign_tracks()
    outcomes = load_outcomes()
    for name in [*tracks, *(stem for stems in united_campaign().values() for stem in stems)]:
        outcome = outcomes.get(name)
        if outcome is None or outcome["status"] == "tooling-blocked":
            # The manifest is authoritative: a later sweep may have finished a
            # track whose lane recorded an earlier failure.
            reconstructed = registered_outcome(name)
            if reconstructed is not None:
                if outcome is not None:
                    reconstructed["lane"] = outcome.get("lane", "-")
                    reconstructed["minutes"] = outcome.get("minutes")
                outcomes[name] = reconstructed
        elif outcome["status"] == "pending-divergence":
            # Promotions land in the manifest first; mirror them here.
            records = [r for r in onboarding.manifest_records() if r[1] == name]
            if records:
                state = onboarding.replay_state(records[0][3])
                for replay in outcome["replays"]:
                    if state.get(replay["name"]) and not replay["exact"]:
                        replay["exact"] = True
                        replay["matched"] = replay["total"]
                if all(r["exact"] for r in outcome["replays"]):
                    outcome["status"] = "exact"
                save_outcome(outcome)
    suite = load_json(SUITE, {})
    notes = load_json(NOTES, [])

    rows = []
    exact_tracks = 0
    pending_tracks = 0
    blocked_tracks = 0
    exact_ticks = 0
    for name in tracks:
        row, status, ticks = status_row(name, series_of(name), outcomes.get(name))
        rows.append(row)
        exact_ticks += ticks
        if status == "exact":
            exact_tracks += 1
        elif status == "pending-divergence":
            pending_tracks += 1
        elif status == "tooling-blocked":
            blocked_tracks += 1

    tests = suite.get("summary", "not run yet")
    text = (
        "# Campaign onboarding status\n\n"
        f"Updated {now()}. Generated by `tools/onboard_campaign_lane.py status`;\n"
        "per-track logs are under `build/onboarding/<track>.log`.\n\n"
        "## Nations campaign totals\n\n"
        f"- Tracks byte-exact (both captures): {exact_tracks}/65\n"
        f"- Tracks with a pending divergence: {pending_tracks}\n"
        f"- Tracks tooling-blocked: {blocked_tracks}\n"
        f"- Tracks not started: {65 - exact_tracks - pending_tracks - blocked_tracks}\n"
        f"- Total byte-exact ticks across registered replays: {exact_ticks:,}\n"
        f"- Test suite: {tests}\n\n"
        "## Nations tracks (Stadium)\n\n"
        "Status: exact, pending-divergence (captures kept under\n"
        "`oracle/results/pending/`, see `analysis/onboarding_queue.md`),\n"
        "tooling-blocked, not-started. Mixed/wall are byte-exact prefix over\n"
        "capture length. Entries/surfaces are track snapshot counts. Centerline is\n"
        "the dense route length in metres. Minutes is wall-clock per track.\n\n"
        + TABLE_HEADER
        + "\n".join(rows) + "\n\n"
        + united_section(outcomes)
        + "## What broke and what was applied\n\n"
        + ("".join(f"- {note}\n" for note in notes) if notes else "- nothing yet\n")
    )
    STATUS_MD.write_text(text)


def add_note(text: str) -> None:
    notes = load_json(NOTES, [])
    notes.append(f"{now()}: {text}")
    ONBOARDING.mkdir(parents=True, exist_ok=True)
    NOTES.write_text(json.dumps(notes, indent=2) + "\n")


def queue_rows() -> list[str]:
    if not QUEUE_MD.exists():
        return []
    return [line for line in QUEUE_MD.read_text().splitlines()
            if line.startswith("| ") and not line.startswith("| Track")]


def write_queue(rows: list[str]) -> None:
    QUEUE_MD.write_text(QUEUE_HEADER + "".join(row + "\n" for row in rows))


def queue_row(track_name: str, replay: dict) -> str:
    mismatches = replay.get("mismatches", [])
    first = mismatches[0] if mismatches else f"{replay['field']} +{replay['word']}"
    return (
        f"| {track_name} | {replay['name']} | {replay['matched']}/{replay['total']} | "
        f"{replay['field']} word {replay['word']} ({first}) | {replay['expected']} | "
        f"{replay['actual']} | {replay['tick']} ({replay['race_time_ms']}) | "
        f"{replay['suspect']}; wheel materials at tick "
        f"{'/'.join(replay.get('wheel_materials_at_tick', []))}, touched so far "
        f"{replay.get('materials_touched_before', [])} | open |"
    )


# --- test suite -------------------------------------------------------------


def run_suite(label: str, fast: bool) -> str:
    """Build everything from HEAD in the private tree and run ctest."""
    onboarding.configure_and_build(all_targets=True)
    arguments = ["ctest", "--test-dir", onboarding.BUILD, "-j2", "--output-on-failure"]
    if fast:
        arguments.extend(["-E", "python_platform"])
    result = onboarding.capture(onboarding.taskset(arguments),
                                env=onboarding.clean_environment(), timeout=1800)
    # ctest prints "100% tests passed out of N" when nothing failed and
    # "P% tests passed, F tests failed out of N" otherwise.
    summary = re.search(r"(\d+)% tests passed(?:, (\d+) tests failed)? out of (\d+)", result.stdout)
    if summary is None:
        onboarding.fail(f"ctest produced no summary: {result.stderr[-2000:]}")
    failed = int(summary.group(2) or 0)
    total = int(summary.group(3))
    failing = re.findall(r"^\s*\d+ - (\S+) \(", result.stdout, re.MULTILINE)
    pending = set()
    labels = onboarding.capture(
        ["ctest", "--test-dir", onboarding.BUILD, "-N", "-L", "pending_divergence"],
        env=onboarding.clean_environment(), timeout=120)
    pending.update(re.findall(r"Test\s+#\d+: (\S+)", labels.stdout))
    baseline = set(load_json(ONBOARDING / "baseline_failures.json", []))
    unexpected = [name for name in failing if name not in pending and name not in baseline]
    text = (
        f"{total - failed}/{total} passed at {git('rev-parse', '--short', 'HEAD').stdout.strip()}"
        f" ({label}{', python_platform excluded' if fast else ''}); "
        f"{len(pending)} registered pending replays"
        + (f", failing: {', '.join(failing)}" if failing else "")
        + (f" (pre-existing at baseline: {', '.join(sorted(baseline & set(failing)))})"
           if baseline & set(failing) else "")
        + (f"; UNEXPECTED failures: {', '.join(unexpected)}" if unexpected else "")
    )
    SUITE.write_text(json.dumps({"summary": text, "failed": failing, "unexpected": unexpected,
                                 "time": now()}, indent=2) + "\n")
    print(f"suite: {text}", flush=True)
    return text


# --- promotion after physics fixes ------------------------------------------


def pending_entries() -> list[tuple[str, str, str, str]]:
    entries = []
    for track_id, track_name, sha256, text in onboarding.manifest_records():
        for name, exact in onboarding.replay_state(text).items():
            if not exact:
                entries.append((track_id, track_name, sha256, name))
    return entries


def promote_pending(force: bool) -> list[str]:
    """Re-run pending replays against HEAD; move exact ones out of pending.

    Caller holds the git lock. Returns the promoted track names.
    """
    head_src = git("rev-parse", "HEAD:src").stdout.strip()
    previous = SRC_TREE.read_text().strip() if SRC_TREE.exists() else ""
    entries = pending_entries()
    if not entries or (head_src == previous and not force):
        SRC_TREE.write_text(head_src + "\n")
        return []
    print(f"src/ changed ({previous[:7]} -> {head_src[:7]}); re-running "
          f"{len(entries)} pending replays", flush=True)
    promoted = []
    promoted_paths: list[Path] = []
    rows = queue_rows()
    with onboarding.locked("shared"):
        onboarding.configure_and_build()
        for track_id, track_name, sha256, name in entries:
            parsed = parse_track(track_name)
            track, vehicle = parsed.snapshot, parsed.vehicle
            reference, inputs = onboarding.reference_paths(track_id, name, False)
            result = onboarding.capture(onboarding.taskset([
                onboarding.REPLAY_TICK, track, vehicle, reference, "input_file", inputs, sha256,
            ]), env=onboarding.clean_environment(), timeout=300)
            summary = onboarding.SUMMARY_RE.search(result.stdout)
            if summary is None:
                print(f"{track_id}_{name}: replay did not run", flush=True)
                continue
            matched, total = int(summary.group(1)), int(summary.group(2))
            outcomes = load_outcomes()
            outcome = outcomes.get(track_name)
            if outcome is not None:
                for replay in outcome["replays"]:
                    if replay["name"] == name:
                        replay["matched"] = matched
            if matched != total:
                print(f"{track_id}_{name}: still {matched}/{total}", flush=True)
                if outcome is not None:
                    save_outcome(outcome)
                continue
            exact_reference, exact_inputs = onboarding.reference_paths(track_id, name, True)
            reference.replace(exact_reference)
            inputs.replace(exact_inputs)
            promoted_paths.extend((exact_reference, exact_inputs, reference.parent))
            if not any(reference.parent.iterdir()):
                reference.parent.rmdir()
            onboarding.register_track(track_id, track_name, sha256, {name: True})
            rows = [
                row.replace("| open |", f"| fixed at {head_src[:7]} |")
                if row.startswith(f"| {track_name} | {name} |") else row
                for row in rows
            ]
            if outcome is not None:
                for replay in outcome["replays"]:
                    if replay["name"] == name:
                        replay["exact"] = True
                if all(r["exact"] for r in outcome["replays"]):
                    outcome["status"] = "exact"
                save_outcome(outcome)
            promoted.append(f"{track_name} {name}")
            print(f"{track_id}_{name}: now {matched}/{total}, promoted", flush=True)
    SRC_TREE.write_text(head_src + "\n")
    if promoted:
        write_queue(rows)
        write_status()
        commit_paths(
            "Promote " + ", ".join(promoted) + " after physics fix",
            [*promoted_paths, ROOT / "oracle/tracks/manifest.txt", QUEUE_MD, STATUS_MD],
        )
    return promoted


# --- lane loop -------------------------------------------------------------


FAILURE_PATTERNS = (
    r"^tmnf (track|route|vehicle|world)[^:]*: .*",
    r"^generate_route_centerline: .*",
    r"^extract_game_assets: .*",
    r"^\w*Error: .*",
    r"timed out after \d+ seconds",
    r"^onboard_track: .*",
)


def classify_failure(log_text: str) -> str:
    """Pick the most specific diagnostic from the end of a track log."""
    tail = [line.strip() for line in log_text.splitlines() if line.strip()][-60:]
    for pattern in FAILURE_PATTERNS:
        for line in reversed(tail):
            match = re.search(pattern, line)
            if match:
                return match.group(0)[:300]
    return tail[-1] if tail else "no output"


def captures_complete(track_name: str) -> bool:
    parsed = parse_track(track_name)
    track_id = parsed.id
    paths = [parsed.snapshot, parsed.route, parsed.vehicle]
    for name in ("mixed", "wall_contact"):
        exact = onboarding.reference_paths(track_id, name, True)
        pending = onboarding.reference_paths(track_id, name, False)
        if not (all(p.is_file() for p in exact) or all(p.is_file() for p in pending)):
            return False
    return all(path.is_file() for path in paths)


def partial_exists(track_name: str) -> bool:
    parsed = parse_track(track_name)
    track_id = parsed.id
    candidates = [
        parsed.snapshot, parsed.route, parsed.vehicle,
        ROOT / f"viewer/scenes/{track_id}_mixed.json",
        ROOT / f"oracle/results/{track_id}_mixed_inputs.bin",
        ROOT / f"oracle/results/{track_id}_wall_contact_inputs.bin",
        ROOT / "oracle/results/pending" / track_id,
    ]
    return any(path.exists() for path in candidates)


def onboard_one(track_name: str, args: argparse.Namespace, mode: str) -> dict:
    """mode: fresh, force (recapture), reuse (post-process existing captures)."""
    LOGS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{track_name}.log"
    result_path = RESULTS / f"{track_name}.json"
    result_path.unlink(missing_ok=True)
    command = [
        sys.executable, ROOT / "tools/onboard_track.py", track_name,
        "--wineprefix", args.wineprefix, "--port", str(args.port),
        "--display", args.display, "--cpu-set", args.cpu_set,
        "--result", result_path,
    ]
    if mode == "force":
        command.append("--force")
    elif mode == "reuse":
        command.append("--reuse-captures")
    started = time.monotonic()
    with log_path.open("ab") as log:
        log.write(f"\n=== {now()} lane {args.lane} {' '.join(map(str, command))}\n".encode())
        log.flush()
        process = subprocess.run([str(c) for c in command], cwd=ROOT, stdout=log,
                                 stderr=subprocess.STDOUT, check=False, timeout=3 * 3600)
    minutes = (time.monotonic() - started) / 60
    if process.returncode != 0 or not result_path.exists():
        return {
            "track_name": track_name, "track_id": parse_track(track_name).id,
            "status": "tooling-blocked", "lane": args.lane, "minutes": round(minutes, 1),
            "error": classify_failure(log_path.read_text(errors="replace")),
            "replays": [], "time": now(),
        }
    result = json.loads(result_path.read_text())
    result.update({"lane": args.lane, "minutes": round(minutes, 1), "time": now()})
    return result


def record_and_commit(outcome: dict) -> None:
    """Persist the outcome, update reports, and commit under the git lock."""
    track_name = outcome["track_name"]
    track_id = outcome["track_id"]
    with onboarding.locked("git"):
        pull_rebase_if_clean()
        save_outcome(outcome)
        if outcome["status"] in ("exact", "pending-divergence"):
            rows = queue_rows()
            for replay in outcome["replays"]:
                if not replay["exact"]:
                    rows = [row for row in rows
                            if not row.startswith(f"| {track_name} | {replay['name']} |")]
                    rows.append(queue_row(track_name, replay))
            write_queue(rows)
            write_status()
            paths = [ROOT / relative for relative in outcome["files"]]
            paths.extend([
                ROOT / "oracle/results/pending" / track_id,
                ASSET_DIR, ROOT / f"viewer/scenes/{track_id}_mixed.json",
                QUEUE_MD, STATUS_MD,
            ])
            with onboarding.locked("shared"):
                commit_paths(f"Onboard {track_name}", paths)
        else:
            write_status()
            commit_paths(f"Record {track_name} onboarding failure", [STATUS_MD])
        promote_pending(force=False)


def run_lane(args: argparse.Namespace) -> int:
    onboarding.set_lane(args.wineprefix, args.port, args.display, args.cpu_set)
    deadline = time.monotonic() + args.hours * 3600
    tracks = args.tracks.split(",")
    tracks = [parse_track(track).name for track in tracks]
    known = known_tracks()
    unknown = [track for track in tracks if track not in known]
    if unknown:
        onboarding.fail(f"unknown campaign tracks: {unknown}")
    if (any(parse_track(track).campaign == "united" for track in tracks)
            and onboarding.layout().flavor != "united"):
        onboarding.fail(f"United tracks need a United prefix, lane has {onboarding.PREFIX}")
    not_reached = []
    onboarded = []
    stop_file = ONBOARDING / f"stop-lane{args.lane}"
    for track_name in tracks:
        if stop_file.exists():
            print(f"lane {args.lane}: stop file present, exiting before {track_name}", flush=True)
            not_reached.append(track_name)
            continue
        if time.monotonic() > deadline:
            not_reached.append(track_name)
            continue
        registered = [r for r in onboarding.manifest_records() if r[1] == track_name]
        if registered and not args.redo_blocked:
            print(f"lane {args.lane}: {track_name} already registered, skipping", flush=True)
            continue
        outcomes = load_outcomes()
        previous = outcomes.get(track_name)
        if previous is not None and previous["status"] != "tooling-blocked":
            print(f"lane {args.lane}: {track_name} already {previous['status']}, skipping",
                  flush=True)
            continue
        mode = "fresh"
        if args.recapture:
            mode = "force"
        elif captures_complete(track_name):
            mode = "reuse"
        elif previous is not None or partial_exists(track_name):
            mode = "force"
        print(f"lane {args.lane}: onboarding {track_name} ({mode})", flush=True)
        outcome = onboard_one(track_name, args, mode)
        if outcome["status"] == "tooling-blocked":
            print(f"lane {args.lane}: {track_name} failed: {outcome['error']}", flush=True)
            if args.retry_blocked and time.monotonic() < deadline:
                mode = "reuse" if captures_complete(track_name) else "force"
                print(f"lane {args.lane}: retrying {track_name} once ({mode})", flush=True)
                retry = onboard_one(track_name, args, mode)
                if retry["status"] != "tooling-blocked":
                    outcome = retry
                else:
                    outcome["error"] = f"{outcome['error']} / retry: {retry['error']}"
                    outcome["minutes"] += retry["minutes"]
        record_and_commit(outcome)
        onboarded.append(track_name)
        print(f"lane {args.lane}: {track_name} -> {outcome['status']} "
              f"in {outcome['minutes']:.1f} min", flush=True)
    if onboarded:
        # One suite for the whole run: a full ctest costs about as much as an
        # onboarding, so per track it doubled the lane's cadence.
        with onboarding.locked("git"):
            with onboarding.locked("shared"):
                run_suite(f"after {len(onboarded)} tracks, {onboarded[0]}..{onboarded[-1]}",
                          fast=True)
            write_status()
            commit_paths("Record lane suite result", [STATUS_MD])
    if not_reached:
        print(f"lane {args.lane}: deadline reached, not started: {', '.join(not_reached)}",
              flush=True)
    print(f"lane {args.lane}: done", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--lane", required=True)
    run.add_argument("--tracks", required=True,
                     help="comma-separated stems: A08-Endurance, DesertA1 or desert/A1")
    run.add_argument("--wineprefix", type=Path, required=True)
    run.add_argument("--port", type=int, required=True)
    run.add_argument("--display", required=True)
    run.add_argument("--cpu-set", required=True)
    run.add_argument("--hours", type=float, default=7.0)
    run.add_argument("--retry-blocked", action="store_true")
    run.add_argument("--redo-blocked", action="store_true",
                     help="re-run tracks recorded as tooling-blocked")
    run.add_argument("--recapture", action="store_true",
                     help="always recapture from the game (--force)")
    record = commands.add_parser("record",
                                 help="commit a finished onboard_track result JSON")
    record.add_argument("track_name")
    record.add_argument("--lane", required=True)
    record.add_argument("--minutes", type=float, required=True)
    commands.add_parser("status")
    note = commands.add_parser("note")
    note.add_argument("text")
    suite = commands.add_parser("suite")
    suite.add_argument("--full", action="store_true")
    suite.add_argument("--label", default="manual")
    promote = commands.add_parser("promote")
    promote.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        return run_lane(args)
    if args.command == "record":
        result_path = RESULTS / f"{args.track_name}.json"
        if not result_path.is_file():
            onboarding.fail(f"no result for {args.track_name}")
        outcome = json.loads(result_path.read_text())
        outcome.update({"lane": args.lane, "minutes": args.minutes, "time": now()})
        record_and_commit(outcome)
        return 0
    if args.command == "status":
        write_status()
        return 0
    if args.command == "note":
        add_note(args.text)
        write_status()
        return 0
    if args.command == "suite":
        with onboarding.locked("shared"):
            run_suite(args.label, fast=not args.full)
        write_status()
        return 0
    if args.command == "promote":
        with onboarding.locked("git"):
            promoted = promote_pending(force=args.force)
        print(f"promoted: {promoted}")
        return 0
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f"onboard_campaign_lane: {error}", file=sys.stderr)
        raise SystemExit(1)
