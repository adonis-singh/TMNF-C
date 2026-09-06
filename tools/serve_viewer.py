#!/usr/bin/env python3
"""Serve the TMNF-C viewer together with the training run registry.

Routes:
  /             static files from viewer/ (index.html, viewer.js, scenes/, assets/)
  /runs/...     static files from build/runs/ (run.json, replays/, evals/, metrics.csv)
  /api/runs     build/runs/index.json joined with every run.json, plus a
                server-evaluated `live` flag per run

Every JSON response carries `Cache-Control: no-store`. A run is live iff its
status is `running` and its heartbeat is at most LIVE_HEARTBEAT_SECONDS old,
measured against this server's clock so browser clock skew is irrelevant.
Timestamps must carry a UTC offset; a naive timestamp makes the run an error
card, matching the writer, so the two sides cannot disagree about liveness.

No directory listings are served, symlinks that resolve outside the served
directory are 404, and a null byte in the path is 400.

Development helpers (never point them at the real build/runs/):
  --write-fixtures [--with-attacks]   create fixture runs in an empty or
                                      fixture-only --runs-dir
  --self-test                         validate the registry reader against the
                                      run contract in a temporary directory

Standard library only.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import tempfile
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

LIVE_HEARTBEAT_SECONDS = 30.0
RUN_STATUSES = frozenset({"running", "finished", "failed"})
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
REPLAY_SCENE_PATTERN = re.compile(r"^replays/[A-Za-z0-9_.-]+\.json(\.gz)?$")
NONE_TYPE = type(None)
RUN_FIELDS: dict[str, tuple[type, ...]] = {
    "run_id": (str,),
    "track_id": (str,),
    "track_name": (str,),
    "algorithm": (str,),
    "started_at": (str,),
    "finished_at": (str, NONE_TYPE),
    "status": (str,),
    "heartbeat_at": (str,),
    "git_commit": (str,),
    "physics_sha256": (str,),
    "seed": (int,),
    "args": (dict,),
    "spectate_url": (str, NONE_TYPE),
    "summary": (dict,),
    "replays": (list,),
}
# summary is {} until the trainer finishes its first update; once a key is
# present it must have the contract type.
SUMMARY_FIELDS: dict[str, tuple[type, ...]] = {
    "updates": (int,),
    "wall_time_s": (int, float),
    "physics_steps": (int,),
    "finishes": (int,),
    "best_lap_ms": (int, NONE_TYPE),
}
REPLAY_FIELDS: dict[str, tuple[type, ...]] = {
    "label": (str,),
    "lap_ms": (int, NONE_TYPE),
    "scene": (str,),
    "minute": (int, float),
}


class RegistryError(Exception):
    """The run registry violates the contract or is mid-rewrite."""


def read_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise RegistryError(f"{path.name} is missing") from None
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise RegistryError(
            f"{path.name} is not valid JSON ({error.msg} at byte {error.pos})"
        ) from None


def parse_iso(value: str, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise RegistryError(f"{name} is not an ISO 8601 timestamp: {value!r}")
    if parsed.tzinfo is None:
        raise RegistryError(
            f"{name} {value!r} has no UTC offset; the writer emits +00:00 and "
            "liveness is not guessed from a naive timestamp"
        )
    return parsed


def check_fields(
    obj: Any, spec: dict[str, tuple[type, ...]], name: str, *, required: bool = True
) -> None:
    if not isinstance(obj, dict):
        raise RegistryError(f"{name} must be a JSON object")
    missing = [key for key in spec if key not in obj]
    if missing and required:
        raise RegistryError(f"{name} is missing fields: {', '.join(missing)}")
    wrong = [
        key
        for key, types in spec.items()
        if key in obj and (not isinstance(obj[key], types) or isinstance(obj[key], bool))
    ]
    if wrong:
        raise RegistryError(f"{name} has wrongly typed fields: {', '.join(wrong)}")


def validate_run(run: Any, run_id: str) -> None:
    check_fields(run, RUN_FIELDS, "run.json")
    if run["run_id"] != run_id:
        raise RegistryError(
            f"run.json run_id {run['run_id']!r} does not match directory {run_id!r}"
        )
    if run["status"] not in RUN_STATUSES:
        raise RegistryError(f"run.json status {run['status']!r} is not running|finished|failed")
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", run["track_id"]):
        raise RegistryError(f"run.json track_id {run['track_id']!r} is not a valid scene key")
    parse_iso(run["started_at"], "started_at")
    parse_iso(run["heartbeat_at"], "heartbeat_at")
    if run["finished_at"] is not None:
        parse_iso(run["finished_at"], "finished_at")
    check_fields(run["summary"], SUMMARY_FIELDS, "run.json summary", required=False)
    for index, replay in enumerate(run["replays"]):
        check_fields(replay, REPLAY_FIELDS, f"run.json replays[{index}]")
        if not REPLAY_SCENE_PATTERN.match(replay["scene"]):
            raise RegistryError(
                f"run.json replays[{index}].scene {replay['scene']!r} "
                "must look like replays/<file>.json"
            )


def heartbeat_age(run: dict[str, Any], now: datetime) -> float:
    return (now - parse_iso(run["heartbeat_at"], "heartbeat_at")).total_seconds()


def load_registry(runs_dir: Path, now: datetime) -> dict[str, Any]:
    """Join index.json with each run.json. Raises RegistryError only when the
    index itself is unreadable; per-run problems become `error` entries."""
    index = read_json(runs_dir / "index.json")
    if not isinstance(index, dict) or not isinstance(index.get("runs"), list):
        raise RegistryError('index.json must be {"runs": [...]}')
    runs: list[dict[str, Any]] = []
    for position, entry in enumerate(index["runs"]):
        run_id = entry.get("run_id") if isinstance(entry, dict) else None
        if not isinstance(run_id, str) or not RUN_ID_PATTERN.match(run_id):
            runs.append({
                "run_id": f"index[{position}]",
                "live": False,
                "error": f"index.json runs[{position}] has no valid run_id",
            })
            continue
        card: dict[str, Any] = {
            key: entry[key]
            for key in ("track_id", "track_name", "algorithm", "status", "started_at")
            if key in entry
        }
        card["run_id"] = run_id
        try:
            run = read_json(runs_dir / run_id / "run.json")
            validate_run(run, run_id)
            age = heartbeat_age(run, now)
            card = {
                **run,
                "live": run["status"] == "running" and age <= LIVE_HEARTBEAT_SECONDS,
                "heartbeat_age_s": round(age, 1),
                "error": None,
            }
        except RegistryError as error:
            card["live"] = False
            card["error"] = f"{run_id}: {error}"
        runs.append(card)
    return {
        "runs": runs,
        "now": now.isoformat(timespec="seconds"),
        "live_heartbeat_s": LIVE_HEARTBEAT_SECONDS,
    }


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    viewer_dir: Path
    runs_dir: Path
    verbose: bool = False


class ViewerHandler(SimpleHTTPRequestHandler):
    server: ViewerServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if urlsplit(self.path).path == "/api/runs":
            self._api_runs()
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        if urlsplit(self.path).path == "/api/runs":
            self.send_error(HTTPStatus.METHOD_NOT_ALLOWED)
            return
        super().do_HEAD()

    def translate_path(self, path: str) -> str:
        clean = urlsplit(path).path
        if clean == "/runs" or clean.startswith("/runs/"):
            self.directory = str(self.server.runs_dir)
            return super().translate_path(clean[len("/runs"):] or "/")
        self.directory = str(self.server.viewer_dir)
        return super().translate_path(clean)

    def send_head(self):  # type: ignore[override]
        if "\x00" in unquote(urlsplit(self.path).path):
            self.send_error(HTTPStatus.BAD_REQUEST, "Null byte in path")
            return None
        # translate_path confines the *lexical* path to the served directory;
        # a symlink inside it could still point anywhere, so confine the
        # resolved path as well.
        target = Path(os.path.realpath(self.translate_path(self.path)))
        root = Path(self.directory)
        if target != root and root not in target.parents:
            self.send_error(HTTPStatus.NOT_FOUND)
            return None
        return super().send_head()

    def list_directory(self, path: str):  # type: ignore[override]
        # Directory listings would expose checkpoints, temp files and the
        # whole registry layout; only files are served.
        self.send_error(HTTPStatus.NOT_FOUND)
        return None

    def end_headers(self) -> None:
        clean = urlsplit(self.path).path
        if clean.endswith((".json", ".json.gz", ".csv")) or clean.startswith("/api/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _api_runs(self) -> None:
        try:
            payload = load_registry(self.server.runs_dir, datetime.now(timezone.utc))
            status = HTTPStatus.OK
        except RegistryError as error:
            payload = {"error": str(error)}
            status = HTTPStatus.SERVICE_UNAVAILABLE
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        if self.server.verbose:
            super().log_message(format, *args)


def make_server(bind: str, port: int, viewer_dir: Path, runs_dir: Path, verbose: bool) -> ViewerServer:
    server = ViewerServer((bind, port), ViewerHandler)
    server.viewer_dir = viewer_dir.resolve()
    server.runs_dir = runs_dir.resolve()
    server.verbose = verbose
    return server


# --- fixtures ----------------------------------------------------------------


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def stretch_scene(data: dict[str, Any], factor: float) -> dict[str, Any]:
    """Return a copy of a viewer scene whose lap takes `factor` times longer.
    Development fixture only: the telemetry is resampled, not re-simulated."""
    lap = data["lap"]
    ticks = lap["ticks"]
    count = int(round(len(ticks) * factor))
    stretched = []
    for index in range(count):
        record = list(ticks[min(int(index / factor), len(ticks) - 1)])
        record[0] = (index + 1) * lap["tickMs"]
        stretched.append(record)

    def scale(tick: int | None) -> int | None:
        return None if tick is None else min(int(round(tick * factor)), count - 1)

    finish_tick = scale(lap["finishTick"])
    return {
        **data,
        "lap": {
            **lap,
            "ticks": stretched,
            "tickCount": count,
            "checkpointTicks": [scale(tick) for tick in lap["checkpointTicks"]],
            "finishTick": finish_tick,
            "finishTimeMs": 0 if finish_tick is None else (finish_tick + 1) * lap["tickMs"],
        },
    }


FIXTURE_INDEX_FORMAT = "tmnf-c-viewer-fixtures"


def write_json(path: Path, payload: Any) -> None:
    """Temp file in the same directory, fsync, rename: a concurrent reader
    sees either the old file or the new one, never a prefix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def swap_route_checkpoints(data: dict[str, Any]) -> dict[str, Any]:
    """Same track, same lap, re-ordered route file: checkpoints 0 and 1 (and
    the lap's crossing ticks) swapped. A ghost built from this must be
    refused because its splits would pair the wrong gates."""
    route = data["route"]
    lap = data["lap"]
    if len(route["checkpoints"]) < 2:
        raise SystemExit("swap_route_checkpoints needs a route with two checkpoints")
    checkpoints = list(route["checkpoints"])
    checkpoints[0], checkpoints[1] = checkpoints[1], checkpoints[0]
    ticks = list(lap["checkpointTicks"])
    ticks[0], ticks[1] = ticks[1], ticks[0]
    return {
        **data,
        "route": {**route, "checkpoints": checkpoints},
        "lap": {**lap, "checkpointTicks": ticks},
    }


def base_run(run_id: str, track_id: str, track_name: str, started: datetime) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "track_id": track_id,
        "track_name": track_name,
        "algorithm": "ppo",
        "started_at": iso(started),
        "finished_at": None,
        "status": "running",
        "heartbeat_at": iso(started),
        "git_commit": "fixture0000000000000000000000000000000000",
        "physics_sha256": "f" * 64,
        "seed": 1,
        "args": {"track": track_id, "seed": 1, "num_envs": 4096, "fixture": True},
        "spectate_url": None,
        "summary": {
            "updates": 0,
            "wall_time_s": 0.0,
            "physics_steps": 0,
            "finishes": 0,
            "best_lap_ms": None,
            "distance_mean_100": 0.0,
            "eval_fullstart": None,
        },
        "replays": [],
    }


def index_entry(run_dir: Path) -> dict[str, Any]:
    """Index summary for one run directory; unreadable run.json still lists
    the run_id so the reader reports it as an error card."""
    try:
        run = read_json(run_dir / "run.json")
        if not isinstance(run, dict):
            raise RegistryError("run.json must be an object")
    except RegistryError:
        return {"run_id": run_dir.name}
    entry = {
        key: run[key]
        for key in ("track_id", "algorithm", "status", "started_at")
        if key in run
    }
    entry["run_id"] = run_dir.name
    return entry


def rebuild_index(runs_dir: Path, extra_entries: list[dict[str, Any]]) -> list[str]:
    """Rewrite index.json from every <run_id>/run.json present (plus any
    deliberately dangling entries), under the writer's index.lock and with
    the writer's temp+rename, so a reader never sees a partial index."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    with (runs_dir / "index.lock").open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            entries = [
                index_entry(path.parent)
                for path in sorted(runs_dir.glob("*/run.json"))
                if not path.parent.name.startswith(".")
            ] + extra_entries
            entries.sort(key=lambda entry: entry.get("started_at", ""), reverse=True)
            write_json(runs_dir / "index.json", {
                "format": FIXTURE_INDEX_FORMAT,
                "version": 1,
                "generated_at": iso(datetime.now(timezone.utc)),
                "runs": entries,
            })
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return [entry["run_id"] for entry in entries]


def refuse_real_registry(runs_dir: Path) -> None:
    """The fixture writer is development-only. It rewrites index.json with
    slim entries and leaves a fake `running` run that the trainer's reaper
    never collects, so it must never touch a registry the writer owns."""
    index_path = runs_dir / "index.json"
    if not index_path.exists():
        return
    try:
        index = read_json(index_path)
    except RegistryError:
        index = None
    if isinstance(index, dict) and index.get("format") == FIXTURE_INDEX_FORMAT:
        return
    raise SystemExit(
        f"refusing to write fixtures: {index_path} is not a fixture index. "
        "Point --runs-dir at an empty scratch directory (headless_check.mjs "
        "does this for you)."
    )


def write_fixtures(runs_dir: Path, scene_path: Path, now: datetime, with_attacks: bool) -> list[str]:
    refuse_real_registry(runs_dir)
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    if scene.get("format") != "tmnf-c-viewer-scene" or scene.get("version") != 3:
        raise SystemExit(f"{scene_path} is not a version 3 viewer scene")
    track_id = scene["gameVisuals"]["scene"]
    best_lap_ms = scene["lap"]["finishTimeMs"]
    slower = stretch_scene(scene, 1.08)
    extra_entries: list[dict[str, Any]] = []

    def register(run: dict[str, Any]) -> None:
        write_json(runs_dir / run["run_id"] / "run.json", run)

    finished_id = f"fixture_{track_id}_finished"
    finished_dir = runs_dir / finished_id
    started = now - timedelta(hours=3)
    finished = base_run(finished_id, track_id, "A01-Race fixture", started)
    finished.update({
        "finished_at": iso(started + timedelta(minutes=60)),
        "status": "finished",
        "heartbeat_at": iso(started + timedelta(minutes=60)),
        "summary": {
            **finished["summary"],
            "updates": 480,
            "wall_time_s": 3600.0,
            "physics_steps": 1_966_080_000,
            "finishes": 12_408,
            "best_lap_ms": best_lap_ms,
            "distance_mean_100": 611.4,
            "eval_fullstart": {"finish_rate": 0.98, "lap_ms": best_lap_ms},
        },
        "replays": [
            {
                "label": "first finish",
                "lap_ms": int(slower["lap"]["finishTimeMs"]),
                "scene": "replays/first_finish.json",
                "minute": 14,
            },
            {
                "label": "best lap",
                "lap_ms": best_lap_ms,
                "scene": "replays/best_lap.json",
                "minute": 52,
            },
        ],
    })
    register(finished)
    (finished_dir / "replays").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(scene_path, finished_dir / "replays" / "best_lap.json")
    write_json(finished_dir / "replays" / "first_finish.json", slower)
    write_json(finished_dir / "evals" / "0014.json", {"minute": 14, "finish_rate": 0.31})
    write_json(finished_dir / "evals" / "0052.json", {"minute": 52, "finish_rate": 0.98})
    (finished_dir / "metrics.csv").write_text(
        "update,wall_time_s,finishes,best_lap_ms\n"
        f"1,7.5,0,\n480,3600.0,12408,{best_lap_ms}\n",
        encoding="utf-8",
    )

    stale_id = "fixture_a10_stale_live"
    stale = base_run(stale_id, "a10", "A10-Acrobatic fixture", now - timedelta(minutes=25))
    stale.update({
        "heartbeat_at": iso(now - timedelta(minutes=10)),
        "spectate_url": "http://127.0.0.1:8765",
        "summary": {
            **stale["summary"],
            "updates": 120,
            "wall_time_s": 900.0,
            "physics_steps": 491_520_000,
            "finishes": 37,
            "best_lap_ms": 10_190,
            "distance_mean_100": 188.0,
        },
    })
    register(stale)

    if with_attacks:
        broken_id = "attack_missing_fields"
        write_json(runs_dir / broken_id / "run.json", {
            "run_id": broken_id,
            "track_id": "a01",
            "status": "finished",
        })

        wrong_id = "attack_wrong_scene_version"
        wrong = base_run(wrong_id, track_id, "A01-Race fixture", now - timedelta(hours=1))
        wrong.update({
            "status": "failed",
            "finished_at": iso(now - timedelta(minutes=50)),
            "heartbeat_at": iso(now - timedelta(minutes=50)),
            "replays": [{
                "label": "obsolete exporter",
                "lap_ms": best_lap_ms,
                "scene": "replays/old_version.json",
                "minute": 3,
            }],
        })
        register(wrong)
        write_json(runs_dir / wrong_id / "replays" / "old_version.json", {**scene, "version": 2})

        # Same track, different route file: checkpoints re-ordered. Listed as
        # a finished run so it can be picked as a ghost from the panel.
        route_id = "attack_route_version"
        swapped = swap_route_checkpoints(scene)
        routed = base_run(route_id, track_id, "A01-Race fixture", now - timedelta(hours=2))
        routed.update({
            "status": "finished",
            "finished_at": iso(now - timedelta(minutes=100)),
            "heartbeat_at": iso(now - timedelta(minutes=100)),
            "summary": {**routed["summary"], "updates": 90, "finishes": 3, "best_lap_ms": best_lap_ms},
            "replays": [{
                "label": "re-ordered route",
                "lap_ms": best_lap_ms,
                "scene": "replays/swapped_route.json",
                "minute": 41,
            }],
        })
        register(routed)
        write_json(runs_dir / route_id / "replays" / "swapped_route.json", swapped)

        # Heartbeat without a UTC offset: the writer never emits one and
        # raises on reading one, so the reader must refuse it too.
        naive_id = "attack_naive_timestamp"
        naive = base_run(naive_id, "e01", "E01 fixture", now - timedelta(minutes=1))
        naive["heartbeat_at"] = (now - timedelta(seconds=5)).replace(tzinfo=None).isoformat(timespec="seconds")
        register(naive)

        extra_entries.append({"run_id": "attack_dangling_index_entry"})

    return rebuild_index(runs_dir, extra_entries)


# --- self-test ---------------------------------------------------------------


def synthetic_scene() -> dict[str, Any]:
    fields = [
        "raceTimeMs", "x", "y", "z", "qx", "qy", "qz", "qw", "speedMps", "rpm", "gear",
        "contactMask", "slidingMask", "inputSteer",
        "wheelSteerFL", "wheelSteerFR", "wheelSteerRL", "wheelSteerRR",
        "wheelDamperFL", "wheelDamperFR", "wheelDamperRL", "wheelDamperRR",
        "wheelSpeedFL", "wheelSpeedFR", "wheelSpeedRL", "wheelSpeedRR",
    ]
    ticks = [[(i + 1) * 10, float(i)] + [0.0] * (len(fields) - 2) for i in range(50)]
    return {
        "format": "tmnf-c-viewer-scene",
        "version": 3,
        "gameVisuals": {"manifest": "assets/game/manifest.json", "scene": "a01"},
        "track": {},
        "route": {
            "length": 98.0,
            "centerlineCount": 50,
            "centerline": [[float(i * 2), 0.0, 0.0, 8.0] for i in range(50)],
            "checkpoints": [
                {"raceIndex": 0, "transform": {"translation": [40, 0, 0]}},
                {"raceIndex": 1, "transform": {"translation": [70, 0, 0]}},
            ],
            "finish": {"transform": {"translation": [98, 0, 0]}},
        },
        "car": {},
        "lap": {
            "tickMs": 10, "fields": fields, "ticks": ticks, "tickCount": 50,
            "finishTimeMs": 400, "checkpointTicks": [20, 30], "finishTick": 39,
        },
    }


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"self-test FAILED: {message}")


def http_get(url: str) -> tuple[int, dict[str, str], bytes]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def self_test() -> None:
    now = datetime(2026, 9, 1, 20, 0, 0, tzinfo=timezone.utc)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        viewer_dir = root / "viewer"
        runs_dir = root / "runs"
        scene_path = viewer_dir / "scenes" / "policy_lap.json"
        write_json(scene_path, synthetic_scene())
        (viewer_dir / "index.html").write_text("<!doctype html>viewer", encoding="utf-8")

        run_ids = write_fixtures(runs_dir, scene_path, now, with_attacks=True)
        expect(len(run_ids) == 7, f"expected 7 fixture runs, got {run_ids}")
        # Re-running against a fixture index is fine; a real index is refused.
        write_fixtures(runs_dir, scene_path, now, with_attacks=True)
        real_dir = root / "real_runs"
        write_json(real_dir / "index.json", {"format": "tmnf-rl-run-index", "version": 1, "runs": []})
        try:
            write_fixtures(real_dir, scene_path, now, with_attacks=False)
        except SystemExit as error:
            expect("refusing" in str(error), "fixture writer refuses a real index")
        else:
            expect(False, "fixture writer must refuse a real index")
        expect(not list(runs_dir.glob("*.tmp")), "no temp files left behind")

        fresh_id = "fresh_live"
        fresh = base_run(fresh_id, "b04", "B04 fixture", now - timedelta(minutes=2))
        fresh["heartbeat_at"] = iso(now - timedelta(seconds=12))
        fresh["spectate_url"] = "http://127.0.0.1:8766"
        fresh["summary"] = {}
        write_json(runs_dir / fresh_id / "run.json", fresh)
        index = read_json(runs_dir / "index.json")
        expect(index["format"] == FIXTURE_INDEX_FORMAT, "fixture index is marked as such")
        expect([entry["run_id"] for entry in index["runs"]] == run_ids, "fixture index lists fixtures")
        index["runs"] += [{"run_id": fresh_id}, {"track_id": "no id"}]
        write_json(runs_dir / "index.json", index)

        registry = load_registry(runs_dir, now)
        by_id = {run["run_id"]: run for run in registry["runs"]}
        expect(len(registry["runs"]) == 9, f"expected 9 cards, got {len(registry['runs'])}")
        expect(by_id["fixture_a01_finished"]["live"] is False, "finished run must not be live")
        expect(by_id["fixture_a01_finished"]["error"] is None, "finished fixture must validate")
        expect(len(by_id["fixture_a01_finished"]["replays"]) == 2, "finished fixture has two replays")
        expect(by_id["fixture_a10_stale_live"]["live"] is False, "stale heartbeat must not be live")
        expect(by_id["fixture_a10_stale_live"]["error"] is None, "stale fixture must validate")
        expect(by_id["fixture_a10_stale_live"]["heartbeat_age_s"] == 600.0, "stale age is 600 s")
        expect(by_id[fresh_id]["live"] is True, "12 s old running heartbeat is live")
        expect(by_id[fresh_id]["error"] is None, "empty summary before the first update is valid")
        naive_card = by_id["attack_naive_timestamp"]
        expect(naive_card["live"] is False, "naive heartbeat must not be live")
        expect("no UTC offset" in naive_card["error"], f"naive timestamp is an error card: {naive_card['error']}")
        expect("missing fields" in by_id["attack_missing_fields"]["error"], "missing fields reported")
        expect("track_name" in by_id["attack_missing_fields"]["error"], "missing field named")
        expect(by_id["attack_missing_fields"]["track_id"] == "a01", "index summary kept on error card")
        expect("run.json is missing" in by_id["attack_dangling_index_entry"]["error"], "dangling entry")
        expect("no valid run_id" in by_id["index[8]"]["error"], "index entry without run_id")
        expect(by_id["attack_wrong_scene_version"]["error"] is None, "wrong scene version is a viewer-side error")
        expect(by_id["attack_route_version"]["error"] is None, "re-ordered route is a viewer-side error")
        swapped = read_json(runs_dir / "attack_route_version" / "replays" / "swapped_route.json")
        expect(swapped["route"]["checkpoints"][0]["raceIndex"] == 1, "swapped fixture re-orders checkpoints")
        expect(swapped["lap"]["checkpointTicks"] == [30, 20], "swapped fixture re-orders crossing ticks")

        later = {run["run_id"]: run for run in load_registry(runs_dir, now + timedelta(seconds=40))["runs"]}
        expect(later[fresh_id]["live"] is False, "heartbeat older than 30 s must drop live")

        write_json(runs_dir / fresh_id / "run.json", {**fresh, "status": "crashed"})
        again = {run["run_id"]: run for run in load_registry(runs_dir, now)["runs"]}
        expect("running|finished|failed" in again[fresh_id]["error"], "invalid status reported")
        write_json(runs_dir / fresh_id / "run.json", {**fresh, "summary": {"updates": "40"}})
        again = {run["run_id"]: run for run in load_registry(runs_dir, now)["runs"]}
        expect("wrongly typed fields: updates" in again[fresh_id]["error"], "summary types enforced")
        write_json(runs_dir / fresh_id / "run.json", fresh)

        stretched = stretch_scene(synthetic_scene(), 1.08)
        expect(stretched["lap"]["tickCount"] == 54, "stretch resamples tick count")
        expect(stretched["lap"]["ticks"][-1][0] == 540, "stretch rewrites race time")
        expect(stretched["lap"]["finishTick"] == 42, "stretch scales finish tick")

        server = make_server("127.0.0.1", 0, viewer_dir, runs_dir, verbose=False)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            code, headers, body = http_get(f"{base}/api/runs")
            expect(code == 200, f"/api/runs returned {code}")
            expect(headers.get("Cache-Control") == "no-store", "/api/runs must be no-store")
            payload = json.loads(body)
            expect(len(payload["runs"]) == 9 and "now" in payload, "/api/runs payload shape")
            expect(payload["live_heartbeat_s"] == LIVE_HEARTBEAT_SECONDS, "threshold published")

            code, headers, body = http_get(f"{base}/runs/fixture_a01_finished/run.json")
            expect(code == 200 and headers.get("Cache-Control") == "no-store", "run.json served no-store")
            expect(json.loads(body)["run_id"] == "fixture_a01_finished", "run.json content")
            code, _, body = http_get(f"{base}/runs/fixture_a01_finished/replays/first_finish.json")
            expect(code == 200 and json.loads(body)["lap"]["tickCount"] == 54, "replay scene served")
            code, _, _ = http_get(f"{base}/runs/does_not_exist/run.json")
            expect(code == 404, "missing run is 404")
            code, _, body = http_get(f"{base}/")
            expect(code == 200 and b"viewer" in body, "viewer index served at /")
            code, headers, body = http_get(f"{base}/scenes/policy_lap.json")
            expect(code == 200 and headers.get("Cache-Control") == "no-store", "scene JSON no-store")
            code, _, body = http_get(f"{base}/runs/..%2f..%2fviewer/index.html")
            expect(code == 404, f"traversal out of runs must fail, got {code}")
            code, _, _ = http_get(f"{base}/runs/fixture_a01_finished/%00")
            expect(code == 400, f"null byte in path must be 400, got {code}")
            code, _, _ = http_get(f"{base}/runs/fixture_a01_finished/run.json%00.txt")
            expect(code == 400, f"embedded null byte must be 400, got {code}")
            for listing in ("/runs/", "/runs/fixture_a01_finished/", "/runs/fixture_a01_finished/replays/", "/scenes/"):
                code, _, body = http_get(f"{base}{listing}")
                expect(code == 404, f"directory listing {listing} must be 404, got {code}")
            (runs_dir / "escape").symlink_to(root, target_is_directory=True)
            (runs_dir / "fixture_a01_finished" / "leak.json").symlink_to(viewer_dir / "index.html")
            code, _, _ = http_get(f"{base}/runs/escape/viewer/index.html")
            expect(code == 404, f"symlinked directory out of runs must be 404, got {code}")
            code, _, _ = http_get(f"{base}/runs/fixture_a01_finished/leak.json")
            expect(code == 404, f"symlinked file out of runs must be 404, got {code}")
            (runs_dir / "fixture_a01_finished" / "inside.json").symlink_to(
                runs_dir / "fixture_a01_finished" / "run.json"
            )
            code, _, body = http_get(f"{base}/runs/fixture_a01_finished/inside.json")
            expect(code == 200 and json.loads(body)["run_id"] == "fixture_a01_finished", "symlink inside runs is fine")

            index_path = runs_dir / "index.json"
            full = index_path.read_text(encoding="utf-8")
            index_path.write_text(full[: len(full) // 2], encoding="utf-8")
            code, headers, body = http_get(f"{base}/api/runs")
            expect(code == 503, f"partial index.json must be 503, got {code}")
            expect("not valid JSON" in json.loads(body)["error"], "partial index error text")
            expect(headers.get("Cache-Control") == "no-store", "503 is no-store")
            index_path.write_text(full, encoding="utf-8")
            code, _, _ = http_get(f"{base}/api/runs")
            expect(code == 200, "registry recovers after rewrite completes")
        finally:
            server.shutdown()
            server.server_close()
    print("self-test OK")


# --- main ----------------------------------------------------------------------


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8801)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--viewer-dir", type=Path, default=repo / "viewer")
    parser.add_argument("--runs-dir", type=Path, default=repo / "build" / "runs")
    parser.add_argument("--verbose", action="store_true", help="log every request")
    parser.add_argument("--write-fixtures", action="store_true",
                        help="write fixture runs into --runs-dir and exit")
    parser.add_argument("--with-attacks", action="store_true",
                        help="with --write-fixtures: also write deliberately broken runs")
    parser.add_argument("--self-test", action="store_true",
                        help="validate the registry reader against the contract and exit")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.write_fixtures:
        run_ids = write_fixtures(
            args.runs_dir,
            args.viewer_dir / "scenes" / "policy_lap.json",
            datetime.now(timezone.utc),
            args.with_attacks,
        )
        print(f"wrote {len(run_ids)} fixture runs to {args.runs_dir}: {', '.join(run_ids)}")
        return

    if not (args.viewer_dir / "index.html").is_file():
        raise SystemExit(f"{args.viewer_dir} does not contain the viewer")
    server = make_server(args.bind, args.port, args.viewer_dir, args.runs_dir, args.verbose)
    print(f"viewer  http://{args.bind}:{args.port}/")
    print(f"runs    {server.runs_dir}  (/runs/, /api/runs)")
    if not server.runs_dir.is_dir():
        print("        run registry directory does not exist yet; /api/runs returns 503 until it does")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
