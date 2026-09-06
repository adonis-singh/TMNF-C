"""Run registry under build/runs: per-run run.json files plus a rebuilt index.

Contract (shared with the viewer):

* ``<root>/index.json`` = ``{"runs": [<run.json content>, ...]}``, newest
  first, written atomically (temp + rename) under an exclusive lock on
  ``<root>/index.lock``. Each entry also carries ``stale`` (running but the
  heartbeat is older than ``STALE_HEARTBEAT_SECONDS``).
* ``<root>/<run_id>/run.json`` is owned by exactly one trainer process and is
  also written atomically. Concurrent trainers never write each other's
  run.json; they only race on the index, which the lock serialises and the
  rebuild-from-files design makes idempotent.
* A run whose process died without writing ``finished``/``failed`` is
  detected at the next index rebuild (heartbeat stale and pid dead on this
  host) and its run.json is rewritten with ``status: failed``.
"""

from __future__ import annotations

import fcntl
import os
import platform
import secrets
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tmnf_rl.tracks import project_root
from tmnf_rl.utils import parse_iso, read_json, utc_now_iso, write_json_atomic


STALE_HEARTBEAT_SECONDS = 60.0
HEARTBEAT_INTERVAL_SECONDS = 5.0
STATUSES = ("running", "finished", "failed")
REQUIRED_KEYS = (
    "run_id",
    "track_id",
    "track_name",
    "algorithm",
    "started_at",
    "finished_at",
    "status",
    "heartbeat_at",
    "git_commit",
    "physics_sha256",
    "seed",
    "args",
    "spectate_url",
    "summary",
    "replays",
)


def default_runs_root() -> Path:
    return project_root() / "build" / "runs"


def infer_trainer(command_line: list[str]) -> str:
    """Name the trainer of a run.json written before the `trainer` field (F26).

    Our own runs recorded `python/tmnf_rl/train.py` (what `-m tmnf_rl.train`
    puts in argv[0]) or the thin `python/train_ppo.py` CLI; anything else is
    a foreign writer and is labelled with its entry point.
    """
    if len(command_line) < 2:
        return "unknown"
    entry = Path(command_line[1])
    if entry.name == "train_ppo.py" or (entry.name == "train.py" and entry.parent.name == "tmnf_rl"):
        return "tmnf_rl.train"
    return f"unknown:{entry.name}"


def pid_start_ticks(pid: int) -> int | None:
    """Field 22 of /proc/<pid>/stat: process start time in clock ticks since
    boot. With the pid it identifies a process even after pid reuse (F28)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # The command name can contain spaces and parentheses; split after it.
    return int(stat[stat.rindex(")") + 2 :].split()[19])


def _pid_alive(pid: int, start_ticks: int | None) -> bool:
    """Whether the recorded process (pid plus start time) still exists.

    A reused pid has a different start time and counts as dead; a pid we may
    not signal (PermissionError) is still checked through /proc, which is
    world-readable; only a run that recorded no start time falls back to the
    signal check alone.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    if start_ticks is None:
        return True
    return pid_start_ticks(pid) == start_ticks


def validate_run_payload(payload: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(f"run.json is missing keys {missing}")
    if payload["status"] not in STATUSES:
        raise ValueError(f"run.json status {payload['status']!r} is invalid")
    if not isinstance(payload["replays"], list):
        raise ValueError("run.json replays must be a list")
    for replay in payload["replays"]:
        for key in ("label", "lap_ms", "scene", "minute"):
            if key not in replay:
                raise ValueError(f"replay entry is missing {key!r}")
    parse_iso(payload["started_at"])
    parse_iso(payload["heartbeat_at"])
    if payload["finished_at"] is not None:
        parse_iso(payload["finished_at"])


class RunRecord:
    """In-memory mirror of one run.json with atomic, locked writes."""

    def __init__(self, path: Path, payload: dict[str, Any]) -> None:
        self.path = Path(path)
        self.payload = payload
        self._lock = threading.Lock()

    @property
    def run_id(self) -> str:
        return str(self.payload["run_id"])

    @property
    def run_dir(self) -> Path:
        return self.path.parent

    def write(self) -> None:
        with self._lock:
            validate_run_payload(self.payload)
            write_json_atomic(self.path, self.payload)

    def update(self, **changes: Any) -> None:
        with self._lock:
            self.payload.update(changes)
            self.payload["heartbeat_at"] = utc_now_iso()
            validate_run_payload(self.payload)
            write_json_atomic(self.path, self.payload)

    def stage(self, **changes: Any) -> None:
        """Update the in-memory payload; the next heartbeat writes it."""
        with self._lock:
            self.payload.update(changes)

    def append_replay(self, replay: dict[str, Any]) -> None:
        with self._lock:
            self.payload["replays"] = [*self.payload["replays"], replay]
            self.payload["heartbeat_at"] = utc_now_iso()
            validate_run_payload(self.payload)
            write_json_atomic(self.path, self.payload)

    def heartbeat(self) -> None:
        self.update()

    def finish(self, status: str, **changes: Any) -> None:
        if status not in ("finished", "failed"):
            raise ValueError("finish status must be finished or failed")
        self.update(status=status, finished_at=utc_now_iso(), **changes)


class RunLockError(RuntimeError):
    pass


class RunLock:
    """Exclusive ``flock`` on ``<run_dir>/run.lock`` for a trainer's lifetime.

    Held from before environment creation until the process exits; the kernel
    drops it when the holder dies (SIGKILL included), so there is no stale
    lock to clean up. A second trainer on the same run fails fast with the
    holder's pid instead of interleaving writes (F23).
    """

    def __init__(self, run_dir: Path) -> None:
        self.path = Path(run_dir) / "run.lock"
        self._file = None

    def acquire(self) -> "RunLock":
        file = self.path.open("a+")
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # The holder writes its identity right after taking the lock and
            # blanks it before releasing; between flock and that write the
            # file is empty (or, in the same window, would name a previous
            # holder), so wait briefly for the current holder's line.
            holder = ""
            deadline = time.monotonic() + 2.0
            while not holder and time.monotonic() < deadline:
                file.seek(0)
                holder = file.read().strip()
                if not holder:
                    time.sleep(0.01)
            file.close()
            raise RunLockError(
                f"run {self.path.parent.name!r} is locked by another trainer "
                f"({holder}); refusing to start a second process on it"
            ) from None
        file.seek(0)
        file.truncate()
        file.write(f"pid {os.getpid()} since {utc_now_iso()} on {platform.node()}\n")
        file.flush()
        os.fsync(file.fileno())
        self._file = file
        return self

    def release(self) -> None:
        if self._file is not None:
            self._file.seek(0)
            self._file.truncate()
            self._file.flush()
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            self._file.close()
            self._file = None

    def abandon(self) -> None:
        """Release a claim that never got a run.json: remove the lock file and
        the directory if nothing else was written, so the run id is reusable."""
        self.release()
        self.path.unlink(missing_ok=True)
        try:
            self.path.parent.rmdir()
        except OSError:
            pass


class RunRegistry:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_runs_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.lock_path = self.root / "index.lock"

    def run_dir(self, run_id: str) -> Path:
        if not run_id or "/" in run_id or run_id.startswith("."):
            raise ValueError(f"invalid run id {run_id!r}")
        return self.root / run_id

    @staticmethod
    def generate_run_id(track_id: str, algorithm: str, seed: int) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"{stamp}_{track_id}_{algorithm}_s{seed}_{secrets.token_hex(2)}"

    def claim_run(self, run_id: str) -> RunLock:
        """Create the run directory (it must not exist) and lock it.

        The trainer calls this before building environments so a duplicate
        ``--run-id`` fails in milliseconds, not after seconds of setup.
        """
        run_dir = self.run_dir(run_id)
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise FileExistsError(
                f"run directory already exists: {run_dir}"
            ) from None
        return RunLock(run_dir).acquire()

    def lock_run(self, run_id: str) -> RunLock:
        """Lock an existing run directory (resume)."""
        run_dir = self.run_dir(run_id)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run {run_id!r} has no directory at {run_dir}")
        return RunLock(run_dir).acquire()

    def create_run(self, payload: dict[str, Any]) -> RunRecord:
        """Write the initial run.json into a claimed (or fresh) run directory."""
        run_dir = self.run_dir(payload["run_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "run.json"
        if path.exists():
            raise FileExistsError(f"run.json already exists: {path}")
        record = RunRecord(path, payload)
        record.write()
        self.rebuild_index()
        return record

    def open_run(self, run_id: str) -> RunRecord:
        path = self.run_dir(run_id) / "run.json"
        if not path.is_file():
            raise FileNotFoundError(f"run {run_id!r} has no run.json at {path}")
        payload = read_json(path)
        validate_run_payload(payload)
        return RunRecord(path, payload)

    def list_run_dirs(self) -> list[Path]:
        return sorted(
            path.parent
            for path in self.root.glob("*/run.json")
            if not path.parent.name.startswith(".")
        )

    def _reap_if_dead(self, payload: dict[str, Any], now: datetime) -> bool:
        if payload["status"] != "running":
            return False
        age = (now - parse_iso(payload["heartbeat_at"])).total_seconds()
        # A heartbeat in the future is a clock jump, not a live run; it is as
        # suspicious as a stale one and goes on to the pid check (F28).
        if 0.0 <= age < STALE_HEARTBEAT_SECONDS:
            return False
        if payload.get("hostname") != platform.node():
            return False
        pid = payload.get("pid")
        start_ticks = payload.get("pid_start_ticks")
        if not isinstance(pid, int) or _pid_alive(pid, start_ticks if isinstance(start_ticks, int) else None):
            return False
        payload["status"] = "failed"
        payload["finished_at"] = payload["heartbeat_at"] if age >= 0.0 else utc_now_iso()
        payload["failure"] = (
            f"heartbeat lost: pid {pid} is dead and the last heartbeat was "
            f"{age:.0f} s old when the index was rebuilt"
        )
        return True

    def mark_failed(self, run_id: str, *, reason: str, force: bool) -> dict[str, Any]:
        """Operator override for a run parked in `running` (F28).

        Without ``force`` the run must look dead from here: stale heartbeat
        and no live process on this host. ``force`` skips that check for the
        cases the reaper cannot decide (another host, unreadable pid).
        """
        record = self.open_run(run_id)
        payload = record.payload
        if payload["status"] != "running":
            raise RuntimeError(f"run {run_id!r} is {payload['status']}, not running")
        if not force:
            age = (datetime.now(timezone.utc) - parse_iso(payload["heartbeat_at"])).total_seconds()
            pid = payload.get("pid")
            start_ticks = payload.get("pid_start_ticks")
            alive = (
                payload.get("hostname") == platform.node()
                and isinstance(pid, int)
                and _pid_alive(pid, start_ticks if isinstance(start_ticks, int) else None)
            )
            if 0.0 <= age < STALE_HEARTBEAT_SECONDS or alive:
                raise RuntimeError(
                    f"run {run_id!r} looks alive (heartbeat {age:.0f} s old, pid {pid} "
                    f"{'alive' if alive else 'not checkable'}); pass --force to override"
                )
        record.finish(
            "failed",
            failure=f"marked failed by operator ({'forced' if force else 'dead'}): {reason}",
        )
        self.rebuild_index()
        return record.payload

    def rebuild_index(self) -> dict[str, Any]:
        """Scan every run.json under the lock and rewrite index.json."""
        with self.lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                now = datetime.now(timezone.utc)
                entries: list[dict[str, Any]] = []
                for run_dir in self.list_run_dirs():
                    path = run_dir / "run.json"
                    # One malformed run.json must never take the whole index
                    # (or the heartbeat thread rebuilding it) down with it, so
                    # every per-run failure becomes an error entry.
                    try:
                        payload = read_json(path)
                        validate_run_payload(payload)
                        if self._reap_if_dead(payload, now):
                            write_json_atomic(path, payload)
                        if "trainer" not in payload:
                            # Foreign writers (F26) predate the field; name the
                            # entry point they recorded so the index shows what
                            # produced the run instead of an empty column.
                            command = (payload.get("provenance") or {}).get("command_line") or []
                            payload["trainer"] = infer_trainer(command)
                        age = (now - parse_iso(payload["heartbeat_at"])).total_seconds()
                        payload["stale"] = (
                            payload["status"] == "running"
                            and age >= STALE_HEARTBEAT_SECONDS
                        )
                    except Exception as error:  # noqa: BLE001
                        entries.append(
                            {
                                "run_id": run_dir.name,
                                "status": "failed",
                                "invalid": True,
                                "failure": (
                                    f"malformed run.json: {type(error).__name__}: {error}"
                                ),
                            }
                        )
                        continue
                    entries.append(payload)
                entries.sort(key=lambda entry: entry.get("started_at", ""), reverse=True)
                index = {
                    "format": "tmnf-rl-run-index",
                    "version": 1,
                    "generated_at": utc_now_iso(),
                    "runs": entries,
                }
                write_json_atomic(self.index_path, index)
                return index
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def read_index(self) -> dict[str, Any]:
        if not self.index_path.is_file():
            return self.rebuild_index()
        return read_json(self.index_path)


class HeartbeatThread(threading.Thread):
    """Refresh heartbeat_at and the index every few seconds while training."""

    def __init__(
        self,
        record: RunRecord,
        registry: RunRegistry,
        interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        super().__init__(name="tmnf-run-heartbeat", daemon=True)
        self.record = record
        self.registry = registry
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self.beats = 0
        self.rebuild_failures = 0
        self.heartbeat_failures = 0
        self.last_error: str | None = None

    def run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            # Neither a failing heartbeat write (disk full, F27) nor a failing
            # index rebuild (another run's broken file, F19) may end this
            # thread: a dead heartbeat thread makes a healthy run look dead
            # and hides the real cause. Both are logged and retried next beat;
            # the trainer's own writes surface a full disk as the run failure.
            try:
                self.record.heartbeat()
                self.beats += 1
            except Exception as error:  # noqa: BLE001
                self.heartbeat_failures += 1
                self.last_error = f"{type(error).__name__}: {error}"
                print(
                    f"tmnf_rl.registry: heartbeat write failed ({self.last_error}); "
                    "heartbeat continues",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            try:
                self.registry.rebuild_index()
            except Exception as error:  # noqa: BLE001
                self.rebuild_failures += 1
                self.last_error = f"{type(error).__name__}: {error}"
                print(
                    f"tmnf_rl.registry: index rebuild failed ({self.last_error}); "
                    "heartbeat continues",
                    file=sys.stderr,
                    flush=True,
                )

    def close(self) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join()


def new_run_payload(
    *,
    run_id: str,
    track_id: str,
    track_name: str,
    algorithm: str,
    seed: int,
    args: dict[str, Any],
    provenance: dict[str, Any],
    spectate_url: str | None,
    run_dir: Path,
    trainer: str = "tmnf_rl.train",
) -> dict[str, Any]:
    now = utc_now_iso()
    return {
        "format": "tmnf-rl-run",
        "version": 1,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "track_id": track_id,
        "track_name": track_name,
        "algorithm": algorithm,
        "started_at": now,
        "finished_at": None,
        "status": "running",
        "heartbeat_at": now,
        "git_commit": provenance["git_commit"],
        "git_dirty": provenance["git_dirty"],
        "physics_sha256": provenance["physics_sha256"],
        "code_sha256": provenance["code_sha256"],
        # The hash producing the newest rows; differs from code_sha256 only
        # after a --resume --ignore-code-hash, which also appends to
        # code_changes (F25).
        "code_sha256_current": provenance["code_sha256"],
        "code_changes": [],
        "seed": seed,
        "args": args,
        "provenance": provenance,
        "spectate_url": spectate_url,
        # Which program wrote this record: foreign trainers (the NN ladder's
        # ppo_pilot.py) share the format with different args, metrics, GPU and
        # physics, and the index shows this column so they are never read as
        # tmnf_rl.train runs (F26).
        "trainer": trainer,
        "pid": os.getpid(),
        "pid_start_ticks": pid_start_ticks(os.getpid()),
        "hostname": platform.node(),
        "summary": {},
        "replays": [],
        "checkpoint": None,
        "resumes": [],
        "failure": None,
        "metrics": "metrics.csv",
        "evaluations": "evaluations.csv",
    }


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="tmnf_rl.registry",
        description="Rebuild build/runs/index.json and print it; or mark a parked run failed.",
    )
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="print the full index")
    commands = parser.add_subparsers(dest="command")
    mark = commands.add_parser(
        "mark-failed",
        help="set a run parked in `running` to failed (pid reuse, clock jump, other host)",
    )
    mark.add_argument("run_id")
    mark.add_argument("--force", action="store_true", help="even if it still looks alive from here")
    mark.add_argument("--reason", default="no reason given")
    args = parser.parse_args(argv)
    registry = RunRegistry(args.runs_root)
    if args.command == "mark-failed":
        payload = registry.mark_failed(args.run_id, reason=args.reason, force=args.force)
        print(json.dumps({"event": "mark_failed", "run_id": args.run_id, "failure": payload["failure"]}))
        return
    index = registry.rebuild_index()
    if args.json:
        print(json.dumps(index, indent=2, sort_keys=True))
        return
    print(
        f"{'run_id':<40} {'trainer':<22} {'physics':<8} {'status':<9} {'stale':<5} "
        f"{'track':<5} {'updates':>8} {'best_lap_ms':>11} eval_best"
    )
    for entry in index["runs"]:
        summary = entry.get("summary") or {}
        print(
            f"{entry.get('run_id', '?'):<40} {str(entry.get('trainer', '?')):<22} "
            f"{str(entry.get('physics_sha256', '?'))[:8]:<8} {entry.get('status', '?'):<9} "
            f"{str(entry.get('stale', '')):<5} {entry.get('track_id', '?'):<5} "
            f"{str(summary.get('updates', '')):>8} {str(summary.get('best_lap_ms', '')):>11} "
            f"{summary.get('eval_best_lap_ms', '')}"
        )


if __name__ == "__main__":
    main()


__all__ = [
    "HEARTBEAT_INTERVAL_SECONDS",
    "STALE_HEARTBEAT_SECONDS",
    "HeartbeatThread",
    "RunLock",
    "RunLockError",
    "RunRecord",
    "RunRegistry",
    "default_runs_root",
    "new_run_payload",
    "pid_start_ticks",
    "validate_run_payload",
]
