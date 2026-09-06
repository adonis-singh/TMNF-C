"""Small shared helpers: atomic JSON, hashing, timestamps, CPU affinity."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CPU_AFFINITY = frozenset({*range(0, 14), *range(16, 30)})
RESERVED_CPUS = frozenset(int(x) for x in os.environ.get("TMNF_RESERVED_CPUS", "").split(",") if x.strip())


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_iso(text: str) -> datetime:
    """Parse an ISO-8601 timestamp; naive (offset-less) values are an error.

    A naive datetime silently poisons every age computation against UTC now
    (TypeError deep inside the index rebuild), so it is rejected here.
    """
    if not isinstance(text, str):
        raise ValueError(f"timestamp must be a string, got {type(text).__name__}")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"timestamp {text!r} has no UTC offset")
    return parsed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write temp file in the same directory, fsync, then rename over target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = json_dumps(payload).encode("utf-8")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        # A failed write (disk full, quota) must not leave a partial temp file
        # next to the target (F27); the target itself is untouched.
        temporary.unlink(missing_ok=True)
        raise


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def enforce_cpu_affinity() -> list[int]:
    """Fail if the process may run on the reserved physical cores."""
    allowed = os.sched_getaffinity(0)
    if allowed & RESERVED_CPUS:
        raise RuntimeError(
            "process affinity includes reserved CPUs "
            f"{sorted(allowed & RESERVED_CPUS)}; start with "
            "taskset -c 0-13,16-29"
        )
    return sorted(allowed)


def require_single_gpu_env() -> str:
    """``CUDA_VISIBLE_DEVICES`` must name exactly one GPU index.

    The trainer never picks a GPU itself: the operator says which physical
    card a run may use (GPU 0, the RTX 5090, for the phase 1 runs) and the
    card's name goes into provenance, where ``reproduce`` compares it.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.isdigit():
        raise RuntimeError(
            "set CUDA_VISIBLE_DEVICES to exactly one GPU index before starting "
            f"TMNF training (got {visible!r})"
        )
    return visible


__all__ = [
    "CPU_AFFINITY",
    "RESERVED_CPUS",
    "enforce_cpu_affinity",
    "json_dumps",
    "parse_iso",
    "read_json",
    "require_single_gpu_env",
    "sha256_file",
    "utc_now_iso",
    "write_json_atomic",
]
