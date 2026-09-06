"""Everything needed to say which code, binary and hardware produced a run."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from tmnf_rl.tracks import project_root
from tmnf_rl.utils import sha256_file


def git_state(root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout
    dirty_paths = sorted(
        line[3:] for line in status.splitlines() if line.strip()
    )
    return {
        "git_commit": commit,
        "git_dirty": bool(dirty_paths),
        "git_dirty_paths": dirty_paths,
    }


def code_sha256(root: Path) -> str:
    """Hash of every tracked-or-not Python file under python/tmnf_rl.

    A dirty tree cannot be pinned by commit, so reproduce compares this hash
    instead of trusting the commit alone.
    """
    package = root / "python" / "tmnf_rl"
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def collect(
    physics_library: Path,
    *,
    seed: int,
    root: Path | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    info: dict[str, Any] = {
        **git_state(root),
        "code_sha256": code_sha256(root),
        "physics_library": str(Path(physics_library).resolve()),
        "physics_sha256": sha256_file(physics_library),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "command_line": [sys.executable, *sys.argv],
        "cwd": os.getcwd(),
        "hostname": platform.node(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "loadavg_at_start": list(os.getloadavg()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": seed,
    }
    if torch_module is not None:
        info["torch_version"] = torch_module.__version__
        info["cuda_version"] = torch_module.version.cuda
        info["cuda_device_name"] = (
            torch_module.cuda.get_device_name(0)
            if torch_module.cuda.is_available()
            else None
        )
        info["cudnn_version"] = torch_module.backends.cudnn.version()
        # The float32 CUDA switches pin_cuda_numerics set; collect() must run
        # after select_device() so this records the pinned state.
        from tmnf_rl.agents.ppo import numerics_state

        info["numerics"] = numerics_state()
    try:
        import numpy

        info["numpy_version"] = numpy.__version__
    except ImportError:
        info["numpy_version"] = None
    return info


# Provenance fields that must match between two processes for their metrics
# to be comparable; `reproduce` refuses when any of them differ.
NUMERICS_FIELDS = (
    "torch_version",
    "cuda_version",
    "cudnn_version",
    "cuda_device_name",
    "numerics",
)


def numerics_differences(recorded: dict[str, Any], current: dict[str, Any]) -> list[dict[str, Any]]:
    """Field-by-field differences in the numerics-relevant provenance."""
    differences: list[dict[str, Any]] = []
    for field in NUMERICS_FIELDS:
        old, new = recorded.get(field), current.get(field)
        if field == "numerics":
            old = old if isinstance(old, dict) else {}
            new = new if isinstance(new, dict) else {}
            for key in sorted(set(old) | set(new)):
                if old.get(key) != new.get(key):
                    differences.append({"field": f"numerics.{key}", "recorded": old.get(key), "current": new.get(key)})
        elif old != new:
            differences.append({"field": field, "recorded": old, "current": new})
    return differences


__all__ = ["NUMERICS_FIELDS", "code_sha256", "collect", "git_state", "numerics_differences"]
