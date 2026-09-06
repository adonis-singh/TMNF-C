"""Viewer scene export from an exact per-tick input schedule."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from tmnf_rl.inputs import InputSchedule
from tmnf_rl.tracks import TrackSpec, project_root


EXPORTER_RELATIVE = Path("build") / "export_viewer_scene"
EXPORT_PADDING_TICKS = 100


def exporter_path(root: Path | None = None) -> Path:
    return (root or project_root()) / EXPORTER_RELATIVE


def require_exporter(root: Path | None = None) -> Path:
    path = exporter_path(root)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; run "
            "`cmake --build build --target export_viewer_scene`"
        )
    return path


def export_scene(
    *,
    spec: TrackSpec,
    schedule: InputSchedule,
    output: Path,
    expected_finish_ms: int,
    root: Path | None = None,
) -> dict[str, Any]:
    """Write ``output`` (scene JSON) and ``output``.inputs.bin.

    The exporter replays the schedule through the World path that is
    byte-exact against the game captures. Its finish time is returned as
    ``finish_ms`` (None when that path does not finish) next to the vec-env's
    ``expected_finish_ms``; since native d09dc00 the two must agree and a
    mismatch is a physics regression. A
    non-finishing World replay is recorded, not raised: the run must survive it.
    """
    root = (root or project_root()).resolve()
    exporter = require_exporter(root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    inputs_path = output.with_suffix(".inputs.bin")
    # The exporter checks the race state at the top of iteration N for the
    # state after N physics steps, so a lap of T ticks needs at least T + 1
    # records. The extra released-input records also let the viewer show the
    # car rolling through the finish.
    inputs_sha256 = schedule.write(inputs_path, schedule.tick_count + EXPORT_PADDING_TICKS)
    result = subprocess.run(
        [
            str(exporter),
            str(spec.track_path(root)),
            str(spec.route_path(root)),
            str(spec.vehicle_path(root)),
            str(inputs_path),
            spec.sha256,
            spec.visual_scene,
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"finish (\d+) ms", result.stdout)
    if match is None:
        raise RuntimeError(f"unexpected exporter output: {result.stdout!r}")
    finish_ms: int | None = int(match.group(1)) or None
    with output.open("r", encoding="utf-8") as handle:
        header = handle.read(200)
    if '"format":"tmnf-c-viewer-scene"' not in header:
        raise RuntimeError(f"exporter wrote an unexpected scene header: {header!r}")
    summaries = [line.removeprefix("lap_summary ") for line in result.stdout.splitlines()
                 if line.startswith("lap_summary ")]
    # Older exporter binaries remain usable; current ones publish the same
    # fields directly, avoiding a second parse of the complete track geometry.
    lap = json.loads(summaries[0]) if len(summaries) == 1 else scene_lap_summary(output)
    if lap["finishTimeMs"] != (finish_ms or 0):
        raise RuntimeError("exporter lap summary disagrees with its finish time")
    return {
        "scene": str(output),
        "inputs": str(inputs_path),
        "inputs_sha256": inputs_sha256,
        "tick_count": schedule.tick_count,
        "finish_ms": finish_ms,
        "env_finish_ms": int(expected_finish_ms),
        "matches_env": finish_ms == int(expected_finish_ms),
        "world_finished": finish_ms is not None,
        "scene_checkpoint_ticks": lap["checkpointTicks"],
        "exporter_stdout": result.stdout.strip(),
    }


def scene_lap_summary(scene_path: Path) -> dict[str, Any]:
    with Path(scene_path).open("r", encoding="utf-8") as handle:
        scene = json.load(handle)
    lap = scene["lap"]
    return {
        "finishTimeMs": lap["finishTimeMs"],
        "checkpointTicks": lap["checkpointTicks"],
        "finishTick": lap["finishTick"],
        "tickCount": lap["tickCount"],
    }


__all__ = ["export_scene", "exporter_path", "require_exporter", "scene_lap_summary"]
