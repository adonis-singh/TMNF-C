"""Track catalogue: fixture paths and pinned challenge hashes per track id."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TrackSpec:
    track_id: str
    name: str
    sha256: str

    @property
    def visual_scene(self) -> str:
        """Game visual scene identifier consumed by export_viewer_scene."""
        return self.track_id

    def track_path(self, root: Path) -> Path:
        return root / "oracle" / "tracks" / f"{self.name}.tmnftrack"

    def vehicle_path(self, root: Path) -> Path:
        """`oracle/vehicles/<Code>-<Environment>.tmnfvehicle`: Nations ids
        are the code (`a08` -> `A08-Stadium`), United ids carry the
        environment (`desert-a1` -> `A1-Desert`)."""
        if "-" in self.track_id:
            environment, code = self.track_id.split("-", 1)
            stem = f"{code.upper()}-{environment.capitalize()}"
        else:
            stem = f"{self.track_id.upper()}-Stadium"
        return root / "oracle" / "vehicles" / f"{stem}.tmnfvehicle"

    def route_path(self, root: Path) -> Path:
        return root / "oracle" / "routes" / f"{self.name}.tmnfroute"

    def missing_fixtures(self, root: Path) -> list[Path]:
        """Fixture files the manifest promises but this tree does not have
        (a track onboarded without committing its files, F32)."""
        return [
            path
            for path in (self.track_path(root), self.vehicle_path(root), self.route_path(root))
            if not path.is_file()
        ]


# A01 predates oracle/tracks/manifest.txt and is not listed there.
_A01 = TrackSpec(
    "a01",
    "A01-Race",
    "f0a870809be99da2cb36ad5df43a2cf63d8f74fe4ac3470ecac68b9e97625dc3",
)


def _load_manifest(root: Path) -> dict[str, TrackSpec]:
    tracks = {}
    if _A01.track_path(root).is_file():
        tracks[_A01.track_id] = _A01
    manifest = root / "oracle" / "tracks" / "manifest.txt"
    for line in (manifest.read_text(encoding="utf-8").splitlines() if manifest.exists() else []):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split("|")
        if len(fields) != 4:
            raise ValueError(f"invalid track manifest line: {line}")
        track_id, name, sha256, _replays = fields
        if track_id in tracks and track_id != "a01":
            raise ValueError(f"duplicate track id in manifest: {track_id}")
        if len(sha256) != 64:
            raise ValueError(f"track {track_id} has a malformed SHA-256")
        tracks[track_id] = TrackSpec(track_id, name, sha256)
    return tracks


_CACHE: dict[Path, dict[str, TrackSpec]] = {}


def track_catalogue(root: Path | None = None) -> dict[str, TrackSpec]:
    resolved = (Path(root) if root is not None else project_root()).resolve()
    if resolved not in _CACHE:
        _CACHE[resolved] = _load_manifest(resolved)
    return _CACHE[resolved]


def track_spec(track_id: str, root: Path | None = None) -> TrackSpec:
    catalogue = track_catalogue(root)
    if track_id not in catalogue:
        raise ValueError(
            f"track {track_id!r} is not installed; local tracks: {sorted(catalogue)}. "
            "See docs/LOCAL_ASSETS.md to generate fixtures from your installation."
        )
    return catalogue[track_id]


__all__ = ["TrackSpec", "project_root", "track_catalogue", "track_spec"]
