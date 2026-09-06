"""The environment dimension of the oracle corpus.

A track is identified by (campaign, environment, code). Two campaigns exist in
the Forever installs:

    nations  GameData/Tracks/Campaigns/Nations/<Series>/<Code>-<Name>.Challenge.Gbx
             65 Stadium tracks, codes A01..E05. Track id is the code: `a01`.
    united   GameData/Tracks/Campaigns/United/Race/<Env>/<Series>/<Env><Code>.Challenge.Gbx
             7 environments x 21 tracks, codes A1..A5, B1..B5, C1..C5, D1..D5, E.
             Track id is `<env>-<code>`: `desert-a1`.

Vehicle snapshots are `oracle/vehicles/<CODE>-<Env>.tmnfvehicle`
(`A01-Stadium`, `A1-Desert`); references are `oracle/results/<id>_<name>.bin`.
The Wine prefix layout (game directory, user documents) is resolved by
`game_layout`, mirroring oracle/game_layout.sh.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import os
import getpass


ROOT = Path(__file__).resolve().parents[1]

# key -> (campaign directory name, collection / pak name)
ENVIRONMENTS: dict[str, tuple[str, str]] = {
    "stadium": ("Stadium", "Stadium"),
    "desert": ("Desert", "Speed"),
    "rally": ("Rally", "Rally"),
    "snow": ("Snow", "Alpine"),
    "island": ("Island", "Island"),
    "coast": ("Coast", "Coast"),
    "bay": ("Bay", "Bay"),
}
UNITED_CODES = tuple(
    f"{series}{number}" for series in "ABCD" for number in range(1, 6)
) + ("E",)

_NATIONS = re.compile(r"([A-E]\d{2})-[A-Za-z][A-Za-z0-9_-]*")
_UNITED = re.compile(r"(Stadium|Desert|Rally|Snow|Island|Coast|Bay)([A-D][1-5]|E)")
_UNITED_SLASH = re.compile(r"([a-z]+)/([A-Za-z0-9]+)")
_UNITED_ID = re.compile(r"([a-z]+)-([a-e][1-5]|e)")
_NATIONS_ID = re.compile(r"[a-e]\d{2}")


@dataclass(frozen=True)
class Track:
    name: str  # challenge stem: A01-Race, DesertA1
    campaign: str  # nations | united
    environment: str  # ENVIRONMENTS key
    code: str  # A01, A1, E

    @property
    def environment_name(self) -> str:
        return ENVIRONMENTS[self.environment][0]

    @property
    def id(self) -> str:
        if self.campaign == "nations":
            return self.code.lower()
        return f"{self.environment}-{self.code.lower()}"

    @property
    def vehicle(self) -> Path:
        return ROOT / f"oracle/vehicles/{self.code}-{self.environment_name}.tmnfvehicle"

    @property
    def snapshot(self) -> Path:
        return ROOT / f"oracle/tracks/{self.name}.tmnftrack"

    @property
    def route(self) -> Path:
        return ROOT / f"oracle/routes/{self.name}.tmnfroute"

    def result(self, name: str, suffix: str = "") -> Path:
        return ROOT / f"oracle/results/{self.id}_{name}{suffix}.bin"

    def challenge(self, game_dir: Path) -> Path:
        if self.campaign == "nations":
            pattern = f"GameData/Tracks/Campaigns/Nations/*/{self.name}.Challenge.Gbx"
        else:
            pattern = (
                f"GameData/Tracks/Campaigns/United/Race/{self.environment_name}"
                f"/*/{self.name}.Challenge.Gbx"
            )
        matches = list(game_dir.glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(
                f"found {len(matches)} campaign challenges for {self.name} under {game_dir}"
            )
        return matches[0]


def parse_track(spec: str) -> Track:
    """Accepts a Nations stem (A08-Endurance), a United stem (DesertA1) or
    the `<env>/<Code>` form (desert/A1)."""
    match = _NATIONS.fullmatch(spec)
    if match is not None:
        return Track(spec, "nations", "stadium", match.group(1))
    match = _UNITED.fullmatch(spec)
    if match is not None:
        return Track(spec, "united", match.group(1).lower(), match.group(2))
    match = _UNITED_SLASH.fullmatch(spec)
    if match is not None and match.group(1) in ENVIRONMENTS:
        code = match.group(2).upper()
        if code not in UNITED_CODES:
            raise RuntimeError(f"{spec}: United codes are A1..D5 and E")
        environment = match.group(1)
        return Track(f"{ENVIRONMENTS[environment][0]}{code}", "united", environment, code)
    raise RuntimeError(
        f"{spec!r} is not a campaign track (A08-Endurance, DesertA1 or desert/A1)"
    )


def track_from_manifest(track_id: str, track_name: str) -> Track:
    track = parse_track(track_name)
    if track.id != track_id:
        raise RuntimeError(f"manifest id {track_id} does not match track {track_name}")
    return track


def environment_of_id(track_id: str) -> str:
    if _NATIONS_ID.fullmatch(track_id):
        return "stadium"
    match = _UNITED_ID.fullmatch(track_id)
    if match is None or match.group(1) not in ENVIRONMENTS:
        raise RuntimeError(f"unknown track id {track_id!r}")
    return match.group(1)


def united_tracks(environment: str) -> tuple[Track, ...]:
    name = ENVIRONMENTS[environment][0]
    return tuple(Track(f"{name}{code}", "united", environment, code) for code in UNITED_CODES)


@dataclass(frozen=True)
class GameLayout:
    flavor: str  # nations | united
    prefix: Path
    game_dir: Path
    user_documents: Path
    user_dir: Path
    tmloader: Path

    @property
    def exe(self) -> Path:
        return self.game_dir / "TmForever.exe"

    @property
    def official_maps(self) -> Path:
        return self.user_dir / "Tracks/Challenges/Official Maps"


def game_layout(prefix: Path) -> GameLayout:
    """Same rules as oracle/game_layout.sh: one Forever install per prefix."""
    prefix = Path(prefix).resolve()
    wine_user = os.environ.get("TMNF_WINE_USER", getpass.getuser())
    if (prefix / "drive_c/TmUnitedForever/TmForever.exe").is_file():
        documents = prefix / "drive_c/users" / wine_user / "Documents"
        return GameLayout(
            "united", prefix, prefix / "drive_c/TmUnitedForever", documents,
            documents / "TrackMania", ROOT / "third_party/TMLoader_united/TMLoader.exe",
        )
    if (prefix / "drive_c/TmNationsForever/TmForever.exe").is_file():
        documents = prefix / "drive_c/users" / wine_user / "TMNFDocuments"
        return GameLayout(
            "nations", prefix, prefix / "drive_c/TmNationsForever", documents,
            documents / "TmForever", ROOT / "third_party/TMLoader/TMLoader.exe",
        )
    raise RuntimeError(f"missing Wine prefix with TmForever.exe: {prefix}")
