"""Download the external data the package reads into ``tmnf_fly.LOCAL`` (local/).

    PYTHONPATH=python build/venv/bin/python -m tmnf_fly.fetch_data [--only malecns|eyemap|mesh|flyvis]

    malecns   MaleCNS v1.0 flat connectome (Janelia FlyEM), three Arrow feather tables,
              1.1 GB                                           -> local/malecns/
    eyemap    Zhao et al. 2025 right-eye lens/cone micro-CT positions and the processed
              eyemap.RData (github.com/reiserlab/eyemap_T4)     -> local/eyemap/data/
    mesh      JRCFIB2022M brain shell (FlyEM MaleCNS ROI meshes) -> local/fly/JRCFIB2022M_brain.ply
    flyvis    flyvis 1.2.0 pretrained optic-lobe ensemble (Lappalainen et al. 2024,
              results_pretrained_models.zip from the flyvis Google Drive folder, the same file
              `flyvis download-pretrained` fetches), unpacked   -> local/fly/flyvis_data/results/

Every file has its size and sha256 pinned; a file already present with the right digest is
not downloaded again. The derived files are built afterwards by ``tmnf_fly.build_data``.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import os
import re
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from tmnf_fly import EYEMAP_DIR, FLY_DIR, LOCAL, MALECNS_DIR

MALECNS_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
EYEMAP_URL = "https://raw.githubusercontent.com/reiserlab/eyemap_T4/main/"
MESH_URL = "https://storage.googleapis.com/flyem-male-cns/rois/pointcloud-shells/"


def flyvis_url() -> str:
    """The pretrained-models zip that `flyvis download-pretrained` fetches: file
    13cJr2nMn89j-jBAd5RduYRJpBcXwoNrC of the flyvis Google Drive folder, read through
    the Drive v3 API with the key the installed flyvis package carries in
    flyvis_cli/download_pretrained_models.py (not copied into this repository)."""
    import flyvis_cli.download_pretrained_models as m
    match = re.search(r'api_key = "([^"]+)"', inspect.getsource(m))
    if match is None:
        raise RuntimeError("flyvis_cli/download_pretrained_models.py no longer carries its Drive API key")
    return f"https://www.googleapis.com/drive/v3/files/13cJr2nMn89j-jBAd5RduYRJpBcXwoNrC?alt=media&key={match.group(1)}"


@dataclass(frozen=True)
class File:
    url: str
    path: Path
    size: int
    sha256: str


DATASETS: dict[str, list[File]] = {
    "malecns": [
        File(MALECNS_URL + "body-annotations-male-cns-v1.0-minconf-0.5.feather",
             MALECNS_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather",
             14483314, "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2"),
        File(MALECNS_URL + "body-neurotransmitters-male-cns-v1.0.feather",
             MALECNS_DIR / "body-neurotransmitters-male-cns-v1.0.feather",
             43282834, "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621"),
        File(MALECNS_URL + "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
             MALECNS_DIR / "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
             1051241946, "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1"),
    ],
    "eyemap": [
        File(EYEMAP_URL + "data/eyemap.RData", EYEMAP_DIR / "data/eyemap.RData",
             182218, "c3f63c69afdc3f381cdafabb1d376b8e25ec5a71bce67550ae42fcdd79cf618e"),
        File(EYEMAP_URL + "data/microCT/20240701_position/lens.csv",
             EYEMAP_DIR / "data/microCT/20240701_position/lens.csv",
             82071, "477fbef0bd60c0a340aded55770c73ddae9fb4a5baf73738bf7ea9c128599c0f"),
        File(EYEMAP_URL + "data/microCT/20240701_position/cone.csv",
             EYEMAP_DIR / "data/microCT/20240701_position/cone.csv",
             84009, "a2d1b6f5b8296e6fb6a26a7d0bdf96ccba1260923ca2a81fb0b1123f2557c355"),
    ],
    "mesh": [
        File(MESH_URL + "JRCFIB2022M_brain.ply", FLY_DIR / "JRCFIB2022M_brain.ply",
             1255587, "13ea41a7ce2677eae00738c23b2be3c2f29ac17c6976c2209998a85efccaa427"),
    ],
    "flyvis": [
        File(flyvis_url(), FLY_DIR / "flyvis_data/results_pretrained_models.zip",
             3417042, "71c78d4070556a536b13b23ee3139cd2788aa2a9d07d430a223b4edead281db1"),
    ],
}
FLYVIS_ZIP_ROOT = "results/"
FLYVIS_MODEL_FILES = 250
PROGRESS_BYTES = 128 << 20


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(f: File) -> None:
    size = f.path.stat().st_size
    if size != f.size:
        raise RuntimeError(f"{f.path}: {size} bytes, expected {f.size}")
    digest = sha256_of(f.path)
    if digest != f.sha256:
        raise RuntimeError(f"{f.path}: sha256 {digest}, expected {f.sha256}")


def fetch(f: File) -> bool:
    """Download unless the file is already present and verified. Returns True if downloaded."""
    if f.path.exists():
        verify(f)
        return False
    f.path.parent.mkdir(parents=True, exist_ok=True)
    part = f.path.with_name(f.path.name + ".part")
    h = hashlib.sha256()
    n = 0
    with urllib.request.urlopen(f.url, timeout=120) as r, part.open("wb") as out:
        while block := r.read(1 << 20):
            out.write(block)
            h.update(block)
            if (n + len(block)) // PROGRESS_BYTES != n // PROGRESS_BYTES:
                print(f"  {f.path.name}: {(n + len(block)) / 1e6:.0f} / {f.size / 1e6:.0f} MB", file=sys.stderr, flush=True)
            n += len(block)
    if n != f.size:
        raise RuntimeError(f"{f.url}: received {n} bytes, expected {f.size}")
    if h.hexdigest() != f.sha256:
        raise RuntimeError(f"{f.url}: sha256 {h.hexdigest()}, expected {f.sha256}")
    os.replace(part, f.path)
    return True


def unpack_flyvis(zip_path: Path) -> int:
    """Extract results/ next to the zip (flyvis's FLYVIS_ROOT_DIR layout). Returns the file count."""
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        bad = [n for n in names if not n.startswith(FLYVIS_ZIP_ROOT) or ".." in n]
        if bad:
            raise RuntimeError(f"{zip_path}: unexpected members {bad[:5]}")
        z.extractall(zip_path.parent)
    files = sum(not n.endswith("/") for n in names)
    if files != FLYVIS_MODEL_FILES:
        raise RuntimeError(f"{zip_path}: {files} files, expected {FLYVIS_MODEL_FILES}")
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=sorted(DATASETS), action="append",
                        help="restrict to these datasets (repeatable); default all")
    args = parser.parse_args()
    names = args.only or list(DATASETS)
    print(f"data root {LOCAL}", flush=True)
    for name in names:
        for f in DATASETS[name]:
            downloaded = fetch(f)
            print(f"{name:8s} {'downloaded' if downloaded else 'present   '} {f.path.relative_to(LOCAL)} "
                  f"({f.size / 1e6:.1f} MB, sha256 {f.sha256[:12]})", flush=True)
            if name == "flyvis":
                files = unpack_flyvis(f.path)
                print(f"{name:8s} unpacked {files} files into {f.path.parent.relative_to(LOCAL)}/results", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
