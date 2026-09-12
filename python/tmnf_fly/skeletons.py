"""MaleCNS v1.0 neuron skeletons: population selection, download, SWC parsing, per-set npz cache.

Coordinates: SWC files are in 8 nm voxel units; everything here is converted to micrometres.

Cache layout (root = tmnf_fly.FLY_DIR, local/fly):
    root/skeletons/<bodyId>.swc            raw SWC downloads
    root/skeleton_sets.json                {set name: [bodyId, ...]}
    root/skeleton_npz/<set name>.npz       packed arrays for one set (see SkeletonSet)
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tmnf_fly import ANNOTATIONS_FEATHER, FLY_DIR

SWC_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/{body_id}.swc"
VOXEL_NM = 8.0

NAMED_TYPES = [
    "DNp01", "DNp02", "DNp04", "DNp11", "DNa01", "DNa02", "DNp09",
    "HSN", "HSE", "HSS",
    "LC4", "LPLC2", "LPLC1", "LC10a", "LC11", "LC6",
]
DN_TYPES = [t for t in NAMED_TYPES if t.startswith("DN")]


@dataclass
class SkeletonSet:
    """Packed skeletons of one population.

    xyz:     (N, 3) float32, micrometres
    parent:  (N,) int32, global node index of the parent, -1 for roots
    radius:  (N,) float32, micrometres
    offsets: (M + 1,) int64, neuron i owns nodes offsets[i]:offsets[i + 1]
    body_ids: (M,) int64
    """

    name: str
    body_ids: np.ndarray
    xyz: np.ndarray
    parent: np.ndarray
    radius: np.ndarray
    offsets: np.ndarray

    def __len__(self) -> int:
        return len(self.body_ids)

    @property
    def n_nodes(self) -> int:
        return len(self.xyz)

    @property
    def node_counts(self) -> np.ndarray:
        return np.diff(self.offsets)

    def save(self, path: Path) -> None:
        np.savez(path, body_ids=self.body_ids, xyz=self.xyz, parent=self.parent,
                 radius=self.radius, offsets=self.offsets)

    @classmethod
    def load(cls, name: str, path: Path) -> "SkeletonSet":
        z = np.load(path)
        return cls(name, z["body_ids"], z["xyz"], z["parent"], z["radius"], z["offsets"])


# ---------------------------------------------------------------- selection

def select_sets(annotations: Path = ANNOTATIONS_FEATHER, seed: int = 0) -> dict[str, list[int]]:
    """Choose the populations to render. Sets are pairwise disjoint."""
    df = pd.read_feather(annotations)
    df = df[df["status"] == "Traced"]
    typ = df["type"].fillna("")
    cls = df["class"].fillna("")
    rng = np.random.default_rng(seed)

    def ids(mask) -> list[int]:
        return sorted(int(b) for b in df.loc[mask, "bodyId"])

    def sample(mask, n: int, taken: set[int]) -> list[int]:
        pool = np.array([b for b in ids(mask) if b not in taken], dtype=np.int64)
        if len(pool) < n:
            raise ValueError(f"only {len(pool)} candidates for a sample of {n}")
        return sorted(int(b) for b in rng.choice(pool, size=n, replace=False))

    sets: dict[str, list[int]] = {}
    for t in NAMED_TYPES:
        sets[t] = ids(typ == t)
    sets["VS"] = ids(typ.str.startswith("VS"))
    sets["DN"] = ids((df["superclass"] == "descending_neuron") & ~typ.isin(DN_TYPES))
    sets["T4"] = sample(typ.str.match(r"^T4[a-d]$"), 300, set())
    sets["T5"] = sample(typ.str.match(r"^T5[a-d]$"), 300, set())
    sets["L1"] = sample(typ == "L1", 300, set())
    sets["Mi1"] = sample(typ == "Mi1", 300, set())
    sets["KC"] = sample(cls == "Kenyon_Cell", 400, set())
    sets["MBON"] = ids(cls == "MBON")
    sets["DAN"] = ids(cls == "DAN")
    taken = {b for v in sets.values() for b in v}
    sets["cb_intrinsic"] = sample(df["superclass"] == "cb_intrinsic", 2000, taken)

    for name, v in sets.items():
        if not v:
            raise ValueError(f"set {name} is empty")
    all_ids = [b for v in sets.values() for b in v]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("sets overlap")
    return sets


# ---------------------------------------------------------------- download

def _fetch(body_id: int, swc_dir: Path) -> int:
    dst = swc_dir / f"{body_id}.swc"
    if dst.exists():
        return 0
    tmp = dst.with_suffix(".part")
    with urllib.request.urlopen(SWC_URL.format(body_id=body_id), timeout=120) as r:
        data = r.read()
    if not data.startswith(b"#") and not data.lstrip()[:1].isdigit():
        raise RuntimeError(f"{body_id}: unexpected SWC payload")
    tmp.write_bytes(data)
    os.replace(tmp, dst)
    return len(data)


def download(body_ids: list[int], swc_dir: Path, workers: int = 32) -> tuple[int, int, float]:
    """Download missing SWCs. Returns (n_downloaded, bytes_downloaded, seconds)."""
    swc_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = 0
    nbytes = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, got in enumerate(ex.map(lambda b: _fetch(b, swc_dir), body_ids)):
            if got:
                n += 1
                nbytes += got
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(body_ids)} files, {nbytes / 1e6:.1f} MB, {time.time() - t0:.0f} s", file=sys.stderr)
    return n, nbytes, time.time() - t0


# ---------------------------------------------------------------- parsing

def parse_swc(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SWC -> (xyz um float32 (N,3), parent index int32 (N,), radius um float32 (N,))."""
    rows = [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
    a = np.array(" ".join(rows).split(), dtype=np.float64).reshape(-1, 7)
    ids = a[:, 0].astype(np.int64)
    par = a[:, 6].astype(np.int64)
    order = np.argsort(ids)
    ids_sorted = ids[order]
    if np.any(np.diff(ids_sorted) == 0):
        raise ValueError(f"{path}: duplicate node ids")
    # map parent ids to row indices; -1 stays -1
    has_parent = par >= 0
    pos = np.searchsorted(ids_sorted, par[has_parent])
    if np.any(pos >= len(ids)) or np.any(ids_sorted[np.minimum(pos, len(ids) - 1)] != par[has_parent]):
        raise ValueError(f"{path}: parent id not found")
    parent = np.full(len(ids), -1, dtype=np.int32)
    parent[has_parent] = order[pos]
    scale = VOXEL_NM / 1000.0
    xyz = (a[:, 2:5] * scale).astype(np.float32)
    radius = (a[:, 5] * scale).astype(np.float32)
    return xyz, parent, radius


def pack_set(name: str, body_ids: list[int], swc_dir: Path) -> SkeletonSet:
    xyzs, parents, radii, offsets = [], [], [], [0]
    for b in body_ids:
        xyz, parent, radius = parse_swc(swc_dir / f"{b}.swc")
        base = offsets[-1]
        parents.append(np.where(parent >= 0, parent + base, -1).astype(np.int32))
        xyzs.append(xyz)
        radii.append(radius)
        offsets.append(base + len(xyz))
    return SkeletonSet(
        name,
        np.asarray(body_ids, dtype=np.int64),
        np.concatenate(xyzs),
        np.concatenate(parents),
        np.concatenate(radii),
        np.asarray(offsets, dtype=np.int64),
    )


# ---------------------------------------------------------------- top level

def build(root: Path = FLY_DIR, seed: int = 0) -> dict[str, SkeletonSet]:
    """Select, download, parse and cache every set. Idempotent."""
    sets_json = root / "skeleton_sets.json"
    if sets_json.exists():
        sets = {k: [int(b) for b in v] for k, v in json.loads(sets_json.read_text()).items()}
    else:
        sets = select_sets(seed=seed)
        root.mkdir(parents=True, exist_ok=True)
        sets_json.write_text(json.dumps(sets, indent=1))
    all_ids = [b for v in sets.values() for b in v]
    print(f"{len(sets)} sets, {len(all_ids)} neurons", file=sys.stderr)
    for k, v in sets.items():
        print(f"  {k:14s} {len(v):5d}", file=sys.stderr)

    n, nbytes, dt = download(all_ids, root / "skeletons")
    print(f"downloaded {n} files, {nbytes / 1e6:.1f} MB in {dt:.1f} s", file=sys.stderr)
    total_bytes = sum((root / "skeletons" / f"{b}.swc").stat().st_size for b in all_ids)
    print(f"on disk: {len(all_ids)} SWC files, {total_bytes / 1e6:.1f} MB", file=sys.stderr)

    npz_dir = root / "skeleton_npz"
    npz_dir.mkdir(exist_ok=True)
    out: dict[str, SkeletonSet] = {}
    t0 = time.time()
    for name, ids in sets.items():
        path = npz_dir / f"{name}.npz"
        if not path.exists():
            pack_set(name, ids, root / "skeletons").save(path)
        out[name] = SkeletonSet.load(name, path)
    nodes = sum(s.n_nodes for s in out.values())
    print(f"packed {len(out)} sets, {nodes} nodes total in {time.time() - t0:.1f} s", file=sys.stderr)
    return out


def load(root: Path = FLY_DIR) -> dict[str, SkeletonSet]:
    """Load cached sets; fails if build() has not run."""
    sets = json.loads((root / "skeleton_sets.json").read_text())
    return {name: SkeletonSet.load(name, root / "skeleton_npz" / f"{name}.npz") for name in sets}


if __name__ == "__main__":
    sets = build()
    for name, s in sets.items():
        c = s.node_counts
        print(f"{name:14s} neurons={len(s):5d} nodes={s.n_nodes:8d} nodes/neuron min={c.min()} med={int(np.median(c))} max={c.max()}")
