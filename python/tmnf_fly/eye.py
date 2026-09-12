"""Drosophila right-eye ommatidia viewing directions (Zhao et al. 2025 micro-CT)
in a head frame (x forward, y left, z up), plus the registration of Zhao's hex
lattice onto the MaleCNS medulla column lattice (assignedOlHex1/2)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather
import rdata
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from tmnf_fly import ANNOTATIONS_FEATHER, EYEMAP_DIR, FLY_DIR

EYEMAP_RDATA = EYEMAP_DIR / "data/eyemap.RData"
MICROCT_DIR = EYEMAP_DIR / "data/microCT/20240701_position"
OMMATIDIA_NPZ = FLY_DIR / "ommatidia.npz"

N_OMMATIDIA = 852
# The two eyes of the micro-CT head are separated along the scanner Y axis.
MICROCT_EYE_SPLIT_Y = 500.0
# Zhao's (ix, iy) lattice has neighbours at +-(1,0), +-(0,1), +-(1,1); MaleCNS
# (hex1, hex2) uses the same convention, so the lattice symmetries are the same.
HEX_NEIGHBOURS = np.array([(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)])


@dataclass
class Eye:
    lens_id: np.ndarray  # (852,) Zhao lens ids 1..852
    hex: np.ndarray  # (852, 2) Zhao (ix, iy)
    dirs_right: np.ndarray  # (852, 3) head-frame unit viewing directions, right eye
    dirs_left: np.ndarray  # (852, 3) mirror image (y -> -y)
    malecns_column_right: np.ndarray  # (852, 2) (hex1, hex2) or -1
    malecns_column_left: np.ndarray  # (852, 2)
    equator_rows: np.ndarray  # (2,) values of ix+iy of the two rows flanking the equator

    def save(self, path: Path = OMMATIDIA_NPZ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.__dict__)

    @staticmethod
    def load(path: Path = OMMATIDIA_NPZ) -> "Eye":
        with np.load(path) as z:
            return Eye(**{k: z[k] for k in z.files})


def azimuth_elevation(dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Degrees. Azimuth positive to the fly's right (Zhao's plotting convention),
    zero straight ahead; elevation positive up."""
    az = np.degrees(np.arctan2(-dirs[..., 1], dirs[..., 0]))
    el = np.degrees(np.arcsin(np.clip(dirs[..., 2], -1.0, 1.0)))
    return az, el


def mollweide(dirs: np.ndarray) -> np.ndarray:
    """(…, 2) Mollweide coordinates, front at the origin, right eye at x > 0.
    x in [-2*sqrt(2), 2*sqrt(2)], y in [-sqrt(2), sqrt(2)]."""
    az, el = azimuth_elevation(dirs)
    lam = np.radians(az)
    phi = np.radians(el)
    theta = phi.copy()
    for _ in range(12):  # Newton iteration for 2*theta + sin(2*theta) = pi*sin(phi)
        f = 2 * theta + np.sin(2 * theta) - np.pi * np.sin(phi)
        theta = theta - f / np.maximum(2 + 2 * np.cos(2 * theta), 1e-9)
    x = 2 * np.sqrt(2) / np.pi * lam * np.cos(theta)
    y = np.sqrt(2) * np.sin(theta)
    return np.stack([x, y], axis=-1)


def hexagon_patches(eye: "Eye") -> np.ndarray:
    """(2*852, 6, 2) hexagon vertices in Mollweide coordinates, left eye first.
    Each hexagon is scaled to half the median projected distance to its lattice
    neighbours; neighbours across the +-180 deg azimuth seam (dorsal pole) are skipped."""
    position = {tuple(h): i for i, h in enumerate(eye.hex)}
    patches = []
    for dirs in (eye.dirs_left, eye.dirs_right):
        xy = mollweide(dirs)
        az = azimuth_elevation(dirs)[0]
        for i, h in enumerate(eye.hex):
            nb = [position[(h[0] + d[0], h[1] + d[1])] for d in HEX_NEIGHBOURS if (h[0] + d[0], h[1] + d[1]) in position]
            nb = [j for j in nb if abs(az[j] - az[i]) < 90.0]
            radius = 0.5 * np.median(np.linalg.norm(xy[nb] - xy[i], axis=1))
            ang = np.arange(6) * np.pi / 3 + np.pi / 6
            patches.append(xy[i] + radius * np.stack([np.cos(ang), np.sin(ang)], axis=1))
    return np.array(patches)


def _kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Proper rotation R minimizing |R source - target|."""
    u, _, vt = np.linalg.svd(source.T @ target)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1.0, 1.0)))


def _fit_sphere_center(points: np.ndarray) -> np.ndarray:
    a = np.c_[2 * points, np.ones(len(points))]
    b = (points**2).sum(1)
    x = np.linalg.lstsq(a, b, rcond=None)[0]
    return x[:3]


def _pca_frame(unit_dirs: np.ndarray) -> np.ndarray:
    mean = _unit(unit_dirs.mean(0))
    axis = np.linalg.eigh(np.cov(unit_dirs.T))[1][:, 2]
    axis = _unit(axis - mean * (axis @ mean))
    return np.c_[mean, axis, np.cross(mean, axis)]


def _match_lenses_to_utp(radial: np.ndarray, utp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """utp_lens_rot is the set of micro-CT lens positions projected on the unit
    sphere in a rotated frame. Recover the rotation and the row (lens id)
    correspondence by ICP from PCA-aligned starts (4 sign choices)."""
    frame_u = _pca_frame(utp)
    frame_r = _pca_frame(radial)
    best = None
    for sign_axis in (1.0, -1.0):
        for sign_cross in (1.0, -1.0):
            rot = frame_u @ (frame_r * np.array([1.0, sign_axis, sign_cross])).T
            tree = cKDTree(utp)
            for _ in range(30):
                _, idx = tree.query(radial @ rot.T)
                rot = _kabsch(radial, utp[idx])
            dist, idx = tree.query(radial @ rot.T)
            resid = np.degrees(2 * np.arcsin(dist / 2))
            if best is None or np.median(resid) < best[0]:
                best = (np.median(resid), rot, idx, resid)
    median, rot, idx, resid = best
    if resid.max() > 0.5 or len(np.unique(idx)) != len(radial):
        raise RuntimeError(f"lens/utp match failed: max resid {resid.max():.3f} deg, unique {len(np.unique(idx))}")
    return rot, idx


def _smooth_on_lattice(dirs: np.ndarray, hexes: np.ndarray) -> np.ndarray:
    """Local linear regression of the direction over the 7-cell hex
    neighbourhood, evaluated at the centre. Reproduces Zhao's `ucl_rot_sm`
    (median difference 0.13 deg) from the raw lens-cone directions."""
    position = {tuple(h): i for i, h in enumerate(hexes)}
    out = np.empty_like(dirs)
    for i, h in enumerate(hexes):
        rows = [(i, (0, 0))]
        for d in HEX_NEIGHBOURS:
            k = (h[0] + d[0], h[1] + d[1])
            if k in position:
                rows.append((position[k], (d[0], d[1])))
        if len(rows) < 4:
            raise RuntimeError(f"lens at hex {tuple(h)} has only {len(rows) - 1} neighbours")
        design = np.array([[1.0, d[0], d[1]] for _, d in rows])
        beta = np.linalg.lstsq(design, dirs[[r for r, _ in rows]], rcond=None)[0]
        out[i] = beta[0]
    return _unit(out)


def _load_microct(name: str) -> np.ndarray:
    df = pd.read_csv(MICROCT_DIR / f"{name}.csv")
    return df[["Position X", "Position Y", "Position Z"]].to_numpy(float)


def _malecns_columns(side: str) -> np.ndarray:
    t = pyarrow.feather.read_feather(ANNOTATIONS_FEATHER)
    t = t[(t.status == "Traced") & t.type.isin(["L1", "L2", "L3", "Mi1"]) & (t.somaSide == side)]
    t = t.dropna(subset=["assignedOlHex1", "assignedOlHex2"])
    cols = t[["assignedOlHex1", "assignedOlHex2"]].astype(int).drop_duplicates().to_numpy()
    return cols


def _hex_symmetries(h: np.ndarray):
    """The 12 symmetries of the hex lattice in (ix, iy) coordinates with
    neighbours +-(1,0), +-(0,1), +-(1,1). Yields (label, transformed)."""
    # Standard axial coordinates (neighbours (1,0),(0,1),(1,-1)) are (ix, -iy).
    q, r = h[:, 0].copy(), -h[:, 1].copy()
    for mirror in (False, True):
        a, b = (r, q) if mirror else (q, r)
        for k in range(6):
            yield (mirror, k), np.c_[a, -b]
            a, b = -b, a + b


def register_lattice(zhao_hex: np.ndarray, malecns_cols: np.ndarray) -> tuple[np.ndarray, int, tuple]:
    """Return (852, 2) MaleCNS (hex1, hex2) per ommatidium (-1 when the cell is
    unoccupied in MaleCNS), the overlap count, and the winning symmetry."""
    occupied = set(map(tuple, malecns_cols))
    best = (0,)
    for label, g in _hex_symmetries(zhao_hex):
        centre = np.round(malecns_cols.mean(0) - g.mean(0)).astype(int)
        for dx in range(-6, 7):
            for dy in range(-6, 7):
                offset = centre + (dx, dy)
                overlap = sum(tuple(p) in occupied for p in g + offset)
                if overlap > best[0]:
                    best = (overlap, label, offset, g + offset)
    overlap, label, offset, mapped = best
    hit = np.array([tuple(p) in occupied for p in mapped])
    mapped = np.where(hit[:, None], mapped, -1)
    return mapped, overlap, (label, tuple(int(o) for o in offset))


def build_eye(verbose: bool = True) -> Eye:
    rda = rdata.read_rda(EYEMAP_RDATA)
    utp = rda["utp_lens_rot"].to_numpy(float)  # rows are lens ids 1..852
    ixy = rda["lens_ixy"].astype(int)
    hexes = np.zeros((N_OMMATIDIA, 2), int)
    hexes[ixy[:, 0] - 1] = ixy[:, 1:]
    ucl = np.asarray(rda["ucl_rot_sm"], float)  # 778 medulla-matched lenses, head frame
    ucl_ids = rda["ucl_rot_sm"].coords["dim_0"].values.astype(int)
    up_rows = ucl_ids[rda["ind_Up_ucl"] - 1]
    down_rows = ucl_ids[rda["ind_Down_ucl"] - 1]

    lenses = _load_microct("lens")
    cones = _load_microct("cone")
    lenses = lenses[lenses[:, 1] > MICROCT_EYE_SPLIT_Y]
    cones = cones[cones[:, 1] > MICROCT_EYE_SPLIT_Y]
    if len(lenses) != N_OMMATIDIA or len(cones) != N_OMMATIDIA:
        raise RuntimeError(f"micro-CT right eye has {len(lenses)} lenses, {len(cones)} cones")
    radial = _unit(lenses - _fit_sphere_center(lenses))
    _, lens_row = _match_lenses_to_utp(radial, utp)
    order = np.argsort(lens_row)  # micro-CT lens index sorted by lens id
    lenses = lenses[order]
    radial = radial[order]

    cost = np.linalg.norm(lenses[:, None] - cones[None], axis=2)
    rows, cone_idx = linear_sum_assignment(cost)
    raw_dirs = _unit(lenses - cones[cone_idx])
    smooth_dirs = _smooth_on_lattice(raw_dirs, hexes)

    rot_head = _kabsch(smooth_dirs[ucl_ids - 1], ucl)
    dirs_right = smooth_dirs @ rot_head.T
    resid = _angle_deg(dirs_right[ucl_ids - 1], ucl)
    if np.median(resid) > 0.5:
        raise RuntimeError(f"head-frame fit residual median {np.median(resid):.2f} deg")
    dirs_left = dirs_right * np.array([1.0, -1.0, 1.0])

    equator_rows = np.array([np.unique(hexes[up_rows - 1].sum(1)).item(), np.unique(hexes[down_rows - 1].sum(1)).item()])

    columns = {}
    for side in ("R", "L"):
        cols = _malecns_columns(side)
        mapped, overlap, sym = register_lattice(hexes, cols)
        columns[side] = mapped
        if verbose:
            print(f"MaleCNS {side}: {len(cols)} columns, overlap {overlap}/{N_OMMATIDIA} = {overlap / N_OMMATIDIA:.4f}, symmetry (mirror, rot60, offset) {sym}")

    if verbose:
        az, el = azimuth_elevation(dirs_right)
        print(f"lens-cone distance um: median {np.median(cost[rows, cone_idx]):.2f}, max {cost[rows, cone_idx].max():.2f}")
        print(f"raw vs smoothed direction deg: median {np.median(_angle_deg(raw_dirs, smooth_dirs)):.3f}, max {_angle_deg(raw_dirs, smooth_dirs).max():.3f}")
        print(f"head-frame vs ucl_rot_sm deg (778): median {np.median(resid):.3f}, p90 {np.percentile(resid, 90):.3f}, max {resid.max():.3f}")
        print(f"skew view-vs-radial deg: median {np.median(_angle_deg(smooth_dirs, radial)):.2f}")
        band = np.abs(el) < 60.0  # azimuth is ill-defined near the dorsal pole
        print(f"right eye azimuth (|el|<60) [{az[band].min():.1f}, {az[band].max():.1f}], elevation [{el.min():.1f}, {el.max():.1f}]")
        print(f"binocular: {np.sum(dirs_right[:, 1] > 0)} right-eye ommatidia look into the left hemisphere; overlap width {-2 * az[band].min():.1f} deg")
        print(f"equator rows ix+iy = {equator_rows.tolist()}, elevation {el[up_rows - 1].mean():.2f} / {el[down_rows - 1].mean():.2f}")

    return Eye(
        lens_id=np.arange(1, N_OMMATIDIA + 1),
        hex=hexes,
        dirs_right=dirs_right,
        dirs_left=dirs_left,
        malecns_column_right=columns["R"],
        malecns_column_left=columns["L"],
        equator_rows=equator_rows,
    )


if __name__ == "__main__":
    eye = build_eye()
    eye.save()
    print(f"wrote {OMMATIDIA_NPZ}")
