"""3D brain-activity renderer: neuron skeletons as polylines, colour driven by per-neuron activity.

Requires an OpenGL context (VTK via GLX on the running display). All coordinates in micrometres.

    sets = skeletons.load()
    r = BrainRenderer(sets, Camera.frontal())
    img = r.frame({name: np.zeros(len(s)) for name, s in sets.items()})        # (H, W, 3) uint8
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyvista as pv
import vtk
from vtk.util import numpy_support

from tmnf_fly import FLY_DIR
from tmnf_fly.skeletons import SkeletonSet

pv.OFF_SCREEN = True

BRAIN_MESH = FLY_DIR / "JRCFIB2022M_brain.ply"  # gs://flyem-male-cns/rois/pointcloud-shells/, nanometres
BRAIN_CENTER = np.array([386.0, 228.0, 211.0], dtype=np.float32)  # centre of the brain shell bounds, um

# activity -> (r, g, b, alpha): 0 dim translucent grey-blue, 0.55 warm orange, 1 hot white-yellow
COLOR_STOPS = np.array([
    [0.00, 0.13, 0.17, 0.30, 0.03],
    [0.20, 0.26, 0.30, 0.46, 0.18],
    [0.45, 0.95, 0.45, 0.12, 0.85],
    [0.75, 1.00, 0.78, 0.30, 1.00],
    [1.00, 1.00, 0.98, 0.88, 1.00],
], dtype=np.float64)
BACKGROUND = (0.015, 0.016, 0.025)
SHELL_COLOR = (0.30, 0.38, 0.60)


@dataclass(frozen=True)
class Camera:
    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    view_up: tuple[float, float, float]
    view_angle: float = 30.0
    orbit_deg_per_s: float = 0.0  # slow orbit around view_up through the focal point; 0 = fixed

    @classmethod
    def frontal(cls, distance: float = 900.0, elevation_deg: float = 14.0, azimuth_deg: float = -24.0,
                orbit_deg_per_s: float = 0.0) -> "Camera":
        """Cinematic three-quarter view from the front (anterior, -z), slightly above (dorsal, -y)."""
        el, az = math.radians(elevation_deg), math.radians(azimuth_deg)
        d = np.array([math.sin(az) * math.cos(el), -math.sin(el), -math.cos(az) * math.cos(el)])
        pos = BRAIN_CENTER + distance * d
        return cls(tuple(pos), tuple(BRAIN_CENTER), (0.0, -1.0, 0.0), 30.0, orbit_deg_per_s)


# ---------------------------------------------------------------- geometry

@dataclass
class Polylines:
    """Decimated line geometry of several sets, ready for one VTK actor."""
    points: np.ndarray    # (N, 3) float32
    segments: np.ndarray  # (S, 2) int64 point indices
    owner: np.ndarray     # (N,) int64 index into the concatenated activity vector of the actor
    n_neurons: int


def decimate(s: SkeletonSet, spacing: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop chain nodes so that kept nodes are >= spacing apart along the arbor.

    Roots, branch points and tips are always kept. Returns (points, segments, owner_neuron_index).
    """
    xyz, parent = s.xyz, s.parent.astype(np.int64)
    n = len(xyz)
    has_par = parent >= 0
    seglen = np.zeros(n, dtype=np.float32)
    seglen[has_par] = np.linalg.norm(xyz[has_par] - xyz[parent[has_par]], axis=1)

    # path length from root by pointer doubling
    d = seglen.astype(np.float64)
    anc = parent.copy()
    while True:
        m = anc >= 0
        if not m.any():
            break
        d_prev, anc_prev = d.copy(), anc.copy()
        d[m] += d_prev[anc_prev[m]]
        anc[m] = anc_prev[anc_prev[m]]

    nchild = np.bincount(parent[has_par], minlength=n)
    bucket = np.floor(d / spacing).astype(np.int64)
    keep = ~has_par | (nchild != 1)
    keep[has_par] |= bucket[has_par] != bucket[parent[has_par]]

    # nearest kept ancestor
    anc = parent.copy()
    while True:
        bad = (anc >= 0) & ~keep[np.maximum(anc, 0)]
        if not bad.any():
            break
        anc[bad] = parent[anc[bad]]

    new_index = np.cumsum(keep) - 1
    kept = np.nonzero(keep)[0]
    kept_anc = anc[kept]
    child = kept[kept_anc >= 0]
    segments = np.stack([new_index[child], new_index[kept_anc[kept_anc >= 0]]], axis=1)
    owner = np.repeat(np.arange(len(s), dtype=np.int64), s.node_counts)[kept]
    return xyz[kept], segments, owner


def build_polylines(sets: list[SkeletonSet], spacing: float) -> Polylines:
    pts, segs, owners = [], [], []
    n_pts = n_neu = 0
    for s in sets:
        p, sg, ow = decimate(s, spacing)
        pts.append(p)
        segs.append(sg + n_pts)
        owners.append(ow + n_neu)
        n_pts += len(p)
        n_neu += len(s)
    return Polylines(np.concatenate(pts), np.concatenate(segs), np.concatenate(owners), n_neu)


def _lookup_table(n: int = 256) -> vtk.vtkLookupTable:
    x = np.linspace(0.0, 1.0, n)
    table = np.stack([np.interp(x, COLOR_STOPS[:, 0], COLOR_STOPS[:, k]) for k in (1, 2, 3, 4)], axis=1) * 255.0
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(n)
    lut.SetRange(0.0, 1.0)
    lut.SetTable(numpy_support.numpy_to_vtk(table.astype(np.uint8), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR))
    return lut


def load_brain_shell(path: Path = BRAIN_MESH) -> pv.PolyData:
    mesh = pv.read(path)
    mesh.points = (mesh.points / 1000.0).astype(np.float32)  # nm -> um
    return mesh


# ---------------------------------------------------------------- renderer

class BrainRenderer:
    """Draw neuron sets as polylines coloured by per-neuron activity in [0, 1].

    sets:    {name: SkeletonSet}; the activity dict passed to frame() must have one array of len(set) per name.
    camera:  Camera; if orbit_deg_per_s != 0 the view orbits with the frame time.
    thick:   set names drawn with thick_width (default: descending neurons).
    """

    def __init__(self, sets: dict[str, SkeletonSet], camera: Camera, size: tuple[int, int] = (1280, 720),
                 spacing_um: float = 2.0, thick: Iterable[str] | None = None,
                 line_width: float = 1.0, thick_width: float = 1.8, brain_mesh: Path | None = BRAIN_MESH,
                 shell_opacity: float = 0.07, depth_peels: int = 8, fxaa: bool = True):
        self.sets = dict(sets)
        self.names = list(self.sets)
        self.camera = camera
        self.size = size
        thick = set(thick) if thick is not None else {n for n in self.names if n == "DN" or n.startswith("DN")}
        for name in thick:
            if name not in self.sets:
                raise KeyError(name)
        self.groups = {"thin": [n for n in self.names if n not in thick], "thick": [n for n in self.names if n in thick]}
        self.groups = {k: v for k, v in self.groups.items() if v}

        t0 = time.time()
        self.plotter = pv.Plotter(off_screen=True, window_size=list(size), lighting="none")
        self.plotter.set_background(BACKGROUND)
        self.plotter.add_light(pv.Light(light_type="headlight", intensity=1.0))
        # order-independent blending of the translucent dim neurons; must be set before the window is created
        self.plotter.enable_depth_peeling(number_of_peels=depth_peels, occlusion_ratio=0.05)
        if fxaa:
            self.plotter.enable_anti_aliasing("fxaa")
        self._actors: dict[str, tuple[pv.PolyData, np.ndarray, Polylines, pv.Actor]] = {}
        self.n_points = self.n_segments = 0
        for group, names in self.groups.items():
            poly = build_polylines([self.sets[n] for n in names], spacing_um)
            cells = np.empty((len(poly.segments), 3), dtype=np.int64)
            cells[:, 0] = 2
            cells[:, 1:] = poly.segments
            mesh = pv.PolyData(poly.points, lines=cells.ravel())
            mesh.point_data["activity"] = np.zeros(len(poly.points), dtype=np.float32)
            actor = self.plotter.add_mesh(
                mesh, scalars="activity", clim=(0.0, 1.0), show_scalar_bar=False,
                line_width=thick_width if group == "thick" else line_width,
                render_lines_as_tubes=group == "thick", lighting=False,
            )
            actor.mapper.lookup_table = _lookup_table()
            actor.mapper.scalar_range = (0.0, 1.0)
            scalars = mesh.point_data["activity"]  # view onto the VTK buffer
            self._actors[group] = (mesh, scalars, poly, actor)
            self.n_points += len(poly.points)
            self.n_segments += len(poly.segments)
        if brain_mesh is not None:
            shell = load_brain_shell(brain_mesh)
            self.plotter.add_mesh(shell, color=SHELL_COLOR, opacity=shell_opacity, smooth_shading=True,
                                  specular=0.3, lighting=True)
        self._apply_camera(0.0)
        self.plotter.show(auto_close=False, interactive=False)  # creates the render window
        self.build_seconds = time.time() - t0
        print(f"BrainRenderer: {sum(len(s) for s in self.sets.values())} neurons, {self.n_points} points, "
              f"{self.n_segments} segments, built in {self.build_seconds:.1f} s", file=sys.stderr)

    # -- helpers

    def _apply_camera(self, t: float) -> None:
        cam = self.plotter.camera
        c = self.camera
        cam.position = c.position
        cam.focal_point = c.focal_point
        cam.up = c.view_up
        cam.view_angle = c.view_angle
        cam.parallel_projection = False
        if c.orbit_deg_per_s:
            cam.Azimuth(c.orbit_deg_per_s * t)
        self.plotter.renderer.ResetCameraClippingRange()

    # -- API

    def frame(self, activity: dict[str, np.ndarray], t: float = 0.0) -> np.ndarray:
        """Render one frame. activity[name] is a float array of len(sets[name]) in [0, 1]. Returns (H, W, 3) uint8."""
        for group, names in self.groups.items():
            mesh, scalars, poly, _ = self._actors[group]
            parts = []
            for n in names:
                a = np.asarray(activity[n], dtype=np.float32)
                if a.shape != (len(self.sets[n]),):
                    raise ValueError(f"activity[{n}] has shape {a.shape}, expected ({len(self.sets[n])},)")
                parts.append(a)
            act = np.concatenate(parts)
            scalars[:] = act[poly.owner]
            scalars.VTKObject.Modified()
        self._apply_camera(t)
        self.plotter.render()
        return self.plotter.screenshot(return_img=True)

    def close(self) -> None:
        vtk.vtkObject.GlobalWarningDisplayOff()  # tearing down the EGL window with depth peeling spams eglMakeCurrent warnings
        self.plotter.close()
        vtk.vtkObject.GlobalWarningDisplayOn()
