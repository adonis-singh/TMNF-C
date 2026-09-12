"""flyvis (Lappalainen et al., Nature 2024) as the visual periphery of the fly brain.

Model
-----
`flyvis` 1.2.0, pretrained task-optimised ensemble `flow/0000` (50 models, trained
on Sintel optic flow, sorted by validation loss: 000 best). One model is loaded from
`local/fly/flyvis_data/results/flow/0000/<NNN>/` at its best validation checkpoint.
Default `flow/0000/001` (validation loss 5.191, second best): of the 50 models it
has the largest minimum T4a-d direction selectivity (0.37-0.46 to drifting
gratings, see `flyvis_validate.py`) with the ensemble-consensus preferred
directions; 000 (loss 5.137) has T4d DSI 0.055 and silent L1/L2 (V < 0 for all
luminance in [0, 1]). Ensemble-wide, only 14/50 models put all four T4 subtypes
on the consensus directions with 4-direction gratings; T5 tuning is less
consistent (3/50).
Connectome `fib25-fib19_v2.2.json`, extent 15: 45669 nodes, 1513231 edges,
65 node types (64 cell types; CT1 is split into the compartments CT1(Lo1) and
CT1(M10)). 63 types are placed on the full hex lattice (721 columns); Lawf1 and
Lawf2 are tiled with stride (123 cells each). Absent (not in the FIB-25 medulla
connectome): all lobula/lobula-plate projection and tangential types (LC4, LC6,
LC10, LC11, LPLC1/2/4, HS, VS, H1/H2, CH, LPi), Dm/Pm amacrines, Tm5 (only Tm5Y,
Tm5a-c), Y cells, Li.

Lattice
-------
Oblique hex coordinates (u, v), integer, |u|,|v|,|u+v| <= 15 -> 721 columns.
Node order within every 721-column type is `flyvis.utils.hex_utils.get_hex_coords(15)`:
u ascending, then v ascending; index 360 is (0, 0). `lattice()` returns the
(721, 2) (u, v) array in this order.
Planar embedding (flyvis `hex_to_pixel(mode="default")`, used for all synthetic
stimuli in the paper): x = 1.5 v, y = -sqrt(3) (u + v/2), which is a regular
lattice with nearest-neighbour spacing sqrt(3); v runs along the horizontal image
axis, u + v/2 runs down the vertical image axis. The angular scale is
`HexEye.omm_width_rad = radians(5.8)` per neighbour step (also
`MovingBar.omm_width`); the Sintel training data were instead rendered with
`BoxEye(extent=15, kernel_size=13)` (13x13 px box mean per hexal at pixel
centres y = 13 (u + v/2), x = 13 v, frames resized so the 31-column lattice fills
the frame height), which has no angular calibration and is anisotropic
(u-step 13 px, v-step 14.5 px). We therefore use the regular 5.8 deg grid.
Viewing directions: the planar lattice point p = 5.8 deg * (x, y) / sqrt(3) is
mapped onto the unit sphere with the azimuthal equidistant projection about the
head +x axis (angular distance from the lattice centre = |p|); lattice +x is
rightward (head -y), lattice +y is up (head +z). Head frame: x forward, y left,
z up; the lattice centre looks along +x (azimuth 0, elevation 0). Radius
15 * 5.8 = 87 deg, i.e. the lattice covers a near-hemisphere. The same mapping is
used for both eyes (no mirroring): T4 subtype -> lattice direction is fixed by
the connectome (measured with drifting gratings in `flyvis_validate.py`, and the
ensemble consensus: T4a: -x (leftward on screen), T4b: +x (rightward), T4c: +y
(up), T4d: -y (down); T5a-d likewise where tuned). With lattice +x = head right,
"T4b" is front-to-back for the right eye but back-to-front for the left eye.

Input contract
--------------
* Stimulus tensor (batch, frames, 1, 721) luminance, one value per column,
  written identically to all eight photoreceptor nodes R1..R8 of that column
  (`Stimulus.input_index` is (8, 721)). The value enters the voltage ODE as an
  additive current with no further scaling: for every node,
      tau_eff dV/dt = -V + bias + sum_edges sign*n_syn*strength*relu(V_source) + x,
      tau_eff = max(tau_type, dt), explicit Euler with step dt.
* Range: Sintel frames are grey = PIL "L" / 255 in [0, 1]; the network's resting
  ("grey") input is 0.5 (`Network.steady_state(value=0.5)`). Training
  augmentation: contrast factor c ~ lognormal(0, 0.2) and brightness
  b ~ N(0, 0.1) applied as clamp(c (x - 0.5) + 0.5 + c b, min=0), plus Gaussian
  white noise std 0.08. So: feed luminance in [0, 1] with background ~0.5;
  values are not further normalised here.
* Time: training dt = 0.02 s (1/50 s; Sintel 24 fps piecewise-constant resampled
  to 50 Hz, 19 frames = 0.38 s per clip). The paper's synthetic-stimulus analyses
  integrate at dt = 1/200 s (`MovingBar.dt`). `Network.simulate` warns for
  dt > 1/50. Most learned time constants sit at 0.0195-0.02 s, i.e. exactly at
  the training-dt clamp, so those cells are instantaneous at dt = 0.02 and
  become 20 ms low-passes at smaller dt. `step(lum, dt)` holds the frame constant
  and integrates it in ceil(dt / 0.02) Euler substeps of dt/n <= 0.02 s.
* Initial state: `Network.simulate(initial_state="auto")` uses the state after
  1.0 s of constant 0.5 input (`steady_state(1.0, dt, batch)`); `fade_in_state`
  ramps the contrast of the first frame instead. `reset()` reproduces the 1 s
  grey steady state at dt = 0.02.
* Read-out: `state.nodes.activity` is the membrane voltage V of every node,
  (batch, 45669); per type via `connectome.nodes.layer_index[type]` (721 node
  indices in lattice order). Downstream cells see relu(V) (`PPNeuronIGRSynapses`,
  activation relu). `step` returns V per type; apply relu for a rate.
"""

from __future__ import annotations

import os

import numpy as np

from tmnf_fly import FLY_DIR

FLYVIS_DATA = FLY_DIR / "flyvis_data"
os.environ["FLYVIS_ROOT_DIR"] = str(FLYVIS_DATA)

import torch  # noqa: E402
import flyvis  # noqa: E402  (sets torch default device to cuda on import)
from flyvis.utils.hex_utils import get_hex_coords  # noqa: E402

DT_TRAIN = 0.02
GREY = 0.5
T_STEADY = 1.0
OMMATIDIAL_ANGLE_DEG = 5.8
EXTENT = 15
N_COLUMNS = 721


def lattice_plane() -> np.ndarray:
    """(721, 2) planar lattice coordinates in degrees: x right, y up, neighbour spacing 5.8."""
    u, v = get_hex_coords(EXTENT)
    x = 1.5 * v
    y = -np.sqrt(3.0) * (u + v / 2.0)
    return np.stack((x, y), axis=1) * (OMMATIDIAL_ANGLE_DEG / np.sqrt(3.0))


def plane_to_directions(plane_deg: np.ndarray) -> np.ndarray:
    """Azimuthal-equidistant map of planar (x right, y up) degrees to head-frame unit vectors."""
    r = np.deg2rad(np.linalg.norm(plane_deg, axis=1))
    unit = np.zeros_like(plane_deg)
    nz = r > 0
    unit[nz] = plane_deg[nz] / np.linalg.norm(plane_deg[nz], axis=1, keepdims=True)
    return np.stack((np.cos(r), -np.sin(r) * unit[:, 0], np.sin(r) * unit[:, 1]), axis=1)


class FlyvisPeriphery:
    """Stateful wrapper around one pretrained flyvis network; see module docstring."""

    def __init__(self, device: str | torch.device = "cuda", model: str = "flow/0000/001", batch_size: int = 1):
        self.device = torch.device(device)
        self.model = model
        torch.set_default_device(self.device)
        self.net = flyvis.NetworkView(model).init_network(checkpoint="best").to(self.device)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.clamp()
        self.params = self.net._param_api()
        conn = self.net.connectome
        self.cell_types = [t.decode() for t in conn.unique_cell_types[:]]
        node_type = conn.nodes.type[:]  # one read; conn.nodes.layer_index[t] costs ~1 s per type
        self.layer_index = {
            t: torch.as_tensor(np.nonzero(node_type == t.encode())[0], device=self.device) for t in self.cell_types
        }
        self.input_index = torch.as_tensor(self.net.stimulus.input_index, device=self.device)  # (8, 721)
        self.n_nodes = self.net.n_nodes
        u, v = get_hex_coords(EXTENT)
        self._uv = np.stack((u, v), axis=1).astype(np.int32)
        self._plane = lattice_plane()
        self._dirs = plane_to_directions(self._plane)
        self.state = None
        self.reset(batch_size)

    def lattice(self) -> tuple[np.ndarray, np.ndarray]:
        """((721, 2) int (u, v), (721, 3) unit viewing directions in the head frame)."""
        return self._uv.copy(), self._dirs.copy()

    def lattice_plane_deg(self) -> np.ndarray:
        """(721, 2) planar lattice coordinates in degrees (x right, y up)."""
        return self._plane.copy()

    @torch.no_grad()
    def reset(self, batch_size: int = 1) -> None:
        """State after 1 s of grey (0.5) input at the training dt, for `batch_size` samples."""
        self.batch_size = batch_size
        self.state = self.net._initial_state(self.params, batch_size)
        grey = torch.full((batch_size, N_COLUMNS), GREY, device=self.device)
        x = self._input_current(grey)
        for _ in range(round(T_STEADY / DT_TRAIN)):
            self.state = self.net._next_state(self.params, self.state, x, DT_TRAIN)

    def _input_current(self, lum: torch.Tensor) -> torch.Tensor:
        x = torch.zeros((lum.shape[0], self.n_nodes), device=self.device)
        x[:, self.input_index] = lum[:, None, :]
        return x

    @torch.no_grad()
    def step(self, luminance, dt: float) -> dict[str, torch.Tensor]:
        """Integrate one frame held for `dt` seconds; returns voltage per cell type, (B, cells)."""
        lum = torch.as_tensor(luminance, dtype=torch.float32, device=self.device)
        if lum.ndim == 1:
            lum = lum[None]
        if lum.shape != (self.batch_size, N_COLUMNS):
            raise ValueError(f"luminance shape {tuple(lum.shape)} != {(self.batch_size, N_COLUMNS)}; call reset(B)")
        n_sub = int(np.ceil(dt / DT_TRAIN - 1e-9))
        if n_sub < 1:
            raise ValueError(f"dt={dt} must be > 0")
        x = self._input_current(lum)
        sub = dt / n_sub
        for _ in range(n_sub):
            self.state = self.net._next_state(self.params, self.state, x, sub)
        return self.activity()

    def activity(self) -> dict[str, torch.Tensor]:
        """Current voltage per cell type, (B, 721) (Lawf1/Lawf2: (B, 123))."""
        act = self.state.nodes.activity
        return {t: act[:, idx] for t, idx in self.layer_index.items()}


__all__ = [
    "FlyvisPeriphery",
    "lattice_plane",
    "plane_to_directions",
    "DT_TRAIN",
    "GREY",
    "OMMATIDIAL_ANGLE_DEG",
    "N_COLUMNS",
]
