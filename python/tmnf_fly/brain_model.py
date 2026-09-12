"""Rate-based dynamics over the MaleCNS graph built by ``tmnf_fly.brain_graph``.

    tau_i dv_i/dt = -v_i + g * sum_j W_ij r_j + I_i,      r = relu(v)

W is the signed, row-normalised CSR from the graph (fixed), g a global synaptic
gain, tau per neuron by superclass (optic lobe / central / descending), I an
external current set on named populations. Euler integration at ``dt`` on the
GPU with ``torch.sparse.mm``.
``clamp`` holds a population's rates at given values every substep (they still
drive their targets through W; their own v is ignored while clamped), which is
how the flyvis periphery is fed in: per 10 ms tick

    for t in flyvis_types:
        for s in ("L", "R"):
            model.clamp(f"{t}/{s}", model.scatter_columns(s, t, act[t][s]))
    model.step({}, 10)

``scatter_columns`` maps a per-column vector (ordered like ``columns/<side>`` in
the graph) onto the neurons of ``<type>/<side>`` through ``column_index``;
neurons of the type without a column get 0.

Population names are ``"<name>"`` (both sides) or ``"<name>/<L|R>"`` using the
``pop/`` index arrays stored in the graph, e.g. ``"L1/R"``, ``"DNp01"``.
"""
from __future__ import annotations

import numpy as np
import torch

TAU_MS_DEFAULT = {"optic": 10.0, "central": 20.0, "descending": 20.0}


class BrainModel:
    def __init__(self, graph: dict[str, np.ndarray], device: str | torch.device, dt: float = 0.001,
                 gain: float = 1.0, tau_ms: dict[str, float] = TAU_MS_DEFAULT) -> None:
        self.device = torch.device(device)
        self.dt = float(dt)
        self.n = int(len(graph["body_id"]))
        n = self.n
        indptr = torch.from_numpy(graph["indptr"].astype(np.int64))
        indices = torch.from_numpy(graph["indices"].astype(np.int64))
        strength = torch.from_numpy(graph["strength"].astype(np.float32))
        self.w = torch.sparse_csr_tensor(indptr, indices, strength, size=(n, n)).to(self.device)
        self.edges = int(len(indices))
        self.gain = float(gain)
        tau = np.full(n, tau_ms["central"], np.float32)
        tau[graph["optic"]] = tau_ms["optic"]
        tau[graph["descending"]] = tau_ms["descending"]
        self.tau = torch.from_numpy(tau).to(self.device) / 1000.0
        self.decay = self.dt / self.tau                       # dt / tau_i
        self.populations = {k[len("pop/"):]: torch.from_numpy(v.astype(np.int64)).to(self.device)
                            for k, v in graph.items() if k.startswith("pop/")}
        self.column_index = torch.from_numpy(graph["column_index"].astype(np.int64)).to(self.device)
        self.n_columns = {s: int(len(graph[f"columns/{s}"])) for s in ("L", "R")}
        self.v = torch.zeros(n, device=self.device)
        self.r = torch.zeros(n, 1, device=self.device)
        self.current = torch.zeros(n, device=self.device)
        self.t = 0.0
        self._clamps: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._clamp_idx = torch.zeros(0, dtype=torch.int64, device=self.device)
        self._clamp_val = torch.zeros(0, device=self.device)
        self._clamp_dirty = False

    # -- population helpers -------------------------------------------------
    def indices(self, name: str) -> torch.Tensor:
        key = name if "/" in name else f"{name}/all"
        if key not in self.populations:
            raise KeyError(f"unknown population {name!r}")
        return self.populations[key]

    def rates(self, name: str) -> torch.Tensor:
        return self.r[self.indices(name), 0]

    def mean_rate(self, name: str) -> float:
        return float(self.rates(name).mean())

    # -- input ----------------------------------------------------------------
    def inject(self, indices: torch.Tensor, values: torch.Tensor | float) -> None:
        self.current[indices] = torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def reset(self) -> None:
        self.v.zero_()
        self.r.zero_()
        self.current.zero_()
        self.t = 0.0
        self.unclamp()

    # -- clamping (flyvis periphery) ----------------------------------------
    def clamp(self, name: str, values: torch.Tensor) -> None:
        idx = self.indices(name)
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if values.shape != idx.shape:
            raise ValueError(f"clamp {name!r}: {tuple(values.shape)} values for {len(idx)} neurons")
        self._clamps[name] = (idx, values)
        self._clamp_dirty = True

    def unclamp(self, name: str | None = None) -> None:
        if name is None:
            self._clamps.clear()
        else:
            del self._clamps[name]
        self._clamp_dirty = True

    def scatter_columns(self, side: str, type_name: str, column_values: torch.Tensor) -> torch.Tensor:
        """Per-column vector (length ``n_columns[side]``) -> per-neuron vector over ``<type>/<side>``."""
        column_values = torch.as_tensor(column_values, dtype=torch.float32, device=self.device)
        if column_values.shape != (self.n_columns[side],):
            raise ValueError(f"expected {self.n_columns[side]} column values for side {side}, got {tuple(column_values.shape)}")
        col = self.column_index[self.indices(f"{type_name}/{side}")]
        return torch.where(col >= 0, column_values[col.clamp(min=0)], torch.zeros((), device=self.device))

    def _refresh_clamps(self) -> None:
        if self._clamps:
            self._clamp_idx = torch.cat([i for i, _ in self._clamps.values()])
            self._clamp_val = torch.cat([v for _, v in self._clamps.values()])
        else:
            self._clamp_idx = self._clamp_idx[:0]
            self._clamp_val = self._clamp_val[:0]
        self._clamp_dirty = False

    # -- integration ----------------------------------------------------------
    @torch.no_grad()
    def substep(self) -> None:
        if self._clamp_dirty:
            self._refresh_clamps()
        drive = torch.sparse.mm(self.w, self.r)[:, 0]
        self.v += self.decay * (self.gain * drive + self.current - self.v)
        self.v[self._clamp_idx] = self._clamp_val          # so unclamping is continuous
        r = torch.relu(self.v)
        r[self._clamp_idx] = self._clamp_val
        self.r = r.unsqueeze(1)
        self.t += self.dt

    @torch.no_grad()
    def step(self, inputs: dict[str, torch.Tensor | float], n_substeps: int) -> None:
        """Set the external current from ``inputs`` (population name -> value(s)) and integrate."""
        self.current.zero_()
        for name, value in inputs.items():
            self.inject(self.indices(name), value)
        for _ in range(n_substeps):
            self.substep()


__all__ = ["BrainModel", "TAU_MS_DEFAULT"]
