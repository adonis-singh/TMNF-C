"""Mushroom body circuit from the MaleCNS v1.0 connectome and the dopamine-gated learner.

Circuit (``extract_circuit`` -> local/fly/mushroom_body.npz):
  ALPN -> KC synapse counts (the fixed expansion), KC -> MBON synapse counts
  (the only plastic synapses), MBON compartments parsed from instance names,
  DAN compartments parsed from instance names or inferred from DAN -> MBON
  synapses, compartment class from the DAN type (PAM = reward, PPL1 =
  punishment). MBONs in reward compartments form the "avoid" group, MBONs in
  punishment compartments the "approach" group (Aso et al. 2014).

Learner (``MushroomBody``):
  sensory input -> fixed random sparse PN features -> real PN->KC matrix ->
  top-k KC code (APL winner-take-all) -> KC->MBON drives -> value =
  gain * (approach - avoid). The only weight change is
  dw = -eta * KC * (DAN_c - baseline) at KC->MBON synapses, with
  DAN_c - baseline = +RPE in PAM compartments and -RPE in PPL1 compartments;
  weights are bounded to [0, connectome weight].
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tmnf_fly import FLY_DIR, MALECNS_DIR

CONNECTOME_DIR = MALECNS_DIR
CIRCUIT_PATH = FLY_DIR / "mushroom_body.npz"

REWARD = 0  # PAM compartment: MBONs promote avoidance, depressed by positive RPE
PUNISHMENT = 1  # PPL1 compartment: MBONs promote approach, depressed by negative RPE
AVOID = 0
APPROACH = 1
EXCLUDED = -1

_LOBE = {"y": "g", "a": "a", "B": "b", "b": "b"}
# Minimum evidence to infer an unnamed DAN's compartment from its outputs.
_INFER_MIN_KC_SYNAPSES = 100
_INFER_MIN_MBON_SYNAPSES = 50


def parse_compartments(instance: str) -> list[str]:
    """``MBON01(y5B'2a)_R`` -> ['g5', "b'2"]; dendritic part before '>' / '<'."""
    match = re.search(r"\(([^)]*)\)", instance)
    if match is None:
        return []
    text = re.split(r"[<>]", match.group(1))[0]
    result: list[str] = []
    lobe = None
    prime = ""
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in _LOBE:
            lobe = _LOBE[ch]
            prime = ""
            if i + 1 < len(text) and text[i + 1] == "'":
                prime = "'"
                i += 1
        elif ch.isdigit() and lobe is not None:
            name = f"{lobe}{prime}{ch}"
            if name not in result:
                result.append(name)
        i += 1
    return result


def extract_circuit(connectome_dir: Path = CONNECTOME_DIR, out: Path = CIRCUIT_PATH) -> dict:
    ann = pd.read_feather(connectome_dir / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    ann = ann[ann["status"] == "Traced"]
    nt = pd.read_feather(connectome_dir / "body-neurotransmitters-male-cns-v1.0.feather")
    nt = nt.set_index("body")["consensus_nt"]

    def population(cls: str) -> pd.DataFrame:
        frame = ann[ann["class"] == cls][["bodyId", "type", "instance"]].copy()
        frame["type"] = frame["type"].fillna("").astype(str)
        frame["instance"] = frame["instance"].fillna("").astype(str)
        return frame.sort_values("bodyId").reset_index(drop=True)

    kc, mbon, dan, pn = population("Kenyon_Cell"), population("MBON"), population("DAN"), population("ALPN")
    bodies = {}
    for name, frame in (("kc", kc), ("mbon", mbon), ("dan", dan), ("pn", pn)):
        bodies[name] = {int(b): i for i, b in enumerate(frame["bodyId"])}

    weights = pd.read_feather(connectome_dir / "connectome-weights-male-cns-v1.0-minconf-0.5.feather")
    all_ids = set().union(*(set(d) for d in bodies.values()))
    weights = weights[weights["body_pre"].isin(all_ids) & weights["body_post"].isin(all_ids)]

    def matrix(pre: str, post: str) -> np.ndarray:
        rows = bodies[pre]
        cols = bodies[post]
        sub = weights[weights["body_pre"].isin(rows) & weights["body_post"].isin(cols)]
        out_matrix = np.zeros((len(rows), len(cols)), dtype=np.float32)
        r = sub["body_pre"].map(rows).to_numpy()
        c = sub["body_post"].map(cols).to_numpy()
        out_matrix[r, c] = sub["weight"].to_numpy(dtype=np.float32)
        return out_matrix

    pn_kc = matrix("pn", "kc")
    kc_mbon = matrix("kc", "mbon")
    dan_kc = matrix("dan", "kc")
    dan_mbon = matrix("dan", "mbon")

    # MBON compartments from instance names.
    mbon_comps = [parse_compartments(s) for s in mbon["instance"]]
    # DAN compartments: names first, otherwise the compartment of the MBON the
    # DAN synapses onto most (only for DANs that actually innervate KCs).
    dan_comps: list[list[str]] = []
    dan_inferred = np.zeros(len(dan), dtype=bool)
    for i, inst in enumerate(dan["instance"]):
        comps = parse_compartments(inst)
        if not comps:
            if dan_kc[i].sum() >= _INFER_MIN_KC_SYNAPSES and dan_mbon[i].max() >= _INFER_MIN_MBON_SYNAPSES:
                comps = list(mbon_comps[int(dan_mbon[i].argmax())])
                dan_inferred[i] = bool(comps)
        dan_comps.append(comps)

    names = sorted({c for comps in mbon_comps + dan_comps for c in comps})
    index = {c: i for i, c in enumerate(names)}
    n_comp = len(names)

    comp_class = np.full(n_comp, EXCLUDED, dtype=np.int64)
    dan_comp = np.zeros((len(dan), n_comp), dtype=bool)
    for i, comps in enumerate(dan_comps):
        if not comps:
            continue
        if not (dan["type"][i].startswith("PAM") or dan["type"][i].startswith("PPL1")):
            raise ValueError(f"DAN {dan['type'][i]} is neither PAM nor PPL1")
        cls = REWARD if dan["type"][i].startswith("PAM") else PUNISHMENT
        for name in comps:
            c = index[name]
            dan_comp[i, c] = True
            if comp_class[c] not in (EXCLUDED, cls):
                raise ValueError(f"compartment {name} receives both PAM and PPL1 DANs")
            comp_class[c] = cls

    mbon_comp = np.zeros((len(mbon), n_comp), dtype=bool)
    mbon_group = np.full(len(mbon), EXCLUDED, dtype=np.int64)
    for i, comps in enumerate(mbon_comps):
        classes = set()
        for c in comps:
            mbon_comp[i, index[c]] = True
            classes.add(int(comp_class[index[c]]))
        classes.discard(EXCLUDED)
        if len(classes) == 1:
            mbon_group[i] = AVOID if classes.pop() == REWARD else APPROACH

    mbon_nt = np.array([str(nt.get(int(b), "")) for b in mbon["bodyId"]])
    kc_type = kc["type"].to_numpy(dtype=str)

    result = {
        "kc_body": kc["bodyId"].to_numpy(np.int64),
        "kc_type": kc_type,
        "pn_body": pn["bodyId"].to_numpy(np.int64),
        "pn_type": pn["type"].to_numpy(dtype=str),
        "mbon_body": mbon["bodyId"].to_numpy(np.int64),
        "mbon_type": mbon["type"].to_numpy(dtype=str),
        "mbon_instance": mbon["instance"].to_numpy(dtype=str),
        "mbon_nt": mbon_nt,
        "mbon_comp": mbon_comp,
        "mbon_group": mbon_group,
        "dan_body": dan["bodyId"].to_numpy(np.int64),
        "dan_type": dan["type"].to_numpy(dtype=str),
        "dan_instance": dan["instance"].to_numpy(dtype=str),
        "dan_comp": dan_comp,
        "dan_inferred": dan_inferred,
        "comp_names": np.array(names),
        "comp_class": comp_class,
        "pn_kc": pn_kc,
        "kc_mbon": kc_mbon,
        "dan_kc_synapses": dan_kc.sum(axis=1),
        "dan_mbon": dan_mbon,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **result)
    return result


def load_circuit(path: Path = CIRCUIT_PATH) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _names(values) -> list[str]:
    return sorted({str(v) for v in values})


def circuit_report(c: dict) -> str:
    lines = []
    pn_kc = c["pn_kc"]
    kc_mbon = c["kc_mbon"]
    kc_in = (pn_kc > 0).sum(axis=0)
    lines.append(f"PNs (ALPN): {len(c['pn_body'])}, with KC output: {int((pn_kc.sum(axis=1) > 0).sum())}")
    lines.append(f"KCs: {len(c['kc_body'])}, with PN input: {int((kc_in > 0).sum())}, PNs per KC (with input): mean {kc_in[kc_in > 0].mean():.2f} median {np.median(kc_in[kc_in > 0]):.0f}")
    lines.append(f"PN->KC edges: {int((pn_kc > 0).sum())}, synapses: {int(pn_kc.sum())}")
    lines.append(f"KC->MBON edges: {int((kc_mbon > 0).sum())}, synapses: {int(kc_mbon.sum())}")
    lines.append(f"MBONs: {len(c['mbon_body'])}, DANs: {len(c['dan_body'])}, compartments: {len(c['comp_names'])}")
    names = c["comp_names"]
    for ci, name in enumerate(names):
        cls = {REWARD: "PAM/reward", PUNISHMENT: "PPL1/punish", EXCLUDED: "no DAN"}[int(c["comp_class"][ci])]
        m = np.flatnonzero(c["mbon_comp"][:, ci])
        d = np.flatnonzero(c["dan_comp"][:, ci])
        mtypes = _names(c["mbon_type"][m])
        dtypes = _names(c["dan_type"][d])
        lines.append(f"  {name:4s} {cls:12s} MBONs {len(m):2d} {mtypes} DANs {len(d):3d} {dtypes}")
    inferred = np.flatnonzero(c["dan_inferred"])
    lines.append("DAN compartments inferred from DAN->MBON synapses: "
                 + str(sorted({f"{c['dan_type'][i]}->{[str(n) for n in names[c['dan_comp'][i]]]}" for i in inferred})))
    unassigned = np.flatnonzero(~c["dan_comp"].any(axis=1))
    lines.append(f"DANs without compartment: {len(unassigned)} " + str(_names(c["dan_type"][unassigned])))
    for g, label in ((AVOID, "avoid (PAM compartments)"), (APPROACH, "approach (PPL1 compartments)"), (EXCLUDED, "excluded")):
        m = np.flatnonzero(c["mbon_group"] == g)
        lines.append(f"MBON group {label}: {len(m)} neurons, types {_names(c['mbon_type'][m])}")
    # Neurotransmitter cross-check: glutamate = avoidance, GABA/ACh = approach (Aso 2014).
    nt_group = np.where(c["mbon_nt"] == "glutamate", AVOID, APPROACH)
    used = c["mbon_group"] != EXCLUDED
    agree = int((nt_group[used] == c["mbon_group"][used]).sum())
    lines.append(f"NT cross-check (glutamate=avoid, GABA/ACh=approach): {agree}/{int(used.sum())} MBONs agree; disagreeing types "
                 + str(_names(c["mbon_type"][used & (nt_group != c["mbon_group"])])))
    return "\n".join(lines)


class MushroomBody:
    """Batched GPU mushroom body: fixed PN features, real PN->KC and KC->MBON wiring,
    dopamine-gated depression at KC->MBON synapses."""

    def __init__(
        self,
        circuit: dict,
        *,
        n_features: int,
        action_codes: np.ndarray,
        device: torch.device,
        seed: int,
        kc_active: int = 200,
        pn_state_inputs: int = 8,
        pn_action_fraction: float = 0.5,
        pn_quantile: float = 0.5,
        gain: float = 40.0,
        alpha: float = 0.02,
    ) -> None:
        self.device = device
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.action_codes = torch.as_tensor(action_codes, dtype=torch.float32, device=device)  # (K, A)
        self.n_actions, self.n_action_dims = self.action_codes.shape
        self.n_features = n_features
        self.kc_active = kc_active
        self.gain = gain
        self.alpha = alpha
        self.pn_state_inputs = pn_state_inputs
        self.pn_action_fraction = pn_action_fraction
        self.pn_quantile = pn_quantile

        pn_kc = torch.as_tensor(circuit["pn_kc"], device=device)
        self.n_pn, self.n_kc = pn_kc.shape
        total = pn_kc.sum(dim=0)
        self.kc_has_input = total > 0
        self.pn_kc = pn_kc / total.clamp_min(1.0)  # weighted mean of a KC's PN inputs
        self.kc_bias = torch.where(self.kc_has_input, torch.zeros_like(total), torch.full_like(total, -float("inf")))

        kc_mbon = torch.as_tensor(circuit["kc_mbon"], device=device)
        group = torch.as_tensor(circuit["mbon_group"], device=device)
        self.mbon_group = group
        self.avoid = group == AVOID
        self.approach = group == APPROACH
        self.n_mbon = kc_mbon.shape[1]
        self.w0 = kc_mbon.clone()  # normalised in calibrate()
        self.w = self.w0.clone()
        self.readout = torch.zeros(self.n_mbon, device=device)  # fixed MBON -> value gains
        self.eta = 0.0

        # Fixed random sparse PN features: each PN reads pn_state_inputs state
        # features; a fraction also reads one action-code dimension.
        u = torch.zeros(self.n_pn, n_features)
        for i in range(self.n_pn):
            cols = torch.randperm(n_features, generator=self.gen)[:pn_state_inputs]
            u[i, cols] = torch.randn(pn_state_inputs, generator=self.gen)
        reads_action = torch.rand(self.n_pn, generator=self.gen) < pn_action_fraction
        act_col = torch.randint(0, self.n_action_dims, (self.n_pn,), generator=self.gen)
        act_sign = torch.sign(torch.randn(self.n_pn, generator=self.gen))
        self.pn_u = u.to(device)
        self.pn_action_mask = torch.zeros(self.n_pn, self.n_action_dims, device=device)
        self.pn_action_mask[torch.arange(self.n_pn)[reads_action], act_col[reads_action]] = act_sign[reads_action].to(device)
        self.action_gain = 1.0
        self.feat_mean = torch.zeros(n_features, device=device)
        self.feat_scale = torch.ones(n_features, device=device)
        self.pn_theta = torch.zeros(self.n_pn, device=device)
        self.pn_scale = torch.ones(self.n_pn, device=device)

    # ----------------------------------------------------------------- sensory
    def _pn_pre(self, features: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        x = (features - self.feat_mean) / self.feat_scale
        return x @ self.pn_u.T + self.action_gain * (codes @ self.pn_action_mask.T)

    def _pn(self, features: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        return torch.relu(self._pn_pre(features, codes) - self.pn_theta) * self.pn_scale

    def kc_code(self, features: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """(B, F) features and (B, A) action codes -> (B, n_kc) bool top-k KC code."""
        h = self._pn(features, codes) @ self.pn_kc + self.kc_bias
        idx = torch.topk(h, self.kc_active, dim=1).indices
        kc = torch.zeros(h.shape[0], self.n_kc, dtype=torch.bool, device=self.device)
        kc.scatter_(1, idx, True)
        return kc

    def kc_all_actions(self, features: torch.Tensor) -> torch.Tensor:
        """(N, F) -> (N, K, n_kc) bool codes for every candidate action."""
        n = features.shape[0]
        f = features.unsqueeze(1).expand(n, self.n_actions, self.n_features).reshape(-1, self.n_features)
        c = self.action_codes.unsqueeze(0).expand(n, -1, -1).reshape(-1, self.n_action_dims)
        return self.kc_code(f, c).view(n, self.n_actions, self.n_kc)

    # ------------------------------------------------------------------- value
    def drives(self, kc: torch.Tensor) -> torch.Tensor:
        return (kc.to(torch.float32) @ self.w) / self.kc_active

    def value(self, kc: torch.Tensor) -> torch.Tensor:
        """MBON approach-minus-avoid drive, (..., n_kc) bool -> (...)."""
        shape = kc.shape[:-1]
        d = self.drives(kc.reshape(-1, self.n_kc))
        return (d @ self.readout).view(shape)

    # --------------------------------------------------------------- calibrate
    @torch.no_grad()
    def calibrate(self, features: torch.Tensor, target_action_overlap: float = 0.5) -> dict:
        """Fix feature normalisation, PN thresholds/gains, action-code gain and the
        KC->MBON normalisation from a batch of calibration states. Nothing here
        is learned from reward."""
        m = features.shape[0]
        self.feat_mean = features.mean(dim=0)
        self.feat_scale = features.std(dim=0).clamp_min(0.05)
        codes = self.action_codes[torch.randint(0, self.n_actions, (m,), generator=self.gen).to(self.device)]

        def set_pn(action_gain: float) -> None:
            self.action_gain = action_gain
            self.pn_theta = torch.zeros(self.n_pn, device=self.device)
            self.pn_scale = torch.ones(self.n_pn, device=self.device)
            z = self._pn_pre(features, codes)
            self.pn_theta = torch.quantile(z, self.pn_quantile, dim=0)
            p = torch.relu(z - self.pn_theta)
            self.pn_scale = 1.0 / p.mean(dim=0).clamp_min(1e-6)

        def action_overlap() -> float:
            sub = features[: min(m, 1024)]
            kc = self.kc_all_actions(sub)
            a = kc[:, 0]
            b = kc[:, 1:]
            inter = (a.unsqueeze(1) & b).sum(dim=2).float()
            return float((inter / self.kc_active).mean())

        lo, hi = 0.0, 64.0
        for _ in range(17):
            mid = 0.5 * (lo + hi)
            set_pn(mid)
            if action_overlap() > target_action_overlap:
                lo = mid
            else:
                hi = mid
        set_pn(0.5 * (lo + hi))
        overlap = action_overlap()

        # KC->MBON normalisation: each MBON's mean drive over calibration codes is 1,
        # the readout gives approach and avoid groups equal weight (value 0 at start).
        kc = self.kc_code(features, codes)
        drive0 = self.drives(kc).mean(dim=0)
        connected = drive0 > 0
        scale = torch.where(connected, 1.0 / drive0.clamp_min(1e-9), torch.zeros_like(drive0))
        self.w0 = self.w0 * scale
        self.w = self.w0.clone()
        self.readout = torch.zeros(self.n_mbon, device=self.device)
        n_app = int((self.approach & connected).sum())
        n_avd = int((self.avoid & connected).sum())
        self.readout[self.approach & connected] = self.gain / n_app
        self.readout[self.avoid & connected] = -self.gain / n_avd
        # Effective TD step: dV per unit RPE for the same code = gain * eta * sum_m |c_m| f_m,
        # f_m = fraction of the active code connected to MBON m. Set eta from alpha.
        conn = (self.w0 > 0).to(torch.float32)
        f = (kc.to(torch.float32) @ conn).mean(dim=0) / self.kc_active  # (n_mbon,)
        eff_app = float((f * self.readout.clamp_min(0)).sum())
        eff_avd = float((f * (-self.readout).clamp_min(0)).sum())
        self.eta = self.alpha / (eff_app + eff_avd)
        return {
            "action_gain": self.action_gain,
            "action_overlap": overlap,
            "pn_active_fraction": float((self._pn(features, codes) > 0).float().mean()),
            "eta": self.eta,
            "dv_per_rpe_approach": self.gain * self.eta * eff_app,
            "dv_per_rpe_avoid": self.gain * self.eta * eff_avd,
            "mbons_connected": int(connected.sum()),
            "approach_mbons_used": n_app,
            "avoid_mbons_used": n_avd,
        }

    # --------------------------------------------------------------- learning
    @torch.no_grad()
    def dopamine_update(self, kc: torch.Tensor, rpe: torch.Tensor) -> float:
        """dw = -eta * KC * (DAN_c - baseline_c) for a batch of (KC code, RPE) pairs.

        PAM (reward) compartments carry DAN - baseline = +RPE and act on the
        avoid MBONs; PPL1 (punishment) compartments carry -RPE and act on the
        approach MBONs. Dopamine above baseline with an active KC depresses the
        synapse, dopamine below baseline with an active KC lets it recover
        (Handler et al. 2019; Bennett et al. 2021). Weights stay in [0, w0].
        Returns mean |dw| over existing synapses."""
        kcf = kc.to(torch.float32)
        coincidence = kcf.T @ rpe  # (n_kc,) sum over the batch of KC x RPE
        before = self.w
        signed = torch.where(self.avoid.unsqueeze(0), coincidence.unsqueeze(1), torch.zeros_like(self.w))
        signed = torch.where(self.approach.unsqueeze(0), -coincidence.unsqueeze(1), signed)
        self.w = torch.minimum((self.w - self.eta * signed).clamp_min(0.0), self.w0)
        exist = self.w0 > 0
        return float((self.w - before).abs()[exist].mean())

    def saturation(self) -> tuple[float, float]:
        """Fraction of existing synapses at the floor (0) and at the ceiling (w0)."""
        exist = self.w0 > 0
        floor = float((self.w[exist] <= 0).float().mean())
        ceiling = float((self.w[exist] >= self.w0[exist]).float().mean())
        return floor, ceiling

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "w": self.w.cpu().numpy(),
            "w0": self.w0.cpu().numpy(),
            "readout": self.readout.cpu().numpy(),
            "mbon_group": self.mbon_group.cpu().numpy(),
            "pn_u": self.pn_u.cpu().numpy(),
            "pn_action_mask": self.pn_action_mask.cpu().numpy(),
            "pn_theta": self.pn_theta.cpu().numpy(),
            "pn_scale": self.pn_scale.cpu().numpy(),
            "feat_mean": self.feat_mean.cpu().numpy(),
            "feat_scale": self.feat_scale.cpu().numpy(),
            "action_codes": self.action_codes.cpu().numpy(),
            "action_gain": np.array(self.action_gain),
            "eta": np.array(self.eta),
            "gain": np.array(self.gain),
            "kc_active": np.array(self.kc_active),
        }

    STATE_TENSORS = ("w", "w0", "readout", "mbon_group", "pn_u", "pn_action_mask", "pn_theta", "pn_scale",
                     "feat_mean", "feat_scale", "action_codes")
    STATE_SCALARS = ("action_gain", "eta", "gain", "kc_active")

    def load_state_dict(self, state) -> None:
        """Restore everything ``state_dict`` saves. Every key is required and every
        tensor must have the shape the circuit and constructor arguments imply."""
        missing = [k for k in self.STATE_TENSORS + self.STATE_SCALARS if k not in state]
        if missing:
            raise KeyError(f"state is missing {missing}")
        for key in self.STATE_TENSORS:
            current = getattr(self, key)
            value = torch.as_tensor(np.asarray(state[key]), dtype=current.dtype, device=self.device)
            if value.shape != current.shape:
                raise ValueError(f"{key}: saved shape {tuple(value.shape)} != expected {tuple(current.shape)}")
            setattr(self, key, value)
        self.action_gain = float(state["action_gain"])
        self.eta = float(state["eta"])
        self.gain = float(state["gain"])
        self.kc_active = int(state["kc_active"])
        self.avoid = self.mbon_group == AVOID
        self.approach = self.mbon_group == APPROACH
        self.n_actions, self.n_action_dims = self.action_codes.shape


if __name__ == "__main__":
    circuit = extract_circuit()
    print(circuit_report(circuit))
    print(f"saved {CIRCUIT_PATH}")
