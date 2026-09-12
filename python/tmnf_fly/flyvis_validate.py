"""Validate the flyvis periphery: T4/T5 direction selectivity, looming, cell-type coverage, throughput.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python taskset -c 16-27 build/venv/bin/python -m tmnf_fly.flyvis_validate

Writes local/fly/flyvis_validation.json.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from tmnf_fly import FLY_DIR
from tmnf_fly.flyvis_periphery import DT_TRAIN, GREY, FlyvisPeriphery

OUT = FLY_DIR / "flyvis_validation.json"

DIRECTIONS = {"+x (right)": (1.0, 0.0), "-x (left)": (-1.0, 0.0), "+y (up)": (0.0, 1.0), "-y (down)": (0.0, -1.0)}
OPPOSITE = {"+x (right)": "-x (left)", "-x (left)": "+x (right)", "+y (up)": "-y (down)", "-y (down)": "+y (up)"}
T4 = ["T4a", "T4b", "T4c", "T4d"]
T5 = ["T5a", "T5b", "T5c", "T5d"]
STEP_TYPES = ["R1", "L1", "L2", "L3", "L4", "L5", "Mi1", "Tm3", "Mi4", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9"]
LOOM_TYPES = ["L1", "L2", "L3", "Mi1", "Tm3", "Mi4", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9", "CT1(Lo1)", "CT1(M10)"] + T4 + T5
ABSENT_CANDIDATES = [
    "LC4", "LPLC1", "LPLC2", "LPLC4", "LC6", "LC9", "LC10", "LC10a", "LC11", "LC12", "LC13", "LC15", "LC16", "LC17",
    "LC18", "LC20", "LC21", "LC22", "LC24", "LC25", "LC26", "LC31", "LT", "HS", "HSN", "HSE", "HSS", "VS", "VS1", "VS2",
    "H1", "H2", "CH", "LPi", "LPi3-4", "LPi4-3", "LPi2-1", "LPi1-2", "Dm", "Dm8", "Pm", "Tm5", "Y3", "TmY", "Li",
]
GRATING_WAVELENGTH_DEG = 30.0
GRATING_TEMPORAL_HZ = 1.0
GRATING_T = 2.0
GRATING_SKIP = 0.5
BAR_SPEED_DEG_S = 30.0
BAR_WIDTH_DEG = 5.8
LOOM_SPEED_DEG_S = 45.0
LOOM_T = 1.0
PRE_T = 0.5
CENTRAL_EXTENT = 6


def run(per: FlyvisPeriphery, frames: np.ndarray, dt: float) -> dict[str, np.ndarray]:
    """frames (T, 721) -> {type: (T, cells)} relu(V) on the CPU. Resets first."""
    per.reset(1)
    out = {t: [] for t in per.cell_types}
    for f in frames:
        act = per.step(f, dt)
        for t in per.cell_types:
            out[t].append(torch.relu(act[t][0]))
    return {t: torch.stack(v).cpu().numpy() for t, v in out.items()}


def central_mask(uv: np.ndarray) -> np.ndarray:
    return np.maximum.reduce([np.abs(uv[:, 0]), np.abs(uv[:, 1]), np.abs(uv[:, 0] + uv[:, 1])]) <= CENTRAL_EXTENT


def grating(plane: np.ndarray, d: tuple[float, float], dt: float) -> np.ndarray:
    n = round(GRATING_T / dt)
    t = np.arange(n) * dt
    phase = plane @ np.asarray(d) / GRATING_WAVELENGTH_DEG
    return GREY + GREY * np.sin(2 * np.pi * (phase[None, :] - GRATING_TEMPORAL_HZ * t[:, None]))


def bar(plane: np.ndarray, d: tuple[float, float], dt: float, on: bool) -> np.ndarray:
    """Bar of width BAR_WIDTH_DEG (lum 1 if on else 0) on grey, sweeping along d at BAR_SPEED across the lattice."""
    proj = plane @ np.asarray(d)
    start, stop = proj.min() - BAR_WIDTH_DEG, proj.max() + BAR_WIDTH_DEG
    n = round((stop - start) / BAR_SPEED_DEG_S / dt)
    pos = start + np.arange(n) * dt * BAR_SPEED_DEG_S
    inside = np.abs(proj[None, :] - pos[:, None]) < BAR_WIDTH_DEG / 2
    return np.where(inside, 1.0 if on else 0.0, GREY)


def loom(plane: np.ndarray, dt: float) -> np.ndarray:
    n = round(LOOM_T / dt)
    r = np.arange(n) * dt * LOOM_SPEED_DEG_S
    dist = np.linalg.norm(plane, axis=1)
    return np.where(dist[None, :] < r[:, None], 0.0, GREY)


def with_pre(frames: np.ndarray, dt: float) -> np.ndarray:
    return np.concatenate([np.full((round(PRE_T / dt), frames.shape[1]), GREY), frames])


def peak_response(r: np.ndarray, n_pre: int, cen: np.ndarray) -> float:
    """Per-cell peak of relu(V) during the stimulus minus that cell's grey baseline, averaged over central cells."""
    base = r[:n_pre, cen].mean(axis=0)
    return float(np.maximum(r[n_pre:, cen].max(axis=0) - base, 0.0).mean())


def dsi_table(responses: dict[str, dict[str, float]]) -> dict[str, dict]:
    table = {}
    for ct, per_dir in responses.items():
        pref = max(per_dir, key=per_dir.get)
        rp, rn = per_dir[pref], per_dir[OPPOSITE[pref]]
        dsi = (rp - rn) / (rp + rn) if rp + rn > 0 else 0.0  # silent cell: not direction selective
        table[ct] = {"pref": pref, "R_pref": rp, "R_null": rn, "DSI": dsi, "R": per_dir}
    return table


def main() -> None:
    t0 = time.perf_counter()
    per = FlyvisPeriphery("cuda")
    load_s = time.perf_counter() - t0
    dt = DT_TRAIN
    uv, _ = per.lattice()
    plane = per.lattice_plane_deg()
    cen = central_mask(uv)
    n_pre = round(PRE_T / dt)
    report: dict = {"model": per.model, "dt": dt, "load_s": load_s}

    present = per.cell_types
    report["cell_types"] = present
    report["absent"] = [c for c in ABSENT_CANDIDATES if c not in present]
    report["absent_present_check"] = [c for c in ABSENT_CANDIDATES if c in present]
    report["n_cells_per_type"] = {t: int(per.layer_index[t].numel()) for t in present}

    # (a) drifting sine gratings
    skip = n_pre + round(GRATING_SKIP / dt)
    grat = {ct: {} for ct in T4 + T5}
    for name, d in DIRECTIONS.items():
        r = run(per, with_pre(grating(plane, d, dt), dt), dt)
        for ct in T4 + T5:
            grat[ct][name] = float(r[ct][skip:, cen].mean())
    report["grating"] = {
        "wavelength_deg": GRATING_WAVELENGTH_DEG, "temporal_hz": GRATING_TEMPORAL_HZ, "speed_deg_s": GRATING_WAVELENGTH_DEG * GRATING_TEMPORAL_HZ,
        "measure": "mean relu(V) over central columns (extent<=6), 0.5 s after onset to 2 s", "dsi": dsi_table(grat),
    }

    # (a') moving bars on grey: ON bar for T4, OFF bar for T5, peak of relu(V) minus grey baseline
    bars = {}
    for group, on in ((T4, True), (T5, False)):
        resp = {ct: {} for ct in group}
        for name, d in DIRECTIONS.items():
            r = run(per, with_pre(bar(plane, d, dt, on), dt), dt)
            for ct in group:
                resp[ct][name] = peak_response(r[ct], n_pre, cen)
        bars["ON" if on else "OFF"] = dsi_table(resp)
    report["bar"] = {
        "speed_deg_s": BAR_SPEED_DEG_S, "width_deg": BAR_WIDTH_DEG, "bg": GREY,
        "measure": "per-cell peak over time of relu(V) minus grey baseline, mean over central columns", "dsi": bars,
    }

    # (b) looming dark disc vs full-field OFF step
    lm = with_pre(loom(plane, dt), dt)
    flash = with_pre(np.zeros((round(LOOM_T / dt), plane.shape[0])), dt)
    r_loom = run(per, lm, dt)
    r_flash = run(per, flash, dt)
    loom_rep = {}
    for ct in LOOM_TYPES:
        base = float(r_loom[ct][:n_pre, cen].mean())
        loom_rep[ct] = {
            "baseline": base,
            "loom_mean": float(r_loom[ct][n_pre:, cen].mean()),
            "loom_peak": float(r_loom[ct][n_pre:, cen].mean(axis=1).max()),
            "flash_mean": float(r_flash[ct][n_pre:, cen].mean()),
            "flash_peak": float(r_flash[ct][n_pre:, cen].mean(axis=1).max()),
        }
    report["loom"] = {"speed_deg_s": LOOM_SPEED_DEG_S, "duration_s": LOOM_T, "disc_lum": 0.0, "bg_lum": GREY, "types": loom_rep}

    # raw voltages (no relu) of early types to full-field luminance steps grey/dark/grey/bright/grey
    seq = np.concatenate([np.full((25, 721), GREY), np.zeros((25, 721)), np.full((25, 721), GREY), np.ones((25, 721)), np.full((25, 721), GREY)])
    per.reset(1)
    volt = {t: [] for t in STEP_TYPES}
    for f in seq:
        act = per.step(f, dt)
        for t in STEP_TYPES:
            volt[t].append(act[t][0][torch.as_tensor(cen, device=per.device)].mean().item())
    report["luminance_step_V"] = {
        t: {"grey": v[20], "dark": v[45], "grey2": v[70], "bright": v[95], "min": min(v), "max": max(v)} for t, v in volt.items()
    }

    # (c) throughput
    thr = {}
    for B in (1, 64):
        per.reset(B)
        x = torch.full((B, 721), GREY, device=per.device)
        for _ in range(10):
            per.step(x, dt)
        torch.cuda.synchronize()
        t = time.perf_counter()
        n = 200
        for _ in range(n):
            per.step(x, dt)
        torch.cuda.synchronize()
        el = (time.perf_counter() - t) / n
        thr[f"B={B}"] = {"ms_per_step": el * 1e3, "steps_per_s": 1 / el, "frames_per_s": B / el, "peak_mem_MB": torch.cuda.max_memory_allocated() / 1e6}
    report["throughput"] = thr

    OUT.write_text(json.dumps(report, indent=1))

    print(f"load {load_s:.1f} s; {len(present)} node types; absent: {report['absent']}")
    if report["absent_present_check"]:
        print("present from candidate list:", report["absent_present_check"])
    for key in ("grating",):
        print(f"\n{key} DSI:")
        for ct, row in report[key]["dsi"].items():
            print(f"  {ct}: pref {row['pref']:<11} DSI {row['DSI']:.3f}  R_pref {row['R_pref']:.4f} R_null {row['R_null']:.4f}  " + " ".join(f"{k}={v:.4f}" for k, v in row["R"].items()))
    for pol, tab in bars.items():
        print(f"\n{pol} bar DSI:")
        for ct, row in tab.items():
            print(f"  {ct}: pref {row['pref']:<11} DSI {row['DSI']:.3f}  R_pref {row['R_pref']:.4f} R_null {row['R_null']:.4f}")
    print("\nloom (baseline / loom_mean / loom_peak / flash_mean / flash_peak):")
    for ct, row in loom_rep.items():
        print(f"  {ct:<9} {row['baseline']:.4f} {row['loom_mean']:.4f} {row['loom_peak']:.4f} {row['flash_mean']:.4f} {row['flash_peak']:.4f}")
    print("\nfull-field step V (grey / dark / grey / bright; min max):")
    for t, row in report["luminance_step_V"].items():
        print(f"  {t:<4} {row['grey']:+.3f} {row['dark']:+.3f} {row['grey2']:+.3f} {row['bright']:+.3f}   {row['min']:+.3f} {row['max']:+.3f}")
    print("\nthroughput:", json.dumps(thr, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
