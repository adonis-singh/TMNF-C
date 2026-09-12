"""Named neuron populations of the MaleCNS v1.0 connectome used by the fly brain model.

Each population is a list of exact MaleCNS `type` strings (or a superclass for
the descending-neuron total). Sides are 'L' / 'R' / 'M' / '' (see
``brain_graph.combined_side``). Run as a script to print the inventory table
and the T4/T5 column-assignment statistics from ``local/fly/brain_full.npz``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from tmnf_fly import FLY_DIR

# name -> exact MaleCNS types
POPULATIONS: dict[str, list[str]] = {
    "R1-R6": ["R1-R6"],
    "R7": ["R7d", "R7p", "R7y", "R7_unclear"],
    "R8": ["R8d", "R8p", "R8y", "R8_unclear"],
    "R7R8_unclear": ["R7R8_unclear"],
    "L1": ["L1"], "L2": ["L2"], "L3": ["L3"],
    "Mi1": ["Mi1"], "Tm3": ["Tm3"], "Tm1": ["Tm1"], "Tm2": ["Tm2"], "Tm9": ["Tm9"],
    "T4a": ["T4a"], "T4b": ["T4b"], "T4c": ["T4c"], "T4d": ["T4d"],
    "T5a": ["T5a"], "T5b": ["T5b"], "T5c": ["T5c"], "T5d": ["T5d"],
    "T4": ["T4a", "T4b", "T4c", "T4d"],
    "T5": ["T5a", "T5b", "T5c", "T5d"],
    "HSN": ["HSN"], "HSE": ["HSE"], "HSS": ["HSS"], "HST": ["HST"],
    "HS": ["HSN", "HSE", "HSS"],
    # MaleCNS names the vertical-system cells as one type 'VS' (9/side) plus VST1, VST2, VSm.
    "VS": ["VS"], "VST1": ["VST1"], "VST2": ["VST2"], "VSm": ["VSm"],
    "VS_all": ["VS", "VST1", "VST2", "VSm"],
    "H1": ["H1"], "H2": ["H2"], "DCH": ["DCH"], "VCH": ["VCH"],
    "CH": ["DCH", "VCH"],
    "LC4": ["LC4"], "LPLC2": ["LPLC2"], "LPLC1": ["LPLC1"], "LPLC4": ["LPLC4"],
    "LC6": ["LC6"], "LC11": ["LC11"], "LC16": ["LC16"],
    "LC10a": ["LC10a"], "LC10b": ["LC10b"], "LC10c": ["LC10c-1", "LC10c-2"], "LC10d": ["LC10d"], "LC10e": ["LC10e"],
    "LC10": ["LC10a", "LC10b", "LC10c-1", "LC10c-2", "LC10d", "LC10e", "LC10_unclear"],
    "DNp01": ["DNp01"], "DNp02": ["DNp02"], "DNp04": ["DNp04"], "DNp11": ["DNp11"],
    "DNa01": ["DNa01"], "DNa02": ["DNa02"], "DNp09": ["DNp09"], "MDN": ["MDN"],
    "DNg02": ["DNg02_a", "DNg02_b", "DNg02_c", "DNg02_d", "DNg02_e", "DNg02_f", "DNg02_g"],
    # main direct targets of HS / VS (pathway audit) and of LC4 / LPLC2 besides the GF
    "DNp15": ["DNp15"], "DNp18": ["DNp18"], "DNg41": ["DNg41"], "DNp20": ["DNp20"], "DNg46": ["DNg46"], "DNp17": ["DNp17"],
    "DNp03": ["DNp03"], "DNg40": ["DNg40"], "DNp103": ["DNp103"],
}
# flyvis cell type -> MaleCNS types. Absent in MaleCNS v1.0: Mi3, Tm28. Renamed: Am -> Lai
# (lamina intrinsic amacrine; MaleCNS 'Am1' is a different, single medulla amacrine), TmY9 -> TmY9a+TmY9b,
# R7 / R8 -> the d/p/y/_unclear subtypes.
FLYVIS_TYPES: dict[str, list[str]] = {
    "R1-R6": ["R1-R6"], "R7": ["R7d", "R7p", "R7y", "R7_unclear"], "R8": ["R8d", "R8p", "R8y", "R8_unclear"],
    "L1": ["L1"], "L2": ["L2"], "L3": ["L3"], "L4": ["L4"], "L5": ["L5"],
    "Lawf1": ["Lawf1"], "Lawf2": ["Lawf2"], "Am": ["Lai"], "C2": ["C2"], "C3": ["C3"], "CT1": ["CT1"],
    "Mi1": ["Mi1"], "Mi2": ["Mi2"], "Mi4": ["Mi4"], "Mi9": ["Mi9"], "Mi10": ["Mi10"], "Mi13": ["Mi13"],
    "Mi14": ["Mi14"], "Mi15": ["Mi15"],
    "T1": ["T1"], "T2": ["T2"], "T2a": ["T2a"], "T3": ["T3"],
    "T4a": ["T4a"], "T4b": ["T4b"], "T4c": ["T4c"], "T4d": ["T4d"],
    "T5a": ["T5a"], "T5b": ["T5b"], "T5c": ["T5c"], "T5d": ["T5d"],
    "Tm1": ["Tm1"], "Tm2": ["Tm2"], "Tm3": ["Tm3"], "Tm4": ["Tm4"], "Tm5Y": ["Tm5Y"], "Tm5a": ["Tm5a"],
    "Tm5b": ["Tm5b"], "Tm5c": ["Tm5c"], "Tm9": ["Tm9"], "Tm16": ["Tm16"], "Tm20": ["Tm20"], "Tm30": ["Tm30"],
    "TmY3": ["TmY3"], "TmY4": ["TmY4"], "TmY5a": ["TmY5a"], "TmY9": ["TmY9a", "TmY9b"], "TmY10": ["TmY10"],
    "TmY13": ["TmY13"], "TmY14": ["TmY14"], "TmY15": ["TmY15"], "TmY18": ["TmY18"],
}
FLYVIS_ABSENT = ("Mi3", "Tm28")
FLYVIS_MALECNS_TYPES = tuple(sorted({t for ts in FLYVIS_TYPES.values() for t in ts}))
for _name, _types in FLYVIS_TYPES.items():
    POPULATIONS.setdefault(_name, _types)
SUPERCLASS_POPULATIONS: dict[str, str] = {"DN": "descending_neuron"}
SIDES = ("L", "R")
# Superclasses whose neurons live in the optic lobes (tau 10 ms in the model).
OPTIC_SUPERCLASSES = ("ol_intrinsic", "ol_sensory")


def population_mask(types: np.ndarray, superclass: np.ndarray, name: str) -> np.ndarray:
    if name in SUPERCLASS_POPULATIONS:
        return superclass == SUPERCLASS_POPULATIONS[name]
    return np.isin(types, POPULATIONS[name])


def population_indices(graph: dict, name: str, side: str | None = None) -> np.ndarray:
    key = f"pop/{name}/{side or 'all'}"
    if key not in graph:
        raise KeyError(f"population {name!r} side {side!r} not in graph")
    return graph[key]


def all_population_keys(types: np.ndarray, superclass: np.ndarray, side: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in list(POPULATIONS) + list(SUPERCLASS_POPULATIONS):
        mask = population_mask(types, superclass, name)
        out[f"pop/{name}/all"] = np.flatnonzero(mask).astype(np.int64)
        for s in SIDES:
            out[f"pop/{name}/{s}"] = np.flatnonzero(mask & (side == s)).astype(np.int64)
    return out


def inventory_table(graph: dict) -> str:
    side = graph["side"]
    rows = [f"{'population':14s} {'L':>6s} {'R':>6s} {'noside':>6s} {'total':>6s}  {'hex':>5s}"]
    for name in list(POPULATIONS) + list(SUPERCLASS_POPULATIONS):
        idx = population_indices(graph, name)
        s = side[idx]
        with_hex = int((graph["hex1"][idx] >= 0).sum())
        rows.append(f"{name:14s} {int((s == 'L').sum()):6d} {int((s == 'R').sum()):6d} "
                    f"{int((~np.isin(s, SIDES)).sum()):6d} {len(idx):6d}  {with_hex:5d}")
    return "\n".join(rows)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else FLY_DIR / "brain_full.npz"
    graph = dict(np.load(path, allow_pickle=True))
    print(inventory_table(graph))
    print(f"\n{'flyvis type':12s} {'malecns':22s} {'L':>5s} {'R':>5s} {'annot':>6s} {'conn':>6s} {'none':>5s}")
    for name, mtypes in FLYVIS_TYPES.items():
        idx = population_indices(graph, name)
        src, s = graph["hex_source"][idx], graph["side"][idx]
        print(f"{name:12s} {'+'.join(mtypes):22s} {int((s == 'L').sum()):5d} {int((s == 'R').sum()):5d} "
              f"{int((src == 'annotation').sum()):6d} {int((src == 'connectivity').sum()):6d} {int((src == '').sum()):5d}")
    print(f"absent in MaleCNS: {FLYVIS_ABSENT}")
    for s in SIDES:
        print(f"columns/{s}: {len(graph[f'columns/{s}'])} distinct (hex1, hex2)")
    untraced = graph["untraced_photoreceptors"]
    print(f"photoreceptors: traced {len(population_indices(graph, 'R1-R6')) + len(population_indices(graph, 'R7')) + len(population_indices(graph, 'R8')) + len(population_indices(graph, 'R7R8_unclear'))}, "
          f"not Traced in annotations {int(untraced)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
