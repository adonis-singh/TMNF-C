"""Audit the visual -> descending pathways in ``local/fly/brain_full.npz`` with raw synapse counts.

    PYTHONPATH=python taskset -c 8-13 build/venv/bin/python -m tmnf_fly.pathway_audit

Writes ``local/fly/pathway_audit.json`` and prints the highlights.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tmnf_fly import FLY_DIR
from tmnf_fly.brain_graph import GRAPH_PATH, load_graph
from tmnf_fly.populations import population_indices


def edges_into(graph: dict, dst: np.ndarray) -> pd.DataFrame:
    """All CSR rows of the neurons in ``dst`` as (pre, post, synapses)."""
    indptr = graph["indptr"]
    starts, ends = indptr[dst], indptr[dst + 1]
    post = np.repeat(dst, ends - starts)
    take = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)]) if len(dst) else np.zeros(0, np.int64)
    return pd.DataFrame({"pre": graph["indices"][take], "post": post, "synapses": graph["synapse_count"][take]})


def edges_between(graph: dict, src: np.ndarray, dst: np.ndarray) -> pd.DataFrame:
    e = edges_into(graph, dst)
    return e[np.isin(e.pre.to_numpy(), src)]


def summarise(graph: dict, e: pd.DataFrame) -> dict:
    sign = graph["sign"][e.pre.to_numpy()]
    nt_by_syn = pd.Series(e.synapses.to_numpy(), index=graph["consensus_nt"][e.pre.to_numpy()]).groupby(level=0).sum()
    return {
        "synapses": int(e.synapses.sum()),
        "edges": int(len(e)),
        "pre_neurons": int(e.pre.nunique()),
        "post_neurons": int(e.post.nunique()),
        "excitatory_synapses": int(e.synapses.to_numpy()[sign > 0].sum()),
        "inhibitory_synapses": int(e.synapses.to_numpy()[sign < 0].sum()),
        "pre_nt_synapses": {k: int(v) for k, v in nt_by_syn.sort_values(ascending=False).items()},
    }


def by_side(graph: dict, src_name: str, dst_name: str) -> dict:
    out = {}
    for s_side in ("L", "R"):
        for d_side in ("L", "R"):
            e = edges_between(graph, population_indices(graph, src_name, s_side), population_indices(graph, dst_name, d_side))
            out[f"{s_side}->{d_side}"] = summarise(graph, e)
    out["all"] = summarise(graph, edges_between(graph, population_indices(graph, src_name), population_indices(graph, dst_name)))
    return out


def top_targets(graph: dict, src: np.ndarray, dst: np.ndarray, n: int = 10) -> list[dict]:
    e = edges_between(graph, src, dst)
    if e.empty:
        return []
    e = e.assign(type=graph["type"][e.post.to_numpy()], side=graph["side"][e.post.to_numpy()],
                 pre_side=graph["side"][e.pre.to_numpy()])
    per_type = e.groupby("type").agg(synapses=("synapses", "sum"), post_neurons=("post", "nunique"),
                                     pre_neurons=("pre", "nunique")).sort_values("synapses", ascending=False).head(n)
    rows = []
    for t, r in per_type.iterrows():
        sub = e[e.type == t]
        sides = sub.groupby("side").synapses.sum()
        ipsi = int(sub[sub.pre_side == sub.side].synapses.sum())
        rows.append({"type": t, "synapses": int(r.synapses), "post_neurons": int(r.post_neurons),
                     "pre_neurons": int(r.pre_neurons), "synapses_to_L": int(sides.get("L", 0)),
                     "synapses_to_R": int(sides.get("R", 0)), "ipsilateral_synapses": ipsi,
                     "contralateral_synapses": int(r.synapses) - ipsi})
    return rows


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else GRAPH_PATH
    graph = load_graph(path)
    dn = population_indices(graph, "DN")
    gf = population_indices(graph, "DNp01")
    dn_not_gf = dn[~np.isin(dn, gf)]
    audit: dict = {"graph": str(path), "min_weight": int(graph["min_weight"]), "edges": int(len(graph["indices"]))}

    audit["looming_to_GF"] = {src: by_side(graph, src, "DNp01") for src in ("LC4", "LPLC2", "LPLC1", "LC6", "LPLC4")}
    audit["ON_OFF_pathway"] = {
        f"{src}->{dst}": summarise(graph, edges_between(graph, population_indices(graph, src), population_indices(graph, dst)))
        for src, dst in (("R1-R6", "L1"), ("R1-R6", "L2"), ("L1", "Mi1"), ("L1", "Tm3"), ("Mi1", "T4"), ("Tm3", "T4"),
                         ("L2", "Tm1"), ("L2", "Tm2"), ("Tm1", "T5"), ("Tm2", "T5"), ("Tm9", "T5"), ("L3", "Tm9"))
    }
    audit["population_nt"] = {
        name: {str(k): int(c) for k, c in zip(*np.unique(graph["consensus_nt"][population_indices(graph, name)], return_counts=True))}
        for name in ("R1-R6", "L1", "L2", "L3", "Mi1", "Tm3", "Tm1", "Tm2", "Tm9", "T4", "T5", "HS", "VS_all", "H2", "CH",
                     "LC4", "LPLC2", "LPLC1", "LC6", "LC10", "LC11", "LC16", "LPLC4", "DNp01", "DNp02", "DNp04", "DNp11",
                     "DNa01", "DNa02", "DNp09", "MDN", "DNg02")
    }
    audit["T4T5_to_tangential"] = {
        f"{src}->{dst}": summarise(graph, edges_between(graph, population_indices(graph, src), population_indices(graph, dst)))
        for src in ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d", "T4", "T5")
        for dst in ("HS", "HST", "VS_all", "H2", "CH")
    }
    audit["tangential_to_DN_top"] = {
        src: top_targets(graph, population_indices(graph, src), dn) for src in ("HS", "VS_all", "H2", "CH", "HST")
    }
    audit["tangential_to_DN_totals"] = {
        src: summarise(graph, edges_between(graph, population_indices(graph, src), dn)) for src in ("HS", "VS_all", "H2", "CH")
    }
    audit["looming_to_DN_not_GF_top"] = {
        src: top_targets(graph, population_indices(graph, src), dn_not_gf, 15) for src in ("LC4", "LPLC2", "LPLC1", "LC6", "LC16", "LC11")
    }
    audit["named_DN_inputs_from_visual_projection"] = {}
    vp = np.flatnonzero(graph["superclass"] == "visual_projection")
    for name in ("DNp01", "DNp02", "DNp04", "DNp11", "DNa01", "DNa02", "DNp09", "MDN", "DNg02"):
        e = edges_between(graph, vp, population_indices(graph, name))
        if e.empty:
            audit["named_DN_inputs_from_visual_projection"][name] = []
            continue
        e = e.assign(type=graph["type"][e.pre.to_numpy()])
        top = e.groupby("type").synapses.sum().sort_values(ascending=False).head(8)
        audit["named_DN_inputs_from_visual_projection"][name] = [{"type": t, "synapses": int(v)} for t, v in top.items()]
    # Total input per named DN, to put the visual numbers in proportion.
    audit["named_DN_total_input"] = {
        name: int(edges_into(graph, population_indices(graph, name)).synapses.sum())
        for name in ("DNp01", "DNp02", "DNp04", "DNp11", "DNa01", "DNa02", "DNp09", "MDN", "DNg02")
    }

    out = FLY_DIR / "pathway_audit.json"
    out.write_text(json.dumps(audit, indent=1))

    for src in ("LC4", "LPLC2"):
        a = audit["looming_to_GF"][src]["all"]
        print(f"{src}->DNp01: {a['synapses']} synapses, {a['edges']} edges, exc {a['excitatory_synapses']} inh {a['inhibitory_synapses']}, "
              f"nt {a['pre_nt_synapses']}; sides " + ", ".join(f"{k} {v['synapses']}" for k, v in audit["looming_to_GF"][src].items() if k != "all"))
    for key, a in audit["ON_OFF_pathway"].items():
        print(f"{key}: {a['synapses']} synapses, exc {a['excitatory_synapses']} inh {a['inhibitory_synapses']}")
    for key in ("T4->HS", "T5->HS", "T4->VS_all", "T5->VS_all", "T4a->HS", "T4b->HS", "T4c->VS_all", "T4d->VS_all"):
        a = audit["T4T5_to_tangential"][key]
        print(f"{key}: {a['synapses']} synapses, {a['edges']} edges, {a['pre_neurons']} pre, exc {a['excitatory_synapses']} inh {a['inhibitory_synapses']}")
    for src in ("HS", "VS_all"):
        print(f"{src}->DN total {audit['tangential_to_DN_totals'][src]['synapses']} synapses; top:")
        for r in audit["tangential_to_DN_top"][src]:
            print(f"   {r['type']:14s} {r['synapses']:5d} syn  post {r['post_neurons']}  ipsi {r['ipsilateral_synapses']} contra {r['contralateral_synapses']}")
    for src in ("LC4", "LPLC2"):
        print(f"{src}->DN (not GF) top:")
        for r in audit["looming_to_DN_not_GF_top"][src][:10]:
            print(f"   {r['type']:14s} {r['synapses']:5d} syn  post {r['post_neurons']}  ipsi {r['ipsilateral_synapses']} contra {r['contralateral_synapses']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
