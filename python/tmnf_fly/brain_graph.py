"""Build the full MaleCNS v1.0 graph (optic lobes kept) for the fly brain model.

    PYTHONPATH=python taskset -c 8-13 build/venv/bin/python -m tmnf_fly.brain_graph \
        [--data local/malecns] [--out local/fly/brain_full.npz] [--min-weight 2]

Retained: every `Traced` neuron. Sign: GABA, glutamate, histamine inhibitory,
everything else excitatory. Edge strength: log1p(synapses) * sign(pre), rows
(post-synaptic neuron) normalised so each neuron's total absolute input is 1.
Side: somaSide, or rootSide for neurons whose soma is outside the volume
(photoreceptors, peripheral sensory neurons). Hex columns: assignedOlHex1/2
from the annotations for the columnar types that carry them; every other
neuron of a flyvis cell type gets the column of its strongest same-side
partner (pre- or post-synaptic) that has an annotated column. ``columns/<side>``
lists the distinct (hex1, hex2) per side and ``column_index`` maps each neuron
into that list (-1: none).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from tmnf_fly import FLY_DIR, MALECNS_DIR
from tmnf_fly.populations import FLYVIS_MALECNS_TYPES, OPTIC_SUPERCLASSES, SIDES, all_population_keys

GRAPH_PATH = FLY_DIR / "brain_full.npz"
MIN_WEIGHT = 2
INHIBITORY = ("gaba", "glutamate", "histamine")
COLUMN_TARGETS = FLYVIS_MALECNS_TYPES


def combined_side(ann: pd.DataFrame) -> np.ndarray:
    side = ann.somaSide.astype(object).where(ann.somaSide.notna(), ann.rootSide.astype(object))
    side = side.where(side.isin(["L", "R", "M"]), "")
    return side.to_numpy(str)


def assign_columns_by_connectivity(hex1: np.ndarray, hex2: np.ndarray, types: np.ndarray, side: np.ndarray,
                                   pre: np.ndarray, post: np.ndarray, count: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Unannotated flyvis-type neurons: column of the strongest same-side partner (pre or post) with an annotated column."""
    annotated = hex1 >= 0
    target = np.isin(types, COLUMN_TARGETS) & ~annotated
    same = side[pre] == side[post]
    a = target[post] & annotated[pre] & same       # partner is presynaptic
    b = target[pre] & annotated[post] & same       # partner is postsynaptic
    df = pd.DataFrame({"neuron": np.concatenate([post[a], pre[b]]), "partner": np.concatenate([pre[a], post[b]]),
                       "count": np.concatenate([count[a], count[b]])})
    best = df.sort_values("count", ascending=False).drop_duplicates("neuron")
    hex1, hex2 = hex1.copy(), hex2.copy()
    hex1[best.neuron.to_numpy()] = hex1[best.partner.to_numpy()]
    hex2[best.neuron.to_numpy()] = hex2[best.partner.to_numpy()]
    source = np.where(annotated, "annotation", np.where(hex1 >= 0, "connectivity", ""))
    return hex1, hex2, source, int(len(best))


def column_tables(hex1: np.ndarray, hex2: np.ndarray, side: np.ndarray) -> dict[str, np.ndarray]:
    """Per side the sorted distinct (hex1, hex2) and a per-neuron index into it (-1 without column)."""
    out = {"column_index": np.full(len(hex1), -1, np.int32)}
    for s in SIDES:
        m = (side == s) & (hex1 >= 0)
        cols, inverse = np.unique(np.stack([hex1[m], hex2[m]], 1), axis=0, return_inverse=True)
        out[f"columns/{s}"] = cols.astype(np.int16)
        out["column_index"][m] = inverse.reshape(-1)
    return out


def build(data: Path, min_weight: int) -> dict[str, np.ndarray]:
    ann = feather.read_feather(data / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    nt = feather.read_feather(data / "body-neurotransmitters-male-cns-v1.0.feather")
    keep = ann[ann.status == "Traced"].sort_values("bodyId").reset_index(drop=True)
    body = keep.bodyId.to_numpy(np.int64)
    n = len(body)
    types = keep.type.fillna("").to_numpy(str)
    superclass = keep.superclass.fillna("unknown").to_numpy(str)
    side = combined_side(keep)
    consensus = nt.set_index("body").consensus_nt.reindex(body).fillna("unclear").to_numpy(str)
    sign = np.where(np.isin(consensus, INHIBITORY), -1.0, 1.0).astype(np.float32)
    hex1 = keep.assignedOlHex1.fillna(-1).to_numpy(np.int16)
    hex2 = keep.assignedOlHex2.fillna(-1).to_numpy(np.int16)
    photoreceptor = ann.type.fillna("").str.match(r"^R[1-8]")
    untraced_photoreceptors = int((photoreceptor & (ann.status != "Traced")).sum())

    weights = feather.read_feather(data / "connectome-weights-male-cns-v1.0-minconf-0.5.feather")
    total_rows = len(weights)
    index = pd.Index(body)
    pre = index.get_indexer(weights.body_pre.to_numpy(np.int64))
    post = index.get_indexer(weights.body_post.to_numpy(np.int64))
    count = weights.weight.to_numpy(np.float32)
    del weights
    traced_pair = (pre >= 0) & (post >= 0)
    traced_edges = int(traced_pair.sum())
    keep_edge = traced_pair & (count >= min_weight)
    pre, post, count = pre[keep_edge], post[keep_edge], count[keep_edge]
    order = np.lexsort((pre, post))
    pre, post, count = pre[order].astype(np.int64), post[order].astype(np.int64), count[order]

    hex1, hex2, hex_source, n_assigned = assign_columns_by_connectivity(hex1, hex2, types, side, pre, post, count)

    strength = np.log1p(count) * sign[pre]
    total = np.zeros(n, np.float64)
    np.add.at(total, post, np.abs(strength))
    strength = (strength / np.where(total[post] > 0, total[post], 1.0)).astype(np.float32)
    indptr = np.zeros(n + 1, np.int64)
    np.add.at(indptr, post + 1, 1)
    indptr = np.cumsum(indptr)

    graph: dict[str, np.ndarray] = dict(
        body_id=body, type=types, superclass=superclass, side=side, consensus_nt=consensus, sign=sign,
        hex1=hex1, hex2=hex2, hex_source=hex_source,
        optic=np.isin(superclass, OPTIC_SUPERCLASSES), descending=superclass == "descending_neuron",
        indptr=indptr, indices=pre, strength=strength, synapse_count=count,
        min_weight=np.int64(min_weight), total_weight_rows=np.int64(total_rows),
        traced_edges=np.int64(traced_edges), untraced_photoreceptors=np.int64(untraced_photoreceptors),
        columns_assigned_by_connectivity=np.int64(n_assigned),
    )
    graph.update(column_tables(hex1, hex2, side))
    graph.update(all_population_keys(types, superclass, side))
    return graph


def load_graph(path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=True))


def write(data: Path = MALECNS_DIR, out: Path = GRAPH_PATH, min_weight: int = MIN_WEIGHT) -> dict[str, np.ndarray]:
    """Build the graph, save it to ``out`` and print its summary."""
    graph = build(data, min_weight)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **graph)
    n = len(graph["body_id"])
    print(f"neurons {n:,} (optic {int(graph['optic'].sum()):,}, descending {int(graph['descending'].sum()):,}); "
          f"weight rows {int(graph['total_weight_rows']):,}, Traced-Traced edges {int(graph['traced_edges']):,}, "
          f"kept >= {min_weight} synapses: {len(graph['indices']):,}; "
          f"inhibitory neurons {int((graph['sign'] < 0).sum()):,}; no input {int((np.diff(graph['indptr']) == 0).sum()):,}; "
          f"columns by connectivity {int(graph['columns_assigned_by_connectivity'])}, "
          f"columns L {len(graph['columns/L'])} R {len(graph['columns/R'])}; sides {dict(zip(*np.unique(graph['side'], return_counts=True)))} -> {out}")
    return graph


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=MALECNS_DIR)
    parser.add_argument("--out", type=Path, default=GRAPH_PATH)
    parser.add_argument("--min-weight", type=int, default=MIN_WEIGHT)
    args = parser.parse_args()
    write(args.data, args.out, args.min_weight)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
