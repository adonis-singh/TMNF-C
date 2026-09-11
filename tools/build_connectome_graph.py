#!/usr/bin/env python3
"""Build the MaleCNS v1.0 policy graph used by `tmnf_rl` `--arch connectome`.

Input: the three flat-connectome tables from gs://flyem-male-cns/v1.0/
(body annotations, per-neuron transmitter consensus, segment-to-segment
weights). Output: one .npz with the retained neurons and a CSR matrix of
signed, in-degree-normalised connection strengths.

Retained: every `Traced` neuron outside the optic lobes (ol_intrinsic,
ol_sensory), i.e. central brain, visual projection neurons and the ventral
nerve cord. Sign: GABA, glutamate and histamine inhibitory, everything else
excitatory (Shiu et al. 2024 convention). Edge strength: log1p(synapses),
rows normalised so each neuron's total absolute input is 1.

    build/venv/bin/python tools/build_connectome_graph.py \
        --data local/malecns --out local/malecns/graph_minw3.npz --min-weight 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

INHIBITORY = {"gaba", "glutamate", "histamine"}
AFFERENT = {"cb_sensory", "vnc_sensory", "sensory_ascending", "ascending_neuron", "visual_projection"}
EFFERENT = {"descending_neuron", "vnc_motor", "cb_motor"}
DROP = {"ol_intrinsic", "ol_sensory"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("local/malecns"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-weight", type=int, default=3, help="minimum synapse count per connection")
    args = parser.parse_args()
    d = args.data
    ann = feather.read_feather(d / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    nt = feather.read_feather(d / "body-neurotransmitters-male-cns-v1.0.feather")
    weights = feather.read_feather(d / "connectome-weights-male-cns-v1.0-minconf-0.5.feather")

    keep = ann[(ann.status == "Traced") & ~ann.superclass.isin(DROP)].copy()
    keep = keep.sort_values("bodyId").reset_index(drop=True)
    body = keep.bodyId.to_numpy(np.int64)
    index = pd.Series(np.arange(len(body)), index=body)
    superclass = keep.superclass.fillna("unknown").to_numpy()
    consensus = nt.set_index("body").consensus_nt.reindex(body).fillna("unclear").to_numpy()
    sign = np.where(np.isin(consensus, list(INHIBITORY)), -1.0, 1.0).astype(np.float32)
    afferent = np.isin(superclass, list(AFFERENT))
    efferent = np.isin(superclass, list(EFFERENT))

    e = weights[weights.body_pre.isin(index.index) & weights.body_post.isin(index.index)]
    e = e[e.weight >= args.min_weight]
    pre = index.reindex(e.body_pre.to_numpy()).to_numpy(np.int64)
    post = index.reindex(e.body_post.to_numpy()).to_numpy(np.int64)
    count = e.weight.to_numpy(np.float32)
    # CSR over the post-synaptic neuron (rows aggregate their inputs).
    order = np.lexsort((pre, post))
    pre, post, count = pre[order], post[order], count[order]
    strength = np.log1p(count) * sign[pre]
    total = np.zeros(len(body), np.float64)
    np.add.at(total, post, np.abs(strength))
    strength = (strength / np.where(total[post] > 0, total[post], 1.0)).astype(np.float32)
    indptr = np.zeros(len(body) + 1, np.int64)
    np.add.at(indptr, post + 1, 1)
    indptr = np.cumsum(indptr)

    np.savez(
        args.out, body_id=body, superclass=superclass, consensus_nt=consensus, sign=sign,
        afferent=afferent, efferent=efferent, indptr=indptr, indices=pre, strength=strength,
        synapse_count=count, min_weight=np.int64(args.min_weight),
        type=keep.type.fillna("").to_numpy(), soma_side=keep.somaSide.fillna("").to_numpy(),
    )
    print(f"neurons {len(body):,} (afferent {afferent.sum():,}, efferent {efferent.sum():,}), "
          f"edges {len(pre):,} at >= {args.min_weight} synapses, inhibitory neurons {(sign < 0).sum():,}; "
          f"isolated (no input) {(np.diff(indptr) == 0).sum():,} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
