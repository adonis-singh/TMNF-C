"""Build the derived data files from the downloads of ``tmnf_fly.fetch_data``, in dependency order.

    PYTHONPATH=python taskset -c 16-27 build/venv/bin/python -m tmnf_fly.build_data [--only eye|graph|mb|skeletons]

    eye        local/fly/ommatidia.npz        eye.build_eye: Zhao micro-CT viewing directions registered on
                                              the MaleCNS medulla column lattice (needs eyemap + malecns)
    graph      local/fly/brain_full.npz       brain_graph.write: the signed, row-normalised MaleCNS graph,
                                              min 2 synapses (needs malecns; reads the 1 GB weights table)
    mb         local/fly/mushroom_body.npz    mushroom_body.extract_circuit: PN/KC/MBON/DAN wiring (needs malecns)
    skeletons  local/fly/skeleton_sets.json   skeletons.build: population selection (seed 0), 6378 SWC
               local/fly/skeleton_npz/        downloads from the FlyEM bucket (~600 MB) packed per set (needs malecns)

The flyvis connectome cache (local/fly/flyvis_data/connectome/) is not built here: flyvis derives it from its
packaged fib25-fib19_v2.2.json the first time a network is loaded (tmnf_fly.flyvis_periphery).
"""

from __future__ import annotations

import argparse
import time

from tmnf_fly import FLY_DIR, LOCAL, brain_graph, eye, mushroom_body, skeletons


def build_eye() -> None:
    eye.build_eye().save()
    print(f"wrote {eye.OMMATIDIA_NPZ}", flush=True)


def build_graph() -> None:
    brain_graph.write()


def build_mb() -> None:
    circuit = mushroom_body.extract_circuit()
    print(mushroom_body.circuit_report(circuit), flush=True)
    print(f"wrote {mushroom_body.CIRCUIT_PATH}", flush=True)


def build_skeletons() -> None:
    sets = skeletons.build()
    print(f"wrote {FLY_DIR / 'skeleton_sets.json'} and {len(sets)} sets under {FLY_DIR / 'skeleton_npz'}", flush=True)


STEPS = {"eye": build_eye, "graph": build_graph, "mb": build_mb, "skeletons": build_skeletons}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(STEPS), action="append",
                        help="restrict to these steps (repeatable); default all, in dependency order")
    args = parser.parse_args()
    names = args.only or list(STEPS)
    print(f"data root {LOCAL}", flush=True)
    for name in names:
        t0 = time.perf_counter()
        print(f"== {name}", flush=True)
        STEPS[name]()
        print(f"== {name} done in {time.perf_counter() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
