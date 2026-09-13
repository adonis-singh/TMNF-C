# tmnf_fly: a fruit fly brain plays TrackMania

The MaleCNS v1.0 connectome (Janelia FlyEM / Google, September 2026: 166,700
neurons of an adult male *Drosophila* central nervous system) wired into the
TMNF-C simulator. Two things drive the car:

1. **A dopamine-trained mushroom body** (`mushroom_body.py`, `train_mb.py`).
   The fly's learning circuit, with its real wiring, learns a track by trial
   and error. No gradients, no neural-network policy: reward-prediction error
   is carried by the PAM and PPL1 dopamine neurons of the connectome and
   depresses Kenyon-cell to MBON synapses. This is the part that learns laps.
2. **A reflex driver** (`drive.py`). A measured compound eye renders the game
   world, a fitted optic-lobe model (flyvis) turns it into cell-type activity,
   that activity is clamped onto the same-typed MaleCNS neurons column by
   column, the MaleCNS wiring is integrated as a rate model, and the car is
   steered from named descending neurons (Giant Fiber brake, LC4/LPLC2 escape
   steering, HS-cell optomotor steering). It avoids walls but finishes
   nothing; it is kept as the fly's-eye view for the videos.

Everything is reproducible from public data. Game assets are not included;
see the [main README](../../README.md) and [local asset setup](../../docs/LOCAL_ASSETS.md).

## Videos

Laps driven by the mushroom body after 5, 30 and 60 minutes of training
(A04-Acrobatic 6.58 s and 6.29 s, A02-Race 21.69 s), composed by `video.py`:
game chase view, the fly's compound-eye view, the MaleCNS skeletons coloured
by activity, the track map and MBON value / dopamine / LC4 traces. The videos,
trained weights, per-decision circuit activity and training logs are in the
[fly-v1 release](https://github.com/adonis-singh/TMNF-C/releases/tag/fly-v1),
not in the repository.

## The mushroom body learner

Circuit, extracted from the connectome tables by `mushroom_body.extract_circuit`:

| | count |
| --- | ---: |
| Antennal-lobe projection neurons (ALPN) | 686 (314 with KC output) |
| Kenyon cells (KC) | 4,064 (3,812 with PN input; 5.9 PNs per KC) |
| PN to KC edges / synapses | 22,586 / 390,928 |
| MBONs | 97 |
| KC to MBON edges / synapses | 61,210 / 463,640 |
| Dopamine neurons (DAN) | 340 (316 PAM, 12 PPL1, 12 without a compartment) |
| Compartments | 15 (a1..a3, a'1..a'3, b1, b2, b'1, b'2, g1..g5) |

How it learns, per 50 ms decision (5 physics ticks):

- The car's 193 state features (the RL encoder's flat observation: pose,
  velocity, wheel contacts, route lookahead) plus a one-hot code of a
  candidate action drive the PNs through fixed random sparse projections.
  PN thresholds and gains are calibrated once from 400 steps of random driving
  and never learned.
- PNs drive the KCs through the real PN to KC synapse counts; the 200 most
  driven KCs (4.9 %) fire, a sparse code of (state, action).
- MBONs read the KC code through the real KC to MBON synapse counts. The
  value of a (state, action) is the summed drive of the approach MBONs minus
  the avoid MBONs, with compartment membership taken from which DAN class
  (PAM reward or PPL1 punishment) innervates each MBON.
- The action with the highest value is taken (epsilon-greedy, 0.1 to 0.02).
- The TD reward-prediction error becomes dopamine: PAM compartments carry
  +RPE and depress the active KC synapses onto their (avoid) MBONs, PPL1
  compartments carry -RPE and depress the active KC synapses onto their
  (approach) MBONs (Handler et al. 2019, Bennett et al. 2021). Weights stay in
  [0, w0]; below-baseline dopamine lets them recover.

The reward is the simulator's potential-shaped race reward (progress along
the route at a reference speed, a small per-tick cost, a lump cost for
failing with budget left), scaled by 10 and clipped to [-5, 5].

### Training statistics

All runs: one GPU, CPU physics with 4 to 6 threads, 64 to 128 parallel cars,
seed 1. "Frozen" is the learned weights replayed without plasticity.

| Track (route, author medal) | Actions | Minutes | Episodes | Finishes | First finish | Best lap | Frozen policy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| A04-Acrobatic (325 m, 5.95 s) | 6 | 5 | 75,108 | 191 | min 3 | 6.65 s | 0/2,048 greedy; 5/2,000 at eps 0.02 (6.58 s) |
| A04-Acrobatic | 6 | 30 | 648,222 | 662 | min 2 | 6.29 s | 0/2,048 greedy; 5/4,000 at eps 0.02 (8.00 s) |
| A02-Race (838 m, 16.25 s) | 6 | 60 | 97,402 | 9 | min 38 | 29.13 s | |
| A02-Race | 3 (gas only) | 60 | 149,232 | 68 | min 1 | 21.69 s | |
| A02-Race | 3 (gas only), 80 % action PNs | 60 | 291,819 | 2,961 | min 2 | 21.85 s | |
| A05-Race (751 m, 16.91 s) | 3 (gas only) | 60 | 152,729 | 42 | min 4 | 20.96 s | |
| C03-Acrobatic (783 m, 13.90 s) | 6 | 60 | 329,444 | 0 | | | mean progress 42 % |

Per-minute learning curve of the 5-minute A04 run (mean over the minute's
episodes):

| Minute | Episodes | Finishes | Best lap | Mean progress | Mean speed |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 13,344 | 0 | | 130 m (40 %) | 22.8 m/s |
| 2 | 15,059 | 0 | | 149 m (46 %) | 28.7 m/s |
| 3 | 16,286 | 17 | 7.18 s | 159 m (49 %) | 32.4 m/s |
| 4 | 15,602 | 75 | 6.88 s | 175 m (54 %) | 34.6 m/s |
| 5 | 14,817 | 99 | 6.65 s | 188 m (58 %) | 35.9 m/s |

What the numbers say. The circuit learns: finish counts rise from 0 to
99 per minute in five minutes on A04, and the value it forms correlates with
speed (r = 0.8 on A02). It does not learn action selection well: candidate
actions share 50 % of their KC code (71 % in the gas-only variant), so the
value differs little between actions and the choice is close to random. On
A04 that is enough because the ramp start gives the speed for free; on the
838 m A02 it crawls at 12 m/s and most episodes time out. Removing the brake
from the repertoire got finishes from minute 1; making 80 % of PNs read the
action code (KC overlap 40 %) raised the finish rate to about 100 per minute
but not the best time. The frozen weights alone rarely finish; the laps in
the videos are the best episodes of training or of an epsilon 0.02 replay,
replayed tick for tick from their recorded decisions (`mb_replay.py`
verifies the finish time against a fresh single-car environment). The three
laps in the videos are in `laps/` as decision sequences with the weights'
digest; `mb_replay --actions laps/<lap>.json` reproduces each finish time.

## The reflex driver

- **Eye**: 852 ommatidia per eye with viewing directions from the micro-CT
  eye map of Zhao et al. 2025 (`eye.py`), registered to the MaleCNS medulla
  columns through the hex lattice; the retina (`retina.py`) ray-casts the
  track geometry with Embree at 852 x 2 directions per 20 ms frame.
- **Optic lobe**: flyvis (Lappalainen et al. 2024), a connectome-constrained
  model of 721 columns and 64 cell types fitted to optic flow; run at its
  training dt of 20 ms (`flyvis_periphery.py`). Each eye's ommatidia are
  resampled onto the flyvis lattice rotated onto that eye's field of view.
- **Brain**: the MaleCNS wiring above 2 synapses (165,122 neurons,
  15.3 M edges, sign from the consensus neurotransmitter) integrated as
  tau dv/dt = -v + W r + I, r = relu(v), 1 ms substeps on the GPU
  (`brain_graph.py`, `brain_model.py`). 56 optic-lobe cell types are
  clamped to flyvis activity per column and side; everything downstream is
  free-running. `pathway_audit.py` checks the pathway facts the controls rely
  on (LC4/LPLC2 to DNp01 excitatory and ipsilateral, T4a to HS, HS to
  DNp15/DNa02) against raw synapse counts.
- **Controls** (`drive.py`, `Reflexes`): brake while the Giant Fiber (DNp01)
  rate exceeds 2.5 x its 1 s running mean; steer away from the side with the
  larger LC4/LPLC2 looming drive plus the HS optic-flow asymmetry; gas
  otherwise. The gains are the engineered part. Sidedness was verified with a
  looming disc on one side (LC4 on that side responds 4x) and a drifting
  grating (T4a/HS direction preference).

It drives away from walls and stalls in narrow sections after 150 to 320 m on
every track tried (A01, Island A1, Bay A1, Snow A1). `drive.py --replay-inputs`
runs the same eye, optic lobe and brain as an observer along a recorded lap,
which is how the mushroom-body videos get their eye and brain panels.

## Reproducing

Build the simulator and generate the game snapshots first (main README).
Then, in the Python environment:

```bash
pip install -r python/tmnf_fly/requirements.txt
export PYTHONPATH=python
python -m tmnf_fly.fetch_data          # MaleCNS tables, eye map, flyvis weights, brain mesh -> local/
python -m tmnf_fly.build_data          # eye, brain graph, mushroom-body circuit, skeleton sets -> local/fly/

# learn A04 for 5 minutes (writes local/fly/mb_a04.csv, mb_a04_weights.npz, mb_a04_best_actions.json)
python -m tmnf_fly.train_mb --track a04 --minutes 5 --num-envs 64
# (--gas-only for the 3-action repertoire, --pn-action-fraction 0.8 for the sharper action code)

# record the best lap's circuit activity and verify the finish time tick for tick
python -m tmnf_fly.mb_replay local/fly/mb_a04_weights.npz --track a04 --envs 1 --episodes 1 \
    --actions local/fly/mb_a04_best_actions.json --out local/fly/mb_a04_showcase.npz

# eye + optic lobe + brain along that lap, the game view, and the composite video
python -m tmnf_fly.drive --track a04 --replay-inputs local/fly/mb_a04_showcase.inputs.bin --out local/fly/drive/a04
build/export_viewer_scene oracle/tracks/A04-Acrobatic.tmnftrack oracle/routes/A04-Acrobatic.tmnfroute \
    oracle/vehicles/A04-Stadium.tmnfvehicle local/fly/mb_a04_showcase.inputs.bin <A04 sha256 from oracle/tracks/manifest.txt> \
    a04 local/fly/mb_a04_showcase.scene.json
python -m tmnf_fly.fly_view local/fly/mb_a04_showcase.scene.json local/fly/a04_chase.mp4 --camera chase
python -m tmnf_fly.video local/fly/drive/a04/observe_a04.npz local/fly/a04_lap.mp4 --track a04 \
    --real-view local/fly/a04_chase.mp4 --mushroom-body local/fly/mb_a04_showcase.npz --training-minutes 5
```

`fly_view.py` needs Node 22+, a Chromium binary (headless WebGL through
ANGLE) and the extracted game visuals under `viewer/assets`. The reflex
driver alone: `python -m tmnf_fly.drive --track a01 --ticks 6000`.

## Files

| | |
| --- | --- |
| `mushroom_body.py` | circuit extraction from the connectome tables; batched GPU mushroom body with the dopamine rule |
| `train_mb.py` | training loop, per-minute CSV log, best-lap JSON |
| `mb_replay.py` | frozen-weight rollouts or forced laps; records KC/MBON/DAN activity, writes the tick schedule, verifies the finish time |
| `laps/` | the decision sequences of the laps in the videos |
| `eye.py`, `retina.py` | ommatidia directions and the Embree retina |
| `flyvis_periphery.py`, `flyvis_validate.py` | flyvis optic lobe on the fly's lattice, and its validation |
| `brain_graph.py`, `brain_model.py`, `populations.py`, `pathway_audit.py` | MaleCNS signed graph, GPU rate model, cell-type inventory, pathway checks |
| `drive.py`, `pose.py` | closed-loop reflex driver and the observer mode |
| `skeletons.py`, `brain_render.py`, `fly_view.py`, `fly_view.mjs`, `video.py` | skeleton download and render, headless game-view capture, composite video |
| `fetch_data.py`, `build_data.py`, `requirements.txt` | public data with digests, derived files, pinned dependencies |

## Data and references

- MaleCNS v1.0 connectome tables: Janelia FlyEM, `gs://flyem-male-cns/v1.0/`
  (body annotations, neurotransmitter predictions, synapse weights at
  confidence 0.5) and per-neuron skeletons.
- Eye map: Zhao, A. et al. (2025), micro-CT ommatidia directions,
  github.com/reiserlab/eyemap_T4.
- flyvis: Lappalainen, J. K. et al. (2024), "Connectome-constrained networks
  predict neural activity across the fly visual system", Nature.
- Dopamine plasticity rule: Handler, A. et al. (2019), Cell; Bennett, J. E. M.
  et al. (2021), Nature Communications; compartment logic after Aso et al.
  (2014) and Hige et al. (2015).
- Giant Fiber / LC4 / LPLC2 escape pathway: von Reyn et al. (2014, 2017),
  Ache et al. (2019); HS optomotor pathway: Fujiwara et al. (2017).

The connectome, eye map and flyvis data keep their own licenses; only the code
here is MIT.
