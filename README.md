# TMNF-C

A deterministic TrackMania Forever simulator for reinforcement learning.
C physics runs at the game's fixed 100 Hz tick rate, with CPU and CUDA vector
environments, Gymnasium bindings, and PPO and TD3 training tools.

This source preview targets `TmForever.exe` 2.11.26. Game assets and captured
research data are not included. Build and run the synthetic checks without a
game installation; supply local game data before running a racing environment.

## Build

The source-only build needs Linux, a C11 compiler, CMake and Python 3:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTMNF_CUDA=OFF
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

This builds the CPU libraries, replay harness and viewer exporter. The tests
exercise synthetic physics, gate geometry, track loading and asset setup
validation. They are not substitutes for game-capture comparisons.

For simulation, follow [local asset setup](docs/LOCAL_ASSETS.md). The current
workflow uses your installed game packs for the physics image and a local
Wine/TMInterface capture lane for track, vehicle and route snapshots. It is
not yet a packs-only importer for a complete racing environment.

The [Python guide](docs/TRAINING.md) covers training and evaluation after
local fixture generation. CUDA is optional; enable `TMNF_CUDA` and set
`CMAKE_CUDA_ARCHITECTURES` for your GPU when configuring the build. Research
validation has covered compute capabilities 8.6 and 12.0.

## Research results

The private research corpus covers 212 Nations and United campaign tracks
across seven vehicle environments. Five retained pure-RL policies beat the
author medal on four tracks. Each listed policy produced an input schedule
whose finish time was reproduced in the game.

| Track | Policy time | Author time |
| --- | ---: | ---: |
| A04-Acrobatic | 5.900 s | 5.950 s |
| A04-Acrobatic, seed 2 | 5.930 s | 5.950 s |
| B05-Race | 25.750 s | 26.280 s |
| C03-Acrobatic | 12.660 s | 13.900 s |
| Rally A1 | 17.690 s | 18.950 s |

These are reported research results; their raw captures and policies are not
part of this source distribution. They are per-track agents trained without
demonstrations or behaviour cloning. Some use snapshot starts from their
own trajectories. Rally uses an intermediate checkpoint; its final seed-1
checkpoint was slower at 19.010 s. Game validation replays saved inputs,
rather than executing the policy online in the game. Cross-track
generalization and superhuman performance are not established.

## Scope

The vector API provides discrete and analog controls, rewards, checkpoint
and finish tracking, same-step autoreset, terminal observations and snapshots.
Training records configurations and code hashes and retains evaluated policies.
Exactness claims apply to tested capture fields and regimes; some private
captures normalize documented memory that the game never writes.

Campaign racing is the focus. Force models 4 and 5 and speed-glitch regimes
remain unvalidated. Route guidance and learning budgets need further work.
The public source-only checks do not establish full game fidelity or a
portable Windows installation.

[ForeverValidator](https://github.com/Skycrafter-dev/ForeverValidator) is
related work with replay validation, a controllable sandbox and a Stadium
CUDA backend. It has broader game-mode coverage. TMNF-C focuses on the RL
workflow and CPU/CUDA support across seven vehicle environments. A current
head-to-head speed advantage has not been established.

## Distribution

The [distribution policy](docs/DISTRIBUTION.md) describes the source boundary.
Local packs, snapshots, captures, replays, textures, generated scenes and
compiled libraries containing locally extracted image data must stay out of
source commits and release uploads. Third-party components retain the notices
listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The project code license is awaiting the owner's selection. This staging
snapshot must not be advertised as an open-source release until that is set.
