# TMNF-C

A TrackMania Forever simulator for reinforcement learning, written in C.
Physics runs at the game's fixed 100 Hz tick rate, with CPU and CUDA vector
environments, Gymnasium bindings, and PPO and TD3 training tools.

The simulator targets `TmForever.exe` 2.11.26 and has been tested on 212
Nations and United campaign tracks across all seven vehicle environments.
Game assets are loaded from locally generated snapshots of your installation.

## Reinforcement learning

The vector environments support discrete and analog controls, checkpoint and
finish tracking, rewards, automatic episode resets and terminal observations.
Snapshots allow rewind, state transfer between environments and training from
states reached by the agent. Training saves configurations, evaluated policies
and checkpoints for resuming runs.

The [training guide](docs/TRAINING.md) covers setup, training and evaluation.

## Build and setup

The CPU build needs Linux, a C11 compiler, CMake and Python 3:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTMNF_CUDA=OFF
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

This builds the libraries, replay harness and viewer exporter, and runs the
synthetic tests. No game installation is needed for this step.

To run a racing environment, follow [local asset setup](docs/LOCAL_ASSETS.md).
It uses your installed game packs and a Wine/TMInterface setup to generate
track, vehicle and route snapshots. Game assets and recorded captures are
not bundled with the repository.

For CUDA, configure with `-DTMNF_CUDA=ON` and set
`-DCMAKE_CUDA_ARCHITECTURES` to your GPU's compute capability, such as `86`
for an RTX 3060. The tested architectures are 8.6 and 12.0.

## Results

Five pure-RL policies beat the author medal on four tracks. Each policy's
saved inputs reproduced the listed finish time in the actual game.

| Track | Policy time | Author time |
| --- | ---: | ---: |
| A04-Acrobatic | 5.900 s | 5.950 s |
| A04-Acrobatic, seed 2 | 5.930 s | 5.950 s |
| B05-Race | 25.750 s | 26.280 s |
| C03-Acrobatic | 12.660 s | 13.900 s |
| Rally A1 | 17.690 s | 18.950 s |

These are per-track agents trained without demonstrations or behaviour
cloning; some train from snapshots of their own trajectories. Rally A1 uses
an intermediate checkpoint. The policy weights and raw captures are not
included in this release. See [validation details](docs/VALIDATION.md) for
test coverage and result qualifications.

## Current limitations

The project focuses on campaign racing. Route guidance and learning budgets
still need work across tracks. Force models 4 and 5 and speed-glitch regimes
remain unvalidated. The supported setup is a Linux source checkout; generating
complete RL fixtures currently requires a running game through Wine/TMInterface.

## License

Original code is [MIT licensed](LICENSE), copyright 2026 adonis-singh.
Game assets are not covered by that license. Bundled libraries and fonts retain
their [third-party notices](THIRD_PARTY_NOTICES.md).
