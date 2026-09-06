# Local game data

Use a legitimate TrackMania Forever installation. This repository does not
supply game executables, packs, extracted geometry, vehicle snapshots,
textures or captured game memory. No account or license bypass is provided.

## Physics image

Install Python's Pillow package and the .NET 10 SDK. Point the extractor at
your installed `Packs` directory:

```bash
python3 -m venv build/assets-venv
build/assets-venv/bin/pip install Pillow
build/assets-venv/bin/python tools/prepare_local_assets.py \
  --packs /path/to/TmNationsForever/Packs
```

Use the United installation for United content. `--dotnet /path/to/dotnet`
selects an SDK outside PATH. The tool builds the pinned GBX.NET helper and
extracts only `TestMaterialHeight.tga`, then checks both the source image and
the generated physics bytes against the supported version's hashes.
Outputs go into ignored `local/`. NuGet downloads code dependencies, not game
assets. The game stores image rows bottom-up; extraction preserves that order.

Reconfigure and rebuild for simulation:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTMNF_CUDA=OFF \
  -DTMNF_GAME_MASK="$PWD/local/game-mask.bin"
cmake --build build -j4
```

A source-only library refuses to create a game world. It does not silently
use a blank image. Rebuilding is required after preparing the local image.
Libraries built with this image contain derived game data; do not upload them
as project releases without a separate redistribution review.

## Track, vehicle and route snapshots

The current complete import path captures snapshots from a locally running
game through Wine and TMInterface. The offline track builder can generate
collision geometry from packs, but does not yet replace vehicle and route
capture. A packs directory alone is therefore insufficient for RL setup.

The research capture tools expect a Wine prefix with the game installed at
`C:\TmNationsForever` or `C:\TmUnitedForever`, TMInterface 2.2.1 through
ModLoader, and the Linesight socket bridge. The Python tools need `numpy`
and `tminterface`; route generation also uses `pygbx`. Native capture helpers
require Wine, Xvfb, a 32-bit MinGW toolchain, curl, unzip, iproute2 and ripgrep.
The tracer is a local mod loaded through Ultimate ASI Loader; its installer
preserves the original `binkw32.dll` as `binkw32Hooked.dll`.

The lane uses `third_party/venv/bin/python`, the trainer uses
`build/venv/bin/python`, and the ModLoader installations live in
`third_party/TMLoader` or `third_party/TMLoader_united`. Configure ModLoader
for your installed game and add Linesight's `Python_Link.as` to the
TMInterface plugins directory. `oracle/prepare_oracle.py` configures the
Nations lane after these dependencies are installed; it also redirects that
prefix's Documents path to its isolated `TMNFDocuments` directory. Run it
only in a prefix dedicated to this project.

Set `TMNF_WINE_USER` if the Wine username differs from the current OS user.
The Nations lane uses `C:\users\<user>\TMNFDocuments`; United uses
`C:\users\<user>\Documents`. Complete the game's first launch and local profile
setup before attempting automated captures.

Inspect the resolved paths before capturing, choosing CPU IDs available on
your machine and an unused port and display:

```bash
build/venv/bin/python tools/onboard_track.py A04-Acrobatic \
  --wineprefix /path/to/dedicated-prefix --port 8488 --display 98 \
  --cpu-set 0,1 --dry-run
```

Remove `--dry-run` to capture and validate the track. The tool generates
track, vehicle and route snapshots, checks accepted inputs and native replay,
and builds route guidance. It writes a local catalogue under `oracle/tracks/`
and keeps captures, caches and viewer outputs in ignored directories.
A reported divergence is a failed comparison, not a successful fidelity check.

This remains a research setup, not a one-command installer. The standalone
source build and physics-image extraction have separate validation from
complete capture-lane setup. Do not infer clean-machine capture support from
a passing synthetic test suite.

## Offline geometry and visuals

`tools/build_track/build_track.py` accepts a challenge and `--packs DIR` to
build collision geometry locally. `tools/extract_game_assets/extract_game_assets.py`
accepts `--game-dir DIR` to generate Stadium viewer assets. Both require their
GBX.NET helper and the user's installed data. Generated content remains local.
