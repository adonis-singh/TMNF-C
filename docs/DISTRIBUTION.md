# Source and local data

The repository contains simulator code, Python bindings, training and capture
tools, synthetic tests, and a viewer. Original project code uses the
[MIT license](../LICENSE). Bundled libraries and fonts retain their
[upstream licenses](../THIRD_PARTY_NOTICES.md).

Game executables, packs, geometry, textures, vehicle and route snapshots,
captures, replays, policy weights, and generated scenes are not included.
See [local asset setup](LOCAL_ASSETS.md) to generate the data needed for
simulation from your installed game.

The physics image is extracted locally and embedded during compilation.
A library built with `TMNF_GAME_MASK` therefore contains derived game data;
the project's source license does not grant redistribution rights to that data.
Generated data and build outputs belong in the ignored local directories.

Run `python3 tools/audit_public_tree.py` to check tracked files for excluded
paths, game formats, unexpected binaries, and embedded physics-image bytes.
CI runs this check alongside the build that requires no game data.
