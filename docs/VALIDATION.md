# Source preview validation

The publication preparation checks the source distribution separately from
the private research corpus. No game data is included in these results.

A fresh Git clone configured and built on Linux with `TMNF_CUDA=OFF`, with
no game installation linked into the checkout. All six registered CTest
checks passed. They cover synthetic physics and vehicle operations, gate
observations, synthetic track generation/loading, malformed local image
rejection and empty track-catalogue behavior.

Local extraction from the installed Nations packs produced the expected
16,384-byte physics image. The image is hash-checked before compilation.
Using that image, the rebuilt CPU library passed the seven private release
capture replays, their expected finish times and environment continuation
checks. These private-data comparisons are reported here; their raw inputs
and captures are not shipped in this source preview.

The public source inventory check passes. It excludes game-data directories,
known binary game formats, unexpected binary files and the previously
embedded image. Viewer font files are the explicit binary exception and
retain their upstream redistribution license.

A passing source inventory check does not establish legal rights in every
piece of reconstructed code. A passing synthetic suite also does not imply
that an arbitrary track, vehicle combination or game state is validated.
