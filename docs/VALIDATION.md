# Validation

## Build and synthetic tests

A fresh Git clone configured and built on Linux with `TMNF_CUDA=OFF`, with
no game installation linked into the checkout. All six registered CTest
checks passed. They cover synthetic physics and vehicle operations, gate
observations, synthetic track generation/loading, malformed local image
rejection and empty track-catalogue behavior.

## Game comparisons

Local extraction from the installed Nations packs produced the expected
16,384-byte physics image. The image is hash-checked before compilation.
Using that image, the rebuilt CPU library passed the seven local
capture replays, their expected finish times and environment continuation
checks. A CUDA build for the RTX 3060 (compute capability 8.6) also replayed
all seven captures byte-exactly. Raw inputs and captures are not included
in the repository.

A04 was imported from a configured local game lane,
starting with no saved vehicle snapshot. The generated track and vehicle
replayed 1,200 mixed-input ticks and 1,400 wall-contact ticks exactly after
the capture workflow's documented normalization. The generated route passed
its regeneration check. Eight CPU environments then ran 256 decisions each
with finite rewards and exact reward continuation after snapshot restore.
This validates one fresh import from a configured lane, not a clean-machine
installer or every campaign track's route quality. Visual extraction was not
part of this headless import check.

## Policy results

The README reports per-track policies evaluated from the official start.
Game validation replays their saved input schedules; it does not execute the
policy online in the game. The Rally A1 result uses an intermediate checkpoint
at 17.690 seconds. Its final seed-1 checkpoint was slower at 19.010 seconds.
These results do not establish cross-track generalization or training success
across seeds. Policy weights and raw captures are not included in the repository.
