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

## Glitch replays

Two TAS input schedules were captured in the game and compared with the CPU
and CUDA simulators. A01-Race's 17.610-second noseboost run by igntuL, lukalyc,
Na'Guul and trabadia matched all 1,782 captured ticks. A12-Speed's
10.100-second run by AurisTFG and Sokko matched all 1,031 ticks, including the
sideways gas/brake/steer sequence and subsequent uberbug launch.
The source replays are TAS Exchange entries
[53](https://tmtas.exchange/api/replay?id=53) and
[128](https://tmtas.exchange/api/replay?id=128).

Both game runs reproduced the recorded finish times with zero accepted-input
mismatches. A01 reached 1,422.3 km/h. A12 accelerated from 312.6 to 485.8 km/h
between captured ticks 656 and 657. Comparisons use the existing normalization
of never-written memory fields and upper material bits; physical state and
surface positions are not normalized. CUDA comparisons used an RTX 3060.

Two additional custom-map TAS replays by Rio matched on CPU and CUDA:
[Bugslide Fun #1](https://tmtas.exchange/api/replay?id=230), 9.810 seconds and
1,002 captured ticks, and
[Lilo BugSlide 004](https://tmtas.exchange/api/replay?id=4091), 7.210 seconds
and 742 ticks. Both reproduced their recorded game finish times with zero
accepted-input mismatches. Bugslide Fun #1 includes a roughly 90-degree
landing followed by a braked sideways turn, independently of the A12
uberbug sequence. The Lilo TAS includes tilted wheel contacts and large
boosts; its map name alone does not establish a conventional sustained
bugslide.

The custom-map checks compare physics without the optional native route
and finish wrapper. Their normalization changed only 408 never-written
turbo-start words and 1,002 roulette words for Bugslide Fun #1, and 2,226
upper material words for Lilo. No physical state or surface positions were
normalized, and no physics changes were needed.

These checks cover the specific maneuvers in those runs, not every glitch
variation or arbitrarily long sustained bugslides. Replay inputs and game
captures are local fixtures.
To capture and compare a campaign replay after local asset setup, use
`tools/wr_replay.py run TRACK REPLAY --tag TAG`; lane options are listed in
`--help`. The saved events JSON includes the native replay exit status and
comparison output.

## Policy results

The README reports per-track policies evaluated from the official start.
Game validation replays their saved input schedules; it does not execute the
policy online in the game. The Rally A1 result uses an intermediate checkpoint
at 17.690 seconds. Its final seed-1 checkpoint was slower at 19.010 seconds.
These results do not establish cross-track generalization or training success
across seeds. Policy weights and raw captures are not included in the repository.
