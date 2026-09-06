"""TMNF-C reinforcement learning research platform.

Modules:

* ``env``          zero-copy Gymnasium vector binding to the native physics
* ``tracks``       track catalogue (fixtures, pinned hashes)
* ``spaces``       observation flattening and action conversion
* ``encoder``      fixed physical-scale features from the 81-float observation
* ``snapshot_starts`` pool of route states from the best trajectories, snapshot starts
* ``agents.ppo``   reference PPO learner, checkpoints, metric schema
* ``evaluation``   full-start deterministic evaluation, schedule replay
* ``inputs``       exact per-tick TMNFRaceInputs schedules
* ``replays``      viewer scene export
* ``registry``     run registry under build/runs
* ``config``       dataclass config, JSON files, strict CLI overrides
* ``provenance``   git/physics/code/hardware fingerprints
* ``spectate``     live telemetry HTTP server

Commands: ``python -m tmnf_rl.{train,evaluate,reproduce,protocol,health_check,
registry}``.
"""

__version__ = "1.0.0"
