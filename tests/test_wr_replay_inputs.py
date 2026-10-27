"""Focused regression tests for replay input filtering."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import wr_replay  # noqa: E402


class ReplayInputFilteringTests(unittest.TestCase):
    def test_subtick_inputs_wait_until_the_next_game_boundary(self):
        for name in ('Steer', 'Gas', 'Accelerate', 'Respawn'):
            for stamp, expected in [(620, 610), (621, 620), (629, 620),
                                    (630, 620), (-1, -10)]:
                with self.subTest(event=name, time=stamp):
                    self.assertEqual(wr_replay._event_time(
                        SimpleNamespace(event_name=name, time=stamp)), expected)

    def test_analog_gas_with_zero_enabled_is_retained(self):
        event = SimpleNamespace(event_name="Gas", flags=1, enabled=0)
        self.assertFalse(wr_replay._should_skip(event))

    def test_button_zero_enabled_is_still_skipped(self):
        event = SimpleNamespace(event_name="Accelerate", flags=0, enabled=0)
        self.assertTrue(wr_replay._should_skip(event))

    def test_schedule_preserves_simultaneous_pedals(self):
        """Replay records may intentionally hold gas and brake together.

        This is distinct from TMInterface's ``load`` keyboard-script behavior:
        the canonical race-input packet carries both digital bits, and the
        native mapper resolves both.  A script playback must not be used to
        infer that the gas bit was absent merely because ``press down`` clears
        a held ``up`` key in the UI layer.
        """
        schedule = wr_replay.build_schedule(
            "0-100 press up\n20-80 press down\n", 12
        )
        rows = list(wr_replay.INPUT_STRUCT.iter_unpack(schedule))
        self.assertEqual((rows[2][11], rows[2][14]), (1, 1))
        self.assertEqual((rows[7][11], rows[7][14]), (1, 1))
        self.assertEqual((rows[8][11], rows[8][14]), (1, 0))
        self.assertEqual(wr_replay.resolve_record(rows[2])[:2], (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
