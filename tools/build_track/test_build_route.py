"""Focused route-builder tests that do not require game assets or Wine."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import build_route


class WaypointMobilTests(unittest.TestCase):
    def test_waypoint_mobil_matches_scene_opposite_ground_fallback(self):
        block = SimpleNamespace(info={}, ground=False, variant=0, sub_variant=0)
        preferred = {'mobil': 'air', 'mobilTreeId': None}
        with patch.object(build_route, 'get_mobil', side_effect=[None, preferred]) as lookup:
            self.assertIs(build_route.waypoint_mobil(block), preferred)
        self.assertEqual(lookup.call_args_list[0].args, ({}, False, 0, 0))
        self.assertEqual(lookup.call_args_list[1].args, ({}, True, 0, 0))

    def test_waypoint_mobil_preserves_preferred_ground_table(self):
        block = SimpleNamespace(info={}, ground=True, variant=2, sub_variant=3)
        preferred = {'mobil': 'ground', 'mobilTreeId': None}
        with patch.object(build_route, 'get_mobil', return_value=preferred) as lookup:
            self.assertIs(build_route.waypoint_mobil(block), preferred)
        lookup.assert_called_once_with({}, True, 2, 3)

    def test_route_omits_finish_without_scene_mobil_but_keeps_checkpoint(self):
        active_finish = SimpleNamespace(info={'wayPointType': 'Finish'}, ground=True,
                                        variant=0, sub_variant=0)
        decorative_finish = SimpleNamespace(info={'wayPointType': 'Finish'}, ground=True,
                                            variant=5, sub_variant=0)
        checkpoint = SimpleNamespace(info={'wayPointType': 'Checkpoint'}, ground=True,
                                     variant=5, sub_variant=0)
        with patch.object(build_route, 'get_mobil',
                          side_effect=lambda _info, _ground, variant, _sub:
                          {'mobil': 'active'} if variant == 0 else None):
            self.assertEqual(build_route.route_finishes(
                SimpleNamespace(blocks=[active_finish, decorative_finish])), [active_finish])
            self.assertEqual(build_route.route_waypoints(
                SimpleNamespace(blocks=[checkpoint]), 'Checkpoint'), [checkpoint])


if __name__ == '__main__':
    unittest.main()
