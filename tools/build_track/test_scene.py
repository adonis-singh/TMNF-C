"""Scene rejection must be atomic and must not create mobils or waypoints."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import build_route
import scene


class MissingBlockInfoTests(unittest.TestCase):
    def challenge(self, allow_missing):
        challenge = scene.Challenge.__new__(scene.Challenge)
        challenge.allow_missing_blocks = allow_missing
        challenge.doc = {'blocks': [
            dict(index=index, name=name, coord=[0, 1, 0], direction=0, flags=0)
            for index, name in [(17, 'Known'), (91, 'Unavailable'), (23, 'Known')]]}
        info = {'type': 'CGameCtnBlockInfoRoad', 'wayPointType': 'None'}

        def lookup(name):
            if name == 'Unavailable':
                raise scene.MissingBlockInfoError('unknown block Unavailable')
            return info

        challenge.assets = SimpleNamespace(block_info=Mock(side_effect=lookup),
                                           block_info_at=lambda _path: None)
        challenge.blocks, challenge.add_list = [], []
        challenge.default_zone = {'blockInfoFlat': 'unused', 'common': {'zoneId': 'ground'}}
        challenge.size = (0, 32, 0)
        challenge.update_field_units = Mock(return_value=True)
        challenge.create_mobil_for_block = Mock()
        return challenge

    def test_validation_omits_unresolved_ids_and_keeps_authored_indices(self):
        challenge = self.challenge(True)
        challenge.init()
        self.assertEqual([block.index for block in challenge.blocks], [17, 23])
        self.assertEqual(challenge.missing_blocks, [{'index': 91, 'name': 'Unavailable'}])
        self.assertEqual(challenge.create_mobil_for_block.call_count, 2)

    def test_strict_scene_loading_rejects_unresolved_ids(self):
        with self.assertRaises(scene.MissingBlockInfoError):
            self.challenge(False).init()

    def test_missing_geometry_for_a_known_block_remains_an_error(self):
        challenge = self.challenge(True)
        challenge.assets.block_info.side_effect = scene.BuildError('missing asset dump')
        with self.assertRaisesRegex(scene.BuildError, 'missing asset dump'):
            challenge.init()
        challenge.create_mobil_for_block.assert_not_called()


class RejectedBlockTests(unittest.TestCase):
    def field_challenge(self):
        challenge = scene.Challenge.__new__(scene.Challenge)
        challenge.size = (32, 32, 32)
        challenge.field = {}
        challenge.allow_ground_mismatch = True
        challenge.is_block_on_ground = Mock(return_value=False)
        return challenge

    def block(self, kind, coord, direction=0, offset=(0, 0, 0)):
        unit = {'relativeOffset': offset, 'clips': [None] * 4}
        return scene.Block(0, kind, coord, direction, 0,
            {'type': 'CGameCtnBlockInfo' + kind, 'airUnits': [unit], 'groundUnits': [unit]})

    def test_undefined_clip_can_pass_bounds_using_previous_absolute_coordinates(self):
        challenge = self.field_challenge()
        preceding = self.block('Road', (7, 8, 5), offset=(1, 0, 0))
        self.assertTrue(challenge.update_field_units(preceding))
        block = self.block('Clip', (7, 8, 7), direction=4)
        self.assertTrue(challenge.update_field_units(block))
        # Bounds uses (8,0,5); placement uses the separate (1,0,0) temporary.
        self.assertEqual(block.units[0].coord, (8, 8, 7))
        self.assertIs(challenge.block_at((8, 8, 7)), block)

    def test_undefined_clip_rejects_when_previous_absolute_coordinates_exceed_bounds(self):
        challenge = self.field_challenge()
        self.assertTrue(challenge.update_field_units(self.block('Road', (16, 6, 17))))
        field_before = {key: list(column) for key, column in challenge.field.items()}
        block = self.block('Clip', (16, 6, 18), direction=4)
        self.assertFalse(challenge.update_field_units(block))
        self.assertEqual(block.units, [])
        self.assertEqual(challenge.field, field_before)

    def test_undefined_clip_without_known_scratch_values_is_not_guessed(self):
        challenge = self.field_challenge()
        with self.assertRaisesRegex(scene.BuildError, 'without initialized field offsets'):
            challenge.update_field_units(self.block('Clip', (1, 1, 1), direction=4))
        self.assertEqual(challenge.field, {})

    def test_authored_empty_pylon_has_no_collision_mobil(self):
        challenge = scene.Challenge.__new__(scene.Challenge)
        challenge.assets = SimpleNamespace(mobil_tree=Mock())
        block = SimpleNamespace(type=scene.PYLON, ground=False, variant=0, sub_variant=0,
            mobil='previous', info={'groundMobils': [[] for _ in range(15)], 'airMobils': []})
        challenge.create_block_mobil(block, None)
        self.assertIsNone(block.mobil)
        challenge.assets.mobil_tree.assert_not_called()

    def test_nonempty_pylon_fallback_is_not_silently_discarded(self):
        challenge = scene.Challenge.__new__(scene.Challenge)
        challenge.assets = SimpleNamespace(mobil_tree=Mock())
        block = SimpleNamespace(type=scene.PYLON, ground=True, variant=0, sub_variant=0,
            info={'groundMobils': [[]], 'airMobils': [[{'mobil': 'nonempty'}]]})
        with self.assertRaisesRegex(scene.BuildError, 'nonempty pylon mobil fallback'):
            challenge.create_block_mobil(block, None)
        challenge.assets.mobil_tree.assert_not_called()

    def test_init_revisits_consecutive_rejections_and_preserves_active_order(self):
        challenge = scene.Challenge.__new__(scene.Challenge)
        blocks = [SimpleNamespace(index=index, type=scene.ROAD, ground=False, units=[],
                                  info={'wayPointType': 'Checkpoint'})
                  for index in [17, 91, 92, 23]]
        challenge.doc = {'blocks': []}
        challenge.blocks, challenge.add_list = blocks.copy(), []
        challenge.assets = SimpleNamespace(block_info_at=lambda _path: None)
        challenge.default_zone = {'blockInfoFlat': 'unused', 'common': {'zoneId': 'ground'}}
        challenge.size = (0, 32, 0)
        challenge.update_field_units = Mock(side_effect=lambda block: block.index not in [91, 92])
        challenge.create_mobil_for_block = Mock()

        challenge.init()

        self.assertEqual([call.args[0].index for call in challenge.update_field_units.call_args_list],
                         [17, 91, 92, 23])
        self.assertEqual([call.args[0].index for call in challenge.create_mobil_for_block.call_args_list],
                         [17, 23])
        self.assertEqual([block.index for block in challenge.blocks], [17, 23])
        self.assertEqual(challenge.add_list, [blocks[0], blocks[3]])
        self.assertEqual(build_route.route_waypoints(challenge, 'Checkpoint'), [blocks[0], blocks[3]])

    def test_one_out_of_bounds_unit_rejects_before_any_field_mutation(self):
        challenge = scene.Challenge.__new__(scene.Challenge)
        challenge.size = (32, 32, 32)
        challenge.is_block_on_ground = Mock(return_value=False)
        challenge.field_unit = Mock(side_effect=AssertionError('field accessed before bounds validation'))
        challenge.create_field_unit = Mock()
        block = SimpleNamespace(type=scene.ROAD, info={}, direction=0, coord=(0, 1, 0), units=[])
        with patch.object(scene, 'unit_infos', return_value=[{}, {}]), \
                patch.object(scene, 'rotated_offset', side_effect=[(0, 0, 0), (32, 0, 0)]):
            self.assertFalse(challenge.update_field_units(block))
        challenge.field_unit.assert_not_called()
        challenge.create_field_unit.assert_not_called()
        self.assertEqual(block.units, [])


class ClipRepairTests(unittest.TestCase):
    def challenge_with_clip(self):
        challenge = scene.Challenge.__new__(scene.Challenge)
        coord = (1, 1, 1)
        default = {'id': 'GrassClip', 'groundMobils': [[{'mobil': {}}]]}
        road_clip = {'id': 'RoadClip'}
        old = SimpleNamespace(type=scene.CLIP, coord=coord, units=[])
        unit = SimpleNamespace(block=old, coord=coord)
        old.units.append(unit)
        field = SimpleNamespace(unit=unit, terrain=scene.T_GROUND)
        neighbor = SimpleNamespace(block=SimpleNamespace(type=scene.CLASSIC),
                                   clips=[None, None, road_clip, None])
        challenge.blocks, challenge.add_list = [old], [old]
        challenge.assets = SimpleNamespace(block_info_at=lambda _path: default)
        zone = {'blockInfoClip': 'grass', 'common': {'depth': 0}}
        challenge.default_zone = zone
        challenge.real_zone = lambda _coord: zone
        challenge.block_at = lambda _coord: field.unit.block if field.unit else None
        challenge.field_unit = lambda _coord: field
        challenge.terrain_at = lambda _coord: scene.T_GROUND
        challenge.unit_at = lambda position: neighbor if position == (1, 1, 2) else None
        challenge.column_modifier = Mock(return_value='Covered')

        def update(block):
            new_unit = SimpleNamespace(block=block, coord=coord)
            block.units.append(new_unit)
            field.unit = new_unit
            return True

        challenge.update_field_units = Mock(side_effect=update)
        challenge.create_mobil_for_block = Mock()
        return challenge, old, field, neighbor

    def test_regenerated_clip_uses_completed_neighbors_and_preserves_column_skin(self):
        challenge, old, field, neighbor = self.challenge_with_clip()
        self.assertTrue(challenge.regenerate_clip(old.coord))
        replacement = challenge.blocks[0]
        self.assertIsNot(replacement, old)
        self.assertIs(field.unit.block, replacement)
        self.assertEqual(challenge.add_list, [replacement])
        self.assertTrue(replacement.ground)
        self.assertEqual(replacement.terrain_modifier, 'Covered')
        challenge.create_mobil_for_block.assert_called_once_with(replacement)

    def test_unconnected_clip_is_removed_without_recreating_geometry(self):
        challenge, old, field, neighbor = self.challenge_with_clip()
        neighbor.clips = [None] * 4
        self.assertTrue(challenge.regenerate_clip(old.coord))
        self.assertEqual(challenge.blocks, [])
        self.assertEqual(challenge.add_list, [])
        self.assertIsNone(field.unit)
        challenge.update_field_units.assert_not_called()
        challenge.create_mobil_for_block.assert_not_called()

    def test_removing_ground_clip_restores_terrain_hidden_by_its_placement(self):
        challenge, old, field, neighbor = self.challenge_with_clip()
        neighbor.clips = [None] * 4
        terrain = SimpleNamespace(terrain_modifier=None)
        challenge.block_at = lambda coord: (terrain if coord[1] == 0
                                           else field.unit.block if field.unit else None)
        self.assertTrue(challenge.regenerate_clip(old.coord))
        self.assertEqual(challenge.add_list, [terrain])
        self.assertEqual(terrain.terrain_modifier, 'Covered')
        challenge.create_mobil_for_block.assert_called_once_with(terrain)

    def test_repair_visits_original_clips_once_and_does_not_revisit_appended_clips(self):
        challenge = scene.Challenge.__new__(scene.Challenge)
        original = [SimpleNamespace(type=scene.CLIP, coord=(i, 1, 0)) for i in [3, 9]]
        challenge.blocks = original.copy()
        visited = []

        def replace(coord):
            visited.append(coord)
            challenge.blocks.pop(0)
            challenge.blocks.append(SimpleNamespace(type=scene.CLIP, coord=(99, 1, 0)))

        challenge.regenerate_clip = replace
        challenge.regenerate_clips()
        self.assertEqual(visited, [(3, 1, 0), (9, 1, 0)])


if __name__ == "__main__":
    unittest.main()
