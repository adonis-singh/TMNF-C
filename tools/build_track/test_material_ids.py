"""Keep explicit inline surface materials when no material node exists."""
import unittest
from unittest.mock import patch

from assets import Assets, BuildError


class SurfaceMaterialTests(unittest.TestCase):
    def test_explicit_metal_slot_without_material_file(self):
        assets = object.__new__(Assets)
        material = dict(file=None, surfaceId='Metal', surfaceIdValue=4, materialSurfaceId=None)
        with patch.object(assets, 'remapped_material', return_value=None):
            self.assertEqual(assets.material_surface_id(material, None), 4)

    def test_loaded_material_still_precedes_inline_value(self):
        assets = object.__new__(Assets)
        material = dict(file='material', surfaceIdValue=4, materialSurfaceId=7)
        with patch.object(assets, 'remapped_material', return_value=None):
            self.assertEqual(assets.material_surface_id(material, None), 7)
        with patch.object(assets, 'remapped_material', return_value='remapped'), patch.object(assets, 'load', return_value={'surfaceId': 10}):
            self.assertEqual(assets.material_surface_id(material, object()), 10)

    def test_unresolved_slot_never_guesses_a_default_material(self):
        assets = object.__new__(Assets)
        with patch.object(assets, 'remapped_material', return_value=None):
            with self.assertRaisesRegex(BuildError, 'no resolved or inline physics ID'):
                assets.material_surface_id(dict(file=None, materialSurfaceId=None, surfaceIdValue=None), None)


if __name__ == '__main__':
    unittest.main()
