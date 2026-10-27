"""Fixed flag supplementation must preserve static geometry and be idempotent."""
import copy
import unittest
from unittest.mock import patch

import build_track
import fp
import trkfile


class FixedDecorationTests(unittest.TestCase):
    def test_appending_flags_preserves_old_tree_and_offsets_new_references(self):
        original = trkfile.Track(
            entries=[trkfile.Entry(
                1, fp.box_to_bytes((0., 0., 0., 1., 1., 1.)),
                fp.iso_to_bytes(fp.IDENTITY), 0x80, 0, 1, 1)],
            surfaces=[trkfile.Surface(0, b'\0')],
            meshes=[trkfile.Mesh(0, 0, 7, 0, b'', b'', b'')],
            corpus_isos=[fp.iso_to_bytes(fp.IDENTITY)])
        fixed = copy.deepcopy(original)
        fixed.entries[0].box = fp.box_to_bytes((10., 0., 0., 1., 1., 1.))
        fixed.surfaces[0].materials = b'\x1c'
        old = copy.deepcopy(original)
        with patch.object(build_track, 'fixed_stadium_flag_corpora', return_value=['flags']), \
                patch.object(build_track, 'geometry_track', return_value=fixed):
            self.assertEqual(build_track.append_fixed_decoration_collisions(original, None, {}), 1)
            self.assertEqual(build_track.append_fixed_decoration_collisions(original, None, {}), 0)
        self.assertEqual(original.entries[1], old.entries[0])
        self.assertEqual(original.entries[0].skip, len(original.entries))
        self.assertEqual(original.surfaces[0], old.surfaces[0])
        self.assertEqual(original.entries[-1].surface, 1)
        self.assertEqual(original.entries[-1].tree, 2)
        self.assertEqual(original.entries[-1].corpus, 2)
        self.assertEqual(original.surfaces[-1].mesh, 1)
        self.assertTrue(original.fixed_decoration_collisions)


if __name__ == '__main__':
    unittest.main()
