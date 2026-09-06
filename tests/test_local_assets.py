"""Tests for local asset validation and empty track catalogues."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class LocalAssets(unittest.TestCase):
    def test_wrong_mask_is_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'bad.bin', Path(tmp)/'out.h'
            source.write_bytes(bytes(16384))
            result = subprocess.run([sys.executable, str(ROOT/'tools/prepare_game_mask.py'),
                                     'header', str(source), str(output)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

    def test_empty_checkout_has_no_installed_tracks(self):
        tracks = module('public_test_tracks', ROOT/'python/tmnf_rl/tracks.py')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'oracle/tracks').mkdir(parents=True)
            (root/'oracle/tracks/manifest.txt').write_text('# local only\n')
            self.assertEqual(tracks.track_catalogue(root), {})
            with self.assertRaisesRegex(ValueError, 'LOCAL_ASSETS'):
                tracks.track_spec('a01', root)


if __name__ == '__main__':
    unittest.main()
