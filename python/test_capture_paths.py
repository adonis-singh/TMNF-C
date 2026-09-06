"""Input scripts must be installed in the selected game's documents folder."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'oracle'))
from game_paths import script_directory
sys.path.pop(0)


@pytest.mark.parametrize('install,documents', [('TmUnitedForever', 'Documents'),
                                              ('TmNationsForever', 'TMNFDocuments')])
def test_capture_script_directory(tmp_path, monkeypatch, install, documents):
    exe = tmp_path / 'drive_c' / install / 'TmForever.exe'
    exe.parent.mkdir(parents=True)
    exe.touch()
    monkeypatch.setenv('TMNF_WINEPREFIX', str(tmp_path))
    assert script_directory() == tmp_path / f'drive_c/users/adityas/{documents}/TMInterface/Scripts'


def test_capture_rejects_a_missing_install(tmp_path, monkeypatch):
    monkeypatch.setenv('TMNF_WINEPREFIX', str(tmp_path))
    with pytest.raises(RuntimeError, match='missing Wine prefix'):
        script_directory()
