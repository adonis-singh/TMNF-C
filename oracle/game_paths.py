"""Capture paths resolved from the selected Forever installation."""
import os
from pathlib import Path
import sys


def script_directory() -> Path:
    # Resolve lazily: schedule decoding does not require a game installation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
    try:
        from track_env import game_layout
    finally:
        sys.path.pop(0)
    prefix = Path(os.environ.get(
        'TMNF_WINEPREFIX', Path(__file__).resolve().parent / 'wineprefix'))
    return game_layout(prefix).user_documents / 'TMInterface/Scripts'
