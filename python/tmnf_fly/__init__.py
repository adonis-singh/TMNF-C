"""Paths shared by the package.

``LOCAL`` is the data root (``<repo>/local`` unless ``TMNF_FLY_LOCAL`` is set):
``tmnf_fly.fetch_data`` downloads the external data into it and
``tmnf_fly.build_data`` writes the derived files next to them.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL = Path(os.environ.get("TMNF_FLY_LOCAL", ROOT / "local")).resolve()
FLY_DIR = LOCAL / "fly"
MALECNS_DIR = LOCAL / "malecns"
EYEMAP_DIR = LOCAL / "eyemap"
ANNOTATIONS_FEATHER = MALECNS_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
