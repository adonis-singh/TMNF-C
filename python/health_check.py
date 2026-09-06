#!/usr/bin/env python3
"""Thin entry point: ``python/health_check.py`` == ``python -m tmnf_rl.health_check``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tmnf_rl.health_check import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv[1:])
