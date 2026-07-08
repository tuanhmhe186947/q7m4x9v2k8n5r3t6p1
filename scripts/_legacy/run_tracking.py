#!/usr/bin/env python3
# ruff: noqa: E402
"""Compatibility wrapper for the canonical `track_videos.py` entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from track_videos import main

if __name__ == "__main__":
    print(
        "[DEPRECATED] Use `python scripts/track_videos.py ...`.",
        file=sys.stderr,
    )
    raise SystemExit(main())
