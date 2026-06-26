#!/usr/bin/env python3
"""Compatibility wrapper for the canonical `track_videos.py` entrypoint."""

from __future__ import annotations

import sys

from track_videos import main


if __name__ == "__main__":
    print(
        "[DEPRECATED] Use `python scripts/track_videos.py ...`.",
        file=sys.stderr,
    )
    raise SystemExit(main())
