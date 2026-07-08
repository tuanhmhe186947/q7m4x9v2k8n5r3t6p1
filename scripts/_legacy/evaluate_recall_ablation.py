#!/usr/bin/env python3
# ruff: noqa: E402
"""Compatibility wrapper for optimize_tracking_metrics.py."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from optimize_tracking_metrics import main

if __name__ == "__main__":
    print(
        "[DEPRECATED] Use `python scripts\\optimize_tracking_metrics.py ...`.",
        file=sys.stderr,
    )
    raise SystemExit(main())
