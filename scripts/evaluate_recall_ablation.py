#!/usr/bin/env python3
"""Compatibility wrapper for optimize_tracking_metrics.py."""

from __future__ import annotations

import sys

from optimize_tracking_metrics import main


if __name__ == "__main__":
    print(
        "[DEPRECATED] Use `python scripts\\optimize_tracking_metrics.py ...`.",
        file=sys.stderr,
    )
    raise SystemExit(main())
