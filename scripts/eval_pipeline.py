#!/usr/bin/env python3
"""Compatibility wrapper for the canonical `evaluate_tracking.py` entrypoint."""

from __future__ import annotations

import sys

from evaluate_tracking import main


if __name__ == "__main__":
    print(
        "[DEPRECATED] Use `python scripts/evaluate_tracking.py ...`.",
        file=sys.stderr,
    )
    raise SystemExit(main())
