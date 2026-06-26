#!/usr/bin/env python3
"""Compatibility wrapper for the canonical `benchmark_tracking_weights.py` entrypoint."""

from __future__ import annotations

import sys

from benchmark_tracking_weights import main


if __name__ == "__main__":
    print(
        "[DEPRECATED] Use `python scripts/benchmark_tracking_weights.py ...`.",
        file=sys.stderr,
    )
    raise SystemExit(main())
