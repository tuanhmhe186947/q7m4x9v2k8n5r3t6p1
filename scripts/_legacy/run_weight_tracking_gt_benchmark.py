#!/usr/bin/env python3
# ruff: noqa: E402
"""Compatibility wrapper for the canonical `benchmark_tracking_weights.py` entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

BENCHMARKS_ROOT = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_ROOT))

from benchmark_tracking_weights import main

if __name__ == "__main__":
    print(
        "[DEPRECATED] Use `python scripts/benchmarks/benchmark_tracking_weights.py ...`.",
        file=sys.stderr,
    )
    raise SystemExit(main())
