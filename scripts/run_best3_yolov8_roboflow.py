#!/usr/bin/env python3
"""Compatibility wrapper for the canonical `evaluate_best3_roboflow.py` entrypoint."""

from __future__ import annotations

import sys

from evaluate_best3_roboflow import main


if __name__ == "__main__":
    print(
        "[DEPRECATED] Use `python scripts/evaluate_best3_roboflow.py ...`.",
        file=sys.stderr,
    )
    raise SystemExit(main())
