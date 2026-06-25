"""Compatibility wrapper for tracking annotation generation.

The implementation lives in ``pig_behavior.data_preparation.tracking_annotation``.
Use ``pig-track-for-annotation`` or import from ``src`` in new code.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    start = start if start.is_dir() else start.parent
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists() or (path / ".git").exists():
            return path
    return start


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pig_behavior.data_preparation.tracking_annotation import (  # noqa: E402
    DEFAULT_MASK_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VIDEO_PATH,
    DEFAULT_WEIGHTS_PATH,
    TrackingConfig,
    TrackingSummary,
    display_tracked_video,
    main,
    run_tracking,
)

__all__ = [
    "DEFAULT_MASK_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_VIDEO_PATH",
    "DEFAULT_WEIGHTS_PATH",
    "TrackingConfig",
    "TrackingSummary",
    "display_tracked_video",
    "main",
    "run_tracking",
]


if __name__ == "__main__":
    raise SystemExit(main())
