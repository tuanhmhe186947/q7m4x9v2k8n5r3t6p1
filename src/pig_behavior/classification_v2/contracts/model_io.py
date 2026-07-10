"""Model I/O contract helpers for classification_v2.

These helpers keep trainer-facing code explicit: model inputs are whitelisted
feature tensors/tables, while identifiers, review fields, labels, and paths stay
outside X even when they are numeric.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_FORBIDDEN_X_PATTERNS = (
    "manual_*",
    "review_*",
    "*behavior*",
    "original_behavior",
    "review_unit_id",
    "window_id",
    "temporal_unit_key",
    "video_key",
    "dataset_id",
    "pig_id",
    "track_id",
    "*_path",
)


def forbidden_x_columns(columns: list[str], patterns: list[str] | tuple[str, ...] | None = None) -> list[str]:
    """Return columns that match audit/label/identifier patterns forbidden in X."""
    active_patterns = tuple(patterns or DEFAULT_FORBIDDEN_X_PATTERNS)
    return sorted(col for col in columns if any(fnmatch(col, pattern) for pattern in active_patterns))


def validate_model_input_columns(
    columns: list[str],
    *,
    forbidden_patterns: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Audit a candidate X schema and fail closed when leakage-prone columns appear."""
    forbidden = forbidden_x_columns(columns, forbidden_patterns)
    return {
        "column_count": int(len(columns)),
        "forbidden_columns": forbidden,
        "valid": not forbidden and bool(columns),
    }


def read_csv_schema(path: Path) -> list[str]:
    """Read only the CSV header, which is enough to validate an X schema cheaply."""
    return list(pd.read_csv(path, nrows=0).columns)
