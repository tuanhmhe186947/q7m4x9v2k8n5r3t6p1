"""Shared fail-closed guards for derived classification_v2 outputs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def require_output_paths_available(
    paths: Iterable[Path],
    *,
    overwrite: bool,
) -> None:
    """Reject accidental replacement of any declared derived artifact."""

    declared = [Path(path) for path in paths]
    existing = [path for path in declared if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Derived output files already exist; pass --overwrite explicitly: "
            f"{existing}"
        )
