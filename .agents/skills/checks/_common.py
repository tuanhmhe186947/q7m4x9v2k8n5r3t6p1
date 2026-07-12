"""Shared read-only helpers for classification_v2 skill checks."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV without modifying it and return its header and rows."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def load_json(path: Path) -> Any:
    """Load one UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value: str) -> bool | None:
    """Parse a strict CSV boolean, returning None for invalid values."""
    cleaned = str(value).strip().lower()
    if cleaned in {"true", "1", "yes", "y"}:
        return True
    if cleaned in {"false", "0", "no", "n"}:
        return False
    return None


def finish(report: dict[str, Any]) -> int:
    """Print a stable JSON audit and return fail-closed exit status."""
    errors = list(report.get("errors", []))
    report["valid"] = not errors
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1
