"""Canonical SHA-256 helpers shared by classification_v2 lineage modules."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

FILE_CHUNK_BYTES = 1024 * 1024


def payload_sha256(payload: Any) -> str:
    """Hash a JSON-compatible payload with deterministic key ordering."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash one file incrementally without loading large caches into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(FILE_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: str) -> bool:
    """Return whether a value is one lowercase hexadecimal SHA-256 digest."""

    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = ["file_sha256", "is_sha256", "payload_sha256"]
