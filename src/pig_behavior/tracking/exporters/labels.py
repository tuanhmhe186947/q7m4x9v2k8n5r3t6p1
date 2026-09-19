"""Label schema export helpers."""

from __future__ import annotations

import json
from pathlib import Path

from pig_behavior.tracking.constants import PIG_LABEL_SCHEMA


def write_labels_json(path: Path) -> None:
    path.write_text(
        json.dumps(PIG_LABEL_SCHEMA, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


__all__ = ["write_labels_json"]
