"""Write complete frame-local/native review evidence semantics."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.review.evidence_semantics import (
    build_evidence_semantics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-local-csv", required=True, type=Path)
    parser.add_argument("--native-evidence-csv", required=True, type=Path)
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument("--code-authority-sha", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    require_output_paths_available(
        [args.output_json],
        overwrite=args.overwrite,
    )
    semantics = build_evidence_semantics(
        pd.read_csv(args.frame_local_csv, low_memory=False),
        pd.read_csv(args.native_evidence_csv, low_memory=False),
        lineage_id=args.lineage_id,
        code_authority_sha=args.code_authority_sha,
    )
    _atomic_json(args.output_json, semantics)
    if semantics["errors"]:
        raise SystemExit(2)
    print(json.dumps({"valid": True, "fields": len(semantics["fields"])}, indent=2))


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
