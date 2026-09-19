"""Apply Hidden review decisions to a new derived frame-feature artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.review.hidden_review_builder import (
    apply_hidden_review_decisions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--decisions-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--confusion-audit-json", type=Path, required=True)
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Smoke/debug only. Full reviewed data must be fail-closed.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = [args.output_csv, args.audit_json, args.confusion_audit_json]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Outputs already exist; use --overwrite explicitly: "
            + ", ".join(str(path) for path in existing)
        )

    frames = pd.read_csv(
        args.input_csv,
        low_memory=False,
        float_precision="round_trip",
    )
    manifest = pd.read_csv(args.manifest_csv, low_memory=False)
    decisions = pd.read_csv(args.decisions_csv, low_memory=False)
    reviewed, audit, confusion = apply_hidden_review_decisions(
        frames,
        manifest,
        decisions,
        require_resolved=not args.allow_unresolved,
    )
    authority = _authority_fields(args)
    audit.update(authority)
    confusion.update(
        {
            **authority,
            "input_rows": int(len(frames)),
            "output_rows": int(len(reviewed)),
            "errors": list(confusion["errors"]),
        }
    )
    _publish_output_transaction(
        reviewed,
        audit,
        confusion,
        outputs,
        overwrite=args.overwrite,
    )
    print(
        "[PASS] Hidden decisions applied without row loss: "
        f"rows={len(reviewed)} corrected={audit['corrected_hidden_rows']}"
    )


def _authority_fields(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "checker_code_authority_sha": _git_head(),
        "input_hashes": {
            "frame_local_primitives_sha256": _sha256_file(args.input_csv),
            "hidden_review_manifest_sha256": _sha256_file(
                args.manifest_csv
            ),
            "hidden_review_decisions_sha256": _sha256_file(
                args.decisions_csv
            ),
        },
        "data_lineage_authority_preserved": True,
        "input_artifacts_regenerated": False,
    }


def _publish_output_transaction(
    reviewed: pd.DataFrame,
    audit: dict[str, Any],
    confusion: dict[str, Any],
    paths: list[Path],
    *,
    overwrite: bool,
) -> None:
    """Stage, validate, and atomically promote the complete apply bundle."""
    token = uuid.uuid4().hex
    temporary = [
        path.with_name(f".{path.name}.{token}.tmp") for path in paths
    ]
    backups = [
        path.with_name(f".{path.name}.{token}.backup") for path in paths
    ]
    published: list[Path] = []
    backed_up: list[tuple[Path, Path]] = []
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
        reviewed.to_csv(temporary[0], index=False, float_format="%.17g")
        _write_json(temporary[1], audit)
        _write_json(temporary[2], confusion)
        _validate_staged_bundle(temporary, expected_rows=len(reviewed))

        if overwrite:
            for target, backup in zip(paths, backups, strict=True):
                if target.exists():
                    _replace_for_commit(target, backup)
                    backed_up.append((target, backup))
        for source, target in zip(temporary, paths, strict=True):
            _replace_for_commit(source, target)
            published.append(target)
    except Exception:
        for target in reversed(published):
            target.unlink(missing_ok=True)
        for target, backup in backed_up:
            if backup.exists():
                os.replace(backup, target)
        raise
    finally:
        for path in [*temporary, *backups]:
            path.unlink(missing_ok=True)


def _validate_staged_bundle(paths: list[Path], *, expected_rows: int) -> None:
    with paths[0].open("r", encoding="utf-8", newline="") as handle:
        staged_rows = sum(1 for _ in csv.reader(handle)) - 1
    if staged_rows != expected_rows:
        raise RuntimeError(
            f"staged apply row count mismatch={staged_rows}->{expected_rows}"
        )
    for path in paths[1:]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("errors"):
            raise RuntimeError(f"staged apply audit contains errors: {path}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _replace_for_commit(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


if __name__ == "__main__":
    main()
