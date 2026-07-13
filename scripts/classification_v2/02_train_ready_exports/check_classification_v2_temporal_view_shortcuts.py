"""Check source, length, padding, timing, and missingness shortcuts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.evaluation.temporal_shortcut_audit import (
    audit_temporal_view_shortcuts,
    write_temporal_shortcut_audit,
)


def parse_args() -> argparse.Namespace:
    """Parse one versioned temporal-view packet and optional mitigation evidence."""

    parser = argparse.ArgumentParser(
        description="Audit classification_v2 temporal-view shortcuts."
    )
    parser.add_argument("--temporal-view-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--mitigation-evidence-json", type=Path)
    parser.add_argument("--direct-accuracy-threshold", type=float, default=0.95)
    parser.add_argument("--minimum-uplift", type=float, default=0.10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run a read-only pattern audit and stop when shortcuts are unmitigated."""

    args = parse_args()
    root = args.temporal_view_dir
    contract = _read_json(root / "temporal_view_contract.json")
    mitigated, evidence = _mitigation_evidence(args.mitigation_evidence_json)
    audit = audit_temporal_view_shortcuts(
        pd.read_csv(root / "temporal_view_selection_manifest.csv"),
        pd.read_csv(root / "fixed6_observed_time_manifest.csv"),
        pd.read_csv(root / "fixed6_normalized_phase_manifest.csv"),
        pd.read_csv(root / "native6_16_manifest.csv"),
        contract,
        direct_accuracy_threshold=args.direct_accuracy_threshold,
        minimum_uplift=args.minimum_uplift,
        mitigated_families=mitigated,
        require_artifact_contract=True,
    )
    audit["temporal_view_dir"] = str(root)
    audit["mitigation_evidence"] = evidence
    audit["dry_run"] = bool(args.dry_run)
    if not args.dry_run:
        write_temporal_shortcut_audit(
            audit,
            args.output_json,
            overwrite=args.overwrite,
        )
    print(json.dumps(audit, indent=2, ensure_ascii=True, allow_nan=False))
    if audit["errors"]:
        raise SystemExit(1)


def _mitigation_evidence(path: Path | None) -> tuple[list[str], dict[str, Any]]:
    """Accept mitigations only from a valid, hash-addressed evidence file."""

    if path is None:
        return [], {"provided": False}
    payload = _read_json(path)
    valid = payload.get("valid") is True
    families = payload.get("mitigated_families", []) if valid else []
    if not isinstance(families, list):
        raise ValueError("mitigation evidence families must be a list")
    return [str(value) for value in families], {
        "provided": True,
        "path": str(path),
        "sha256": _sha256(path),
        "valid": valid,
        "accepted_families": [str(value) for value in families],
    }


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject missing or non-object payloads."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    """Hash evidence bytes for experiment lineage."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
