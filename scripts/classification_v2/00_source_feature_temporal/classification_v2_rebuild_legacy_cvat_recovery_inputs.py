"""Rebuild legacy recovery scaffold and six-anchor tables from native CVAT."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from legacy_burst_recovery.cvat_anchor_rebuild import (
    build_legacy_recovery_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cvat-export-root", type=Path, required=True)
    parser.add_argument("--metadata-scaffold-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--behavior-authority-slot", type=int, default=0)
    parser.add_argument(
        "--min-anchor-count",
        type=int,
        default=6,
        help="Canonical compatibility option; only the value 6 is accepted.",
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    audit_path = output_dir / "legacy_cvat_recovery_input_audit.json"
    issues_path = output_dir / "legacy_cvat_recovery_input_issues.csv"
    center_path = output_dir / "legacy_center_keyframes_from_cvat.csv"
    anchors_path = output_dir / "legacy_six_anchor_bboxes_from_cvat.csv"
    manifest_path = output_dir / "legacy_recovery_input_manifest.json"
    targets = [audit_path, issues_path, center_path, anchors_path, manifest_path]
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing recovery-input artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    center, anchors, audit, issues = build_legacy_recovery_inputs(
        cvat_export_root=args.cvat_export_root,
        metadata_scaffold_csv=args.metadata_scaffold_csv,
        behavior_authority_slot=args.behavior_authority_slot,
        min_anchor_count=args.min_anchor_count,
    )
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    issues.to_csv(issues_path, index=False)
    print(f"status={audit['status']}")
    print(f"audit={audit_path}")
    print(f"issues={issues_path}")

    if audit["errors"]:
        print("errors:")
        for error in audit["errors"]:
            print(f"- {error}")
        raise SystemExit(2)
    if args.audit_only:
        print("audit-only: recovery input CSV files were not written")
        return

    center.to_csv(center_path, index=False)
    anchors.to_csv(anchors_path, index=False)
    manifest = {
        "schema_version": 1,
        "audit_path": str(audit_path),
        "audit_sha256": _sha256(audit_path),
        "center_csv": str(center_path),
        "center_rows": int(len(center)),
        "center_sha256": _sha256(center_path),
        "anchor_csv": str(anchors_path),
        "anchor_rows": int(len(anchors)),
        "anchor_sha256": _sha256(anchors_path),
        "behavior_authority_slot": args.behavior_authority_slot,
        "min_anchor_count": args.min_anchor_count,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"center={center_path} rows={len(center)}")
    print(f"anchors={anchors_path} rows={len(anchors)}")
    print(f"manifest={manifest_path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
