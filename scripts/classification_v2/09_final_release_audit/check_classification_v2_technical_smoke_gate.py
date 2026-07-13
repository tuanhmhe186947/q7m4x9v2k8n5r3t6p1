"""Consolidate bounded data-science smoke evidence without training approval."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.contracts.technical_smoke_gate import (
    audit_technical_smoke_gate,
)

DEFAULT_ROOT = Path("outputs/classification_v2/rebuilds/scientific_smoke_v1")


def parse_args() -> argparse.Namespace:
    """Parse versioned smoke roots and an explicit audit destination."""

    parser = argparse.ArgumentParser(
        description=(
            "Reconcile the bounded legacy+CVAT scientific smoke chain. This "
            "never authorizes reviewed training data or full OOF."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repeat-root", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing technical gate JSON explicitly.",
    )
    return parser.parse_args()


def main() -> None:
    """Load stage audits, verify repeatability, and write one final gate."""

    args = parse_args()
    root = args.root
    repeat_root = args.repeat_root or root / "05_fail_closed_recheck"
    output_path = args.output_json or root / "audits" / "technical_smoke_gate.json"
    require_output_paths_available([output_path], overwrite=args.overwrite)

    paths = _audit_paths(root)
    payloads, preload_errors = _load_payloads(paths)
    repeatability = _repeatability_audit(root, repeat_root)
    decision_files = _decision_files(root / "03_review_units")
    gate = audit_technical_smoke_gate(
        payloads,
        repeatability=repeatability,
        decision_files=decision_files,
        preload_errors=preload_errors,
    )
    gate["inputs"] = {name: str(path) for name, path in paths.items()}
    gate["repeat_root"] = str(repeat_root)
    gate["artifact_sha256"] = {
        str(path): _sha256(path)
        for path in [*paths.values(), *_repeatability_paths(root, repeat_root)]
        if path.exists()
    }
    gate["code_state"] = _git_state()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": gate["status"],
                "technical_pass": gate["technical_pass"],
                "counts": gate["counts"],
                "repeatability": gate["repeatability"],
                "human_gate_blockers": gate["human_gate_blockers"],
                "errors": gate["errors"],
                "warnings": gate["warnings"],
                "output_json": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not gate["technical_pass"]:
        raise SystemExit(2)


def _audit_paths(root: Path) -> dict[str, Path]:
    """Return the fixed stage artifacts for one versioned smoke root."""

    return {
        "scope": root / "00_scope" / "smoke_scope_audit.json",
        "enhanced": root / "01_enhanced" / "enhanced_audit.json",
        "sequence": root / "02_sequence" / "sequence_window_audit.json",
        "review_units": root / "03_review_units" / "review_unit_audit.json",
        "temporal_evidence": root / "audits" / "temporal_evidence_audit.json",
        "train_ready": root / "04_train_ready" / "train_ready_audit.json",
        "feature_semantics": root / "audits" / "feature_semantics_audit.json",
        "spatial_validation": root
        / "audits"
        / "spatial_sequence_validation.json",
    }


def _load_payloads(
    paths: dict[str, Path],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Load required JSONs and retain every missing/invalid file as an error."""

    payloads: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for name, path in paths.items():
        if not path.exists():
            errors.append(f"missing_audit={name}:{path}")
            payloads[name] = {}
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid_audit={name}:{path}:{exc}")
            payloads[name] = {}
            continue
        if not isinstance(payload, dict):
            errors.append(f"audit_not_object={name}:{path}")
            payloads[name] = {}
            continue
        payloads[name] = payload
    return payloads, errors


def _repeatability_paths(root: Path, repeat_root: Path) -> list[Path]:
    """List original and rerun CSVs whose bytes must remain identical."""

    pairs = _repeatability_pairs(root, repeat_root)
    return [path for pair in pairs for path in pair]


def _repeatability_pairs(
    root: Path,
    repeat_root: Path,
) -> list[tuple[Path, Path]]:
    """Pair each scientific table with its explicit-overwrite bounded rerun."""

    original_enhanced = root / "01_enhanced"
    original_sequence = root / "02_sequence"
    repeat_enhanced = repeat_root / "enhanced"
    repeat_sequence = repeat_root / "sequence"
    return [
        (
            original_enhanced / "spatiotemporal_frame_features_enhanced.csv",
            repeat_enhanced / "spatiotemporal_frame_features_enhanced.csv",
        ),
        (
            original_sequence
            / "training_ready_frame_features_harmonized_preview.csv",
            repeat_sequence
            / "training_ready_frame_features_harmonized_preview.csv",
        ),
        (
            original_sequence / "temporal_label_intervals.csv",
            repeat_sequence / "temporal_label_intervals.csv",
        ),
        (
            original_sequence / "sequence_window_manifest.csv",
            repeat_sequence / "sequence_window_manifest.csv",
        ),
        (
            original_sequence / "sequence_window_features.csv",
            repeat_sequence / "sequence_window_features.csv",
        ),
    ]


def _repeatability_audit(root: Path, repeat_root: Path) -> dict[str, Any]:
    """Hash paired CSVs and make missing artifacts an explicit mismatch."""

    comparisons: list[dict[str, Any]] = []
    for original, repeated in _repeatability_pairs(root, repeat_root):
        original_hash = _sha256(original) if original.exists() else None
        repeated_hash = _sha256(repeated) if repeated.exists() else None
        comparisons.append(
            {
                "original": str(original),
                "repeated": str(repeated),
                "original_sha256": original_hash,
                "repeated_sha256": repeated_hash,
                "match": (
                    original_hash is not None
                    and repeated_hash is not None
                    and original_hash == repeated_hash
                ),
            }
        )
    return {
        "pair_count": len(comparisons),
        "matching_pair_count": sum(bool(item["match"]) for item in comparisons),
        "all_match": bool(comparisons)
        and all(bool(item["match"]) for item in comparisons),
        "comparisons": comparisons,
    }


def _decision_files(review_dir: Path) -> list[str]:
    """Reject accidental synthetic human decisions inside the smoke output."""

    if not review_dir.exists():
        return []
    return sorted(
        str(path)
        for path in review_dir.rglob("*.csv")
        if "decision" in path.name.lower()
    )


def _sha256(path: Path) -> str:
    """Hash one artifact without loading a potentially large CSV into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, Any]:
    """Record code SHA and dirty paths without treating user edits as failures."""

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    dirty_paths = [line for line in status.stdout.splitlines() if line.strip()]
    return {
        "git_sha": sha.stdout.strip() if sha.returncode == 0 else None,
        "dirty_worktree": bool(dirty_paths),
        "dirty_paths": dirty_paths,
    }


if __name__ == "__main__":
    main()
