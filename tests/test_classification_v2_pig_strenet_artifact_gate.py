from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pig_behavior.classification_v2.contracts.pig_strenet_artifact_run import (
    audit_pig_strenet_artifact_run,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_valid_run(root: Path, input_csv: Path) -> None:
    names = [
        "pair_manifest.csv",
        "history_features.csv",
        "stabilized_difference_summary.csv",
    ]
    for name in names:
        (root / name).write_text("key\nvalue\n", encoding="utf-8")

    input_sha = _sha256(input_csv)
    artifact_audit = {
        "valid": True,
        "media_contract_valid": True,
        "input_sha256": input_sha,
        "difference": {
            "status": "PASS",
            "missing_available_frame_slots": 0,
        },
        "packed_tensors": {
            "roi_visual_pixels": {
                "status": "PASS",
                "missing_expected_rows": 0,
            }
        },
    }
    _write_json(root / "pig_strenet_artifact_audit.json", artifact_audit)

    entries = []
    for path in sorted(root.iterdir()):
        if path.name == "artifact_manifest.json":
            continue
        entries.append(
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _write_json(
        root / "artifact_manifest.json",
        {"files": entries},
    )
    _write_json(
        root / "run_manifest.json",
        {
            "valid": True,
            "run_scope": "full",
            "input": {"sha256": input_sha},
            "artifact_manifest": {
                "sha256": _sha256(root / "artifact_manifest.json")
            },
        },
    )


def test_valid_pig_strenet_run_passes(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    input_csv = tmp_path / "harmonized.csv"
    input_csv.write_text("temporal_unit_key\nunit-1\n", encoding="utf-8")
    _write_valid_run(root, input_csv)

    audit = audit_pig_strenet_artifact_run(
        root,
        input_csv=input_csv,
        expected_run_scope="full",
    )

    assert audit["status"] == "PASS"
    assert audit["errors"] == []


def test_invalid_media_contract_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    input_csv = tmp_path / "harmonized.csv"
    input_csv.write_text("temporal_unit_key\nunit-1\n", encoding="utf-8")
    _write_valid_run(root, input_csv)

    audit_path = root / "pig_strenet_artifact_audit.json"
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["media_contract_valid"] = False
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    audit = audit_pig_strenet_artifact_run(
        root,
        input_csv=input_csv,
        expected_run_scope="full",
    )

    assert audit["status"] == "FAIL"
    assert "pig_strenet_media_contract_valid_false" in audit["errors"]


def test_scope_and_input_hash_are_bound(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    input_csv = tmp_path / "harmonized.csv"
    input_csv.write_text("temporal_unit_key\nunit-1\n", encoding="utf-8")
    _write_valid_run(root, input_csv)
    input_csv.write_text("temporal_unit_key\nunit-2\n", encoding="utf-8")

    audit = audit_pig_strenet_artifact_run(
        root,
        input_csv=input_csv,
        expected_run_scope="smoke",
    )

    assert audit["status"] == "FAIL"
    assert "pig_strenet_run_scope_mismatch:full!=smoke" in audit["errors"]
    assert "pig_strenet_input_sha256_mismatch" in audit["errors"]
