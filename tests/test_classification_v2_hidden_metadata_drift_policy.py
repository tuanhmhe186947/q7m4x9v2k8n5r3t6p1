from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.classification_v2.review.hidden_review_builder import (
    audit_hidden_decision_coverage,
)

ROOT = Path(__file__).resolve().parents[1]
COVERAGE_SCRIPT = (
    ROOT
    / "scripts/classification_v2/01_review_units_gui/"
    "check_hidden_review_decision_coverage.py"
)
SCIENCE_SCRIPT = (
    ROOT
    / "scripts/classification_v2/01_review_units_gui/"
    "check_hidden_review_scientific_gate.py"
)
V6_ROOT = (
    ROOT
    / "human_review_workspace/classification_v2/"
    "c2v2_human_review_20260722_reviewer01_v6"
)
V6_HIDDEN = V6_ROOT / "data/03_hidden_review"
V6_MANIFEST = V6_HIDDEN / "hidden_review_unit_manifest.csv"
V6_DECISIONS = V6_ROOT / "human_decisions/hidden/hidden_review_decisions.csv"
V6_DESIGN = V6_HIDDEN / "hidden_review_scientific_design.json"


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hidden_review_item_id": ["item-1", "item-2", "item-3"],
            "hidden_before_review": ["No", "No", "Yes"],
            "source_type": ["cvat_tracking_xml"] * 3,
            "video_key": ["video-1"] * 3,
            "source_video_path": ["video-1.mp4"] * 3,
            "frame_uid": ["frame-1", "frame-2", "frame-3"],
            "frame_index": [1, 2, 3],
            "temporal_unit_key": ["unit-1", "unit-1", "unit-1"],
            "pig_id": ["ID_1"] * 3,
            "track_id": ["track-1"] * 3,
            "crop_sha256": ["crop-1", "crop-2", "crop-3"],
            "hidden_review_cohort": ["cohort-a"] * 3,
            "hidden_false_negative_risk_score": [0.1, 0.2, 0.3],
            "hidden_false_negative_risk_reasons": ["a", "b", "c"],
            "unapproved_metadata": ["locked"] * 3,
        }
    )


def _decisions(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    copied_metadata = [
        "source_type",
        "video_key",
        "source_video_path",
        "frame_uid",
        "frame_index",
        "temporal_unit_key",
        "pig_id",
        "track_id",
        "crop_sha256",
        "hidden_review_cohort",
        "hidden_false_negative_risk_score",
        "hidden_false_negative_risk_reasons",
        "unapproved_metadata",
    ]
    for _, item in manifest.iterrows():
        row = {
            "hidden_review_item_id": item["hidden_review_item_id"],
            "hidden_before_review": item["hidden_before_review"],
            "hidden_after_review": item["hidden_before_review"],
            "hidden_review_status": "reviewed",
            "hidden_review_confidence": "high",
            "hidden_review_reason": "synthetic_test",
            "hidden_reviewer": "pytest",
            "hidden_reviewed_at": "2026-07-22T00:00:00",
        }
        row.update({column: item[column] for column in copied_metadata})
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("hidden_review_cohort", "changed"),
        ("hidden_false_negative_risk_reasons", "changed"),
        ("hidden_false_negative_risk_score", 99.0),
    ],
)
def test_approved_risk_metadata_drift_is_nonfatal_and_audited(
    column: str,
    value: object,
) -> None:
    manifest = _manifest()
    decisions = _decisions(manifest)
    decisions.loc[0, column] = value

    audit = audit_hidden_decision_coverage(manifest, decisions)

    assert audit["errors"] == []
    assert audit["decision_metadata_drift_counts"] == {column: 1}
    assert audit["decision_metadata_drift_unique_items"] == 1
    assert audit["metadata_drift_policy"] == (
        "derived_review_and_audit_metadata_is_nonfatal"
    )
    assert audit["warnings"]


def test_both_approved_risk_fields_keep_exact_counts() -> None:
    manifest = _manifest()
    decisions = _decisions(manifest)
    decisions.loc[[0, 1], "hidden_false_negative_risk_reasons"] = "changed"
    decisions.loc[[1], "hidden_false_negative_risk_score"] = 99.0

    audit = audit_hidden_decision_coverage(manifest, decisions)

    assert audit["errors"] == []
    assert audit["decision_metadata_drift_counts"] == {
        "hidden_false_negative_risk_reasons": 2,
        "hidden_false_negative_risk_score": 1,
    }
    assert audit["decision_metadata_drift_unique_items"] == 2


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("source_video_path", "mismatch"),
        ("frame_uid", "mismatch"),
        ("frame_index", 999),
        ("temporal_unit_key", "mismatch"),
        ("pig_id", "mismatch"),
        ("track_id", "mismatch"),
        ("crop_sha256", "mismatch"),
        ("unapproved_metadata", "mismatch"),
    ],
)
def test_unapproved_identity_media_span_or_metadata_drift_fails_closed(
    column: str,
    value: object,
) -> None:
    manifest = _manifest()
    decisions = _decisions(manifest)
    decisions.loc[0, column] = value

    audit = audit_hidden_decision_coverage(manifest, decisions)

    assert f"decision_metadata_mismatch:{column}=1" in audit["errors"]


def test_canonical_and_decision_payload_mismatch_fail_closed() -> None:
    manifest = _manifest()
    decisions = _decisions(manifest)
    decisions.loc[0, "hidden_review_item_id"] = "unknown-item"
    key_audit = audit_hidden_decision_coverage(manifest, decisions)
    assert "missing_decision_items=1" in key_audit["errors"]
    assert "unknown_decision_items=1" in key_audit["errors"]

    decisions = _decisions(manifest)
    decisions.loc[0, "hidden_before_review"] = "Yes"
    payload_audit = audit_hidden_decision_coverage(manifest, decisions)
    assert "stale_hidden_before_review=1" in payload_audit["errors"]


def test_missing_unknown_duplicate_and_conflicting_decisions_fail_closed() -> None:
    manifest = _manifest()
    decisions = _decisions(manifest)
    missing = audit_hidden_decision_coverage(manifest, decisions.iloc[:-1])
    assert "missing_decision_items=1" in missing["errors"]

    unknown_rows = pd.concat(
        [decisions, decisions.iloc[[0]].assign(hidden_review_item_id="unknown")],
        ignore_index=True,
    )
    unknown = audit_hidden_decision_coverage(manifest, unknown_rows)
    assert "unknown_decision_items=1" in unknown["errors"]

    duplicate_rows = pd.concat([decisions, decisions.iloc[[0]]], ignore_index=True)
    duplicate = audit_hidden_decision_coverage(manifest, duplicate_rows)
    assert "duplicate_decision_items=1" in duplicate["errors"]

    conflict = decisions.iloc[[0]].copy()
    conflict["hidden_after_review"] = "Yes"
    conflict_rows = pd.concat([decisions, conflict], ignore_index=True)
    conflicting = audit_hidden_decision_coverage(manifest, conflict_rows)
    assert "duplicate_decision_items=1" in conflicting["errors"]


def test_blank_key_unsupported_status_and_malformed_payload_fail_closed() -> None:
    manifest = _manifest()
    decisions = _decisions(manifest)
    decisions.loc[0, "hidden_review_item_id"] = ""
    blank = audit_hidden_decision_coverage(manifest, decisions)
    assert "blank_decision_items=1" in blank["errors"]

    decisions = _decisions(manifest)
    decisions.loc[0, "hidden_review_status"] = "invented"
    unsupported = audit_hidden_decision_coverage(manifest, decisions)
    assert "unsupported_decision_status=1" in unsupported["errors"]

    decisions = _decisions(manifest)
    decisions.loc[0, "hidden_review_confidence"] = "low"
    malformed = audit_hidden_decision_coverage(manifest, decisions)
    assert "low_confidence_requires_unclear=1" in malformed["errors"]


HIDDEN_EXTERNAL_REASON = (
    "OPTIONAL_EXTERNAL_HIDDEN_V6_FIXTURE_UNAVAILABLE:"
    "supply the versioned v6 human-review bundle"
)


def _all_files_readable(paths: tuple[Path, ...]) -> bool:
    try:
        for path in paths:
            with path.open("rb") as handle:
                handle.read(1)
    except OSError:
        return False
    return True


@pytest.mark.skipif(
    not _all_files_readable((V6_MANIFEST, V6_DECISIONS, V6_DESIGN)),
    reason=HIDDEN_EXTERNAL_REASON,
)
def test_actual_v6_coverage_and_scientific_cli_pass_with_drift(
    tmp_path: Path,
) -> None:
    coverage_path = tmp_path / "coverage.json"
    coverage = subprocess.run(
        [
            sys.executable,
            str(COVERAGE_SCRIPT),
            "--manifest-csv",
            str(V6_MANIFEST),
            "--decisions-csv",
            str(V6_DECISIONS),
            "--audit-json",
            str(coverage_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert coverage.returncode == 0, coverage.stderr
    coverage_audit = json.loads(coverage_path.read_text(encoding="utf-8"))
    assert coverage_audit["decision_metadata_drift_counts"] == {
        "hidden_false_negative_risk_reasons": 82,
        "hidden_false_negative_risk_score": 16,
    }
    assert coverage_audit["errors"] == []
    assert len(coverage_audit["checker_code_sha"]) == 40

    science_path = tmp_path / "science.json"
    science = subprocess.run(
        [
            sys.executable,
            str(SCIENCE_SCRIPT),
            "--manifest-csv",
            str(V6_MANIFEST),
            "--decisions-csv",
            str(V6_DECISIONS),
            "--design-json",
            str(V6_DESIGN),
            "--audit-json",
            str(science_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert science.returncode == 0, science.stderr
    science_audit = json.loads(science_path.read_text(encoding="utf-8"))
    assert science_audit["status"] == "PASS"
    assert science_audit["errors"] == []
    assert science_audit["decision_metadata_drift_counts"] == {
        "hidden_false_negative_risk_reasons": 82,
        "hidden_false_negative_risk_score": 16,
    }
