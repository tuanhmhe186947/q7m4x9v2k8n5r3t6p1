from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.classification_v2.review.hidden_review_migration import (
    HUMAN_PAYLOAD_COLUMNS,
    carry_forward_hidden_review_decisions,
)

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_MANIFEST = (
    ROOT
    / "human_review_workspace/classification_v2/"
    "c2v2_human_review_20260720_reviewer01_v1/"
    "data/03_hidden_review/hidden_review_unit_manifest.csv"
)
CURRENT_MANIFEST = (
    ROOT
    / "human_review_workspace/classification_v2/"
    "c2v2_human_review_20260722_reviewer01_v5/"
    "data/03_hidden_review/hidden_review_unit_manifest.csv"
)
PREVIOUS_DECISIONS = (
    ROOT
    / "human_review_workspace/classification_v2/"
    "c2v2_human_review_20260720_reviewer01_v1/"
    "human_decisions/hidden/hidden_review_decisions.csv"
)
CARRY_SCRIPT = (
    ROOT
    / "scripts/classification_v2/01_review_units_gui/"
    "classification_v2_carry_forward_hidden_review_decisions.py"
)
HIDDEN_EXTERNAL_REASON = (
    "OPTIONAL_EXTERNAL_HIDDEN_V6_FIXTURE_UNAVAILABLE:"
    "supply the versioned hidden carry-forward bundle"
)


def _all_files_readable(paths: tuple[Path, ...]) -> bool:
    try:
        for path in paths:
            with path.open("rb") as handle:
                handle.read(1)
    except OSError:
        return False
    return True


def _production_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(PREVIOUS_MANIFEST, low_memory=False),
        pd.read_csv(CURRENT_MANIFEST, low_memory=False),
        pd.read_csv(PREVIOUS_DECISIONS, low_memory=False),
    )


@pytest.mark.skipif(
    not _all_files_readable(
        (
            PREVIOUS_MANIFEST,
            CURRENT_MANIFEST,
            PREVIOUS_DECISIONS,
        )
    ),
    reason=HIDDEN_EXTERNAL_REASON,
)
def test_actual_hidden_carry_partition_and_determinism() -> None:
    previous, current, decisions = _production_tables()
    carried, audit = carry_forward_hidden_review_decisions(
        previous,
        current,
        decisions,
    )
    repeated, repeated_audit = carry_forward_hidden_review_decisions(
        previous,
        current.sample(frac=1.0, random_state=20260722),
        decisions,
    )

    assert audit["errors"] == []
    assert audit["valid"] is True
    assert audit["previous_manifest_items"] == 5240
    assert audit["current_manifest_items"] == 5233
    assert audit["exact_common_items"] == 5227
    assert audit["carried_decision_items"] == 5227
    assert audit["old_only_items"] == 13
    assert audit["old_only_decision_items"] == 13
    assert audit["new_only_items"] == 6
    assert audit["unknown_decision_items"] == 0
    assert audit["identity_mismatches"] == 0
    assert audit["span_mismatches"] == 0
    assert audit["media_mismatches"] == 0
    assert audit["positional_matches"] == 0
    assert len(carried) == 5227
    assert set(carried["hidden_review_item_id"]).isdisjoint(
        audit["old_only_item_ids"]
    )
    common_before = decisions.loc[
        decisions["hidden_review_item_id"].isin(
            carried["hidden_review_item_id"]
        )
    ].reset_index(drop=True)
    for column in HUMAN_PAYLOAD_COLUMNS:
        pd.testing.assert_series_equal(
            common_before[column],
            carried[column].reset_index(drop=True),
            check_names=False,
        )
    pd.testing.assert_frame_equal(carried, repeated)
    assert audit == repeated_audit


@pytest.mark.skipif(
    not _all_files_readable(
        (
            PREVIOUS_MANIFEST,
            CURRENT_MANIFEST,
            PREVIOUS_DECISIONS,
        )
    ),
    reason=HIDDEN_EXTERNAL_REASON,
)
def test_actual_hidden_carry_cli_dry_run_and_apply(tmp_path: Path) -> None:
    dry_output = tmp_path / "dry_run_must_not_exist.csv"
    dry_audit = tmp_path / "dry_run_audit.json"
    dry = _run_cli(
        "--dry-run",
        PREVIOUS_DECISIONS,
        dry_output,
        dry_audit,
    )
    assert dry.returncode == 0, dry.stderr
    assert not dry_output.exists()
    dry_payload = json.loads(dry_audit.read_text(encoding="utf-8"))
    assert dry_payload["carried_decision_items"] == 5227
    assert dry_payload["old_only_decision_items"] == 13
    assert dry_payload["new_only_items"] == 6

    apply_output = tmp_path / "carried.csv"
    apply_audit = tmp_path / "apply_audit.json"
    applied = _run_cli(
        "--apply",
        PREVIOUS_DECISIONS,
        apply_output,
        apply_audit,
    )
    assert applied.returncode == 0, applied.stderr
    assert len(pd.read_csv(apply_output, low_memory=False)) == 5227
    apply_payload = json.loads(apply_audit.read_text(encoding="utf-8"))
    assert apply_payload["output_written"] is True
    assert apply_payload["carried_decision_items"] == 5227


@pytest.mark.skipif(
    not _all_files_readable(
        (
            PREVIOUS_MANIFEST,
            CURRENT_MANIFEST,
            PREVIOUS_DECISIONS,
        )
    ),
    reason=HIDDEN_EXTERNAL_REASON,
)
def test_actual_hidden_carry_failure_does_not_promote_csv(
    tmp_path: Path,
) -> None:
    _, _, decisions = _production_tables()
    decisions.loc[0, "hidden_review_item_id"] = "unknown-to-previous"
    invalid_decisions = tmp_path / "invalid_decisions.csv"
    decisions.to_csv(invalid_decisions, index=False)
    output = tmp_path / "must_not_exist.csv"
    audit_path = tmp_path / "failed_apply_audit.json"
    result = _run_cli(
        "--apply",
        invalid_decisions,
        output,
        audit_path,
    )

    assert result.returncode != 0
    assert not output.exists()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["unknown_decision_items"] == 1
    assert audit["output_written"] is False


@pytest.mark.skipif(
    not _all_files_readable(
        (
            PREVIOUS_MANIFEST,
            CURRENT_MANIFEST,
            PREVIOUS_DECISIONS,
        )
    ),
    reason=HIDDEN_EXTERNAL_REASON,
)
def test_actual_hidden_carry_rejects_duplicate_and_conflicting_decisions(
) -> None:
    previous, current, decisions = _production_tables()
    duplicate = decisions.iloc[[0]].copy()
    conflicting = duplicate.copy()
    conflicting["hidden_after_review"] = (
        "No" if duplicate.iloc[0]["hidden_after_review"] == "Yes" else "Yes"
    )

    for extra in (duplicate, conflicting):
        changed = pd.concat([decisions, extra], ignore_index=True)
        _, audit = carry_forward_hidden_review_decisions(
            previous,
            current,
            changed,
        )
        assert audit["valid"] is False
        assert any(
            error.startswith("duplicate_decision_item_id=")
            for error in audit["errors"]
        )


@pytest.mark.skipif(
    not _all_files_readable(
        (
            PREVIOUS_MANIFEST,
            CURRENT_MANIFEST,
            PREVIOUS_DECISIONS,
        )
    ),
    reason=HIDDEN_EXTERNAL_REASON,
)
def test_actual_hidden_carry_csv_promotion_failure_is_transactional(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "output-is-a-directory"
    output_directory.mkdir()
    audit_path = tmp_path / "failed_promotion_audit.json"
    result = _run_cli(
        "--apply",
        PREVIOUS_DECISIONS,
        output_directory,
        audit_path,
        overwrite=True,
    )

    assert result.returncode != 0
    assert output_directory.is_dir()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["output_written"] is False
    assert audit["output_decisions_sha256"] is None
    assert any(
        error.startswith("decision_csv_promotion_failed=")
        for error in audit["errors"]
    )


@pytest.mark.skipif(
    not _all_files_readable(
        (
            PREVIOUS_MANIFEST,
            CURRENT_MANIFEST,
            PREVIOUS_DECISIONS,
        )
    ),
    reason=HIDDEN_EXTERNAL_REASON,
)
def test_actual_common_mismatch_categories_fail_closed() -> None:
    previous, current, decisions = _production_tables()
    item_id = decisions.iloc[0]["hidden_review_item_id"]
    row = current["hidden_review_item_id"].eq(item_id)
    mutations = (
        ("pig_id", "wrong-actor", "identity_mismatches"),
        ("frame_index", -1, "span_mismatches"),
        ("image_name", "wrong-media.png", "media_mismatches"),
    )
    for column, value, field in mutations:
        changed = current.copy()
        changed.loc[row, column] = value
        _, audit = carry_forward_hidden_review_decisions(
            previous,
            changed,
            decisions,
        )
        assert audit["valid"] is False
        assert audit[field] == 1


def _run_cli(
    mode: str,
    decisions_path: Path,
    output_path: Path,
    audit_path: Path,
    *,
    overwrite: bool = False,
) -> subprocess.CompletedProcess[str]:
    arguments = [
        sys.executable,
        str(CARRY_SCRIPT),
        "--previous-manifest-csv",
        str(PREVIOUS_MANIFEST),
        "--current-manifest-csv",
        str(CURRENT_MANIFEST),
        "--decisions-csv",
        str(decisions_path),
        "--output-decisions-csv",
        str(output_path),
        "--audit-json",
        str(audit_path),
        mode,
    ]
    if overwrite:
        arguments.append("--overwrite")
    return subprocess.run(
        arguments,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
