from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GUI_SCRIPT = (
    ROOT
    / "scripts/classification_v2/01_review_units_gui/review_hidden_quality_gui.py"
)
V6_ROOT = (
    ROOT
    / "human_review_workspace/classification_v2/"
    "c2v2_human_review_20260722_reviewer01_v6"
)
V6_MANIFEST = V6_ROOT / "data/03_hidden_review/hidden_review_unit_manifest.csv"
V6_DECISIONS = (
    V6_ROOT / "human_decisions/hidden/hidden_review_decisions.csv"
)
V6_NEW_ITEM_KEYS = frozenset(
    {
        "hidden_item_v2_e697bb6dd8ada8044745ae96",
        "hidden_item_v2_9d7eb6d50d2212786c572d6b",
        "hidden_item_v2_ca00126225ba3127b762bc03",
        "hidden_item_v2_c0f8291f3ecc9eeccba6bdcb",
        "hidden_item_v2_89bb8d44d7192ad1d36270e6",
        "hidden_item_v2_eb0df38cff70ee8717ca9a56",
    }
)


def _module() -> dict[str, object]:
    return runpy.run_path(str(GUI_SCRIPT))


def _production_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(V6_MANIFEST, low_memory=False),
        pd.read_csv(V6_DECISIONS, low_memory=False),
    )


def _historical_carried_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    """Recreate the immutable 5,227-row pre-GUI authority in memory only."""
    return decisions.loc[
        ~decisions["hidden_review_item_id"].astype(str).isin(V6_NEW_ITEM_KEYS)
    ].reset_index(drop=True)


def _reviewed_row(
    row: pd.Series,
    decision_columns: list[str],
) -> dict[str, str]:
    record = {
        column: "" if pd.isna(row.get(column, "")) else str(row.get(column, ""))
        for column in decision_columns
    }
    record.update(
        {
            "hidden_review_item_id": str(row["hidden_review_item_id"]),
            "hidden_before_review": str(row["hidden_before_review"]),
            "hidden_after_review": "No",
            "hidden_review_status": "reviewed",
            "hidden_review_confidence": "high",
            "hidden_review_reason": "clearly_visible",
            "hidden_reviewer": "production-regression",
            "hidden_reviewed_at": "2026-07-22T00:00:00",
        }
    )
    return record


def test_v6_exact_worklist_replaces_the_old_368_row_suffix() -> None:
    module = _module()
    builder = module["build_review_worklist"]
    manifest, decisions = _production_tables()
    decisions = _historical_carried_decisions(decisions)
    resolved_keys = set(decisions["hidden_review_item_id"].astype(str))
    manifest_keys = manifest["hidden_review_item_id"].astype(str)
    first_unresolved_position = next(
        index
        for index, key in enumerate(manifest_keys)
        if key not in resolved_keys
    )
    old_suffix_items = len(manifest.iloc[first_unresolved_position:])

    worklist, audit = builder(
        manifest,
        decisions,
        include_reviewed=False,
        resume_back=0,
    )

    assert old_suffix_items == 368
    assert audit == {
        **audit,
        "manifest_items": 5233,
        "input_decision_rows": 5227,
        "stored_decision_rows": 5227,
        "covered_items": 5227,
        "resolved_items": 5227,
        "unresolved_items": 6,
        "resume_back_requested": 0,
        "revisit_items": 0,
        "worklist_items": 6,
        "duplicate_manifest_keys": 0,
        "duplicate_decision_keys": 0,
        "unknown_decision_keys": 0,
        "blank_manifest_keys": 0,
        "blank_decision_keys": 0,
        "errors": [],
    }
    assert worklist["hidden_review_item_id"].tolist() == audit[
        "worklist_item_ids"
    ]

    joined = manifest.merge(
        decisions,
        on="hidden_review_item_id",
        suffixes=("_manifest", "_decision"),
    )
    reason_drift = joined["hidden_false_negative_risk_reasons_manifest"].fillna("").astype(str).ne(
        joined["hidden_false_negative_risk_reasons_decision"].fillna("").astype(str)
    )
    manifest_score = pd.to_numeric(
        joined["hidden_false_negative_risk_score_manifest"],
        errors="coerce",
    )
    decision_score = pd.to_numeric(
        joined["hidden_false_negative_risk_score_decision"],
        errors="coerce",
    )
    assert int(reason_drift.sum()) == 82
    assert int(manifest_score.sub(decision_score).abs().gt(1e-12).sum()) == 16


def test_v6_one_and_all_new_decisions_reduce_only_the_unresolved_set() -> None:
    module = _module()
    builder = module["build_review_worklist"]
    decision_columns = module["DECISION_COLUMNS"]
    manifest, decisions = _production_tables()
    decisions = _historical_carried_decisions(decisions)
    original = decisions.copy(deep=True)
    decision_keys = set(decisions["hidden_review_item_id"].astype(str))
    missing = manifest.loc[
        ~manifest["hidden_review_item_id"].astype(str).isin(decision_keys)
    ]
    additions = pd.DataFrame(
        [_reviewed_row(row, decision_columns) for _, row in missing.iterrows()],
        columns=decision_columns,
    )

    one = pd.concat([decisions, additions.head(1)], ignore_index=True)
    one_worklist, one_audit = builder(
        manifest,
        one,
        include_reviewed=False,
        resume_back=0,
    )
    assert len(one) == 5228
    assert one_audit["unresolved_items"] == 5
    assert len(one_worklist) == 5
    pd.testing.assert_frame_equal(
        one.iloc[:5227].reset_index(drop=True),
        original,
        check_dtype=False,
    )

    complete = pd.concat([decisions, additions], ignore_index=True)
    complete_worklist, complete_audit = builder(
        manifest,
        complete,
        include_reviewed=False,
        resume_back=0,
    )
    assert len(complete) == 5233
    assert complete_audit["unresolved_items"] == 0
    assert complete_worklist.empty


def test_v6_no_op_close_keeps_carried_csv_byte_identical(tmp_path: Path) -> None:
    module = _module()
    copied = tmp_path / "hidden_review_decisions.csv"
    original_bytes = V6_DECISIONS.read_bytes()
    copied.write_bytes(original_bytes)
    closed: list[str] = []
    fake_app = SimpleNamespace(
        reader=SimpleNamespace(close=lambda: closed.append("reader")),
        root=SimpleNamespace(destroy=lambda: closed.append("root")),
        decision_path=copied,
        decisions={},
    )

    module["HiddenQualityReviewApp"]._save_and_exit(fake_app)

    assert copied.read_bytes() == original_bytes
    assert closed == ["reader", "root"]
