from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def gui_module() -> dict[str, object]:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "classification_v2"
        / "01_review_units_gui"
        / "review_hidden_quality_gui.py"
    )
    return runpy.run_path(str(script_path))


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hidden_review_item_id": ["item-1", "item-2", "item-3", "item-4"],
        }
    )


def _decision(item_id: str) -> dict[str, str]:
    return {
        "hidden_review_item_id": item_id,
        "hidden_after_review": "No",
        "hidden_review_status": "reviewed",
        "hidden_review_confidence": "high",
        "hidden_review_reason": "clearly_visible",
    }


def _decision_frame(
    gui_module: dict[str, object],
    item_ids: list[str],
) -> pd.DataFrame:
    columns = gui_module["DECISION_COLUMNS"]
    rows = [_decision(item_id) for item_id in item_ids]
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame[columns]


def test_resume_back_reopens_only_the_requested_prior_items(
    gui_module: dict[str, object],
) -> None:
    select_review_items = gui_module["select_review_items"]
    decisions = {
        "item-1": _decision("item-1"),
        "item-2": _decision("item-2"),
    }

    items = select_review_items(
        _manifest(),
        decisions,
        include_reviewed=False,
        resume_back=1,
    )

    assert items["hidden_review_item_id"].tolist() == [
        "item-2",
        "item-3",
        "item-4",
    ]


def test_default_resume_keeps_completed_items_hidden(
    gui_module: dict[str, object],
) -> None:
    select_review_items = gui_module["select_review_items"]
    decisions = {
        "item-1": _decision("item-1"),
        "item-2": _decision("item-2"),
    }

    items = select_review_items(
        _manifest(),
        decisions,
        include_reviewed=False,
        resume_back=0,
    )

    assert items["hidden_review_item_id"].tolist() == ["item-3", "item-4"]


@pytest.mark.parametrize(
    ("resolved", "expected"),
    [
        (["item-2", "item-4"], ["item-1", "item-3"]),
        (["item-1", "item-2", "item-3"], ["item-4"]),
        (["item-2", "item-3", "item-4"], ["item-1"]),
        (["item-1", "item-2", "item-3", "item-4"], []),
        ([], ["item-1", "item-2", "item-3", "item-4"]),
    ],
)
def test_exact_unresolved_set_difference_preserves_manifest_order(
    gui_module: dict[str, object],
    resolved: list[str],
    expected: list[str],
) -> None:
    builder = gui_module["build_review_worklist"]
    items, audit = builder(
        _manifest(),
        _decision_frame(gui_module, resolved),
        include_reviewed=False,
        resume_back=0,
    )

    assert audit["errors"] == []
    assert audit["unresolved_items"] == len(expected)
    assert items["hidden_review_item_id"].tolist() == expected


def test_resume_back_rejects_conflicting_or_out_of_range_requests(
    gui_module: dict[str, object],
) -> None:
    select_review_items = gui_module["select_review_items"]

    with pytest.raises(ValueError, match="include-reviewed"):
        select_review_items(
            _manifest(),
            {},
            include_reviewed=True,
            resume_back=1,
        )
    with pytest.raises(ValueError, match="manifest item count"):
        select_review_items(
            _manifest(),
            {},
            include_reviewed=False,
            resume_back=5,
        )


def test_resume_back_prepends_only_bounded_resolved_revisits(
    gui_module: dict[str, object],
) -> None:
    builder = gui_module["build_review_worklist"]
    manifest = pd.DataFrame(
        {"hidden_review_item_id": [f"item-{index}" for index in range(1, 8)]}
    )
    decisions = _decision_frame(
        gui_module,
        ["item-1", "item-2", "item-4", "item-6"],
    )

    first, first_audit = builder(
        manifest,
        decisions,
        include_reviewed=False,
        resume_back=2,
    )
    repeated, repeated_audit = builder(
        manifest,
        decisions.sample(frac=1.0, random_state=20260722),
        include_reviewed=False,
        resume_back=2,
    )

    assert first["hidden_review_item_id"].tolist() == [
        "item-1",
        "item-2",
        "item-3",
        "item-5",
        "item-7",
    ]
    assert first["_worklist_role"].tolist() == [
        "revisit",
        "revisit",
        "unresolved",
        "unresolved",
        "unresolved",
    ]
    assert first_audit["revisit_items"] == 2
    pd.testing.assert_frame_equal(first, repeated)
    assert first_audit == repeated_audit


def test_mutable_risk_metadata_does_not_requeue_resolved_items(
    gui_module: dict[str, object],
) -> None:
    builder = gui_module["build_review_worklist"]
    manifest = _manifest().assign(
        hidden_false_negative_risk_score=[0.1, 0.2, 0.3, 0.4],
        hidden_false_negative_risk_reasons=["new"] * 4,
    )
    decisions = _decision_frame(
        gui_module,
        ["item-1", "item-2", "item-3"],
    )
    decisions["hidden_false_negative_risk_score"] = "999"
    decisions["hidden_false_negative_risk_reasons"] = "stale"

    items, audit = builder(
        manifest,
        decisions,
        include_reviewed=False,
        resume_back=0,
    )

    assert audit["errors"] == []
    assert items["hidden_review_item_id"].tolist() == ["item-4"]


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("duplicate", "duplicate_decision_keys=1"),
        ("conflicting", "duplicate_decision_keys=1"),
        ("unknown", "unknown_decision_keys=1"),
        ("unsupported", "unsupported_decision_status=1"),
        ("malformed", "malformed_decision_rows=1"),
        ("blank", "blank_decision_keys=1"),
    ],
)
def test_worklist_validation_fails_closed_on_invalid_decisions(
    gui_module: dict[str, object],
    mutation: str,
    expected_error: str,
) -> None:
    builder = gui_module["build_review_worklist"]
    decisions = _decision_frame(gui_module, ["item-1", "item-2"])
    if mutation == "duplicate":
        decisions = pd.concat([decisions, decisions.iloc[[0]]], ignore_index=True)
    elif mutation == "conflicting":
        conflict = decisions.iloc[[0]].copy()
        conflict["hidden_after_review"] = "Yes"
        conflict["hidden_review_reason"] = "occluded_by_pig"
        decisions = pd.concat([decisions, conflict], ignore_index=True)
    elif mutation == "unknown":
        decisions.loc[0, "hidden_review_item_id"] = "unknown"
    elif mutation == "unsupported":
        decisions.loc[0, "hidden_review_status"] = "invented"
    elif mutation == "malformed":
        decisions.loc[0, "hidden_after_review"] = ""
    elif mutation == "blank":
        decisions.loc[0, "hidden_review_item_id"] = ""

    _, audit = builder(
        _manifest(),
        decisions,
        include_reviewed=False,
        resume_back=0,
    )

    assert expected_error in audit["errors"]


def test_worklist_validation_rejects_blank_or_duplicate_manifest_keys(
    gui_module: dict[str, object],
) -> None:
    builder = gui_module["build_review_worklist"]
    decisions = _decision_frame(gui_module, [])
    manifest = _manifest()
    manifest.loc[0, "hidden_review_item_id"] = ""
    manifest.loc[1, "hidden_review_item_id"] = "item-3"

    _, audit = builder(
        manifest,
        decisions,
        include_reviewed=False,
        resume_back=0,
    )

    assert "blank_manifest_keys=1" in audit["errors"]
    assert "duplicate_manifest_keys=1" in audit["errors"]


def test_resume_back_relabel_replaces_one_row_and_undo_restores_it(
    gui_module: dict[str, object],
    tmp_path: Path,
) -> None:
    select_review_items = gui_module["select_review_items"]
    write_decisions = gui_module["write_decisions"]
    load_decisions = gui_module["load_decisions"]
    decisions = {
        item_id: _decision(item_id)
        for item_id in ("item-1", "item-2", "item-3")
    }
    decision_path = tmp_path / "hidden_review_decisions.csv"
    write_decisions(decision_path, decisions)
    before = load_decisions(decision_path)
    original = {item_id: record.copy() for item_id, record in before.items()}

    items = select_review_items(
        _manifest(),
        before,
        include_reviewed=False,
        resume_back=2,
    )
    assert items["hidden_review_item_id"].tolist() == [
        "item-2",
        "item-3",
        "item-4",
    ]

    previous = before["item-2"].copy()
    relabeled = previous.copy()
    relabeled.update(
        {
            "hidden_after_review": "Yes",
            "hidden_review_reason": "occluded_or_not_visible",
            "hidden_reviewed_at": "2026-07-20T16:00:00",
        }
    )
    before["item-2"] = relabeled
    write_decisions(decision_path, before)
    after_relabel = load_decisions(decision_path)

    assert len(after_relabel) == 3
    assert len(set(after_relabel)) == 3
    assert after_relabel["item-2"]["hidden_after_review"] == "Yes"
    assert after_relabel["item-1"] == original["item-1"]
    assert after_relabel["item-3"] == original["item-3"]

    after_relabel["item-2"] = previous
    write_decisions(decision_path, after_relabel)

    assert load_decisions(decision_path) == original


def test_no_op_save_and_exit_preserves_decision_csv_bytes(
    gui_module: dict[str, object],
    tmp_path: Path,
) -> None:
    decision_path = tmp_path / "hidden_review_decisions.csv"
    original = b"hidden_review_item_id,opaque\r\nitem-1,preserved\r\n"
    decision_path.write_bytes(original)

    closed: list[str] = []
    fake_app = SimpleNamespace(
        reader=SimpleNamespace(close=lambda: closed.append("reader")),
        root=SimpleNamespace(destroy=lambda: closed.append("root")),
        decision_path=decision_path,
        decisions={"item-1": _decision("item-1")},
    )
    gui_module["HiddenQualityReviewApp"]._save_and_exit(fake_app)

    assert decision_path.read_bytes() == original
    assert closed == ["reader", "root"]
