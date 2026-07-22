from __future__ import annotations

import runpy
from pathlib import Path

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
