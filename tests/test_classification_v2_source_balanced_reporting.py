import pandas as pd
import pytest

from pig_behavior.classification_v2.evaluation.source_balanced_reporting import (
    build_source_balanced_native_report,
)


def test_source_balancing_operates_on_native_units_and_preserves_exclusions() -> None:
    """Overlapping windows cannot increase source quota, and excluded units remain audited."""

    predictions, metadata = _inputs()

    units, selection, report = build_source_balanced_native_report(
        predictions,
        metadata,
        expected_fold_count=2,
        paper_facing_run_verified=False,
    )

    assert len(units) == 6
    assert len(selection) == 6
    assert int(selection["source_balance_keep"].sum()) == 4
    drink = selection.loc[selection["behavior_true"].eq("drink") & selection["source_balance_keep"]]
    assert drink["source_type"].value_counts().to_dict() == {
        "cvat_tracking_xml": 1,
        "legacy_recovered": 1,
    }
    assert report["valid"] is True
    assert report["paper_facing_ready"] is False
    assert report["excluded_native_unit_rows"] == 2


def test_source_balancing_rejects_source_conflict_inside_native_unit() -> None:
    """A native review unit cannot be assigned to two source domains."""

    predictions, metadata = _inputs()
    metadata.loc[metadata["window_id"].eq("w1b"), "source_type"] = "cvat_tracking_xml"

    with pytest.raises(ValueError, match="conflicts"):
        build_source_balanced_native_report(predictions, metadata)


def test_single_source_never_marks_units_as_source_matched() -> None:
    """A one-domain pilot must retain units but assign zero matched quota."""

    predictions, metadata = _inputs()
    metadata["source_type"] = "legacy_recovered"

    _, selection, report = build_source_balanced_native_report(predictions, metadata)

    assert report["valid"] is False
    assert int(selection["source_balance_keep"].sum()) == 0
    assert set(selection["source_balance_reason"]) == {"label_missing_one_or_more_sources"}


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        ("w1a", "u1", "fold_a", "drink", "drink"),
        ("w1b", "u1", "fold_a", "drink", "drink"),
        ("w2", "u2", "fold_a", "drink", "eat"),
        ("w3", "u3", "fold_b", "drink", "drink"),
        ("w4", "u4", "fold_a", "eat", "eat"),
        ("w5", "u5", "fold_b", "eat", "drink"),
        ("w6", "u6", "fold_a", "drink", "drink"),
    ]
    predictions = pd.DataFrame(
        rows,
        columns=["window_id", "temporal_unit_key", "oof_fold_id", "behavior_true", "behavior_pred"],
    )
    predictions["window_sample_weight"] = 1.0
    predictions["window_valid_for_main_train"] = True
    metadata = pd.DataFrame(
        {
            "window_id": predictions["window_id"],
            "source_type": [
                "legacy_recovered",
                "legacy_recovered",
                "legacy_recovered",
                "cvat_tracking_xml",
                "legacy_recovered",
                "cvat_tracking_xml",
                "legacy_recovered",
            ],
        }
    )
    return predictions, metadata
