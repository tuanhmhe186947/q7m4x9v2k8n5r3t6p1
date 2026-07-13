import pandas as pd
import pytest

from pig_behavior.classification_v2.review.review_unit_builder import (
    _input_contract_errors,
    _validate_window_review_overlay_keys,
)


def _windows() -> pd.DataFrame:
    return pd.DataFrame({"window_id": ["window-0", "window-1"]})


def _review_overlay() -> pd.DataFrame:
    return pd.DataFrame({"window_id": ["window-0"]})


def test_window_review_overlay_accepts_unique_window_subset() -> None:
    _validate_window_review_overlay_keys(_windows(), _review_overlay())


def test_window_review_overlay_rejects_duplicate_decision_key() -> None:
    overlay = pd.concat([_review_overlay(), _review_overlay()], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate_window_review_id_rows=2"):
        _validate_window_review_overlay_keys(_windows(), overlay)


def test_window_review_overlay_rejects_unknown_window_key() -> None:
    overlay = pd.DataFrame({"window_id": ["window-missing"]})

    with pytest.raises(ValueError, match="unknown_window_review_id_rows=1"):
        _validate_window_review_overlay_keys(_windows(), overlay)


def test_review_unit_contract_rejects_unit_without_window_coverage() -> None:
    intervals = pd.DataFrame({"temporal_unit_key": ["unit-0"]})
    windows = pd.DataFrame({"window_id": ["window-0"]})
    units = pd.DataFrame(
        {
            "temporal_unit_key": ["unit-0"],
            "affected_window_count": [0],
        }
    )

    errors = _input_contract_errors(intervals, windows, units)

    assert "review_units_without_window_coverage=1" in errors
