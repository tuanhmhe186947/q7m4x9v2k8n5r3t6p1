from __future__ import annotations

import pandas as pd
import pytest

from pig_behavior.classification_v2.evaluation.prediction_schema_contract import (
    check_prediction_schema,
)


def _valid_prediction_row() -> dict[str, object]:
    """Return the smallest valid row for the prediction exchange contract."""

    return {
        "temporal_unit_key": "unit-0",
        "window_id": "window-0",
        "behavior_true": "stand",
        "behavior_pred": "stand",
        "window_sample_weight": 1.0,
        "window_valid_for_main_train": True,
        "oof_fold_id": "fold-0",
        "experiment_role": "oof_test",
    }


@pytest.mark.parametrize(
    "identifier_column",
    ["identifier_schema_version", "scene_frame_uid", "frame_uid"],
)
def test_prediction_exchange_rejects_frame_identifiers(
    identifier_column: str,
) -> None:
    """Frame/object identifiers are lineage fields, never metric payload fields."""

    row = _valid_prediction_row()
    row[identifier_column] = "forbidden-value"

    audit = check_prediction_schema(pd.DataFrame([row]))

    assert audit["valid"] is False
    assert audit["errors"] == [
        f"forbidden_prediction_columns=['{identifier_column}']"
    ]
