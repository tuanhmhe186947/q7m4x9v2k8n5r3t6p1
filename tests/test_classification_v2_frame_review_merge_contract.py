import pandas as pd
import pytest

from pig_behavior.classification_v2.features.review_policy import (
    _merge_review_decisions,
)


def _frames() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_row_index": [0, 1, 2],
            "behavior": ["stand", "eat", "lying"],
        }
    )


def _decision(row_index: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_row_index": [row_index],
            "manual_review_decision": ["accept"],
            "manual_corrected_behavior": [""],
            "manual_label_strength": ["strong"],
            "manual_training_action": ["include"],
            "manual_sample_weight": [1.0],
            "manual_note": ["checked"],
        }
    )


def test_frame_review_merge_preserves_rows_and_applies_unique_payload() -> None:
    merged = _merge_review_decisions(_frames(), _decision())

    assert len(merged) == 3
    assert merged.loc[merged["review_row_index"].eq(1),
                      "manual_review_decision"].item() == "accept"


def test_frame_review_merge_rejects_duplicate_decision_key() -> None:
    decisions = pd.concat([_decision(), _decision()], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate_decision_review_key_rows=2"):
        _merge_review_decisions(_frames(), decisions)


def test_frame_review_merge_rejects_unmatched_decision_key() -> None:
    with pytest.raises(ValueError, match="unmatched_decision_review_key_rows=1"):
        _merge_review_decisions(_frames(), _decision(row_index=99))


def test_frame_review_merge_rejects_duplicate_frame_key() -> None:
    frames = pd.concat([_frames(), _frames().iloc[[1]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate_frame_review_key_rows=2"):
        _merge_review_decisions(frames, _decision())
