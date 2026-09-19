from __future__ import annotations

import pandas as pd
import pytest

from pig_behavior.classification_v2.review.composite_review_authority import (
    CompositeReviewContractError,
    ReviewLayer,
    compose_behavior_review_layers,
)


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "review_unit_id": "key-1",
                "temporal_unit_key": "key-1",
                "behavior_label": "social-nose",
                "video_key": "video-1",
                "track_id": "1",
                "unit_start_frame": "0",
                "unit_end_frame": "5",
            },
            {
                "review_unit_id": "key-2",
                "temporal_unit_key": "key-2",
                "behavior_label": "stand",
                "video_key": "video-1",
                "track_id": "1",
                "unit_start_frame": "6",
                "unit_end_frame": "11",
            },
        ]
    )


def _layer(
    name: str,
    key: str,
    incoming: str,
    decision: str,
    output: str,
) -> ReviewLayer:
    corrected = output if decision == "corrected" else ""
    return ReviewLayer(
        name=name,
        decisions=pd.DataFrame(
            [
                {
                    "review_unit_id": key,
                    "temporal_unit_key": key,
                    "behavior_label": incoming,
                    "manual_review_decision": decision,
                    "manual_corrected_behavior": corrected,
                    "manual_label_strength": "strong",
                    "manual_sample_weight": "1.0",
                    "manual_note": "",
                }
            ]
        ),
        quality=pd.DataFrame(
            [
                {
                    "review_unit_id": key,
                    "original_behavior": incoming,
                    "reviewed_behavior": output,
                    "source_label_error_confirmed": (
                        "YES" if decision == "corrected" else "NO"
                    ),
                    "error_pattern": "OTHER_CLEAR_SOURCE_LABEL_ERROR",
                    "review_confidence": "HIGH",
                    "selection_assessment": "SOURCE_LABEL_ERROR_FOUND",
                }
            ]
        ),
    )


def test_compose_layers_applies_accept_relative_to_prior_review() -> None:
    result = compose_behavior_review_layers(
        _source(),
        [
            _layer("primary", "key-1", "social-nose", "corrected", "fight"),
            _layer("v3", "key-1", "fight", "accept", "fight"),
            _layer("micro", "key-1", "fight", "corrected", "move"),
            _layer("micro", "key-2", "stand", "accept", "stand"),
        ],
    )

    decisions = result["decisions"].set_index("temporal_unit_key")
    assert decisions.at["key-1", "behavior_label"] == "social-nose"
    assert decisions.at["key-1", "manual_review_decision"] == "corrected"
    assert decisions.at["key-1", "manual_corrected_behavior"] == "move"
    assert decisions.at["key-2", "manual_review_decision"] == "accept"
    assert result["audit"]["composite_reviewed_keys"] == 2
    assert result["audit"]["layer_audits"][1]["overlap_with_prior_layers"] == 1


def test_compose_layers_rejects_input_label_drift() -> None:
    with pytest.raises(
        CompositeReviewContractError,
        match="layer_input_behavior_mismatch",
    ):
        compose_behavior_review_layers(
            _source(),
            [
                _layer(
                    "primary",
                    "key-1",
                    "social-nose",
                    "corrected",
                    "fight",
                ),
                _layer("v3", "key-1", "social-nose", "accept", "social-nose"),
            ],
        )


def test_compose_layers_records_later_reversion_to_source() -> None:
    result = compose_behavior_review_layers(
        _source(),
        [
            _layer("primary", "key-1", "social-nose", "corrected", "fight"),
            _layer("v3", "key-1", "fight", "corrected", "social-nose"),
        ],
    )

    decision = result["decisions"].iloc[0]
    assert decision["manual_review_decision"] == "accept"
    assert decision["manual_corrected_behavior"] == ""
    assert result["audit"]["ever_changed_from_source"] == 1
    assert result["audit"]["reverted_to_source_by_later_review"] == 1
