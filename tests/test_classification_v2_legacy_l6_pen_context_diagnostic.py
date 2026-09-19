from __future__ import annotations

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.evaluation.legacy_development_l6_pen_context_diagnostic import (
    MODES,
    build_boundary_diagnostics,
    build_per_class_cluster_bootstrap,
    build_per_class_diagnostic,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS


def _predictions() -> dict[str, pd.DataFrame]:
    rows = 245
    true = np.asarray(
        [VALID_BEHAVIORS[index % len(VALID_BEHAVIORS)] for index in range(rows)]
    )
    zero = true.copy()
    zero[np.flatnonzero(true == "move")[:3]] = "explore"
    zero[np.flatnonzero(true == "lying")[:2]] = "sitting"
    availability = zero.copy()
    pen = zero.copy()
    pen[np.flatnonzero(true == "move")[:2]] = "move"
    pen[np.flatnonzero(true == "lying")[:4]] = "sitting"
    outputs: dict[str, pd.DataFrame] = {}
    for mode, predicted in zip(
        MODES,
        (zero, availability, pen),
        strict=True,
    ):
        frame = pd.DataFrame(
            {
                "temporal_unit_key": [f"unit-{index:03d}" for index in range(rows)],
                "video_key": [f"video-{index % 33:02d}" for index in range(rows)],
                "behavior_label": true,
                "predicted_label": predicted,
            }
        )
        probabilities = np.full(
            (rows, len(VALID_BEHAVIORS)),
            0.3 / (len(VALID_BEHAVIORS) - 1),
            dtype=float,
        )
        for index, label in enumerate(predicted):
            probabilities[index, VALID_BEHAVIORS.index(label)] = 0.7
        for index, label in enumerate(VALID_BEHAVIORS):
            frame[f"prob_{label.replace('-', '_')}"] = probabilities[:, index]
        outputs[mode] = frame
    return outputs


def _exposure() -> pd.DataFrame:
    rows = 245
    fractions = np.resize(np.asarray([0.0, 0.25, 0.75]), rows)
    return pd.DataFrame(
        {
            "temporal_unit_key": [f"unit-{index:03d}" for index in range(rows)],
            "unique_frame_count": 15,
            "near_boundary_frame_fraction": fractions,
            "mean_signed_distance_n": 0.1,
            "min_signed_distance_n": 0.05,
            "mean_clearance_box_ratio": 1.0,
            "min_clearance_box_ratio": 0.8,
            "mean_bbox_inside_ratio": 0.99,
            "min_bbox_inside_ratio": 0.95,
            "unique_motion_pair_count": 14,
            "mean_approach_speed": 0.01,
            "max_approach_speed": 0.02,
            "mean_retreat_speed": 0.01,
            "max_retreat_speed": 0.02,
            "mean_parallel_speed": 0.01,
            "max_parallel_speed": 0.02,
            "boundary_stratum": np.select(
                [fractions == 0.0, fractions < 0.5],
                ["interior_only", "intermittent_boundary"],
                default="persistent_boundary",
            ),
        }
    )


def test_pen_utility_per_class_reports_gain_harm_and_nll() -> None:
    result = build_per_class_diagnostic(_predictions()).set_index(
        "behavior_label"
    )

    assert len(result) == 10
    assert result.loc["move", "pen_minus_zero_recall"] > 0.0
    assert result.loc["lying", "pen_minus_zero_recall"] < 0.0
    assert np.isfinite(result.filter(like="true_nll").to_numpy()).all()


def test_pen_utility_cluster_bootstrap_is_native_and_video_grouped() -> None:
    result = build_per_class_cluster_bootstrap(
        _predictions(),
        iterations=2000,
        seed=20260717,
    )

    assert result["video_clusters"] == 33
    assert result["iterations"] == 2000
    assert result["classes"]["move"]["pen_minus_zero_f1"][
        "valid_iterations"
    ] == 2000


def test_pen_boundary_diagnostic_preserves_native_universe() -> None:
    boundary, classes, native = build_boundary_diagnostics(
        _predictions(),
        _exposure(),
    )

    assert len(native) == 245
    assert native["temporal_unit_key"].nunique() == 245
    assert len(boundary) == 9
    assert len(classes) == 30
    assert set(boundary["boundary_stratum"]) == {
        "interior_only",
        "intermittent_boundary",
        "persistent_boundary",
    }
