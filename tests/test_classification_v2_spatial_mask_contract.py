"""Regression tests for the audited spatial validity-mask contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from pig_behavior.classification_v2.features.motion_schema import (
    MOTION_FEATURE_NAMES,
    MOTION_REQUIRED_MASKS,
)
from pig_behavior.classification_v2.models.spatial_tcn import (
    SpatialTCNClassifier,
    SpatialTCNConfig,
)
from pig_behavior.classification_v2.spatial_sequence_export import (
    _motion_feature_validity_from_values,
    _rebase_window_motion,
    _rebase_window_social_motion,
    _social_feature_validity,
    _view_motion_pair_masks,
)
from pig_behavior.classification_v2.training.fold_preprocessing import (
    fit_fold_preprocessing,
)


def _motion_contract_fixture() -> tuple[list[str], np.ndarray]:
    names = [
        "cx_n",
        "cy_n",
        "bw_n",
        "bh_n",
        "area_n",
        "aspect_ratio",
        *MOTION_FEATURE_NAMES,
        *MOTION_REQUIRED_MASKS,
        "bbox_valid",
        "actor_bbox_valid",
        "geometry_feature_valid",
        "spatiotemporal_feature_valid",
    ]
    values = np.zeros((3, len(names)), dtype=np.float64)
    for row, center in enumerate((0.0, 0.0, 0.2)):
        values[row, names.index("cx_n")] = center
        values[row, names.index("bw_n")] = 0.2
        values[row, names.index("bh_n")] = 0.1
        values[row, names.index("area_n")] = 0.02
        values[row, names.index("aspect_ratio")] = 2.0
        for quality in (
            "bbox_valid",
            "actor_bbox_valid",
            "geometry_feature_valid",
            "spatiotemporal_feature_valid",
        ):
            values[row, names.index(quality)] = 1.0
    return names, values


def test_motion_rebase_masks_first_and_cross_unit_pairs() -> None:
    names, values = _motion_contract_fixture()
    rebased, _ = _rebase_window_motion(
        values,
        names,
        np.array([0, 1, 2], dtype=np.int32),
        timestamps=np.array([0.0, 1.0, 2.0]),
        temporal_unit_keys=np.array(["unit-a", "unit-b", "unit-b"]),
    )
    valid_index = names.index("valid_motion_pair")
    available_index = names.index("motion_feature_available")
    velocity_index = names.index("velocity_valid")
    assert rebased[:, valid_index].tolist() == [0.0, 0.0, 1.0]
    assert rebased[:, available_index].tolist() == [0.0, 0.0, 1.0]
    assert rebased[:, velocity_index].tolist() == [0.0, 0.0, 1.0]
    assert rebased[2, names.index("vx_n_per_second")] == pytest.approx(0.2)
    adjacent, sparse = _view_motion_pair_masks(
        np.array([0, 1, 2]),
        np.array([0.0, 1.0, 2.0]),
        np.array([True, True, True]),
        np.array(["unit-a", "unit-b", "unit-b"]),
    )
    assert adjacent.tolist() == [0.0, 0.0, 1.0]
    assert sparse.tolist() == [0.0, 0.0, 0.0]


def test_stationary_pair_is_valid_zero_motion() -> None:
    names, values = _motion_contract_fixture()
    rebased, _ = _rebase_window_motion(
        values[:2],
        names,
        np.array([0, 1], dtype=np.int32),
        timestamps=np.array([0.0, 1.0]),
        temporal_unit_keys=np.array(["unit-a", "unit-a"]),
    )
    assert rebased[:, names.index("valid_motion_pair")].tolist() == [0.0, 1.0]
    assert rebased[1, names.index("speed_n_per_second")] == 0.0


def test_nonpositive_motion_time_gap_is_invalid() -> None:
    names, values = _motion_contract_fixture()
    rebased, _ = _rebase_window_motion(
        values[:2],
        names,
        np.array([0, 1], dtype=np.int32),
        timestamps=np.array([0.0, 0.0]),
        temporal_unit_keys=np.array(["unit-a", "unit-a"]),
    )
    assert rebased[:, names.index("valid_motion_pair")].tolist() == [0.0, 0.0]
    assert rebased[:, names.index("motion_feature_available")].tolist() == [
        0.0,
        0.0,
    ]


def test_velocity_validity_does_not_imply_acceleration_validity() -> None:
    names, values = _motion_contract_fixture()
    rebased, _ = _rebase_window_motion(
        values[:2],
        names,
        np.array([0, 1], dtype=np.int32),
        timestamps=np.array([0.0, 1.0]),
        temporal_unit_keys=np.array(["unit-a", "unit-a"]),
    )
    validity = _motion_feature_validity_from_values(rebased, names)
    assert validity[1, [0, 1, 6]].tolist() == [1.0, 1.0, 1.0]
    assert validity[1, [8, 9, 10, 11]].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_social_zero_counts_remain_valid_without_neighbor() -> None:
    names = [
        "nearest_dist_n",
        "nearest_pair_iou",
        "nearest_pair_overlap_ratio",
        "social_density_near_count",
        "social_contact_count",
        "partner_distance_delta_n",
        "approach_speed_n_per_second",
        "retreat_speed_n_per_second",
        "pair_contact_with_nearest",
        "aggression_score_proxy_per_second",
        "social_context_valid",
        "social_neighbor_available",
    ]
    values = np.zeros((1, len(names)), dtype=np.float64)
    values[0, names.index("social_context_valid")] = 1.0
    validity = _social_feature_validity(
        values,
        names,
        np.array([""]),
        np.array([0]),
        np.array([0.0]),
        np.array([True]),
        np.array(["unit-a"]),
    )
    assert validity[0, :3].tolist() == [0.0, 0.0, 0.0]
    assert validity[0, 3:5].tolist() == [1.0, 1.0]
    assert validity[0, 8] == 0.0


def test_social_pair_cannot_cross_temporal_unit() -> None:
    names = [
        "nearest_dist_n",
        "cx_n",
        "cy_n",
        "bbox_valid",
        "partner_distance_delta_n",
        "approach_speed_n_per_second",
        "retreat_speed_n_per_second",
    ]
    values = np.zeros((2, len(names)), dtype=np.float64)
    values[:, names.index("bbox_valid")] = 1.0
    values[:, names.index("nearest_dist_n")] = [0.4, 0.2]
    rebased, audit = _rebase_window_social_motion(
        values,
        names,
        np.array([0, 1]),
        np.array(["pig-b", "pig-b"]),
        timestamps=np.array([0.0, 1.0]),
        temporal_unit_keys=np.array(["unit-a", "unit-b"]),
    )
    assert audit["valid_pairs"] == 0
    assert rebased[1, names.index("partner_distance_delta_n")] == 0.0


def test_fold_fit_excludes_invalid_motion_placeholders() -> None:
    values = np.empty((2, 2, 12), dtype=np.float32)
    values[0, 0, :] = 1.0
    values[0, 1, :] = 1000.0
    values[1, 0, :] = 3.0
    values[1, 1, :] = 5.0
    validity = np.ones_like(values, dtype=np.float32)
    validity[0, 1, :] = 0.0
    arrays = {
        "motion_delta": values.astype(np.float32),
        "motion_feature_validity_mask": validity,
        "length_mask": np.ones((2, 2), dtype=np.float32),
        "observed_mask": np.ones((2, 2), dtype=np.float32),
        "spatial_quality_mask": np.ones((2, 2), dtype=np.float32),
    }
    state = fit_fold_preprocessing(
        pd.DataFrame(
            {
                "window_id": ["w0", "w1"],
                "grouped_role": ["train", "train"],
                "eligible": [True, True],
            }
        ),
        arrays,
        {"motion_delta": [f"motion_{index}" for index in range(12)]},
        fold_id="fold-0",
        snapshot_sha256="a" * 64,
        config_sha256="b" * 64,
        spatial_audit_sha256="c" * 64,
        feature_groups=("motion_delta",),
        standardized_groups=("motion_delta",),
    )
    assert state.statistics["motion_delta"]["mean"] == [3.0] * 12
    assert state.statistics["motion_delta"]["finite_value_count"] == [3] * 12


def test_required_feature_validity_mask_fails_closed() -> None:
    model = SpatialTCNClassifier(
        SpatialTCNConfig(input_dims={"motion_delta": 12}, num_classes=3)
    )
    with pytest.raises(ValueError, match="feature validity masks"):
        model(
            {"motion_delta": torch.zeros((1, 2, 12))},
            length_mask=torch.ones((1, 2)),
        )


def test_spatial_tcn_mask_changes_output_without_changing_placeholder() -> None:
    torch.manual_seed(7)
    model = SpatialTCNClassifier(
        SpatialTCNConfig(input_dims={"motion_delta": 12}, num_classes=3)
    ).eval()
    values = torch.zeros((1, 3, 12), dtype=torch.float32)
    common = torch.ones((1, 3), dtype=torch.float32)
    observed = torch.ones((1, 3), dtype=torch.float32)
    valid = torch.ones((1, 3, 12), dtype=torch.float32)
    invalid_first = valid.clone()
    invalid_first[:, 0, :] = 0.0
    with torch.no_grad():
        valid_logits = model(
            {"motion_delta": values},
            length_mask=common,
            observed_mask=observed,
            feature_validity_masks={"motion_delta": valid},
        )
        invalid_logits = model(
            {"motion_delta": values},
            length_mask=common,
            observed_mask=observed,
            feature_validity_masks={"motion_delta": invalid_first},
        )
    assert not torch.allclose(valid_logits, invalid_logits)


def test_motion_invalidity_remains_distinct_from_padding() -> None:
    torch.manual_seed(11)
    model = SpatialTCNClassifier(
        SpatialTCNConfig(input_dims={"motion_delta": 12}, num_classes=3)
    ).eval()
    values = torch.zeros((1, 2, 12), dtype=torch.float32)
    observed = torch.ones((1, 2), dtype=torch.float32)
    validity = torch.ones((1, 2, 12), dtype=torch.float32)
    validity[:, 0, :] = 0.0
    with torch.no_grad():
        invalid_logits = model(
            {"motion_delta": values},
            length_mask=torch.ones((1, 2)),
            observed_mask=observed,
            feature_validity_masks={"motion_delta": validity},
        )
        padding_logits = model(
            {"motion_delta": values},
            length_mask=torch.tensor([[0.0, 1.0]]),
            observed_mask=observed,
            feature_validity_masks={"motion_delta": validity},
        )
    assert not torch.allclose(invalid_logits, padding_logits)
