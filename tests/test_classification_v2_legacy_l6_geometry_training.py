from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder import (
    TemporalLadderSelection,
)
from pig_behavior.classification_v2.training.legacy_development_l6_geometry import (
    EXPECTED_PARAMETER_COUNT,
    FEATURE_DIM,
    GEOMETRY_DIM,
    MODEL_INPUT_DIM,
    LegacyL6GeometryClassifier,
    _validate_config_payload,
    build_confusion_group_report,
    build_geometry_view,
    fit_geometry_normalization,
)
from pig_behavior.classification_v2.training.legacy_development_l6_geometry_cache import (
    GEOMETRY_FEATURE_NAMES,
    LINEAGE_SCOPE,
    SEQUENCE_LENGTH,
)


@dataclass
class _FakeBase:
    windows: pd.DataFrame
    sequences: np.ndarray
    observed_mask: np.ndarray
    time_delta: np.ndarray
    targets: np.ndarray
    sample_weights: np.ndarray

    def load_sequences(self, positions: np.ndarray) -> np.ndarray:
        return self.sequences[np.asarray(positions, dtype=np.int64)].copy()


@dataclass
class _FakeCache:
    geometry: np.ndarray
    availability: np.ndarray
    window_index: pd.DataFrame
    slot_index: pd.DataFrame
    audit: dict[str, Any]

    def load_geometry(self, rows: np.ndarray | None = None) -> np.ndarray:
        if rows is None:
            return self.geometry.copy()
        return self.geometry[np.asarray(rows, dtype=np.int64)].copy()

    def load_availability(self, rows: np.ndarray | None = None) -> np.ndarray:
        if rows is None:
            return self.availability.copy()
        return self.availability[np.asarray(rows, dtype=np.int64)].copy()


def _synthetic_inputs() -> tuple[
    _FakeBase,
    _FakeCache,
    TemporalLadderSelection,
]:
    windows = pd.DataFrame(
        {
            "window_id": ["w0", "w1", "w2"],
            "temporal_unit_key": ["u0", "u1", "u2"],
            "l5_role": ["train", "train", "validation"],
            "source_type": ["legacy_recovered"] * 3,
            "dataset_id": ["legacy_recovered_16f"] * 3,
        }
    )
    sequences = np.zeros(
        (3, SEQUENCE_LENGTH, FEATURE_DIM),
        dtype=np.float32,
    )
    observed = np.ones((3, SEQUENCE_LENGTH), dtype=np.bool_)
    base = _FakeBase(
        windows=windows,
        sequences=sequences,
        observed_mask=observed,
        time_delta=np.tile(
            np.arange(SEQUENCE_LENGTH, dtype=np.float32),
            (3, 1),
        ),
        targets=np.asarray([0, 1, 2], dtype=np.int64),
        sample_weights=np.full(3, 0.25, dtype=np.float32),
    )
    frame_numbers = np.asarray(
        [
            [0, 1, 2, 3, 4, 5],
            [3, 4, 5, 6, 7, 8],
            [100, 101, 102, 103, 104, 105],
        ],
        dtype=np.float64,
    )
    geometry = np.stack(
        [frame_numbers + offset for offset in range(GEOMETRY_DIM)],
        axis=2,
    ).astype(np.float32)
    slot_rows: list[dict[str, Any]] = []
    for cache_row, values in enumerate(frame_numbers.astype(int)):
        for slot_index, value in enumerate(values):
            slot_rows.append(
                {
                    "cache_row": cache_row,
                    "slot_index": slot_index,
                    "frame_uid": f"frame-{value}",
                }
            )
    cache = _FakeCache(
        geometry=geometry,
        availability=np.ones((3, SEQUENCE_LENGTH), dtype=np.bool_),
        window_index=windows.assign(
            cache_row=np.arange(3, dtype=np.int64),
            lineage_scope=LINEAGE_SCOPE,
            human_review_complete=False,
        ),
        slot_index=pd.DataFrame.from_records(slot_rows),
        audit={
            "manifest_sha256": "a" * 64,
            "outer_holdout_slots_materialized": 0,
            "source_media_reads": 0,
        },
    )
    selection = TemporalLadderSelection(
        manifest=pd.DataFrame({"window_id": ["w0", "w1", "w2"]}),
        train_positions=np.asarray([0, 1], dtype=np.int64),
        validation_positions=np.asarray([2], dtype=np.int64),
        audit={"selection_content_sha256": "b" * 64},
    )
    return base, cache, selection


def test_geometry_normalization_uses_unique_train_frames_only() -> None:
    _, cache, selection = _synthetic_inputs()

    state = fit_geometry_normalization(cache, selection)

    assert state.feature_names == GEOMETRY_FEATURE_NAMES
    assert state.train_window_rows == 2
    assert state.train_slot_exposures == 12
    assert state.unique_train_frame_rows == 9
    assert state.duplicate_train_slot_exposures == 3
    assert state.validation_rows_read_for_fit == 0
    assert state.outer_holdout_rows_read_for_fit == 0
    assert len(state.state_sha256) == 64


@pytest.mark.parametrize(
    "path",
    [
        Path(
            "configs/classification_v2/"
            "legacy_development_l6_geometry_short_v1.json"
        ),
        Path(
            "configs/classification_v2/"
            "legacy_development_l6_geometry_short_v2.json"
        ),
    ],
)
def test_short_config_locks_canonical_source_and_claim_boundary(
    path: Path,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))

    _validate_config_payload(payload)

    assert payload["canonical_source_name"] == "legacy_16f"
    changed = copy.deepcopy(payload)
    changed["canonical_source_name"] = "legacy"
    with pytest.raises(ValueError, match="canonical_source_name"):
        _validate_config_payload(changed)
    changed_claim = copy.deepcopy(payload)
    changed_claim["q2_claim_allowed"] = True
    with pytest.raises(ValueError, match="q2_claim_allowed"):
        _validate_config_payload(changed_claim)


def test_geometry_modes_are_fixed_width_and_missing_safe() -> None:
    base, cache, selection = _synthetic_inputs()
    normalization = fit_geometry_normalization(cache, selection)
    positions = np.asarray([0, 1], dtype=np.int64)

    zero = build_geometry_view(
        base,
        cache,
        mode="parameter_matched_zero",
        normalization=normalization,
    ).load_sequences(positions)
    availability = build_geometry_view(
        base,
        cache,
        mode="availability_only",
        normalization=normalization,
    ).load_sequences(positions)
    geometry_view = build_geometry_view(
        base,
        cache,
        mode="geometry",
        normalization=normalization,
    )
    geometry = geometry_view.load_sequences(positions)
    missing = geometry_view.with_missing_modality().load_sequences(positions)

    expected_shape = (2, SEQUENCE_LENGTH, MODEL_INPUT_DIM)
    assert zero.shape == expected_shape
    assert availability.shape == expected_shape
    assert geometry.shape == expected_shape
    assert np.count_nonzero(zero[..., FEATURE_DIM:]) == 0
    assert np.count_nonzero(availability[..., FEATURE_DIM:-1]) == 0
    assert np.all(availability[..., -1] == 1.0)
    assert np.isfinite(geometry).all()
    assert np.count_nonzero(geometry[..., FEATURE_DIM:-1]) > 0
    assert np.count_nonzero(missing[..., FEATURE_DIM:]) == 0


def test_geometry_classifier_is_parameter_matched_and_mask_safe() -> None:
    model = LegacyL6GeometryClassifier(
        temporal_encoder_name="masked_mean",
        hidden_dim=128,
        dropout=0.1,
        transformer_layers=1,
        transformer_heads=4,
    ).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        EXPECTED_PARAMETER_COUNT
    )
    features = torch.zeros((2, SEQUENCE_LENGTH, MODEL_INPUT_DIM))
    observed = torch.ones((2, SEQUENCE_LENGTH))
    observed[:, -1] = 0.0
    features[:, -1] = torch.nan
    time_delta = torch.arange(SEQUENCE_LENGTH).repeat(2, 1).float()

    with torch.inference_mode():
        first = model(features, observed, time_delta=time_delta)
        features[:, -1] = 1_000.0
        second = model(features, observed, time_delta=time_delta)

    assert first.shape == (2, len(VALID_BEHAVIORS))
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_confusion_groups_are_predeclared_and_complete() -> None:
    per_class = pd.DataFrame(
        {
            "behavior_label": list(VALID_BEHAVIORS),
            "support": [2] * len(VALID_BEHAVIORS),
            "true_positive": [2] * len(VALID_BEHAVIORS),
            "f1": [1.0] * len(VALID_BEHAVIORS),
        }
    )
    confusion = pd.DataFrame(
        np.eye(len(VALID_BEHAVIORS), dtype=np.int64) * 2,
        index=VALID_BEHAVIORS,
        columns=VALID_BEHAVIORS,
    ).reset_index(names="true_behavior")

    report = build_confusion_group_report(
        per_class,
        confusion,
        mode="geometry",
        missing_modality=False,
    )

    assert set(report["confusion_group"]) == {
        "rare",
        "interaction",
        "feeding",
        "posture",
        "locomotion_exploration",
    }
    assert report["accuracy"].eq(1.0).all()
    assert report["macro_f1"].eq(1.0).all()
    assert report["predicted_inside_group_rate"].eq(1.0).all()
