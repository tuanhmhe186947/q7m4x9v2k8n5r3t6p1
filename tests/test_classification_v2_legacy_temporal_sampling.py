from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.training.legacy_development_l5_cached_data import (
    FEATURE_DIM,
    LegacyL5CachedFeatureView,
)
from pig_behavior.classification_v2.training.legacy_development_temporal_sampling import (
    VIEW_SPECS,
    derive_temporal_sampling_view,
)


def _base_view(tmp_path: Path) -> LegacyL5CachedFeatureView:
    feature_path = tmp_path / "features.npy"
    np.save(
        feature_path,
        np.arange(48 * FEATURE_DIM, dtype=np.float32).reshape(48, FEATURE_DIM),
    )
    windows = pd.DataFrame(
        {
            "window_id": ["window-0", "window-1", "window-2"],
            "temporal_unit_key": ["unit-0", "unit-1", "unit-2"],
            "recording_group_id": ["recording-0"] * 3,
            "video_key": ["video-0", "video-1", "video-2"],
            "source_type": ["legacy_recovered"] * 3,
            "dataset_id": ["legacy_recovered_16f"] * 3,
            "behavior_label": ["lying", "sitting", "move"],
            "oof_fold_id": ["fold-0"] * 3,
            "l5_role": ["train", "train", "validation"],
        }
    )
    feature_rows = np.arange(48, dtype=np.int64).reshape(3, 16)
    time_delta = np.full((3, 16), 0.2, dtype=np.float32)
    time_delta[:, 0] = 0.0
    return LegacyL5CachedFeatureView(
        feature_tensor_path=feature_path,
        feature_tensor_sha256="0" * 64,
        control_id="V1",
        temporal_view_name="legacy_t16_centered_matched_observed_time",
        sequence_length=16,
        windows=windows,
        fold_manifest=pd.DataFrame(),
        feature_rows=feature_rows,
        observed_mask=np.ones((3, 16), dtype=np.bool_),
        time_delta=time_delta,
        targets=np.asarray([5, 8, 7], dtype=np.int64),
        sample_weights=np.ones(3, dtype=np.float64),
        audit={"valid": True},
    )


@pytest.mark.parametrize(
    ("view_id", "offsets"),
    [
        ("c6_contiguous_centered", [5, 6, 7, 8, 9, 10]),
        ("c8_contiguous_centered", [4, 5, 6, 7, 8, 9, 10, 11]),
        ("s6_uniform_span16", [0, 3, 6, 9, 12, 15]),
    ],
)
def test_derives_exact_native_offsets_without_row_loss(
    tmp_path: Path,
    view_id: str,
    offsets: list[int],
) -> None:
    base = _base_view(tmp_path)

    result = derive_temporal_sampling_view(base, view_id)

    assert result.audit["native_frame_offsets"] == offsets
    assert len(result.view.windows) == len(base.windows)
    assert result.audit["rows_dropped"] == 0
    assert result.audit["labels_changed"] == 0
    assert result.view.feature_rows.tolist() == base.feature_rows[:, offsets].tolist()
    assert result.view.observed_mask.all()
    assert result.slot_manifest["native_frame_offset"].tolist() == offsets * 3
    assert not result.slot_manifest[["window_id", "slot_index"]].duplicated().any()


def test_uniform_span16_recomputes_real_elapsed_deltas(tmp_path: Path) -> None:
    result = derive_temporal_sampling_view(
        _base_view(tmp_path),
        "s6_uniform_span16",
    )

    expected = np.asarray([0.0, 0.6, 0.6, 0.6, 0.6, 0.6], dtype=np.float32)
    assert np.allclose(result.view.time_delta, expected[None, :])
    assert result.audit["temporal_span_frames"] == 16
    assert result.view.sequence_length == 6


def test_contiguous_views_keep_consecutive_elapsed_deltas(tmp_path: Path) -> None:
    for view_id in ("c6_contiguous_centered", "c8_contiguous_centered"):
        result = derive_temporal_sampling_view(_base_view(tmp_path), view_id)
        assert np.allclose(result.view.time_delta[:, 0], 0.0)
        assert np.allclose(result.view.time_delta[:, 1:], 0.2)


def test_rejects_duplicate_native_units(tmp_path: Path) -> None:
    base = _base_view(tmp_path)
    base.windows.loc[1, "temporal_unit_key"] = "unit-0"

    with pytest.raises(ValueError, match="one row per native unit"):
        derive_temporal_sampling_view(base, "s6_uniform_span16")


def test_contract_excludes_contiguous_t16_candidate() -> None:
    assert set(VIEW_SPECS) == {
        "c6_contiguous_centered",
        "c8_contiguous_centered",
        "s6_uniform_span16",
    }
    assert all(spec["sequence_length"] in {6, 8} for spec in VIEW_SPECS.values())
