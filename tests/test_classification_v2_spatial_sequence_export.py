from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pig_behavior.classification_v2.features.motion_schema import (
    MOTION_FEATURE_NAMES,
    MOTION_SCHEMA_DIMENSION,
    MOTION_SCHEMA_HASH,
    MOTION_SCHEMA_ID,
    MOTION_SCHEMA_VERSION,
    MotionSchemaError,
)
from pig_behavior.classification_v2.features.spatial_schema import (
    SPATIAL_PREDICTIVE_FEATURES,
    SPATIAL_PREDICTIVE_GROUP_NAMES,
    SPATIAL_SCHEMA_HASH,
    SPATIAL_SCHEMA_TOTAL_DIMENSION,
    SpatialSchemaError,
    canonical_spatial_feature_groups,
    load_current_spatial_tensor_bundle,
    spatial_schema_metadata,
)
from pig_behavior.classification_v2.features.spatiotemporal import (
    _add_temporal_deltas,
)
from pig_behavior.classification_v2.spatial_sequence_export import (
    LEGACY_SPATIAL_FRAME_FEATURES,
    export_spatial_sequences,
)


def _windows(*, start: int = 0, end: int = 1) -> pd.DataFrame:
    indices = list(range(start, end + 1))
    offsets = [value - start for value in indices]
    window_id = f"track-a|win={len(indices)}|{start}-{end}"
    return pd.DataFrame(
        {
            "window_id": [window_id],
            "object_track_key": ["track-a"],
            "window_start_frame": [start],
            "window_end_frame": [end],
            "window_length_frames": [len(indices)],
            "feature_computation_grain": ["FINAL_VIEW_FEATURES"],
            "pair_scope_key": [window_id],
            "view_type": [f"T{len(indices)}_contiguous"],
            "sampling_pattern": ["contiguous"],
            "selected_frame_offsets": [str(offsets).replace(" ", "")],
            "selected_frame_indices": [str(indices).replace(" ", "")],
            "selected_timestamps_seconds": [
                str([value / 30.0 for value in indices]).replace(" ", "")
            ],
            "pair_delta_frames": [
                str([1] * max(0, len(indices) - 1)).replace(" ", "")
            ],
            "pair_delta_seconds": [
                str([1.0 / 30.0] * max(0, len(indices) - 1)).replace(
                    " ",
                    "",
                )
            ],
            "pair_recomputed_for_view": [True],
            "aggregate_recomputed_for_view": [True],
        }
    )


def _frames() -> pd.DataFrame:
    return _with_motion_contract(pd.DataFrame(
        {
            "object_track_key": ["track-a", "track-a"],
            "frame_index": [0, 1],
            "timestamp_sec": [0.0, 1.0 / 30.0],
            "cx_n": [0.25, 0.30],
            "cy_n": [0.40, 0.42],
            "bbox_valid": [True, True],
            "nearest_partner_key": ["", ""],
        }
    ))


def _with_motion_contract(frames: pd.DataFrame) -> pd.DataFrame:
    out = frames.copy()
    count = len(out)
    out["source_type"] = out.get("source_type", "cvat_tracking_xml")
    out["dataset_id"] = out.get("dataset_id", "fixture")
    out["video_key"] = out.get("video_key", "video-a")
    out["temporal_unit_key"] = out.get(
        "temporal_unit_key",
        out["object_track_key"].astype(str) + "|unit",
    )
    out["bw_n"] = out.get("bw_n", pd.Series([0.2] * count))
    out["bh_n"] = out.get("bh_n", pd.Series([0.1] * count))
    out["area_n"] = out.get("area_n", out["bw_n"] * out["bh_n"])
    out["aspect_ratio"] = out.get(
        "aspect_ratio",
        out["bw_n"] / out["bh_n"],
    )
    out["box_diag_n"] = out.get(
        "box_diag_n",
        np.hypot(out["bw_n"], out["bh_n"]),
    )
    out["bbox_valid"] = out.get("bbox_valid", True)
    out = _add_temporal_deltas(out)
    available = out.groupby("temporal_unit_key")[
        "velocity_valid"
    ].transform("any")
    out["motion_feature_available"] = available
    out["motion_schema_id"] = MOTION_SCHEMA_ID
    out["motion_schema_version"] = MOTION_SCHEMA_VERSION
    out["motion_schema_dimension"] = MOTION_SCHEMA_DIMENSION
    out["motion_schema_feature_names"] = json.dumps(
        list(MOTION_FEATURE_NAMES),
        separators=(",", ":"),
    )
    out["motion_schema_hash"] = MOTION_SCHEMA_HASH
    additions: dict[str, object] = {}
    for group in ("roi_class_relation", "social_relation"):
        for feature_name in SPATIAL_PREDICTIVE_FEATURES[group]:
            if feature_name not in out:
                additions[feature_name] = 0.0
    for roi_class in ("feeder", "drinker", "toy"):
        availability = f"roi_{roi_class}_available"
        if availability not in out:
            additions[availability] = False
    if "nearest_partner_key" not in out:
        additions["nearest_partner_key"] = ""
    if "social_neighbor_available" not in out:
        partner = out.get(
            "nearest_partner_key",
            pd.Series([""] * count, index=out.index),
        )
        additions["social_neighbor_available"] = (
            partner.fillna("").astype(str).str.strip().ne("")
        )
    return pd.concat(
        [out, pd.DataFrame(additions, index=out.index)],
        axis=1,
    ).copy()


def _write_current_bundle(
    root: Path,
    *,
    audit_override: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    result = export_spatial_sequences(_windows(), _frames())
    npz_path = root / "X_spatial_sequences.npz"
    audit_path = root / "spatial_sequence_audit.json"
    np.savez_compressed(npz_path, **result.arrays)
    audit = dict(result.audit)
    if audit_override is not None:
        audit.update(audit_override)
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return npz_path, audit_path


def test_spatial_export_audits_complete_alignment_without_row_loss() -> None:
    result = export_spatial_sequences(_windows(), _frames())

    assert result.audit["input_window_rows"] == 1
    assert result.audit["aligned_window_rows"] == 1
    assert result.audit["input_frame_rows"] == 2
    assert result.audit["aligned_frame_rows"] == 2
    assert result.audit["invalid_frame_alignment_rows"] == 0
    assert result.audit["duplicate_frame_alignment_rows"] == 0
    assert result.audit["observed_frame_slots"] == 2


@pytest.mark.parametrize("invalid_frame_index", [None, "bad", 0.5])
def test_spatial_export_rejects_invalid_frame_alignment_rows(
    invalid_frame_index: object,
) -> None:
    frames = _frames()
    frames["frame_index"] = frames["frame_index"].astype(object)
    frames.loc[1, "frame_index"] = invalid_frame_index

    with pytest.raises(
        ValueError,
        match=r"Frame alignment contract failed: invalid_rows=1",
    ):
        export_spatial_sequences(_windows(), frames)


def test_spatial_export_rejects_missing_track_key_instead_of_dropping_row() -> None:
    frames = _frames()
    frames.loc[1, "object_track_key"] = pd.NA

    with pytest.raises(
        ValueError,
        match=r"Frame alignment contract failed: invalid_rows=1",
    ):
        export_spatial_sequences(_windows(), frames)


def test_spatial_export_rejects_duplicate_track_frame_alignment() -> None:
    frames = _frames()
    frames.loc[1, "frame_index"] = 0

    with pytest.raises(
        ValueError,
        match=r"duplicate_frame_alignment_rows=2",
    ):
        export_spatial_sequences(_windows(), frames)


def test_spatial_export_rejects_inconsistent_window_span() -> None:
    windows = _windows()
    windows.loc[0, "window_length_frames"] = 3

    with pytest.raises(
        ValueError,
        match=r"Final-view selected-slot count mismatch",
    ):
        export_spatial_sequences(windows, _frames())


def test_spatial_motion_is_rebased_inside_each_window() -> None:
    frames = _with_motion_contract(pd.DataFrame(
        {
            "object_track_key": ["track-a"] * 3,
            "frame_index": [0, 1, 2],
            "timestamp_sec": [0.0, 1.0 / 30.0, 2.0 / 30.0],
            "cx_n": [0.0, 0.5, 0.6],
            "cy_n": [0.0, 0.0, 0.0],
            "bw_n": [0.2, 0.2, 0.2],
            "bh_n": [0.1, 0.1, 0.1],
            "area_n": [0.02, 0.02, 0.02],
            "aspect_ratio": [2.0, 2.0, 2.0],
            "vx_n_per_second": [0.0, 15.0, 99.0],
            "vy_n_per_second": [0.0, 0.0, 99.0],
            "speed_n_per_second": [0.0, 15.0, 99.0],
            "abs_tangential_acceleration_n_per_second2": [
                0.0,
                450.0,
                99.0,
            ],
            "abs_direction_change_rad": [0.0, 1.0, 1.0],
            "bbox_valid": [True, True, True],
        }
    ))
    windows = _windows(start=1, end=2)

    result = export_spatial_sequences(windows, frames)
    names = result.feature_names["motion_delta"]
    motion = result.arrays["motion_delta"][0]

    assert motion[0, names.index("vx_n_per_second")] == 0.0
    assert motion[0, names.index("speed_n_per_second")] == 0.0
    assert motion[1, names.index("vx_n_per_second")] == pytest.approx(3.0)
    assert motion[1, names.index("speed_n_per_second")] == pytest.approx(3.0)
    assert (
        motion[
            1,
            names.index("tangential_acceleration_n_per_second2"),
        ]
        == 0.0
    )
    assert motion[1, names.index("direction_change_rad")] == 0.0
    assert result.arrays["vector_acceleration_valid_mask"][0, 1] == 0.0
    assert result.audit["motion_rebased_windows"] == 1


def test_spatial_social_motion_is_rebased_inside_each_window() -> None:
    frames = _with_motion_contract(pd.DataFrame(
        {
            "object_track_key": ["track-a"] * 3,
            "frame_index": [0, 1, 2],
            "timestamp_sec": [0.0, 1.0 / 30.0, 2.0 / 30.0],
            "cx_n": [0.0, 0.5, 0.6],
            "cy_n": [0.0, 0.0, 0.0],
            "bw_n": [0.2, 0.2, 0.2],
            "bh_n": [0.1, 0.1, 0.1],
            "nearest_pig_id": ["ID_2"] * 3,
            "nearest_partner_key": ["track-b"] * 3,
            "roi_feeder_available": [True, True, True],
            "roi_drinker_available": [False, False, False],
            "roi_toy_available": [True, True, True],
            "nearest_dist_n": [0.5, 0.2, 0.1],
            "partner_distance_delta_n": [0.0, -0.3, -99.0],
            "approach_speed_n_per_second": [0.0, 9.0, 99.0],
            "retreat_speed_n_per_second": [0.0, 0.0, 99.0],
            "pair_contact_with_nearest": [True, True, True],
            "social_density_near_count": [0.0, 0.0, 0.0],
            "aggression_score_proxy_per_second": [0.0, 99.0, 99.0],
            "speed_n_per_second": [0.0, 15.0, 99.0],
            "bbox_valid": [True, True, True],
        }
    ))
    windows = _windows(start=1, end=2)

    result = export_spatial_sequences(windows, frames)
    names = result.feature_names["social_relation"]
    social = result.arrays["social_relation"][0]
    assert social[0, names.index("partner_distance_delta_n")] == 0.0
    assert social[0, names.index("approach_speed_n_per_second")] == 0.0
    assert social[0, names.index("aggression_score_proxy_per_second")] == 0.0
    assert social[1, names.index("partner_distance_delta_n")] == pytest.approx(
        -0.1
    )
    assert social[1, names.index("approach_speed_n_per_second")] == pytest.approx(
        3.0
    )
    assert social[1, names.index("aggression_score_proxy_per_second")] == (
        pytest.approx(6.0)
    )
    assert result.arrays["roi_validity_mask"][0, 0].tolist() == [1.0, 0.0, 1.0]
    assert result.arrays["social_validity_mask"][0, 0] == 1.0
    selected = {
        feature
        for group_features in result.feature_names.values()
        for feature in group_features
    }
    assert "nearest_pig_id" not in selected
    assert result.audit["social_rebased_windows"] == 1


def test_current_social_export_does_not_fallback_to_legacy_pig_id() -> None:
    frames = _frames().drop(columns=["nearest_partner_key"])
    frames["nearest_pig_id"] = "ID_2"
    frames["nearest_dist_n"] = 0.25

    with pytest.raises(
        ValueError,
        match="Missing canonical social identity column",
    ):
        export_spatial_sequences(_windows(), frames)


def test_sparse_s6_at16_uses_exact_selected_frames_and_sparse_pairs() -> None:
    indices = [0, 3, 6, 9, 12, 15]
    frames = _with_motion_contract(pd.DataFrame(
        {
            "object_track_key": ["track-a"] * 16,
            "frame_index": list(range(16)),
            "timestamp_sec": [value / 30.0 for value in range(16)],
            "cx_n": [value / 30.0 for value in range(16)],
            "cy_n": [0.0] * 16,
            "bw_n": [0.2] * 16,
            "bh_n": [0.1] * 16,
            "area_n": [0.02] * 16,
            "aspect_ratio": [2.0] * 16,
            "speed_n_per_second": [999.0] * 16,
            "bbox_valid": [True] * 16,
        }
    ))
    window_id = "track-a|view=S6@16|0-15"
    windows = pd.DataFrame(
        {
            "window_id": [window_id],
            "object_track_key": ["track-a"],
            "window_start_frame": [0],
            "window_end_frame": [15],
            "window_length_frames": [6],
            "feature_computation_grain": ["FINAL_VIEW_FEATURES"],
            "pair_scope_key": [window_id],
            "view_type": ["S6@16"],
            "sampling_pattern": [
                "uniform_sparse_offsets_0_3_6_9_12_15"
            ],
            "selected_frame_offsets": ["[0,3,6,9,12,15]"],
            "selected_frame_indices": ["[0,3,6,9,12,15]"],
            "selected_timestamps_seconds": [
                "[0.0,0.1,0.2,0.3,0.4,0.5]"
            ],
            "pair_delta_frames": ["[3,3,3,3,3]"],
            "pair_delta_seconds": ["[0.1,0.1,0.1,0.1,0.1]"],
            "pair_recomputed_for_view": [True],
            "aggregate_recomputed_for_view": [True],
        }
    )

    result = export_spatial_sequences(windows, frames)
    names = result.feature_names["motion_delta"]
    speed = result.arrays["motion_delta"][0, :, names.index(
        "speed_n_per_second"
    )]

    assert result.arrays["frame_index_sequence"][0].tolist() == indices
    assert result.arrays["adjacent_motion_pair_mask"][0].tolist() == [
        0.0,
    ] * 6
    assert result.arrays["sparse_velocity_pair_mask"][0].tolist() == [
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert speed.tolist() == pytest.approx([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])


@pytest.mark.parametrize("group", SPATIAL_PREDICTIVE_GROUP_NAMES)
@pytest.mark.parametrize("position", ("first", "middle", "last"))
def test_all_groups_reject_missing_required_source_columns(
    group: str,
    position: str,
) -> None:
    frames = _frames()
    names = SPATIAL_PREDICTIVE_FEATURES[group]
    index = {
        "first": 0,
        "middle": len(names) // 2,
        "last": len(names) - 1,
    }[position]

    error_type = (
        MotionSchemaError if group == "motion_delta" else SpatialSchemaError
    )
    error_pattern = (
        "missing_required_motion_features"
        if group == "motion_delta"
        else "missing_required_source_columns"
    )
    with pytest.raises(error_type, match=error_pattern):
        export_spatial_sequences(
            _windows(),
            frames.drop(columns=[names[index]]),
        )


@pytest.mark.parametrize("group", SPATIAL_PREDICTIVE_GROUP_NAMES)
@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("swap", "ordered_names_mismatch"),
        ("duplicate", "duplicate_names"),
        ("unknown", "unexpected_names"),
        ("blank", "blank_names"),
        ("null", "null_or_non_string_names"),
        ("case", "unexpected_names"),
        ("whitespace", "whitespace_normalized_names"),
    ),
)
def test_all_groups_reject_noncanonical_declarations(
    group: str,
    mutation: str,
    error: str,
) -> None:
    frames = _frames()
    schema = canonical_spatial_feature_groups()
    names: list[object] = list(schema[group])
    if mutation == "swap":
        names[0], names[1] = names[1], names[0]
    elif mutation == "duplicate":
        names[1] = names[0]
    elif mutation == "unknown":
        names.insert(1, "unknown_predictive_feature")
        frames["unknown_predictive_feature"] = 0.0
    elif mutation == "blank":
        names[0] = ""
    elif mutation == "null":
        names[0] = None
    elif mutation == "case":
        names[0] = str(names[0]).upper()
    elif mutation == "whitespace":
        names[0] = f" {names[0]}"
    schema[group] = names  # type: ignore[assignment]

    error_type = (
        MotionSchemaError if group == "motion_delta" else SpatialSchemaError
    )
    motion_error = {
        "swap": "motion_feature_order_mismatch",
        "duplicate": "duplicate_motion_features",
        "unknown": "unexpected_motion_features",
        "blank": "unexpected_motion_features",
        "null": "unexpected_motion_features",
        "case": "unexpected_motion_features",
        "whitespace": "unexpected_motion_features",
    }[mutation]
    with pytest.raises(
        error_type,
        match=motion_error if group == "motion_delta" else error,
    ):
        export_spatial_sequences(
            _windows(),
            frames,
            feature_schema=schema,
        )


@pytest.mark.parametrize("group", SPATIAL_PREDICTIVE_GROUP_NAMES)
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("group_schema_version", "stale.v0"),
        ("group_schema_hash", "0" * 64),
    ),
)
def test_all_groups_reject_stale_group_schema_authority(
    group: str,
    field: str,
    value: str,
) -> None:
    metadata = spatial_schema_metadata()
    group_index = list(SPATIAL_PREDICTIVE_GROUP_NAMES).index(group)
    metadata["groups"][group_index][field] = value

    with pytest.raises(SpatialSchemaError, match="spatial_schema_groups"):
        export_spatial_sequences(
            _windows(),
            _frames(),
            spatial_schema_manifest=metadata,
        )


def test_export_rejects_incompatible_predictive_dtype() -> None:
    frames = _frames()
    frames["cx_n"] = "not-a-number"

    with pytest.raises(ValueError, match="Incompatible predictive source dtype"):
        export_spatial_sequences(_windows(), frames)


def test_export_rejects_declared_dimension_disagreement() -> None:
    metadata = spatial_schema_metadata()
    metadata["group_dimensions"]["social_relation"] = 9

    with pytest.raises(
        SpatialSchemaError,
        match="spatial_schema_group_dimensions_mismatch",
    ):
        export_spatial_sequences(
            _windows(),
            _frames(),
            spatial_schema_manifest=metadata,
        )


def test_legacy_schema_cannot_bypass_current_export() -> None:
    with pytest.raises(
        SpatialSchemaError,
        match="POLICY_CURRENT_ONLY_FAIL_CLOSED",
    ):
        export_spatial_sequences(
            _windows(),
            _frames(),
            feature_schema=LEGACY_SPATIAL_FRAME_FEATURES,
        )


def test_current_export_is_exact_fixed_width_and_masked() -> None:
    result = export_spatial_sequences(_windows(), _frames())

    assert result.feature_names == canonical_spatial_feature_groups()
    assert sum(
        array.shape[-1]
        for name, array in result.arrays.items()
        if name in SPATIAL_PREDICTIVE_GROUP_NAMES
    ) == SPATIAL_SCHEMA_TOTAL_DIMENSION == 46
    assert result.arrays["roi_class_relation"].shape[-1] == 18
    assert result.arrays["social_relation"].shape[-1] == 10
    assert not result.arrays["roi_validity_mask"].any()
    assert not result.arrays["social_validity_mask"].any()
    assert not result.arrays["roi_class_relation"].any()
    assert not result.arrays["social_relation"].any()


def test_loader_rejects_sidecar_order_and_hash_disagreement(
    tmp_path: Path,
) -> None:
    result = export_spatial_sequences(_windows(), _frames())
    feature_names = deepcopy(result.audit["feature_names"])
    feature_names["bbox_xywh_n"][0:2] = reversed(
        feature_names["bbox_xywh_n"][0:2]
    )
    npz_path, audit_path = _write_current_bundle(
        tmp_path,
        audit_override={"feature_names": feature_names},
    )

    with pytest.raises(SpatialSchemaError, match="ordered_names_mismatch"):
        load_current_spatial_tensor_bundle(npz_path, audit_path)


def test_loader_accepts_exact_current_order_and_rejects_stale_hash(
    tmp_path: Path,
) -> None:
    npz_path, audit_path = _write_current_bundle(tmp_path)
    arrays, audit = load_current_spatial_tensor_bundle(
        npz_path,
        audit_path,
    )
    assert audit["spatial_schema"]["schema_hash"] == SPATIAL_SCHEMA_HASH
    assert {
        group: arrays[group].shape[-1]
        for group in SPATIAL_PREDICTIVE_GROUP_NAMES
    } == {
        group: len(SPATIAL_PREDICTIVE_FEATURES[group])
        for group in SPATIAL_PREDICTIVE_GROUP_NAMES
    }

    arrays["bbox_xywh_n"][0, 0, 0] = 0.123
    np.savez_compressed(npz_path, **arrays)
    with pytest.raises(SpatialSchemaError, match="content hash mismatch"):
        load_current_spatial_tensor_bundle(npz_path, audit_path)


def test_bounded_spatial_model_forward_uses_canonical_dimensions() -> None:
    pytest.importorskip("torchvision")
    from pig_behavior.classification_v2.models.spatial_tcn import (
        SpatialTCNClassifier,
        SpatialTCNConfig,
    )

    result = export_spatial_sequences(_windows(), _frames())
    model = SpatialTCNClassifier(
        SpatialTCNConfig(
            input_dims={
                group: len(SPATIAL_PREDICTIVE_FEATURES[group])
                for group in SPATIAL_PREDICTIVE_GROUP_NAMES
            },
            num_classes=10,
            hidden_dim=8,
            dropout=0.0,
        )
    )
    features = {
        group: torch.from_numpy(result.arrays[group])
        for group in SPATIAL_PREDICTIVE_GROUP_NAMES
    }
    logits = model(
        features,
        length_mask=torch.from_numpy(result.arrays["length_mask"]),
        observed_mask=torch.from_numpy(result.arrays["observed_mask"]),
    )
    assert logits.shape == (1, 10)


def test_repeated_current_export_has_identical_declared_content_hash() -> None:
    first = export_spatial_sequences(_windows(), _frames())
    second = export_spatial_sequences(_windows(), _frames())

    assert (
        first.audit["spatial_tensor_content_hash"]
        == second.audit["spatial_tensor_content_hash"]
    )
    for name in first.arrays:
        np.testing.assert_array_equal(first.arrays[name], second.arrays[name])
