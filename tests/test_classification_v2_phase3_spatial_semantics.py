from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.features.motion_schema import (
    MOTION_SCHEMA_HASH,
)
from pig_behavior.classification_v2.features.native_evidence_contract import (
    check_native_review_evidence,
)
from pig_behavior.classification_v2.features.social import (
    build_static_social_context_features,
)
from pig_behavior.classification_v2.features.spatial_semantics import (
    AXIS_DISTANCE_METRIC_ID,
    AXIS_DISTANCE_METRIC_VERSION,
    DIAGONAL_DISTANCE_METRIC_ID,
    DIAGONAL_DISTANCE_METRIC_VERSION,
    ROI_AGGREGATION_VERSION,
    SOCIAL_IDENTITY_VERSION,
    SOCIAL_TIE_BREAK_RULE,
    SOCIAL_TIE_BREAK_VERSION,
    axis_normalized_image_distance,
    diagonal_normalized_image_distance,
    is_target_roi_model_forbidden,
)
from pig_behavior.classification_v2.features.spatiotemporal import (
    EnhancedFeatureConfig,
    _add_social_context_columns,
    _add_temporal_deltas,
    _add_temporal_unit_aggregates,
    audit_enhanced_spatiotemporal_features,
    build_enhanced_spatiotemporal_features,
)
from pig_behavior.classification_v2.spatial_sequence_export import (
    export_spatial_sequences,
)
from pig_behavior.classification_v2.train_ready_features import (
    select_window_feature_columns,
)


def _row(
    *,
    actor: str,
    frame: int = 0,
    cx_n: float = 0.5,
    cy_n: float = 0.5,
    video: str = "video-a",
    unit: str | None = None,
    pig_id: str | None = None,
    bbox_valid: bool = True,
    image_width: float = 1000.0,
    image_height: float = 500.0,
) -> dict[str, object]:
    center_x = cx_n * image_width
    center_y = cy_n * image_height
    width = 20.0
    height = 20.0
    return {
        "source_type": "cvat_tracking_xml",
        "dataset_id": "fixture",
        "video_key": video,
        "scene_frame_uid": f"{video}|frame={frame}",
        "frame_uid": f"{video}|frame={frame}|actor={actor}",
        "frame_index": frame,
        "timestamp_sec": float(frame),
        "object_track_key": f"{video}|track={actor}",
        "temporal_unit_key": unit or f"{video}|unit={actor}",
        "pig_id": actor if pig_id is None else pig_id,
        "track_id": actor,
        "object_id": f"object-{actor}",
        "behavior": "stand",
        "bbox_valid": bbox_valid,
        "x1": center_x - width / 2.0,
        "y1": center_y - height / 2.0,
        "x2": center_x + width / 2.0,
        "y2": center_y + height / 2.0,
        "bbox_area": width * height,
        "image_width": image_width,
        "image_height": image_height,
        "cx_n": cx_n,
        "cy_n": cy_n,
        "bw_n": width / image_width if image_width > 0 else np.nan,
        "bh_n": height / image_height if image_height > 0 else np.nan,
        "area_n": (
            width * height / (image_width * image_height)
            if image_width > 0 and image_height > 0
            else np.nan
        ),
        "aspect_ratio": 1.0,
        "box_diag_n": math.hypot(
            width / image_width if image_width > 0 else np.nan,
            height / image_height if image_height > 0 else np.nan,
        ),
        "observed_mask": True,
    }


def _social_rows(order: tuple[str, ...] = ("A", "B", "C")) -> pd.DataFrame:
    positions = {"A": 0.5, "B": 0.4, "C": 0.6}
    return pd.DataFrame(
        [_row(actor=actor, cx_n=positions[actor]) for actor in order]
    )


def _social_actor(
    output: pd.DataFrame,
    actor: str,
) -> pd.Series:
    key = f"video-a|track={actor}"
    return output.loc[output["object_track_key"].eq(key)].iloc[0]


def _add_pair_support(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["delta_frame_prev"] = (
        out.groupby(
            [
                "source_type",
                "dataset_id",
                "video_key",
                "object_track_key",
                "temporal_unit_key",
            ]
        )["frame_index"].diff()
    )
    out["motion_delta_seconds"] = (
        out.groupby(
            [
                "source_type",
                "dataset_id",
                "video_key",
                "object_track_key",
                "temporal_unit_key",
            ]
        )["timestamp_sec"].diff()
    )
    out["adjacent_motion_pair_valid"] = out["delta_frame_prev"].eq(1)
    out["motion_velocity_pair_valid"] = (
        out["motion_delta_seconds"].gt(0)
        & out["delta_frame_prev"].gt(0)
    )
    out["speed_n_per_frame"] = 0.0
    out["speed_n_per_second"] = 0.0
    return out


def _roi_unit(
    available: list[bool],
    contact: list[bool],
) -> pd.DataFrame:
    rows = pd.DataFrame(
        [
            _row(actor="A", frame=index)
            for index in range(len(available))
        ]
    )
    temporal = _add_temporal_deltas(rows)
    temporal["roi_target_available"] = available
    temporal["roi_target_contact"] = contact
    temporal["roi_target_near"] = contact
    return _add_temporal_unit_aggregates(temporal)


def _window_and_frame(column: str, value: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows = pd.DataFrame(
        {
            "window_id": ["window-a"],
            "object_track_key": ["track-a"],
            "window_start_frame": [0],
            "window_end_frame": [0],
            "window_length_frames": [1],
        }
    )
    frames = pd.DataFrame(
        {
            "object_track_key": ["track-a"],
            "frame_index": [0],
            column: [value],
        }
    )
    return windows, frames


def test_axis_normalized_metric_formula_non_square() -> None:
    assert axis_normalized_image_distance(100, 0, 1000, 500) == pytest.approx(
        0.1
    )
    assert axis_normalized_image_distance(0, 100, 1000, 500) == pytest.approx(
        0.2
    )


def test_diagonal_metric_is_isotropic_in_pixel_space() -> None:
    expected = 100.0 / math.sqrt(1000.0**2 + 500.0**2)
    horizontal = diagonal_normalized_image_distance(100, 0, 1000, 500)
    vertical = diagonal_normalized_image_distance(0, 100, 1000, 500)
    assert horizontal == pytest.approx(expected)
    assert vertical == pytest.approx(expected)


@pytest.mark.parametrize(
    ("width", "height"),
    [(0.0, 500.0), (1000.0, 0.0), (math.nan, 500.0)],
)
def test_invalid_image_dimensions_make_distance_unavailable(
    width: float,
    height: float,
) -> None:
    assert math.isnan(axis_normalized_image_distance(1, 1, width, height))
    assert math.isnan(
        diagonal_normalized_image_distance(1, 1, width, height)
    )


def test_coincident_valid_centers_are_measured_zero() -> None:
    assert axis_normalized_image_distance(0, 0, 1000, 500) == 0.0
    assert diagonal_normalized_image_distance(0, 0, 1000, 500) == 0.0


def test_distance_metric_versions_and_physical_claims() -> None:
    assert AXIS_DISTANCE_METRIC_ID == "image_axis_normalized_distance"
    assert DIAGONAL_DISTANCE_METRIC_ID == "image_diagonal_normalized_distance"
    assert AXIS_DISTANCE_METRIC_VERSION.endswith(".v1")
    assert DIAGONAL_DISTANCE_METRIC_VERSION.endswith(".v1")


def test_equal_distance_neighbor_uses_canonical_key() -> None:
    result = build_static_social_context_features(_social_rows())
    actor = _social_actor(result, "A")
    assert actor["nearest_partner_key"] == "video-a|track=B"
    assert actor["nearest_tie_count"] == 2
    assert actor["nearest_tie_break_rule"] == SOCIAL_TIE_BREAK_RULE
    assert actor["nearest_dist_n"] == pytest.approx(0.1)


def test_equal_distance_selection_is_row_order_invariant() -> None:
    first = build_static_social_context_features(
        _social_rows(("A", "B", "C"))
    )
    second = build_static_social_context_features(
        _social_rows(("C", "A", "B"))
    )
    columns = [
        "object_track_key",
        "nearest_partner_key",
        "nearest_dist_n",
        "nearest_distance_diagonal",
        "nearest_tie_count",
    ]
    first = first[columns].sort_values("object_track_key").reset_index(drop=True)
    second = second[columns].sort_values("object_track_key").reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)


def test_actor_never_selects_itself() -> None:
    result = build_static_social_context_features(_social_rows())
    assert (
        result["nearest_partner_key"].ne(result["object_track_key"])
    ).all()


def test_blank_pig_id_does_not_remove_stable_identity() -> None:
    rows = _social_rows()
    rows["pig_id"] = ""
    result = build_static_social_context_features(rows)
    assert _social_actor(result, "A")["nearest_partner_key"] == (
        "video-a|track=B"
    )


def test_duplicate_pig_id_does_not_collapse_tracks() -> None:
    rows = _social_rows()
    rows.loc[
        rows["object_track_key"].isin(
            ["video-a|track=B", "video-a|track=C"]
        ),
        "pig_id",
    ] = "duplicate"
    result = build_static_social_context_features(rows)
    actor = _social_actor(result, "A")
    assert actor["nearest_partner_key"] == "video-a|track=B"
    assert actor["nearest_tie_count"] == 2


def test_same_pig_id_in_two_videos_does_not_form_neighbor() -> None:
    rows = pd.DataFrame(
        [
            _row(actor="A", video="video-a", pig_id="same"),
            _row(actor="A", video="video-b", pig_id="same"),
        ]
    )
    result = build_static_social_context_features(rows)
    assert result["nearest_neighbor_available"].eq(False).all()
    assert result["nearest_partner_key"].eq("").all()


def test_no_valid_neighbor_is_not_zero_distance() -> None:
    result = build_static_social_context_features(
        pd.DataFrame([_row(actor="A")])
    )
    assert not result.loc[0, "nearest_neighbor_available"]
    assert pd.isna(result.loc[0, "nearest_dist_n"])
    assert pd.isna(result.loc[0, "nearest_distance_diagonal"])


def test_invalid_partner_geometry_is_excluded() -> None:
    rows = _social_rows()
    rows.loc[rows["object_track_key"].eq("video-a|track=B"), "bbox_valid"] = False
    result = build_static_social_context_features(rows)
    assert _social_actor(result, "A")["nearest_partner_key"] == (
        "video-a|track=C"
    )


def test_partner_continuity_uses_stable_key() -> None:
    rows = pd.DataFrame(
        [
            _row(actor=actor, frame=frame)
            for frame in (0, 1)
            for actor in ("A", "B")
        ]
    )
    result = _add_social_context_columns(
        _add_pair_support(rows),
        EnhancedFeatureConfig(),
    )
    actor = result.loc[result["object_track_key"].eq("video-a|track=A")]
    assert not actor.iloc[0]["partner_continuity_valid"]
    assert actor.iloc[1]["partner_continuity_valid"]
    assert actor.iloc[1]["same_partner_as_previous"]
    assert not actor.iloc[1]["partner_switch"]


def test_partner_switch_is_measured_only_when_both_neighbors_exist() -> None:
    rows = pd.DataFrame(
        [
            _row(actor="A", frame=0),
            _row(actor="B", frame=0),
            _row(actor="A", frame=1),
            _row(actor="C", frame=1),
        ]
    )
    result = _add_social_context_columns(
        _add_pair_support(rows),
        EnhancedFeatureConfig(),
    )
    actor = result.loc[result["object_track_key"].eq("video-a|track=A")]
    assert actor.iloc[1]["partner_continuity_valid"]
    assert actor.iloc[1]["partner_switch"]
    assert not actor.iloc[1]["same_partner_as_previous"]


def test_missing_neighbor_is_distinct_from_partner_switch() -> None:
    rows = pd.DataFrame(
        [
            _row(actor="A", frame=0),
            _row(actor="B", frame=0),
            _row(actor="A", frame=1),
        ]
    )
    result = _add_social_context_columns(
        _add_pair_support(rows),
        EnhancedFeatureConfig(),
    )
    actor = result.loc[result["object_track_key"].eq("video-a|track=A")]
    assert actor.iloc[1]["no_neighbor"]
    assert not actor.iloc[1]["partner_continuity_valid"]
    assert not actor.iloc[1]["partner_switch"]


def test_partner_continuity_resets_at_temporal_unit() -> None:
    rows = pd.DataFrame(
        [
            _row(actor="A", frame=0, unit="unit-a"),
            _row(actor="B", frame=0, unit="unit-b"),
            _row(actor="A", frame=1, unit="unit-a-next"),
            _row(actor="B", frame=1, unit="unit-b-next"),
        ]
    )
    result = _add_social_context_columns(
        _add_pair_support(rows),
        EnhancedFeatureConfig(),
    )
    actor = result.loc[result["object_track_key"].eq("video-a|track=A")]
    assert actor["partner_continuity_valid"].eq(False).all()


def test_partner_continuity_resets_at_actor_identity_change() -> None:
    rows = pd.DataFrame(
        [
            _row(actor="A0", frame=0),
            _row(actor="B", frame=0),
            _row(actor="A1", frame=1),
            _row(actor="B", frame=1),
        ]
    )
    result = _add_social_context_columns(
        _add_pair_support(rows),
        EnhancedFeatureConfig(),
    )
    actors = result.loc[result["object_track_key"].isin(
        ["video-a|track=A0", "video-a|track=A1"]
    )]
    assert actors["partner_continuity_valid"].eq(False).all()


def test_roi_partial_availability_uses_available_denominator() -> None:
    result = _roi_unit(
        [True, True, True, False, False],
        [True, True, False, False, False],
    ).iloc[0]
    assert result["observed_frame_count"] == 5
    assert result["target_roi_available_frame_count"] == 3
    assert result["target_roi_contact_frame_count"] == 2
    assert result["target_roi_availability_ratio_unit"] == pytest.approx(0.6)
    assert result["target_roi_contact_ratio_unit"] == pytest.approx(2.0 / 3.0)
    assert result["target_roi_contact_ratio_unit"] != pytest.approx(0.4)
    assert result["target_roi_unit_available"]
    assert result["roi_aggregation_version"] == ROI_AGGREGATION_VERSION


def test_zero_roi_availability_is_unavailable_not_measured_no_contact() -> None:
    result = _roi_unit(
        [False] * 5,
        [False] * 5,
    ).iloc[0]
    assert result["target_roi_available_frame_count"] == 0
    assert result["target_roi_contact_frame_count"] == 0
    assert result["target_roi_contact_ratio_unit"] == 0.0
    assert not result["target_roi_unit_available"]
    assert not result["target_roi_contact_available"]


def test_invalid_current_geometry_is_not_roi_available() -> None:
    rows = pd.DataFrame(
        [_row(actor="A", frame=index) for index in range(3)]
    )
    rows.loc[1, "bbox_valid"] = False
    temporal = _add_temporal_deltas(rows)
    temporal["roi_target_available"] = True
    temporal["roi_target_contact"] = True
    temporal["roi_target_near"] = True
    result = _add_temporal_unit_aggregates(temporal).iloc[0]
    assert result["observed_frame_count"] == 3
    assert result["target_roi_available_frame_count"] == 2
    assert result["target_roi_contact_frame_count"] == 2
    assert result["target_roi_availability_ratio_unit"] == pytest.approx(
        2.0 / 3.0
    )
    assert result["target_roi_contact_ratio_unit"] == 1.0


@pytest.mark.parametrize(
    ("available", "contact", "expected"),
    [
        ([True] * 5, [False] * 5, 0.0),
        ([True] * 5, [True] * 5, 1.0),
        ([True, False, True], [False, False, True], 0.5),
    ],
)
def test_roi_contact_ratio_extremes_and_mixed_geometry(
    available: list[bool],
    contact: list[bool],
    expected: float,
) -> None:
    result = _roi_unit(available, contact).iloc[0]
    assert result["target_roi_contact_ratio_unit"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "column",
    [
        "target_roi_contact",
        "target_roi_distance",
        "target_roi_contact_ratio_unit",
        "label_selected_roi_class_indicator",
        "roi_target_contact",
    ],
)
def test_forbidden_target_roi_export_fails(column: str) -> None:
    windows, frames = _window_and_frame(column)
    with pytest.raises(ValueError, match="Forbidden spatial feature"):
        export_spatial_sequences(
            windows,
            frames,
            feature_schema={"requested": [column]},
        )


def test_label_independent_roi_feature_is_allowed() -> None:
    windows, frames = _window_and_frame("roi_feeder_contact")
    exported = export_spatial_sequences(
        windows,
        frames,
        feature_schema={"roi_class_relation": ["roi_feeder_contact"]},
    )
    assert exported.feature_names["roi_class_relation"] == [
        "roi_feeder_contact"
    ]


def test_target_roi_explicit_train_ready_whitelist_fails() -> None:
    frame = pd.DataFrame({"target_roi_contact": [1.0]})
    with pytest.raises(ValueError, match="forbidden"):
        select_window_feature_columns(
            frame,
            feature_whitelist=["target_roi_contact"],
        )


def test_target_roi_policy_authority_is_fail_closed() -> None:
    assert is_target_roi_model_forbidden("target_roi_contact")
    assert is_target_roi_model_forbidden("roi_target_min_dist_n")
    assert not is_target_roi_model_forbidden("roi_feeder_min_dist_n")


def test_phase2_motion_schema_hash_is_frozen() -> None:
    assert MOTION_SCHEMA_HASH == (
        "ec0c511b5f5198240492be49c0492e543"
        "c9e38eb4a4ff446259b958c2a59963b"
    )


def test_phase3_social_versions_are_explicit() -> None:
    result = build_static_social_context_features(_social_rows())
    assert result["social_identity_version"].eq(
        SOCIAL_IDENTITY_VERSION
    ).all()
    assert result["social_tie_break_version"].eq(
        SOCIAL_TIE_BREAK_VERSION
    ).all()


def test_native_producer_and_independent_checker_agree() -> None:
    source = pd.DataFrame(
        [
            _row(actor=actor, frame=frame)
            for frame in (0, 1, 2)
            for actor in ("A", "B")
        ]
    )
    output = build_enhanced_spatiotemporal_features(source)
    code_sha = "1" * 40
    input_sha = "2" * 64
    contract_sha = "3" * 64
    output["code_authority_sha"] = code_sha
    output["input_sha256"] = input_sha
    output["contract_manifest_sha256"] = contract_sha
    producer_audit = audit_enhanced_spatiotemporal_features(
        output,
        input_rows=len(source),
        code_sha=code_sha,
        input_sha256=input_sha,
        contract_manifest_sha256=contract_sha,
    )
    assert producer_audit["errors"] == []
    checked = check_native_review_evidence(
        source,
        output,
        producer_audit=producer_audit,
        code_sha=code_sha,
        input_sha256=input_sha,
        contract_manifest_sha256=contract_sha,
    )
    assert checked["valid"], checked["errors"]
    assert checked["phase3_spatial_checks"]["self_neighbor_count"] == 0
    assert checked["phase3_spatial_checks"]["roi_denominator_errors"] == 0


def test_phase3_producer_is_deterministic_after_canonical_sort() -> None:
    source = pd.DataFrame(
        [
            _row(actor=actor, frame=frame)
            for frame in (0, 1)
            for actor in ("A", "B", "C")
        ]
    )
    first = build_enhanced_spatiotemporal_features(source)
    second = build_enhanced_spatiotemporal_features(
        source.sample(frac=1.0, random_state=17)
    )
    columns = [
        "object_track_key",
        "frame_index",
        "nearest_partner_key",
        "nearest_dist_n",
        "nearest_distance_diagonal",
        "nearest_tie_count",
        "partner_continuity_valid",
        "same_partner_as_previous",
    ]
    ordering = ["object_track_key", "frame_index"]
    first = first[columns].sort_values(ordering).reset_index(drop=True)
    second = second[columns].sort_values(ordering).reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)
