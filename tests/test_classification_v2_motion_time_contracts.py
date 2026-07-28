from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.evaluation.final_view_contract_audit import (
    audit_final_view_contract,
    audit_pre_review_structural_view_availability,
)
from pig_behavior.classification_v2.features.motion_schema import (
    MOTION_FEATURE_NAMES,
    MOTION_SCHEMA_DIMENSION,
    MOTION_SCHEMA_HASH,
    MOTION_SCHEMA_ID,
    MOTION_SCHEMA_VERSION,
)
from pig_behavior.classification_v2.features.pen_context import (
    _add_pen_temporal_derivatives,
)
from pig_behavior.classification_v2.features.sequence_windows import (
    SequenceWindowConfig,
    _aggregate_window_features,
    _window_timing_summary,
    build_sequence_windows,
)
from pig_behavior.classification_v2.features.spatial_schema import (
    SPATIAL_PREDICTIVE_FEATURES,
)
from pig_behavior.classification_v2.features.spatiotemporal import (
    _add_roi_temporal_columns,
    _add_temporal_deltas,
)
from pig_behavior.classification_v2.features.temporal_evidence import (
    TemporalEvidenceConfig,
    summarize_social_motion_dynamics,
    summarize_temporal_evidence,
)
from pig_behavior.classification_v2.merge_sources import (
    audit_merged_frame_objects,
)
from pig_behavior.classification_v2.sources.legacy_recovered_csv import (
    load_legacy_frame_objects,
)
from pig_behavior.classification_v2.sources.temporal_provenance import (
    CANONICAL_TIMESTAMP_SOURCE,
    apply_source_frame_clock,
    audit_source_frame_clock,
)
from pig_behavior.classification_v2.spatial_sequence_export import (
    export_spatial_sequences,
)
from pig_behavior.classification_v2.train_ready_features import (
    select_window_feature_columns,
)


def _motion_rows(
    frames: list[int],
    timestamps: list[float],
    centers: list[float],
    units: list[str],
) -> pd.DataFrame:
    row_count = len(frames)
    return pd.DataFrame(
        {
            "source_type": ["legacy_recovered"] * row_count,
            "dataset_id": ["legacy"] * row_count,
            "video_key": ["video-a"] * row_count,
            "object_track_key": ["track-a"] * row_count,
            "temporal_unit_key": units,
            "frame_index": frames,
            "timestamp_sec": timestamps,
            "cx_n": centers,
            "cy_n": [0.5] * row_count,
            "bw_n": [0.2] * row_count,
            "bh_n": [0.1] * row_count,
            "area_n": [0.02] * row_count,
            "aspect_ratio": [2.0] * row_count,
            "box_diag_n": [np.hypot(0.2, 0.1)] * row_count,
            "bbox_valid": [True] * row_count,
        }
    )


def _attach_motion_v2_contract(frames: pd.DataFrame) -> pd.DataFrame:
    out = _add_temporal_deltas(frames)
    out["motion_feature_available"] = out.groupby(
        "temporal_unit_key"
    )["velocity_valid"].transform("any")
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
    return pd.concat(
        [out, pd.DataFrame(additions, index=out.index)],
        axis=1,
    ).copy()


def test_same_fps_legacy_and_cvat_t6_have_same_physical_clock() -> None:
    source_frames = list(range(100, 106))
    base = pd.DataFrame(
        {
            "frame_index": source_frames,
            "timestamp_sec": np.linspace(10.0, 11.0, 6),
            "timestamp_source": "input_timestamp",
        }
    )
    legacy = apply_source_frame_clock(
        base.assign(source_type="legacy_recovered"),
        source_fps=30.0,
        preserve_input_as_acquisition=True,
    )
    cvat = apply_source_frame_clock(
        base.assign(source_type="cvat_tracking_xml"),
        source_fps=30.0,
        preserve_input_as_acquisition=False,
    )

    assert np.diff(legacy["timestamp_sec"]).tolist() == pytest.approx(
        [1.0 / 30.0] * 5
    )
    assert np.diff(cvat["timestamp_sec"]).tolist() == pytest.approx(
        [1.0 / 30.0] * 5
    )
    legacy_span = (
        legacy["timestamp_sec"].iloc[-1]
        - legacy["timestamp_sec"].iloc[0]
    )
    cvat_span = (
        cvat["timestamp_sec"].iloc[-1]
        - cvat["timestamp_sec"].iloc[0]
    )
    assert legacy_span == pytest.approx(5.0 / 30.0)
    assert cvat_span == pytest.approx(5.0 / 30.0)


def test_source_frame_clock_audit_rejects_wrong_timestamp() -> None:
    rows = apply_source_frame_clock(
        pd.DataFrame({"frame_index": [0, 1]}),
        source_fps=30.0,
        preserve_input_as_acquisition=False,
    )
    rows.loc[1, "timestamp_sec"] = 0.2

    audit = audit_source_frame_clock(rows)

    assert "timestamp_formula_mismatch_rows=1" in audit["errors"]


@pytest.mark.parametrize(
    "column",
    [
        "frame_index",
        "source_frame_index",
        "relative_frame_index",
        "native_offset",
        "source_fps",
        "timestamp_sec",
        "acquisition_timestamp_sec",
    ],
)
def test_temporal_provenance_columns_are_forbidden_from_model_x(
    column: str,
) -> None:
    windows = pd.DataFrame({column: [1.0]})
    with pytest.raises(ValueError, match="forbidden feature"):
        select_window_feature_columns(
            windows,
            feature_whitelist=[column],
        )


def test_source_clock_rejects_silent_per_video_fps_override() -> None:
    rows = pd.DataFrame(
        {
            "frame_index": [0, 1],
            "source_fps": [25.0, 25.0],
        }
    )
    with pytest.raises(ValueError, match="source_fps disagrees"):
        apply_source_frame_clock(
            rows,
            source_fps=30.0,
            preserve_input_as_acquisition=False,
        )


def test_view_contract_recommends_shared_t6_and_sparse_ablation() -> None:
    rows: list[dict[str, object]] = []
    for source in ("legacy_recovered", "cvat_tracking_xml"):
        for view, slots, span in (
            ("T6_contiguous", 6, 5.0 / 30.0),
            ("T8_contiguous", 8, 7.0 / 30.0),
        ):
            rows.append(
                {
                    "source_type": source,
                    "view_type": view,
                    "sampling_pattern": "contiguous",
                    "physical_span_seconds": span,
                    "expected_slot_count": slots,
                    "observed_slot_count": slots,
                    "pair_delta_frames": "[1,1,1,1,1]",
                    "pair_delta_seconds": "shared_30fps",
                    "primary_cross_source_eligible": True,
                }
            )
    rows.append(
        {
            "source_type": "legacy_recovered",
            "view_type": "S6@16",
            "sampling_pattern": "uniform_sparse_offsets_0_3_6_9_12_15",
            "physical_span_seconds": 15.0 / 30.0,
            "expected_slot_count": 6,
            "observed_slot_count": 6,
            "pair_delta_frames": "[3,3,3,3,3]",
            "pair_delta_seconds": "sparse_30fps",
            "primary_cross_source_eligible": False,
        }
    )

    audit = audit_final_view_contract(pd.DataFrame(rows))

    assert audit["errors"] == []
    assert audit["primary_view_recommendation"] == "T6_contiguous"
    ablations = {
        row["view_type"]: row["reason"]
        for row in audit["ablation_view_recommendations"]
    }
    assert ablations["S6@16"] == "legacy_only_sparse_ablation_not_primary"


def test_view_timing_source_shortcut_blocks_primary_recommendation() -> None:
    rows = []
    for source, span, pair_seconds in (
        ("legacy_recovered", 0.5, "legacy_only_timing"),
        ("cvat_tracking_xml", 5.0 / 30.0, "cvat_only_timing"),
    ):
        for _ in range(2):
            rows.append(
                {
                    "source_type": source,
                    "view_type": "T6_contiguous",
                    "sampling_pattern": "contiguous",
                    "physical_span_seconds": span,
                    "expected_slot_count": 6,
                    "observed_slot_count": 6,
                    "pair_delta_frames": "[1,1,1,1,1]",
                    "pair_delta_seconds": pair_seconds,
                    "primary_cross_source_eligible": True,
                }
            )

    audit = audit_final_view_contract(pd.DataFrame(rows))

    report = audit["source_predictability_from_view_metadata"]["by_view"]
    assert report["T6_contiguous"]["near_direct_source_signature"] is True
    assert audit["primary_view_recommendation"] is None
    assert "no_cross_source_primary_view_without_metadata_shortcut" in audit[
        "errors"
    ]


def test_structural_view_availability_is_not_review_authority() -> None:
    records = []
    for frame in range(16):
        records.append(
            {
                "source_type": "legacy_recovered",
                "object_track_key": "legacy-track",
                "temporal_unit_key": "legacy-unit",
                "frame_index": frame,
            }
        )
    for frame in range(24):
        records.append(
            {
                "source_type": "cvat_tracking_xml",
                "object_track_key": "cvat-track",
                "temporal_unit_key": f"cvat-unit-{frame // 6}",
                "frame_index": frame,
            }
        )

    audit = audit_pre_review_structural_view_availability(
        pd.DataFrame(records),
        source_fps=30.0,
    )

    assert audit["errors"] == []
    assert audit["scope"] == "PRE_REVIEW_STRUCTURAL_ONLY"
    assert audit["review_eligibility_applied"] is False
    assert audit["not_train_ready_authority"] is True
    assert audit["primary_view_recommendation"] == "T6_contiguous"
    availability = audit["source_by_view_availability"]
    assert availability["legacy_recovered"]["S6@16"] == 1
    assert availability["cvat_tracking_xml"]["S6@16"] == 0


def test_legacy_loader_preserves_times_txt_as_audit_only(
    tmp_path,
) -> None:
    source = pd.DataFrame(
        {
            "image_key": ["scene-0", "scene-1"],
            "image_name": ["f10.jpg", "f11.jpg"],
            "source_video_key": ["video-a", "video-a"],
            "tracklet_id": ["track-a", "track-a"],
            "pig_id": ["ID_1", "ID_1"],
            "behavior": ["stand", "stand"],
            "frame_index": [10, 11],
            "relative_frame_index": [0, 1],
            "timestamp_sec": [1.62, 1.78],
            "timestamp_source": ["times_txt", "times_txt"],
            "x1": [10.0, 11.0],
            "y1": [10.0, 10.0],
            "x2": [20.0, 21.0],
            "y2": [20.0, 20.0],
        }
    )
    path = tmp_path / "legacy_frame_object_annotations.csv"
    source.to_csv(path, index=False)

    loaded = load_legacy_frame_objects(path, source_fps=30.0)

    assert loaded["source_frame_index"].tolist() == [10, 11]
    assert loaded["native_offset"].tolist() == [0, 1]
    assert loaded["timestamp_sec"].tolist() == pytest.approx(
        [10.0 / 30.0, 11.0 / 30.0]
    )
    assert loaded["acquisition_timestamp_sec"].tolist() == pytest.approx(
        [1.62, 1.78]
    )
    assert loaded["timestamp_source"].eq(CANONICAL_TIMESTAMP_SOURCE).all()

    merged_audit = audit_merged_frame_objects(loaded)
    assert merged_audit["timestamp_clock_audit"]["status"] == "pass"
    tampered = loaded.copy()
    tampered.loc[tampered.index[-1], "timestamp_sec"] = 99.0
    tampered_audit = audit_merged_frame_objects(tampered)
    assert any(
        "timestamp_formula_mismatch_rows=1" in error
        for error in tampered_audit["errors"]
    )


def test_motion_resets_at_native_unit_and_roi_boundary() -> None:
    rows = _motion_rows(
        [0, 1, 2, 3],
        [0.0, 0.1, 0.2, 0.3],
        [0.0, 0.1, 0.9, 1.0],
        ["unit-a", "unit-a", "unit-b", "unit-b"],
    )
    rows["behavior"] = "eat"
    rows["roi_target_class"] = "feeder"
    rows["roi_target_available"] = True
    rows["roi_target_contact"] = [False, False, True, True]
    rows["roi_target_near"] = [False, False, True, True]
    rows["roi_target_center_inside"] = False

    motion = _add_temporal_deltas(rows)
    result = _add_roi_temporal_columns(motion)

    unit_start = result["temporal_unit_key"].ne(
        result["temporal_unit_key"].shift(1)
    )
    assert result.loc[unit_start, "speed_n_per_second"].isna().all()
    assert result.loc[
        unit_start,
        "tangential_acceleration_n_per_second2",
    ].isna().all()
    assert not result.loc[unit_start, "adjacent_motion_pair_valid"].any()
    assert not bool(result.loc[2, "roi_target_entry_event"])


def test_gap_pair_is_sparse_velocity_not_contiguous_path() -> None:
    rows = _motion_rows(
        [0, 2],
        [0.0, 0.2],
        [0.0, 0.2],
        ["unit-a", "unit-a"],
    )
    motion = _add_temporal_deltas(rows)
    evidence = summarize_temporal_evidence(
        rows,
        expected_start=0,
        expected_end=2,
    )

    assert int(motion["adjacent_motion_pair_valid"].sum()) == 0
    assert int(motion["sparse_velocity_pair_valid"].sum()) == 1
    assert motion["displacement_n"].sum() == 0.0
    assert motion.loc[1, "speed_n_per_second"] == pytest.approx(1.0)
    assert evidence["trajectory_path_length_n"] == 0.0
    assert evidence["trajectory_sparse_path_length_n"] == pytest.approx(0.2)


@pytest.mark.parametrize("fps", [10.0, 30.0])
def test_physical_speed_is_fps_invariant(fps: float) -> None:
    velocity = 0.3
    rows = _motion_rows(
        [0, 1, 2],
        [0.0, 1.0 / fps, 2.0 / fps],
        [0.0, velocity / fps, 2.0 * velocity / fps],
        ["unit-a"] * 3,
    )
    evidence = summarize_temporal_evidence(
        rows,
        config=TemporalEvidenceConfig(),
    )

    assert evidence["motion_speed_n_per_second_p50"] == pytest.approx(
        velocity
    )
    assert evidence["motion_active_ratio_per_second"] == 1.0


def test_exact_window_recompute_ignores_inherited_pair_columns() -> None:
    rows = _motion_rows(
        [1, 2],
        [0.1, 0.2],
        [0.5, 0.6],
        ["unit-a", "unit-a"],
    )
    rows["speed_n_per_frame"] = [999.0, 999.0]
    rows["speed_n_per_second"] = [999.0, 999.0]
    rows["displacement_n"] = [999.0, 999.0]

    summary = _aggregate_window_features(
        rows,
        0.2,
        expected_start=1,
        expected_end=2,
        evidence_config=SequenceWindowConfig().temporal_evidence_config(),
    )

    assert summary["speed_n_per_second_mean_window"] == pytest.approx(1.0)
    assert summary["path_length_n_window"] == pytest.approx(0.1)


def test_final_view_rejects_parent_final_view_artifact() -> None:
    rows = _motion_rows(
        [1, 2],
        [0.1, 0.2],
        [0.5, 0.6],
        ["unit-a", "unit-a"],
    )
    rows["feature_computation_grain"] = "FINAL_VIEW_FEATURES"
    rows["pair_recomputed_for_view"] = True
    rows["aggregate_recomputed_for_view"] = True

    with pytest.raises(ValueError, match="another final-view artifact"):
        _aggregate_window_features(
            rows,
            0.2,
            expected_start=1,
            expected_end=2,
            evidence_config=SequenceWindowConfig().temporal_evidence_config(),
        )


def test_final_view_rejects_mismatched_native_pair_scope() -> None:
    rows = _motion_rows(
        [1, 2],
        [0.1, 0.2],
        [0.5, 0.6],
        ["unit-a", "unit-a"],
    )
    rows["feature_computation_grain"] = "NATIVE_UNIT_REVIEW_EVIDENCE"
    rows["pair_scope_key"] = "unit-b"

    with pytest.raises(ValueError, match="does not match temporal_unit_key"):
        _aggregate_window_features(
            rows,
            0.2,
            expected_start=1,
            expected_end=2,
            evidence_config=SequenceWindowConfig().temporal_evidence_config(),
        )


def test_social_physical_rate_is_invariant_to_frame_rate() -> None:
    summaries = []
    for fps in (10.0, 30.0):
        timestamps = [0.0, 1.0 / fps, 2.0 / fps]
        rows = _motion_rows(
            [0, 1, 2],
            timestamps,
            [0.0, 0.01, 0.02],
            ["unit-a"] * 3,
        )
        rows["nearest_pig_id"] = "ID_2"
        rows["nearest_dist_n"] = [1.0 - 0.3 * value for value in timestamps]
        rows["pair_contact_with_nearest"] = True
        rows["social_density_near_count"] = 0.0
        summaries.append(summarize_social_motion_dynamics(rows))

    assert summaries[0]["approach_speed_n_per_second_max"] == pytest.approx(
        0.3
    )
    assert summaries[1]["approach_speed_n_per_second_max"] == pytest.approx(
        0.3
    )
    assert summaries[0]["approach_speed_max"] != pytest.approx(
        summaries[1]["approach_speed_max"]
    )


def test_social_motion_mean_excludes_nonexistent_first_pair() -> None:
    rows = _motion_rows(
        [0, 1, 2],
        [0.0, 0.1, 0.2],
        [0.0, 0.1, 0.2],
        ["unit-a"] * 3,
    )
    rows["nearest_pig_id"] = "ID_2"
    rows["nearest_dist_n"] = [1.0, 0.9, 0.8]
    rows["pair_contact_with_nearest"] = True
    rows["social_density_near_count"] = 0.0

    summary = summarize_social_motion_dynamics(rows)

    assert summary["aggression_score_proxy_mean"] == pytest.approx(0.2)
    assert summary["aggression_score_proxy_n_per_second_mean"] == pytest.approx(
        2.0
    )


def test_legacy_behavior_template_shift_and_rolling_reset_by_native_unit() -> None:
    script = Path(
        "scripts/classification_v2/01_review_units_gui/"
        "build_behavior_review_templates.py"
    )
    add_review_attributes = runpy.run_path(str(script))["add_review_attributes"]
    rows = pd.DataFrame(
        {
            "source_type": ["legacy_recovered"] * 4,
            "dataset_id": ["dataset"] * 4,
            "video_key": ["video"] * 4,
            "track_id": ["track-a"] * 4,
            "pig_id": ["ID_1"] * 4,
            "temporal_unit_key": ["unit-a", "unit-a", "unit-b", "unit-b"],
            "frame_index": [0, 1, 2, 3],
            "timestamp_sec": [0.0, 0.1, 0.2, 0.3],
            "cx_n": [0.0, 1.0, 100.0, 101.0],
            "cy_n": [0.0] * 4,
            "behavior": ["move"] * 4,
            "bbox_valid": [True] * 4,
        }
    )

    result = add_review_attributes(
        rows,
        motion_low_threshold=0.18,
        motion_strong_threshold=0.75,
        boundary_frame_gap=12,
        window_radius=1,
    )

    assert result.loc[2, "adjacent_motion_pair_valid_auto"] == np.False_
    assert result.loc[2, "step_speed_n_per_second_auto"] == pytest.approx(0.0)
    assert result.loc[2, "window_speed_mean_n_per_second_auto"] == (
        pytest.approx(5.0)
    )


def test_sparse_duration_separates_declared_and_observed_timing() -> None:
    rows = _motion_rows(
        [0, 5],
        [0.0, 5.0 / 30.0],
        [0.0, 0.1],
        ["unit-a", "unit-a"],
    )
    timing = _window_timing_summary(
        rows,
        start=0,
        end=5,
        expected_slot_count=6,
        default_fps=None,
    )

    assert timing["declared_window_duration_seconds"] == pytest.approx(0.2)
    assert timing["observed_timestamp_span_seconds"] == pytest.approx(5 / 30)
    assert timing["adjacent_observed_duration_seconds"] == 0.0
    assert timing["effective_observation_rate_hz"] == pytest.approx(6.0)
    assert timing["adjacent_pair_coverage_ratio"] == 0.0


def _pen_rows(centers: list[tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for frame_index, (center_x, center_y) in enumerate(centers):
        rows.append(
            {
                "source_type": "legacy_recovered",
                "dataset_id": "legacy",
                "video_key": "video-a",
                "object_track_key": "track-a",
                "temporal_unit_key": "unit-a",
                "frame_index": frame_index,
                "timestamp_sec": frame_index / 10.0,
                "x1": center_x - 1.0,
                "y1": center_y - 1.0,
                "x2": center_x + 1.0,
                "y2": center_y + 1.0,
                "image_width": 100.0,
                "image_height": 100.0,
                "pen_context_available": True,
                "pen_center_inside": True,
                "pen_center_signed_distance_n": center_x / np.hypot(100, 100),
                "pen_boundary_inward_normal_x": 1.0,
                "pen_boundary_inward_normal_y": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_pen_projection_pure_normal_and_tangent_invariants() -> None:
    normal = _add_pen_temporal_derivatives(
        _pen_rows([(10.0, 10.0), (12.0, 10.0)])
    )
    tangent = _add_pen_temporal_derivatives(
        _pen_rows([(10.0, 10.0), (10.0, 12.0)])
    )

    assert normal.loc[1, "pen_parallel_speed_n_per_second"] == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert tangent.loc[1, "pen_normal_speed_n_per_second"] == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert tangent.loc[1, "pen_approach_speed_n_per_second"] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_invalid_geometry_is_zeroed_and_quality_masked() -> None:
    frames = _motion_rows(
        [0, 1],
        [0.0, 0.1],
        [0.0, 0.1],
        ["unit-a", "unit-a"],
    )
    frames.loc[1, "bbox_valid"] = False
    frames["vx_n_per_second"] = [0.0, 1.0e9]
    frames = _attach_motion_v2_contract(frames)
    windows = pd.DataFrame(
        {
            "window_id": ["track-a|win=2|0-1"],
            "object_track_key": ["track-a"],
            "window_start_frame": [0],
            "window_end_frame": [1],
            "window_length_frames": [2],
            "feature_computation_grain": ["FINAL_VIEW_FEATURES"],
            "pair_scope_key": ["track-a|win=2|0-1"],
            "view_type": ["T2_contiguous"],
            "sampling_pattern": ["contiguous"],
            "selected_frame_offsets": ["[0,1]"],
            "selected_frame_indices": ["[0,1]"],
            "selected_timestamps_seconds": ["[0.0,0.1]"],
            "pair_delta_frames": ["[1]"],
            "pair_delta_seconds": ["[0.1]"],
            "pair_recomputed_for_view": [True],
            "aggregate_recomputed_for_view": [True],
        }
    )

    exported = export_spatial_sequences(windows, frames)

    assert exported.arrays["spatial_quality_mask"][0].tolist() == [1.0, 0.0]
    assert exported.arrays["motion_delta"][0, 1].sum() == 0.0
    assert exported.arrays["observed_mask"][0].tolist() == [1.0, 1.0]


def _reviewed_cvat_rows() -> pd.DataFrame:
    frames = list(range(12))
    return pd.DataFrame(
        {
            "source_type": ["cvat_tracking_xml"] * 12,
            "dataset_id": ["cvat"] * 12,
            "video_key": ["video-cvat"] * 12,
            "frame_uid": [f"video-cvat::f{frame:06d}" for frame in frames],
            "frame_index": frames,
            "relative_frame_index": frames,
            "timestamp_sec": [frame / 30.0 for frame in frames],
            "pig_id": ["ID_1"] * 12,
            "track_id": ["1"] * 12,
            "behavior": ["stand"] * 12,
            "bbox_valid": [True] * 12,
            "hidden": ["No"] * 12,
            "cx_n": [frame / 100.0 for frame in frames],
            "cy_n": [0.5] * 12,
            "bw_n": [0.2] * 12,
            "bh_n": [0.1] * 12,
            "area_n": [0.02] * 12,
            "aspect_ratio": [2.0] * 12,
            "behavior_review_decision_present": [True] * 12,
            "behavior_review_label_resolved": [True] * 12,
            "behavior_review_include_in_training": [True] * 12,
            "behavior_reviewed_final": ["stand"] * 12,
        }
    )


def test_cvat_t8_requires_both_intervals_reviewed_and_eligible() -> None:
    rows = _reviewed_cvat_rows()
    _, _, accepted = build_sequence_windows(
        rows,
        window_lengths=[8],
        behavior_review_requirement="full_native_unit_review_required",
    )
    first = accepted.sort_values("window_start_frame").iloc[0]
    assert first["num_temporal_units_window"] == 2
    assert bool(first["window_valid_for_main_train"]) is True
    assert first["adjacent_motion_pair_count_window"] == 7
    assert first["feature_computation_grain"] == "FINAL_VIEW_FEATURES"
    assert first["pair_scope_key"] == first["window_id"]
    assert bool(first["pair_recomputed_for_view"]) is True
    assert bool(first["aggregate_recomputed_for_view"]) is True

    rows.loc[rows["frame_index"] >= 6, "behavior_review_decision_present"] = False
    _, _, rejected = build_sequence_windows(
        rows,
        window_lengths=[8],
        behavior_review_requirement="full_native_unit_review_required",
    )
    first = rejected.sort_values("window_start_frame").iloc[0]
    assert bool(first["window_valid_for_main_train"]) is False
    assert first["human_reviewed_behavior_consistency_status"] == "partial"


def test_cvat_t8_requires_same_resolved_final_label() -> None:
    rows = _reviewed_cvat_rows()
    rows.loc[rows["frame_index"] >= 6, "behavior_reviewed_final"] = "move"

    _, _, windows = build_sequence_windows(
        rows,
        window_lengths=[8],
        behavior_review_requirement="full_native_unit_review_required",
    )

    first = windows.sort_values("window_start_frame").iloc[0]
    assert bool(first["window_valid_for_main_train"]) is False
    assert first["human_reviewed_behavior_consistency_status"] == "transition"
    assert "behavior_review_transition" in first["window_exclusion_reason"]


def _reviewed_legacy_rows() -> pd.DataFrame:
    frames = list(range(16))
    return pd.DataFrame(
        {
            "source_type": ["legacy_recovered"] * 16,
            "dataset_id": ["legacy"] * 16,
            "video_key": ["legacy-burst"] * 16,
            "frame_uid": [f"legacy-burst::f{frame:06d}" for frame in frames],
            "frame_index": frames,
            "relative_frame_index": frames,
            "timestamp_sec": [frame / 10.0 for frame in frames],
            "pig_id": ["ID_1"] * 16,
            "track_id": ["1"] * 16,
            "behavior": ["move"] * 16,
            "bbox_valid": [True] * 16,
            "hidden": ["No"] * 16,
            "cx_n": [frame / 100.0 for frame in frames],
            "cy_n": [0.5] * 16,
            "bw_n": [0.2] * 16,
            "bh_n": [0.1] * 16,
            "area_n": [0.02] * 16,
            "aspect_ratio": [2.0] * 16,
            "behavior_review_decision_present": [True] * 16,
            "behavior_review_label_resolved": [True] * 16,
            "behavior_review_include_in_training": [True] * 16,
            "behavior_reviewed_final": ["move"] * 16,
        }
    )


def test_local_offsets_do_not_prove_contiguous_source_frames() -> None:
    rows = _reviewed_legacy_rows().iloc[:6].copy()
    source_frames = np.arange(0, 30, 5)
    rows["frame_index"] = source_frames
    rows["source_frame_index"] = source_frames
    rows["relative_frame_index"] = np.arange(6)
    rows["native_offset"] = np.arange(6)
    rows["source_fps"] = 30.0
    rows["timestamp_sec"] = source_frames / 30.0
    rows["timestamp_source"] = CANONICAL_TIMESTAMP_SOURCE

    with pytest.raises(
        ValueError,
        match="not contiguous decoded source frames",
    ):
        build_sequence_windows(
            rows,
            window_lengths=[6],
            behavior_review_requirement=(
                "full_native_unit_review_required"
            ),
        )


def test_final_s6_at16_recomputes_sparse_view_from_reviewed_burst() -> None:
    rows = _reviewed_legacy_rows()
    _, _, windows = build_sequence_windows(
        rows,
        window_lengths=[16],
        behavior_review_requirement="full_native_unit_review_required",
        include_legacy_sparse_s6_at16=True,
    )

    t16 = windows.loc[windows["view_type"].eq("T16_contiguous")].iloc[0]
    sparse = windows.loc[windows["view_type"].eq("S6@16")].iloc[0]
    assert sparse["selected_frame_indices"] == "[0,3,6,9,12,15]"
    assert sparse["pair_delta_frames"] == "[3,3,3,3,3]"
    assert sparse["adjacent_motion_pair_count_window"] == 0
    assert sparse["sparse_velocity_pair_count_window"] == 5
    assert sparse["path_length_n_window"] == 0.0
    assert sparse["sparse_path_length_n_window"] == pytest.approx(0.15)
    assert sparse["speed_n_per_second_mean_window"] == pytest.approx(
        t16["speed_n_per_second_mean_window"]
    )
    assert bool(sparse["primary_cross_source_eligible"]) is False
    assert sparse["pair_scope_key"] == sparse["window_id"]


def test_s6_at16_requires_review_of_unselected_underlying_frame() -> None:
    rows = _reviewed_legacy_rows()
    rows.loc[1, "behavior_review_decision_present"] = False

    _, _, windows = build_sequence_windows(
        rows,
        window_lengths=[16],
        behavior_review_requirement="full_native_unit_review_required",
        include_legacy_sparse_s6_at16=True,
    )

    sparse = windows.loc[windows["view_type"].eq("S6@16")].iloc[0]
    assert bool(sparse["window_valid_for_main_train"]) is False
    assert sparse["human_reviewed_behavior_consistency_status"] == "unreviewed"
