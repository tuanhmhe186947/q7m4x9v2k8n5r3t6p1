from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.features.pig_strenet_artifacts import (
    _segment_summary,
    availability_columns,
    build_pig_strenet_artifacts,
    compute_stabilized_difference_maps,
    model_x_columns,
)
from pig_behavior.classification_v2.features.pig_strenet_checkpoint import (
    PigSTRENetCheckpointError,
    PigSTRENetCheckpointStore,
)
from pig_behavior.classification_v2.review.pig_strenet_review_evidence import (
    PIG_REVIEW_EVIDENCE_COLUMNS,
    attach_pig_strenet_review_evidence,
    build_pig_strenet_review_evidence,
)


def _frames(
    *,
    source_type: str = "legacy_recovered",
    missing_history: bool = False,
    actual_offset: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for relative_index in range(16 if source_type == "legacy_recovered" else 12):
        if missing_history and relative_index < 6:
            continue
        frame_index = actual_offset + relative_index
        scene = f"scene-{frame_index}"
        for actor_index in range(2):
            track = f"track-{actor_index}"
            unit = f"{source_type}|unit-{actor_index}"
            row: dict[str, object] = {
                "object_track_key": track,
                "temporal_unit_key": unit,
                "source_type": source_type,
                "dataset_id": "synthetic",
                "video_key": "video-1",
                "frame_index": frame_index,
                "frame_uid": f"{track}|frame-{frame_index}",
                "scene_frame_uid": scene,
                "relative_frame_index": relative_index,
                "label_window_start": 6 if source_type != "legacy_recovered" else 0,
                "label_window_end": 11 if source_type != "legacy_recovered" else 15,
                "human_review_complete": False,
                "behavior_label": "fight" if actor_index == 0 else "explore",
                "lineage_scope": "synthetic",
                "crop_path": "",
                "cx_n": 0.2 + 0.2 * actor_index + 0.01 * frame_index,
                "cy_n": 0.3,
                "speed_n_per_frame": 0.01 * (frame_index + 1),
                "displacement_n": 0.01,
                "abs_accel_n_per_frame2": 0.001,
                "abs_direction_change_rad": 0.1,
                "nearest_dist_n": 0.2 - 0.001 * frame_index,
                "nearest_dist_delta": -0.001,
                "approach_speed_n_per_frame": 0.001,
                "separation_speed_n_per_frame": 0.0,
                "nearest_pair_iou": 0.1,
                "nearest_pair_overlap_ratio": 0.2,
                "pair_contact_with_nearest": frame_index >= 8,
                "nearest_track_id": f"track-{1 - actor_index}",
                "timestamp_sec": relative_index / 6.0,
            }
            for roi in ("feeder", "drinker", "toy"):
                row.update(
                    {
                        f"roi_{roi}_available": True,
                        f"roi_{roi}_min_dist_n": 0.1 + 0.01 * frame_index,
                        f"roi_{roi}_max_overlap_ratio": 0.2,
                        f"roi_{roi}_max_iou": 0.1,
                        f"roi_{roi}_center_inside": False,
                        f"roi_{roi}_near": frame_index >= 4,
                        f"roi_{roi}_contact": frame_index >= 8,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def test_legacy_history_artifacts_preserve_event_mass_and_controls() -> None:
    artifacts = build_pig_strenet_artifacts(_frames())

    assert len(artifacts.pair_manifest) == 2
    assert artifacts.pair_manifest["derived_view"].eq("legacy_derived_6f").all()
    mass = artifacts.pair_manifest.groupby("native_event_id")["event_weight"].sum()
    assert mass.tolist() == [1.0, 1.0]
    assert len(artifacts.slot_manifest) == 24
    assert len(artifacts.roi_dynamics) == 72
    assert len(artifacts.control_matrix) == 16
    assert set(artifacts.control_matrix["control_id"]) == {
        "T0",
        "T1",
        "H0",
        "HA",
        "HS",
        "HR",
        "HRev",
        "PM",
    }
    assert artifacts.audit["target_selected_roi_used"] is False
    assert artifacts.audit["behavior_selected_partner_used"] is False
    assert not any("target_roi" in name for name in model_x_columns(artifacts.history_features))
    assert not set(availability_columns(artifacts.history_features)).intersection(
        model_x_columns(artifacts.history_features)
    )
    injected = artifacts.history_features.assign(unexpected_numeric_feature=999.0)
    assert "unexpected_numeric_feature" not in model_x_columns(injected)
    assert "history_speed_n_per_second_mean" in model_x_columns(injected)
    for _, group in artifacts.social_edges.groupby(
        ["pair_id", "slot_index", "actor_node_key"], sort=False
    ):
        assert group["neighbor_rank"].tolist() == [1, 2, 3]
    assert artifacts.social_edges["edge_available"].sum() == 24


def test_artifact_builder_reports_bounded_phase_progress() -> None:
    events: list[tuple[str, int | None, int | None]] = []

    build_pig_strenet_artifacts(
        _frames(),
        progress_callback=lambda phase, completed, total: events.append(
            (phase, completed, total)
        ),
    )

    phases = {phase for phase, _, _ in events}
    assert {
        "normalize_frames",
        "build_pairs_and_slots",
        "build_history_features",
        "build_roi_dynamics",
        "build_roi_visual_selection",
        "build_social_graph",
        "build_artifact_audit",
    }.issubset(phases)
    assert any(
        phase == "build_pairs_and_slots" and completed == total
        for phase, completed, total in events
    )


def test_checkpoint_resume_matches_uninterrupted_artifacts(tmp_path: Path) -> None:
    frames = _frames()
    expected = build_pig_strenet_artifacts(frames)
    identity = {"input_sha256": "fixture", "top_k_neighbors": 3}

    class InterruptAfterFirstSocialChunk(PigSTRENetCheckpointStore):
        def save_social_chunk(self, **kwargs: object) -> None:
            super().save_social_chunk(**kwargs)
            if kwargs["end_pair"] == 1:
                raise RuntimeError("synthetic interruption")

    interrupted = InterruptAfterFirstSocialChunk(
        tmp_path / "checkpoints",
        identity=identity,
        resume=False,
        social_chunk_pairs=1,
    )
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        build_pig_strenet_artifacts(
            frames,
            checkpoint_store=interrupted,
        )

    resumed_store = PigSTRENetCheckpointStore(
        tmp_path / "checkpoints",
        identity=identity,
        resume=True,
        social_chunk_pairs=1,
    )
    actual = build_pig_strenet_artifacts(
        frames,
        checkpoint_store=resumed_store,
    )

    for name in (
        "pair_manifest",
        "slot_manifest",
        "history_features",
        "roi_dynamics",
        "roi_visual_selection",
        "social_nodes",
        "social_edges",
        "control_matrix",
    ):
        pd.testing.assert_frame_equal(
            getattr(actual, name),
            getattr(expected, name),
        )
    assert actual.audit == expected.audit


def test_checkpoint_resume_rejects_authority_drift(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    PigSTRENetCheckpointStore(
        root,
        identity={"input_sha256": "left"},
        resume=False,
    )

    with pytest.raises(
        PigSTRENetCheckpointError,
        match="checkpoint identity mismatch",
    ):
        PigSTRENetCheckpointStore(
            root,
            identity={"input_sha256": "right"},
            resume=True,
        )


def test_legacy_actual_frame_coordinates_are_not_relative_coordinates() -> None:
    artifacts = build_pig_strenet_artifacts(_frames(actual_offset=100))

    pair = artifacts.pair_manifest.iloc[0]
    assert pair["history_start_relative"] == 0
    assert pair["history_end_relative"] == 5
    assert pair["target_start_relative"] == 6
    assert pair["target_end_relative"] == 11
    assert pair["history_window_start_frame"] == 100
    assert pair["history_window_end_frame"] == 105
    assert pair["target_window_start_frame"] == 106
    assert pair["target_window_end_frame"] == 111
    assert artifacts.slot_manifest.iloc[0]["frame_index"] == 100


def test_multiple_legacy_views_conserve_native_event_mass() -> None:
    artifacts = build_pig_strenet_artifacts(
        _frames(),
        legacy_target_starts=(6, 8),
    )

    assert len(artifacts.pair_manifest) == 4
    assert artifacts.pair_manifest["event_weight"].eq(0.5).all()
    mass = artifacts.pair_manifest.groupby("native_event_id")["event_weight"].sum()
    assert mass.eq(1.0).all()


def test_pig_strenet_pair_motion_resets_at_history_target_boundary() -> None:
    artifacts = build_pig_strenet_artifacts(_frames())
    pair_id = artifacts.pair_manifest.iloc[0]["pair_id"]
    rank_one = artifacts.social_edges.loc[
        artifacts.social_edges["pair_id"].eq(pair_id)
        & artifacts.social_edges["neighbor_rank"].eq(1)
    ].sort_values("slot_index")

    assert rank_one.loc[
        rank_one["slot_index"].isin([0, 6]),
        "pair_motion_energy_n_per_second2",
    ].eq(0.0).all()
    assert rank_one.loc[
        rank_one["slot_index"].isin([1, 7]),
        "pair_motion_energy_n_per_second2",
    ].gt(0.0).all()
    assert rank_one.iloc[-1]["pair_contact_duration_sec"] == pytest.approx(
        0.5
    )


def test_cvat_history_is_missing_without_prior_frames() -> None:
    artifacts = build_pig_strenet_artifacts(
        _frames(source_type="cvat_tracking_xml", missing_history=True)
    )

    assert len(artifacts.pair_manifest) == 2
    assert artifacts.pair_manifest["derived_view"].eq("cvat_target_6f").all()
    assert not artifacts.pair_manifest["history_complete"].any()
    assert artifacts.pair_manifest["history_available_ratio"].eq(0.0).all()
    assert artifacts.history_features["history_expected_frame_count"].eq(6).all()
    assert artifacts.history_features["history_gap_count"].eq(6).all()
    assert not artifacts.history_features[
        "history_target_transition_available"
    ].any()
    transition_columns = [
        "activity_delta_history_to_target",
        "speed_delta_history_to_target",
        "stationary_to_motion_score",
        "motion_to_stationary_score",
        "stationary_per_second_to_motion_score",
        "motion_to_stationary_per_second_score",
        "speed_n_per_second_delta_history_to_target",
        "distance_delta_history_to_target",
        "approach_to_contact_score",
        "contact_persistence_score",
        "contact_to_separation_score",
        "partner_change_count",
        "shape_change_history_to_target",
        "feeder_approach_to_engagement",
        "feeder_engagement_to_departure",
        "drinker_approach_to_engagement",
        "drinker_engagement_to_departure",
        "toy_approach_to_engagement",
        "toy_engagement_to_departure",
    ]
    assert artifacts.history_features[transition_columns].eq(0.0).all().all()
    assert "history_target_transition_available" in availability_columns(
        artifacts.history_features
    )
    assert "history_target_transition_available" not in model_x_columns(
        artifacts.history_features
    )
    xml_hr = artifacts.control_matrix.query("control_id == 'HR'")
    assert xml_hr["history_window_spec"].str.startswith("actual[").all()


def test_complete_history_enables_transition_features() -> None:
    frames = _frames(source_type="cvat_tracking_xml")
    for _, actor in frames.groupby("object_track_key", sort=False):
        ordered = actor.sort_values("relative_frame_index")
        history_end = float(
            ordered.loc[
                ordered["relative_frame_index"].eq(5),
                "cx_n",
            ].iloc[0]
        )
        target = ordered["relative_frame_index"].ge(6)
        relative = ordered.loc[target, "relative_frame_index"].to_numpy()
        frames.loc[ordered.index[target], "cx_n"] = (
            history_end + 0.04 * (relative - 5)
        )
    frames["speed_n_per_frame"] = 999.0
    frames["speed_n_per_second"] = 999.0
    artifacts = build_pig_strenet_artifacts(
        frames
    )

    assert artifacts.history_features[
        "history_target_transition_available"
    ].all()
    assert artifacts.history_features[
        "activity_n_per_second_delta_history_to_target"
    ].ne(0.0).all()
    assert artifacts.history_features[
        "speed_n_per_second_delta_history_to_target"
    ].tolist() == pytest.approx([0.18, 0.18])
    pair = artifacts.pair_manifest.iloc[0]
    assert pair["target_duration_sec"] == pytest.approx(1.0)
    assert pair["target_observed_timestamp_span_seconds"] == pytest.approx(
        5.0 / 6.0
    )


def test_segment_pair_aggregates_exclude_nonexistent_or_invalid_pairs() -> None:
    segment = pd.DataFrame(
        {
            "frame_index": list(range(6)),
            "timestamp_sec": [float(value) for value in range(6)],
            "source_fps": [1.0] * 6,
            "cx_n": [0.0, 1.0, 3.0, 6.0, 10.0, 15.0],
            "cy_n": [0.0] * 6,
            "bbox_valid": [True] * 6,
            "nearest_track_id": ["track-b"] * 6,
            "nearest_dist_n": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
            "pair_contact_with_nearest": [True] * 6,
        }
    )

    summary = _segment_summary(segment, "history", expected_count=6)

    assert summary[
        "history_tangential_acceleration_n_per_second2_abs_mean"
    ] == (
        pytest.approx(1.0)
    )
    assert summary["history_approach_ratio_per_second"] == pytest.approx(1.0)


def test_review_evidence_masks_missing_history_and_ignores_labels() -> None:
    artifacts = build_pig_strenet_artifacts(
        _frames(source_type="cvat_tracking_xml", missing_history=True)
    )
    pairs = artifacts.pair_manifest.copy()
    pairs["behavior_label_audit_only"] = ["fight", "sitting"]

    evidence, audit = build_pig_strenet_review_evidence(
        pairs,
        artifacts.history_features,
        roi_dynamics=artifacts.roi_dynamics,
        social_edges=artifacts.social_edges,
    )

    assert audit["valid"] is True
    assert audit["transition_invalid_pairs"] == 2
    assert not evidence["review_pig_history_transition_available"].any()
    assert evidence["review_pig_motion_transition_score"].eq(0.0).all()
    assert evidence["review_pig_social_phase_score"].eq(0.0).all()
    assert not any("behavior" in column for column in evidence.columns)


def test_review_evidence_attach_preserves_labels_and_rows() -> None:
    artifacts = build_pig_strenet_artifacts(
        _frames(source_type="cvat_tracking_xml")
    )
    evidence, _ = build_pig_strenet_review_evidence(
        artifacts.pair_manifest,
        artifacts.history_features,
        roi_dynamics=artifacts.roi_dynamics,
        social_edges=artifacts.social_edges,
    )
    units = artifacts.pair_manifest[
        ["temporal_unit_key", "behavior_label_audit_only"]
    ].rename(columns={"behavior_label_audit_only": "behavior_temporal_final"})

    attached = attach_pig_strenet_review_evidence(units, evidence)

    assert len(attached) == len(units)
    assert attached["behavior_temporal_final"].equals(
        units["behavior_temporal_final"]
    )
    assert set(PIG_REVIEW_EVIDENCE_COLUMNS).issubset(attached.columns)
    assert attached["review_pig_history_transition_available"].all()


def test_difference_maps_mask_missing_slots() -> None:
    crops = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    crops[1, 2:5, 2:5] = 255
    maps, summary, pair_valid = compute_stabilized_difference_maps(
        crops,
        np.array([True, True, False]),
    )

    assert maps.shape == (2, 8, 8)
    assert pair_valid.tolist() == [True, False]
    assert summary["pair_valid"].tolist() == [True, False]
    assert summary.loc[1, "diff_mean"] == 0.0
