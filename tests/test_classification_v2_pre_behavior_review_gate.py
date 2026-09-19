from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.features.sequence_windows import (
    audit_sequence_windows,
    build_sequence_windows,
)
from pig_behavior.classification_v2.review.review_authority import (
    OFFICIAL_SCOPE,
    SMOKE_SCOPE,
    build_review_authority_manifest,
)
from pig_behavior.classification_v2.sources.temporal_provenance import (
    CANONICAL_TIMESTAMP_SOURCE,
)

CODE_SHA = "4" * 40


def _reviewed_rows(
    *,
    source_type: str,
    frames: list[int],
    labels: list[str] | None = None,
) -> pd.DataFrame:
    labels = labels or ["move"] * len(frames)
    if len(labels) != len(frames):
        raise ValueError("labels must align with frames")
    legacy = source_type == "legacy_recovered"
    unit_keys = [
        "legacy-unit" if legacy else f"cvat-unit-{frame // 6}"
        for frame in frames
    ]
    return pd.DataFrame(
        {
            "source_type": [source_type] * len(frames),
            "dataset_id": ["legacy" if legacy else "cvat"] * len(frames),
            "video_key": ["video-a"] * len(frames),
            "frame_uid": [f"video-a::f{frame:06d}" for frame in frames],
            "frame_index": frames,
            "source_frame_index": frames,
            "source_fps": [30.0] * len(frames),
            "timestamp_sec": [frame / 30.0 for frame in frames],
            "timestamp_source": [CANONICAL_TIMESTAMP_SOURCE] * len(frames),
            "relative_frame_index": [
                frame - frames[0] if legacy else frame % 6
                for frame in frames
            ],
            "native_offset": [
                frame - frames[0] if legacy else frame % 6
                for frame in frames
            ],
            "pig_id": ["ID_1"] * len(frames),
            "track_id": ["1"] * len(frames),
            "object_track_key": ["track-a"] * len(frames),
            "temporal_unit_key": unit_keys,
            "behavior": labels,
            "behavior_temporal_final": labels,
            "bbox_valid": [True] * len(frames),
            "hidden": ["No"] * len(frames),
            "hidden_is_trusted": [True] * len(frames),
            "hidden_review_status": ["reviewed"] * len(frames),
            "cx_n": [frame / 100.0 for frame in frames],
            "cy_n": [0.5] * len(frames),
            "bw_n": [0.2] * len(frames),
            "bh_n": [0.1] * len(frames),
            "area_n": [0.02] * len(frames),
            "aspect_ratio": [2.0] * len(frames),
            "spatiotemporal_feature_valid": [True] * len(frames),
            "feature_computation_grain": ["FRAME_LOCAL_PRIMITIVES"]
            * len(frames),
            "pair_scope_key": [""] * len(frames),
            "behavior_review_decision_present": [True] * len(frames),
            "behavior_review_label_resolved": [True] * len(frames),
            "behavior_review_include_in_training": [True] * len(frames),
            "behavior_reviewed_final": labels,
        }
    )


@pytest.mark.parametrize(
    ("length", "expected_pairs", "expected_span"),
    [
        (6, 5, 5.0 / 30.0),
        (8, 7, 7.0 / 30.0),
        (12, 11, 11.0 / 30.0),
        (16, 15, 15.0 / 30.0),
    ],
)
def test_contiguous_exact_views_recompute_expected_pairs(
    length: int,
    expected_pairs: int,
    expected_span: float,
) -> None:
    rows = _reviewed_rows(
        source_type="cvat_tracking_xml",
        frames=list(range(18)),
    )
    _, _, windows = build_sequence_windows(
        rows,
        window_lengths=[length],
        behavior_review_requirement="full_native_unit_review_required",
    )
    window = windows.sort_values("window_start_frame").iloc[0]

    assert window["selected_frame_indices"] == json.dumps(
        list(range(length)),
        separators=(",", ":"),
    )
    assert json.loads(window["pair_delta_frames"]) == [1] * expected_pairs
    assert np.asarray(
        json.loads(window["pair_delta_seconds"]),
    ) == pytest.approx(np.full(expected_pairs, 1.0 / 30.0))
    assert window["adjacent_motion_pair_count_window"] == expected_pairs
    assert window["physical_span_seconds"] == pytest.approx(expected_span)
    assert bool(window["pair_recomputed_for_view"]) is True
    assert bool(window["aggregate_recomputed_for_view"]) is True
    assert window["pair_scope_key"] == window["window_id"]
    assert bool(window["window_valid_for_main_train"]) is True
    assert audit_sequence_windows(windows)["errors"] == []


def test_t6_starting_mid_track_excludes_pre_window_pair() -> None:
    rows = _reviewed_rows(
        source_type="legacy_recovered",
        frames=list(range(16)),
    )
    _, _, windows = build_sequence_windows(
        rows,
        window_lengths=[6],
        legacy_window_stride=1,
        behavior_review_requirement="full_native_unit_review_required",
    )
    window = windows.loc[windows["window_start_frame"].eq(2)].iloc[0]

    assert window["selected_frame_indices"] == "[2,3,4,5,6,7]"
    assert window["adjacent_motion_pair_count_window"] == 5
    assert window["path_length_n_window"] == pytest.approx(0.05)
    assert window["displacement_n_window"] == pytest.approx(0.05)


def test_cvat_t8_cross_unit_pair_is_recomputed_not_native_aggregate() -> None:
    rows = _reviewed_rows(
        source_type="cvat_tracking_xml",
        frames=list(range(12)),
    )
    rows["path_length_n_unit"] = 99_999.0
    rows["speed_mean_unit"] = 99_999.0
    _, _, windows = build_sequence_windows(
        rows,
        window_lengths=[8],
        behavior_review_requirement="full_native_unit_review_required",
    )
    window = windows.sort_values("window_start_frame").iloc[0]

    assert window["num_temporal_units_window"] == 2
    assert window["adjacent_motion_pair_count_window"] == 7
    assert window["path_length_n_window"] == pytest.approx(0.07)
    assert window["speed_n_per_second_mean_window"] == pytest.approx(0.3)
    assert window["path_length_n_window"] != 99_999.0


def test_cvat_behavior_boundary_and_pending_unit_fail_closed() -> None:
    labels = ["move"] * 6 + ["stand"] * 6
    transition = _reviewed_rows(
        source_type="cvat_tracking_xml",
        frames=list(range(12)),
        labels=labels,
    )
    _, _, transition_windows = build_sequence_windows(
        transition,
        window_lengths=[8],
        behavior_review_requirement="full_native_unit_review_required",
    )
    transition_window = transition_windows.iloc[0]
    assert bool(transition_window["window_valid_for_main_train"]) is False
    assert transition_window[
        "human_reviewed_behavior_consistency_status"
    ] == "transition"
    assert "behavior_review_transition" in transition_window[
        "window_exclusion_reason"
    ]

    pending = _reviewed_rows(
        source_type="cvat_tracking_xml",
        frames=list(range(12)),
    )
    pending.loc[
        pending["frame_index"].ge(6),
        "behavior_review_decision_present",
    ] = False
    _, _, pending_windows = build_sequence_windows(
        pending,
        window_lengths=[8],
        behavior_review_requirement="full_native_unit_review_required",
    )
    pending_window = pending_windows.iloc[0]
    assert bool(pending_window["window_valid_for_main_train"]) is False
    assert bool(pending_window["all_temporal_units_behavior_reviewed"]) is False


def test_hidden_invalid_frame_excludes_main_window() -> None:
    rows = _reviewed_rows(
        source_type="cvat_tracking_xml",
        frames=list(range(12)),
    )
    rows.loc[3, "hidden"] = "Yes"
    rows.loc[3, "bbox_valid"] = False
    _, _, windows = build_sequence_windows(
        rows,
        window_lengths=[8],
        behavior_review_requirement="full_native_unit_review_required",
    )
    window = windows.iloc[0]

    assert window["bbox_valid_ratio_window"] == pytest.approx(7.0 / 8.0)
    assert window["hidden_ratio_trusted_window"] == pytest.approx(1.0 / 8.0)
    assert bool(window["window_valid_for_main_train"]) is False
    assert "bbox_valid_ratio_below_threshold" in window[
        "window_exclusion_reason"
    ]


def test_s6_at16_has_sparse_identity_and_real_delta_time() -> None:
    rows = _reviewed_rows(
        source_type="legacy_recovered",
        frames=list(range(16)),
    )
    _, _, windows = build_sequence_windows(
        rows,
        window_lengths=[16],
        behavior_review_requirement="full_native_unit_review_required",
        include_legacy_sparse_s6_at16=True,
    )
    sparse = windows.loc[windows["view_type"].eq("S6@16")].iloc[0]
    contiguous = windows.loc[
        windows["view_type"].eq("T16_contiguous")
    ].iloc[0]

    assert sparse["sampling_pattern"] == (
        "uniform_sparse_offsets_0_3_6_9_12_15"
    )
    assert json.loads(sparse["pair_delta_frames"]) == [3] * 5
    assert np.asarray(
        json.loads(sparse["pair_delta_seconds"]),
    ) == pytest.approx(np.full(5, 3.0 / 30.0))
    assert sparse["physical_span_seconds"] == pytest.approx(15.0 / 30.0)
    assert sparse["adjacent_motion_pair_count_window"] == 0
    assert sparse["sparse_velocity_pair_count_window"] == 5
    assert sparse["path_length_n_window"] == 0.0
    assert sparse["sparse_path_length_n_window"] == pytest.approx(0.15)
    assert contiguous["path_length_n_window"] == pytest.approx(0.15)
    assert bool(sparse["primary_cross_source_eligible"]) is False


def test_final_view_rejects_imported_final_aggregate() -> None:
    rows = _reviewed_rows(
        source_type="cvat_tracking_xml",
        frames=list(range(6)),
    )
    rows["feature_computation_grain"] = "FINAL_VIEW_FEATURES"
    rows["pair_scope_key"] = "another-window"
    rows["pair_recomputed_for_view"] = True
    rows["aggregate_recomputed_for_view"] = True

    with pytest.raises(ValueError, match="another final-view artifact"):
        build_sequence_windows(
            rows,
            window_lengths=[6],
            behavior_review_requirement="full_native_unit_review_required",
        )


def test_review_authority_manifest_is_deterministic_and_media_bound(
    tmp_path: Path,
) -> None:
    source_artifacts, artifacts = _authority_files(tmp_path)
    kwargs = {
        "code_authority_sha": CODE_SHA,
        "code_dirty": False,
        "lineage_id": "c2v2_human_review_20260722_reviewer01_v6",
        "authority_scope": OFFICIAL_SCOPE,
        "source_artifacts": source_artifacts,
        "artifacts": artifacts,
        "timestamp_fps_contract": _timestamp_contract(),
        "evidence_semantics": _evidence_semantics(),
    }
    first = build_review_authority_manifest(**kwargs)
    second = build_review_authority_manifest(**kwargs)

    assert first["valid"] is True
    assert first["authorizes_behavior_gui"] is True
    assert first["review_authority_sha256"] == second[
        "review_authority_sha256"
    ]
    assert first["behavior_review_unit_identity"]["units"] == 2

    dirty = build_review_authority_manifest(
        **{**kwargs, "code_dirty": True}
    )
    assert dirty["valid"] is False
    assert dirty["authorizes_behavior_gui"] is False
    assert "official_authority_requires_clean_code" in dirty["errors"]

    pd.DataFrame({"frame_uid": ["different-media"]}).to_csv(
        artifacts["media_authority"],
        index=False,
    )
    changed = build_review_authority_manifest(**kwargs)
    assert changed["valid"] is True
    assert changed["review_authority_sha256"] != first[
        "review_authority_sha256"
    ]


def test_review_authority_rejects_pair_columns_and_stopped_v3(
    tmp_path: Path,
) -> None:
    stopped = tmp_path / "c2v2_human_review_20260721_reviewer01_v3"
    stopped.mkdir()
    source_artifacts, artifacts = _authority_files(stopped)
    frame_local = pd.read_csv(artifacts["frame_local"])
    frame_local["speed_n_per_second"] = 1.0
    frame_local.to_csv(artifacts["frame_local"], index=False)

    manifest = build_review_authority_manifest(
        code_authority_sha=CODE_SHA,
        code_dirty=False,
        lineage_id="c2v2_human_review_20260722_reviewer01_v6",
        authority_scope=OFFICIAL_SCOPE,
        source_artifacts=source_artifacts,
        artifacts=artifacts,
        timestamp_fps_contract=_timestamp_contract(),
        evidence_semantics=_evidence_semantics(),
    )

    assert manifest["valid"] is False
    assert manifest["authorizes_behavior_gui"] is False
    assert any(
        error.startswith("frame_local_contains_pair_or_aggregate")
        for error in manifest["errors"]
    )
    assert any(
        error.startswith("official_authority_references_stopped_v3")
        for error in manifest["errors"]
    )

    smoke = build_review_authority_manifest(
        code_authority_sha=CODE_SHA,
        code_dirty=True,
        lineage_id="pre_behavior_review_smoke",
        authority_scope=SMOKE_SCOPE,
        source_artifacts=source_artifacts,
        artifacts={
            **artifacts,
            "frame_local": _clean_frame_local(stopped),
        },
        timestamp_fps_contract=_timestamp_contract(),
        evidence_semantics=_evidence_semantics(),
    )
    assert smoke["valid"] is True
    assert smoke["official_review_authority"] is False
    assert smoke["authorizes_behavior_gui"] is False


def _authority_files(
    root: Path,
) -> tuple[dict[str, Path], dict[str, Path]]:
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source.csv"
    pd.DataFrame({"source_frame_index": [0, 1]}).to_csv(source, index=False)
    frame_local = _clean_frame_local(root)
    hidden = root / "hidden.csv"
    harmonized = root / "harmonized.csv"
    temporal = root / "temporal.csv"
    pig = root / "pig.json"
    review = root / "review.csv"
    media = root / "media.csv"
    timestamp = root / "timestamp.json"
    semantics = root / "semantics.json"
    pd.DataFrame({"frame_uid": ["f0", "f1"], "hidden": ["No", "No"]}).to_csv(
        hidden,
        index=False,
    )
    pd.DataFrame({"frame_uid": ["f0", "f1"], "behavior": ["move", "move"]}).to_csv(
        harmonized,
        index=False,
    )
    pd.DataFrame(
        {
            "temporal_unit_key": ["u0", "u1"],
            "label_window_start": [0, 6],
            "label_window_end": [5, 11],
            "source_type": ["cvat_tracking_xml"] * 2,
            "video_key": ["v"] * 2,
            "object_track_key": ["t"] * 2,
        }
    ).to_csv(temporal, index=False)
    pig.write_text(
        json.dumps({"schema_version": "pig.v1", "rows": 2}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "review_unit_id": ["u0", "u1"],
            "unit_start_frame": [0, 6],
            "unit_end_frame": [5, 11],
            "source_type": ["cvat_tracking_xml"] * 2,
            "video_key": ["v"] * 2,
            "object_track_key": ["t"] * 2,
        }
    ).to_csv(review, index=False)
    pd.DataFrame(
        {
            "frame_uid": ["f0", "f1"],
            "crop_path": ["c0.jpg", "c1.jpg"],
            "source_video_path": ["v.mp4", "v.mp4"],
        }
    ).to_csv(media, index=False)
    timestamp.write_text(json.dumps(_timestamp_contract()), encoding="utf-8")
    semantics.write_text(json.dumps(_evidence_semantics()), encoding="utf-8")
    return {"source_frames": source}, {
        "frame_local": frame_local,
        "hidden_reviewed_frames": hidden,
        "harmonized_frames": harmonized,
        "temporal_native_units": temporal,
        "pig_strenet_evidence": pig,
        "behavior_review_units": review,
        "media_authority": media,
        "timestamp_fps_contract": timestamp,
        "evidence_semantics": semantics,
    }


def _clean_frame_local(root: Path) -> Path:
    path = root / "frame_local.csv"
    pd.DataFrame(
        {
            "feature_computation_grain": ["FRAME_LOCAL_PRIMITIVES"] * 2,
            "frame_uid": ["f0", "f1"],
            "source_frame_index": [0, 1],
            "timestamp_sec": [0.0, 1.0 / 30.0],
            "cx_n": [0.1, 0.2],
            "nearest_pair_iou": [0.0, 0.0],
        }
    ).to_csv(path, index=False)
    return path


def _timestamp_contract() -> dict[str, object]:
    return {
        "source_fps": 30.0,
        "formula": "timestamp_sec=source_frame_index/source_fps",
        "valid": True,
        "errors": [],
    }


def _evidence_semantics() -> dict[str, object]:
    return {
        "evidence_column_semantic_version": "classification_v2.review.v1",
        "feature_grain": "NATIVE_UNIT_REVIEW_EVIDENCE",
        "valid": True,
        "errors": [],
    }
