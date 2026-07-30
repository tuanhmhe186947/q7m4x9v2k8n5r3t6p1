from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GUI_SCRIPT = (
    ROOT
    / "scripts"
    / "classification_v2"
    / "01_review_units_gui"
    / "review_final_behavior_gui_v1.py"
)
QUALITY_SCRIPT = GUI_SCRIPT.with_name("final_behavior_label_quality.py")
VIEW_SCRIPT = (
    ROOT
    / "scripts"
    / "classification_v2"
    / "01_review_units_gui"
    / "build_final_behavior_review_view.py"
)
SCOPE_SCRIPT = (
    ROOT
    / "scripts"
    / "classification_v2"
    / "01_review_units_gui"
    / "build_combined_final_behavior_review_scope.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_cvat_final_display_keeps_target_and_extended_context_separate() -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_cvat")
    unit = pd.Series(
        {
            "source_type": "cvat_tracking_xml",
            "display_frame_indices": "100,101,102,103,104,105",
            "review_pig_history_display_frame_indices": "94,95,96,97,98,99",
            "final_context_frame_indices": (
                "70,76,82,87,93,99,106,112,118,123,129,135"
            ),
            "final_playback_frame_indices": ",".join(
                str(frame) for frame in range(70, 136)
            ),
        }
    )

    displayed = module.final_display_frames(unit)

    assert displayed == sorted(set(displayed))
    assert set(range(100, 106)).issubset(displayed)
    assert 70 in displayed
    assert 135 in displayed
    assert module.BASE.decision_scope_complete(unit, displayed)
    assert module.final_playback_frames(unit) == list(range(70, 136))
    assert module.playback_frames_for_scope(
        unit,
        module.PLAYBACK_SCOPE_TARGET,
    ) == list(range(100, 106))
    assert module.playback_frames_for_scope(
        unit,
        module.PLAYBACK_SCOPE_CONTEXT,
    ) == list(range(70, 136))
    assert module.playback_frame_role(unit, 100) == "TARGET"
    assert module.playback_frame_role(unit, 99) == "CONTEXT"
    assert module.target_interval(unit) == (100, 105, 6)
    scene_keys = module.final_review_scene_frame_keys(
        pd.DataFrame([unit])
    )
    assert ("cvat_tracking_xml", "", "", 70) in scene_keys
    assert ("cvat_tracking_xml", "", "", 135) in scene_keys


def test_legacy_final_display_ignores_fabricated_context() -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_legacy")
    targets = list(range(2, 18))
    unit = pd.Series(
        {
            "source_type": "legacy_recovered",
            "display_frame_indices": ",".join(map(str, targets)),
            "review_pig_history_display_frame_indices": "0,1",
            "final_context_frame_indices": "18,19",
            "final_playback_frame_indices": ",".join(map(str, targets)),
        }
    )

    assert module.final_display_frames(unit) == targets
    assert module.final_playback_frames(unit) == targets
    assert module.playback_frames_for_scope(
        unit,
        module.PLAYBACK_SCOPE_TARGET,
    ) == targets


def test_final_summary_hides_selection_and_model_hints() -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_summary")
    unit = pd.Series(
        {
            "source_type": "cvat_tracking_xml",
            "behavior_label": "eat",
            "display_frame_indices": "10,11,12,13,14,15",
            "final_context_frame_indices": "4,5,6,16,17,18",
            "review_reason_codes": "behavior_evidence_conflict",
            "candidate_tier": "TIER_1_HARD_MANDATORY",
            "risk_score": 1.0,
        }
    )

    summary = module.format_final_summary(unit, [], 12)

    assert "Nhãn hiện tại (chưa xác nhận): eat" in summary
    assert "behavior_evidence_conflict" not in summary
    assert "TIER_1_HARD_MANDATORY" not in summary
    assert "risk_score" not in summary
    assert "score" not in summary.casefold()
    assert "DECISION TARGET: f10-f15 (6 frames)" in summary


def test_playback_status_keeps_target_and_context_bounds_visible() -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_playback_status")
    unit = pd.Series(
        {
            "source_type": "cvat_tracking_xml",
            "display_frame_indices": "100,101,102,103,104,105",
            "final_playback_frame_indices": ",".join(
                str(frame) for frame in range(70, 136)
            ),
        }
    )
    frames = module.playback_frames_for_scope(
        unit,
        module.PLAYBACK_SCOPE_TARGET,
    )

    status = module.format_playback_status(
        unit,
        frames,
        0,
        module.PLAYBACK_SCOPE_TARGET,
    )

    assert "current f100" in status
    assert "DECISION TARGET f100-f105 (6 frames)" in status
    assert "FULL CONTEXT f70-f135" in status


def test_resume_index_can_backtrack_from_next_unreviewed() -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_resume")
    review_ids = [f"item-{index}" for index in range(100)]
    decided = set(review_ids[:50])

    assert module.calculate_resume_index(review_ids, decided, 0) == 50
    assert module.calculate_resume_index(review_ids, decided, 7) == 43


def test_review_progress_keeps_cursor_and_completed_counts_distinct() -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_progress")

    assert module.format_review_progress(172, 2729, 637) == (
        "Đã hoàn tất: 637/2729 · Còn lại: 2092 · "
        "Mục đang mở (danh sách gốc): 172/2729"
    )


def test_scene_frame_loader_retains_all_actors_for_requested_frames(
    tmp_path: Path,
) -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_frame_filter")
    rows = [
        {
            "source_type": "cvat",
            "dataset_id": "set-a",
            "video_key": "video-a",
            "frame_index": 5,
            "pig_id": "pig-1",
            "x1": 1,
            "y1": 2,
            "x2": 11,
            "y2": 12,
        },
        {
            "source_type": "cvat",
            "dataset_id": "set-a",
            "video_key": "video-a",
            "frame_index": 5,
            "pig_id": "pig-2",
            "x1": 21,
            "y1": 22,
            "x2": 31,
            "y2": 32,
        },
        {
            "source_type": "cvat",
            "dataset_id": "set-a",
            "video_key": "video-a",
            "frame_index": 6,
            "pig_id": "pig-1",
            "x1": 1,
            "y1": 2,
            "x2": 11,
            "y2": 12,
        },
        {
            "source_type": "cvat",
            "dataset_id": "set-a",
            "video_key": "video-b",
            "frame_index": 5,
            "pig_id": "pig-3",
            "x1": 1,
            "y1": 2,
            "x2": 11,
            "y2": 12,
        },
    ]
    features_path = tmp_path / "frames.csv"
    pd.DataFrame(rows).to_csv(features_path, index=False)

    frames = module.BASE.load_gui_frame_features(
        features_path,
        scene_frame_keys={("cvat", "set-a", "video-a", 5)},
        chunk_rows=1,
    )

    assert frames["pig_id"].tolist() == ["pig-1", "pig-2"]
    assert frames["frame_index"].tolist() == [5, 5]


def test_final_frame_store_queries_only_requested_frames(
    tmp_path: Path,
) -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_frame_store")
    rows = [
        {
            "source_type": "cvat",
            "dataset_id": "set-a",
            "video_key": "video-a",
            "frame_index": 5,
            "pig_id": "pig-1",
            "x1": 1,
            "y1": 2,
            "x2": 11,
            "y2": 12,
        },
        {
            "source_type": "cvat",
            "dataset_id": "set-a",
            "video_key": "video-a",
            "frame_index": 5,
            "pig_id": "pig-2",
            "x1": 21,
            "y1": 22,
            "x2": 31,
            "y2": 32,
        },
        {
            "source_type": "cvat",
            "dataset_id": "set-a",
            "video_key": "video-a",
            "frame_index": 6,
            "pig_id": "pig-1",
            "x1": 1,
            "y1": 2,
            "x2": 11,
            "y2": 12,
        },
    ]
    features_path = tmp_path / "frames.csv"
    cache_path = tmp_path / "frames.sqlite3"
    pd.DataFrame(rows).to_csv(features_path, index=False)

    store = module.FinalBehaviorFrameStore(
        features_path,
        cache_path,
        scene_frame_keys={("cvat", "set-a", "video-a", 5)},
        chunk_rows=1,
    )
    selected = store.query(
        source_type="cvat",
        dataset_id="set-a",
        video_key="video-a",
        frame_indices=[5],
    )
    excluded = store.query(
        source_type="cvat",
        dataset_id="set-a",
        video_key="video-a",
        frame_indices=[6],
    )

    assert cache_path.is_file()
    assert selected["pig_id"].tolist() == ["pig-1", "pig-2"]
    assert excluded.empty


def test_final_frame_store_reuses_valid_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_frame_store_reuse")
    features_path = tmp_path / "frames.csv"
    cache_path = tmp_path / "frames.sqlite3"
    pd.DataFrame(
        [
            {
                "source_type": "cvat",
                "dataset_id": "set-a",
                "video_key": "video-a",
                "frame_index": 5,
                "pig_id": "pig-1",
                "x1": 1,
                "y1": 2,
                "x2": 11,
                "y2": 12,
            }
        ]
    ).to_csv(features_path, index=False)
    keys = {("cvat", "set-a", "video-a", 5)}
    module.FinalBehaviorFrameStore(
        features_path,
        cache_path,
        scene_frame_keys=keys,
        chunk_rows=1,
    )

    def fail_if_rescanned(*args, **kwargs):
        raise AssertionError("valid SQLite cache rescanned the source CSV")

    monkeypatch.setattr(
        module.BASE,
        "iter_gui_frame_feature_chunks",
        fail_if_rescanned,
    )
    reused = module.FinalBehaviorFrameStore(
        features_path,
        cache_path,
        scene_frame_keys=keys,
        chunk_rows=1,
    )

    assert reused.query(
        source_type="cvat",
        dataset_id="set-a",
        video_key="video-a",
        frame_indices=[5],
    )["pig_id"].tolist() == ["pig-1"]


def test_final_frame_store_rebuilds_when_requested_frames_change(
    tmp_path: Path,
) -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_frame_store_rebuild")
    features_path = tmp_path / "frames.csv"
    cache_path = tmp_path / "frames.sqlite3"
    pd.DataFrame(
        [
            {
                "source_type": "cvat",
                "dataset_id": "set-a",
                "video_key": "video-a",
                "frame_index": frame_index,
                "pig_id": f"pig-{frame_index}",
                "x1": 1,
                "y1": 2,
                "x2": 11,
                "y2": 12,
            }
            for frame_index in (5, 6)
        ]
    ).to_csv(features_path, index=False)
    module.FinalBehaviorFrameStore(
        features_path,
        cache_path,
        scene_frame_keys={("cvat", "set-a", "video-a", 5)},
        chunk_rows=1,
    )

    rebuilt = module.FinalBehaviorFrameStore(
        features_path,
        cache_path,
        scene_frame_keys={("cvat", "set-a", "video-a", 6)},
        chunk_rows=1,
    )

    assert rebuilt.query(
        source_type="cvat",
        dataset_id="set-a",
        video_key="video-a",
        frame_indices=[5],
    ).empty
    assert rebuilt.query(
        source_type="cvat",
        dataset_id="set-a",
        video_key="video-a",
        frame_indices=[6],
    )["pig_id"].tolist() == ["pig-6"]

    changed_rows = pd.read_csv(features_path)
    changed_rows.loc[
        changed_rows["frame_index"].eq(6),
        "pig_id",
    ] = "pig-6-source-changed"
    changed_rows.to_csv(features_path, index=False)
    source_rebuilt = module.FinalBehaviorFrameStore(
        features_path,
        cache_path,
        scene_frame_keys={("cvat", "set-a", "video-a", 6)},
        chunk_rows=1,
    )
    assert source_rebuilt.query(
        source_type="cvat",
        dataset_id="set-a",
        video_key="video-a",
        frame_indices=[6],
    )["pig_id"].tolist() == ["pig-6-source-changed"]


def test_rendered_media_retention_keeps_current_and_prefetched_items() -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_media_retention")
    cache = module.BASE.RenderedImageCache(max_items=3)
    image = module.Image.new("RGB", (8, 8), "white")
    cache.put("previous", image)
    cache.put("current", image)
    cache.put("next", image)

    cache.retain_only({"current", "next"})

    assert len(cache) == 2
    assert cache.get("previous") is None
    assert cache.get("current") is not None


def test_requested_review_index_requires_one_exact_match() -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_requested_start")
    review_ids = ["unit-a", "unit-b", "unit-c"]

    assert module.requested_review_index(review_ids, "") is None
    assert module.requested_review_index(review_ids, "unit-b") == 1

    try:
        module.requested_review_index(review_ids, "missing")
    except ValueError as exc:
        assert "matches=0" in str(exc)
    else:
        raise AssertionError("missing requested review unit was accepted")


def test_final_review_order_groups_date_video_actor_then_time() -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_order")
    units = pd.DataFrame(
        [
            {
                "review_unit_id": "late-video-a-track-2",
                "recording_date": "2019-11-30",
                "video_key": "video-a",
                "source_type": "cvat_tracking_xml",
                "dataset_id": "set-a",
                "object_track_key": "track-2",
                "track_id": "2",
                "pig_id": "pig-2",
                "unit_start_frame": 30,
                "unit_end_frame": 35,
                "review_priority": 999,
            },
            {
                "review_unit_id": "early-video-b-track-1",
                "recording_date": "2019-11-29",
                "video_key": "video-b",
                "source_type": "cvat_tracking_xml",
                "dataset_id": "set-a",
                "object_track_key": "track-1",
                "track_id": "1",
                "pig_id": "pig-1",
                "unit_start_frame": 10,
                "unit_end_frame": 15,
                "review_priority": 500,
            },
            {
                "review_unit_id": "early-video-a-track-2-later",
                "recording_date": "2019-11-29",
                "video_key": "video-a",
                "source_type": "cvat_tracking_xml",
                "dataset_id": "set-a",
                "object_track_key": "track-2",
                "track_id": "2",
                "pig_id": "pig-2",
                "unit_start_frame": 40,
                "unit_end_frame": 45,
                "review_priority": 100,
            },
            {
                "review_unit_id": "early-video-a-track-1",
                "recording_date": "2019-11-29",
                "video_key": "video-a",
                "source_type": "cvat_tracking_xml",
                "dataset_id": "set-a",
                "object_track_key": "track-1",
                "track_id": "1",
                "pig_id": "pig-1",
                "unit_start_frame": 20,
                "unit_end_frame": 25,
                "review_priority": 1,
            },
            {
                "review_unit_id": "early-video-a-track-2-earlier",
                "recording_date": "2019-11-29",
                "video_key": "video-a",
                "source_type": "cvat_tracking_xml",
                "dataset_id": "set-a",
                "object_track_key": "track-2",
                "track_id": "2",
                "pig_id": "pig-2",
                "unit_start_frame": 10,
                "unit_end_frame": 15,
                "review_priority": 200,
            },
        ]
    )

    ordered = module.order_final_review_units(units)

    assert ordered["review_unit_id"].tolist() == [
        "early-video-a-track-1",
        "early-video-a-track-2-earlier",
        "early-video-a-track-2-later",
        "early-video-b-track-1",
        "late-video-a-track-2",
    ]
    assert set(ordered.columns) == set(units.columns)
    assert len(ordered) == len(units)


def test_media_is_bounded_without_changing_source_pixels() -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_media_fit")
    source = module.Image.new("RGB", (1920, 1114), "red")

    fitted = module.fit_media_for_display(
        source,
        max_width=760,
        max_height=610,
    )

    assert fitted.width <= 760
    assert fitted.height <= 610
    assert source.size == (1920, 1114)
    assert fitted.getpixel((0, 0)) == (255, 0, 0)


def test_final_gui_contact_sheet_cache_avoids_rerender() -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_contact_sheet_cache")
    gui = module.FinalBehaviorReviewGui.__new__(module.FinalBehaviorReviewGui)
    gui.contact_sheet_cache = module.BASE.RenderedImageCache(max_items=2)
    gui.decisions = {"unit-1": {"manual_review_decision": "accept"}}
    frames = pd.DataFrame({"frame_index": [1, 2]})
    calls = {"render": 0}

    def render(unit, matched_frames):
        calls["render"] += 1
        return module.Image.new("RGB", (32, 24), "white"), ["diagnostic"]

    gui._frame_rows_for_unit = lambda unit: frames
    gui._make_contact_sheet = render
    unit = pd.Series({"review_unit_id": "unit-1"})

    first, first_diagnostics, first_count = gui._contact_sheet_for_unit(unit)
    first.paste("red", (0, 0, 1, 1))
    second, second_diagnostics, second_count = gui._contact_sheet_for_unit(unit)

    assert calls["render"] == 1
    assert first_diagnostics == second_diagnostics == ["diagnostic"]
    assert first_count == second_count == 2
    assert second.getpixel((0, 0)) == (255, 255, 255)
    assert gui.decisions == {"unit-1": {"manual_review_decision": "accept"}}


def test_final_gui_prefetch_restores_current_playback_rows() -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_prefetch_restore")
    gui = module.FinalBehaviorReviewGui.__new__(module.FinalBehaviorReviewGui)
    gui.units = pd.DataFrame([{"review_unit_id": "unit-2"}])
    gui._prefetch_after_id = "idle-1"
    previous_scene = pd.DataFrame({"frame_index": [10]})
    previous_actor = pd.DataFrame({"frame_index": [10]})
    gui._current_scene_rows = previous_scene
    gui._current_actor_rows = previous_actor

    def prepare(unit):
        gui._current_scene_rows = pd.DataFrame({"frame_index": [20]})
        gui._current_actor_rows = pd.DataFrame({"frame_index": [20]})

    gui._prepare_current_media_rows = prepare
    gui._contact_sheet_for_unit = lambda unit: (
        module.Image.new("RGB", (32, 24), "white"),
        [],
        1,
    )

    gui._prefetch_contact_sheet(0)

    assert gui._prefetch_after_id is None
    assert gui._current_scene_rows is previous_scene
    assert gui._current_actor_rows is previous_actor


def test_review_window_stays_inside_common_laptop_screen() -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_window_size")

    assert module.review_window_dimensions(1366, 768) == (1334, 700)
    assert module.review_window_dimensions(1920, 1080) == (1500, 940)


def test_supported_label_records_selector_false_positive_without_unresolved() -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_supported_quality")
    unit = {
        "review_unit_id": "non-a",
        "behavior_label": "stand",
        "final_scope_component": "ROI_DIRECTION_CORRECTED_NONINTERACTION",
    }
    decision = {
        "manual_review_decision": "accept",
        "manual_corrected_behavior": "",
        "manual_label_strength": "strong",
    }

    record = module.build_quality_record(unit, decision)

    assert record["label_status"] == "SUPPORTED"
    assert record["source_label_error_confirmed"] == "NO"
    assert record["selection_assessment"] == (
        "SELECTOR_FLAGGED_BUT_SOURCE_LABEL_SUPPORTED"
    )
    assert "UNRESOLVED" not in set(record.values())


def test_corrected_label_confirms_source_error_and_requires_pattern() -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_corrected_quality")
    unit = {
        "review_unit_id": "non-b",
        "behavior_label": "drink",
        "final_scope_component": "ROI_DIRECTION_CORRECTED_NONINTERACTION",
    }
    decision = {
        "manual_review_decision": "corrected",
        "manual_corrected_behavior": "explore",
        "manual_label_strength": "medium",
    }

    try:
        module.build_quality_record(unit, decision)
    except ValueError as exc:
        assert "clear-error pattern" in str(exc)
    else:
        raise AssertionError("corrected label accepted without error pattern")

    record = module.build_quality_record(
        unit,
        decision,
        error_pattern="ROI_PROXIMITY_ONLY_FALSE_POSITIVE",
    )
    assert record["label_status"] == "SOURCE_LABEL_ERROR_CONFIRMED"
    assert record["source_label_error_confirmed"] == "YES"
    assert record["reviewed_behavior"] == "explore"
    assert record["review_confidence"] == "MEDIUM"
    assert record["selection_assessment"] == (
        "SELECTOR_FLAG_CONFIRMED_SOURCE_LABEL_ERROR"
    )


def test_defer_is_workflow_only_and_exclude_is_technical_only() -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_nonlabel_states")
    quality_module = _load(
        QUALITY_SCRIPT,
        "final_behavior_label_quality_contract",
    )
    unit = {
        "review_unit_id": "fight-a",
        "behavior_label": "fight",
        "final_scope_component": (
            "POST_CALIBRATION_FULL_INTERACTION_CENSUS"
        ),
    }
    pending = {
        "manual_review_decision": "pending",
        "manual_corrected_behavior": "",
        "manual_label_strength": "strong",
    }
    excluded = {
        "manual_review_decision": "exclude",
        "manual_corrected_behavior": "",
        "manual_label_strength": "boundary",
    }

    assert module.build_quality_record(unit, pending) is None
    record = module.build_quality_record(unit, excluded)
    assert record["label_status"] == "TECHNICAL_DEFECT"
    assert record["source_label_error_confirmed"] == "NOT_APPLICABLE"
    assert record["error_pattern"] == (
        "TECHNICAL_MEDIA_OR_PRESENTATION_DEFECT"
    )
    assert set(quality_module.QUALITY_COLUMNS) == set(
        quality_module.MODEL_X_FORBIDDEN_COLUMNS
    )


def test_existing_accepts_are_derived_but_corrections_require_attribution() -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_quality_migration")
    gui = module.FinalBehaviorReviewGui.__new__(
        module.FinalBehaviorReviewGui
    )
    gui.units = pd.DataFrame(
        [
            {
                "review_unit_id": "accepted",
                "behavior_label": "stand",
                "final_scope_component": (
                    "ROI_DIRECTION_CORRECTED_NONINTERACTION"
                ),
            },
            {
                "review_unit_id": "corrected",
                "behavior_label": "drink",
                "final_scope_component": (
                    "ROI_DIRECTION_CORRECTED_NONINTERACTION"
                ),
            },
        ]
    )
    gui.decisions = {
        "accepted": {
            "manual_review_decision": "accept",
            "manual_corrected_behavior": "",
            "manual_label_strength": "strong",
        },
        "corrected": {
            "manual_review_decision": "corrected",
            "manual_corrected_behavior": "explore",
            "manual_label_strength": "strong",
        },
    }
    gui.label_quality_records = {}

    gui._derive_supported_quality_records()

    assert set(gui.label_quality_records) == {"accepted"}


def test_context_sampler_spans_available_window_deterministically() -> None:
    module = _load(VIEW_SCRIPT, "final_behavior_view_builder")
    values = list(range(70, 100))

    sampled = module._sample_evenly(values, 6)

    assert sampled == [70, 76, 82, 87, 93, 99]
    assert module._sample_evenly(values, 6) == sampled


def test_playback_window_is_continuous_and_context_sampler_is_sparse() -> None:
    module = _load(VIEW_SCRIPT, "final_behavior_view_playback")
    unit = pd.Series(
        {
            "source_type": "cvat_tracking_xml",
            "display_frame_indices": "100,101,102,103,104,105",
        }
    )
    available = list(range(1, 250))

    playback = module._playback_indices(unit, available)
    context = module._context_indices(unit, playback)

    assert playback == list(range(10, 196))
    assert context == [10, 28, 46, 63, 81, 99, 106, 124, 142, 159, 177, 195]
    assert not set(range(100, 106)).intersection(context)


def test_playback_frame_band_identifies_scope_without_altering_pixels() -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_playback_band")
    source = module.Image.new("RGB", (100, 10), "red")

    rendered = module.compose_playback_frame(
        source,
        frame_index=101,
        role="TARGET",
        target_start=100,
        target_end=105,
        playback_start=70,
        playback_end=135,
    )

    assert rendered.size == (100, 44)
    assert rendered.getpixel((0, 0)) == (255, 242, 204)
    assert rendered.getpixel((0, 34)) == (255, 0, 0)
    assert rendered.getpixel((48, 27)) == (192, 0, 0)


def test_current_media_rows_keep_actor_and_neutral_scene_context() -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_media_rows")
    gui = module.FinalBehaviorReviewGui.__new__(
        module.FinalBehaviorReviewGui
    )
    frame_rows = pd.DataFrame(
        [
            {
                "source_type": "cvat_tracking_xml",
                "dataset_id": "set-a",
                "video_key": "video-a",
                "pig_id": "pig-1",
                "object_track_key": "actor",
                "track_id": "1",
                "frame_index": 100,
            },
            {
                "source_type": "cvat_tracking_xml",
                "dataset_id": "set-a",
                "video_key": "video-a",
                "pig_id": "pig-1",
                "object_track_key": "actor",
                "track_id": "1",
                "frame_index": 101,
            },
            {
                "source_type": "cvat_tracking_xml",
                "dataset_id": "set-a",
                "video_key": "video-a",
                "pig_id": "pig-2",
                "object_track_key": "neighbor",
                "track_id": "2",
                "frame_index": 100,
            },
        ]
    )

    class FrameStoreStub:
        def query(self, **kwargs):
            del kwargs
            return frame_rows.copy()

    gui.frame_store = FrameStoreStub()
    unit = pd.Series(
        {
            "source_type": "cvat_tracking_xml",
            "dataset_id": "set-a",
            "video_key": "video-a",
            "pig_id": "pig-1",
            "object_track_key": "actor",
            "track_id": "1",
        }
    )

    gui._prepare_current_media_rows(unit)

    assert len(gui._current_scene_rows) == 3
    assert gui._current_actor_rows["frame_index"].tolist() == [100, 101]


def test_combined_scope_is_corrected_noninteraction_plus_full_interaction() -> None:
    module = _load(SCOPE_SCRIPT, "combined_final_behavior_scope")
    candidates = pd.DataFrame(
        [
            {
                "review_unit_id": "non-a",
                "review_template": "motion",
                "requires_partner_context": False,
                "behavior_label": "stand",
            },
            {
                "review_unit_id": "non-b",
                "review_template": "roi",
                "requires_partner_context": False,
                "behavior_label": "drink",
            },
            {
                "review_unit_id": "fight-a",
                "review_template": "interaction",
                "requires_partner_context": True,
                "behavior_label": "fight",
            },
            {
                "review_unit_id": "social-a",
                "review_template": "interaction",
                "requires_partner_context": True,
                "behavior_label": "social-nose",
            },
        ]
    )
    corrected = pd.DataFrame({"review_unit_id": ["non-b"]})

    combined, audit = module.build_combined_scope(candidates, corrected)

    assert combined["review_unit_id"].tolist() == [
        "non-b",
        "fight-a",
        "social-a",
    ]
    assert combined["final_scope_component"].tolist() == [
        "ROI_DIRECTION_CORRECTED_NONINTERACTION",
        "POST_CALIBRATION_FULL_INTERACTION_CENSUS",
        "POST_CALIBRATION_FULL_INTERACTION_CENSUS",
    ]
    assert audit["component_overlap_count"] == 0
    assert audit["full_interaction_census_count"] == 2


def test_combined_scope_rejects_inconsistent_interaction_partition() -> None:
    module = _load(SCOPE_SCRIPT, "combined_scope_inconsistent_partition")
    candidates = pd.DataFrame(
        [
            {
                "review_unit_id": "fight-a",
                "review_template": "interaction",
                "requires_partner_context": False,
                "behavior_label": "fight",
            }
        ]
    )
    corrected = pd.DataFrame({"review_unit_id": []})

    try:
        module.build_combined_scope(candidates, corrected)
    except ValueError as exc:
        assert "partitions differ" in str(exc)
    else:
        raise AssertionError("inconsistent interaction partition accepted")


def test_combined_scope_requires_full_census_calibration_decision() -> None:
    module = _load(SCOPE_SCRIPT, "combined_scope_calibration_decision")
    decision = {
        "post_calibration_decision": module.FULL_CENSUS_DECISION,
        "selected_rule_id": None,
        "confirmation_authorized": False,
        "ledger_sha256": "ledger-hash",
    }

    module.validate_full_census_decision(
        decision,
        calibration_ledger_sha256="ledger-hash",
    )
    decision["post_calibration_decision"] = (
        "DECISION_A_KEEP_CURRENT_991_WITH_REVALIDATION"
    )
    try:
        module.validate_full_census_decision(
            decision,
            calibration_ledger_sha256="ledger-hash",
        )
    except ValueError as exc:
        assert "not full census" in str(exc)
    else:
        raise AssertionError("selective interaction decision accepted")
