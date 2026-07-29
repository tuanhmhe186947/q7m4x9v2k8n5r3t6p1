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
VIEW_SCRIPT = (
    ROOT
    / "scripts"
    / "classification_v2"
    / "01_review_units_gui"
    / "build_final_behavior_review_view.py"
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
    assert module.playback_frame_role(unit, 100) == "TARGET"
    assert module.playback_frame_role(unit, 99) == "CONTEXT"


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


def test_resume_index_can_backtrack_from_next_unreviewed() -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_resume")
    review_ids = [f"item-{index}" for index in range(100)]
    decided = set(review_ids[:50])

    assert module.calculate_resume_index(review_ids, decided, 0) == 50
    assert module.calculate_resume_index(review_ids, decided, 7) == 43


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
    source = module.Image.new("RGB", (20, 10), "red")

    rendered = module.compose_playback_frame(
        source,
        frame_index=101,
        role="TARGET",
    )

    assert rendered.size == (20, 44)
    assert rendered.getpixel((0, 0)) == (255, 242, 204)
    assert rendered.getpixel((0, 34)) == (255, 0, 0)


def test_current_media_rows_keep_actor_and_neutral_scene_context() -> None:
    module = _load(GUI_SCRIPT, "final_behavior_gui_media_rows")
    gui = module.FinalBehaviorReviewGui.__new__(
        module.FinalBehaviorReviewGui
    )
    gui.frames = pd.DataFrame(
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
