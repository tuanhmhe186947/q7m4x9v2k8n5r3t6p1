"""Contracts for the frozen State 8 development evaluator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "tracking"
    / "evaluate_reconciled_methods_standard_v2.py"
)
SPEC = importlib.util.spec_from_file_location("state8_evaluator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_transfer_decision_rule_is_predeclared() -> None:
    assert MODULE.AGGREGATE_METRIC_COLUMNS["idp"] == "id_precision"
    assert MODULE.AGGREGATE_METRIC_COLUMNS["idr"] == "id_recall"
    realtime = {
        "hota": 0.8,
        "idf1": 0.8,
        "idsw_standard": 10.0,
        "wrong_id_matched_frames": 20.0,
    }
    assert (
        MODULE.classify_transfer_signal(
            realtime,
            {
                "hota": 0.81,
                "idf1": 0.8,
                "idsw_standard": 9.0,
                "wrong_id_matched_frames": 20.0,
            },
            harmful_changes=0,
            final_changed_rows=1,
        )
        == "TRANSFER_SIGNAL_POSITIVE"
    )
    assert (
        MODULE.classify_transfer_signal(
            realtime,
            {
                "hota": 0.79,
                "idf1": 0.79,
                "idsw_standard": 11.0,
                "wrong_id_matched_frames": 21.0,
            },
            harmful_changes=1,
            final_changed_rows=1,
        )
        == "TRANSFER_DEGRADES_RF"
    )
    assert (
        MODULE.classify_transfer_signal(
            realtime,
            {
                "hota": 0.79,
                "idf1": 0.79,
                "idsw_standard": 9.0,
                "wrong_id_matched_frames": 21.0,
            },
            harmful_changes=1,
            final_changed_rows=1,
        )
        == "TRANSFER_SIGNAL_MIXED"
    )


def test_contiguous_episode_count_uses_video_track_boundaries() -> None:
    keys = {
        ("video_a", 1, "Pig_1"),
        ("video_a", 2, "Pig_1"),
        ("video_a", 4, "Pig_1"),
        ("video_a", 2, "Pig_2"),
        ("video_b", 1, "Pig_1"),
    }
    assert MODULE.contiguous_episode_count(keys) == 4


def test_state8_tool_cannot_run_detector_tracker_or_current_b1() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert ".track(" not in source
    assert ".predict(" not in source
    assert "run_tracking(" not in source
    assert "ReplayDetector" not in source
    assert "B1_hybrid_bytetrack" not in source
    assert "unseen" not in source.lower() or "unseen_files_accessed" in source
