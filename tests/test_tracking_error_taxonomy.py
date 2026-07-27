from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "tracking"
    / "build_rf_acc23_error_taxonomy.py"
)
SPEC = importlib.util.spec_from_file_location("rf_acc23_taxonomy", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_grouping_merges_pair_swap_and_conserves_rows() -> None:
    rows = pd.DataFrame(
        [
            {"video_stem": "v", "frame": 10, "gt_id": "A", "pred_id": "B",
             "event": "id_mismatch"},
            {"video_stem": "v", "frame": 10, "gt_id": "B", "pred_id": "A",
             "event": "id_switch_and_mismatch"},
            {"video_stem": "v", "frame": 11, "gt_id": "A", "pred_id": "B",
             "event": "id_mismatch"},
            {"video_stem": "v", "frame": 12, "gt_id": "A", "pred_id": "A",
             "previous_pred_id": "B", "event": "id_switch"},
            {"video_stem": "v", "frame": 40, "gt_id": "A", "pred_id": "B",
             "event": "id_mismatch"},
        ]
    )

    events = MODULE.group_error_events(rows)

    assert len(events) == 2
    assert sum(item["wrong_id_matched_frames"] for item in events) == 4
    assert events[0]["start_frame"] == 10
    assert events[0]["end_frame"] == 11
    assert events[0]["wrong_id_matched_frames"] == 3
    assert events[0]["id_switch_rows"] == 2


def test_grouping_requires_shared_identity_to_bridge_frames() -> None:
    rows = pd.DataFrame(
        [
            {"video_stem": "v", "frame": 10, "gt_id": "A", "pred_id": "B",
             "event": "id_mismatch"},
            {"video_stem": "v", "frame": 11, "gt_id": "C", "pred_id": "D",
             "event": "id_mismatch"},
        ]
    )

    assert len(MODULE.group_error_events(rows)) == 2


def test_classification_prioritizes_gt_authority_and_birth() -> None:
    ambiguous = {
        "gt_authority": "UNRESOLVED_SOURCE_AUTHORITY",
        "start_frame": 0,
    }
    birth = {
        "gt_authority": "SUFFICIENT_FOR_EVENT_TAXONOMY",
        "start_frame": 0,
    }

    assert MODULE.classify_event(ambiguous)[0] == "GT_OR_EVALUATION_AMBIGUITY"
    assert MODULE.classify_event(birth)[0] == "TRACK_BIRTH_OR_DUPLICATE_TRACK"


def test_classification_uses_measured_hidden_overlap_evidence() -> None:
    event = {
        "gt_authority": "SUFFICIENT_FOR_EVENT_TAXONOMY",
        "start_frame": 100,
        "hidden_count_max": 1,
        "overlap_pair_count_max": 2,
        "missing_detection_count_max": 1,
    }

    primary, secondary = MODULE.classify_event(event)

    assert primary == "OCCLUSION_OWNER_LOSS"
    assert secondary == ["DETECTION_MISS_OR_DROPOUT"]


def test_output_root_refuses_overwrite(tmp_path: Path) -> None:
    with pytest.raises(MODULE.AuditError, match="refusing to overwrite"):
        MODULE.build_outputs(
            MODULE.AuditInputs(
                evaluation_root=tmp_path,
                prediction_root=tmp_path,
                shadow_pairs=tmp_path / "pairs.csv",
                output_root=tmp_path,
            )
        )
