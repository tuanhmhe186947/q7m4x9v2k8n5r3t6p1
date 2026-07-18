from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".agents"
    / "skills"
    / "tracking-experiment-guardian"
    / "scripts"
    / "replay_post_video_identity.py"
)
SPEC = importlib.util.spec_from_file_location(
    "tracking_identity_replay_skill",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
IDENTITY_REPLAY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = IDENTITY_REPLAY
SPEC.loader.exec_module(IDENTITY_REPLAY)


def _shape(
    frame: int,
    fixed_id: int,
    *,
    id_value: str,
    hidden: bool = False,
    score: float = 0.9,
    points: list[float] | None = None,
) -> dict:
    return {
        "label": f"Pig_{fixed_id}",
        "frame": frame,
        "points": points or [0.0, 0.0, 12.0, 12.0],
        "outside": False,
        "occluded": hidden,
        "score": score,
        "source": "file",
        "attributes": [
            {"name": "ID", "value": id_value},
            {"name": "Behavior", "value": "lying"},
            {"name": "Hidden", "value": "Yes" if hidden else "No"},
        ],
        "elements": [],
    }


def _old_repair_parent() -> list[dict]:
    shapes = []
    for frame in range(9, 16):
        repaired = frame >= 10
        partner = _shape(
            frame,
            1,
            id_value="ID_2" if repaired else "ID_1",
            points=[0.0, 0.0, 12.0, 12.0],
        )
        hidden = _shape(
            frame,
            2,
            id_value="ID_1" if repaired else "ID_2",
            hidden=frame in {10, 11},
            score=0.3 if frame in {10, 11} else 0.9,
            points=[1.0, 1.0, 11.0, 11.0],
        )
        shapes.extend([partner, hidden])
    return shapes


def _id_by_key(shapes: list[dict]) -> dict[tuple[str, int], str]:
    return {
        (shape["label"], shape["frame"]): IDENTITY_REPLAY.id_value(shape)
        for shape in shapes
    }


def _config() -> object:
    return IDENTITY_REPLAY.ReplayConfig(
        min_hidden_frames=2,
        max_hidden_frames=3,
        min_overlap_iou=0.4,
        max_hidden_median_score=0.5,
        start_back_frames=1,
        min_suffix_frames=6,
    )


def _valid_plan() -> dict:
    return {
        "schema_version": "tracking_h5_hidden_suffix_commit_plan_v1",
        "status": "frozen_before_replay",
        "priority": "hybrid_bytetrack_residual_first",
        "candidate": {
            "name": "hidden_suffix_commit_after_run_v1",
            "gt_used_to_generate_prediction": False,
            "video_or_frame_hardcode_allowed": False,
            "parameters": {
                "min_hidden_frames": 8,
                "max_hidden_frames": 15,
                "min_overlap_iou": 0.7,
                "max_hidden_median_score": 0.5,
                "start_back_frames": 7,
                "min_suffix_frames": 600,
            },
        },
        "identity_replay_contract": {
            "allowed_payload_change": ["ID attribute"],
            "geometry_replay_contract_allowed": False,
        },
        "evaluation_contract": {
            "include_hidden": True,
            "hidden_is_optimization_target": False,
            "rule_combo": "iou0_area0_condarea0_merge0",
            "generated_mp4_allowed": False,
            "classification_scope_allowed": False,
        },
    }


def test_identity_replay_delays_commit_until_after_hidden_run() -> None:
    parent = _old_repair_parent()
    candidate, events = IDENTITY_REPLAY.replay_hidden_suffix_commit_boundary(
        parent,
        _config(),
    )

    parent_ids = _id_by_key(parent)
    candidate_ids = _id_by_key(candidate)
    assert parent_ids[("Pig_1", 10)] == "ID_2"
    assert parent_ids[("Pig_2", 10)] == "ID_1"
    for frame in {10, 11}:
        assert candidate_ids[("Pig_1", frame)] == "ID_1"
        assert candidate_ids[("Pig_2", frame)] == "ID_2"
    for frame in range(12, 16):
        assert candidate_ids[("Pig_1", frame)] == "ID_2"
        assert candidate_ids[("Pig_2", frame)] == "ID_1"
    assert events == [
        {
            "hidden_label": "Pig_2",
            "partner_label": "Pig_1",
            "run_start": 10,
            "run_end": 11,
            "run_length": 2,
            "hidden_median_score": 0.3,
            "max_partner_iou": 0.694444,
            "old_commit_start": 10,
            "candidate_commit_start": 12,
            "common_suffix_frames": 6,
            "changed_frames": [10, 11],
        }
    ]
    assert [
        IDENTITY_REPLAY.payload_without_id(shape) for shape in parent
    ] == [IDENTITY_REPLAY.payload_without_id(shape) for shape in candidate]


def test_identity_replay_does_not_invent_an_unapplied_parent_repair() -> None:
    parent = _old_repair_parent()
    for shape in parent:
        fixed_id = IDENTITY_REPLAY.label_identity(shape)
        IDENTITY_REPLAY.set_id_value(shape, f"ID_{fixed_id}")

    candidate, events = IDENTITY_REPLAY.replay_hidden_suffix_commit_boundary(
        parent,
        _config(),
    )

    assert events == []
    assert candidate == parent


def test_identity_replay_requires_both_tracks_visible_at_commit() -> None:
    parent = _old_repair_parent()
    partner_at_commit = next(
        shape
        for shape in parent
        if shape["label"] == "Pig_1" and shape["frame"] == 12
    )
    next(
        attribute
        for attribute in partner_at_commit["attributes"]
        if attribute["name"] == "Hidden"
    )["value"] = "Yes"

    candidate, events = IDENTITY_REPLAY.replay_hidden_suffix_commit_boundary(
        parent,
        _config(),
    )

    assert events == []
    assert candidate == parent


def test_identity_replay_plan_rejects_non_id_payload(tmp_path: Path) -> None:
    plan_payload = _valid_plan()
    plan_payload["identity_replay_contract"]["allowed_payload_change"] = [
        "ID attribute",
        "points",
    ]
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(plan_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not isolate the ID attribute"):
        IDENTITY_REPLAY.validate_plan(plan, IDENTITY_REPLAY.CANDIDATE)


def test_identity_replay_rejects_cli_parameter_drift() -> None:
    args = SimpleNamespace(
        min_hidden_frames=9,
        max_hidden_frames=None,
        min_overlap_iou=None,
        max_hidden_median_score=None,
        start_back_frames=None,
        min_suffix_frames=None,
    )

    with pytest.raises(ValueError, match="min_hidden_frames"):
        IDENTITY_REPLAY.validated_replay_config(_valid_plan(), args)


def test_identity_replay_rejects_score_window_override(tmp_path: Path) -> None:
    window = tmp_path / "windows.csv"
    window.write_text(
        "video_stem,episode_id,first_switch_frame,last_switch_frame,"
        "switch_event_rows,score_start_frame,score_end_frame\n"
        "Pigs291119_000233_30fps,event,1111,1114,4,1104,1119\n",
        encoding="utf-8",
    )
    plan = _valid_plan()
    plan["parent"] = {"video": "Pigs291119_000233_30fps"}
    plan["frozen_window"] = {
        "path": str(window),
        "directory": str(tmp_path),
        "filename": window.name,
        "manifest_sha256": IDENTITY_REPLAY.file_sha256(window),
        "event_id": "event",
        "score_frames": [1104, 1119],
        "switch_frames": [1111, 1114],
        "parent_remapped_idsw": 4,
    }
    args = SimpleNamespace(score_start_frame=1105, score_end_frame=1119)

    with pytest.raises(ValueError, match="override"):
        IDENTITY_REPLAY.validate_frozen_window(plan, args)
