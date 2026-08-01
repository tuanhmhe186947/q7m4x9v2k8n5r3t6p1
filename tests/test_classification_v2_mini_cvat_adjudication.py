from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from pig_behavior.classification_v2.review.mini_cvat_adjudication import (
    MiniCvatActorAttributes,
    MiniCvatFrameAnnotation,
    load_mini_cvat_sidecar,
    validate_mini_cvat_state,
    write_mini_cvat_sidecar,
)


def _attributes() -> dict[str, MiniCvatActorAttributes]:
    return {
        "ID_4": MiniCvatActorAttributes(
            actor_scope_id="ID_4",
            original_pig_id="ID_4",
            reviewed_pig_id="ID_4",
            original_behavior="fight",
            reviewed_behavior="fight",
        )
    }


def _annotation(frame_index: int) -> MiniCvatFrameAnnotation:
    return MiniCvatFrameAnnotation(
        actor_scope_id="ID_4",
        frame_index=frame_index,
        source_frame_index=frame_index,
        original_object_track_key=f"actor-4-{frame_index}",
        original_track_id="track-4",
        original_pig_id="ID_4",
        reviewed_pig_id="ID_4",
        bbox_mode="SOURCE_BBOX",
        x1=1.0,
        y1=2.0,
        x2=11.0,
        y2=12.0,
        original_hidden="No",
        reviewed_hidden="No",
    )


def test_mini_cvat_sidecar_round_trip_preserves_frame_scope(tmp_path: Path) -> None:
    annotations = {
        ("ID_4", 3): replace(_annotation(3), reviewed_pig_id="ID_5"),
        ("ID_4", 4): _annotation(4),
    }
    write_mini_cvat_sidecar(
        tmp_path,
        reviewer="reviewer-a",
        source_type="cvat_tracking_xml",
        dataset_id="set-a",
        video_key="scene/001",
        editable_actor_ids=("ID_4",),
        frame_indices=(3, 4),
        actor_attributes=_attributes(),
        frame_annotations=annotations,
    )
    loaded_attributes, loaded_annotations = load_mini_cvat_sidecar(
        tmp_path,
        source_type="cvat_tracking_xml",
        dataset_id="set-a",
        video_key="scene/001",
        editable_actor_ids=("ID_4",),
        frame_indices=(3, 4),
    )
    assert loaded_attributes == _attributes()
    assert loaded_annotations == annotations


def test_mini_cvat_load_migrates_v1_frame_identity(tmp_path: Path) -> None:
    annotation = asdict(_annotation(3))
    annotation.pop("reviewed_pig_id")
    payload = {
        "schema": "classification_v2.mini_cvat_adjudication.v1",
        "reviewer": "reviewer-a",
        "source_type": "cvat_tracking_xml",
        "dataset_id": "set-a",
        "video_key": "scene/001",
        "editable_actor_ids": ["ID_4"],
        "frame_indices": [3],
        "actor_attributes": [asdict(_attributes()["ID_4"])],
        "frame_annotations": [annotation],
        "source_annotations_changed": "NO",
        "behavior_decisions_changed": "NO",
        "model_x_forbidden": "YES",
    }
    sidecar = tmp_path / "mini_cvat_adjudication.json"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    _, annotations = load_mini_cvat_sidecar(
        tmp_path,
        source_type="cvat_tracking_xml",
        dataset_id="set-a",
        video_key="scene/001",
        editable_actor_ids=("ID_4",),
        frame_indices=(3,),
    )
    assert annotations[("ID_4", 3)].reviewed_pig_id == "ID_4"


def test_mini_cvat_completeness_validation_lists_pending_frame() -> None:
    errors = validate_mini_cvat_state(
        _attributes(),
        {("ID_4", 3): _annotation(3)},
        editable_actor_ids=("ID_4",),
        frame_indices=(3, 4),
        require_complete=True,
    )
    assert errors == ["mini_cvat_frame_pending=ID_4:4"]


def test_mini_cvat_rejects_duplicate_reviewed_actor_ids() -> None:
    attributes = _attributes() | {
        "ID_5": MiniCvatActorAttributes(
            actor_scope_id="ID_5",
            original_pig_id="ID_5",
            reviewed_pig_id="ID_4",
            original_behavior="move",
            reviewed_behavior="move",
        )
    }
    errors = validate_mini_cvat_state(
        attributes,
        {},
        editable_actor_ids=("ID_4", "ID_5"),
        frame_indices=(3,),
        require_complete=True,
    )
    assert "mini_cvat_duplicate_reviewed_pig_id=3:ID_4" in errors
