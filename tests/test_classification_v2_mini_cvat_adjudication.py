from __future__ import annotations

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
        bbox_mode="SOURCE_BBOX",
        x1=1.0,
        y1=2.0,
        x2=11.0,
        y2=12.0,
        original_hidden="No",
        reviewed_hidden="No",
    )


def test_mini_cvat_sidecar_round_trip_preserves_burst_and_frame_scope(
    tmp_path: Path,
) -> None:
    annotations = {("ID_4", 3): _annotation(3), ("ID_4", 4): _annotation(4)}
    write_mini_cvat_sidecar(
        tmp_path,
        reviewer="reviewer",
        source_type="legacy_recovered",
        dataset_id="legacy",
        video_key="scene/001",
        editable_actor_ids=("ID_4",),
        frame_indices=(3, 4),
        actor_attributes=_attributes(),
        frame_annotations=annotations,
    )
    loaded = load_mini_cvat_sidecar(
        tmp_path,
        source_type="legacy_recovered",
        dataset_id="legacy",
        video_key="scene/001",
        editable_actor_ids=("ID_4",),
        frame_indices=(3, 4),
    )
    assert loaded == (_attributes(), annotations)


def test_mini_cvat_requires_complete_frames_for_finalization() -> None:
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
    assert "mini_cvat_duplicate_reviewed_pig_id=ID_4" in errors
