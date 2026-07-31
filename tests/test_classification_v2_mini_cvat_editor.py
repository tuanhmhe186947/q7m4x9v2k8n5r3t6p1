from __future__ import annotations

from pig_behavior.classification_v2.review.identity_continuity_adjudication import (
    CORRECTED_BBOX_MODE,
    FrameCandidate,
)
from pig_behavior.classification_v2.review.mini_cvat_editor import (
    MiniCvatEditorState,
    begin_bbox_drag,
    preview_bbox_drag,
)


def _candidate(
    actor_id: str,
    frame_index: int,
    x1: float,
    behavior: str,
) -> FrameCandidate:
    suffix = actor_id.removeprefix("ID_")
    return FrameCandidate(
        frame_index=frame_index,
        source_frame_index=frame_index,
        object_track_key=f"actor-{suffix}-{frame_index}",
        track_id=f"track-{suffix}",
        pig_id=actor_id,
        x1=x1,
        y1=10.0,
        x2=x1 + 20.0,
        y2=30.0,
        source_video_path="scene.mp4",
        behavior=behavior,
        hidden="No",
    )


def _state() -> MiniCvatEditorState:
    return MiniCvatEditorState(
        editable_actor_ids=("ID_4", "ID_5", "ID_6"),
        frame_indices=(3, 4),
        candidates_by_frame={
            3: (
                _candidate("ID_4", 3, 10.0, "fight"),
                _candidate("ID_5", 3, 40.0, "move"),
                _candidate("ID_6", 3, 70.0, "explore"),
            ),
            4: (
                _candidate("ID_4", 4, 12.0, "fight"),
                _candidate("ID_5", 4, 42.0, "move"),
                _candidate("ID_6", 4, 72.0, "explore"),
            ),
        },
        actor_attributes={},
        frame_annotations={},
    )


def test_resize_handle_wins_over_move_and_preview_is_stable() -> None:
    intent = begin_bbox_drag(
        (10.0, 10.0),
        (10.0, 10.0, 30.0, 30.0),
        scale=1.0,
        offset=(0.0, 0.0),
    )

    assert intent is not None
    assert intent.mode == "resize"
    assert intent.resize_handle == "nw"
    assert preview_bbox_drag(
        intent,
        (15.0, 16.0),
        scale=1.0,
        source_size=(100, 100),
    ) == (15.0, 16.0, 30.0, 30.0)


def test_move_preview_clamps_bbox_without_changing_extent() -> None:
    intent = begin_bbox_drag(
        (35.0, 35.0),
        (10.0, 10.0, 60.0, 60.0),
        scale=1.0,
        offset=(0.0, 0.0),
    )

    assert intent is not None
    assert intent.mode == "move"
    assert preview_bbox_drag(
        intent,
        (-100.0, -100.0),
        scale=1.0,
        source_size=(100, 100),
    ) == (0.0, 0.0, 50.0, 50.0)


def test_frame_id_swap_is_atomic_and_does_not_change_burst_behavior() -> None:
    state = _state()
    draft = state.change_draft(
        state.draft("ID_4", 3),
        reviewed_pig_id="6",
    )

    result = state.save_frame(draft)

    assert result.swapped_actor_scope_id == "ID_6"
    assert state.effective_reviewed_id("ID_4", 3) == "ID_6"
    assert state.effective_reviewed_id("ID_6", 3) == "ID_4"
    assert state.effective_reviewed_id("ID_4", 4) == "ID_4"
    assert state.effective_reviewed_id("ID_6", 4) == "ID_6"
    assert state.reviewed_behavior("ID_4") == "fight"
    assert state.reviewed_behavior("ID_6") == "explore"


def test_behavior_save_changes_only_actor_burst_behavior() -> None:
    state = _state()
    draft = state.change_draft(
        state.draft("ID_4", 3),
        reviewed_pig_id="ID_6",
    )
    state.save_frame(draft)

    state.save_behavior("ID_4", "social-nose")

    assert state.reviewed_behavior("ID_4") == "social-nose"
    assert state.effective_reviewed_id("ID_4", 3) == "ID_6"
    assert state.effective_reviewed_id("ID_4", 4) == "ID_4"


def test_unsaved_bbox_draft_does_not_mutate_resumable_state() -> None:
    state = _state()
    original = state.draft("ID_5", 3)
    changed = state.change_draft(
        original,
        bbox=(45.0, 12.0, 68.0, 35.0),
        bbox_mode=CORRECTED_BBOX_MODE,
    )

    assert changed.dirty is True
    assert state.draft("ID_5", 3) == original
    assert state.saved_frame_count("ID_5") == 0
