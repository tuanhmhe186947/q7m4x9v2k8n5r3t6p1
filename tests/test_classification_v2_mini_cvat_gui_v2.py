from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from pig_behavior.classification_v2.review.identity_continuity_adjudication import (
    FrameCandidate,
)
from pig_behavior.classification_v2.review.mini_cvat_editor import (
    MiniCvatEditorState,
)

ROOT = Path(__file__).resolve().parents[1]
GUI_SCRIPT = (
    ROOT
    / "scripts"
    / "classification_v2"
    / "01_review_units_gui"
    / "review_identity_continuity_gui_v2.py"
)


def _load_gui():
    spec = importlib.util.spec_from_file_location(
        "classification_v2_mini_cvat_gui_v2_test",
        GUI_SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_video_path_deduplicates_flat_video_guess(
    tmp_path: Path,
) -> None:
    module = _load_gui()
    video_key = "Pigs291119_000216_30fps"
    video_path = tmp_path / f"{video_key}.mp4"
    video_path.write_bytes(b"video-placeholder")

    resolved = module.resolve_video_path(
        (SimpleNamespace(video_key=video_key),),
        {},
        tmp_path,
    )

    assert resolved == video_path.resolve()


def _candidate(actor_id: str, x1: float) -> FrameCandidate:
    return FrameCandidate(
        frame_index=3,
        source_frame_index=3,
        object_track_key=f"actor-{actor_id}",
        track_id=f"track-{actor_id}",
        pig_id=actor_id,
        x1=x1,
        y1=10.0,
        x2=x1 + 20.0,
        y2=30.0,
        source_video_path="scene.mp4",
        behavior="fight",
        hidden="No",
    )


class _Canvas:
    def __init__(self) -> None:
        self.rectangles: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.texts: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def create_rectangle(self, *args: object, **kwargs: object) -> None:
        self.rectangles.append((args, kwargs))

    def create_text(self, *args: object, **kwargs: object) -> None:
        self.texts.append((args, kwargs))

    def grab_release(self) -> None:
        return None


def _state() -> MiniCvatEditorState:
    return MiniCvatEditorState(
        editable_actor_ids=("ID_4", "ID_5"),
        frame_indices=(3,),
        candidates_by_frame={
            3: (
                _candidate("ID_4", 10.0),
                _candidate("ID_5", 40.0),
                _candidate("ID_9", 70.0),
            )
        },
        actor_attributes={},
        frame_annotations={},
    )


def test_view_only_bbox_is_muted_and_never_gets_resize_handles() -> None:
    module = _load_gui()
    gui = module.MiniCvatGuiV2.__new__(module.MiniCvatGuiV2)
    gui.canvas = _Canvas()
    gui.display_scale = 1.0
    gui.display_offset = (0.0, 0.0)

    gui._draw_view_only_bbox(_candidate("ID_9", 70.0))

    assert len(gui.canvas.rectangles) == 1
    rectangle_kwargs = gui.canvas.rectangles[0][1]
    assert rectangle_kwargs["tags"] == ("bbox-view-only",)
    assert rectangle_kwargs["dash"] == (5, 3)
    assert len(gui.canvas.texts) == 1
    assert "chỉ xem" in str(gui.canvas.texts[0][1]["text"])


def test_resume_cursor_selects_first_unsaved_actor_in_first_unsaved_frame() -> None:
    module = _load_gui()
    state = _state()
    saved = state.change_draft(
        state.draft("ID_4", 3),
        reviewed_hidden="No",
    )
    state.save_frame(saved)

    gui = module.MiniCvatGuiV2.__new__(module.MiniCvatGuiV2)
    gui.frames = (3,)
    gui.frame_position = 0
    gui.state = state
    gui.config = type(
        "Config",
        (),
        {"editable_pig_ids": ("ID_4", "ID_5")},
    )()

    assert gui._resume_frame_position() == 0
    assert gui._resume_actor_id() == "ID_5"


def test_reset_all_session_changes_replaces_trial_state(
    monkeypatch,
) -> None:
    module = _load_gui()
    gui = module.MiniCvatGuiV2.__new__(module.MiniCvatGuiV2)
    gui.config = type(
        "Config",
        (),
        {
            "editable_pig_ids": ("ID_4", "ID_5"),
            "reviewer": "tester",
        },
    )()
    gui.frames = (3,)
    gui.frame_position = 0
    gui.active_actor_id = "ID_5"
    gui.candidates_by_frame = {
        3: (
            _candidate("ID_4", 10.0),
            _candidate("ID_5", 40.0),
        )
    }
    gui.state = _state()
    gui.state.save_frame(
        gui.state.change_draft(
            gui.state.draft("ID_4", 3),
            reviewed_hidden="No",
        )
    )
    gui.draft = gui.state.draft("ID_5", 3)
    gui.drag_intent = None
    gui.drag_preview = None
    gui.add_bbox_mode = False
    gui.add_start = None
    gui.canvas = _Canvas()
    gui.root = object()
    gui.status_var = type(
        "Status",
        (),
        {"set": lambda _self, _value: None},
    )()
    gui._load_draft_into_controls = lambda: None
    gui._render = lambda: None
    persisted: list[bool] = []
    gui._persist = lambda: persisted.append(True)
    monkeypatch.setattr(module.messagebox, "askyesno", lambda *_a, **_k: True)

    gui.reset_all_session_changes()

    assert persisted == [True]
    assert gui.active_actor_id == "ID_4"
    assert gui.frame_position == 0
    assert gui.state.frame_annotations == {}
    assert set(gui.state.actor_attributes) == {"ID_4", "ID_5"}
