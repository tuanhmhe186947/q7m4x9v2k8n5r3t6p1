from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image

from pig_behavior.classification_v2.review.identity_continuity_adjudication import (
    FrameCandidate,
    IdentityAdjudicationError,
    IdentityCase,
)

ROOT = Path(__file__).resolve().parents[1]
GUI_SCRIPT = (
    ROOT
    / "scripts"
    / "classification_v2"
    / "01_review_units_gui"
    / "review_identity_continuity_gui.py"
)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, GUI_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _candidate(key: str, x1: float) -> FrameCandidate:
    return FrameCandidate(
        frame_index=3,
        source_frame_index=103,
        object_track_key=key,
        track_id=key.replace("actor-", "track-"),
        pig_id=key.replace("actor-", "ID_"),
        x1=x1,
        y1=10.0,
        x2=x1 + 20.0,
        y2=30.0,
        source_video_path="C:/source/scene.mp4",
    )


def _case() -> IdentityCase:
    return IdentityCase(
        review_item_id="item-a",
        review_unit_id="unit-a",
        source_type="legacy_recovered",
        dataset_id="legacy",
        video_key="scene/001",
        original_pig_id="ID_a",
        original_track_id="track-a",
        original_object_track_key="actor-a",
        frame_indices=(3,),
    )


def test_render_distinguishes_original_and_selected_box_without_behavior_label() -> None:
    module = _load("identity_continuity_gui_overlay")
    source = Image.new("RGB", (100, 60), "white")
    candidates = (_candidate("actor-a", 10.0), _candidate("actor-b", 50.0))

    rendered = module.render_identity_frame(
        source,
        candidates,
        (_case(),),
        {"unit-a": "actor-b"},
        "unit-a",
    )

    assert source.getpixel((10, 10)) == (255, 255, 255)
    assert rendered.getpixel((10, 30)) == (255, 217, 102)
    assert rendered.getpixel((50, 30)) == (0, 166, 90)
    assert "behavior" not in module.candidate_label(candidates[0]).casefold()


def test_click_mapping_uses_display_scale_and_smallest_overlapping_box() -> None:
    module = _load("identity_continuity_gui_click")
    large = FrameCandidate(
        frame_index=3,
        source_frame_index=103,
        object_track_key="large",
        track_id="large",
        pig_id="large",
        x1=10.0,
        y1=10.0,
        x2=80.0,
        y2=80.0,
        source_video_path="C:/source/scene.mp4",
    )
    small = FrameCandidate(
        frame_index=3,
        source_frame_index=103,
        object_track_key="small",
        track_id="small",
        pig_id="small",
        x1=20.0,
        y1=20.0,
        x2=40.0,
        y2=40.0,
        source_video_path="C:/source/scene.mp4",
    )

    selected = module.candidate_at_display_point(
        (large, small),
        x=70.0,
        y=70.0,
        scale=2.0,
        offset=(10, 10),
    )

    assert selected is small
    assert module.candidate_at_display_point((large,), 0.0, 0.0, 1.0, (0, 0)) is None


class _Frame:
    shape = (60, 100, 3)


class _Capture:
    def __init__(self, decoded_position: float, *, seek_ok: bool = True) -> None:
        self.decoded_position = decoded_position
        self.seek_ok = seek_ok
        self.requested: int | None = None

    def set(self, _property: int, frame_index: int) -> bool:
        self.requested = frame_index
        return self.seek_ok

    def read(self) -> tuple[bool, _Frame]:
        return True, _Frame()

    def get(self, _property: int) -> float:
        return self.decoded_position


class _Canvas:
    def __init__(self) -> None:
        self.cursor = ""
        self.created: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def delete(self, _item: object) -> None:
        return None

    def configure(self, *, cursor: str) -> None:
        self.cursor = cursor

    def create_rectangle(self, *args: object, **kwargs: object) -> int:
        self.created.append((args, kwargs))
        return len(self.created)

    def create_text(self, *args: object, **kwargs: object) -> int:
        self.created.append((args, kwargs))
        return len(self.created)


def test_canvas_drag_normalizes_clamps_and_rejects_tiny_boxes() -> None:
    module = _load("identity_continuity_gui_bbox_math")

    assert module.canvas_drag_to_source_bbox(
        (90.0, 70.0),
        (10.0, 10.0),
        scale=0.5,
        offset=(10, 10),
        source_size=(120, 100),
    ) == (0.0, 0.0, 120.0, 100.0)
    assert (
        module.canvas_drag_to_source_bbox(
            (10.0, 10.0),
            (11.0, 11.0),
            scale=1.0,
            offset=(0, 0),
            source_size=(120, 100),
        )
        is None
    )


def test_minicvat_geometry_supports_handles_move_resize_and_clamping() -> None:
    module = _load("identity_continuity_gui_minicvat_geometry")

    canvas_bbox = module.source_bbox_to_canvas(
        (10.0, 20.0, 50.0, 60.0),
        scale=2.0,
        offset=(5, 7),
    )
    assert canvas_bbox == (25.0, 47.0, 105.0, 127.0)
    assert module.hit_test_bbox_handle((25.0, 47.0), canvas_bbox) == "nw"
    assert module.hit_test_bbox_handle((65.0, 127.0), canvas_bbox) == "s"
    assert module.canvas_point_inside_bbox((65.0, 80.0), canvas_bbox)
    assert module.transform_source_bbox(
        (10.0, 10.0, 30.0, 30.0),
        delta=(-20.0, 50.0),
        operation="move",
        source_size=(100, 60),
    ) == (0.0, 40.0, 20.0, 60.0)
    assert module.transform_source_bbox(
        (10.0, 10.0, 30.0, 30.0),
        delta=(15.0, 20.0),
        operation="se",
        source_size=(100, 60),
    ) == (10.0, 10.0, 45.0, 50.0)


def test_canvas_release_moves_source_bbox_and_saves_correction() -> None:
    module = _load("identity_continuity_gui_bbox_move")
    gui = module.IdentityContinuityGui.__new__(module.IdentityContinuityGui)
    gui.finalized = False
    gui.cases = (_case(),)
    gui.active_case_position = 0
    gui.all_frames = (3,)
    gui.current_frame_position = 0
    gui.selections = {("unit-a", 3): "actor-a"}
    gui.exclusions = {}
    gui.bbox_edits = {}
    gui.candidates_by_frame = {3: (_candidate("actor-a", 10.0),)}
    gui._bbox_draw_mode = None
    gui._bbox_drag_start = (15.0, 15.0)
    gui._bbox_drag_rectangle = None
    gui._bbox_interaction = "move"
    gui._bbox_resize_handle = None
    gui._bbox_origin = (10.0, 10.0, 30.0, 30.0)
    gui._bbox_preview = gui._bbox_origin
    gui._display_scale = 1.0
    gui._display_offset = (0, 0)
    gui._source_image_size = (100, 60)
    gui.canvas = _Canvas()
    gui.status_var = type("Status", (), {"set": lambda _self, _value: None})()
    gui.save = lambda *, silent=False: True
    gui.show_current_frame = lambda: None

    gui._on_canvas_release(type("Event", (), {"x": 25, "y": 20})())

    edit = gui.bbox_edits[("unit-a", 3)]
    assert edit.mode == module.CORRECTED_BBOX_MODE
    assert edit.source_object_track_key == "actor-a"
    assert edit.bbox == (20.0, 15.0, 40.0, 35.0)
    assert gui.selections[("unit-a", 3)] == "actor-a"


def test_handle_drag_resizes_source_bbox_and_autosaves() -> None:
    module = _load("identity_continuity_gui_bbox_resize")
    gui = module.IdentityContinuityGui.__new__(module.IdentityContinuityGui)
    gui.finalized = False
    gui.cases = (_case(),)
    gui.active_case_position = 0
    gui.all_frames = (3,)
    gui.current_frame_position = 0
    gui.selections = {("unit-a", 3): "actor-a"}
    gui.exclusions = {}
    gui.bbox_edits = {}
    gui.candidates_by_frame = {3: (_candidate("actor-a", 10.0),)}
    gui._bbox_draw_mode = None
    gui._bbox_drag_start = None
    gui._bbox_drag_rectangle = None
    gui._bbox_interaction = None
    gui._bbox_resize_handle = None
    gui._bbox_origin = None
    gui._bbox_preview = None
    gui._display_scale = 1.0
    gui._display_offset = (0, 0)
    gui._source_image_size = (100, 60)
    gui.canvas = _Canvas()
    gui.status_var = type("Status", (), {"set": lambda _self, _value: None})()
    gui.save = lambda *, silent=False: True
    gui.show_current_frame = lambda: None

    gui._on_canvas_press(type("Event", (), {"x": 30, "y": 30})())
    assert gui._bbox_interaction == "resize"
    assert gui._bbox_resize_handle == "se"
    gui._on_canvas_drag(type("Event", (), {"x": 40, "y": 35})())
    gui._on_canvas_release(type("Event", (), {"x": 40, "y": 35})())

    edit = gui.bbox_edits[("unit-a", 3)]
    assert edit.mode == module.CORRECTED_BBOX_MODE
    assert edit.bbox == (10.0, 10.0, 40.0, 35.0)


def test_added_bbox_can_be_moved_after_creation() -> None:
    module = _load("identity_continuity_gui_added_bbox_edit")
    gui = module.IdentityContinuityGui.__new__(module.IdentityContinuityGui)
    gui.finalized = False
    gui.cases = (_case(),)
    gui.active_case_position = 0
    gui.all_frames = (3,)
    gui.current_frame_position = 0
    gui.selections = {}
    gui.exclusions = {}
    gui.bbox_edits = {}
    gui.candidates_by_frame = {3: (_candidate("actor-a", 10.0),)}
    gui._bbox_draw_mode = module.ADDED_BBOX_MODE
    gui._bbox_drag_start = (40.0, 10.0)
    gui._bbox_drag_rectangle = None
    gui._bbox_interaction = "add"
    gui._bbox_resize_handle = None
    gui._bbox_origin = None
    gui._bbox_preview = None
    gui._display_scale = 1.0
    gui._display_offset = (0, 0)
    gui._source_image_size = (100, 60)
    gui.canvas = _Canvas()
    gui.status_var = type("Status", (), {"set": lambda _self, _value: None})()
    gui.save = lambda *, silent=False: True
    gui.show_current_frame = lambda: None

    gui._on_canvas_release(type("Event", (), {"x": 70, "y": 35})())
    first_edit = gui.bbox_edits[("unit-a", 3)]
    assert first_edit.mode == module.ADDED_BBOX_MODE
    assert first_edit.bbox == (40.0, 10.0, 70.0, 35.0)

    gui._on_canvas_press(type("Event", (), {"x": 50, "y": 20})())
    gui._on_canvas_drag(type("Event", (), {"x": 55, "y": 25})())
    gui._on_canvas_release(type("Event", (), {"x": 55, "y": 25})())

    moved_edit = gui.bbox_edits[("unit-a", 3)]
    assert moved_edit.mode == module.ADDED_BBOX_MODE
    assert moved_edit.bbox == (45.0, 15.0, 75.0, 40.0)


def test_selecting_source_candidate_clears_stale_bbox_edit() -> None:
    module = _load("identity_continuity_gui_clear_stale_edit")
    gui = module.IdentityContinuityGui.__new__(module.IdentityContinuityGui)
    gui.finalized = False
    gui.cases = (_case(),)
    gui.active_case_position = 0
    gui.all_frames = (3,)
    gui.current_frame_position = 0
    gui.selections = {("unit-a", 3): "actor-a"}
    gui.exclusions = {}
    gui.bbox_edits = {
        ("unit-a", 3): module.BoundingBoxEdit(
            module.CORRECTED_BBOX_MODE,
            11.0,
            11.0,
            31.0,
            31.0,
            "actor-a",
        )
    }
    gui.save = lambda *, silent=False: True
    gui.show_current_frame = lambda: None
    gui.status_var = type("Status", (), {"set": lambda _self, _value: None})()

    gui.select_candidate(_candidate("actor-a", 10.0))

    assert gui.bbox_edits == {}


def test_exact_seek_and_candidate_bounds_fail_closed() -> None:
    module = _load("identity_continuity_gui_exact_seek")
    capture = _Capture(decoded_position=104.0)

    assert module.decode_exact_source_frame(capture, 103).shape == (60, 100, 3)
    assert capture.requested == 103

    with pytest.raises(RuntimeError, match="decoded_source_frame_mismatch"):
        module.decode_exact_source_frame(_Capture(decoded_position=103.0), 103)
    with pytest.raises(RuntimeError, match="cannot_seek_source_frame"):
        module.decode_exact_source_frame(
            _Capture(decoded_position=104.0, seek_ok=False),
            103,
        )

    module.assert_candidate_bounds(
        (_candidate("actor-a", 10.0),),
        review_frame_index=3,
        image_width=100,
        image_height=60,
    )
    out_of_bounds = _candidate("actor-c", 90.0)
    with pytest.raises(RuntimeError, match="candidate_bbox_outside_decoded_video"):
        module.assert_candidate_bounds(
            (out_of_bounds,),
            review_frame_index=3,
            image_width=100,
            image_height=60,
        )


def test_video_path_must_equal_declared_source_authority(tmp_path: Path) -> None:
    module = _load("identity_continuity_gui_video_authority")
    declared = tmp_path / "declared" / "scene.mp4"
    unrelated = tmp_path / "unrelated" / "scene.mp4"
    declared.parent.mkdir()
    unrelated.parent.mkdir()
    declared.write_bytes(b"declared")
    unrelated.write_bytes(b"unrelated")
    candidate = FrameCandidate(
        frame_index=3,
        source_frame_index=103,
        object_track_key="actor-a",
        track_id="track-a",
        pig_id="ID_a",
        x1=10.0,
        y1=10.0,
        x2=30.0,
        y2=30.0,
        source_video_path=str(declared),
    )
    candidates = {3: (candidate,)}

    assert module.resolve_video_path(candidates, declared) == declared.resolve()
    with pytest.raises(
        IdentityAdjudicationError,
        match="not_bound_to_declared_source_video",
    ):
        module.resolve_video_path(candidates, unrelated)


def test_cli_paths_reject_behavior_ledger_before_file_access(tmp_path: Path) -> None:
    module = _load("identity_continuity_gui_cli_path_guard")
    forbidden = (
        tmp_path
        / "human_review_workspace"
        / "classification_v2"
        / "run"
        / "human_decisions"
        / "behavior"
        / "review_view.csv"
    )
    config = module.IdentityGuiConfig(
        review_units_csv=forbidden,
        frame_features_csv=tmp_path / "frames.csv",
        output_dir=tmp_path / "output",
        reviewer="reviewer",
        review_item_ids=("item-a",),
    )

    with pytest.raises(
        IdentityAdjudicationError,
        match="cli_path_is_behavior_ledger",
    ):
        module.assert_safe_cli_input_paths(config)

    protected_candidate = FrameCandidate(
        frame_index=3,
        source_frame_index=103,
        object_track_key="actor-a",
        track_id="track-a",
        pig_id="ID_a",
        x1=10.0,
        y1=10.0,
        x2=30.0,
        y2=30.0,
        source_video_path=str(forbidden),
    )
    with pytest.raises(
        IdentityAdjudicationError,
        match="declared_source_video_path_is_behavior_ledger",
    ):
        module.resolve_video_path({3: (protected_candidate,)}, None)


def test_navigation_stays_inside_active_case_frame_scope() -> None:
    module = _load("identity_continuity_gui_scope_navigation")
    case = IdentityCase(
        review_item_id="item-a",
        review_unit_id="unit-a",
        source_type="legacy_recovered",
        dataset_id="legacy",
        video_key="scene/001",
        original_pig_id="ID_a",
        original_track_id="track-a",
        original_object_track_key="actor-a",
        frame_indices=(3, 7, 12),
    )

    assert module.step_case_frame(case.frame_indices, 3, 1) == 7
    assert module.step_case_frame(case.frame_indices, 12, 1) == 3
    assert module.step_case_frame(case.frame_indices, 4, 1) == 3
    assert module.first_pending_case_frame(
        case,
        {("unit-a", 3): "actor-a"},
    ) == 7


def test_finalization_marker_locks_and_can_be_explicitly_reopened(
    tmp_path: Path,
) -> None:
    module = _load("identity_continuity_gui_finalization")
    frame_path = tmp_path / module.FRAME_SIDECAR_NAME
    case_path = tmp_path / module.CASE_SIDECAR_NAME
    frame_path.write_text("frame\n", encoding="utf-8")
    case_path.write_text("case\n", encoding="utf-8")

    module.write_finalization_marker(tmp_path, reviewer="reviewer-a")
    locked = module.load_finalization_marker(tmp_path)
    assert locked is not None
    assert locked["status"] == module.FINALIZED_STATUS

    module.reopen_finalization_marker(tmp_path, reviewer="reviewer-b")
    reopened = module.load_finalization_marker(tmp_path)
    assert reopened is not None
    assert reopened["status"] == module.REOPENED_STATUS


def test_autosave_failure_restores_in_memory_selection() -> None:
    module = _load("identity_continuity_gui_rollback")
    gui = module.IdentityContinuityGui.__new__(module.IdentityContinuityGui)
    gui.finalized = False
    gui.cases = (_case(),)
    gui.active_case_position = 0
    gui.all_frames = (3,)
    gui.current_frame_position = 0
    gui.selections = {("unit-a", 3): "actor-a"}
    gui.exclusions = {}
    gui.save = lambda *, silent: False
    gui.show_current_frame = lambda: None
    gui.status_var = type("Status", (), {"set": lambda _self, _value: None})()

    gui.select_candidate(_candidate("actor-b", 50.0))

    assert gui.selections == {("unit-a", 3): "actor-a"}


def test_autosave_failure_restores_exclusion_state(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load("identity_continuity_gui_exclusion_rollback")
    gui = module.IdentityContinuityGui.__new__(module.IdentityContinuityGui)
    gui.finalized = False
    gui.cases = (_case(),)
    gui.active_case_position = 0
    gui.all_frames = (3,)
    gui.current_frame_position = 0
    gui.selections = {("unit-a", 3): "actor-a"}
    gui.exclusions = {}
    gui.save = lambda *, silent: False
    gui.show_current_frame = lambda: None
    gui.status_var = type("Status", (), {"set": lambda _self, _value: None})()
    gui.root = object()
    monkeypatch.setattr(module.simpledialog, "askstring", lambda *_args, **_kwargs: "reason")

    gui.exclude_active_case()

    assert gui.selections == {("unit-a", 3): "actor-a"}
    assert gui.exclusions == {}


def test_out_of_scope_selection_is_rejected_before_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load("identity_continuity_gui_scope_selection")
    gui = module.IdentityContinuityGui.__new__(module.IdentityContinuityGui)
    gui.finalized = False
    gui.cases = (_case(),)
    gui.active_case_position = 0
    gui.all_frames = (3, 4)
    gui.current_frame_position = 1
    gui.selections = {}
    gui.exclusions = {}
    gui.root = object()
    gui.save = lambda *, silent: pytest.fail("out-of-scope selection must not save")
    monkeypatch.setattr(module.messagebox, "showerror", lambda *_args, **_kwargs: None)

    gui.select_candidate(_candidate("actor-b", 50.0))

    assert gui.selections == {}


def test_prefetch_is_scheduled_after_idle_without_tk_window() -> None:
    module = _load("identity_continuity_gui_idle_prefetch")

    class FakeRoot:
        def __init__(self) -> None:
            self.callback = None
            self.cancelled: list[str] = []

        def after_idle(self, callback):
            self.callback = callback
            return "idle-1"

        def after_cancel(self, callback_id: str) -> None:
            self.cancelled.append(callback_id)

    gui = module.IdentityContinuityGui.__new__(module.IdentityContinuityGui)
    gui.root = FakeRoot()
    gui._prefetch_after_id = None
    called: list[str] = []
    gui._prefetch_adjacent_frame = lambda: called.append("prefetch")

    gui._schedule_adjacent_prefetch()

    assert called == []
    assert gui._prefetch_after_id == "idle-1"
    assert gui.root.callback is not None
    gui.root.callback()
    assert called == ["prefetch"]
