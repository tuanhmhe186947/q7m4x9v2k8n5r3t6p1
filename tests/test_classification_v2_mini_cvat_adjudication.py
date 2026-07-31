from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from pig_behavior.classification_v2.review.mini_cvat_adjudication import (
    MiniCvatActorAttributes,
    MiniCvatFrameAnnotation,
    load_mini_cvat_sidecar,
    validate_mini_cvat_state,
    write_mini_cvat_sidecar,
)


class _Var:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


def _load_gui_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "classification_v2" / "01_review_units_gui"
    path /= "review_identity_continuity_gui.py"
    spec = importlib.util.spec_from_file_location("mini_cvat_gui_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_gui_identity_change_swaps_existing_actor_scope() -> None:
    module = _load_gui_module()
    from pig_behavior.classification_v2.review.identity_continuity_adjudication import (
        FrameCandidate,
    )

    gui = module.IdentityContinuityGui.__new__(module.IdentityContinuityGui)
    gui.finalized = False
    gui.mini_cvat_enabled = True
    gui.editable_pig_ids = ("ID_4", "ID_5")
    gui.active_pig_id = "ID_5"
    gui.all_frames = (3,)
    gui.current_frame_position = 0
    gui.candidates_by_frame = {
        3: (
            FrameCandidate(
                3,
                3,
                "actor-4",
                "track-4",
                "ID_4",
                1.0,
                1.0,
                11.0,
                11.0,
                "scene.mp4",
                "fight",
                "No",
            ),
            FrameCandidate(
                3,
                3,
                "actor-5",
                "track-5",
                "ID_5",
                20.0,
                20.0,
                30.0,
                30.0,
                "scene.mp4",
                "move",
                "No",
            ),
        )
    }
    gui.mini_actor_attributes = {
        "ID_5": MiniCvatActorAttributes(
            "ID_5", "ID_5", "ID_4", "move", "move"
        )
    }
    gui.mini_frame_annotations = {}
    gui.mini_selected_keys = {}
    gui.mini_reviewed_id_var = _Var("ID_4")
    gui.mini_behavior_var = _Var("move")
    gui.status_var = _Var("")
    gui.save = lambda silent=True: True
    gui.show_current_frame = lambda: None
    gui.cancel_bbox_drawing = lambda silent=True: None
    gui.apply_mini_actor_attributes()
    assert gui.mini_actor_attributes["ID_5"].reviewed_pig_id == "ID_4"
    assert gui.mini_actor_attributes["ID_4"].reviewed_pig_id == "ID_5"


def test_mini_cvat_actor_apply_normalizes_bare_numeric_reviewed_id() -> None:
    module = _load_gui_module()
    from pig_behavior.classification_v2.review.identity_continuity_adjudication import (
        FrameCandidate,
    )

    gui = module.IdentityContinuityGui.__new__(module.IdentityContinuityGui)
    gui.finalized = False
    gui.mini_cvat_enabled = True
    gui.editable_pig_ids = ("ID_4", "ID_5")
    gui.active_pig_id = "ID_4"
    gui.all_frames = (3,)
    gui.current_frame_position = 0
    gui.candidates_by_frame = {
        3: (
            FrameCandidate(
                3,
                3,
                "actor-4",
                "track-4",
                "ID_4",
                1.0,
                1.0,
                11.0,
                11.0,
                "scene.mp4",
                "fight",
                "No",
            ),
            FrameCandidate(
                3,
                3,
                "actor-5",
                "track-5",
                "ID_5",
                20.0,
                20.0,
                30.0,
                30.0,
                "scene.mp4",
                "move",
                "No",
            ),
        )
    }
    gui.mini_actor_attributes = {}
    gui.mini_frame_annotations = {}
    gui.mini_selected_keys = {}
    gui.mini_reviewed_id_var = _Var("5")
    gui.mini_behavior_var = _Var("fight")
    gui.status_var = _Var("")
    gui.save = lambda silent=True: True
    gui.show_current_frame = lambda: None
    gui.cancel_bbox_drawing = lambda silent=True: None

    gui.apply_mini_actor_attributes()

    assert gui.mini_reviewed_id_var.get() == "ID_5"
    assert gui.mini_actor_attributes["ID_4"].reviewed_pig_id == "ID_5"
    assert gui.mini_actor_attributes["ID_5"].reviewed_pig_id == "ID_4"


def test_mini_cvat_save_current_frame_applies_actor_mapping() -> None:
    module = _load_gui_module()
    from pig_behavior.classification_v2.review.identity_continuity_adjudication import (
        FrameCandidate,
    )

    gui = module.IdentityContinuityGui.__new__(module.IdentityContinuityGui)
    gui.finalized = False
    gui.mini_cvat_enabled = True
    gui.editable_pig_ids = ("ID_4", "ID_5")
    gui.active_pig_id = "ID_4"
    gui.all_frames = (3,)
    gui.current_frame_position = 0
    gui.candidates_by_frame = {
        3: (
            FrameCandidate(
                3,
                3,
                "actor-4",
                "track-4",
                "ID_4",
                1.0,
                1.0,
                11.0,
                11.0,
                "scene.mp4",
                "fight",
                "No",
            ),
            FrameCandidate(
                3,
                3,
                "actor-5",
                "track-5",
                "ID_5",
                20.0,
                20.0,
                30.0,
                30.0,
                "scene.mp4",
                "move",
                "No",
            ),
        )
    }
    gui.mini_actor_attributes = {}
    gui.mini_frame_annotations = {}
    gui.mini_selected_keys = {}
    gui.mini_reviewed_id_var = _Var("5")
    gui.mini_behavior_var = _Var("fight")
    gui.mini_hidden_var = _Var("No")
    gui.status_var = _Var("")
    gui.save = lambda silent=True: True
    gui.show_current_frame = lambda: None
    gui.cancel_bbox_drawing = lambda silent=True: None

    gui.save_mini_current_frame()

    assert gui.mini_reviewed_id_var.get() == "ID_5"
    assert gui.mini_actor_attributes["ID_4"].reviewed_pig_id == "ID_5"
    assert gui.mini_actor_attributes["ID_5"].reviewed_pig_id == "ID_4"
    assert gui.mini_frame_annotations[("ID_4", 3)].reviewed_hidden == "No"


def test_mini_cvat_display_button_selects_owner_scope() -> None:
    module = _load_gui_module()
    from pig_behavior.classification_v2.review.identity_continuity_adjudication import (
        FrameCandidate,
    )

    gui = module.IdentityContinuityGui.__new__(module.IdentityContinuityGui)
    gui.finalized = False
    gui.mini_cvat_enabled = True
    gui.editable_pig_ids = ("ID_4", "ID_5")
    gui.active_pig_id = "ID_5"
    gui.all_frames = (3,)
    gui.current_frame_position = 0
    gui.candidates_by_frame = {
        3: (
            FrameCandidate(
                3,
                3,
                "actor-4",
                "track-4",
                "ID_4",
                1.0,
                1.0,
                11.0,
                11.0,
                "scene.mp4",
                "fight",
                "No",
            ),
            FrameCandidate(
                3,
                3,
                "actor-5",
                "track-5",
                "ID_5",
                20.0,
                20.0,
                30.0,
                30.0,
                "scene.mp4",
                "move",
                "No",
            ),
        )
    }
    gui.mini_actor_attributes = {
        "ID_4": MiniCvatActorAttributes(
            "ID_4",
            "ID_4",
            "ID_5",
            "fight",
            "fight",
        ),
        "ID_5": MiniCvatActorAttributes(
            "ID_5",
            "ID_5",
            "ID_4",
            "move",
            "move",
        ),
    }
    gui.mini_frame_annotations = {}
    gui.mini_selected_keys = {}
    gui.status_var = _Var("")
    gui.show_current_frame = lambda: None
    gui.cancel_bbox_drawing = lambda silent=True: None

    gui.select_mini_display_actor("5")

    assert gui.active_pig_id == "ID_4"
    assert gui.status_var.get().startswith("Đang sửa source scope ID_4")


def test_mini_cvat_reset_identity_fields_restores_saved_values() -> None:
    module = _load_gui_module()

    gui = module.IdentityContinuityGui.__new__(module.IdentityContinuityGui)
    gui.finalized = False
    gui.mini_cvat_enabled = True
    gui.editable_pig_ids = ("ID_4", "ID_5")
    gui.active_pig_id = "ID_4"
    gui.mini_actor_attributes = {
        "ID_4": MiniCvatActorAttributes(
            "ID_4",
            "ID_4",
            "ID_5",
            "fight",
            "fight",
        )
    }
    gui.mini_reviewed_id_var = _Var("ID_5")
    gui.mini_behavior_var = _Var("move")
    gui.status_var = _Var("")
    gui.show_current_frame = lambda: None

    gui.reset_mini_actor_identity_fields()

    assert gui.mini_reviewed_id_var.get() == "ID_5"
    assert gui.mini_behavior_var.get() == "fight"
    assert gui.status_var.get().startswith("Đã khôi phục reviewed ID đã lưu")
