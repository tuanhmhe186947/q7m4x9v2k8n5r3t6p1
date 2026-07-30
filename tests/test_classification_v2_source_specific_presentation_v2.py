from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from types import MethodType

import pandas as pd
import pytest
from PIL import Image

from pig_behavior.classification_v2.review.source_specific_blinded_presentation_v2 import (
    ACTOR_COLOR,
    CVAT_CONTEXT_MODE,
    CVAT_RENDER_MODE,
    LEGACY_CONTEXT_MODE,
    LEGACY_NOTICE_TEXT,
    LEGACY_RENDER_MODE,
    NEUTRAL_NEIGHBOR_COLOR,
    PRESENTATION_SEMANTIC_HASH,
    PRESENTATION_TEMPLATE,
    PRESENTATION_VERSION,
    SourceSpecificPresentationError,
    build_media_authority_v2,
    canonical_presentation_contract_v2,
    compose_source_specific_contact_sheet,
    derive_calibration_outcome,
    presentation_semantic_hash_v2,
    public_display_text_v2,
    render_neutral_context_v2,
    validate_media_authority_v2,
)

ROOT = Path(__file__).resolve().parents[1]
GUI_PATH = (
    ROOT
    / "scripts"
    / "classification_v2"
    / "01_review_units_gui"
    / "review_interaction_blind_calibration_gui_v2.py"
)


def _load_gui_module() -> object:
    spec = importlib.util.spec_from_file_location(
        "_test_source_specific_gui_v2",
        GUI_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GUI = _load_gui_module()


def _unit(
    *,
    source_type: str = "cvat_tracking_xml",
    targets: str = "10,11,12,13,14,15",
    history: str = "",
) -> pd.Series:
    is_cvat = source_type == "cvat_tracking_xml"
    return pd.Series(
        {
            "calibration_item_id": "calibration_item_000001",
            "review_key": "review_1",
            "split": "CALIBRATION_DEVELOPMENT_SET",
            "presentation_order": 1,
            "source_type": source_type,
            "dataset_id": "dataset",
            "video_key": "video",
            "object_track_key": "actor",
            "pig_id": "pig_actor",
            "track_id": "track_actor",
            "context_mode": (
                CVAT_CONTEXT_MODE if is_cvat else LEGACY_CONTEXT_MODE
            ),
            "render_mode": CVAT_RENDER_MODE if is_cvat else LEGACY_RENDER_MODE,
            "actor_identity_semantics": (
                "red_bbox_is_reviewed_actor"
                if is_cvat
                else "entire_crop_is_reviewed_actor"
            ),
            "neighbor_context_available": is_cvat,
            "full_frame_context_available": is_cvat,
            "presentation_template": PRESENTATION_TEMPLATE,
            "presentation_version": PRESENTATION_VERSION,
            "presentation_semantic_hash": PRESENTATION_SEMANTIC_HASH,
            "target_frame_indices": targets,
            "history_frame_indices": history,
            "display_frame_indices": (
                ",".join(value for value in (history, targets) if value)
            ),
            "target_frame_count": len(targets.split(",")) if targets else 0,
            "history_frame_count": len(history.split(",")) if history else 0,
            "frame_order_contract": (
                "HISTORY_ASCENDING_THEN_TARGET_ASCENDING_NO_DUPLICATES"
            ),
            "media_authority": "media_1",
            "render_available": True,
            "render_failure_reason": "",
            "unit_start_frame": 10,
            "unit_end_frame": 15 if is_cvat else 25,
        }
    )


def _frame_rows(
    frames: list[int],
    *,
    neighbors: int = 1,
) -> pd.DataFrame:
    records = []
    for frame_index in frames:
        records.append(
            {
                "source_type": "cvat_tracking_xml",
                "dataset_id": "dataset",
                "video_key": "video",
                "frame_index": frame_index,
                "pig_id": "pig_actor",
                "track_id": "track_actor",
                "object_track_key": "actor",
                "x1": 10,
                "y1": 20,
                "x2": 30,
                "y2": 50,
            }
        )
        for neighbor in range(neighbors):
            x1 = 40 + neighbor * 25
            records.append(
                {
                    "source_type": "cvat_tracking_xml",
                    "dataset_id": "dataset",
                    "video_key": "video",
                    "frame_index": frame_index,
                    "pig_id": f"pig_neighbor_{neighbor}",
                    "track_id": f"track_neighbor_{neighbor}",
                    "object_track_key": f"neighbor_{neighbor}",
                    "x1": x1,
                    "y1": 20,
                    "x2": x1 + 20,
                    "y2": 50,
                }
            )
    return pd.DataFrame.from_records(records)


def _delegate(frames: pd.DataFrame) -> object:
    delegate = GUI.SourceSpecificMediaDelegate.__new__(
        GUI.SourceSpecificMediaDelegate
    )
    delegate.frames = frames.copy()
    delegate.frames["frame_index"] = pd.to_numeric(
        delegate.frames["frame_index"],
        errors="coerce",
    )
    delegate.video_cache = {}
    delegate._decode_cvat_full_frame = MethodType(
        lambda self, actor: Image.new("RGB", (120, 80), "white"),
        delegate,
    )
    return delegate


def _legacy_delegate(crop: Image.Image | None) -> object:
    delegate = _delegate(
        pd.DataFrame(
            [
                {
                    "source_type": "legacy_recovered",
                    "dataset_id": "dataset",
                    "video_key": "video",
                    "frame_index": frame,
                    "pig_id": "pig_actor",
                    "track_id": "track_actor",
                    "object_track_key": "actor",
                    "x1": 1,
                    "y1": 1,
                    "x2": 8,
                    "y2": 8,
                }
                for frame in range(10, 26)
            ]
        )
    )
    delegate._legacy_crop_image = MethodType(
        lambda self, actor: None if crop is None else crop.copy(),
        delegate,
    )
    return delegate


def _v1_inputs(
    *,
    source_type: str = "legacy_recovered",
    targets: str = ",".join(str(frame) for frame in range(10, 26)),
    history: str = "10,11,12,13,14,15",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    blinded = pd.DataFrame(
        [
            {
                "calibration_item_id": "calibration_item_000001",
                "media_authority_key": "media_1",
                "frozen_subset": "CALIBRATION_DEVELOPMENT_SET",
                "presentation_order": 1,
                "sampling_config_hash": "a" * 64,
            }
        ]
    )
    media = pd.DataFrame(
        [
            {
                "calibration_item_id": "calibration_item_000001",
                "media_authority_key": "media_1",
                "review_unit_id": "review_1",
                "temporal_unit_key": "temporal_1",
                "source_type": source_type,
                "dataset_id": "dataset",
                "video_key": "video",
                "recording_date": "2026-01-01",
                "object_track_key": "actor",
                "pig_id": "pig_actor",
                "track_id": "track_actor",
                "unit_start_frame": int(targets.split(",")[0]),
                "unit_end_frame": int(targets.split(",")[-1]),
                "display_frame_indices": targets,
                "review_pig_history_display_frame_indices": history,
            }
        ]
    )
    return blinded, media


def test_declared_and_effective_runtime_contracts_match() -> None:
    assert GUI.runtime_contract_matches_declared()
    assert (
        GUI.effective_runtime_presentation_contract_v2()
        == canonical_presentation_contract_v2()
    )


def test_cvat_actor_one_neighbor_is_red_and_neutral() -> None:
    image = Image.new("RGB", (120, 80), "white")
    rendered = render_neutral_context_v2(
        image,
        _frame_rows([10], neighbors=1),
        actor_identity="object:actor",
    )
    assert rendered.getpixel((10, 20)) == (255, 0, 0)
    assert rendered.getpixel((40, 20)) == (127, 127, 127)
    assert ACTOR_COLOR == "#ff0000"
    assert NEUTRAL_NEIGHBOR_COLOR == "#7f7f7f"


def test_cvat_actor_multiple_neighbors_are_all_neutral() -> None:
    image = Image.new("RGB", (140, 80), "white")
    rendered = render_neutral_context_v2(
        image,
        _frame_rows([10], neighbors=3),
        actor_identity="object:actor",
    )
    for x1 in (40, 65, 90):
        assert rendered.getpixel((x1, 20)) == (127, 127, 127)
    colors = rendered.getcolors(rendered.width * rendered.height)
    assert colors is not None
    assert (0, 176, 80) not in {color for _, color in colors}


def test_cvat_actor_without_neighbor_is_valid() -> None:
    rendered = render_neutral_context_v2(
        Image.new("RGB", (120, 80), "white"),
        _frame_rows([10], neighbors=0),
        actor_identity="object:actor",
    )
    assert rendered.getpixel((10, 20)) == (255, 0, 0)


def test_cvat_target_and_history_are_visibly_separated() -> None:
    delegate = _delegate(_frame_rows(list(range(4, 16)), neighbors=1))
    unit = _unit(history="4,5,6,7,8,9")
    sheet, diagnostics, audits = delegate.make_source_specific_sheet(unit)
    assert not diagnostics
    assert [item["role"] for item in audits[:6]] == ["CONTEXT"] * 6
    assert [item["role"] for item in audits[6:]] == ["TARGET"] * 6
    assert sheet.width > 0 and sheet.height > 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("presentation_template", ""),
        ("presentation_template", "unknown"),
        ("render_mode", "unknown"),
    ],
)
def test_missing_or_unknown_dispatch_fails_closed(
    field: str,
    value: str,
) -> None:
    unit = _unit()
    unit[field] = value
    delegate = _delegate(_frame_rows(list(range(10, 16))))
    with pytest.raises(SourceSpecificPresentationError):
        delegate.make_source_specific_sheet(unit)


def test_cvat_incomplete_target_and_missing_actor_fail_closed() -> None:
    blinded, media = _v1_inputs(
        source_type="cvat_tracking_xml",
        targets="10,11,12,13,14",
        history="",
    )
    built = build_media_authority_v2(
        blinded,
        media,
        producer_sha="b" * 40,
        input_hashes={},
    )
    assert not bool(built.iloc[0]["render_available"])
    assert "target_frame_count" in built.iloc[0]["render_failure_reason"]
    delegate = _delegate(_frame_rows([10, 11, 12, 13, 14]))
    with pytest.raises(SourceSpecificPresentationError):
        delegate.make_source_specific_sheet(_unit())


def test_legacy_complete_actor_crop_is_direct_and_overlay_free() -> None:
    crop = Image.new("RGB", (32, 24), (12, 34, 56))
    delegate = _legacy_delegate(crop)
    unit = _unit(
        source_type="legacy_recovered",
        targets=",".join(str(frame) for frame in range(10, 26)),
    )
    actor = delegate.frames.iloc[0]
    rendered, audit = delegate.render_source_frame(unit, actor)
    assert rendered.tobytes() == crop.tobytes()
    assert audit["runtime_renderer"] == LEGACY_RENDER_MODE
    assert audit["legacy_direct_crop"]
    assert (255, 0, 0) not in {
        color for _, color in (rendered.getcolors(1000) or [])
    }


def test_legacy_notice_and_actor_crop_only_public_text() -> None:
    text = public_display_text_v2(
        item_number=1,
        item_count=10,
        calibration_item_id="calibration_item_000001",
        target_count=16,
        context_count=0,
        context_mode=LEGACY_CONTEXT_MODE,
    )
    assert LEGACY_NOTICE_TEXT in text
    for token in (
        "fight",
        "candidate",
        "reason",
        "score",
        "selector",
        "stratum",
    ):
        assert token not in text.casefold()


def test_legacy_overlapping_history_is_removed_without_duplicate_display() -> None:
    blinded, media = _v1_inputs()
    built = build_media_authority_v2(
        blinded,
        media,
        producer_sha="b" * 40,
        input_hashes={},
    )
    row = built.iloc[0]
    assert row["history_frame_indices"] == ""
    assert row["target_frame_count"] == 16
    assert row["display_frame_count"] == 16
    assert len(row["display_frame_indices"].split(",")) == 16


def test_legacy_missing_crop_incomplete_burst_and_invalid_order_fail() -> None:
    unit = _unit(
        source_type="legacy_recovered",
        targets=",".join(str(frame) for frame in range(10, 26)),
    )
    delegate = _legacy_delegate(None)
    with pytest.raises(SourceSpecificPresentationError):
        delegate.make_source_specific_sheet(unit)
    blinded, media = _v1_inputs(targets="10,11,12")
    incomplete = build_media_authority_v2(
        blinded,
        media,
        producer_sha="b" * 40,
        input_hashes={},
    )
    assert not bool(incomplete.iloc[0]["render_available"])
    blinded, media = _v1_inputs(
        targets="10,12,11,13,14,15,16,17,18,19,20,21,22,23,24,25"
    )
    invalid_order = build_media_authority_v2(
        blinded,
        media,
        producer_sha="b" * 40,
        input_hashes={},
    )
    assert "invalid_target_frame_order" in (
        invalid_order.iloc[0]["render_failure_reason"]
    )


def test_contact_sheet_rejects_duplicate_or_reordered_frames() -> None:
    crop = Image.new("RGB", (20, 20), "white")
    with pytest.raises(SourceSpecificPresentationError):
        compose_source_specific_contact_sheet(
            [
                ("TARGET", 2, crop, "ok"),
                ("TARGET", 1, crop, "ok"),
            ],
            context_mode=LEGACY_CONTEXT_MODE,
        )
    with pytest.raises(SourceSpecificPresentationError):
        compose_source_specific_contact_sheet(
            [
                ("TARGET", 1, crop, "ok"),
                ("TARGET", 1, crop, "ok"),
            ],
            context_mode=LEGACY_CONTEXT_MODE,
        )


def test_hash_binds_notice_and_every_legacy_semantic_field() -> None:
    contract = canonical_presentation_contract_v2()
    original = presentation_semantic_hash_v2(contract)
    legacy_fields = (
        "render_mode",
        "actor_identity_semantics",
        "neighbor_context_available",
        "full_frame_context_available",
        "visible_notice",
    )
    for field in legacy_fields:
        changed = deepcopy(contract)
        value = changed["source_modes"][LEGACY_CONTEXT_MODE][field]
        changed["source_modes"][LEGACY_CONTEXT_MODE][field] = (
            not value if isinstance(value, bool) else f"{value}_changed"
        )
        assert presentation_semantic_hash_v2(changed) != original
    assert original != (
        "9eba97958b100e18bef7e8a216e0bd890"
        "e4f7cb2e6777c1eda90c6641b76fd3f"
    )


def test_render_is_deterministic_and_never_writes_decisions(
    tmp_path: Path,
) -> None:
    unit = _unit()
    delegate = _delegate(_frame_rows(list(range(10, 16)), neighbors=2))
    first, _, _ = delegate.make_source_specific_sheet(unit)
    second, _, _ = delegate.make_source_specific_sheet(unit)
    assert first.tobytes() == second.tobytes()
    assert list(tmp_path.iterdir()) == []


def test_rendered_image_cache_is_bounded_and_returns_copies() -> None:
    cache = GUI._MEDIA.RenderedImageCache(max_items=2)
    white = Image.new("RGB", (8, 8), "white")
    cache.put("first", white, metadata=("first",))
    white.paste("red", (0, 0, 1, 1))

    first = cache.get("first")
    assert first is not None
    first_image, first_metadata = first
    assert first_image.getpixel((0, 0)) == (255, 255, 255)
    assert first_metadata == ("first",)

    cache.put("second", Image.new("RGB", (8, 8), "green"))
    cache.put("third", Image.new("RGB", (8, 8), "blue"))

    assert len(cache) == 2
    assert cache.get("first") is None
    assert cache.get("second") is not None
    assert cache.get("third") is not None


def test_v2_display_sheet_cache_preserves_decisions_and_avoids_rerender() -> None:
    class FakeMedia:
        def __init__(self) -> None:
            self.calls = 0

        def make_source_specific_sheet(
            self,
            unit: pd.Series,
        ) -> tuple[Image.Image, list[str], list[dict[str, object]]]:
            self.calls += 1
            return Image.new("RGB", (1600, 1000), "white"), [], []

    gui = GUI.SourceSpecificCalibrationGui.__new__(
        GUI.SourceSpecificCalibrationGui
    )
    gui.sheet_cache = GUI._MEDIA.RenderedImageCache(max_items=2)
    gui.media = FakeMedia()
    gui.decisions = {"review_1": {"reviewed_behavior": "fight"}}
    original_decisions = gui.decisions.copy()
    unit = _unit()

    first = gui._display_sheet_for_unit(unit)
    first.paste("red", (0, 0, 1, 1))
    second = gui._display_sheet_for_unit(unit)

    assert gui.media.calls == 1
    assert second.size == (1000, 625)
    assert second.getpixel((0, 0)) == (255, 255, 255)
    assert gui.decisions == original_decisions


def test_v2_prefetch_failure_leaves_item_for_interactive_fallback() -> None:
    class FailingMedia:
        def __init__(self) -> None:
            self.calls = 0

        def make_source_specific_sheet(
            self,
            unit: pd.Series,
        ) -> tuple[Image.Image, list[str], list[dict[str, object]]]:
            self.calls += 1
            raise SourceSpecificPresentationError("missing media")

    gui = GUI.SourceSpecificCalibrationGui.__new__(
        GUI.SourceSpecificCalibrationGui
    )
    gui.sheet_cache = GUI._MEDIA.RenderedImageCache(max_items=2)
    gui.media = FailingMedia()
    gui.units = pd.DataFrame([_unit()])
    gui.decisions = {"review_1": {"reviewed_behavior": "fight"}}
    gui._prefetch_after_id = "idle-1"

    gui._prefetch_sheet(0)

    assert gui.media.calls == 1
    assert gui.sheet_cache.get("review_1") is None
    assert gui.decisions == {"review_1": {"reviewed_behavior": "fight"}}


def test_v2_sequential_frames_do_not_repeat_expensive_video_seeks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.seek_calls: list[int] = []

        def isOpened(self) -> bool:
            return True

        def set(self, property_id: int, frame_index: int) -> None:
            self.seek_calls.append(frame_index)

        def read(self) -> tuple[bool, object]:
            return True, GUI.np.zeros((4, 4, 3), dtype=GUI.np.uint8)

    capture = FakeCapture()

    class FakeCv2:
        CAP_PROP_POS_FRAMES = 1
        COLOR_BGR2RGB = 2

        @staticmethod
        def VideoCapture(path: str) -> FakeCapture:
            return capture

        @staticmethod
        def cvtColor(frame: object, conversion: int) -> object:
            return frame

    monkeypatch.setattr(GUI._MEDIA, "cv2", FakeCv2)
    delegate = GUI.SourceSpecificMediaDelegate.__new__(
        GUI.SourceSpecificMediaDelegate
    )
    delegate.video_cache = {}
    delegate.video_next_frame = {}
    delegate._resolve_video_path = MethodType(
        lambda self, actor: Path("synthetic.mp4"),
        delegate,
    )

    delegate._decode_cvat_full_frame(pd.Series({"frame_index": 10}))
    delegate._decode_cvat_full_frame(pd.Series({"frame_index": 11}))
    delegate._decode_cvat_full_frame(pd.Series({"frame_index": 15}))

    assert capture.seek_calls == [10, 15]


def test_four_calibration_outcomes_remain_distinct() -> None:
    assert (
        derive_calibration_outcome(
            provisional_behavior="fight",
            reviewed_behavior="social-nose",
            visual_reviewability="reviewable",
        )
        == "CORRECTION_REQUIRED"
    )
    assert (
        derive_calibration_outcome(
            provisional_behavior="fight",
            reviewed_behavior="fight",
            visual_reviewability="reviewable",
        )
        == "LABEL_SUPPORTED"
    )
    assert (
        derive_calibration_outcome(
            provisional_behavior="fight",
            reviewed_behavior="unclear",
            visual_reviewability="visually_unresolved",
        )
        == "VISUALLY_UNRESOLVED"
    )
    assert (
        derive_calibration_outcome(
            provisional_behavior="fight",
            reviewed_behavior="unreviewable",
            visual_reviewability="technical_authority_defect",
        )
        == "TECHNICAL_AUTHORITY_DEFECT"
    )


def test_serialized_false_availability_does_not_become_true() -> None:
    media = pd.DataFrame(
        [
            _unit(
                source_type="legacy_recovered",
                targets=",".join(str(frame) for frame in range(10, 26)),
            )
        ]
    )
    media["neighbor_context_available"] = "False"
    media["full_frame_context_available"] = "False"
    audit = validate_media_authority_v2(media)
    assert audit["valid"]
