from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
from PIL import Image

from pig_behavior.classification_v2.review.source_specific_blinded_presentation_v2 import (
    PRESENTATION_SEMANTIC_HASH,
    presentation_semantic_hash_v2,
)

ACCEPTED_PRESENTATION_V2_HASH = (
    "71d63afef4e0084d5abf51f19a4a07c6acaeceb15f470d6ee2dd79b3e5c8be38"
)
EXPECTED_TARGET_FRAMES = list(range(12, 28))
OVERLAPPING_HISTORY_INPUT = list(range(12, 18))


def _load_gui_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "classification_v2"
        / "01_review_units_gui"
        / "review_temporal_unit_gui.py"
    )
    spec = importlib.util.spec_from_file_location(
        "legacy_noninteraction_scope_gui",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _legacy_unit(targets: list[int] | None = None) -> pd.Series:
    target_frames = EXPECTED_TARGET_FRAMES if targets is None else targets
    return pd.Series(
        {
            "review_unit_id": "legacy-unit",
            "source_type": "legacy_recovered",
            "unit_start_frame": 12,
            "unit_end_frame": 27,
            "display_frame_indices": ",".join(
                str(value) for value in target_frames
            ),
            "review_pig_history_display_frame_indices": ",".join(
                str(value) for value in OVERLAPPING_HISTORY_INPUT
            ),
            "review_pig_history_available_ratio": 1.0,
        }
    )


def _gui_without_tk(
    module: ModuleType,
    tmp_path: Path,
) -> object:
    gui = module.ReviewUnitGui.__new__(module.ReviewUnitGui)
    gui.config = module.GuiConfig(
        review_units_csv=tmp_path / "unused_units.csv",
        frame_features_csv=tmp_path / "unused_frames.csv",
        output_dir=tmp_path / "unused_output",
        raw_root=tmp_path,
    )
    return gui


def test_legacy_scope_uses_all_sixteen_targets_and_zero_history() -> None:
    module = _load_gui_module()
    gui = module.ReviewUnitGui.__new__(module.ReviewUnitGui)
    unit = _legacy_unit()

    targets = gui._display_frames(unit)
    history = gui._history_display_frames(unit)
    displayed = gui._all_display_frames(unit)

    assert module.legacy_noninteraction_scope_errors(unit) == []
    assert targets == EXPECTED_TARGET_FRAMES
    assert history == []
    assert displayed == EXPECTED_TARGET_FRAMES
    assert len(displayed) - len(set(displayed)) == 0
    assert displayed == sorted(displayed)
    assert all(
        gui._display_frame_role(unit, frame_index) == "T"
        for frame_index in EXPECTED_TARGET_FRAMES[:6]
    )


def test_legacy_heading_never_calls_a_target_frame_history() -> None:
    module = _load_gui_module()
    gui = module.ReviewUnitGui.__new__(module.ReviewUnitGui)
    unit = _legacy_unit()

    heading = module.review_scope_heading(unit)

    assert heading == "DECISION TARGET — ALL 16 FRAMES"
    assert "CONTEXT" not in heading
    assert "HISTORY" not in heading
    assert "NOT DECISION TARGET" not in heading
    assert {
        gui._display_frame_role(unit, frame_index)
        for frame_index in EXPECTED_TARGET_FRAMES
    } == {"T"}


def test_incomplete_or_misordered_legacy_scope_fails_closed() -> None:
    module = _load_gui_module()

    incomplete = module.legacy_noninteraction_scope_errors(
        _legacy_unit(EXPECTED_TARGET_FRAMES[:-1])
    )
    misordered_frames = EXPECTED_TARGET_FRAMES.copy()
    misordered_frames[0], misordered_frames[1] = (
        misordered_frames[1],
        misordered_frames[0],
    )
    misordered = module.legacy_noninteraction_scope_errors(
        _legacy_unit(misordered_frames)
    )
    duplicated = module.legacy_noninteraction_scope_errors(
        _legacy_unit(
            EXPECTED_TARGET_FRAMES[:-1] + [EXPECTED_TARGET_FRAMES[-2]]
        )
    )

    assert "target_frame_count=15 expected=16" in incomplete
    assert "target_frame_order_not_chronological" in misordered
    assert "target_frames_do_not_match_native_unit_bounds" in misordered
    assert "duplicate_target_frame_indices=1" in duplicated


def test_gui_loader_rejects_an_incomplete_legacy_burst(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_gui_module()
    manifest = tmp_path / "legacy_units.csv"
    pd.DataFrame(
        [
            {
                "review_unit_id": "legacy-unit",
                "review_unit_type": "legacy_sequence",
                "source_type": "legacy_recovered",
                "dataset_id": "legacy",
                "video_key": "video",
                "pig_id": "pig",
                "unit_start_frame": 12,
                "unit_end_frame": 27,
                "display_frame_indices": ",".join(
                    str(value) for value in EXPECTED_TARGET_FRAMES[:-1]
                ),
                "behavior_label": "stand",
            }
        ]
    ).to_csv(manifest, index=False)
    monkeypatch.setattr(
        module,
        "audit_review_unit_contract",
        lambda _units: {"errors": []},
    )
    monkeypatch.setattr(
        module,
        "validate_candidate_gui_manifest",
        lambda _units: [],
    )
    gui = module.ReviewUnitGui.__new__(module.ReviewUnitGui)
    gui.config = module.GuiConfig(
        review_units_csv=manifest,
        frame_features_csv=tmp_path / "unused.csv",
        output_dir=tmp_path / "output",
        source_type="legacy_recovered",
    )

    with pytest.raises(
        SystemExit,
        match="Legacy non-interaction target scope failed",
    ):
        gui._load_units(manifest)


def test_missing_legacy_crop_fails_closed(tmp_path: Path) -> None:
    module = _load_gui_module()
    gui = _gui_without_tk(module, tmp_path)
    unit = _legacy_unit()
    row = pd.Series(
        {
            "frame_index": EXPECTED_TARGET_FRAMES[0],
            "crop_path": str(tmp_path / "missing.png"),
        }
    )

    _, diagnostic = gui._image_for_row(unit, row)

    assert diagnostic == "missing_legacy_crop_path"


def test_cvat_target_history_behavior_is_unchanged() -> None:
    module = _load_gui_module()
    gui = module.ReviewUnitGui.__new__(module.ReviewUnitGui)
    history = list(range(4, 10))
    targets = list(range(10, 16))
    unit = pd.Series(
        {
            "source_type": "cvat_tracking_xml",
            "display_frame_indices": ",".join(
                str(value) for value in targets
            ),
            "review_pig_history_display_frame_indices": ",".join(
                str(value) for value in history
            ),
            "review_pig_history_available_ratio": 1.0,
        }
    )

    assert gui._history_display_frames(unit) == history
    assert gui._all_display_frames(unit) == history + targets
    assert gui._display_frame_role(unit, history[0]) == "H"
    assert gui._display_frame_role(unit, targets[0]) == "T"
    assert module.review_scope_heading(unit) == ""


def test_legacy_contact_sheet_render_is_deterministic(
    tmp_path: Path,
) -> None:
    module = _load_gui_module()
    gui = _gui_without_tk(module, tmp_path)
    crop = tmp_path / "actor.png"
    Image.new("RGB", (32, 24), (31, 47, 59)).save(crop)
    rows = pd.DataFrame(
        [
            {
                "frame_index": frame_index,
                "crop_path": str(crop),
            }
            for frame_index in EXPECTED_TARGET_FRAMES
        ]
    )
    unit = _legacy_unit()

    first, first_diagnostics = gui._make_contact_sheet(unit, rows)
    second, second_diagnostics = gui._make_contact_sheet(unit, rows)

    assert first_diagnostics == []
    assert second_diagnostics == []
    assert first.size == second.size
    assert first.tobytes() == second.tobytes()


def test_interaction_calibration_presentation_v2_hash_is_unchanged() -> None:
    assert PRESENTATION_SEMANTIC_HASH == ACCEPTED_PRESENTATION_V2_HASH
    assert presentation_semantic_hash_v2() == ACCEPTED_PRESENTATION_V2_HASH
