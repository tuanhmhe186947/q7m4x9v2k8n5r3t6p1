from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = ROOT / "scripts" / "classification_v2" / "01_review_units_gui"
SOURCE_DIR = ROOT / "scripts" / "classification_v2" / "00_source_feature_temporal"


def _load_script(name: str, filename: str):
    """Load one operator script without executing its CLI entrypoint."""
    spec = importlib.util.spec_from_file_location(name, REVIEW_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script: {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_source_script(name: str, filename: str):
    """Load a source/temporal checker without executing its CLI."""
    spec = importlib.util.spec_from_file_location(name, SOURCE_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source script: {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _decision_row(unit_id: str, decision: str = "accept") -> dict[str, object]:
    """Return the complete GUI decision schema for focused audit tests."""
    coverage = _load_script(
        "review_decision_coverage_schema",
        "check_review_unit_decision_coverage.py",
    )
    row = {column: "" for column in coverage.REQUIRED_COLUMNS}
    row.update(
        {
            "review_unit_id": unit_id,
            "manual_review_decision": decision,
            "behavior_label": "stand",
            "original_behavior": "stand",
        }
    )
    if decision == "accept":
        row["manual_label_strength"] = "strong"
        row["manual_training_action"] = "main_train"
        row["manual_sample_weight"] = 1.0
    elif decision == "corrected":
        row["manual_label_strength"] = "medium"
        row["manual_training_action"] = "correct_and_keep"
        row["manual_sample_weight"] = 1.0
    elif decision == "exclude":
        row["manual_label_strength"] = "boundary"
        row["manual_training_action"] = "exclude"
        row["manual_sample_weight"] = 0.0
    return row


def _native_unit(source: str, start: int, behavior: str = "stand") -> dict[str, object]:
    """Create one complete canonical native review-unit record."""
    if source == "cvat_tracking_xml":
        count = 6
        unit_type = "cvat_interval_6"
        scope = "cvat_interval_6f"
    else:
        count = 16
        unit_type = "legacy_burst_16"
        scope = "whole_legacy_burst_16f"
    end = start + count - 1
    unit_id = f"{source}|unit={start}-{end}"
    return {
        "review_item_id": f"item-{start}",
        "review_unit_id": unit_id,
        "review_unit_type": unit_type,
        "temporal_unit_key": unit_id,
        "source_type": source,
        "dataset_id": "dataset-a",
        "video_key": "video-a",
        "pig_id": "ID_1",
        "track_id": "1",
        "object_track_key": f"{source}|dataset-a|video-a|track=1|pig=ID_1",
        "unit_start_frame": start,
        "unit_end_frame": end,
        "unit_frame_count": count,
        "display_frame_indices": ",".join(str(value) for value in range(start, end + 1)),
        "review_template": "motion",
        "behavior_label": behavior,
        "apply_scope": scope,
    }


def _decision_for_unit(
    unit: dict[str, object],
    decision: str,
) -> dict[str, object]:
    row = _decision_row(str(unit["review_unit_id"]), decision)
    for column in [
        "review_item_id",
        "review_unit_id",
        "review_unit_type",
        "temporal_unit_key",
        "source_type",
        "dataset_id",
        "video_key",
        "pig_id",
        "track_id",
        "object_track_key",
        "unit_start_frame",
        "unit_end_frame",
        "display_frame_indices",
        "review_template",
        "behavior_label",
        "apply_scope",
    ]:
        row[column] = unit[column]
    row["original_behavior"] = unit["behavior_label"]
    return row


def test_gui_resumes_existing_decisions(tmp_path: Path) -> None:
    gui_module = _load_script("review_gui_resume", "review_temporal_unit_gui.py")
    output_dir = tmp_path / "review"
    output_dir.mkdir()
    pd.DataFrame([_decision_row("unit_a")]).to_csv(
        output_dir / "behavior_unit_review_decisions.csv",
        index=False,
    )

    gui = gui_module.ReviewUnitGui.__new__(gui_module.ReviewUnitGui)
    gui.config = gui_module.GuiConfig(
        review_units_csv=tmp_path / "units.csv",
        frame_features_csv=tmp_path / "frames.csv",
        output_dir=output_dir,
    )
    gui.units = pd.DataFrame({"review_unit_id": ["unit_a"]})

    decisions = gui._load_existing_decisions()
    assert list(decisions) == ["unit_a"]
    assert decisions["unit_a"]["manual_review_decision"] == "accept"


def test_gui_rejects_duplicate_existing_decisions(tmp_path: Path) -> None:
    gui_module = _load_script("review_gui_duplicate", "review_temporal_unit_gui.py")
    output_dir = tmp_path / "review"
    output_dir.mkdir()
    pd.DataFrame([_decision_row("unit_a"), _decision_row("unit_a")]).to_csv(
        output_dir / "behavior_unit_review_decisions.csv",
        index=False,
    )
    gui = gui_module.ReviewUnitGui.__new__(gui_module.ReviewUnitGui)
    gui.config = gui_module.GuiConfig(
        review_units_csv=tmp_path / "units.csv",
        frame_features_csv=tmp_path / "frames.csv",
        output_dir=output_dir,
    )
    gui.units = pd.DataFrame({"review_unit_id": ["unit_a"]})

    with pytest.raises(SystemExit, match="duplicate review_unit_id"):
        gui._load_existing_decisions()


def test_interaction_gui_draws_actor_and_nearest_partner_on_full_frame() -> None:
    gui_module = _load_script(
        "review_gui_interaction_context",
        "review_temporal_unit_gui.py",
    )
    gui = gui_module.ReviewUnitGui.__new__(gui_module.ReviewUnitGui)
    gui.frames = pd.DataFrame(
        [
            {
                "source_type": "cvat_tracking_xml",
                "dataset_id": "dataset-a",
                "video_key": "video-a",
                "frame_index": 0,
                "object_track_key": "actor",
                "pig_id": "ID_1",
                "x1": 10,
                "y1": 10,
                "x2": 30,
                "y2": 30,
            },
            {
                "source_type": "cvat_tracking_xml",
                "dataset_id": "dataset-a",
                "video_key": "video-a",
                "frame_index": 0,
                "object_track_key": "partner",
                "pig_id": "ID_2",
                "x1": 40,
                "y1": 10,
                "x2": 60,
                "y2": 30,
            },
        ]
    )
    actor = gui.frames.iloc[0]
    image = gui._interaction_context_image(np.zeros((80, 120, 3), dtype=np.uint8), actor)

    assert image.size == (120, 80)
    assert image.getpixel((10, 10)) == (255, 0, 0)
    assert image.getpixel((40, 10)) == (0, 176, 80)


def test_decision_coverage_requires_no_missing_or_pending() -> None:
    coverage = _load_script(
        "review_decision_coverage",
        "check_review_unit_decision_coverage.py",
    )
    manifest = pd.DataFrame({"review_unit_id": ["unit_a", "unit_b"]})
    incomplete = pd.DataFrame([_decision_row("unit_a", "pending")])

    audit = coverage.audit_decision_coverage(
        manifest,
        incomplete,
        require_complete=True,
    )
    assert "missing_review_unit_count=1" in audit["errors"]
    assert "pending_review_unit_count=1" in audit["errors"]


def test_decision_coverage_accepts_complete_unique_review() -> None:
    coverage = _load_script(
        "review_decision_coverage_complete",
        "check_review_unit_decision_coverage.py",
    )
    manifest = pd.DataFrame({"review_unit_id": ["unit_a", "unit_b"]})
    decisions = pd.DataFrame(
        [_decision_row("unit_a", "accept"), _decision_row("unit_b", "exclude")]
    )

    audit = coverage.audit_decision_coverage(
        manifest,
        decisions,
        require_complete=True,
    )
    assert audit["errors"] == []
    assert audit["covered_review_units"] == 2


def test_complete_review_rejects_review_later_action() -> None:
    coverage = _load_script(
        "review_decision_coverage_review_later",
        "check_review_unit_decision_coverage.py",
    )
    manifest = pd.DataFrame({"review_unit_id": ["unit_a"]})
    row = _decision_row("unit_a", "accept")
    row["manual_training_action"] = "review_later"

    audit = coverage.audit_decision_coverage(
        manifest,
        pd.DataFrame([row]),
        require_complete=True,
    )
    assert "review_later_unit_count=1" in audit["errors"]


def test_apply_action_aliases_are_fail_closed() -> None:
    apply_module = _load_script(
        "review_apply_action_aliases",
        "classification_v2_apply_review_unit_decisions.py",
    )
    assert apply_module._to_bool_action("review_later", "accept") is False
    assert apply_module._default_weight("accept", "review_later") == 0.0
    assert apply_module._default_weight("accept", "low_weight_train") == 0.5


def test_apply_corrected_cvat_decision_has_exact_six_frame_scope() -> None:
    apply_module = _load_script(
        "review_apply_cvat_scope",
        "classification_v2_apply_review_unit_decisions.py",
    )
    unit = _native_unit("cvat_tracking_xml", 1020)
    frames = pd.DataFrame(
        {
            "temporal_unit_key": [unit["temporal_unit_key"]] * 6,
            "source_type": [unit["source_type"]] * 6,
            "frame_index": list(range(1020, 1026)),
            "behavior": ["stand"] * 6,
        }
    )
    decision = _decision_for_unit(unit, "corrected")
    decision["manual_corrected_behavior"] = "move"
    normalized, errors, _ = apply_module.normalize_decisions(pd.DataFrame([decision]))
    assert errors == []

    reviewed, audit = apply_module.apply_decisions_to_frames(
        frames,
        pd.DataFrame([unit]),
        normalized,
    )
    assert len(reviewed) == len(frames)
    assert reviewed["behavior"].eq("move").all()
    assert audit["changed_behavior_frames"] == 6
    assert audit["missing_review_unit_count"] == 0
    checker = _load_script(
        "review_apply_scope_checker",
        "check_apply_review_unit_decisions_output.py",
    )
    assert checker._applied_scope_errors(reviewed, pd.DataFrame([unit])) == []
    incomplete = reviewed.iloc[:-1].copy()
    scope_errors = checker._applied_scope_errors(incomplete, pd.DataFrame([unit]))
    assert f"applied_frame_scope_mismatch={unit['review_unit_id']}" in scope_errors


def test_apply_exclusion_has_exact_legacy_sixteen_frame_scope() -> None:
    apply_module = _load_script(
        "review_apply_legacy_scope",
        "classification_v2_apply_review_unit_decisions.py",
    )
    unit = _native_unit("legacy_recovered", 32)
    frames = pd.DataFrame(
        {
            "temporal_unit_key": [unit["temporal_unit_key"]] * 16,
            "source_type": [unit["source_type"]] * 16,
            "frame_index": list(range(32, 48)),
            "behavior": ["stand"] * 16,
        }
    )
    decision = _decision_for_unit(unit, "exclude")
    normalized, errors, _ = apply_module.normalize_decisions(pd.DataFrame([decision]))
    assert errors == []

    reviewed, audit = apply_module.apply_decisions_to_frames(
        frames,
        pd.DataFrame([unit]),
        normalized,
    )
    assert len(reviewed) == len(frames)
    assert (~reviewed["review_include_in_training"]).all()
    assert audit["excluded_frames"] == 16


def test_duplicate_active_decisions_fail_instead_of_keep_last() -> None:
    apply_module = _load_script(
        "review_apply_duplicate_fail",
        "classification_v2_apply_review_unit_decisions.py",
    )
    rows = pd.DataFrame([_decision_row("unit-a"), _decision_row("unit-a")])
    _, errors, _ = apply_module.normalize_decisions(rows)
    assert "duplicate_decision_rows=2" in errors


def test_review_unit_contract_rejects_wrong_cvat_length() -> None:
    from pig_behavior.classification_v2.review.behavior_review_contract import (
        audit_review_unit_contract,
    )

    unit = _native_unit("cvat_tracking_xml", 1020)
    unit["unit_end_frame"] = 1026
    audit = audit_review_unit_contract(pd.DataFrame([unit]))
    assert "wrong_native_frame_count=cvat_tracking_xml:rows=1" in audit["errors"]


def test_legacy_series_fallback_preserves_scalar_rows() -> None:
    from pig_behavior.classification_v2.sources.legacy_recovered_csv import (
        _first_existing_series,
    )

    source = pd.DataFrame(index=[4, 7])
    fallback = pd.Series([11.0, 22.0], index=source.index)
    result = _first_existing_series(source, ["missing"], default=fallback)
    pd.testing.assert_series_equal(result, fallback)
    assert all(not isinstance(value, pd.Series) for value in result)


def test_cvat_anchor_case_is_checked_across_layers() -> None:
    checker = _load_source_script(
        "cvat_anchor_case_checker",
        "check_classification_v2_cvat_anchor_case.py",
    )
    video = "Pigs281119_000085_30fps"
    enhanced = pd.DataFrame(
        {
            "video_key": [video] * 6,
            "pig_id": ["ID_4"] * 6,
            "frame_index": list(range(1020, 1026)),
            "behavior": ["social-nose", "stand", "stand", "stand", "stand", "stand"],
        }
    )
    intervals = pd.DataFrame(
        {
            "video_key": [video],
            "pig_id": ["ID_4"],
            "label_window_start": [1020],
            "behavior_temporal_final": ["social-nose"],
        }
    )
    units = pd.DataFrame(
        {
            "video_key": [video],
            "pig_id": ["ID_4"],
            "unit_start_frame": [1020],
            "behavior_label": ["social-nose"],
            "review_template": ["interaction"],
        }
    )

    audit = checker.audit_anchor_case(
        enhanced,
        intervals,
        units,
        video_key=video,
        pig_id="ID_4",
        anchor=1020,
        expected_behavior="social-nose",
        expected_template="interaction",
    )
    assert audit["valid"] is True
    assert audit["errors"] == []
    json.dumps(audit)


def test_publication_split_writes_explicit_train_ready_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = ROOT / "scripts" / "classification_v2" / "02_train_ready_exports"
    spec = importlib.util.spec_from_file_location(
        "publication_split_explicit_output",
        script / "classification_v2_build_publication_folds.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load publication-fold builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    manifest = tmp_path / "windows.csv"
    pd.DataFrame(
        {
            "window_id": ["w1", "w2", "w3"],
            "behavior_window_label": ["stand", "lying", "eat"],
            "window_valid_for_main_train": [True, True, True],
            "source_type": ["cvat_tracking_xml"] * 3,
            "dataset_id": ["d1", "d2", "d3"],
            "video_key": [
                "Pigs281119_000085_30fps",
                "Pigs291119_000231_30fps",
                "Pigs301119_000327_30fps",
            ],
        }
    ).to_csv(manifest, index=False)
    output_dir = tmp_path / "protocol"
    split_path = tmp_path / "train_ready" / "split_manifest.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classification_v2_build_publication_folds.py",
            "--manifest-csv",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--split-output-csv",
            str(split_path),
            "--group-level",
            "recording_date",
        ],
    )

    module.main()
    split = pd.read_csv(split_path, low_memory=False)
    assert len(split) == 3
    assert split["window_id"].is_unique
    assert split["recording_group_id"].nunique() == 3
