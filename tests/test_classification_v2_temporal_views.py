from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.evaluation.temporal_shortcut_audit import (
    audit_temporal_view_shortcuts,
    write_temporal_shortcut_audit,
)
from pig_behavior.classification_v2.features.temporal_views import (
    MODEL_TENSOR_COLUMNS,
    build_temporal_views,
    write_temporal_view_outputs,
)


def test_temporal_views_preserve_windows_units_and_fixed_slot_identity() -> None:
    windows, frames, intervals = _inputs()

    result = build_temporal_views(windows, frames, intervals)

    assert len(result.selection_manifest) == len(windows)
    assert result.selection_manifest["window_id"].tolist() == windows["window_id"].tolist()
    assert int(result.selection_manifest["fixed6_keep"].sum()) == 2
    assert len(result.fixed6_observed_time_manifest) == 12
    assert len(result.fixed6_normalized_phase_manifest) == 12
    assert len(result.native6_16_manifest) == 22
    assert result.audit["all_native_units_in_fixed6"] is True
    assert result.audit["rows_dropped"] == 0
    assert result.audit["labels_changed"] == 0
    observed = result.fixed6_observed_time_manifest
    phase = result.fixed6_normalized_phase_manifest
    assert observed["slot_key"].tolist() == phase["slot_key"].tolist()
    assert not observed["padding_mask"].any()
    assert observed.groupby("view_item_id").size().eq(6).all()
    phase_values = phase.groupby("view_item_id", sort=False)["time_value"].apply(list)
    assert all(values == pytest.approx(np.linspace(0.0, 1.0, 6)) for values in phase_values)
    assert "source_type" not in MODEL_TENSOR_COLUMNS
    assert "source_native_length_audit" not in MODEL_TENSOR_COLUMNS


def test_temporal_views_are_deterministic_under_frame_reordering() -> None:
    windows, frames, intervals = _inputs()

    first = build_temporal_views(windows, frames, intervals)
    second = build_temporal_views(
        windows,
        frames.sample(frac=1.0, random_state=17).reset_index(drop=True),
        intervals,
    )

    pd.testing.assert_frame_equal(
        first.fixed6_observed_time_manifest,
        second.fixed6_observed_time_manifest,
    )
    pd.testing.assert_frame_equal(first.native6_16_manifest, second.native6_16_manifest)
    assert first.audit["fixed6_slot_key_sha256"] == second.audit["fixed6_slot_key_sha256"]


def test_contiguous_t6_may_cross_two_reviewed_native_intervals() -> None:
    intervals = pd.DataFrame(
        {
            "temporal_unit_key": ["unit-a", "unit-b"],
            "source_type": ["cvat_tracking_xml"] * 2,
            "object_track_key": ["track-a"] * 2,
            "label_window_start": [0, 6],
            "label_window_end": [5, 11],
            "label_frame_count": [6, 6],
        }
    )
    frames = pd.DataFrame(
        {
            "frame_uid": [f"frame-{value}" for value in range(12)],
            "source_type": ["cvat_tracking_xml"] * 12,
            "object_track_key": ["track-a"] * 12,
            "temporal_unit_key": ["unit-a"] * 6 + ["unit-b"] * 6,
            "frame_index": list(range(12)),
            "timestamp_sec": [value / 30.0 for value in range(12)],
            "bbox_valid": [True] * 12,
            "spatiotemporal_feature_valid": [True] * 12,
        }
    )
    window_id = "track-a|T6|3-8"
    windows = pd.DataFrame(
        {
            "window_id": [window_id],
            "source_type": ["cvat_tracking_xml"],
            "object_track_key": ["track-a"],
            "window_start_frame": [3],
            "window_end_frame": [8],
            "window_length_frames": [6],
            "temporal_unit_keys_json": ['["unit-a","unit-b"]'],
            "feature_computation_grain": ["FINAL_VIEW_FEATURES"],
            "pair_scope_key": [window_id],
            "view_type": ["T6_contiguous"],
            "sampling_pattern": ["contiguous"],
            "selected_frame_indices": ["[3,4,5,6,7,8]"],
            "pair_recomputed_for_view": [True],
            "aggregate_recomputed_for_view": [True],
        }
    )

    result = build_temporal_views(windows, frames, intervals)

    assert result.fixed6_observed_time_manifest[
        "temporal_unit_key"
    ].tolist() == ["unit-a"] * 3 + ["unit-b"] * 3


def test_missing_frame_is_masked_without_losing_expected_slot() -> None:
    windows, frames, intervals = _inputs()
    frames = frames.loc[~frames["frame_uid"].eq("legacy-frame-3")].reset_index(drop=True)

    result = build_temporal_views(windows, frames, intervals)

    fixed = result.fixed6_observed_time_manifest
    slot = fixed.loc[fixed["frame_index_expected_audit"].eq(3)].iloc[0]
    assert bool(slot["observed_mask"]) is False
    assert slot["frame_uid_audit"] == ""
    assert result.audit["fixed6_missing_observed_slots"] == 1
    assert len(result.native6_16_manifest) == 22


def test_duplicate_frame_alignment_is_rejected() -> None:
    windows, frames, intervals = _inputs()
    duplicate = frames.iloc[[0]].copy()
    duplicate["frame_uid"] = "another-frame-uid"
    frames = pd.concat([frames, duplicate], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate frame alignment"):
        build_temporal_views(windows, frames, intervals)


def test_duplicate_window_id_is_rejected() -> None:
    windows, frames, intervals = _inputs()
    windows.loc[1, "window_id"] = windows.loc[2, "window_id"]

    with pytest.raises(ValueError, match="duplicate_temporal_view_input_window_id_rows"):
        build_temporal_views(windows, frames, intervals)


def test_wrong_source_native_length_is_rejected() -> None:
    windows, frames, intervals = _inputs()
    intervals.loc[0, "label_window_end"] = 14
    intervals.loc[0, "label_frame_count"] = 15

    with pytest.raises(ValueError, match="source native-length contract mismatch"):
        build_temporal_views(windows, frames, intervals)


def test_frame_outside_declared_native_interval_is_rejected() -> None:
    windows, frames, intervals = _inputs()
    extra = frames.iloc[[0]].copy()
    extra["frame_uid"] = "legacy-frame-outside"
    extra["frame_index"] = 16
    frames = pd.concat([frames, extra], ignore_index=True)

    with pytest.raises(ValueError, match="frame rows outside native interval=1"):
        build_temporal_views(windows, frames, intervals)


def test_window_native_source_mismatch_is_rejected() -> None:
    windows, frames, intervals = _inputs()
    windows.loc[2, "source_type"] = "cvat_tracking_xml"

    with pytest.raises(ValueError, match="window/native source mismatch"):
        build_temporal_views(windows, frames, intervals)


def test_nonmonotonic_timestamps_are_rejected() -> None:
    windows, frames, intervals = _inputs()
    frames.loc[frames["frame_uid"].eq("legacy-frame-4"), "timestamp_sec"] = 0.01

    with pytest.raises(ValueError, match="nonmonotonic observed timestamps"):
        build_temporal_views(windows, frames, intervals)


def test_fixed6_requires_coverage_for_every_native_unit() -> None:
    windows, frames, intervals = _inputs()
    windows = windows.loc[~windows["window_id"].eq("legacy-fixed6")].reset_index(
        drop=True
    )

    with pytest.raises(ValueError, match="native_units_missing_from_fixed6=1"):
        build_temporal_views(windows, frames, intervals)


def test_temporal_view_writer_requires_explicit_overwrite(tmp_path: Path) -> None:
    result = build_temporal_views(*_inputs())
    paths = write_temporal_view_outputs(result, tmp_path)

    assert all(Path(path).exists() for path in paths.values())
    with pytest.raises(FileExistsError, match="already exist"):
        write_temporal_view_outputs(result, tmp_path)
    write_temporal_view_outputs(result, tmp_path, overwrite=True)


def test_structural_shortcut_audit_accepts_shared_fixed_patterns() -> None:
    result = build_temporal_views(*_inputs())

    audit = _shortcut_audit(result)

    assert audit["valid"] is True
    assert audit["fixed6_length_pattern_shared_across_sources"] is True
    assert audit["fixed6_padding_pattern_shared_across_sources"] is True
    assert audit["phase_timing_pattern_shared_across_sources"] is True
    assert audit["native_length_confound_expected"] is True
    assert audit["training_authorized"] is False


def test_shortcut_audit_rejects_sparse_six_as_primary_t6() -> None:
    result = build_temporal_views(*_inputs())
    selection = result.selection_manifest.copy()
    legacy = selection["window_id"].eq("legacy-fixed6")
    selection.loc[legacy, "view_type"] = "S6@16"
    selection.loc[legacy, "sampling_pattern"] = "sparse_0_3_6_9_12_15"

    audit = audit_temporal_view_shortcuts(
        selection,
        result.fixed6_observed_time_manifest,
        result.fixed6_normalized_phase_manifest,
        result.native6_16_manifest,
        result.contract,
    )

    assert audit["valid"] is False
    assert any(
        "selection_fixed6_keep_does_not_match_exact_T6_contiguous" in error
        for error in audit["errors"]
    )


def test_phase_timing_source_signature_stops_training() -> None:
    result = build_temporal_views(*_inputs())
    phase = result.fixed6_normalized_phase_manifest.copy()
    cvat = phase["source_type"].eq("cvat_tracking_xml")
    phase.loc[cvat, "time_value"] = phase.loc[cvat, "time_value"] + 0.01

    audit = audit_temporal_view_shortcuts(
        result.selection_manifest,
        result.fixed6_observed_time_manifest,
        phase,
        result.native6_16_manifest,
        result.contract,
    )

    assert audit["valid"] is False
    assert audit["training_stop_required"] is True
    assert any("normalized_phase:timing" in error for error in audit["errors"])


def test_reordered_phase_slots_are_rejected() -> None:
    result = build_temporal_views(*_inputs())
    phase = result.fixed6_normalized_phase_manifest.copy()
    phase = pd.concat([phase.iloc[[1]], phase.iloc[[0]], phase.iloc[2:]], ignore_index=True)

    audit = audit_temporal_view_shortcuts(
        result.selection_manifest,
        result.fixed6_observed_time_manifest,
        phase,
        result.native6_16_manifest,
        result.contract,
    )

    assert audit["valid"] is False
    assert any("slot_sequence_invalid" in error for error in audit["errors"])
    assert any("exact_membership_and_order" in error for error in audit["errors"])


def test_availability_shortcut_requires_versioned_mitigation() -> None:
    result = build_temporal_views(*_inputs())
    observed = result.fixed6_observed_time_manifest.copy()
    phase = result.fixed6_normalized_phase_manifest.copy()
    legacy = observed["source_type"].eq("legacy_recovered")
    observed.loc[legacy, "actor_context_available_mask"] = True
    phase.loc[legacy, "actor_context_available_mask"] = True

    blocked = audit_temporal_view_shortcuts(
        result.selection_manifest,
        observed,
        phase,
        result.native6_16_manifest,
        result.contract,
    )
    mitigated = audit_temporal_view_shortcuts(
        result.selection_manifest,
        observed,
        phase,
        result.native6_16_manifest,
        result.contract,
        mitigated_families=["availability"],
    )

    assert blocked["valid"] is False
    assert mitigated["valid"] is True
    assert any("declared_mitigated" in warning for warning in mitigated["warnings"])


def test_source_metadata_nearly_determining_behavior_is_rejected() -> None:
    result = build_temporal_views(*_inputs())
    selection = result.selection_manifest.copy()
    selected = selection["fixed6_keep"]
    selection.loc[selected & selection["source_type"].eq("legacy_recovered"),
                  "behavior_window_label"] = "stand"
    selection.loc[selected & selection["source_type"].eq("cvat_tracking_xml"),
                  "behavior_window_label"] = "drink"

    audit = audit_temporal_view_shortcuts(
        selection,
        result.fixed6_observed_time_manifest,
        result.fixed6_normalized_phase_manifest,
        result.native6_16_manifest,
        result.contract,
    )

    assert audit["valid"] is False
    assert any("nearly_determines_behavior" in error for error in audit["errors"])


def test_window_length_nearly_determining_behavior_is_rejected() -> None:
    result = build_temporal_views(*_inputs())
    selection = result.selection_manifest.copy()
    selection.loc[selection["window_length_frames"].eq(16),
                  "behavior_window_label"] = "lying"
    selection.loc[selection["window_length_frames"].eq(6),
                  "behavior_window_label"] = "stand"

    audit = audit_temporal_view_shortcuts(
        selection,
        result.fixed6_observed_time_manifest,
        result.fixed6_normalized_phase_manifest,
        result.native6_16_manifest,
        result.contract,
    )

    assert audit["valid"] is False
    report = audit["label_shortcut_reports"]["all_window_length_to_behavior"]
    assert report["near_direct_target_signature"] is True


def test_forbidden_model_tensor_metadata_is_rejected() -> None:
    result = build_temporal_views(*_inputs())
    contract = dict(result.contract)
    contract["model_tensor_columns"] = [*MODEL_TENSOR_COLUMNS, "source_type"]

    audit = audit_temporal_view_shortcuts(
        result.selection_manifest,
        result.fixed6_observed_time_manifest,
        result.fixed6_normalized_phase_manifest,
        result.native6_16_manifest,
        contract,
    )

    assert audit["valid"] is False
    assert any("forbidden_temporal_model_fields" in error for error in audit["errors"])


def test_shortcut_audit_writer_requires_overwrite(tmp_path: Path) -> None:
    result = build_temporal_views(*_inputs())
    audit = _shortcut_audit(result)
    output = tmp_path / "temporal_shortcut_audit.json"

    write_temporal_shortcut_audit(audit, output)

    assert json.loads(output.read_text(encoding="utf-8"))["valid"] is True
    with pytest.raises(FileExistsError, match="already exists"):
        write_temporal_shortcut_audit(audit, output)


def test_temporal_view_builder_cli_dry_run_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windows, frames, intervals = _inputs()
    window_csv = tmp_path / "windows.csv"
    frame_csv = tmp_path / "frames.csv"
    interval_csv = tmp_path / "intervals.csv"
    output_dir = tmp_path / "not-written"
    windows.to_csv(window_csv, index=False)
    frames.to_csv(frame_csv, index=False)
    intervals.to_csv(interval_csv, index=False)
    script = Path(
        "scripts/classification_v2/02_train_ready_exports/"
        "classification_v2_build_temporal_views.py"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--window-manifest",
            str(window_csv),
            "--harmonized-frame-csv",
            str(frame_csv),
            "--temporal-interval-csv",
            str(interval_csv),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
    )

    runpy.run_path(str(script), run_name="__main__")

    assert output_dir.exists() is False


def test_temporal_shortcut_cli_dry_run_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view_dir = tmp_path / "views"
    output_json = tmp_path / "not-written.json"
    write_temporal_view_outputs(build_temporal_views(*_inputs()), view_dir)
    script = Path(
        "scripts/classification_v2/02_train_ready_exports/"
        "check_classification_v2_temporal_view_shortcuts.py"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--temporal-view-dir",
            str(view_dir),
            "--output-json",
            str(output_json),
            "--dry-run",
        ],
    )

    runpy.run_path(str(script), run_name="__main__")

    assert output_json.exists() is False


def test_persisted_artifact_contract_detects_truncated_manifest(
    tmp_path: Path,
) -> None:
    view_dir = tmp_path / "views"
    write_temporal_view_outputs(build_temporal_views(*_inputs()), view_dir)
    contract = json.loads(
        (view_dir / "temporal_view_contract.json").read_text(encoding="utf-8")
    )
    selection = pd.read_csv(view_dir / "temporal_view_selection_manifest.csv")
    observed = pd.read_csv(view_dir / "fixed6_observed_time_manifest.csv").iloc[:-1]
    phase = pd.read_csv(view_dir / "fixed6_normalized_phase_manifest.csv")
    native = pd.read_csv(view_dir / "native6_16_manifest.csv")

    audit = audit_temporal_view_shortcuts(
        selection,
        observed,
        phase,
        native,
        contract,
        require_artifact_contract=True,
    )

    assert audit["valid"] is False
    assert any("artifact_contract_row_mismatch" in error for error in audit["errors"])


def _shortcut_audit(result):
    return audit_temporal_view_shortcuts(
        result.selection_manifest,
        result.fixed6_observed_time_manifest,
        result.fixed6_normalized_phase_manifest,
        result.native6_16_manifest,
        result.contract,
    )


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    intervals = pd.DataFrame(
        {
            "temporal_unit_key": ["legacy-unit", "cvat-unit"],
            "source_type": ["legacy_recovered", "cvat_tracking_xml"],
            "object_track_key": ["legacy-track", "cvat-track"],
            "label_window_start": [0, 100],
            "label_window_end": [15, 105],
            "label_frame_count": [16, 6],
        }
    )
    windows = pd.DataFrame(
        {
            "window_id": ["legacy-native16", "cvat-fixed6", "legacy-fixed6"],
            "source_type": [
                "legacy_recovered",
                "cvat_tracking_xml",
                "legacy_recovered",
            ],
            "object_track_key": ["legacy-track", "cvat-track", "legacy-track"],
            "window_start_frame": [0, 100, 0],
            "window_end_frame": [15, 105, 5],
            "window_length_frames": [16, 6, 6],
            "feature_computation_grain": ["FINAL_VIEW_FEATURES"] * 3,
            "pair_scope_key": [
                "legacy-native16",
                "cvat-fixed6",
                "legacy-fixed6",
            ],
            "view_type": [
                "T16_contiguous",
                "T6_contiguous",
                "T6_contiguous",
            ],
            "sampling_pattern": ["contiguous"] * 3,
            "selected_frame_indices": [
                json.dumps(list(range(16)), separators=(",", ":")),
                json.dumps(list(range(100, 106)), separators=(",", ":")),
                json.dumps(list(range(6)), separators=(",", ":")),
            ],
            "pair_recomputed_for_view": [True] * 3,
            "aggregate_recomputed_for_view": [True] * 3,
            "temporal_unit_keys_json": [
                '["legacy-unit"]',
                '["cvat-unit"]',
                '["legacy-unit"]',
            ],
            "behavior_window_label": ["stand", "stand", "stand"],
            "window_valid_for_main_train": [True, True, True],
        }
    )
    records: list[dict[str, object]] = []
    for source, object_key, unit_key, start, length, prefix in [
        ("legacy_recovered", "legacy-track", "legacy-unit", 0, 16, "legacy"),
        ("cvat_tracking_xml", "cvat-track", "cvat-unit", 100, 6, "cvat"),
    ]:
        for slot in range(length):
            records.append(
                {
                    "frame_uid": f"{prefix}-frame-{slot}",
                    "source_type": source,
                    "object_track_key": object_key,
                    "temporal_unit_key": unit_key,
                    "frame_index": start + slot,
                    "timestamp_sec": slot / 30.0,
                    "bbox_valid": True,
                    "spatiotemporal_feature_valid": True,
                    "roi_feeder_available": True,
                    "roi_drinker_available": True,
                    "roi_toy_available": True,
                    "nearest_pig_id": "ID_2",
                }
            )
    return windows, pd.DataFrame.from_records(records), intervals
