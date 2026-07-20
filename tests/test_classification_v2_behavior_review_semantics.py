import importlib.util
import sys
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.features.sequence_windows import (
    audit_sequence_windows,
    build_sequence_windows,
)
from pig_behavior.classification_v2.features.temporal_harmonization import (
    build_temporal_label_intervals,
    harmonize_temporal_labels,
)
from pig_behavior.classification_v2.review.review_unit_builder import (
    ReviewUnitConfig,
    build_review_units,
)
from pig_behavior.classification_v2.train_ready_features import (
    select_window_feature_columns,
)

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "classification_v2"
    / "01_review_units_gui"
    / "classification_v2_apply_review_unit_decisions.py"
)
SPEC = importlib.util.spec_from_file_location("behavior_review_apply_semantics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
apply_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = apply_module
SPEC.loader.exec_module(apply_module)
apply_decisions_to_frames = apply_module.apply_decisions_to_frames
normalize_decisions = apply_module.normalize_decisions

CHECKER_SCRIPT = SCRIPT.parent / "check_behavior_reviewed_sequence_windows.py"
CHECKER_SPEC = importlib.util.spec_from_file_location(
    "behavior_reviewed_sequence_checker", CHECKER_SCRIPT
)
assert CHECKER_SPEC is not None and CHECKER_SPEC.loader is not None
checker_module = importlib.util.module_from_spec(CHECKER_SPEC)
sys.modules[CHECKER_SPEC.name] = checker_module
CHECKER_SPEC.loader.exec_module(checker_module)

GUI_CHECKER_SCRIPT = SCRIPT.parent / "check_review_unit_gui_contract.py"
GUI_CHECKER_SPEC = importlib.util.spec_from_file_location(
    "review_unit_gui_contract_checker", GUI_CHECKER_SCRIPT
)
assert GUI_CHECKER_SPEC is not None and GUI_CHECKER_SPEC.loader is not None
gui_checker_module = importlib.util.module_from_spec(GUI_CHECKER_SPEC)
sys.modules[GUI_CHECKER_SPEC.name] = gui_checker_module
GUI_CHECKER_SPEC.loader.exec_module(gui_checker_module)


def _frames(reviewed: bool) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_type": ["cvat_tracking_xml"] * 6,
            "dataset_id": ["d"] * 6,
            "video_key": ["v"] * 6,
            "frame_index": list(range(6)),
            "pig_id": ["1"] * 6,
            "track_id": ["1"] * 6,
            "behavior": ["stand"] * 6,
            "behavior_review_decision_present": [reviewed] * 6,
            "behavior_review_label_resolved": [reviewed] * 6,
            "behavior_review_include_in_training": [reviewed] * 6,
            "behavior_reviewed_final": ["stand"] * 6,
            "bbox_valid": [True] * 6,
        }
    )


def _two_cvat_units(
    original: tuple[str, str] = ("stand", "stand"),
    final: tuple[str, str] = ("stand", "stand"),
    decisions: tuple[bool, bool] = (True, True),
    eligible: tuple[bool, bool] = (True, True),
) -> pd.DataFrame:
    unit = [0] * 6 + [1] * 6
    return pd.DataFrame(
        {
            "source_type": ["cvat_tracking_xml"] * 12,
            "dataset_id": ["d"] * 12,
            "video_key": ["v"] * 12,
            "frame_index": list(range(12)),
            "pig_id": ["1"] * 12,
            "track_id": ["1"] * 12,
            "behavior": [original[index] for index in unit],
            "behavior_review_decision_present": [decisions[index] for index in unit],
            "behavior_review_label_resolved": [decisions[index] for index in unit],
            "behavior_review_include_in_training": [eligible[index] for index in unit],
            "behavior_reviewed_final": [final[index] for index in unit],
            "bbox_valid": [True] * 12,
        }
    )


def test_annotation_stable_unreviewed_is_not_main_train() -> None:
    _, intervals, windows = build_sequence_windows(_frames(False), window_lengths=[6])
    row = windows.iloc[0]
    assert row["sequence_label_status"] == "stable"
    assert row["human_reviewed_behavior_consistency_status"] == "unreviewed"
    assert bool(row["window_valid_for_main_train"]) is False
    assert float(row["window_sample_weight"]) == 0.0
    assert audit_sequence_windows(windows, intervals)["errors"] == []


def test_all_native_units_reviewed_can_be_main_train() -> None:
    _, intervals, windows = build_sequence_windows(_frames(True), window_lengths=[6])
    row = windows.iloc[0]
    assert row["human_reviewed_behavior_consistency_status"] == "stable"
    assert float(row["behavior_review_coverage_ratio_window"]) == 1.0
    assert bool(row["all_temporal_units_behavior_train_eligible"]) is True
    assert bool(row["window_valid_for_main_train"]) is True
    assert audit_sequence_windows(windows, intervals)["errors"] == []


def test_temporal_harmonization_feeds_native_only_builder_without_windows(
    tmp_path,
) -> None:
    frames = pd.concat(
        [
            _frames(False),
            pd.DataFrame(
                {
                    "source_type": ["legacy_recovered"] * 16,
                    "dataset_id": ["d"] * 16,
                    "video_key": ["legacy-v"] * 16,
                    "frame_index": list(range(100, 116)),
                    "pig_id": ["2"] * 16,
                    "track_id": ["2"] * 16,
                    "behavior": ["lying"] * 16,
                    "bbox_valid": [True] * 16,
                }
            ),
        ],
        ignore_index=True,
    )
    harmonized = harmonize_temporal_labels(frames)
    intervals = build_temporal_label_intervals(harmonized)
    intervals_csv = tmp_path / "temporal_intervals_standalone.csv"
    intervals.to_csv(intervals_csv, index=False)
    audit = build_review_units(
        ReviewUnitConfig(
            intervals_csv=intervals_csv,
            output_dir=tmp_path / "review_units",
            include_all_retained_native_units=True,
        )
    )
    assert audit["rows"]["windows"] is None
    assert audit["rows"]["review_units"] == 2
    manifest = pd.read_csv(audit["templates"]["full_review_unit_manifest"]["path"])
    assert set(manifest["review_unit_type"]) == {"cvat_interval_6", "legacy_burst_16"}
    assert manifest["review_unit_id"].is_unique
    assert not manifest["behavior_review_cohort"].eq("behavior_not_selected").any()
    template_ids: set[str] = set()
    for name, details in audit["templates"].items():
        if name.endswith("_review_unit_template"):
            template = pd.read_csv(details["path"])
            template_ids.update(template.get("review_unit_id", pd.Series(dtype=str)))
    assert template_ids == set(manifest["review_unit_id"])
    gui_audit = gui_checker_module.validate_gui_contract(manifest, harmonized)
    assert gui_audit["errors"] == []


def test_partial_review_coverage_blocks_cvat_eight_frame_window() -> None:
    _, _, windows = build_sequence_windows(
        _two_cvat_units(decisions=(True, False), eligible=(True, False)),
        window_lengths=[8],
    )
    row = windows.iloc[0]
    assert row["human_reviewed_behavior_consistency_status"] == "partial"
    assert float(row["behavior_review_coverage_ratio_window"]) == 0.5
    assert bool(row["window_valid_for_main_train"]) is False


def test_correction_can_change_stable_annotation_to_reviewed_transition() -> None:
    _, _, windows = build_sequence_windows(
        _two_cvat_units(final=("stand", "move")),
        window_lengths=[12],
    )
    row = windows.iloc[0]
    assert row["annotation_consistency_status"] == "stable"
    assert row["human_reviewed_behavior_consistency_status"] == "transition"
    assert bool(row["window_valid_for_main_train"]) is False


def test_correction_can_change_annotation_transition_to_reviewed_stable() -> None:
    _, _, windows = build_sequence_windows(
        _two_cvat_units(original=("stand", "move"), final=("stand", "stand")),
        window_lengths=[12],
    )
    row = windows.iloc[0]
    assert row["annotation_consistency_status"] == "transition"
    assert row["human_reviewed_behavior_consistency_status"] == "stable"
    assert bool(row["window_valid_for_main_train"]) is True


def test_excluded_or_uncertain_unit_prevents_bridge() -> None:
    for decisions, eligible, expected in [
        ((True, True), (True, False), "excluded"),
        ((True, False), (True, False), "partial"),
    ]:
        _, _, windows = build_sequence_windows(
            _two_cvat_units(decisions=decisions, eligible=eligible),
            window_lengths=[8, 12],
        )
        spanning = windows[windows["window_start_frame"].eq(0)]
        assert spanning["human_reviewed_behavior_consistency_status"].eq(expected).all()
        assert (~spanning["window_valid_for_main_train"].astype(bool)).all()
        assert spanning["window_sample_weight"].eq(0.0).all()


def test_partial_review_coverage_main_train_false() -> None:
    # partial review coverage -> main_train false
    df = pd.DataFrame(
        {
            "source_type": ["cvat_tracking_xml"] * 12,
            "dataset_id": ["d"] * 12,
            "video_key": ["v"] * 12,
            "frame_index": list(range(12)),
            "pig_id": ["1"] * 12,
            "track_id": ["1"] * 12,
            "behavior": ["stand"] * 12,
            "behavior_review_decision_present": [True] * 12,
            "behavior_review_label_resolved": [True] * 12,
            "behavior_review_include_in_training": [True] * 12,
            "behavior_reviewed_final": ["stand"] * 12,
            "bbox_valid": [True] * 12,
            "temporal_unit_key": ["unit0"] * 6 + ["unit1"] * 6,
        }
    )
    df.loc[6:11, "behavior_review_decision_present"] = False
    _, intervals, windows = build_sequence_windows(df, window_lengths=[8])
    row = windows.iloc[0]
    assert row["human_reviewed_behavior_consistency_status"] == "partial"
    assert bool(row["window_valid_for_main_train"]) is False
    assert float(row["window_sample_weight"]) == 0.0


def test_stable_to_transition_after_correction() -> None:
    # stable -> transition sau correction
    df = pd.DataFrame(
        {
            "source_type": ["cvat_tracking_xml"] * 12,
            "dataset_id": ["d"] * 12,
            "video_key": ["v"] * 12,
            "frame_index": list(range(12)),
            "pig_id": ["1"] * 12,
            "track_id": ["1"] * 12,
            "behavior": ["stand"] * 12,
            "behavior_review_decision_present": [True] * 12,
            "behavior_review_label_resolved": [True] * 12,
            "behavior_review_include_in_training": [True] * 12,
            "behavior_reviewed_final": ["stand"] * 12,
            "bbox_valid": [True] * 12,
            "temporal_unit_key": ["unit0"] * 6 + ["unit1"] * 6,
        }
    )
    df.loc[6:11, "behavior_reviewed_final"] = "eat"
    _, intervals, windows = build_sequence_windows(df, window_lengths=[8])
    row = windows.iloc[0]
    assert row["human_reviewed_behavior_consistency_status"] == "transition"
    assert bool(row["window_valid_for_main_train"]) is False


def test_transition_to_stable_after_correction() -> None:
    # transition -> stable sau correction
    df = _frames(True)
    df.loc[0:2, "behavior"] = "stand"
    df.loc[3:5, "behavior"] = "eat"
    df["behavior_reviewed_final"] = "stand"
    _, intervals, windows = build_sequence_windows(df, window_lengths=[6])
    row = windows.iloc[0]
    assert row["human_reviewed_behavior_consistency_status"] == "stable"
    assert bool(row["window_valid_for_main_train"]) is True


def test_uncertain_and_excluded_units_cut_reviewed_behavior_run() -> None:
    # uncertain unit cat reviewed behavior run
    # excluded unit cat reviewed behavior run
    # khong bridge qua excluded/uncertain

    # Excluded unit
    df_ex = _frames(True)
    df_ex["behavior_review_include_in_training"] = False
    _, _, windows_ex = build_sequence_windows(df_ex, window_lengths=[6])
    assert windows_ex.iloc[0]["human_reviewed_behavior_consistency_status"] == "excluded"
    assert bool(windows_ex.iloc[0]["window_valid_for_main_train"]) is False

    # Uncertain unit (resolved = False)
    df_un = _frames(True)
    df_un["behavior_review_label_resolved"] = False
    _, _, windows_un = build_sequence_windows(df_un, window_lengths=[6])
    assert windows_un.iloc[0]["human_reviewed_behavior_consistency_status"] == "unresolved"
    assert bool(windows_un.iloc[0]["window_valid_for_main_train"]) is False


def test_cvat_8frame_needs_all_intervals_reviewed_and_same_label() -> None:
    # CVAT 8-frame can tat ca interval lien quan reviewed va cung final label
    df = pd.DataFrame(
        {
            "source_type": ["cvat_tracking_xml"] * 12,
            "dataset_id": ["d"] * 12,
            "video_key": ["v"] * 12,
            "frame_index": list(range(12)),
            "pig_id": ["1"] * 12,
            "track_id": ["1"] * 12,
            "behavior": ["stand"] * 12,
            "behavior_review_decision_present": [True] * 12,
            "behavior_review_label_resolved": [True] * 12,
            "behavior_review_include_in_training": [True] * 12,
            "behavior_reviewed_final": ["stand"] * 12,
            "bbox_valid": [True] * 12,
            "temporal_unit_key": ["unit0"] * 6 + ["unit1"] * 6,
        }
    )
    # If both are stand, an 8-frame window should be stable and valid
    _, _, windows = build_sequence_windows(df, window_lengths=[8])
    assert windows.iloc[0]["human_reviewed_behavior_consistency_status"] == "stable"
    assert bool(windows.iloc[0]["window_valid_for_main_train"]) is True

    # If one interval has different label (e.g. eat)
    df.loc[6:11, "behavior_reviewed_final"] = "eat"
    _, _, windows2 = build_sequence_windows(df, window_lengths=[8])
    assert windows2.iloc[0]["human_reviewed_behavior_consistency_status"] == "transition"
    assert bool(windows2.iloc[0]["window_valid_for_main_train"]) is False

    # If one interval is unreviewed
    df2 = df.copy()
    df2.loc[6:11, "behavior_review_decision_present"] = False
    _, _, windows3 = build_sequence_windows(df2, window_lengths=[8])
    assert windows3.iloc[0]["human_reviewed_behavior_consistency_status"] == "partial"
    assert bool(windows3.iloc[0]["window_valid_for_main_train"]) is False


def test_partial_span_apply_fail() -> None:
    # partial-span apply fail
    frames = pd.DataFrame(
        {
            "temporal_unit_key": ["unit0"] * 5,  # missing 1 frame
            "source_type": ["cvat_tracking_xml"] * 5,
            "frame_index": list(range(5)),
            "behavior": ["stand"] * 5,
        }
    )
    review_units = pd.DataFrame(
        {
            "review_unit_id": ["unit0"],
            "temporal_unit_key": ["unit0"],
            "source_type": ["cvat_tracking_xml"],
            "unit_start_frame": [0],
            "unit_end_frame": [5],
            "review_unit_type": ["cvat_interval_6"],
            "dataset_id": ["d"],
            "video_key": ["v"],
            "pig_id": ["1"],
            "track_id": ["1"],
            "object_track_key": ["track0"],
            "review_template": ["motion"],  # for "stand"
            "behavior_label": ["stand"],
            "apply_scope": ["cvat_interval_6f"],
        }
    )
    decisions = pd.DataFrame(
        {
            "review_unit_id": ["unit0"],
            "manual_review_decision": ["accept"],
            "manual_corrected_behavior": [""],
            "manual_training_action": ["main_train"],
            "manual_sample_weight": [1.0],
            "manual_label_strength": ["strong"],
            "manual_note": ["ok"],
            "temporal_unit_key": ["unit0"],
            "review_template": ["motion"],
            "behavior_label": ["stand"],
            "original_behavior": ["stand"],
            "apply_scope": ["cvat_interval_6f"],
        }
    )
    _, audit = apply_decisions_to_frames(frames, review_units, decisions)
    assert len(audit["unmatched_decisions"]) > 0
    assert "unit0" in audit["unmatched_decisions"][0]


def test_duplicate_conflicting_decisions_fail() -> None:
    # duplicate/conflicting decisions fail
    decisions = pd.DataFrame(
        {
            "review_item_id": ["item1", "item2"],
            "review_unit_id": ["unit0", "unit0"],  # duplicate
            "review_unit_type": ["cvat_interval_6", "cvat_interval_6"],
            "temporal_unit_key": ["unit0", "unit0"],
            "source_type": ["cvat_tracking_xml", "cvat_tracking_xml"],
            "dataset_id": ["d", "d"],
            "video_key": ["v", "v"],
            "pig_id": ["1", "1"],
            "track_id": ["1", "1"],
            "object_track_key": ["track0", "track0"],
            "unit_start_frame": [0, 0],
            "unit_end_frame": [5, 5],
            "display_frame_indices": ["0,1,2,3,4,5", "0,1,2,3,4,5"],
            "review_template": ["roi", "roi"],
            "behavior_label": ["stand", "stand"],
            "original_behavior": ["stand", "stand"],
            "review_reason": ["risk", "risk"],
            "apply_scope": ["cvat_interval_6", "cvat_interval_6"],
            "manual_review_decision": ["accept", "corrected"],  # conflicting
            "manual_corrected_behavior": ["", "eat"],
            "manual_label_strength": ["strong", "strong"],
            "manual_training_action": ["main_train", "main_train"],
            "manual_sample_weight": [1.0, 1.0],
            "manual_note": ["ok", "ok"],
        }
    )
    _, errors, _ = normalize_decisions(decisions)
    assert "duplicate_decision_rows=2" in errors


def test_independent_reviewed_window_checker_requires_full_rebuild(tmp_path) -> None:
    _, _, windows = build_sequence_windows(_frames(True), window_lengths=[6])
    reviewed_frames = tmp_path / "reviewed_frame_features.csv"
    reviewed_frames.touch()
    review_units = pd.DataFrame({"review_unit_id": ["unit0"]})
    audit = checker_module.audit_reviewed_windows(
        windows,
        review_units,
        {"input_csv": str(reviewed_frames), "parameters": {"build_strategy": "full_rebuild"}},
        reviewed_frames,
    )
    assert audit["errors"] == []
    reused = checker_module.audit_reviewed_windows(
        windows,
        review_units,
        {
            "input_csv": str(reviewed_frames),
            "parameters": {
                "build_strategy": "reuse_unreviewed_window_structure_with_review_overlay"
            },
        },
        reviewed_frames,
    )
    assert "final_sequence_build_not_full_rebuild" in reused["errors"]


def test_behavior_review_audit_columns_do_not_leak_into_model_features() -> None:
    _, _, windows = build_sequence_windows(_frames(True), window_lengths=[6])
    selected = select_window_feature_columns(windows)
    assert not any(column.startswith("behavior_review") for column in selected)
    assert "all_temporal_units_behavior_reviewed" not in selected
    assert "all_temporal_units_behavior_label_resolved" not in selected
    assert "all_temporal_units_behavior_train_eligible" not in selected


def test_reviewed_window_semantics_are_deterministic() -> None:
    first = build_sequence_windows(_two_cvat_units(), window_lengths=[6, 8, 12])
    second = build_sequence_windows(_two_cvat_units(), window_lengths=[6, 8, 12])
    for left, right in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(left, right, check_like=False)
