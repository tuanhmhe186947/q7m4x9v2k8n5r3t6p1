from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.datasets.legacy_unreviewed_development import (
    LEGACY_DEVELOPMENT_SCOPE,
    LEGACY_TEMPORAL_MODEL_VIEW_SPECS,
    build_legacy_unreviewed_development_manifests,
)
from pig_behavior.classification_v2.training.temporal_view_loader import (
    load_temporal_view_tensors,
)

_LOADER_CHECKER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "classification_v2"
    / "04_baselines_smokes"
    / "check_classification_v2_temporal_view_loader.py"
)
_LOADER_CHECKER_SPEC = importlib.util.spec_from_file_location(
    "classification_v2_temporal_view_loader_checker",
    _LOADER_CHECKER_PATH,
)
assert _LOADER_CHECKER_SPEC is not None
assert _LOADER_CHECKER_SPEC.loader is not None
_LOADER_CHECKER = importlib.util.module_from_spec(_LOADER_CHECKER_SPEC)
_LOADER_CHECKER_SPEC.loader.exec_module(_LOADER_CHECKER)


def _fixture_tables() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    source_rows: list[dict[str, object]] = []
    harmonized_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    starts_by_length = {
        6: [0, 3, 6, 9],
        8: [0, 3, 6],
        12: [0, 3],
        16: [0],
    }
    for unit_index, behavior in enumerate(("eat", "fight")):
        dataset_id = "legacy_recovered_16f"
        video_key = f"pigs0{unit_index + 1}1119/000100/color.mp4"
        track_id = f"tracklet_{unit_index:08d}"
        pig_id = f"ID_{unit_index + 1}"
        object_key = (
            f"legacy_recovered|{dataset_id}|{video_key}|"
            f"track={track_id}|pig={pig_id}"
        )
        unit_key = f"{object_key}|legacy_sequence"
        for frame_index in range(16):
            source_rows.append(
                {
                    "source_type": "legacy_recovered",
                    "dataset_id": dataset_id,
                    "video_key": video_key,
                    "clip_id": f"burst-{unit_index}",
                    "track_id": track_id,
                    "pig_id": pig_id,
                    "relative_frame_index": frame_index,
                    "behavior": behavior,
                    "bbox_valid": True,
                    "include_in_training": True,
                    "use_for_main_eval": True,
                    "hidden": "No",
                }
            )
            harmonized_rows.append(
                {
                    "source_type": "legacy_recovered",
                    "dataset_id": dataset_id,
                    "video_key": video_key,
                    "object_track_key": object_key,
                    "track_id": track_id,
                    "pig_id": pig_id,
                    "frame_index": frame_index,
                    "timestamp_sec": frame_index * 0.2,
                    "temporal_unit_key": unit_key,
                    "behavior_temporal_final": behavior,
                    "bbox_valid": True,
                    "spatiotemporal_feature_valid": True,
                    "include_in_training": True,
                    "use_for_main_eval": True,
                }
            )
        interval_rows.append(
            {
                "temporal_unit_key": unit_key,
                "source_type": "legacy_recovered",
                "dataset_id": dataset_id,
                "video_key": video_key,
                "object_track_key": object_key,
                "pig_id": pig_id,
                "track_id": track_id,
                "label_window_start": 0,
                "label_window_end": 15,
                "label_frame_count": 16,
                "observed_frame_count": 16,
                "expected_observed_frame_count": 16,
                "temporal_interval_complete": True,
                "behavior_temporal_final": behavior,
                "behavior_consistency_in_interval": True,
                "bbox_valid_ratio_interval": 1.0,
                "hidden_ratio_interval": 0.0,
                "spatiotemporal_feature_valid_ratio_interval": 1.0,
            }
        )
        for length, starts in starts_by_length.items():
            for start in starts:
                end = start + length - 1
                window_rows.append(
                    {
                        "window_id": (
                            f"{object_key}|win={length}|{start}-{end}"
                        ),
                        "source_type": "legacy_recovered",
                        "dataset_id": dataset_id,
                        "video_key": video_key,
                        "object_track_key": object_key,
                        "window_length_frames": length,
                        "window_start_frame": start,
                        "window_end_frame": end,
                        "temporal_unit_keys_json": json.dumps([unit_key]),
                        "num_temporal_units_window": 1,
                        "behavior_window_label": behavior,
                        "window_valid_for_main_train": True,
                    }
                )
    return (
        pd.DataFrame(source_rows),
        pd.DataFrame(harmonized_rows),
        pd.DataFrame(interval_rows),
        pd.DataFrame(window_rows),
    )


def test_builds_balanced_and_matched_temporal_tiers(tmp_path: Path) -> None:
    tables = build_legacy_unreviewed_development_manifests(*_fixture_tables())

    assert tables.audit["errors"] == []
    assert tables.audit["valid_for_bounded_development"] is True
    assert tables.audit["human_review_complete"] is False
    comparison = tables.audit["temporal_input_comparison_contract"]
    assert comparison["controlled_tiers_frames"] == [6, 8, 12, 16]
    assert comparison["changed_scientific_family"] == (
        "temporal_input_length_only"
    )
    assert len(tables.source_units) == 2
    assert len(tables.native_units) == 2
    assert len(tables.all_sliding_windows) == 20
    assert len(tables.matched_windows) == 8
    assert set(tables.native_units["lineage_scope"]) == {
        LEGACY_DEVELOPMENT_SCOPE
    }

    mass = tables.all_sliding_windows.groupby(
        ["temporal_unit_key", "temporal_tier"]
    )["tier_event_mass_weight"].sum()
    assert mass.eq(1.0).all()
    matched_starts = (
        tables.matched_windows.groupby("temporal_tier")["window_start_frame"]
        .unique()
        .map(lambda values: values.tolist())
        .to_dict()
    )
    assert matched_starts == {
        "T6": [6],
        "T8": [3],
        "T12": [3],
        "T16": [0],
    }
    selection = tables.temporal_selection
    assert len(selection) == len(tables.all_sliding_windows)
    assert set(tables.temporal_slot_manifests) == set(
        LEGACY_TEMPORAL_MODEL_VIEW_SPECS
    )
    expected_selected = {
        "legacy_t6_all_sliding_keep": 8,
        "legacy_t6_centered_matched_keep": 2,
        "legacy_t8_all_sliding_keep": 6,
        "legacy_t8_centered_matched_keep": 2,
        "legacy_t12_all_sliding_keep": 4,
        "legacy_t12_centered_matched_keep": 2,
        "legacy_t16_all_sliding_keep": 2,
        "legacy_t16_centered_matched_keep": 2,
    }
    assert {
        column: int(selection[column].sum())
        for column in expected_selected
    } == expected_selected
    for view_name, spec in LEGACY_TEMPORAL_MODEL_VIEW_SPECS.items():
        slots = tables.temporal_slot_manifests[view_name]
        selected_mask = selection[str(spec["selection_column"])].to_numpy()
        selected_count = int(selected_mask.sum())
        assert len(slots) == selected_count * int(spec["sequence_length"])
        assert set(slots["temporal_view_name"]) == {view_name}
        first = slots.loc[slots["slot_index"].eq(0), "time_delta"]
        later = slots.loc[slots["slot_index"].gt(0), "time_delta"]
        assert first.eq(0.0).all()
        assert np.isclose(later, 0.2).all()
        slot_path = tmp_path / str(spec["slot_manifest_filename"])
        slots.to_csv(slot_path, index=False)
        tensors = load_temporal_view_tensors(
            slot_path,
            expected_window_ids=selection["window_id"],
            selected_mask=selected_mask,
            expected_view_name=view_name,
            expected_sequence_length=int(spec["sequence_length"]),
        )
        assert tensors.time_delta.shape == (
            len(selection),
            int(spec["sequence_length"]),
        )
        assert np.isfinite(tensors.time_delta[selected_mask]).all()
        assert np.isnan(tensors.time_delta[~selected_mask]).all()


def test_real_tier_loader_audit_accepts_complete_packet(
    tmp_path: Path,
) -> None:
    selection = _write_temporal_tier_packet(tmp_path)

    audit = _LOADER_CHECKER.run_legacy_tier_loader_audit(tmp_path)

    assert audit["valid"] is True
    assert audit["errors"] == []
    assert audit["window_universe_rows"] == len(selection)
    assert audit["loaded_view_count"] == 8
    for view in audit["views"].values():
        assert view["shape_valid"] is True
        assert view["selected_nonempty"] is True
        assert view["selected_time_delta_finite"] is True
        assert view["unselected_time_delta_nan"] is True
        assert view["selected_timing_valid"] is True
        assert view["selected_observed"] is True
        assert view["unselected_masks_clear"] is True


def test_real_tier_loader_audit_rejects_dirty_selection_boolean(
    tmp_path: Path,
) -> None:
    selection = _write_temporal_tier_packet(tmp_path)
    column = "legacy_t6_all_sliding_keep"
    selection[column] = selection[column].astype(object)
    selection.loc[0, column] = "not-a-boolean"
    selection.to_csv(
        tmp_path / "temporal_tier_selection_manifest.csv",
        index=False,
    )

    audit = _LOADER_CHECKER.run_legacy_tier_loader_audit(tmp_path)

    assert audit["valid"] is False
    assert audit["loaded_view_count"] == 7
    assert any(
        error.startswith("legacy_t6_all_sliding_observed_time:ValueError")
        for error in audit["errors"]
    )


def test_accepts_absolute_frame_indices_inside_each_burst() -> None:
    source, harmonized, intervals, windows = _fixture_tables()
    frame_offset = 100
    harmonized["frame_index"] += frame_offset
    intervals["label_window_start"] += frame_offset
    intervals["label_window_end"] += frame_offset
    windows["window_start_frame"] += frame_offset
    windows["window_end_frame"] += frame_offset

    tables = build_legacy_unreviewed_development_manifests(
        source,
        harmonized,
        intervals,
        windows,
    )

    assert tables.audit["errors"] == []
    assert tables.audit["harmonized_frame_audit"][
        "invalid_frame_index_rows"
    ] == 0
    assert tables.native_units["harmonized_interval_bounds_match"].all()
    assert tables.native_units["native_unit_valid_for_development"].all()


def test_rejects_harmonized_interval_boundary_drift() -> None:
    source, harmonized, intervals, windows = _fixture_tables()
    intervals.loc[0, "label_window_start"] += 1
    intervals.loc[0, "label_window_end"] += 1

    with pytest.raises(
        ValueError,
        match="harmonized_interval_bound_mismatch_units=1",
    ):
        build_legacy_unreviewed_development_manifests(
            source,
            harmonized,
            intervals,
            windows,
        )


def test_invalid_unit_is_retained_with_zero_training_mass() -> None:
    source, harmonized, intervals, windows = _fixture_tables()
    first_track = source.loc[0, "track_id"]
    source.loc[source["track_id"].eq(first_track), "include_in_training"] = False

    tables = build_legacy_unreviewed_development_manifests(
        source,
        harmonized,
        intervals,
        windows,
    )

    invalid = tables.native_units.loc[
        ~tables.native_units["native_unit_valid_for_development"]
    ]
    assert len(invalid) == 1
    assert "source_excluded" in invalid.iloc[0]["native_unit_exclusion_reason"]
    invalid_key = invalid.iloc[0]["temporal_unit_key"]
    invalid_windows = tables.all_sliding_windows.loc[
        tables.all_sliding_windows["temporal_unit_key"].eq(invalid_key)
    ]
    assert len(invalid_windows) == 10
    assert invalid_windows["tier_event_mass_weight"].eq(0.0).all()


def test_rejects_window_that_crosses_two_native_units() -> None:
    source, harmonized, intervals, windows = _fixture_tables()
    keys = intervals["temporal_unit_key"].tolist()
    windows.loc[0, "temporal_unit_keys_json"] = json.dumps(keys)
    windows.loc[0, "num_temporal_units_window"] = 2

    with pytest.raises(ValueError, match="exactly one native unit"):
        build_legacy_unreviewed_development_manifests(
            source,
            harmonized,
            intervals,
            windows,
        )


def test_rejects_missing_temporal_tier_window_without_dropping_unit() -> None:
    source, harmonized, intervals, windows = _fixture_tables()
    first_key = intervals.loc[0, "temporal_unit_key"]
    unit_json = json.dumps([first_key])
    remove = windows["temporal_unit_keys_json"].eq(unit_json) & windows[
        "window_length_frames"
    ].eq(16)
    windows = windows.loc[~remove].reset_index(drop=True)

    with pytest.raises(ValueError, match="missing_native_tier_pairs=1"):
        build_legacy_unreviewed_development_manifests(
            source,
            harmonized,
            intervals,
            windows,
        )


def test_rejects_wrong_stride_lattice_even_when_window_count_matches() -> None:
    source, harmonized, intervals, windows = _fixture_tables()
    wrong = windows["window_length_frames"].eq(6) & windows[
        "window_start_frame"
    ].eq(9)
    first_index = windows.index[wrong][0]
    windows.loc[first_index, "window_start_frame"] = 8
    windows.loc[first_index, "window_end_frame"] = 13

    with pytest.raises(ValueError, match="tier_window_lattice_mismatches=1"):
        build_legacy_unreviewed_development_manifests(
            source,
            harmonized,
            intervals,
            windows,
        )


def _write_temporal_tier_packet(tmp_path: Path) -> pd.DataFrame:
    tables = build_legacy_unreviewed_development_manifests(*_fixture_tables())
    selection = tables.temporal_selection.copy()
    selection.to_csv(
        tmp_path / "temporal_tier_selection_manifest.csv",
        index=False,
    )
    for view_name, spec in LEGACY_TEMPORAL_MODEL_VIEW_SPECS.items():
        manifest = tables.temporal_slot_manifests[view_name]
        manifest.to_csv(
            tmp_path / str(spec["slot_manifest_filename"]),
            index=False,
        )
    return selection
