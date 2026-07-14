from __future__ import annotations

import json

import pandas as pd
import pytest

from pig_behavior.classification_v2.datasets.legacy_unreviewed_development import (
    LEGACY_DEVELOPMENT_SCOPE,
    build_legacy_unreviewed_development_manifests,
)


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


def test_builds_balanced_and_matched_temporal_tiers() -> None:
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
