import pandas as pd
import pytest

from pig_behavior.classification_v2.metadata.recording_groups import (
    assign_publication_splits,
    build_recording_group_manifest,
    parse_ratios,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["w-a", "w-b", "w-c"],
            "source_type": ["cvat_tracking_xml"] * 3,
            "dataset_id": ["dataset-a", "dataset-b", "dataset-c"],
            "video_key": [
                "Pigs281119_000085_30fps",
                "Pigs291119_000231_30fps",
                "Pigs301119_000327_30fps",
            ],
            "behavior_window_label": ["lying", "stand", "fight"],
            "window_valid_for_main_train": [True, True, True],
        }
    )


def test_recording_date_manifest_and_split_preserve_every_window() -> None:
    rows = _rows()
    groups = build_recording_group_manifest(rows)
    splits = assign_publication_splits(
        rows,
        groups.manifest,
        ratios={"train": 1 / 3, "val": 1 / 3, "test": 1 / 3},
    )

    assert groups.audit["recording_group_count"] == 3
    assert splits.audit["row_count_delta"] == 0
    assert splits.audit["leakage_group_count"] == 0
    assert splits.audit["leakage_video_count"] == 0
    assert splits.audit["errors"] == []
    assert set(splits.split_manifest["window_id"]) == set(rows["window_id"])


def test_manual_date_override_recomputes_group_without_explicit_group() -> None:
    rows = _rows().iloc[[0]].copy()
    manual = rows[["source_type", "dataset_id", "video_key"]].copy()
    manual["canonical_recording_date"] = "2019-12-31"
    manual["recording_group_id"] = None
    manual["biological_subject_scope_known"] = "false"

    groups = build_recording_group_manifest(rows, manual_metadata=manual)

    assert groups.manifest.loc[0, "recording_group_id"] == "date=2019-12-31"
    assert groups.audit["biological_subject_scope_known"] is False


def test_recording_manifest_rejects_unmatched_manual_key() -> None:
    rows = _rows().iloc[[0]].copy()
    manual = rows[["source_type", "dataset_id", "video_key"]].copy()
    manual["video_key"] = "Pigs291119_999999_30fps"

    with pytest.raises(ValueError, match="unmatched source/video keys"):
        build_recording_group_manifest(rows, manual_metadata=manual)


def test_recording_date_group_rejects_unknown_or_invalid_date() -> None:
    rows = _rows().iloc[[0]].copy()
    rows["video_key"] = "video_without_date"
    rows["dataset_id"] = "dataset_without_date"

    with pytest.raises(ValueError, match="requires known recording dates"):
        build_recording_group_manifest(rows)

    rows["video_key"] = "Pigs321319_000001_30fps"
    with pytest.raises(ValueError, match="Invalid recording date token"):
        build_recording_group_manifest(rows)


def test_publication_split_rejects_duplicate_id_and_invalid_bool() -> None:
    rows = _rows()
    groups = build_recording_group_manifest(rows).manifest
    duplicate = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate split window_id rows=2"):
        assign_publication_splits(
            duplicate,
            groups,
            ratios={"train": 1 / 3, "val": 1 / 3, "test": 1 / 3},
        )

    invalid = rows.copy()
    invalid["window_valid_for_main_train"] = ["true", "unknown", "false"]
    with pytest.raises(ValueError, match="invalid window_valid_for_main_train"):
        assign_publication_splits(
            invalid,
            groups,
            ratios={"train": 1 / 3, "val": 1 / 3, "test": 1 / 3},
        )


def test_publication_split_rejects_duplicate_group_manifest_key() -> None:
    rows = _rows()
    groups = build_recording_group_manifest(rows).manifest
    groups = pd.concat([groups, groups.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate recording-group key rows=2"):
        assign_publication_splits(
            rows,
            groups,
            ratios={"train": 1 / 3, "val": 1 / 3, "test": 1 / 3},
        )


def test_ratio_parser_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        parse_ratios("nan,0.5,0.5")
