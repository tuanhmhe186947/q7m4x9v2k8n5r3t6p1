from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.data import classification_dataset as dataset
from pig_behavior.data.classification_features import (
    apply_first_task_frame_behavior_authority,
    clean_merged_annotations,
)

BEHAVIORS = ["stand", "fight"]


def _authority_fixture() -> pd.DataFrame:
    task_frames = [10, 11, 12, 13, 4, 14]
    rows = []
    for order, task_frame in enumerate(task_frames):
        rows.append(
            {
                "task": "task_0",
                "frame": task_frame,
                "img_name": f"burst_color_deadbeef_400_f{order * 3}_k{order}.jpg",
                "image_path": f"/images/frame_{task_frame}.jpg",
                "width": 100,
                "height": 80,
                "x1": -2.0 if order == 0 else 10.0,
                "y1": 10.0,
                "x2": 105.0 if order == 0 else 30.0,
                "y2": 40.0,
                "pig_id": "ID_1",
                "behavior": "fight" if order == 4 else "stand",
                "hidden": "Yes" if order == 2 else "No",
                "hidden_attribute_present": True,
                "group_id": "burst_color_deadbeef_400",
                "order": order,
                "is_burst_first_task_frame": order == 4,
                "annotation_format": "xml",
            }
        )
    return pd.DataFrame(rows)


def test_first_task_frame_overrides_majority_without_changing_hidden() -> None:
    source = _authority_fixture()

    output = apply_first_task_frame_behavior_authority(source, BEHAVIORS)

    assert len(output) == len(source)
    assert output["behavior"].eq("fight").all()
    assert output["behavior_before_authority"].tolist() == source[
        "behavior"
    ].tolist()
    assert output["behavior_authority_slot"].eq(4).all()
    assert output["behavior_authority_task_frame"].eq(4).all()
    assert output["hidden"].tolist() == source["hidden"].tolist()


def test_cleaning_preserves_out_of_image_bbox_and_flags_it() -> None:
    source = _authority_fixture()

    output = clean_merged_annotations(
        source,
        BEHAVIORS,
        drop_hidden=False,
    )

    assert len(output) == len(source)
    assert output.loc[0, ["x1", "x2"]].tolist() == [-2.0, 105.0]
    assert bool(output.loc[0, "bbox_outside_image"])
    assert output["hidden"].tolist() == source["hidden"].tolist()


def test_duplicate_actor_slot_fails_closed() -> None:
    source = pd.concat(
        [_authority_fixture(), _authority_fixture().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate_actor_slot"):
        apply_first_task_frame_behavior_authority(source, BEHAVIORS)


def test_incomplete_actor_anchor_set_fails_closed() -> None:
    source = _authority_fixture().iloc[:-1].copy()

    with pytest.raises(ValueError, match="incomplete_actor_anchor_set"):
        apply_first_task_frame_behavior_authority(source, BEHAVIORS)


def _write_source_files(root: Path) -> tuple[Path, Path]:
    cvat_root = root / "cvat"
    for index in range(4):
        task_dir = cvat_root / f"task_{index}"
        data_dir = task_dir / "data"
        data_dir.mkdir(parents=True)
        (task_dir / "task.json").write_text("{}", encoding="utf-8")
        (data_dir / "manifest.jsonl").write_text("{}\n", encoding="utf-8")
        annotation_name = "annotations.xml" if index < 3 else "annotations.json"
        annotation_content = "<annotations/>" if index < 3 else "[]"
        (task_dir / annotation_name).write_text(
            annotation_content,
            encoding="utf-8",
        )
        if index == 0:
            (task_dir / "annotations.json").write_text("[]", encoding="utf-8")
    roi_path = root / "roi.json"
    roi_path.write_text(json.dumps({"images": [], "annotations": []}), encoding="utf-8")
    return cvat_root, roi_path


def test_source_records_select_xml_before_json(tmp_path: Path) -> None:
    cvat_root, roi_path = _write_source_files(tmp_path)

    records = dataset.collect_source_records(cvat_root, roi_path)
    task_zero = [record for record in records if record["scope"] == "task_0"]

    assert any(record["role"] == "annotation_xml" for record in task_zero)
    assert not any(record["role"] == "annotation_json" for record in task_zero)
    assert all(len(record["sha256"]) == 64 for record in records)


def test_output_under_project_data_is_rejected(tmp_path: Path) -> None:
    cvat_root, roi_path = _write_source_files(tmp_path)
    args = dataset.parse_args(
        [
            "--cvat-export-root",
            str(cvat_root),
            "--roi-coco-json",
            str(roi_path),
            "--output-dir",
            str(dataset.PROJECT_ROOT / "data" / "processed" / "forbidden"),
            "--dry-run",
        ]
    )

    with pytest.raises(ValueError, match="immutable project data"):
        dataset.validate_build_paths(args)


def test_dry_run_writes_no_dataset_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cvat_root, roi_path = _write_source_files(tmp_path)
    source = _authority_fixture()
    image_root = tmp_path / "images"
    image_root.mkdir()
    for index, _row in source.iterrows():
        image_path = image_root / f"{index}.jpg"
        image_path.touch()
        source.loc[index, "image_path"] = str(image_path)

    monkeypatch.setattr(dataset, "load_all_cvat_tasks", lambda _: source.copy())
    monkeypatch.setattr(dataset, "load_behaviors_from_project", lambda _: BEHAVIORS)
    monkeypatch.setattr(
        dataset,
        "add_training_features",
        lambda frame, _: frame.assign(
            behavior_coarse="posture",
            in_feeder=False,
            in_drinker=False,
            in_toy=False,
        ),
    )
    output_dir = tmp_path / "derived"
    output_dir.mkdir()

    dataset.main(
        [
            "--cvat-export-root",
            str(cvat_root),
            "--roi-coco-json",
            str(roi_path),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    assert list(output_dir.iterdir()) == []


def test_actor_exclusion_policy_preserves_explicit_row_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cvat_root, roi_path = _write_source_files(tmp_path)
    source = _authority_fixture()
    image_root = tmp_path / "images"
    image_root.mkdir()
    for index, _row in source.iterrows():
        image_path = image_root / f"{index}.jpg"
        image_path.touch()
        source.loc[index, "image_path"] = str(image_path)
    extra = source.copy()
    extra["group_id"] = "excluded_group"
    extra["pig_id"] = "ID_2"
    source = pd.concat([source, extra], ignore_index=True)

    monkeypatch.setattr(dataset, "load_all_cvat_tasks", lambda _: source.copy())
    monkeypatch.setattr(dataset, "load_behaviors_from_project", lambda _: BEHAVIORS)
    monkeypatch.setattr(
        dataset,
        "add_training_features",
        lambda frame, _: frame.assign(
            behavior_coarse="posture",
            in_feeder=False,
            in_drinker=False,
            in_toy=False,
        ),
    )
    policy = tmp_path / "excluded_actor_keys.csv"
    policy.write_text(
        "group_id,pig_id,reason\n"
        "excluded_group,ID_2,operator-confirmed incomplete task_3 actor\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "derived"

    dataset.main(
        [
            "--cvat-export-root",
            str(cvat_root),
            "--roi-coco-json",
            str(roi_path),
            "--output-dir",
            str(output_dir),
            "--exclude-actor-key-csv",
            str(policy),
        ]
    )

    lineage = json.loads(
        (output_dir / dataset.OUT_LINEAGE_NAME).read_text(encoding="utf-8")
    )
    assert lineage["row_counts"] == {
        "raw": 12,
        "excluded": 6,
        "retained": 6,
        "validated": 6,
        "feature": 6,
    }
    assert lineage["actor_exclusion_policy"]["key_count"] == 1
