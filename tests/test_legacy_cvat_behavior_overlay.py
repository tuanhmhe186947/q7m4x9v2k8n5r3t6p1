from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from legacy_burst_recovery.anchor_builder import build_anchor_records
from legacy_burst_recovery.csv_loader import validate_legacy_dataframe
from legacy_burst_recovery.cvat_anchor_rebuild import (
    build_legacy_recovery_inputs,
)
from legacy_burst_recovery.cvat_behavior_overlay import (
    apply_cvat_k0_behavior_authority,
)
from legacy_burst_recovery.cvat_recovery_validation import (
    validate_cvat_recovered_dense,
)
from legacy_burst_recovery.export_legacy_annotations import (
    build_export_audit,
    build_frame_object_csv,
    normalize_source_video_key,
)
from legacy_burst_recovery.legacy_gt_loader import hidden_seed_for_frame
from pig_behavior.classification_v2.sources.legacy_recovered_csv import (
    load_legacy_frame_objects,
)

OFFSETS = [0, 3, 6, 9, 12, 15]


def _write_cvat_task(
    root: Path,
    labels: dict[str, dict[str, str]],
    *,
    later_behavior: str = "stand",
    hidden_by_slot: dict[int, str] | None = None,
) -> Path:
    task_dir = root / "task_0"
    data_dir = task_dir / "data"
    data_dir.mkdir(parents=True)

    image_records: list[dict[str, object]] = []
    for group_id in labels:
        for slot, source_frame in enumerate(OFFSETS):
            image_records.append(
                {
                    "name": f"{group_id}_f{source_frame}_k{slot}",
                    "extension": ".jpg",
                    "width": 1280,
                    "height": 720,
                    "group_id": group_id,
                    "slot": slot,
                }
            )
    image_records.sort(key=lambda item: str(item["name"]))

    manifest = [{"version": "1.1"}, {"type": "images"}]
    manifest.extend(
        {
            "name": item["name"],
            "extension": item["extension"],
            "width": item["width"],
            "height": item["height"],
        }
        for item in image_records
    )
    (data_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(item) for item in manifest) + "\n",
        encoding="utf-8",
    )

    shapes: list[dict[str, object]] = []
    for task_frame, image in enumerate(image_records):
        group_id = str(image["group_id"])
        slot = int(image["slot"])
        for pig_index, (pig_id, k0_behavior) in enumerate(
            labels[group_id].items(),
            start=1,
        ):
            behavior = k0_behavior if slot == 0 else later_behavior
            shapes.append(
                {
                    "type": "rectangle",
                    "label": "Pig",
                    "frame": task_frame,
                    "outside": False,
                    "points": [
                        10.0 * pig_index + slot,
                        20.0,
                        50.0 + slot,
                        80.0,
                    ],
                    "attributes": [
                        {"name": "ID", "value": pig_id},
                        {"name": "Behavior", "value": behavior},
                        {
                            "name": "Hidden",
                            "value": (hidden_by_slot or {}).get(slot, "No"),
                        },
                    ],
                }
            )
    (task_dir / "annotations.json").write_text(
        json.dumps([{"version": 0, "tags": [], "shapes": shapes, "tracks": []}]),
        encoding="utf-8",
    )
    (task_dir / "task.json").write_text(
        json.dumps({"name": "synthetic", "subset": "Train"}),
        encoding="utf-8",
    )
    return task_dir


def _dense_rows(labels: dict[str, dict[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_index, (group_id, pigs) in enumerate(labels.items()):
        for pig_index, pig_id in enumerate(pigs, start=1):
            for frame_index in range(16):
                rows.append(
                    {
                        "group_id": group_id,
                        "sample_id": f"{group_id}_{pig_id}",
                        "tracklet_id": f"track_{group_index}_{pig_index}",
                        "pig_id": pig_id,
                        "frame_index": frame_index,
                        "behavior": "stand",
                        "hidden": "Yes" if frame_index == 4 else "No",
                        "x1": float(10 * pig_index),
                        "y1": 20.0,
                        "x2": 50.0,
                        "y2": 80.0,
                        "include_in_training": True,
                        "training_tier": "clean",
                        "qa_status": "ok",
                    }
                )
    return pd.DataFrame(rows)


def _write_scaffold(
    path: Path,
    labels: dict[str, dict[str, str]],
) -> Path:
    rows: list[dict[str, object]] = []
    for group_id, pigs in labels.items():
        for pig_id, behavior in pigs.items():
            rows.append(
                {
                    "video_final": f"/source/{group_id}/color.mp4",
                    "day_final": "pigs010101",
                    "group_id": group_id,
                    "sample_id": f"{group_id}_{pig_id}",
                    "img_name": f"{group_id}_f9_k3.jpg",
                    "frames": "0|3|6|9|12|15",
                    "pig_id": pig_id,
                    "x1": 10.0,
                    "y1": 20.0,
                    "x2": 50.0,
                    "y2": 80.0,
                    "behavior": behavior,
                    "hidden": "No",
                    "center_frame_from_img": 9,
                    "center_frame_final": 9,
                    "frame_mismatch": False,
                    "match_source": "synthetic_scaffold",
                    "source_video_key": "pigs010101/000001",
                    "trigger_type": "stale_trigger",
                    "roi_name": "stale_roi",
                    "near_roi": True,
                    "distance_to_roi": 0.0,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_k0_is_burst_authority_not_global_task_frame(tmp_path: Path) -> None:
    labels = {
        "burst_color_aaa11111_100": {"ID_1": "explore", "ID_2": "lying"},
        "burst_color_bbb22222_200": {"ID_1": "drink"},
    }
    _write_cvat_task(tmp_path, labels)
    dense = _dense_rows(labels)
    original_non_behavior = dense[["hidden", "x1", "y1", "x2", "y2"]].copy()

    out, audit, discrepancies = apply_cvat_k0_behavior_authority(dense, tmp_path)

    assert len(out) == len(dense) == 48
    assert not out.duplicated(["group_id", "pig_id", "frame_index"]).any()
    pd.testing.assert_frame_equal(
        out[["hidden", "x1", "y1", "x2", "y2"]].reset_index(drop=True),
        original_non_behavior.reset_index(drop=True),
    )
    for group_id, pigs in labels.items():
        for pig_id, behavior in pigs.items():
            selected = out[out["group_id"].eq(group_id) & out["pig_id"].eq(pig_id)]
            assert selected["behavior"].eq(behavior).all()
            assert selected["legacy_behavior_authority_slot"].eq(0).all()
            assert selected["label_source"].eq(
                "cvat_native_k0_burst_authority"
            ).all()

    second_group = out[out["group_id"].eq("burst_color_bbb22222_200")]
    assert second_group["legacy_behavior_authority_task_frame"].eq(6).all()
    assert audit["counts"]["dense_rows_input"] == audit["counts"]["dense_rows_output"]
    assert audit["counts"]["matched_behavior_keys"] == 3
    assert discrepancies.empty


def test_missing_k0_is_retained_but_excluded(tmp_path: Path) -> None:
    cvat_labels = {"burst_color_aaa11111_100": {"ID_1": "explore"}}
    dense_labels = {
        "burst_color_aaa11111_100": {
            "ID_1": "explore",
            "ID_2": "sitting",
        }
    }
    _write_cvat_task(tmp_path, cvat_labels)
    dense = _dense_rows(dense_labels)

    out, audit, discrepancies = apply_cvat_k0_behavior_authority(dense, tmp_path)

    missing = out[out["pig_id"].eq("ID_2")]
    assert len(missing) == 16
    assert missing["behavior"].isna().all()
    assert (~missing["include_in_training"]).all()
    assert missing["legacy_behavior_authority_status"].eq(
        "missing_k0_excluded"
    ).all()
    assert missing["label_source"].eq(
        "missing_cvat_k0_no_training_label"
    ).all()
    assert audit["counts"]["dense_keys_missing_k0"] == 1
    assert discrepancies["join_status"].tolist() == ["dense_missing_k0"]


def test_frame_export_preserves_k0_provenance(tmp_path: Path) -> None:
    labels = {"burst_color_aaa11111_100": {"ID_1": "drink"}}
    _write_cvat_task(tmp_path, labels)
    dense = _dense_rows(labels)
    dense["behavior"] = "drink"
    overlaid, authority_audit, _ = apply_cvat_k0_behavior_authority(
        dense,
        tmp_path,
    )
    overlaid["hidden_source"] = "cvat_native_anchor"
    overlaid["hidden_is_trusted"] = False
    overlaid["hidden_review_status"] = "seed_unreviewed"
    overlaid["hidden_trust_status"] = "untrusted_cvat_seed"
    overlaid["visibility_quality"] = "cvat_anchor_seed_unreviewed"
    overlaid["hidden_seed_method"] = "cvat_anchor_exact"

    exported = build_frame_object_csv(
        dense_df=overlaid,
        image_width=1280,
        image_height=720,
        training_only=False,
        dataset_id="synthetic_legacy",
        source_type="legacy_recovered",
        expected_sequence_length=16,
        anchor_relative_frames=OFFSETS,
        expected_pig_count=8,
        fps=None,
        require_full_8_for_eval=False,
    )

    assert len(exported) == 16
    assert exported["behavior"].eq("drink").all()
    assert exported["label_source"].eq("cvat_native_k0_burst_authority").all()
    assert exported["legacy_behavior_authority_status"].eq(
        "authoritative_k0"
    ).all()
    assert exported["sequence_complete"].all()
    export_audit = build_export_audit(
        overlaid,
        exported,
        behavior_authority_audit=authority_audit,
        training_only=False,
    )
    assert export_audit["errors"] == []

    export_path = tmp_path / "legacy_frame_object_annotations.csv"
    exported.to_csv(export_path, index=False)
    canonical = load_legacy_frame_objects(export_path)
    assert len(canonical) == 16
    assert canonical["label_source"].eq(
        "cvat_native_k0_burst_authority"
    ).all()
    assert (~canonical["hidden_is_trusted"].astype(bool)).all()
    assert canonical["hidden_source"].eq("cvat_native_anchor").all()
    assert canonical["hidden_review_status"].eq("seed_unreviewed").all()


def test_export_audit_rejects_stale_dense_behavior_overlay(tmp_path: Path) -> None:
    labels = {"burst_color_aaa11111_100": {"ID_1": "explore"}}
    _write_cvat_task(tmp_path, labels)
    dense = _dense_rows(labels)
    overlaid, authority_audit, _ = apply_cvat_k0_behavior_authority(
        dense,
        tmp_path,
    )
    for column, value in {
        "hidden_source": "cvat_native_anchor",
        "hidden_is_trusted": False,
        "hidden_review_status": "seed_unreviewed",
        "hidden_trust_status": "untrusted_cvat_seed",
        "visibility_quality": "cvat_anchor_seed_unreviewed",
        "hidden_seed_method": "cvat_anchor_exact",
    }.items():
        overlaid[column] = value
    exported = build_frame_object_csv(
        dense_df=overlaid,
        image_width=1280,
        image_height=720,
        training_only=False,
        dataset_id="synthetic_legacy",
        source_type="legacy_recovered",
        expected_sequence_length=16,
        anchor_relative_frames=OFFSETS,
        expected_pig_count=8,
        fps=None,
        require_full_8_for_eval=False,
    )

    audit = build_export_audit(
        overlaid,
        exported,
        behavior_authority_audit=authority_audit,
        training_only=False,
    )

    assert audit["status"] == "FAIL"
    assert "stale_dense_behavior_rows_changed_by_overlay=16" in audit["errors"]


def test_recovery_inputs_add_actor_and_keep_each_anchor_bbox(tmp_path: Path) -> None:
    group_id = "burst_color_aaa11111_100"
    cvat_labels = {group_id: {"ID_1": "explore", "ID_2": "drink"}}
    scaffold_labels = {group_id: {"ID_1": "stand"}}
    _write_cvat_task(
        tmp_path / "cvat",
        cvat_labels,
        hidden_by_slot={0: "No", 1: "No", 2: "Yes", 3: "Yes"},
    )
    scaffold_path = _write_scaffold(
        tmp_path / "scaffold.csv",
        scaffold_labels,
    )

    center, anchors, audit, issues = build_legacy_recovery_inputs(
        cvat_export_root=tmp_path / "cvat",
        metadata_scaffold_csv=scaffold_path,
    )

    assert audit["errors"] == []
    assert audit["counts"]["new_k0_keys"] == 1
    assert audit["counts"]["behavior_disagreement_rows_mapped_to_k0"] == 10
    assert audit["behavior_disagreement_by_slot_mapped_to_k0"] == {
        "0": 0,
        "1": 2,
        "2": 2,
        "3": 2,
        "4": 2,
        "5": 2,
    }
    assert len(center) == 2
    assert len(anchors) == 12
    assert set(center["pig_id"]) == {"ID_1", "ID_2"}
    assert center.set_index("pig_id").loc["ID_2", "behavior"] == "drink"
    center_id1 = center.set_index("pig_id").loc["ID_1"]
    assert center_id1["bbox_anchor_slot"] == 0
    assert center_id1["center_frame_from_img"] == 0
    assert center_id1["center_frame_final"] == 0
    assert not bool(center_id1["frame_mismatch"])
    assert center_id1[["x1", "y1", "x2", "y2"]].tolist() == [
        10.0,
        20.0,
        50.0,
        80.0,
    ]
    assert center_id1["hidden"] == "No"
    stale_spatial_columns = {
        "trigger_type",
        "roi_name",
        "near_roi",
        "distance_to_roi",
    }
    assert stale_spatial_columns.isdisjoint(center.columns)
    assert stale_spatial_columns.isdisjoint(anchors.columns)
    id1 = anchors[anchors["pig_id"].eq("ID_1")].sort_values("legacy_order")
    assert id1["behavior"].eq("explore").all()
    assert id1["behavior_before_k0_mapping"].tolist() == [
        "explore",
        "stand",
        "stand",
        "stand",
        "stand",
        "stand",
    ]
    assert id1["x1"].tolist() == [10.0 + slot for slot in range(6)]
    assert id1["hidden"].tolist() == ["No", "No", "Yes", "Yes", "No", "No"]
    assert (~id1["hidden_is_trusted"]).all()
    assert id1["hidden_review_status"].eq("seed_unreviewed").all()
    assert issues["code"].eq("new_k0_actor_key").any()

    accepted, rejected = validate_legacy_dataframe(center)
    assert rejected.empty
    recovery_records = build_anchor_records(
        accepted,
        timestamps_by_video={},
        track_end_mode="legacy_last",
    )
    records_by_pig = {record["pig_id"]: record for record in recovery_records}
    assert records_by_pig["ID_1"]["behavior"] == "explore"
    assert records_by_pig["ID_2"]["behavior"] == "drink"
    assert records_by_pig["ID_1"]["dense_frame_indices"] == list(range(16))
    assert records_by_pig["ID_2"]["dense_frame_indices"] == list(range(16))


def test_source_video_key_supports_session_suffix_and_derivation(
    tmp_path: Path,
) -> None:
    video_final = (
        "/content/drive/MyDrive/pig_data_unzipped/"
        "pigs101219b/PIGS101219/000034/color.mp4"
    )
    video_hash = hashlib.md5(video_final.encode("utf-8")).hexdigest()[:8]
    group_id = f"burst_color_{video_hash}_100"
    labels = {group_id: {"ID_1": "explore"}}
    _write_cvat_task(tmp_path / "cvat", labels)
    scaffold_path = _write_scaffold(tmp_path / "scaffold.csv", labels)
    scaffold = pd.read_csv(scaffold_path)
    scaffold["day_final"] = scaffold["day_final"].astype(object)
    scaffold["video_final"] = scaffold["video_final"].astype(object)
    scaffold["source_video_key"] = scaffold["source_video_key"].astype(object)
    scaffold["day_final"] = "pigs101219b"
    scaffold["video_final"] = video_final
    scaffold["source_video_key"] = ""
    scaffold.to_csv(scaffold_path, index=False)

    center, anchors, audit, issues = build_legacy_recovery_inputs(
        cvat_export_root=tmp_path / "cvat",
        metadata_scaffold_csv=scaffold_path,
    )

    assert normalize_source_video_key(scaffold.loc[0, "video_final"]) == (
        "pigs101219b/000034"
    )
    assert audit["errors"] == []
    assert audit["counts"]["source_video_key_groups_derived"] == 1
    assert center["source_video_key"].eq("pigs101219b/000034").all()
    assert anchors["source_video_key"].eq("pigs101219b/000034").all()
    assert issues["code"].eq("source_video_key_derived").any()


def test_source_video_key_mismatch_fails_closed(tmp_path: Path) -> None:
    group_id = "burst_color_aaa11111_100"
    labels = {group_id: {"ID_1": "explore"}}
    _write_cvat_task(tmp_path / "cvat", labels)
    scaffold_path = _write_scaffold(tmp_path / "scaffold.csv", labels)
    scaffold = pd.read_csv(scaffold_path)
    scaffold["video_final"] = (
        "/content/drive/MyDrive/pig_data_unzipped/"
        "pigs010101/PIGS010101/000001/color.mp4"
    )
    scaffold["source_video_key"] = "pigs010101/999999"
    scaffold.to_csv(scaffold_path, index=False)

    center, anchors, audit, issues = build_legacy_recovery_inputs(
        cvat_export_root=tmp_path / "cvat",
        metadata_scaffold_csv=scaffold_path,
    )

    assert audit["status"] == "FAIL"
    assert audit["counts"]["source_video_key_path_mismatch_groups"] == 1
    assert center.empty
    assert anchors.empty
    assert issues["code"].eq("source_video_key_video_path_mismatch").any()


def test_group_video_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    group_id = "burst_color_aaa11111_100"
    labels = {group_id: {"ID_1": "explore"}}
    _write_cvat_task(tmp_path / "cvat", labels)
    scaffold_path = _write_scaffold(tmp_path / "scaffold.csv", labels)
    scaffold = pd.read_csv(scaffold_path)
    scaffold["video_final"] = (
        "/content/drive/MyDrive/pig_data_unzipped/"
        "pigs010101/PIGS010101/000001/color.mp4"
    )
    scaffold["source_video_key"] = "pigs010101/000001"
    scaffold.to_csv(scaffold_path, index=False)

    center, anchors, audit, issues = build_legacy_recovery_inputs(
        cvat_export_root=tmp_path / "cvat",
        metadata_scaffold_csv=scaffold_path,
    )

    assert audit["status"] == "FAIL"
    assert audit["counts"]["group_video_hashes_checked"] == 1
    assert audit["counts"]["group_video_hash_mismatch_groups"] == 1
    assert center.empty
    assert anchors.empty
    assert issues["code"].eq("group_video_hash_mismatch").any()


def test_recovery_input_threshold_cannot_be_lowered(tmp_path: Path) -> None:
    group_id = "burst_color_aaa11111_100"
    labels = {group_id: {"ID_1": "explore"}}
    _write_cvat_task(tmp_path / "cvat", labels)
    scaffold_path = _write_scaffold(tmp_path / "scaffold.csv", labels)

    with pytest.raises(ValueError, match="exactly six CVAT anchors"):
        build_legacy_recovery_inputs(
            cvat_export_root=tmp_path / "cvat",
            metadata_scaffold_csv=scaffold_path,
            min_anchor_count=5,
        )


def test_missing_cvat_hidden_attribute_fails_closed(tmp_path: Path) -> None:
    group_id = "burst_color_aaa11111_100"
    labels = {group_id: {"ID_1": "explore"}}
    task_dir = _write_cvat_task(tmp_path / "cvat", labels)
    annotation_path = task_dir / "annotations.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload[0]["shapes"][0]["attributes"] = [
        item
        for item in payload[0]["shapes"][0]["attributes"]
        if item["name"] != "Hidden"
    ]
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")
    scaffold_path = _write_scaffold(tmp_path / "scaffold.csv", labels)

    center, anchors, audit, issues = build_legacy_recovery_inputs(
        cvat_export_root=tmp_path / "cvat",
        metadata_scaffold_csv=scaffold_path,
    )

    assert audit["status"] == "FAIL"
    assert center.empty
    assert anchors.empty
    assert any("invalid_hidden_rows" in error for error in audit["errors"])
    assert issues["code"].eq("invalid_hidden_attribute").any()


def test_recovery_inputs_exclude_actor_without_all_six_anchors(
    tmp_path: Path,
) -> None:
    group_id = "burst_color_aaa11111_100"
    labels = {group_id: {"ID_1": "explore", "ID_2": "drink"}}
    task_dir = _write_cvat_task(tmp_path / "cvat", labels)
    annotation_path = task_dir / "annotations.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload[0]["shapes"] = [
        shape
        for shape in payload[0]["shapes"]
        if not (
            next(
                item["value"]
                for item in shape["attributes"]
                if item["name"] == "ID"
            )
            == "ID_2"
            and int(shape["frame"]) not in {0, 3}
        )
    ]
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")
    scaffold_path = _write_scaffold(
        tmp_path / "scaffold.csv",
        {group_id: {"ID_1": "stand"}},
    )

    center, anchors, audit, issues = build_legacy_recovery_inputs(
        cvat_export_root=tmp_path / "cvat",
        metadata_scaffold_csv=scaffold_path,
    )

    assert audit["errors"] == []
    assert audit["counts"]["excluded_below_min_anchor_count"] == 1
    assert center["pig_id"].tolist() == ["ID_1"]
    assert anchors["pig_id"].eq("ID_1").all()
    excluded = issues[
        issues["code"].eq("incomplete_anchor_set")
        & issues["pig_id"].eq("ID_2")
    ]
    assert excluded["severity"].tolist() == ["excluded"]


def test_recovery_inputs_fail_on_duplicate_anchor_identity(tmp_path: Path) -> None:
    group_id = "burst_color_aaa11111_100"
    labels = {group_id: {"ID_1": "explore"}}
    task_dir = _write_cvat_task(tmp_path / "cvat", labels)
    annotation_path = task_dir / "annotations.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    duplicate = dict(payload[0]["shapes"][1])
    duplicate["attributes"] = [dict(item) for item in duplicate["attributes"]]
    payload[0]["shapes"].append(duplicate)
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")
    scaffold_path = _write_scaffold(
        tmp_path / "scaffold.csv",
        {group_id: {"ID_1": "stand"}},
    )

    center, anchors, audit, issues = build_legacy_recovery_inputs(
        cvat_export_root=tmp_path / "cvat",
        metadata_scaffold_csv=scaffold_path,
    )

    assert audit["status"] == "FAIL"
    assert any("duplicate_anchor_identity_rows" in item for item in audit["errors"])
    assert center.empty
    assert anchors.empty
    assert issues["code"].eq("duplicate_anchor_identity").any()


def test_duplicate_k0_authority_is_rejected(tmp_path: Path) -> None:
    labels = {"burst_color_aaa11111_100": {"ID_1": "explore"}}
    task_dir = _write_cvat_task(tmp_path, labels)
    path = task_dir / "annotations.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    duplicate = dict(payload[0]["shapes"][0])
    duplicate["attributes"] = [dict(item) for item in duplicate["attributes"]]
    payload[0]["shapes"].append(duplicate)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate_authority_keys"):
        apply_cvat_k0_behavior_authority(_dense_rows(labels), tmp_path)


def _recovered_validation_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_id = "burst_color_aaa11111_100"
    pig_id = "ID_1"
    anchor_frames = [3, 6, 9, 12, 15, 18]
    center = pd.DataFrame(
        [
            {
                "group_id": group_id,
                "pig_id": pig_id,
                "behavior": "explore",
                "frames": "3|6|9|12|15|18",
                "behavior_authority_slot": 0,
                "center_frame_from_img": 3,
                "center_frame_final": 3,
                "frame_mismatch": False,
                "bbox_anchor_slot": 0,
                "hidden": "No",
                "x1": 10.0,
                "y1": 20.0,
                "x2": 50.0,
                "y2": 80.0,
            }
        ]
    )
    anchor_hidden = ["No", "No", "Yes", "Yes", "No", "No"]
    anchors = pd.DataFrame(
        [
            {
                "group_id": group_id,
                "pig_id": pig_id,
                "frame_index": frame_index,
                "legacy_order": slot,
                "behavior": "explore",
                "hidden": anchor_hidden[slot],
                "x1": 10.0 + slot,
                "y1": 20.0,
                "x2": 50.0 + slot,
                "y2": 80.0,
            }
            for slot, frame_index in enumerate(anchor_frames)
        ]
    )
    dense_rows: list[dict[str, object]] = []
    anchor_by_frame = anchors.set_index("frame_index")
    hidden_anchor_map = {
        int(row.frame_index): {"hidden": row.hidden}
        for row in anchors.itertuples(index=False)
    }
    for frame_index in range(3, 19):
        is_anchor = frame_index in anchor_by_frame.index
        bbox = (
            anchor_by_frame.loc[frame_index]
            if is_anchor
            else pd.Series({"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0})
        )
        hidden_seed = hidden_seed_for_frame(
            frame_index,
            hidden_anchor_map,
            fallback_hidden="No",
        )
        dense_rows.append(
            {
                "group_id": group_id,
                "pig_id": pig_id,
                "frame_index": frame_index,
                "behavior": "explore",
                **hidden_seed,
                "x1": bbox["x1"],
                "y1": bbox["y1"],
                "x2": bbox["x2"],
                "y2": bbox["y2"],
                "bbox_source": "gt_legacy" if is_anchor else "interpolated",
                "legacy_gt_bbox_available": is_anchor,
            }
        )
    return center, anchors, pd.DataFrame(dense_rows)


def test_recovered_dense_keeps_k0_behavior_and_each_cvat_anchor_bbox() -> None:
    center, anchors, dense = _recovered_validation_fixture()

    audit = validate_cvat_recovered_dense(center, anchors, dense)

    assert audit["status"] == "PASS"
    assert audit["errors"] == []
    assert audit["counts"]["dense_rows"] == 16
    assert audit["behavior_checks"] == {
        "anchor_behavior_not_mapped_from_k0": 0,
        "dense_behavior_not_mapped_from_k0": 0,
    }
    assert all(count == 0 for count in audit["anchor_checks"].values())
    assert all(count == 0 for count in audit["center_checks"].values())


def test_recovered_dense_validator_rejects_behavior_or_bbox_drift() -> None:
    center, anchors, dense = _recovered_validation_fixture()
    anchors.loc[anchors["frame_index"].eq(6), "behavior"] = "stand"
    dense.loc[dense["frame_index"].eq(4), "behavior"] = "stand"
    dense.loc[dense["frame_index"].eq(9), "x1"] += 2.0

    audit = validate_cvat_recovered_dense(center, anchors, dense)

    assert audit["status"] == "FAIL"
    assert "anchor_behavior_not_mapped_from_k0=1" in audit["errors"]
    assert "dense_behavior_not_mapped_from_k0=1" in audit["errors"]
    assert "cvat_anchor_bbox_mismatches=1" in audit["errors"]


def test_recovered_dense_validator_rejects_center_k0_drift() -> None:
    center, anchors, dense = _recovered_validation_fixture()
    center.loc[0, "x1"] += 1.0
    center.loc[0, "hidden"] = "Yes"
    center.loc[0, "center_frame_final"] = 4

    audit = validate_cvat_recovered_dense(center, anchors, dense)

    assert audit["status"] == "FAIL"
    assert "center_k0_frame_contract_mismatches=1" in audit["errors"]
    assert "center_k0_bbox_mismatches=1" in audit["errors"]
    assert "center_k0_hidden_mismatches=1" in audit["errors"]


def test_hidden_seed_uses_anchor_exact_agreement_and_conservative_transition() -> None:
    anchors = {
        0: {"hidden": "No"},
        3: {"hidden": "No"},
        6: {"hidden": "Yes"},
    }

    exact = hidden_seed_for_frame(3, anchors, fallback_hidden="No")
    agreement = hidden_seed_for_frame(1, anchors, fallback_hidden="No")
    transition = hidden_seed_for_frame(4, anchors, fallback_hidden="No")

    assert exact["hidden"] == "No"
    assert exact["hidden_seed_method"] == "cvat_anchor_exact"
    assert agreement["hidden"] == "No"
    assert agreement["hidden_seed_method"] == "cvat_anchor_pair_agreement"
    assert transition["hidden"] == "Yes"
    assert transition["hidden_seed_method"] == (
        "cvat_anchor_transition_conservative"
    )
    assert exact["hidden_is_trusted"] is False
    assert agreement["hidden_is_trusted"] is False
    assert transition["hidden_is_trusted"] is False


def test_recovered_dense_validator_rejects_hidden_drift_or_false_trust() -> None:
    center, anchors, dense = _recovered_validation_fixture()
    dense.loc[dense["frame_index"].eq(4), "hidden"] = "Yes"
    dense.loc[dense["frame_index"].eq(5), "hidden_is_trusted"] = True

    audit = validate_cvat_recovered_dense(center, anchors, dense)

    assert audit["status"] == "FAIL"
    assert "dense_hidden_seed_mismatches=1" in audit["errors"]
    assert "dense_hidden_incorrectly_trusted=1" in audit["errors"]
