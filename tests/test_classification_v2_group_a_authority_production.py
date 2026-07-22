from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.contracts.timestamp_fps import (
    build_timestamp_fps_contract,
)
from pig_behavior.classification_v2.features.frame_local import (
    FRAME_LOCAL_GRAIN,
    audit_frame_local_primitives,
    build_frame_local_primitives,
    forbidden_frame_local_columns,
    frame_local_schema_payload,
)
from pig_behavior.classification_v2.features.social import (
    build_static_social_context_features,
)
from pig_behavior.classification_v2.features.spatiotemporal import (
    build_enhanced_spatiotemporal_features,
)
from pig_behavior.classification_v2.features.temporal_harmonization import (
    attach_structural_temporal_unit_identity,
)
from pig_behavior.classification_v2.review.evidence_semantics import (
    build_evidence_semantics,
)
from pig_behavior.classification_v2.review.media_authority import (
    build_behavior_review_media_authority,
    finalize_media_authority_summary,
)
from pig_behavior.classification_v2.review.review_authority import (
    OFFICIAL_SCOPE,
    REQUIRED_COMPONENT_GATE_KEYS,
    build_review_authority_manifest,
)

CODE_SHA = "a" * 40
LINEAGE = "c2v2_human_review_20260722_reviewer01_v6"


def test_frame_local_preserves_rows_keys_clock_and_forbids_pairs(
    tmp_path: Path,
) -> None:
    source = _source_rows()
    output = _build_frame_local(source, tmp_path)
    audit = audit_frame_local_primitives(source, output)

    assert audit["errors"] == []
    assert output["frame_uid"].tolist() == source["frame_uid"].tolist()
    assert output["feature_computation_grain"].eq(FRAME_LOCAL_GRAIN).all()
    assert output["timestamp_sec"].to_numpy() == pytest.approx(
        output["source_frame_index"].to_numpy() / 30.0
    )
    assert output["acquisition_timestamp_sec"].tolist() == source[
        "timestamp_sec"
    ].tolist()
    assert forbidden_frame_local_columns(output.columns) == []
    assert "temporal_unit_key" in output
    assert output["temporal_unit_key"].str.endswith("|anchor=0").all()
    assert output.groupby("temporal_unit_key", sort=False).size().eq(6).all()
    assert "pen_distance_delta_n_per_second" not in output
    assert "speed_n_per_second" not in output


def test_frame_local_is_deterministic_and_static_social_is_order_invariant(
    tmp_path: Path,
) -> None:
    source = _source_rows()
    first = _build_frame_local(source, tmp_path)
    second = _build_frame_local(source, tmp_path)
    pd.testing.assert_frame_equal(first, second)

    base = first.drop(columns=[column for column in first if column.startswith("nearest_")])
    base = base.drop(
        columns=[column for column in base if column.startswith("social_")]
        + ["pair_contact_with_nearest"]
    )
    ordered = build_static_social_context_features(base)
    shuffled = build_static_social_context_features(
        base.sample(frac=1.0, random_state=7)
    )
    columns = [
        "frame_uid",
        "nearest_pig_id",
        "nearest_dist_n",
        "nearest_pair_iou",
    ]
    pd.testing.assert_frame_equal(
        ordered[columns].sort_values("frame_uid").reset_index(drop=True),
        shuffled[columns].sort_values("frame_uid").reset_index(drop=True),
    )


def test_forbidden_pair_semantics_registry_is_fail_closed() -> None:
    assert forbidden_frame_local_columns(
        ["cx_n", "speed_n_per_second", "prev_timestamp_sec"]
    ) == ["prev_timestamp_sec", "speed_n_per_second"]
    assert forbidden_frame_local_columns(["temporal_unit_key"]) == []


def test_structural_temporal_identity_covers_cvat_and_legacy_units() -> None:
    cvat = pd.DataFrame(
        {
            "source_type": ["cvat_tracking_xml"] * 12,
            "dataset_id": ["cvat"] * 12,
            "video_key": ["video-cvat"] * 12,
            "frame_index": list(range(12)),
            "pig_id": ["ID_1"] * 12,
            "track_id": ["1"] * 12,
        }
    )
    cvat_identity = attach_structural_temporal_unit_identity(cvat)
    assert cvat_identity["temporal_unit_key"].nunique() == 2
    assert cvat_identity.loc[:5, "temporal_unit_key"].nunique() == 1
    assert cvat_identity.loc[6:, "temporal_unit_key"].nunique() == 1

    legacy = pd.DataFrame(
        {
            "source_type": ["legacy_recovered"] * 16,
            "dataset_id": ["legacy"] * 16,
            "video_key": ["burst-color"] * 16,
            "frame_index": list(range(16)),
            "relative_frame_index": list(range(16)),
            "pig_id": ["ID_2"] * 16,
            "track_id": ["2"] * 16,
        }
    )
    legacy_identity = attach_structural_temporal_unit_identity(legacy)
    assert legacy_identity["temporal_unit_key"].nunique() == 1
    assert legacy_identity["temporal_unit_key"].str.endswith(
        "|legacy_sequence"
    ).all()

    drift = cvat.copy()
    drift["temporal_unit_key"] = "wrong-unit"
    with pytest.raises(ValueError, match="disagrees with structural identity"):
        attach_structural_temporal_unit_identity(drift)


def test_frame_local_independent_checker_detects_content_drift(
    tmp_path: Path,
) -> None:
    source = _source_rows()
    output = _build_frame_local(source, tmp_path)
    source_path = tmp_path / "source.csv"
    output_path = tmp_path / "frame_local.csv"
    schema_path = tmp_path / "schema.json"
    builder_audit_path = tmp_path / "builder_audit.json"
    source.to_csv(source_path, index=False)
    output.to_csv(output_path, index=False)
    schema = frame_local_schema_payload(output)
    schema.update({"lineage_id": LINEAGE, "code_authority_sha": CODE_SHA})
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    builder_audit = audit_frame_local_primitives(source, output)
    builder_audit.update(
        {"lineage_id": LINEAGE, "code_authority_sha": CODE_SHA}
    )
    builder_audit_path.write_text(json.dumps(builder_audit), encoding="utf-8")
    command = [
        sys.executable,
        "scripts/classification_v2/00_source_feature_temporal/"
        "check_classification_v2_frame_local_primitives.py",
        "--source-csv",
        str(source_path),
        "--frame-local-csv",
        str(output_path),
        "--roi-coco",
        str(tmp_path / "roi.json"),
        "--pen-mask",
        str(tmp_path / "mask.png"),
        "--expected-pen-mask-sha256",
        "",
        "--schema-json",
        str(schema_path),
        "--builder-audit-json",
        str(builder_audit_path),
        "--lineage-id",
        LINEAGE,
        "--code-authority-sha",
        CODE_SHA,
        "--output-json",
        str(tmp_path / "checker.json"),
    ]
    passed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert passed.returncode == 0, passed.stderr

    drift = pd.read_csv(output_path, low_memory=False)
    drift.loc[0, "nearest_dist_n"] = 999.0
    drift.to_csv(output_path, index=False)
    command[-1] = str(tmp_path / "checker_drift.json")
    failed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert failed.returncode == 2


def test_frame_local_output_passes_hidden_production_structural_gate(
    tmp_path: Path,
) -> None:
    frame_local = _build_frame_local(_source_rows(), tmp_path)
    input_path = tmp_path / "frame_local_for_hidden.csv"
    output_dir = tmp_path / "hidden_smoke"
    frame_local.to_csv(input_path, index=False)
    command = [
        sys.executable,
        "scripts/classification_v2/01_review_units_gui/"
        "classification_v2_build_hidden_review_units.py",
        "--input-csv",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--design-scope",
        "smoke",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    audit = json.loads(
        (output_dir / "hidden_review_template_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["structural_audit"]["errors"] == []
    assert audit["structural_audit"]["temporal_unit_count"] == 2


def test_timestamp_contract_accepts_30fps_and_rejects_wrong_clock(
    tmp_path: Path,
) -> None:
    frame = _build_frame_local(_source_rows(), tmp_path)
    source_artifact = tmp_path / "source_lineage.json"
    source_artifact.write_text("{}", encoding="utf-8")
    kwargs = {
        "lineage_id": LINEAGE,
        "code_authority_sha": CODE_SHA,
        "source_lineage_artifacts": {"merged": source_artifact},
        "video_fps_authority": {
            "Pigs291119_000231_30fps": {
                "authority": "decoded_video_container_metadata",
                "fps": 30.0,
            }
        },
    }
    contract = build_timestamp_fps_contract(frame, **kwargs)
    assert contract["errors"] == []

    wrong = frame.copy()
    wrong.loc[wrong["source_frame_index"].eq(1), "timestamp_sec"] = 0.2
    failed = build_timestamp_fps_contract(wrong, **kwargs)
    assert failed["valid"] is False
    assert any("timestamp_formula_mismatch" in item for item in failed["errors"])


def test_evidence_semantics_detects_scope_drift_and_model_x_leakage(
    tmp_path: Path,
) -> None:
    frame_local = _build_frame_local(_source_rows(), tmp_path)
    native = build_enhanced_spatiotemporal_features(frame_local)
    semantics = build_evidence_semantics(frame_local, native)
    assert semantics["errors"] == []
    assert semantics["fields"]["behavior"]["model_x_eligibility"] == "forbidden"
    assert semantics["fields"]["review_feature_reason_auto"][
        "model_x_eligibility"
    ] == "forbidden"
    assert semantics["fields"]["use_for_roi_training"][
        "model_x_eligibility"
    ] == "forbidden"
    assert semantics["fields"]["frame_index"]["model_x_eligibility"] == "forbidden"
    assert semantics["mask_semantics"]["observed_mask"] != semantics[
        "mask_semantics"
    ]["spatial_quality_mask"]

    drift = native.copy()
    drift.loc[drift.index[0], "pair_scope_key"] = "wrong-unit"
    failed = build_evidence_semantics(frame_local, drift)
    assert any("native_pair_scope_mismatch" in item for item in failed["errors"])


def test_media_authority_binds_frames_actor_span_and_basename(
    tmp_path: Path,
) -> None:
    frames, units, video_root, crop_root = _media_fixture(tmp_path)
    index, summary = build_behavior_review_media_authority(
        units,
        frames,
        video_root=video_root,
        legacy_crop_root=crop_root,
    )
    index_path = tmp_path / "media_index.csv"
    index.to_csv(index_path, index=False)
    final = finalize_media_authority_summary(summary, index_csv=index_path)
    repeated_index, repeated_summary = build_behavior_review_media_authority(
        units,
        frames,
        video_root=video_root,
        legacy_crop_root=crop_root,
    )

    assert final["errors"] == []
    assert final["valid"] is True
    assert index["all_scene_media_available"].all()
    assert index["all_actor_media_available"].all()
    assert index.iloc[0]["selected_source_frames"] == "[0,1,2,3,4,5]"
    pd.testing.assert_frame_equal(index, repeated_index)
    assert summary == repeated_summary

    wrong_span = units.copy()
    wrong_span.loc[0, "unit_end_frame"] = 6
    _, failed_span = build_behavior_review_media_authority(
        wrong_span,
        frames,
        video_root=video_root,
        legacy_crop_root=crop_root,
    )
    assert any("media_frame_span_mismatch" in item for item in failed_span["errors"])

    missing = frames.copy()
    missing.loc[0, "scene_image_path"] = str(tmp_path / "missing.png")
    _, failed_media = build_behavior_review_media_authority(
        units,
        missing,
        video_root=video_root,
        legacy_crop_root=crop_root,
    )
    assert any("missing_scene_media" in item for item in failed_media["errors"])

    wrong_actor = units.copy()
    wrong_actor.loc[0, "object_track_key"] = "wrong-actor"
    _, failed_actor = build_behavior_review_media_authority(
        wrong_actor,
        frames,
        video_root=video_root,
        legacy_crop_root=crop_root,
    )
    assert any("no_matching_actor" in item for item in failed_actor["errors"])


def test_media_authority_rejects_wrong_000231_basename_and_v3_reference(
    tmp_path: Path,
) -> None:
    frames, units, video_root, crop_root = _media_fixture(tmp_path)
    wrong = tmp_path / "wrong_video_frame.png"
    cv2.imwrite(str(wrong), np.zeros((100, 100, 3), dtype=np.uint8))
    frames["scene_image_path"] = str(wrong)
    _, basename = build_behavior_review_media_authority(
        units,
        frames,
        video_root=video_root,
        legacy_crop_root=crop_root,
    )
    assert any("wrong_video_basename" in item for item in basename["errors"])

    v3 = frames.copy()
    v3["crop_path"] = v3["crop_path"].astype(str) + (
        "c2v2_human_review_20260721_reviewer01_v3"
    )
    _, stopped = build_behavior_review_media_authority(
        units,
        v3,
        video_root=video_root,
        legacy_crop_root=crop_root,
    )
    assert stopped["valid"] is False
    assert any("stopped_v3_media_reference" in item for item in stopped["errors"])


def test_official_authority_requires_all_clean_same_lineage_gates(
    tmp_path: Path,
) -> None:
    source, artifacts, timestamp, semantics, gates = _authority_fixture(tmp_path)
    kwargs = {
        "code_authority_sha": CODE_SHA,
        "code_dirty": False,
        "lineage_id": LINEAGE,
        "authority_scope": OFFICIAL_SCOPE,
        "source_artifacts": source,
        "artifacts": artifacts,
        "timestamp_fps_contract": timestamp,
        "evidence_semantics": semantics,
        "component_gates": gates,
        "actual_head_sha": CODE_SHA,
        "tracked_code_clean": True,
        "require_full_component_gates": True,
    }
    manifest = build_review_authority_manifest(**kwargs)
    assert manifest["errors"] == []
    assert manifest["authorizes_behavior_gui"] is True
    assert manifest["authorizes_final_view_build"] is False
    assert manifest["authorizes_training"] is False

    stopped_v5 = build_review_authority_manifest(
        **{
            **kwargs,
            "lineage_id": "c2v2_human_review_20260722_reviewer01_v5",
        }
    )
    assert stopped_v5["authorizes_behavior_gui"] is False
    assert "official_authority_requires_v6_lineage_id" in stopped_v5["errors"]

    first_gate = next(iter(gates.values()))
    first_gate.write_text(
        json.dumps({"lineage_id": LINEAGE, "valid": True, "errors": [], "x": 1}),
        encoding="utf-8",
    )
    hash_drift = build_review_authority_manifest(**kwargs)
    assert hash_drift["review_authority_sha256"] != manifest[
        "review_authority_sha256"
    ]

    drift = build_review_authority_manifest(
        **{**kwargs, "actual_head_sha": "b" * 40}
    )
    assert drift["authorizes_behavior_gui"] is False
    dirty = build_review_authority_manifest(
        **{**kwargs, "tracked_code_clean": False}
    )
    assert dirty["authorizes_behavior_gui"] is False
    missing = dict(gates)
    missing.pop("media_authority")
    absent = build_review_authority_manifest(
        **{**kwargs, "component_gates": missing}
    )
    assert absent["authorizes_behavior_gui"] is False


def _source_rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for frame in range(6):
        for actor, x1 in (("ID_1", 10.0 + frame), ("ID_2", 50.0)):
            records.append(
                {
                    "source_type": "cvat_tracking_xml",
                    "dataset_id": "cvat",
                    "video_key": "Pigs291119_000231_30fps",
                    "scene_frame_uid": f"scene-{frame}",
                    "frame_uid": f"scene-{frame}-{actor}",
                    "object_id_in_image": actor,
                    "frame_index": frame,
                    "pig_id": actor,
                    "track_id": actor,
                    "behavior": "move",
                    "hidden": "No",
                    "bbox_valid": True,
                    "x1": x1,
                    "y1": 20.0,
                    "x2": x1 + 20.0,
                    "y2": 50.0,
                    "image_width": 100,
                    "image_height": 100,
                    "timestamp_sec": 1000.0 + frame * 0.16,
                    "timestamp_source": "times.txt",
                }
            )
    return pd.DataFrame.from_records(records)


def _build_frame_local(source: pd.DataFrame, root: Path) -> pd.DataFrame:
    roi = root / "roi.json"
    roi.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "width": 100, "height": 100}],
                "categories": [{"id": 1, "name": "feeder"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 20, 100],
                        "segmentation": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    mask = root / "mask.png"
    image = np.zeros((100, 100), dtype=np.uint8)
    image[5:95, 5:95] = 255
    cv2.imwrite(str(mask), image)
    return build_frame_local_primitives(
        source,
        roi_coco_path=roi,
        pen_mask_path=mask,
        expected_pen_mask_sha256=None,
    )


def _media_fixture(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    video_root = root / "videos"
    crop_root = root / "crops"
    video_root.mkdir()
    crop_root.mkdir()
    rows: list[dict[str, object]] = []
    for frame in range(6):
        scene = root / f"Pigs291119_000231_30fps_frame_{frame}.png"
        crop = crop_root / f"actor_{frame}.png"
        cv2.imwrite(str(scene), np.zeros((100, 100, 3), dtype=np.uint8))
        cv2.imwrite(str(crop), np.zeros((30, 20, 3), dtype=np.uint8))
        rows.append(
            {
                "temporal_unit_key": "unit-1020",
                "source_type": "cvat_tracking_xml",
                "dataset_id": "cvat",
                "video_key": "Pigs291119_000231_30fps",
                "frame_index": frame,
                "source_frame_index": frame,
                "pig_id": "ID_1",
                "track_id": "1",
                "object_track_key": "track-1",
                "x1": 10,
                "y1": 20,
                "x2": 30,
                "y2": 50,
                "image_width": 100,
                "image_height": 100,
                "scene_image_path": str(scene),
                "crop_path": str(crop),
                "nearest_pig_id": "ID_2",
                "social_context_valid": True,
            }
        )
    units = pd.DataFrame(
        {
            "review_unit_id": ["review-1020"],
            "temporal_unit_key": ["unit-1020"],
            "source_type": ["cvat_tracking_xml"],
            "dataset_id": ["cvat"],
            "video_key": ["Pigs291119_000231_30fps"],
            "pig_id": ["ID_1"],
            "track_id": ["1"],
            "object_track_key": ["track-1"],
            "unit_start_frame": [0],
            "unit_end_frame": [5],
            "display_frame_indices": ["[0,1,2,3,4,5]"],
        }
    )
    return pd.DataFrame(rows), units, video_root, crop_root


def _authority_fixture(tmp_path: Path):
    lineage_root = tmp_path / LINEAGE
    lineage_root.mkdir()
    source_path = lineage_root / "source.csv"
    pd.DataFrame({"source_frame_index": [0]}).to_csv(source_path, index=False)
    artifacts: dict[str, Path] = {}
    for name in (
        "frame_local",
        "hidden_reviewed_frames",
        "harmonized_frames",
        "temporal_native_units",
        "behavior_review_units",
    ):
        path = lineage_root / f"{name}.csv"
        artifacts[name] = path
    pd.DataFrame(
        {
            "feature_computation_grain": [FRAME_LOCAL_GRAIN],
            "frame_uid": ["f0"],
            "source_frame_index": [0],
            "timestamp_sec": [0.0],
        }
    ).to_csv(artifacts["frame_local"], index=False)
    pd.DataFrame({"frame_uid": ["f0"], "hidden": ["No"]}).to_csv(
        artifacts["hidden_reviewed_frames"], index=False
    )
    pd.DataFrame({"frame_uid": ["f0"], "behavior": ["move"]}).to_csv(
        artifacts["harmonized_frames"], index=False
    )
    pd.DataFrame(
        {
            "temporal_unit_key": ["u0"],
            "label_window_start": [0],
            "label_window_end": [5],
        }
    ).to_csv(artifacts["temporal_native_units"], index=False)
    pd.DataFrame(
        {
            "review_unit_id": ["u0"],
            "unit_start_frame": [0],
            "unit_end_frame": [5],
        }
    ).to_csv(artifacts["behavior_review_units"], index=False)
    for name in ("pig_strenet_evidence", "media_authority"):
        path = lineage_root / f"{name}.json"
        path.write_text(json.dumps({"valid": True, "errors": []}), encoding="utf-8")
        artifacts[name] = path
    gates: dict[str, Path] = {}
    for name in REQUIRED_COMPONENT_GATE_KEYS:
        path = lineage_root / f"gate_{name}.json"
        path.write_text(
            json.dumps({"lineage_id": LINEAGE, "valid": True, "errors": []}),
            encoding="utf-8",
        )
        gates[name] = path
    timestamp = {"lineage_id": LINEAGE, "valid": True, "errors": []}
    semantics = {
        "lineage_id": LINEAGE,
        "valid": True,
        "errors": [],
        "evidence_column_semantic_version": "test.v1",
    }
    timestamp_path = lineage_root / "timestamp_fps_contract.json"
    timestamp_path.write_text(json.dumps(timestamp), encoding="utf-8")
    semantics_path = lineage_root / "evidence_semantics.json"
    semantics_path.write_text(json.dumps(semantics), encoding="utf-8")
    artifacts["timestamp_fps_contract"] = timestamp_path
    artifacts["evidence_semantics"] = semantics_path
    return {"merged": source_path}, artifacts, timestamp, semantics, gates
