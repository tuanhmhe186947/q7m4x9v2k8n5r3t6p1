from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.classification_v2.contracts.identifiers import (
    FRAME_OBJECT_IDENTIFIER_VERSION,
)
from pig_behavior.classification_v2.contracts.source_to_window_lineage import (
    audit_source_to_window_lineage,
)
from pig_behavior.classification_v2.contracts.window_alignment import (
    ordered_window_id_sha256,
    require_ordered_window_ids,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS


def _load_identifier_lineage_checker():
    """Load the numbered operator script without making it a package API."""

    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "classification_v2"
        / "09_final_release_audit"
        / "check_classification_v2_identifier_v2_lineage.py"
    )
    spec = importlib.util.spec_from_file_location(
        "classification_v2_identifier_lineage_checker",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_identifier_lineage_checker_reads_existing_csv(tmp_path: Path) -> None:
    checker = _load_identifier_lineage_checker()
    csv_path = tmp_path / "bounded.csv"
    pd.DataFrame({"window_id": ["window-0"]}).to_csv(csv_path, index=False)
    errors: list[str] = []

    rows = checker._read_csv(csv_path, "bounded", errors)

    assert rows["window_id"].tolist() == ["window-0"]
    assert errors == []


def test_source_to_window_lineage_accepts_complete_bounded_fixture() -> None:
    result = audit_source_to_window_lineage(**_valid_inputs())

    assert result["technical_pass"] is True
    assert result["status"] == "PASS_IDENTIFIER_V2_TECHNICAL_HUMAN_REVIEW_BLOCKED"
    assert not any(result["authorization"].values())
    assert len(result["human_review_blockers"]) == 2


def test_source_to_window_lineage_rejects_frame_uid_drift() -> None:
    inputs = _valid_inputs()
    geometry = inputs["frame_stages"]["geometry"].copy()
    geometry.loc[0, "frame_uid"] = "unexpected-frame-object"
    inputs["frame_stages"]["geometry"] = geometry

    result = audit_source_to_window_lineage(**inputs)

    assert result["technical_pass"] is False
    assert any("missing_frame_uid_count=1" in error for error in result["errors"])


def test_source_to_window_lineage_rejects_reordered_image_windows() -> None:
    inputs = _valid_inputs()
    inputs["image_window_manifest"] = (
        inputs["image_window_manifest"].iloc[::-1].reset_index(drop=True)
    )

    result = audit_source_to_window_lineage(**inputs)

    assert result["technical_pass"] is False
    assert any("window_order_mismatch_rows" in error for error in result["errors"])


def test_source_to_window_lineage_accepts_full_interaction_packet() -> None:
    inputs = _valid_inputs()
    sequence_manifest = inputs["sequence_manifest"]
    assert isinstance(sequence_manifest, pd.DataFrame)
    inputs["interaction_window_manifest"] = sequence_manifest.copy()
    expected_hash = ordered_window_id_sha256(
        sequence_manifest["window_id"]
    )
    artifact_audits = inputs["artifact_audits"]
    assert isinstance(artifact_audits, dict)
    artifact_audits["interaction_context"] = {
        "window_alignment": {
            "reference_ordered_window_id_sha256": expected_hash,
            "comparisons": {
                "interaction_context_windows": {
                    "ordered_window_id_sha256": expected_hash,
                }
            },
        },
        "errors": [],
    }
    inputs["require_interaction_lineage"] = True

    result = audit_source_to_window_lineage(**inputs)

    assert result["technical_pass"] is True
    assert result["full_multimodal_lineage_complete"] is True


def test_source_to_window_lineage_rejects_missing_required_interaction() -> None:
    inputs = _valid_inputs()
    inputs["require_interaction_lineage"] = True

    result = audit_source_to_window_lineage(**inputs)

    assert result["technical_pass"] is False
    assert "required_interaction_lineage_incomplete" in result["errors"]


def test_ordered_window_contract_rejects_blank_and_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="blank_split_window_ids=1"):
        require_ordered_window_ids(
            "split",
            pd.Series(["window-0", ""]),
        )
    with pytest.raises(ValueError, match="duplicate_split_window_id_rows=2"):
        require_ordered_window_ids(
            "split",
            pd.Series(["window-0", "window-0"]),
        )


def _valid_inputs() -> dict[str, object]:
    labels = list(VALID_BEHAVIORS)
    rows = len(labels)
    sources = [
        "legacy_recovered" if index % 2 == 0 else "cvat_tracking_xml"
        for index in range(rows)
    ]
    context = pd.DataFrame(
        {
            "identifier_schema_version": [FRAME_OBJECT_IDENTIFIER_VERSION] * rows,
            "scene_frame_uid": [f"scene-{index}" for index in range(rows)],
            "frame_uid": [f"frame-{index}" for index in range(rows)],
            "source_type": sources,
            "dataset_id": [f"dataset-{index % 2}" for index in range(rows)],
            "video_key": [f"video-{index % 2}" for index in range(rows)],
            "clip_id": [f"clip-{index % 2}" for index in range(rows)],
            "task_id": [f"task-{index % 2}" for index in range(rows)],
            "pig_id": [f"pig-{index}" for index in range(rows)],
            "track_id": [f"track-{index}" for index in range(rows)],
            "object_track_key": [f"track-{index}" for index in range(rows)],
            "frame_index": list(range(rows)),
        }
    )
    frame_stages = {
        name: context.copy()
        for name in ["context", "geometry", "roi", "enhanced", "harmonized"]
    }
    window_ids = pd.Series([f"window-{index}" for index in range(rows)])
    sequence_manifest = pd.DataFrame(
        {
            "window_id": window_ids,
            "source_type": sources,
        }
    )
    sequence_features = pd.DataFrame(
        {
            "window_id": window_ids,
            "behavior_window_label": labels,
        }
    )
    expected_hash = ordered_window_id_sha256(window_ids)
    artifact_audits = {
        "train_ready": {
            "window_alignment": {
                "reference_ordered_window_id_sha256": expected_hash,
            },
            "feature_selection": {"errors": []},
        },
        "spatial": {
            "window_alignment": {
                "reference_ordered_window_id_sha256": expected_hash,
            },
            "errors": [],
        },
        "image_context": {
            "window_alignment": {
                "reference_ordered_window_id_sha256": expected_hash,
                "comparisons": {
                    "image_context_windows": {
                        "ordered_window_id_sha256": expected_hash,
                    }
                },
            },
            "errors": [],
        },
    }
    return {
        "frame_stages": frame_stages,
        "sequence_manifest": sequence_manifest,
        "sequence_features": sequence_features,
        "image_frame_manifest": context.iloc[::-1].reset_index(drop=True),
        "image_window_manifest": sequence_manifest.copy(),
        "x_columns": ["speed_mean_n_per_frame"],
        "artifact_audits": artifact_audits,
        "artifact_row_counts": {
            "X_window_features": rows,
            "y_behavior": rows,
            "train_mask": rows,
            "sample_weight": rows,
        },
        "spatial_array_rows": {
            "bbox_xywh_n": rows,
            "length_mask": rows,
        },
        "preload_errors": [],
    }
