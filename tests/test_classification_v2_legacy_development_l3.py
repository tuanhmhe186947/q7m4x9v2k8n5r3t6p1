from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.contracts.legacy_development_l3 import (
    AVAILABILITY_COLUMNS,
    IMAGE_AVAILABILITY_COLUMNS,
    LEGACY_L3_SCHEMA_VERSION,
    MASK_ONLY_SPATIAL_GROUPS,
    PREDICTIVE_SPATIAL_GROUPS,
    audit_legacy_feature_contract,
    audit_legacy_shortcuts,
    build_legacy_artifact_manifest,
    verify_legacy_artifact_manifest,
    verify_legacy_snapshot,
)
from pig_behavior.classification_v2.contracts.temporal_tier_contract import (
    LEGACY_TEMPORAL_MODEL_VIEW_SPECS,
)
from pig_behavior.classification_v2.datasets.legacy_unreviewed_development import (
    LEGACY_DEVELOPMENT_SCOPE,
    LEGACY_SOURCE,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.spatial_sequence_export import (
    SPATIAL_FRAME_FEATURES,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

CONTRACT_PATH = Path(
    "configs/classification_v2/legacy_development_input_contract_v1.json"
)
CHECKER_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "classification_v2"
    / "03_image_cache_context"
    / "check_classification_v2_legacy_development_l3.py"
)
CHECKER_SPEC = importlib.util.spec_from_file_location(
    "classification_v2_legacy_development_l3_checker",
    CHECKER_PATH,
)
assert CHECKER_SPEC is not None and CHECKER_SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(CHECKER_SPEC)
CHECKER_SPEC.loader.exec_module(CHECKER)


def test_legacy_l3_feature_contract_freezes_predictive_and_mask_roles() -> None:
    audit = _feature_audit()

    assert audit["valid"] is True
    assert audit["predictive_spatial_groups"] == list(
        PREDICTIVE_SPATIAL_GROUPS
    )
    assert audit["mask_only_spatial_groups"] == list(MASK_ONLY_SPATIAL_GROUPS)
    assert not set(audit["predictive_feature_whitelist"]).intersection(
        audit["mask_only_feature_whitelist"]
    )
    assert audit["unblocked_forbidden_probe_columns"] == []


def test_legacy_l3_feature_contract_rejects_availability_as_predictive_x() -> None:
    contract = _contract()
    contract["feature_selection"]["predictive_spatial_groups"].append(
        "quality_mask"
    )

    audit = audit_legacy_feature_contract(
        contract,
        available_frame_columns=_all_spatial_columns(),
    )

    assert audit["valid"] is False
    assert "predictive_spatial_group_order_mismatch" in audit["errors"]
    assert any(
        error.startswith("predictive_mask_feature_overlap=")
        for error in audit["errors"]
    )


def test_legacy_l3_shortcut_audit_accepts_balanced_fixed_tiers() -> None:
    inputs = _shortcut_inputs()

    audit = audit_legacy_shortcuts(
        **inputs,
        feature_contract_audit=_feature_audit(),
    )

    assert audit["valid"] is True
    assert audit["source_probe"]["model_fit_performed"] is False
    assert audit["source_probe"]["source_count"] == 1
    assert audit["length_shortcut"][
        "maximum_class_distribution_delta_across_lengths"
    ] == 0.0
    assert all(
        report["mask_counts"]["padding_mask"]["true_rows"] == 0
        for report in audit["temporal_view_reports"].values()
    )


def test_legacy_l3_shortcut_audit_rejects_length_and_padding_drift() -> None:
    inputs = _shortcut_inputs()
    selection = inputs["temporal_selection"].copy()
    selection = selection.loc[
        selection["window_length_frames"].eq(
            selection["temporal_unit_key"].str.extract(r"(\d+)$")[0]
            .astype(int)
            .mod(4)
            .map({0: 6, 1: 8, 2: 12, 3: 16})
        )
    ].reset_index(drop=True)
    inputs["temporal_selection"] = selection
    first_view = next(iter(inputs["temporal_views"]))
    slots = inputs["temporal_views"][first_view].copy()
    slots.loc[0, "padding_mask"] = True
    inputs["temporal_views"][first_view] = slots

    audit = audit_legacy_shortcuts(
        **inputs,
        feature_contract_audit=_feature_audit(),
    )

    assert audit["valid"] is False
    assert any(
        error.startswith("temporal_length_label_distribution_drift=")
        for error in audit["errors"]
    )
    assert any("padding_mask_contract" in error for error in audit["errors"])


def test_legacy_l3_artifact_manifest_detects_postfreeze_drift(
    tmp_path: Path,
) -> None:
    table_path = tmp_path / "table.csv"
    tensor_path = tmp_path / "tensor.npy"
    pd.DataFrame({"key": ["a", "b"]}).to_csv(table_path, index=False)
    np.save(tensor_path, np.zeros((2, 4, 4, 3), dtype=np.uint8))
    manifest = build_legacy_artifact_manifest(
        {
            "table": ("fixture", table_path),
            "tensor": ("fixture", tensor_path),
        }
    )

    assert verify_legacy_artifact_manifest(manifest)["valid"] is True

    pd.DataFrame({"key": ["a", "changed"]}).to_csv(table_path, index=False)
    drifted = verify_legacy_artifact_manifest(manifest)
    assert drifted["valid"] is False
    assert "frozen_artifact_sha_mismatch=table" in drifted["errors"]


def test_legacy_l3_snapshot_binds_noncyclic_hashes(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    contract_path = tmp_path / "contract.json"
    feature_path = tmp_path / "feature.json"
    shortcut_path = tmp_path / "shortcut.json"
    for path in (contract_path, feature_path, shortcut_path):
        path.write_text("{}", encoding="utf-8")
    pd.DataFrame({"artifact_name": ["x"]}).to_csv(
        manifest_path,
        index=False,
    )
    snapshot = {
        "schema_version": LEGACY_L3_SCHEMA_VERSION,
        "lineage_scope": LEGACY_DEVELOPMENT_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "model_training_authorized": False,
        "accuracy_f1_comparison_authorized": False,
        "artifact_manifest_sha256": file_sha256(manifest_path),
        "feature_contract_sha256": file_sha256(contract_path),
        "feature_audit_sha256": file_sha256(feature_path),
        "shortcut_audit_sha256": file_sha256(shortcut_path),
        "frozen_contract": {
            "temporal_views": list(LEGACY_TEMPORAL_MODEL_VIEW_SPECS),
            "image_size": 160,
            "predictive_spatial_groups": list(PREDICTIVE_SPATIAL_GROUPS),
            "mask_only_spatial_groups": list(MASK_ONLY_SPATIAL_GROUPS),
        },
    }

    audit = verify_legacy_snapshot(
        snapshot,
        artifact_manifest_path=manifest_path,
        feature_contract_path=contract_path,
        feature_audit_path=feature_path,
        shortcut_audit_path=shortcut_path,
    )
    assert audit["valid"] is True

    shortcut_path.write_text('{"drift": true}', encoding="utf-8")
    drifted = verify_legacy_snapshot(
        snapshot,
        artifact_manifest_path=manifest_path,
        feature_contract_path=contract_path,
        feature_audit_path=feature_path,
        shortcut_audit_path=shortcut_path,
    )
    assert drifted["valid"] is False
    assert "snapshot_hash_mismatch=shortcut_audit_sha256" in drifted["errors"]


def test_legacy_l3_preview_inventory_walks_sharded_directories(
    tmp_path: Path,
) -> None:
    preview = tmp_path / "source" / "video" / "pig" / "frame.jpg"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"preview")

    assert CHECKER._preview_jpg_paths(tmp_path) == [preview]


def test_legacy_l3_reuses_pixel_evidence_only_under_exact_bindings(
    tmp_path: Path,
) -> None:
    paths = {}
    for name in (
        "packed_tensor",
        "cache_manifest",
        "packed_index",
        "packed_audit",
        "image_frames",
        "image_windows",
    ):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("utf-8"))
        paths[name] = path
    artifact_hashes = {
        name: file_sha256(path)
        for name, path in paths.items()
        if name != "packed_tensor"
    }
    pixel_section = {
        "all_pixel_checked_rows": CHECKER.EXPECTED_FRAME_ROWS,
        "invalid_source_cache_tensors": 0,
        "packed_loader_failures": 0,
        "packed_pixel_mismatches": 0,
        "source_media_fallback_reads": 0,
        "image_load_audit": {
            "packed_image_cache_hits": CHECKER.EXPECTED_FRAME_ROWS,
            "disk_image_cache_misses": 0,
            "source_image_loads": 0,
        },
        "errors": [],
        "valid": True,
    }
    evidence = {
        "schema_version": LEGACY_L3_SCHEMA_VERSION,
        "status": "PASS_LEGACY_DEVELOPMENT_L3",
        "valid": True,
        "checker_source_sha256": file_sha256(CHECKER_PATH),
        "packed_tensor_sha256": file_sha256(paths["packed_tensor"]),
        "artifact_hashes": artifact_hashes,
        "packed_pixel_and_loader_audit": pixel_section,
    }
    evidence_path = tmp_path / "prior_l3.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    reused = CHECKER._reuse_packed_pixel_evidence(evidence_path, paths)
    assert reused["valid"] is True
    assert reused["reused"] is True

    paths["packed_index"].write_bytes(b"drift")
    drifted = CHECKER._reuse_packed_pixel_evidence(evidence_path, paths)
    assert drifted["valid"] is False
    assert "reused_pixel_artifact_sha_mismatch=packed_index" in drifted["errors"]


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _all_spatial_columns() -> list[str]:
    return [
        feature
        for features in SPATIAL_FRAME_FEATURES.values()
        for feature in features
    ]


def _feature_audit() -> dict:
    return audit_legacy_feature_contract(
        _contract(),
        available_frame_columns=_all_spatial_columns(),
    )


def _shortcut_inputs() -> dict:
    native_rows = []
    selection_rows = []
    image_rows = []
    enhanced_rows = []
    for index, label in enumerate(VALID_BEHAVIORS):
        unit = f"unit-{index}"
        native_rows.append(
            {
                "temporal_unit_key": unit,
                "source_type": LEGACY_SOURCE,
                "behavior_label": label,
                "lineage_scope": LEGACY_DEVELOPMENT_SCOPE,
                "human_review_complete": False,
            }
        )
        for length in (6, 8, 12, 16):
            selection_rows.append(
                {
                    "window_id": f"{unit}-t{length}",
                    "temporal_unit_key": unit,
                    "window_length_frames": length,
                    "lineage_scope": LEGACY_DEVELOPMENT_SCOPE,
                    "human_review_complete": False,
                }
            )
        image_row = {
            "source_type": LEGACY_SOURCE,
            "lineage_scope": LEGACY_DEVELOPMENT_SCOPE,
            "human_review_complete": False,
        }
        image_row.update({column: True for column in IMAGE_AVAILABILITY_COLUMNS})
        image_rows.append(image_row)
        enhanced_row = {
            "behavior": label,
            "source_type": LEGACY_SOURCE,
            "lineage_scope": LEGACY_DEVELOPMENT_SCOPE,
            "human_review_complete": False,
        }
        enhanced_row.update({column: True for column in AVAILABILITY_COLUMNS})
        enhanced_row["social_missing_mask"] = bool(index == 0)
        enhanced_rows.append(enhanced_row)

    temporal_views = {}
    for view_name, spec in LEGACY_TEMPORAL_MODEL_VIEW_SPECS.items():
        length = int(spec["sequence_length"])
        rows = []
        for unit_index in range(len(VALID_BEHAVIORS)):
            item = f"{view_name}|unit-{unit_index}"
            for slot in range(length):
                rows.append(
                    {
                        "temporal_view_name": view_name,
                        "view_item_id": item,
                        "source_type": LEGACY_SOURCE,
                        "slot_index": slot,
                        "declared_sequence_length": length,
                        "length_mask": True,
                        "observed_mask": True,
                        "timing_valid_mask": True,
                        "padding_mask": False,
                        "lineage_scope": LEGACY_DEVELOPMENT_SCOPE,
                        "human_review_complete": False,
                    }
                )
        temporal_views[view_name] = pd.DataFrame(rows)
    return {
        "native_units": pd.DataFrame(native_rows),
        "temporal_selection": pd.DataFrame(selection_rows),
        "temporal_views": temporal_views,
        "image_frames": pd.DataFrame(image_rows),
        "enhanced_frames": pd.DataFrame(enhanced_rows),
    }
