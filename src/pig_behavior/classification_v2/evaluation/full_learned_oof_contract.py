"""Readiness gate for full learned native-OOF multimodal evaluation.

This module does not train the model. It verifies that the audited tensors,
native temporal folds, image context manifests, and model/input contracts are
present and mutually aligned before a long paper-facing learned evaluation is
allowed to be registered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class FullLearnedOofContractConfig:
    """Paths for checking the full learned OOF evaluation contract."""

    contract_json: Path = Path(
        "configs/classification_v2/full_learned_oof_contract_v1.json"
    )
    output_json: Path = Path(
        "outputs/classification_v2/model_design/"
        "full_learned_oof_contract_audit.json"
    )


def check_full_learned_oof_contract(config: FullLearnedOofContractConfig) -> dict[str, Any]:
    """Validate data readiness and claim boundaries for full learned OOF training."""

    errors: list[str] = []
    warnings: list[str] = []
    contract = _read_json(config.contract_json, errors, "contract")
    inputs = {name: Path(path) for name, path in contract.get("required_inputs", {}).items()}
    input_report = {name: _path_report(path) for name, path in inputs.items()}
    for name, report in input_report.items():
        if not report["exists"]:
            errors.append(f"missing_required_input={name}:{report['path']}")

    alignment_report: dict[str, Any] = {}
    if not errors:
        alignment_report = _build_alignment_report(inputs, errors, warnings)
    required_record = Path(str(contract.get("required_record", "")))
    record_report = _path_report(required_record)
    paper_ready = bool(record_report["exists"] and not errors)
    if not record_report["exists"]:
        warnings.append(f"full_oof_record_missing={required_record}")

    audit = {
        "schema_version": "classification_v2_full_learned_oof_contract_audit_v1",
        "contract_json": str(config.contract_json),
        "paper_claim_level": contract.get("paper_claim_level"),
        "external_generalization_claim": bool(contract.get("external_generalization_claim")),
        "statistical_unit": contract.get("statistical_unit"),
        "split_policy": contract.get("split_policy"),
        "input_report": input_report,
        "alignment_report": alignment_report,
        "required_model_inputs": contract.get("required_model_inputs", []),
        "forbidden_model_inputs": contract.get("forbidden_model_inputs", []),
        "required_output_schema": contract.get("required_output_schema", {}),
        "required_record": record_report,
        "paper_ready": paper_ready,
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def _build_alignment_report(
    inputs: dict[str, Path],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    """Check row alignment across train-ready windows, tensors, folds, and context manifests."""

    y = pd.read_csv(inputs["y_behavior_csv"]).iloc[:, 0].fillna("").astype(str)
    train_mask = _read_bool(inputs["train_mask_csv"])
    split = pd.read_csv(inputs["split_manifest_csv"], low_memory=False)
    image_windows = pd.read_csv(inputs["image_window_context_manifest_csv"], low_memory=False)
    image_frames = pd.read_csv(inputs["image_frame_context_manifest_csv"], low_memory=False)
    sequence = pd.read_csv(
        inputs["sequence_manifest_csv"],
        usecols=["window_id", "temporal_unit_keys_window", "num_temporal_units_window"],
        low_memory=False,
    )
    folds = pd.read_csv(
        inputs["native_oof_fold_manifest_csv"],
        usecols=["temporal_unit_key", "oof_fold_id", "native_unit_valid_for_main_eval"],
        low_memory=False,
    )
    interaction = pd.read_csv(inputs["interaction_context_manifest_csv"], low_memory=False)
    arrays = np.load(inputs["x_spatial_sequences_npz"])

    expected_rows = int(len(y))
    row_counts = {
        "y_behavior": expected_rows,
        "train_mask": int(len(train_mask)),
        "split_manifest": int(len(split)),
        "image_window_context_manifest": int(len(image_windows)),
        "interaction_context_manifest": int(len(interaction)),
    }
    row_counts.update(
        {
            f"spatial_array_{name}": int(value.shape[0])
            for name, value in arrays.items()
        }
    )
    mismatched = {name: count for name, count in row_counts.items() if count != expected_rows}
    if mismatched:
        errors.append(f"row_count_mismatch={mismatched}")

    sequence_join = split[["window_id"]].merge(
        sequence,
        on="window_id",
        how="left",
        validate="one_to_one",
    )
    single_unit = pd.to_numeric(sequence_join["num_temporal_units_window"], errors="coerce").eq(1)
    native_join = sequence_join.loc[single_unit].rename(
        columns={"temporal_unit_keys_window": "temporal_unit_key"}
    )
    native_join = native_join.merge(folds, on="temporal_unit_key", how="left")
    native_valid = _to_bool(
        native_join["native_unit_valid_for_main_eval"]
    ) & native_join["oof_fold_id"].notna()
    if int(native_valid.sum()) == 0:
        errors.append("native_oof_valid_window_rows_zero")
    complete_image = _to_bool(image_windows["window_image_context_complete"])
    if int(complete_image.sum()) == 0:
        errors.append("complete_image_context_rows_zero")
    if "frame_uid" not in image_frames.columns:
        errors.append("image_frame_context_missing_frame_uid")
    scene_column = (
        "scene_frame_uid"
        if "scene_frame_uid" in image_frames.columns
        else "frame_uid"
    )

    return {
        "row_counts": row_counts,
        "row_count_mismatches": mismatched,
        "train_ready_rows": expected_rows,
        "train_mask_true_rows": int(train_mask.sum()),
        "single_native_unit_window_rows": int(single_unit.sum()),
        "native_oof_valid_window_rows": int(native_valid.sum()),
        "native_oof_fold_count": int(native_join.loc[native_valid, "oof_fold_id"].nunique()),
        "complete_image_context_rows": int(complete_image.sum()),
        "image_frame_context_rows": int(len(image_frames)),
        "image_frame_context_unique_frames": int(
            image_frames[scene_column].nunique()
        )
        if scene_column in image_frames.columns
        else 0,
        "image_frame_context_unique_objects": int(
            image_frames["frame_uid"].nunique()
        )
        if "frame_uid" in image_frames.columns
        else 0,
        "interaction_context_ready_rows": int(
            _to_bool(interaction["scene_partner_context_ready"]).sum()
        ),
    }


def _path_report(path: Path) -> dict[str, Any]:
    """Return basic existence and size metadata for an evidence path."""

    exists = path.exists()
    report: dict[str, Any] = {"path": str(path), "exists": exists}
    if exists:
        report["size_bytes"] = int(path.stat().st_size)
    return report


def _read_json(path: Path, errors: list[str], name: str) -> dict[str, Any]:
    """Read JSON and retain missing/invalid state in the audit errors."""

    if not path.exists():
        errors.append(f"missing_{name}={path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_bool(path: Path) -> pd.Series:
    """Read a one-column CSV mask and normalize truthy strings."""

    return _to_bool(pd.read_csv(path).iloc[:, 0])


def _to_bool(series: pd.Series) -> pd.Series:
    """Convert bool/string flags to strict booleans for audit counts."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})
