from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.contracts.temporal_tier_contract import (
    LEGACY_TEMPORAL_MODEL_VIEW_SPECS,
)
from pig_behavior.classification_v2.training.temporal_view_loader import (
    load_temporal_view_tensors,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit fixed-six timing loading with a synthetic manifest."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_design/"
            "temporal_view_loader_audit.json"
        ),
    )
    parser.add_argument(
        "--legacy-tier-root",
        type=Path,
        default=None,
        help="Optionally audit all eight views in a bounded real-data packet.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_temporal_loader_audit()
    if args.legacy_tier_root is not None:
        real_packet = run_legacy_tier_loader_audit(args.legacy_tier_root)
        result["legacy_tier_real_packet"] = real_packet
        result["errors"].extend(real_packet["errors"])
        result["valid"] = not result["errors"]
    if not args.dry_run:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if not result["valid"]:
        raise SystemExit(1)


def run_temporal_loader_audit() -> dict[str, Any]:
    """Exercise order, missing-row preservation, and corruption rejection."""

    errors: list[str] = []
    corruption_rejected = False
    with tempfile.TemporaryDirectory(prefix="c2v2_time_loader_") as temp_dir:
        path = Path(temp_dir) / "fixed6_observed_time_manifest.csv"
        frame = _manifest(["window-0", "window-2"])
        frame.to_csv(path, index=False)
        tensors = load_temporal_view_tensors(
            path,
            expected_window_ids=["window-0", "window-1", "window-2"],
            selected_mask=np.array([True, False, True]),
            expected_view_name="fixed6_observed_time",
        )
        if tensors.time_delta.shape != (3, 6):
            errors.append(f"unexpected_time_delta_shape={tensors.time_delta.shape}")
        if not np.isnan(tensors.time_delta[1]).all():
            errors.append("unselected_window_not_nan_masked")
        corrupted = frame.copy()
        corrupted.loc[1, "slot_key"] = corrupted.loc[0, "slot_key"]
        corrupted.to_csv(path, index=False)
        try:
            load_temporal_view_tensors(
                path,
                expected_window_ids=["window-0", "window-1", "window-2"],
                selected_mask=np.array([True, False, True]),
                expected_view_name="fixed6_observed_time",
            )
        except ValueError:
            corruption_rejected = True
        if not corruption_rejected:
            errors.append("duplicate_slot_key_not_rejected")
    return {
        "schema_version": "classification_v2.temporal_loader_audit.v1",
        "window_universe_rows": 3,
        "selected_window_rows": 2,
        "expected_tensor_shape": [3, 6],
        "unselected_rows_preserved": 1,
        "duplicate_slot_key_rejected": corruption_rejected,
        "optimizer_steps": 0,
        "full_dataset_read": False,
        "errors": errors,
        "valid": not errors,
    }


def run_legacy_tier_loader_audit(root: Path) -> dict[str, Any]:
    """Load every bounded real-data tier through the strict tensor loader."""

    selection_path = root / "temporal_tier_selection_manifest.csv"
    errors: list[str] = []
    views: dict[str, dict[str, Any]] = {}
    try:
        selection = pd.read_csv(selection_path, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return {
            "root": str(root),
            "selection_path": str(selection_path),
            "views": views,
            "errors": [f"selection_manifest_unreadable={exc}"],
            "valid": False,
        }
    if "window_id" not in selection.columns:
        errors.append("selection_manifest_missing_window_id")
        window_ids = pd.Series(dtype="object")
    else:
        window_ids = selection["window_id"]

    for view_name, spec in LEGACY_TEMPORAL_MODEL_VIEW_SPECS.items():
        selection_column = str(spec["selection_column"])
        sequence_length = int(spec["sequence_length"])
        manifest_path = root / str(spec["slot_manifest_filename"])
        if selection_column not in selection.columns:
            errors.append(f"{view_name}:missing_selection_column")
            continue
        try:
            selected = _strict_bool_mask(
                selection[selection_column],
                selection_column,
            )
            tensors = load_temporal_view_tensors(
                manifest_path,
                expected_window_ids=window_ids,
                selected_mask=selected,
                expected_view_name=view_name,
                expected_sequence_length=sequence_length,
            )
            expected_shape = (len(selection), sequence_length)
            shape_valid = bool(
                tensors.time_delta.shape == expected_shape
                and tensors.timing_valid_mask.shape == expected_shape
                and tensors.observed_mask.shape == expected_shape
            )
            selected_nonempty = bool(selected.any())
            selected_finite = bool(
                np.isfinite(tensors.time_delta[selected]).all()
            )
            unselected_nan = bool(
                np.isnan(tensors.time_delta[~selected]).all()
            )
            selected_timing_valid = bool(
                tensors.timing_valid_mask[selected].all()
            )
            selected_observed = bool(
                tensors.observed_mask[selected].all()
            )
            unselected_masks_clear = bool(
                not tensors.timing_valid_mask[~selected].any()
                and not tensors.observed_mask[~selected].any()
            )
            if not (
                shape_valid
                and selected_nonempty
                and selected_finite
                and unselected_nan
                and selected_timing_valid
                and selected_observed
                and unselected_masks_clear
            ):
                errors.append(f"{view_name}:tensor_mask_contract_failed")
            views[view_name] = {
                "sequence_length": sequence_length,
                "selected_windows": int(selected.sum()),
                "tensor_shape": list(tensors.time_delta.shape),
                "shape_valid": shape_valid,
                "selected_nonempty": selected_nonempty,
                "selected_time_delta_finite": selected_finite,
                "unselected_time_delta_nan": unselected_nan,
                "selected_timing_valid": selected_timing_valid,
                "selected_observed": selected_observed,
                "unselected_masks_clear": unselected_masks_clear,
                "manifest_sha256": tensors.audit["sha256"],
            }
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{view_name}:{type(exc).__name__}:{exc}")

    return {
        "schema_version": "classification_v2.legacy_tier_loader_audit.v1",
        "root": str(root),
        "selection_path": str(selection_path),
        "window_universe_rows": int(len(selection)),
        "expected_view_count": len(LEGACY_TEMPORAL_MODEL_VIEW_SPECS),
        "loaded_view_count": len(views),
        "human_review_complete": False,
        "optimizer_steps": 0,
        "views": views,
        "errors": errors,
        "valid": not errors,
    }


def _strict_bool_mask(series: pd.Series, name: str) -> np.ndarray:
    """Parse CSV booleans without treating arbitrary text as true."""

    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise ValueError(f"{name} contains null values")
        return series.to_numpy(dtype=np.bool_)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f"}
    invalid = ~normalized.isin(truthy | falsy)
    if invalid.any():
        raise ValueError(f"{name} contains invalid boolean values")
    return normalized.isin(truthy).to_numpy(dtype=np.bool_)


def _manifest(window_ids: list[str]) -> pd.DataFrame:
    rows = []
    for item_order, window_id in enumerate(window_ids):
        view_item_id = f"fixed6|{window_id}"
        for slot_index in range(6):
            rows.append(
                {
                    "temporal_view_name": "fixed6_observed_time",
                    "view_item_id": view_item_id,
                    "parent_window_id": window_id,
                    "item_order": item_order,
                    "slot_index": slot_index,
                    "slot_key": f"{view_item_id}|slot={slot_index}",
                    "declared_sequence_length": 6,
                    "time_delta": 0.0 if slot_index == 0 else 0.2,
                    "length_mask": True,
                    "observed_mask": True,
                    "timing_valid_mask": True,
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
