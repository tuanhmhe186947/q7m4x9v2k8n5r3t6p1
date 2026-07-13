from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_temporal_loader_audit()
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
