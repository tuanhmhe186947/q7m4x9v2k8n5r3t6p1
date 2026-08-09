"""Materialize inner-only S1 eligibility, weights, and coverage authorities.

This is a derived-artifact command.  It consumes the authoritative effective
window index and grouped role metadata, never feature/RGB payloads.  The
resulting root is immutable: a retry must use a new root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.datasets.fold_event_weights import (
    build_fold_event_weight_manifest,
)
from pig_behavior.classification_v2.datasets.primary_temporal_eligibility import (
    PRIMARY_S1_ALLOWED_ROLES,
    build_primary_s1_validation_native_population,
    build_primary_s1_view_role_overlay,
    load_primary_s1_temporal_eligibility,
)

_VIEW_TYPES = (
    "T6_contiguous",
    "T8_contiguous",
    "T12_contiguous",
    "T16_contiguous",
)
_SELECTION_COLUMN = "primary_s1_keep"


def parse_args() -> argparse.Namespace:
    """Parse the immutable derived-artifact contract."""

    parser = argparse.ArgumentParser(
        description="Materialize inner-only S1 primary eligibility authorities."
    )
    parser.add_argument("--effective-window-index-csv", type=Path, required=True)
    parser.add_argument("--native-role-authority-csv", type=Path, required=True)
    parser.add_argument("--effective-window-index-sha256", required=True)
    parser.add_argument("--native-role-authority-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold-id", default="FOLD_3")
    return parser.parse_args()


def main() -> None:
    """Write only inner-role derived artifacts and their audit."""

    args = parse_args()
    _require_inputs(args.effective_window_index_csv, args.native_role_authority_csv)
    _require_empty_root(args.output_root)

    result = load_primary_s1_temporal_eligibility(
        args.effective_window_index_csv,
        args.native_role_authority_csv,
        fold_id=args.fold_id,
        requested_roles=PRIMARY_S1_ALLOWED_ROLES,
        expected_window_index_sha256=args.effective_window_index_sha256,
        expected_native_role_authority_sha256=args.native_role_authority_sha256,
    )
    inner = _stage1_inner_only(result.windows)
    role_authority = pd.read_csv(args.native_role_authority_csv, low_memory=False)
    args.output_root.mkdir(parents=True, exist_ok=False)

    eligibility_path = args.output_root / "fold3_primary_inner_eligibility.csv"
    _write_csv_atomic(eligibility_path, inner)
    views: dict[str, dict[str, Any]] = {}
    view_native_keys: list[set[str]] = []
    for view_type in _VIEW_TYPES:
        summary, native_keys = _materialize_view(
            inner,
            role_authority,
            fold_id=args.fold_id,
            view_type=view_type,
            output_root=args.output_root,
        )
        views[view_type] = summary
        view_native_keys.append(native_keys)

    common_keys = set.intersection(*view_native_keys)
    common = _common_inner_native_population(
        role_authority,
        fold_id=args.fold_id,
        keys=common_keys,
    )
    common_path = args.output_root / "fold3_common_t6_t8_t12_t16_native_units.csv"
    _write_csv_atomic(common_path, common)
    audit = {
        "schema_version": "classification_v2.s1_primary_derived_artifacts.v1",
        "fold_id": args.fold_id,
        "allowed_roles": sorted(PRIMARY_S1_ALLOWED_ROLES),
        "outer_payloads_opened": False,
        "inputs": {
            "effective_window_index": _artifact(args.effective_window_index_csv),
            "native_role_authority": _artifact(args.native_role_authority_csv),
        },
        "primary_eligibility": {
            "path": str(eligibility_path),
            "sha256": _sha256(eligibility_path),
            "rows": int(len(inner)),
            "materialization_invariants": {
                "row_index_preserved": result.audit["window_row_index_preserved"],
                "retained_window_feature_reuse": result.audit[
                    "retained_window_feature_reuse"
                ],
                "outer_rows_written": 0,
                "stage1_view_types": list(_VIEW_TYPES),
            },
        },
        "views": views,
        "common_t6_t8_t12_t16": {
            "path": str(common_path),
            "sha256": _sha256(common_path),
            "native_units": int(len(common)),
            "class_support": _class_support(common),
        },
    }
    audit_path = args.output_root / "s1_primary_derived_artifacts_audit.json"
    _write_json_atomic(audit_path, audit)
    print(json.dumps({**audit, "audit_path": str(audit_path)}, indent=2))


def _materialize_view(
    inner: pd.DataFrame,
    role_authority: pd.DataFrame,
    *,
    fold_id: str,
    view_type: str,
    output_root: Path,
) -> tuple[dict[str, Any], set[str]]:
    view = inner.loc[inner["view_type"].eq(view_type)].copy()
    if view.empty:
        raise ValueError(f"primary S1 inner population lacks {view_type}")
    overlay, overlay_audit = build_primary_s1_view_role_overlay(
        view,
        role_authority,
        fold_id=fold_id,
        view_type=view_type,
    )
    overlay = overlay.loc[overlay["role"].isin(PRIMARY_S1_ALLOWED_ROLES)].copy()
    weights_input = view.copy()
    weights_input["window_valid_for_main_train"] = weights_input[
        "primary_s1_eligible"
    ]
    weights_input["window_sample_weight"] = weights_input[
        "primary_s1_effective_sample_weight"
    ]
    selection = weights_input[["window_id", "primary_s1_eligible"]].rename(
        columns={"primary_s1_eligible": _SELECTION_COLUMN}
    )
    tables = build_fold_event_weight_manifest(
        weights_input,
        overlay,
        selection=selection,
        selection_col=_SELECTION_COLUMN,
    )
    validation_population, validation_audit = (
        build_primary_s1_validation_native_population(
            view,
            overlay,
            fold_id=fold_id,
        )
    )
    label = view_type.split("_", maxsplit=1)[0].lower()
    weights_path = output_root / f"fold3_{label}_event_weights.csv"
    class_path = output_root / f"fold3_{label}_class_summary.csv"
    event_path = output_root / f"fold3_{label}_event_summary.csv"
    validation_path = output_root / f"fold3_{label}_validation_native_units.csv"
    input_path = output_root / f"fold3_{label}_primary_windows.csv"
    roles_path = output_root / f"fold3_{label}_primary_roles.csv"
    selection_path = output_root / f"fold3_{label}_primary_selection.csv"
    _write_csv_atomic(input_path, weights_input)
    _write_csv_atomic(roles_path, overlay)
    _write_csv_atomic(selection_path, selection)
    _write_csv_atomic(weights_path, tables.weights)
    _write_csv_atomic(class_path, tables.class_summary)
    _write_csv_atomic(event_path, tables.event_summary)
    _write_csv_atomic(validation_path, validation_population)
    _assert_zero_mixed_training_rows(view, tables.weights)
    eligible = _strict_bool(view["primary_s1_eligible"])
    native_keys = _native_keys(view.loc[eligible, "temporal_unit_keys_json"])
    train_mask = eligible & view["primary_s1_role"].eq("train")
    validation_mask = eligible & view["primary_s1_role"].eq("validation")
    return (
        {
            "view_type": view_type,
            "eligible_train_windows": int(train_mask.sum()),
            "eligible_validation_windows": int(validation_mask.sum()),
            "eligible_inner_native_units": int(len(native_keys)),
            "mixed_label_training_rows": 0,
            "event_weight_train_only": "PASS",
            "view_role_overlay": overlay_audit,
            "event_weight_audit": tables.audit,
            "validation_population": {
                "path": str(validation_path),
                "sha256": _sha256(validation_path),
                **validation_audit,
            },
            "event_weights": {
                "path": str(weights_path),
                "sha256": _sha256(weights_path),
                "window_input_path": str(input_path),
                "window_input_sha256": _sha256(input_path),
                "roles_path": str(roles_path),
                "roles_sha256": _sha256(roles_path),
                "selection_path": str(selection_path),
                "selection_sha256": _sha256(selection_path),
                "selection_column": _SELECTION_COLUMN,
                "class_summary_path": str(class_path),
                "class_summary_sha256": _sha256(class_path),
                "event_summary_path": str(event_path),
                "event_summary_sha256": _sha256(event_path),
            },
        },
        native_keys,
    )


def _stage1_inner_only(windows: pd.DataFrame) -> pd.DataFrame:
    """Discard non-inner and non-Stage-1 rows before any artifact is written."""

    inner = windows.loc[
        windows["primary_s1_role"].isin(PRIMARY_S1_ALLOWED_ROLES)
        & windows["view_type"].isin(_VIEW_TYPES)
    ].copy()
    if inner.empty:
        raise ValueError("primary S1 eligibility produced no inner rows")
    return inner.reset_index(drop=True)


def _common_inner_native_population(
    roles: pd.DataFrame,
    *,
    fold_id: str,
    keys: set[str],
) -> pd.DataFrame:
    common = roles.loc[
        roles["outer_fold_id"].astype(str).eq(fold_id)
        & roles["role"].isin(PRIMARY_S1_ALLOWED_ROLES)
        & roles["temporal_unit_key"].astype(str).isin(keys),
        ["temporal_unit_key", "role", "behavior_label"],
    ].copy()
    if len(common) != len(keys):
        raise ValueError("common cohort native role binding is incomplete")
    return common.sort_values("temporal_unit_key", kind="mergesort").reset_index(
        drop=True
    )


def _assert_zero_mixed_training_rows(
    view: pd.DataFrame,
    weights: pd.DataFrame,
) -> None:
    mixed_ids = set(
        view.loc[
            view["primary_s1_eligibility_status"].eq("MIXED_LABEL")
            & view["primary_s1_role"].eq("train"),
            "window_id",
        ].astype(str)
    )
    if not mixed_ids:
        return
    selected = weights.loc[weights["window_id"].astype(str).isin(mixed_ids)]
    if selected["window_valid_for_fold_training_weight"].any():
        raise ValueError("mixed-label row reached fold-local training weights")


def _native_keys(values: pd.Series) -> set[str]:
    keys: set[str] = set()
    for value in values:
        parsed = json.loads(str(value))
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("eligible primary window lacks native-unit keys")
        keys.update(str(key) for key in parsed)
    return keys


def _class_support(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in frame["behavior_label"].value_counts().sort_index().items()
    }


def _strict_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise ValueError("primary eligibility contains invalid boolean")
    return normalized.isin({"true", "1"})


def _require_inputs(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing S1 primary authority inputs={missing}")


def _require_empty_root(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"immutable S1 output root already exists={path}")


def _artifact(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
