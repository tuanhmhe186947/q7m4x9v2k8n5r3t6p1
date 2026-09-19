"""Source/domain control views for classification_v2.

The train-ready dataset intentionally preserves both legacy recovered bursts and
CVAT tracking intervals. Those sources differ in annotation cadence, crop/video
origin, and spatial statistics, so a model can learn source/domain shortcuts.
This module builds an auditable matched view without rewriting X/y artifacts:
every original window remains in the selection manifest, while a deterministic
``domain_control_keep`` flag marks the source-balanced subset for controlled
experiments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.contracts.model_io import forbidden_x_columns, read_csv_schema


@dataclass(slots=True)
class SourceDomainControlResult:
    """Container returned by source-domain view builders."""

    selection_manifest: pd.DataFrame
    audit: dict[str, Any]


def build_source_domain_control_view(
    split_manifest: pd.DataFrame,
    y_behavior: pd.DataFrame,
    train_mask: pd.Series,
    *,
    x_columns: list[str],
    forbidden_patterns: list[str] | tuple[str, ...] | None = None,
) -> SourceDomainControlResult:
    """Build a deterministic source-balanced view over existing train-ready rows.

    Balancing is performed independently inside each ``split`` x behavior label
    x ``window_length_frames`` stratum. The smallest source count in a stratum
    defines the matched quota. Rows outside the quota are not deleted; they are
    retained with an explicit exclusion reason so later reports can quantify
    exactly what was not used by the controlled view.
    """

    work = _validated_work_frame(split_manifest, y_behavior, train_mask)
    source_labels = sorted(work.loc[work["domain_control_eligible"], "source_type"].dropna().astype(str).unique())
    work["domain_control_stratum_key"] = _stratum_key(work)
    work["domain_control_keep"] = False
    work["domain_control_reason"] = "ineligible_train_mask_or_split"
    work["domain_control_source_rank"] = pd.NA
    work["domain_control_matched_quota"] = 0

    # Rank within source after deterministic sorting. This creates a stable
    # matched subset and avoids random row loss when the view is regenerated.
    eligible = work[work["domain_control_eligible"]].copy()
    if not eligible.empty:
        eligible["_source_row_index"] = eligible.index
        eligible = eligible.sort_values(
            ["split", "behavior_label", "window_length_frames", "source_type", "window_id"]
        )
        eligible["domain_control_source_rank"] = (
            eligible.groupby(["domain_control_stratum_key", "source_type"]).cumcount() + 1
        )
        quotas = _matched_quotas(eligible, source_labels).set_index("domain_control_stratum_key")[
            "domain_control_matched_quota"
        ]
        eligible["domain_control_matched_quota"] = (
            eligible["domain_control_stratum_key"].map(quotas).fillna(0).astype(int)
        )
        eligible["domain_control_keep"] = (
            eligible["domain_control_matched_quota"].gt(0)
            & eligible["domain_control_source_rank"].le(eligible["domain_control_matched_quota"])
        )
        eligible["domain_control_reason"] = "source_matched_keep"
        eligible.loc[
            eligible["domain_control_matched_quota"].eq(0),
            "domain_control_reason",
        ] = "stratum_missing_one_or_more_sources"
        eligible.loc[
            eligible["domain_control_matched_quota"].gt(0) & ~eligible["domain_control_keep"],
            "domain_control_reason",
        ] = "above_source_matched_quota"
        update_cols = [
            "domain_control_keep",
            "domain_control_reason",
            "domain_control_source_rank",
            "domain_control_matched_quota",
        ]
        # Merge resets the DataFrame index, so update through the preserved
        # original row index rather than relying on positional coincidence.
        work.loc[eligible["_source_row_index"], update_cols] = eligible[update_cols].to_numpy()

    ordered = _ordered_selection_columns(work)
    audit = audit_source_domain_control_view(
        ordered,
        x_columns=x_columns,
        forbidden_patterns=forbidden_patterns,
        source_labels=source_labels,
    )
    return SourceDomainControlResult(selection_manifest=ordered, audit=audit)


def build_source_domain_control_from_paths(config: dict[str, Any]) -> SourceDomainControlResult:
    """Load declared artifacts and build the source-domain selection manifest."""

    root = Path(config["train_ready_root"])
    split_manifest = pd.read_csv(root / config.get("split_manifest", "split_manifest.csv"), low_memory=False)
    y_behavior = pd.read_csv(root / config.get("y_behavior", "y_behavior.csv"), low_memory=False)
    train_mask = _read_bool(root / config.get("train_mask", "train_mask.csv"))
    x_columns = read_csv_schema(root / config.get("tabular_x", "X_window_features.csv"))
    return build_source_domain_control_view(
        split_manifest,
        y_behavior,
        train_mask,
        x_columns=x_columns,
        forbidden_patterns=config.get("forbidden_x_patterns"),
    )


def audit_source_domain_control_view(
    selection_manifest: pd.DataFrame,
    *,
    x_columns: list[str],
    forbidden_patterns: list[str] | tuple[str, ...] | None,
    source_labels: list[str],
) -> dict[str, Any]:
    """Validate source balance, row preservation, and source leakage guards."""

    errors: list[str] = []
    duplicate_windows = int(selection_manifest["window_id"].duplicated().sum())
    forbidden = forbidden_x_columns(x_columns, forbidden_patterns)
    if duplicate_windows:
        errors.append(f"duplicate_window_id={duplicate_windows}")
    if not source_labels:
        errors.append("missing_source_labels")
    if forbidden:
        errors.append(f"forbidden_x_columns={forbidden}")

    kept = selection_manifest[_to_bool(selection_manifest["domain_control_keep"])]
    balance_after = _source_counts(kept, ["split", "behavior_label", "window_length_frames"])
    imbalanced_after = _imbalanced_strata(balance_after, source_labels)
    if imbalanced_after:
        errors.append(f"matched_view_imbalanced_strata={len(imbalanced_after)}")

    return {
        "rows": int(len(selection_manifest)),
        "eligible_rows": int(_to_bool(selection_manifest["domain_control_eligible"]).sum()),
        "kept_rows": int(len(kept)),
        "excluded_rows": int(len(selection_manifest) - len(kept)),
        "source_labels": source_labels,
        "duplicate_window_id": duplicate_windows,
        "x_column_count": int(len(x_columns)),
        "forbidden_x_columns": forbidden,
        "source_counts_before": selection_manifest["source_type"].value_counts(dropna=False).to_dict(),
        "source_counts_kept": kept["source_type"].value_counts(dropna=False).to_dict(),
        "reason_counts": selection_manifest["domain_control_reason"].value_counts(dropna=False).to_dict(),
        "balanced_strata_after_count": int(len(balance_after) - len(imbalanced_after)),
        "imbalanced_strata_after_count": int(len(imbalanced_after)),
        "imbalanced_strata_after_examples": imbalanced_after[:20],
        "warnings": [
            "Use this matched view for source/domain-control experiments.",
            "It is not a replacement for the full train-ready dataset.",
            "Report high source shortcut audits under video/session-safe validation.",
        ],
        "errors": errors,
        "valid": not errors,
    }


def write_source_domain_control_outputs(
    result: SourceDomainControlResult,
    *,
    output_dir: Path,
) -> dict[str, str]:
    """Write the selection manifest and audit JSON produced by S5."""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "source_domain_selection_manifest.csv"
    audit_path = output_dir / "source_domain_control_audit.json"
    result.selection_manifest.to_csv(manifest_path, index=False)
    audit_path.write_text(json.dumps(result.audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"selection_manifest": str(manifest_path), "audit_json": str(audit_path)}


def _validated_work_frame(
    split_manifest: pd.DataFrame,
    y_behavior: pd.DataFrame,
    train_mask: pd.Series,
) -> pd.DataFrame:
    required = [
        "window_id",
        "split",
        "source_type",
        "window_length_frames",
        "window_valid_for_main_train",
    ]
    missing = [col for col in required if col not in split_manifest.columns]
    if missing:
        raise ValueError(f"split_manifest missing columns: {missing}")
    if "behavior_window_label" in split_manifest.columns:
        behavior = split_manifest["behavior_window_label"]
    elif "behavior_window_label" in y_behavior.columns:
        behavior = y_behavior["behavior_window_label"]
    else:
        behavior = y_behavior.iloc[:, 0]
    if len(split_manifest) != len(y_behavior) or len(split_manifest) != len(train_mask):
        raise ValueError(
            f"row mismatch split={len(split_manifest)} y={len(y_behavior)} train_mask={len(train_mask)}"
        )

    work = split_manifest[
        [
            "window_id",
            "split",
            "source_type",
            "dataset_id",
            "video_key",
            "object_track_key",
            "pig_id",
            "track_id",
            "window_length_frames",
            "window_start_frame",
            "window_end_frame",
            "sequence_label_status",
            "window_valid_for_main_train",
            "window_sample_weight",
        ]
    ].copy()
    work["behavior_label"] = behavior.astype(str)
    work["train_mask"] = _to_bool(train_mask)
    work["window_valid_for_main_train"] = _to_bool(work["window_valid_for_main_train"])
    work["domain_control_eligible"] = work["train_mask"] & work["window_valid_for_main_train"]
    return work


def _matched_quotas(eligible: pd.DataFrame, source_labels: list[str]) -> pd.DataFrame:
    counts = (
        eligible.groupby(["domain_control_stratum_key", "source_type"])["window_id"]
        .count()
        .rename("rows")
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for stratum, group in counts.groupby("domain_control_stratum_key"):
        present = {str(source): int(rows) for source, rows in zip(group["source_type"], group["rows"], strict=False)}
        if not source_labels or any(source not in present for source in source_labels):
            quota = 0
        else:
            quota = min(present[source] for source in source_labels)
        rows.append({"domain_control_stratum_key": stratum, "domain_control_matched_quota": int(quota)})
    return pd.DataFrame(rows)


def _ordered_selection_columns(work: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "window_id",
        "split",
        "source_type",
        "behavior_label",
        "window_length_frames",
        "window_start_frame",
        "window_end_frame",
        "dataset_id",
        "video_key",
        "object_track_key",
        "pig_id",
        "track_id",
        "sequence_label_status",
        "window_valid_for_main_train",
        "window_sample_weight",
        "train_mask",
        "domain_control_eligible",
        "domain_control_keep",
        "domain_control_reason",
        "domain_control_stratum_key",
        "domain_control_source_rank",
        "domain_control_matched_quota",
    ]
    # Preserve the original train-ready row order so snapshot key-alignment can
    # prove that this control manifest is a row-wise mask over the same windows.
    return work[columns].reset_index(drop=True)


def _source_counts(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*group_cols, "source_type", "rows"])
    return frame.groupby([*group_cols, "source_type"])["window_id"].count().rename("rows").reset_index()


def _imbalanced_strata(counts: pd.DataFrame, source_labels: list[str]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    if counts.empty:
        return examples
    for keys, group in counts.groupby(["split", "behavior_label", "window_length_frames"]):
        present = {str(source): int(rows) for source, rows in zip(group["source_type"], group["rows"], strict=False)}
        values = [present.get(source, 0) for source in source_labels]
        if len(set(values)) > 1:
            examples.append(
                {
                    "split": str(keys[0]),
                    "behavior_label": str(keys[1]),
                    "window_length_frames": int(keys[2]),
                    "source_counts": present,
                }
            )
    return examples


def _stratum_key(work: pd.DataFrame) -> pd.Series:
    return (
        work["split"].astype(str)
        + "|label="
        + work["behavior_label"].astype(str)
        + "|len="
        + work["window_length_frames"].astype(str)
    )


def _read_bool(path: Path) -> pd.Series:
    return _to_bool(pd.read_csv(path).iloc[:, 0])


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})
