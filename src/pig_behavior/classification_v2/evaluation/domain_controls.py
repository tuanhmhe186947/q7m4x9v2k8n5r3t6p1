"""Source/domain control views that preserve every classification_v2 window."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_source_matched_views(windows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mark source and matched-length controls without dropping or relabeling rows."""

    required = {
        "window_id",
        "source_type",
        "behavior_window_label",
        "window_length_frames",
        "window_valid_for_main_train",
    }
    missing = sorted(required.difference(windows.columns))
    if missing:
        raise ValueError(f"source-matched view input missing columns: {missing}")
    if windows["window_id"].duplicated().any():
        raise ValueError("duplicate window_id in source-matched view input")
    result = windows.copy()
    valid = _to_bool(result["window_valid_for_main_train"])
    source = result["source_type"].astype(str)
    length = pd.to_numeric(result["window_length_frames"], errors="coerce")
    result["view_combined"] = valid
    result["view_cvat_only"] = valid & source.eq("cvat_tracking_xml")
    result["view_legacy_only"] = valid & source.eq("legacy_recovered")
    result["view_matched_6frame"] = valid & length.eq(6)
    result["source_class_balance_keep"] = False
    result["source_class_balance_rank"] = 0
    result["source_class_balance_quota"] = 0
    candidates = result.loc[result["view_matched_6frame"]].copy()
    sources = sorted(candidates["source_type"].astype(str).unique())
    quotas: dict[str, int] = {}
    counts = candidates.groupby(["behavior_window_label", "source_type"])["window_id"].count()
    for label in sorted(candidates["behavior_window_label"].astype(str).unique()):
        values = [int(counts.get((label, source_name), 0)) for source_name in sources]
        quotas[label] = min(values) if len(sources) >= 2 and all(value > 0 for value in values) else 0
    ordered = candidates.sort_values(
        ["behavior_window_label", "source_type", "window_id"], kind="mergesort"
    )
    ranks = ordered.groupby(["behavior_window_label", "source_type"]).cumcount() + 1
    result.loc[ordered.index, "source_class_balance_rank"] = ranks.astype(int)
    result["source_class_balance_quota"] = (
        result["behavior_window_label"].astype(str).map(quotas).fillna(0).astype(int)
    )
    result["source_class_balance_keep"] = (
        result["view_matched_6frame"]
        & result["source_class_balance_quota"].gt(0)
        & result["source_class_balance_rank"].le(result["source_class_balance_quota"])
    )
    result["source_control_exclusion_reason"] = "not_valid_for_main_train"
    result.loc[valid, "source_control_exclusion_reason"] = "valid_not_in_matched_6frame"
    result.loc[result["view_matched_6frame"], "source_control_exclusion_reason"] = (
        "matched_6frame_above_source_class_quota"
    )
    result.loc[result["source_class_balance_keep"], "source_control_exclusion_reason"] = (
        "source_class_matched_keep"
    )
    audit = {
        "schema_version": "classification_v2_source_matched_views_v1",
        "rows_input": int(len(windows)),
        "rows_output": int(len(result)),
        "duplicate_window_id": int(result["window_id"].duplicated().sum()),
        "source_counts": source.value_counts().sort_index().to_dict(),
        "view_counts": {
            column: int(_to_bool(result[column]).sum())
            for column in [
                "view_combined",
                "view_cvat_only",
                "view_legacy_only",
                "view_matched_6frame",
                "source_class_balance_keep",
            ]
        },
        "matched_6frame_source_behavior_counts": _contingency(
            result.loc[result["view_matched_6frame"]]
        ),
        "source_class_balanced_counts": _contingency(
            result.loc[result["source_class_balance_keep"]]
        ),
        "rows_dropped": 0,
        "labels_changed": 0,
        "errors": [],
        "valid": len(result) == len(windows) and not result["window_id"].duplicated().any(),
    }
    return result, audit


def grouped_source_probe(
    features: pd.DataFrame,
    window_metadata: pd.DataFrame,
    event_mapping: pd.DataFrame,
    grouped_roles: pd.DataFrame,
    *,
    max_iter: int = 500,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Predict source on outer-test groups with scaler/model fit on train groups only."""

    if not (len(features) == len(window_metadata) == len(event_mapping)):
        raise ValueError(
            "source probe row alignment mismatch: "
            f"features={len(features)}, metadata={len(window_metadata)}, events={len(event_mapping)}"
        )
    if "window_id" not in event_mapping or not window_metadata["window_id"].astype(str).reset_index(
        drop=True
    ).equals(event_mapping["window_id"].astype(str).reset_index(drop=True)):
        raise ValueError("source probe window/event row-order alignment mismatch")
    if features.select_dtypes(exclude=[np.number]).columns.tolist():
        raise ValueError("source probe features must be an explicit numeric feature table")
    if "window_valid_for_main_train" not in window_metadata:
        raise ValueError("source probe metadata missing window_valid_for_main_train")
    base = window_metadata[["window_id", "source_type", "window_valid_for_main_train"]].copy()
    base["temporal_unit_key"] = event_mapping["temporal_unit_keys_window"].astype(str).to_numpy()
    base["row_index"] = np.arange(len(base), dtype=np.int64)
    predictions: list[pd.DataFrame] = []
    fold_audits: list[dict[str, Any]] = []
    for fold_id in sorted(grouped_roles["outer_fold_id"].astype(str).unique()):
        roles = grouped_roles.loc[
            grouped_roles["outer_fold_id"].astype(str).eq(fold_id),
            ["temporal_unit_key", "role"],
        ]
        merged = base.merge(roles, on="temporal_unit_key", how="left", validate="many_to_one")
        eligible = _to_bool(merged["window_valid_for_main_train"])
        train = merged["role"].eq("train") & eligible
        test = merged["role"].eq("test") & eligible
        if not train.any() or not test.any():
            raise ValueError(f"empty grouped source probe split={fold_id}")
        y_train = merged.loc[train, "source_type"].astype(str)
        y_test = merged.loc[test, "source_type"].astype(str)
        if y_train.nunique() < 2:
            raise ValueError(f"source probe training fold has one source={fold_id}")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=max_iter, class_weight="balanced", random_state=0),
        )
        model.fit(features.iloc[merged.loc[train, "row_index"].to_numpy(dtype=int)], y_train)
        predicted = model.predict(features.iloc[merged.loc[test, "row_index"].to_numpy(dtype=int)])
        part = merged.loc[test, ["window_id", "temporal_unit_key", "source_type"]].copy()
        part["outer_fold_id"] = fold_id
        part["source_predicted"] = predicted
        predictions.append(part)
        fold_audits.append(
            {
                "outer_fold_id": fold_id,
                "train_rows": int(train.sum()),
                "test_rows": int(test.sum()),
                "train_source_counts": y_train.value_counts().sort_index().to_dict(),
                "test_source_counts": y_test.value_counts().sort_index().to_dict(),
                "balanced_accuracy_supported": _supported_balanced_accuracy(y_test, predicted),
                "test_source_class_count": int(y_test.nunique()),
                "two_source_test_support": bool(y_test.nunique() == 2),
                "scaler_fit_on_train_only": True,
                "validation_and_test_excluded_from_fit": True,
            }
        )
    out = pd.concat(predictions, ignore_index=True)
    if out["window_id"].duplicated().any():
        raise ValueError("grouped source probe emitted duplicate OOF window predictions")
    pooled = float(balanced_accuracy_score(out["source_type"], out["source_predicted"]))
    audit = {
        "schema_version": "classification_v2_grouped_source_probe_v1",
        "feature_count": int(features.shape[1]),
        "oof_prediction_rows": int(len(out)),
        "oof_fold_count": int(out["outer_fold_id"].nunique()),
        "pooled_balanced_accuracy": pooled,
        "folds": fold_audits,
        "source_identifier_in_features": False,
        "interpretation": "internal source/domain shortcut diagnostic, not external generalization",
        "warnings": ["High source predictability can indicate source-correlated geometry or missingness."],
        "errors": [],
        "valid": True,
    }
    return out, audit


def audit_domain_feature_shift(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    """Quantify feature missingness and standardized mean shift by source and behavior."""

    required = ["source_type", "behavior_window_label", "window_valid_for_main_train"]
    missing = [column for column in required if column not in metadata]
    if missing or len(features) != len(metadata):
        raise ValueError(
            f"domain shift input mismatch: missing={missing}, features={len(features)}, metadata={len(metadata)}"
        )
    valid = _to_bool(metadata["window_valid_for_main_train"])
    x = features.loc[valid].apply(pd.to_numeric, errors="coerce")
    meta = metadata.loc[valid].reset_index(drop=True)
    x = x.reset_index(drop=True)
    sources = sorted(meta["source_type"].astype(str).unique())
    if len(sources) != 2:
        raise ValueError(f"domain shift audit requires exactly two sources, observed={sources}")
    source_stats: dict[str, Any] = {}
    for source in sources:
        subset = x.loc[meta["source_type"].astype(str).eq(source)]
        source_stats[source] = {
            "rows": int(len(subset)),
            "missing_rate": subset.isna().mean().astype(float).to_dict(),
            "mean": subset.mean().astype(float).to_dict(),
            "std": subset.std(ddof=0).astype(float).to_dict(),
        }
    left, right = sources
    left_mask = meta["source_type"].astype(str).eq(left)
    right_mask = meta["source_type"].astype(str).eq(right)
    mean_delta = x.loc[left_mask].mean() - x.loc[right_mask].mean()
    pooled_scale = np.sqrt(
        (x.loc[left_mask].var(ddof=0) + x.loc[right_mask].var(ddof=0)) / 2.0
    ).replace(0.0, np.nan)
    smd = (mean_delta / pooled_scale).replace([np.inf, -np.inf], np.nan)
    ranked = smd.abs().sort_values(ascending=False, na_position="last")
    return {
        "schema_version": "classification_v2_domain_feature_shift_v1",
        "eligible_rows": int(valid.sum()),
        "feature_count": int(features.shape[1]),
        "sources": sources,
        "source_statistics": source_stats,
        "standardized_mean_difference": smd.astype(object).where(smd.notna(), None).to_dict(),
        "top_absolute_smd_features": [
            {"feature": feature, "absolute_smd": float(ranked[feature])}
            for feature in ranked.index[:20]
            if pd.notna(ranked[feature])
        ],
        "source_behavior_counts": {
            behavior: {source: int(count) for source, count in values.items()}
            for behavior, values in pd.crosstab(
                meta["source_type"], meta["behavior_window_label"]
            ).to_dict().items()
        },
        "source_identifier_in_features": False,
        "camera_layout_metadata_status": "not_available_in_current_window_metadata",
        "camera_safe_claim_allowed": False,
        "claim_boundary": "recording-date/video-safe internal validation only",
        "errors": [],
        "valid": True,
    }


def _contingency(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    table = pd.crosstab(frame["source_type"], frame["behavior_window_label"])
    return {
        label: {source: int(count) for source, count in values.items()}
        for label, values in table.to_dict().items()
    }


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _supported_balanced_accuracy(true: pd.Series, predicted: np.ndarray) -> float:
    recalls = []
    predicted_series = pd.Series(predicted, index=true.index)
    for label in sorted(true.astype(str).unique()):
        mask = true.astype(str).eq(label)
        recalls.append(float(predicted_series.loc[mask].astype(str).eq(label).mean()))
    return float(np.mean(recalls)) if recalls else 0.0
