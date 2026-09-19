"""Canonical evaluation contract for concatenated behavior OOF predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.evaluation.calibration import (
    probability_calibration_metrics,
)
from pig_behavior.classification_v2.evaluation.metrics import evaluate_predictions
from pig_behavior.classification_v2.evaluation.native_unit_metrics import (
    evaluate_native_oof,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

PROBABILITY_ROW_SUM_TOLERANCE = 1e-4
NLL_CLIP_EPSILON = 1e-12


def evaluate_behavior_oof(
    predictions: pd.DataFrame | str | Path,
    canonical_manifest: pd.DataFrame | str | Path,
    class_order: Sequence[str] = VALID_BEHAVIORS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate, aggregate, and score one complete concatenated behavior OOF.

    Predictions are joined one-to-one from ``window_id`` to the canonical
    manifest's ``target_id``. The returned headline is computed once on the
    pooled native-unit rows; per-fold metrics are diagnostic only.
    """

    labels = _locked_class_order(class_order)
    prediction_frame = _read_frame(predictions, "predictions")
    manifest_frame = _read_frame(canonical_manifest, "canonical_manifest")
    _require_columns(
        prediction_frame,
        [
            "window_id",
            "fold_id",
            "oof_fold_id",
            "source_type",
            "true_label",
            "predicted_label",
        ],
        "predictions",
    )
    _require_columns(
        manifest_frame,
        [
            "target_id",
            "source_type",
            "dataset_id",
            "video_key",
            "object_track_key",
            "behavior",
            "outer_fold_id",
            "native_unit_id",
        ],
        "canonical_manifest",
    )
    integrity = _validate_identity_population(prediction_frame, manifest_frame)
    probabilities = _validated_probabilities(prediction_frame, labels)
    joined = _join_canonical_authority(prediction_frame, manifest_frame, labels)
    joined.loc[:, [f"prob_{label}" for label in labels]] = probabilities

    temporal_keys = _source_aware_temporal_keys(joined)
    native_input = _native_prediction_input(joined, temporal_keys, labels)
    fold_assignments = _native_fold_assignments(joined, temporal_keys)
    native_rows, native_audit = evaluate_native_oof(
        native_input,
        fold_assignments,
    )
    if not native_audit.get("valid", False):
        errors = native_audit.get("errors", [])
        raise ValueError(f"native OOF evaluation failed: {errors}")

    fold_lookup = pd.DataFrame(
        {
            "temporal_unit_key": temporal_keys,
            "fold_id": joined["fold_id"].astype(str).to_numpy(),
        }
    )
    native_rows = native_rows.merge(
        fold_lookup,
        on="temporal_unit_key",
        how="left",
        validate="one_to_one",
    )
    evaluable = native_rows.loc[native_rows["native_metric_include"]].copy()
    pooled = native_audit["pooled_metrics"]
    nll = _multiclass_nll(evaluable, labels)
    fold_metrics = {
        str(fold_id): _fold_metric_bundle(group, labels)
        for fold_id, group in evaluable.groupby("fold_id", sort=True)
    }
    integrity["joined_rows"] = int(len(joined))
    integrity["each_manifest_row_predicted_exactly_once"] = bool(
        len(joined) == len(manifest_frame)
    )
    report = {
        "schema_version": "classification_v2_canonical_behavior_oof_v1",
        "valid": True,
        "statistical_unit": "source_aware_native_temporal_unit",
        "class_order": labels,
        "integrity": integrity,
        "headline_definition": "pooled_concatenated_native_oof_macro_f1",
        "fold_metrics_are_diagnostic_only": True,
        "macro_f1": float(pooled["macro_f1"]),
        "nll": nll,
        "nll_definition": "-mean(log(clip(p_true,1e-12,1)))",
        "per_class": pooled["per_class"],
        "confusion_matrix": pooled["confusion_matrix"],
        "pooled_metrics": pooled,
        "fold_metrics": fold_metrics,
        "outer_fold_metrics": native_audit["fold_metrics"],
        "native_aggregation_audit": native_audit,
    }
    return native_rows, report


def per_class_metric_delta(
    candidate: Mapping[str, Mapping[str, float | int]],
    baseline: Mapping[str, Mapping[str, float | int]],
    *,
    metric: str = "f1",
    class_order: Sequence[str] = VALID_BEHAVIORS,
) -> dict[str, float]:
    """Subtract per-class metrics by class name, never by array position."""

    labels = _locked_class_order(class_order)
    expected = set(labels)
    if set(candidate) != expected:
        raise ValueError(
            "candidate class keys must exactly equal the locked class order"
        )
    if set(baseline) != expected:
        raise ValueError(
            "baseline class keys must exactly equal the locked class order"
        )
    delta: dict[str, float] = {}
    for label in labels:
        if metric not in candidate[label] or metric not in baseline[label]:
            raise ValueError(f"missing per-class metric {metric!r} for {label!r}")
        delta[label] = float(candidate[label][metric]) - float(
            baseline[label][metric]
        )
    return delta


def _locked_class_order(class_order: Sequence[str]) -> list[str]:
    labels = list(class_order)
    if labels != list(VALID_BEHAVIORS):
        raise ValueError("class_order must equal VALID_BEHAVIORS in canonical order")
    return labels


def _read_frame(value: pd.DataFrame | str | Path, name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            raise ValueError(f"{name} CSV does not exist: {path}")
        return pd.read_csv(path, low_memory=False)
    raise TypeError(f"{name} must be a pandas DataFrame or CSV path")


def _require_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _clean_ids(frame: pd.DataFrame, column: str, name: str) -> pd.Series:
    values = frame[column].fillna("").astype(str).str.strip()
    if values.eq("").any():
        raise ValueError(f"blank {name} identities: {int(values.eq('').sum())}")
    return values


def _validate_identity_population(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
) -> dict[str, int | bool]:
    prediction_ids = _clean_ids(predictions, "window_id", "prediction window_id")
    manifest_ids = _clean_ids(manifest, "target_id", "manifest target_id")
    duplicate_predictions = int(prediction_ids.duplicated(keep=False).sum())
    duplicate_manifest = int(manifest_ids.duplicated(keep=False).sum())
    if duplicate_predictions:
        raise ValueError(
            f"duplicate prediction window_id rows: {duplicate_predictions}"
        )
    if duplicate_manifest:
        raise ValueError(f"duplicate manifest target_id rows: {duplicate_manifest}")
    prediction_set = set(prediction_ids)
    manifest_set = set(manifest_ids)
    missing = sorted(manifest_set - prediction_set)
    extra = sorted(prediction_set - manifest_set)
    if missing:
        raise ValueError(
            f"missing prediction identities: count={len(missing)}, examples={missing[:5]}"
        )
    if extra:
        raise ValueError(
            f"extra prediction identities: count={len(extra)}, examples={extra[:5]}"
        )
    return {
        "prediction_rows": int(len(predictions)),
        "manifest_rows": int(len(manifest)),
        "joined_rows": 0,
        "duplicate_prediction_identity_count": duplicate_predictions,
        "duplicate_manifest_identity_count": duplicate_manifest,
        "missing_prediction_identity_count": len(missing),
        "extra_prediction_identity_count": len(extra),
        "each_manifest_row_predicted_exactly_once": False,
    }


def _validated_probabilities(
    predictions: pd.DataFrame,
    labels: list[str],
) -> np.ndarray:
    expected = [f"prob_{label}" for label in labels]
    observed = [column for column in predictions if column.startswith("prob_")]
    if observed != expected:
        raise ValueError(
            "probability column order must exactly match class_order: "
            f"expected={expected}, observed={observed}"
        )
    numeric = predictions[expected].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("probabilities must be finite")
    if (values < 0.0).any():
        raise ValueError("probabilities must be non-negative")
    row_sums = values.sum(axis=1)
    deviation = np.abs(row_sums - 1.0)
    if (deviation > PROBABILITY_ROW_SUM_TOLERANCE).any():
        invalid = int((deviation > PROBABILITY_ROW_SUM_TOLERANCE).sum())
        raise ValueError(
            "probability row sums must be within 1e-4 of one before "
            f"normalization: invalid_rows={invalid}"
        )
    return values / row_sums[:, None]


def _join_canonical_authority(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    labels: list[str],
) -> pd.DataFrame:
    prediction_frame = predictions.copy()
    prediction_frame["_prediction_order"] = np.arange(len(prediction_frame))
    joined = prediction_frame.merge(
        manifest,
        left_on="window_id",
        right_on="target_id",
        how="inner",
        suffixes=("_prediction", "_manifest"),
        sort=False,
        validate="one_to_one",
    ).sort_values("_prediction_order", kind="stable")
    joined = joined.reset_index(drop=True)
    _validate_joined_authority(joined, labels)
    return joined


def _validate_joined_authority(joined: pd.DataFrame, labels: list[str]) -> None:
    valid_labels = set(labels)
    for column in [
        "target_id",
        "source_type_manifest",
        "dataset_id",
        "video_key",
        "object_track_key",
        "behavior",
        "outer_fold_id",
    ]:
        blank = joined[column].fillna("").astype(str).str.strip().eq("")
        if blank.any():
            raise ValueError(
                f"blank canonical {column} rows: {int(blank.sum())}"
            )
    for column in ["true_label", "predicted_label"]:
        values = joined[column].fillna("").astype(str)
        invalid = sorted(set(values) - valid_labels)
        if invalid:
            raise ValueError(f"invalid {column} values: {invalid}")
    checks = [
        ("true_label", "behavior", "true labels disagree with canonical manifest"),
        (
            "source_type_prediction",
            "source_type_manifest",
            "prediction source_type disagrees with canonical manifest",
        ),
        (
            "oof_fold_id",
            "outer_fold_id",
            "prediction oof_fold_id disagrees with canonical manifest",
        ),
    ]
    for left, right, message in checks:
        mismatch = joined[left].fillna("").astype(str).ne(
            joined[right].fillna("").astype(str)
        )
        if mismatch.any():
            raise ValueError(f"{message}: {int(mismatch.sum())} rows")
    for redundant, authority in [
        ("y_true", "true_label"),
        ("y_pred", "predicted_label"),
    ]:
        if redundant in joined.columns:
            mismatch = joined[redundant].fillna("").astype(str).ne(
                joined[authority].fillna("").astype(str)
            )
            if mismatch.any():
                raise ValueError(
                    f"{redundant} disagrees with {authority}: {int(mismatch.sum())} rows"
                )
    fold_ids = joined["fold_id"].fillna("").astype(str).str.strip()
    if fold_ids.eq("").any():
        raise ValueError(f"blank fold_id rows: {int(fold_ids.eq('').sum())}")


def _source_aware_temporal_keys(joined: pd.DataFrame) -> pd.Series:
    source = joined["source_type_manifest"].astype(str)
    valid_sources = {"legacy_recovered", "cvat_tracking_xml"}
    invalid_sources = sorted(set(source) - valid_sources)
    if invalid_sources:
        raise ValueError(f"invalid canonical source_type values: {invalid_sources}")
    native_id = joined["native_unit_id"].fillna("").astype(str).str.strip()
    legacy_missing = source.eq("legacy_recovered") & native_id.eq("")
    if legacy_missing.any():
        raise ValueError(
            "legacy rows require canonical native_unit_id: "
            f"{int(legacy_missing.sum())} rows"
        )
    authority_id = native_id.where(source.eq("legacy_recovered"), joined["target_id"])
    temporal_keys = source + "|" + authority_id.astype(str)
    if temporal_keys.duplicated().any():
        raise ValueError("source-aware native temporal identities are not unique")
    return temporal_keys.rename("temporal_unit_key")


def _native_prediction_input(
    joined: pd.DataFrame,
    temporal_keys: pd.Series,
    labels: list[str],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "window_id": joined["window_id"].astype(str),
            "temporal_unit_key": temporal_keys,
            "true_label": joined["true_label"].astype(str),
            "predicted_label": joined["predicted_label"].astype(str),
            "oof_fold_id": joined["oof_fold_id"].astype(str),
        }
    )
    for label in labels:
        frame[f"prob_{label}"] = joined[f"prob_{label}"].to_numpy(dtype=float)
    return frame


def _native_fold_assignments(
    joined: pd.DataFrame,
    temporal_keys: pd.Series,
) -> pd.DataFrame:
    source = joined["source_type_manifest"].astype(str)
    dataset = joined["dataset_id"].astype(str)
    video = joined["video_key"].astype(str)
    return pd.DataFrame(
        {
            "temporal_unit_key": temporal_keys,
            "recording_group_id": source + "|" + dataset + "|" + video,
            "outer_fold_id": joined["outer_fold_id"].astype(str),
            "behavior_label": joined["behavior"].astype(str),
            "source_type": source,
            "video_key": video,
            "native_unit_valid_for_main_eval": True,
        }
    )


def _fold_metric_bundle(frame: pd.DataFrame, labels: list[str]) -> dict[str, Any]:
    metrics = evaluate_predictions(
        frame,
        y_true_col="true_label",
        y_pred_col="native_predicted_behavior",
        label_order=labels,
    )
    probability_columns = [f"prob_{label}" for label in labels]
    probabilities = frame[probability_columns].to_numpy(dtype=np.float64)
    label_to_index = {label: index for index, label in enumerate(labels)}
    targets = np.asarray(
        [label_to_index[label] for label in frame["true_label"]],
        dtype=np.int64,
    )
    calibration = probability_calibration_metrics(
        probabilities,
        targets,
        ece_bins=15,
    )
    return {
        "rows": int(len(frame)),
        "macro_f1": float(metrics["macro_f1"]),
        "macro_f1_supported": float(metrics["macro_f1_supported"]),
        "nll": float(calibration["negative_log_likelihood"]),
    }


def _multiclass_nll(frame: pd.DataFrame, labels: list[str]) -> float:
    probability_columns = [f"prob_{label}" for label in labels]
    probabilities = frame[probability_columns].to_numpy(dtype=np.float64)
    label_to_index = {label: index for index, label in enumerate(labels)}
    targets = np.asarray(
        [label_to_index[label] for label in frame["true_label"]],
        dtype=np.int64,
    )
    true_probability = probabilities[np.arange(len(targets)), targets]
    return float(
        -np.log(np.clip(true_probability, NLL_CLIP_EPSILON, 1.0)).mean()
    )
