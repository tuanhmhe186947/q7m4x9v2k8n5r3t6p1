"""Support-aware post-hoc diagnostics for the legacy pen-context short run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.evaluation.metrics import evaluate_predictions
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

SCHEMA = "classification_v2.legacy_l6.pen_context_utility_diagnostic.v1"
CONFIG_SCHEMA = (
    "classification_v2.legacy_l6.pen_context_utility_diagnostic_config.v1"
)
MODES = ("parameter_matched_zero", "availability_only", "pen_context")
FEATURE_NAMES = (
    "pen_center_signed_distance_n",
    "pen_center_clearance_box_ratio",
    "pen_bbox_inside_ratio",
    "pen_distance_delta_n_per_frame",
    "pen_approach_speed_n_per_frame",
    "pen_retreat_speed_n_per_frame",
    "pen_parallel_speed_n_per_frame",
)
STATIC_FEATURES = FEATURE_NAMES[:3]
EXPECTED_NATIVE_UNITS = 245
EXPECTED_VIDEO_CLUSTERS = 33
EXPECTED_VALIDATION_WINDOWS = 980
DECLARED_FRAMES_PER_NATIVE_UNIT = 16
EXPECTED_EXPOSED_FRAMES_PER_NATIVE_UNIT = 15
EXPECTED_EXPOSED_PAIRS_PER_NATIVE_UNIT = 14


def write_pen_context_utility_diagnostic(
    config_path: Path,
    *,
    project_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Run the hash-bound diagnostic and write exclusive evidence artifacts."""

    root = project_root.resolve()
    resolved_config = config_path.resolve()
    config = _read_json(resolved_config)
    _validate_config(config)
    _validate_spec(root, config["implementation"], "implementation")
    _validate_spec(root, config["decision_artifact"], "decision artifact")
    predictions = _load_predictions(root, config["predictions"])
    exposure = _load_native_exposure(root, config["pen_cache"], predictions)
    per_class = build_per_class_diagnostic(predictions)
    bootstrap = build_per_class_cluster_bootstrap(
        predictions,
        iterations=int(config["analysis_contract"]["bootstrap_iterations"]),
        seed=int(config["analysis_contract"]["bootstrap_seed"]),
    )
    boundary, class_boundary, native = build_boundary_diagnostics(
        predictions,
        exposure,
    )
    output = config["output"]
    paths = {
        name: _resolve_inside(root, str(output[name]))
        for name in (
            "summary_json",
            "per_class_csv",
            "boundary_strata_csv",
            "class_boundary_csv",
            "native_exposure_csv",
        )
    }
    _write_csv_exclusive(paths["per_class_csv"], per_class)
    _write_csv_exclusive(paths["boundary_strata_csv"], boundary)
    _write_csv_exclusive(paths["class_boundary_csv"], class_boundary)
    _write_csv_exclusive(paths["native_exposure_csv"], native)
    csv_artifacts = {
        name: {
            "path": str(path),
            "sha256": file_sha256(path),
            "rows": int(
                {
                    "per_class_csv": len(per_class),
                    "boundary_strata_csv": len(boundary),
                    "class_boundary_csv": len(class_boundary),
                    "native_exposure_csv": len(native),
                }[name]
            ),
        }
        for name, path in paths.items()
        if name != "summary_json"
    }
    support = {
        str(row.behavior_label): {
            "support": int(row.support),
            "reliability_tier": str(row.reliability_tier),
        }
        for row in per_class.itertuples(index=False)
    }
    summary = {
        "schema_version": SCHEMA,
        "status": "PASS_LEGACY_L6_PEN_CONTEXT_UTILITY_DIAGNOSTIC",
        "analysis_scope": "post_hoc_exploratory_not_promotion_evidence",
        "lineage_scope": "legacy-only-unreviewed-development",
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "promotion_decision_changed": False,
        "full_pen_context_expansion_authorized": False,
        "config_path": str(resolved_config),
        "config_sha256": file_sha256(resolved_config),
        "decision_artifact": _bound_summary(
            root,
            config["decision_artifact"],
        ),
        "prediction_inputs": {
            mode: _bound_summary(root, config["predictions"][mode])
            for mode in MODES
        },
        "pen_cache_manifest": _bound_summary(
            root,
            config["pen_cache"]["manifest"],
        ),
        "common_universe": {
            "native_units": EXPECTED_NATIVE_UNITS,
            "video_clusters": EXPECTED_VIDEO_CLUSTERS,
            "validation_windows": EXPECTED_VALIDATION_WINDOWS,
            "declared_frames_per_native_unit": DECLARED_FRAMES_PER_NATIVE_UNIT,
            "exposed_unique_frames_per_native_unit": (
                EXPECTED_EXPOSED_FRAMES_PER_NATIVE_UNIT
            ),
            "exposed_unique_pairs_per_native_unit": (
                EXPECTED_EXPOSED_PAIRS_PER_NATIVE_UNIT
            ),
            "unexposed_declared_frames_per_native_unit": 1,
            "exact_prediction_metadata_equality": True,
            "exact_prediction_exposure_join": True,
        },
        "boundary_contract": {
            "near_boundary_definition": (
                "pen_center_clearance_box_ratio <= 1.0"
            ),
            "interior_only": "near_boundary_frame_fraction == 0",
            "intermittent_boundary": (
                "0 < near_boundary_frame_fraction < 0.5"
            ),
            "persistent_boundary": "near_boundary_frame_fraction >= 0.5",
            "overlapping_t6_slots_deduplicated_by": "frame_uid",
            "exposure_scope": "15_unique_frames_covered_by_four_t6_windows",
        },
        "class_support": support,
        "per_class_cluster_bootstrap": bootstrap,
        "descriptive_findings": _descriptive_findings(per_class),
        "artifacts": csv_artifacts,
        "limitations": [
            "post_hoc_analysis_does_not_override_predeclared_promotion_gate",
            "legacy_only_unreviewed_labels",
            "rare_classes_have_low_or_single_unit_support",
            "boundary_strata_are_associational_not_feature_attribution",
            "single_fixed_camera_pen_calibration",
            "three_epoch_short_training_not_full_convergence",
            "four_t6_windows_expose_15_of_16_declared_native_frames",
        ],
        "errors": [],
        "valid": True,
    }
    _write_json_exclusive(paths["summary_json"], summary)
    return paths["summary_json"], summary


def build_per_class_diagnostic(
    predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compare per-class hard metrics, paired correctness, and true-label NLL."""

    _validate_prediction_universe(predictions)
    rows: list[dict[str, Any]] = []
    zero = predictions["parameter_matched_zero"]
    pen = predictions["pen_context"]
    for label in VALID_BEHAVIORS:
        row: dict[str, Any] = {
            "behavior_label": label,
            "support": int(zero["behavior_label"].astype(str).eq(label).sum()),
        }
        mode_metrics: dict[str, dict[str, float | int]] = {}
        for mode in MODES:
            metrics = _single_class_metrics(predictions[mode], label)
            mode_metrics[mode] = metrics
            prefix = _mode_prefix(mode)
            for name, value in metrics.items():
                row[f"{prefix}_{name}"] = value
        zero_correct = zero["predicted_label"].astype(str).eq(label)
        pen_correct = pen["predicted_label"].astype(str).eq(label)
        true_label = zero["behavior_label"].astype(str).eq(label)
        row.update(
            {
                "candidate_only_correct": int(
                    (true_label & pen_correct & ~zero_correct).sum()
                ),
                "baseline_only_correct": int(
                    (true_label & zero_correct & ~pen_correct).sum()
                ),
                "pen_minus_zero_precision": float(
                    mode_metrics["pen_context"]["precision"]
                    - mode_metrics["parameter_matched_zero"]["precision"]
                ),
                "pen_minus_zero_recall": float(
                    mode_metrics["pen_context"]["recall"]
                    - mode_metrics["parameter_matched_zero"]["recall"]
                ),
                "pen_minus_zero_f1": float(
                    mode_metrics["pen_context"]["f1"]
                    - mode_metrics["parameter_matched_zero"]["f1"]
                ),
                "pen_minus_availability_f1": float(
                    mode_metrics["pen_context"]["f1"]
                    - mode_metrics["availability_only"]["f1"]
                ),
                "pen_minus_zero_true_nll": float(
                    mode_metrics["pen_context"]["true_nll"]
                    - mode_metrics["parameter_matched_zero"]["true_nll"]
                ),
                "pen_minus_availability_true_nll": float(
                    mode_metrics["pen_context"]["true_nll"]
                    - mode_metrics["availability_only"]["true_nll"]
                ),
                "reliability_tier": _reliability_tier(int(row["support"])),
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def build_per_class_cluster_bootstrap(
    predictions: dict[str, pd.DataFrame],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap class-specific F1/NLL deltas by video without window rows."""

    if iterations < 1000:
        raise ValueError("pen diagnostic bootstrap requires at least 1000 draws")
    _validate_prediction_universe(predictions)
    reference = predictions["parameter_matched_zero"]
    clusters = sorted(reference["video_key"].astype(str).unique())
    if len(clusters) != EXPECTED_VIDEO_CLUSTERS:
        raise ValueError(f"pen diagnostic video clusters={len(clusters)}")
    labels = list(VALID_BEHAVIORS)
    counts = np.zeros((len(clusters), len(labels), len(MODES), 4), dtype=float)
    for cluster_index, cluster in enumerate(clusters):
        cluster_mask = reference["video_key"].astype(str).eq(cluster).to_numpy()
        for label_index, label in enumerate(labels):
            true = reference["behavior_label"].astype(str).eq(label).to_numpy()
            true_cluster = true & cluster_mask
            support = int(true_cluster.sum())
            for mode_index, mode in enumerate(MODES):
                frame = predictions[mode]
                predicted = frame["predicted_label"].astype(str).eq(label).to_numpy()
                tp = int((true_cluster & predicted).sum())
                fp = int((cluster_mask & ~true & predicted).sum())
                fn = support - tp
                nll_sum = float(
                    _row_true_nll(frame.loc[true_cluster], label).sum()
                )
                counts[cluster_index, label_index, mode_index] = (
                    tp,
                    fp,
                    fn,
                    nll_sum,
                )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0,
        len(clusters),
        size=(iterations, len(clusters)),
    )
    totals = counts[sampled].sum(axis=1)
    rows: dict[str, Any] = {}
    for label_index, label in enumerate(labels):
        f1_by_mode: dict[str, np.ndarray] = {}
        nll_by_mode: dict[str, np.ndarray] = {}
        for mode_index, mode in enumerate(MODES):
            values = totals[:, label_index, mode_index]
            denominator = 2.0 * values[:, 0] + values[:, 1] + values[:, 2]
            f1_by_mode[mode] = np.divide(
                2.0 * values[:, 0],
                denominator,
                out=np.full(iterations, np.nan),
                where=denominator > 0,
            )
            support = values[:, 0] + values[:, 2]
            nll_by_mode[mode] = np.divide(
                values[:, 3],
                support,
                out=np.full(iterations, np.nan),
                where=support > 0,
            )
        rows[label] = {
            "support": int(
                reference["behavior_label"].astype(str).eq(label).sum()
            ),
            "reliability_tier": _reliability_tier(
                int(reference["behavior_label"].astype(str).eq(label).sum())
            ),
            "pen_minus_zero_f1": _bootstrap_interval(
                f1_by_mode["pen_context"]
                - f1_by_mode["parameter_matched_zero"]
            ),
            "pen_minus_availability_f1": _bootstrap_interval(
                f1_by_mode["pen_context"] - f1_by_mode["availability_only"]
            ),
            "pen_minus_zero_true_nll": _bootstrap_interval(
                nll_by_mode["pen_context"]
                - nll_by_mode["parameter_matched_zero"]
            ),
        }
    return {
        "cluster_unit": "video_key",
        "native_unit": "temporal_unit_key",
        "iterations": iterations,
        "seed": seed,
        "video_clusters": len(clusters),
        "classes": rows,
    }


def build_boundary_diagnostics(
    predictions: dict[str, pd.DataFrame],
    exposure: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Describe paired effects by semantic pen-boundary exposure strata."""

    _validate_prediction_universe(predictions)
    native = exposure.merge(
        predictions["parameter_matched_zero"][
            ["temporal_unit_key", "video_key", "behavior_label"]
        ],
        on="temporal_unit_key",
        how="inner",
        validate="one_to_one",
    )
    if len(native) != EXPECTED_NATIVE_UNITS:
        raise ValueError(f"pen native exposure join rows={len(native)}")
    for mode in MODES:
        frame = predictions[mode][
            ["temporal_unit_key", "predicted_label", *list(_probability_columns())]
        ].copy()
        frame = frame.rename(
            columns={
                "predicted_label": f"predicted_{mode}",
                **{
                    column: f"{column}_{mode}"
                    for column in _probability_columns()
                },
            }
        )
        native = native.merge(
            frame,
            on="temporal_unit_key",
            how="inner",
            validate="one_to_one",
        )
    native["zero_correct"] = native["predicted_parameter_matched_zero"].eq(
        native["behavior_label"]
    )
    native["pen_correct"] = native["predicted_pen_context"].eq(
        native["behavior_label"]
    )
    boundary_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for stratum, group in native.groupby("boundary_stratum", sort=True):
        for mode in MODES:
            evaluated = _evaluate_native_slice(group, mode)
            boundary_rows.append(
                {
                    "boundary_stratum": str(stratum),
                    "mode": mode,
                    "native_units": len(group),
                    "video_clusters": int(group["video_key"].nunique()),
                    **evaluated,
                }
            )
        for label in VALID_BEHAVIORS:
            true_rows = group.loc[group["behavior_label"].eq(label)]
            zero_metrics = _single_class_native_metrics(group, label, MODES[0])
            pen_metrics = _single_class_native_metrics(group, label, MODES[2])
            class_rows.append(
                {
                    "boundary_stratum": str(stratum),
                    "behavior_label": label,
                    "support": len(true_rows),
                    "reliability_tier": _reliability_tier(len(true_rows)),
                    "zero_recall": zero_metrics["recall"],
                    "pen_recall": pen_metrics["recall"],
                    "pen_minus_zero_recall": (
                        pen_metrics["recall"] - zero_metrics["recall"]
                    ),
                    "zero_f1": zero_metrics["f1"],
                    "pen_f1": pen_metrics["f1"],
                    "pen_minus_zero_f1": pen_metrics["f1"] - zero_metrics["f1"],
                    "candidate_only_correct": int(
                        (true_rows["pen_correct"] & ~true_rows["zero_correct"]).sum()
                    ),
                    "baseline_only_correct": int(
                        (~true_rows["pen_correct"] & true_rows["zero_correct"]).sum()
                    ),
                    "pen_minus_zero_true_nll": _slice_nll_delta(
                        true_rows,
                        label,
                    ),
                }
            )
    return (
        pd.DataFrame.from_records(boundary_rows),
        pd.DataFrame.from_records(class_rows),
        native,
    )


def _load_predictions(
    root: Path,
    specs: object,
) -> dict[str, pd.DataFrame]:
    values = _object(specs, "predictions")
    if set(values) != set(MODES):
        raise ValueError("pen diagnostic prediction mode set drift")
    predictions: dict[str, pd.DataFrame] = {}
    for mode in MODES:
        path = _validate_spec(root, values[mode], f"predictions.{mode}")
        frame = pd.read_csv(path)
        required = {
            "temporal_unit_key",
            "video_key",
            "behavior_label",
            "predicted_label",
            "pen_context_mode",
            "missing_modality",
            *_probability_columns(),
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{mode} predictions missing={missing}")
        if set(frame["pen_context_mode"].astype(str)) != {mode}:
            raise ValueError(f"{mode} mode column drift")
        if frame["missing_modality"].astype(str).str.lower().ne("false").any():
            raise ValueError(f"{mode} unexpectedly marks missing modality")
        predictions[mode] = frame.sort_values(
            "temporal_unit_key",
            kind="mergesort",
        ).reset_index(drop=True)
    _validate_prediction_universe(predictions)
    return predictions


def _validate_prediction_universe(
    predictions: dict[str, pd.DataFrame],
) -> None:
    if set(predictions) != set(MODES):
        raise ValueError("pen diagnostic prediction dictionaries differ")
    reference: pd.DataFrame | None = None
    for mode in MODES:
        frame = predictions[mode]
        if len(frame) != EXPECTED_NATIVE_UNITS:
            raise ValueError(f"{mode} native prediction rows={len(frame)}")
        if frame["temporal_unit_key"].astype(str).duplicated().any():
            raise ValueError(f"{mode} duplicate native units")
        metadata = frame[
            ["temporal_unit_key", "video_key", "behavior_label"]
        ].astype(str).reset_index(drop=True)
        if reference is None:
            reference = metadata
        elif not metadata.equals(reference):
            raise ValueError(f"{mode} native metadata differs")
        probabilities = frame[list(_probability_columns())].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all():
            raise ValueError(f"{mode} probabilities are nonfinite")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError(f"{mode} probabilities do not sum to one")
    assert reference is not None
    if reference["video_key"].nunique() != EXPECTED_VIDEO_CLUSTERS:
        raise ValueError("pen diagnostic video-cluster count drift")


def _load_native_exposure(
    root: Path,
    specs: object,
    predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    cache = _object(specs, "pen_cache")
    for name in ("manifest", "window_index", "slot_index", "pen_tensor"):
        _validate_spec(root, cache[name], f"pen_cache.{name}")
    manifest = _read_json(_resolve_inside(root, cache["manifest"]["path"]))
    if manifest.get("valid") is not True or manifest.get("errors") != []:
        raise ValueError("pen cache manifest is not valid")
    window_path = _resolve_inside(root, cache["window_index"]["path"])
    slots_path = _resolve_inside(root, cache["slot_index"]["path"])
    tensor_path = _resolve_inside(root, cache["pen_tensor"]["path"])
    windows = pd.read_csv(window_path)
    validation = windows.loc[windows["l5_role"].astype(str).eq("validation")].copy()
    if len(validation) != EXPECTED_VALIDATION_WINDOWS:
        raise ValueError(f"pen validation windows={len(validation)}")
    prediction_units = set(
        predictions["parameter_matched_zero"]["temporal_unit_key"].astype(str)
    )
    if set(validation["temporal_unit_key"].astype(str)) != prediction_units:
        raise ValueError("pen cache and prediction native units differ")
    slots = pd.read_csv(
        slots_path,
        usecols=[
            "cache_row",
            "slot_index",
            "frame_uid",
            "pen_pair_uid",
        ],
    )
    validation_rows = set(validation["cache_row"].astype(int))
    slots = slots.loc[slots["cache_row"].astype(int).isin(validation_rows)].copy()
    expected_slots = EXPECTED_VALIDATION_WINDOWS * 6
    if len(slots) != expected_slots:
        raise ValueError(f"pen validation slots={len(slots)}")
    tensor = np.load(tensor_path, mmap_mode="r")
    if tensor.ndim != 3 or tensor.shape[1:] != (6, len(FEATURE_NAMES)):
        raise ValueError(f"pen tensor shape={tensor.shape}")
    cache_rows = slots["cache_row"].to_numpy(dtype=np.int64)
    slot_rows = slots["slot_index"].to_numpy(dtype=np.int64)
    values = np.asarray(tensor[cache_rows, slot_rows], dtype=np.float64)
    for index, name in enumerate(FEATURE_NAMES):
        slots[name] = values[:, index]
    slots = slots.merge(
        validation[["cache_row", "temporal_unit_key"]],
        on="cache_row",
        how="left",
        validate="many_to_one",
    )
    _validate_repeated_static_features(slots)
    static = slots.drop_duplicates(
        ["temporal_unit_key", "frame_uid"],
        keep="first",
    )
    static["near_boundary"] = static[
        "pen_center_clearance_box_ratio"
    ].le(1.0)
    native = static.groupby("temporal_unit_key", sort=True).agg(
        unique_frame_count=("frame_uid", "nunique"),
        near_boundary_frame_fraction=("near_boundary", "mean"),
        mean_signed_distance_n=("pen_center_signed_distance_n", "mean"),
        min_signed_distance_n=("pen_center_signed_distance_n", "min"),
        mean_clearance_box_ratio=("pen_center_clearance_box_ratio", "mean"),
        min_clearance_box_ratio=("pen_center_clearance_box_ratio", "min"),
        mean_bbox_inside_ratio=("pen_bbox_inside_ratio", "mean"),
        min_bbox_inside_ratio=("pen_bbox_inside_ratio", "min"),
    ).reset_index()
    if set(native["unique_frame_count"].astype(int)) != {
        EXPECTED_EXPOSED_FRAMES_PER_NATIVE_UNIT
    }:
        raise ValueError("pen native units do not contain 15 exposed frames")
    pairs = slots.loc[slots["pen_pair_uid"].fillna("").astype(str).ne("")].copy()
    pairs = pairs.drop_duplicates(
        ["temporal_unit_key", "pen_pair_uid"],
        keep="first",
    )
    pair_native = pairs.groupby("temporal_unit_key", sort=True).agg(
        unique_motion_pair_count=("pen_pair_uid", "nunique"),
        mean_approach_speed=("pen_approach_speed_n_per_frame", "mean"),
        max_approach_speed=("pen_approach_speed_n_per_frame", "max"),
        mean_retreat_speed=("pen_retreat_speed_n_per_frame", "mean"),
        max_retreat_speed=("pen_retreat_speed_n_per_frame", "max"),
        mean_parallel_speed=("pen_parallel_speed_n_per_frame", "mean"),
        max_parallel_speed=("pen_parallel_speed_n_per_frame", "max"),
    ).reset_index()
    native = native.merge(
        pair_native,
        on="temporal_unit_key",
        how="left",
        validate="one_to_one",
    )
    if set(native["unique_motion_pair_count"].astype(int)) != {
        EXPECTED_EXPOSED_PAIRS_PER_NATIVE_UNIT
    }:
        raise ValueError("pen native units do not contain 14 exposed frame pairs")
    native["boundary_stratum"] = native[
        "near_boundary_frame_fraction"
    ].map(_boundary_stratum)
    return native


def _single_class_metrics(frame: pd.DataFrame, label: str) -> dict[str, float | int]:
    true = frame["behavior_label"].astype(str).eq(label)
    predicted = frame["predicted_label"].astype(str).eq(label)
    tp = int((true & predicted).sum())
    support = int(true.sum())
    predicted_count = int(predicted.sum())
    precision = tp / predicted_count if predicted_count else 0.0
    recall = tp / support if support else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    true_nll = float(_row_true_nll(frame.loc[true], label).mean()) if support else 0.0
    return {
        "predicted_count": predicted_count,
        "true_positive": tp,
        "false_positive": predicted_count - tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_nll": true_nll,
    }


def _single_class_native_metrics(
    frame: pd.DataFrame,
    label: str,
    mode: str,
) -> dict[str, float]:
    true = frame["behavior_label"].astype(str).eq(label)
    predicted = frame[f"predicted_{mode}"].astype(str).eq(label)
    tp = int((true & predicted).sum())
    support = int(true.sum())
    predicted_count = int(predicted.sum())
    precision = tp / predicted_count if predicted_count else 0.0
    recall = tp / support if support else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _evaluate_native_slice(frame: pd.DataFrame, mode: str) -> dict[str, float | int]:
    temporary = pd.DataFrame(
        {
            "true": frame["behavior_label"].astype(str),
            "predicted": frame[f"predicted_{mode}"].astype(str),
        }
    )
    metrics = evaluate_predictions(
        temporary,
        y_true_col="true",
        y_pred_col="predicted",
        label_order=list(VALID_BEHAVIORS),
    )
    nll = _native_mode_nll(frame, mode)
    return {
        "supported_class_count": metrics["supported_label_count"],
        "accuracy": metrics["accuracy"],
        "macro_f1_global_10_class": metrics["macro_f1"],
        "macro_f1_supported": metrics["macro_f1_supported"],
        "mean_true_nll": nll,
    }


def _native_mode_nll(frame: pd.DataFrame, mode: str) -> float:
    values: list[float] = []
    for label in VALID_BEHAVIORS:
        selected = frame.loc[frame["behavior_label"].astype(str).eq(label)]
        if selected.empty:
            continue
        column = f"{_probability_column(label)}_{mode}"
        probabilities = selected[column].to_numpy(dtype=float)
        values.extend((-np.log(np.clip(probabilities, 1e-12, 1.0))).tolist())
    return float(np.mean(values)) if values else 0.0


def _slice_nll_delta(frame: pd.DataFrame, label: str) -> float:
    if frame.empty:
        return 0.0
    probability = _probability_column(label)
    pen = -np.log(
        np.clip(
            frame[f"{probability}_pen_context"].to_numpy(dtype=float),
            1e-12,
            1.0,
        )
    )
    zero = -np.log(
        np.clip(
            frame[f"{probability}_parameter_matched_zero"].to_numpy(dtype=float),
            1e-12,
            1.0,
        )
    )
    return float((pen - zero).mean())


def _row_true_nll(frame: pd.DataFrame, label: str) -> np.ndarray:
    if frame.empty:
        return np.asarray([], dtype=float)
    values = frame[_probability_column(label)].to_numpy(dtype=float)
    return -np.log(np.clip(values, 1e-12, 1.0))


def _bootstrap_interval(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {
            "valid_iterations": 0,
            "bootstrap_mean": None,
            "ci_low": None,
            "ci_high": None,
        }
    return {
        "valid_iterations": int(len(finite)),
        "bootstrap_mean": float(finite.mean()),
        "ci_low": float(np.quantile(finite, 0.025)),
        "ci_high": float(np.quantile(finite, 0.975)),
        "fraction_positive": float(np.mean(finite > 0.0)),
        "fraction_negative": float(np.mean(finite < 0.0)),
    }


def _validate_repeated_static_features(slots: pd.DataFrame) -> None:
    for feature in STATIC_FEATURES:
        conflicts = slots.groupby(
            ["temporal_unit_key", "frame_uid"],
            sort=False,
        )[feature].nunique(dropna=False)
        if conflicts.gt(1).any():
            raise ValueError(f"pen repeated frame conflicts feature={feature}")


def _boundary_stratum(value: float) -> str:
    ratio = float(value)
    if ratio == 0.0:
        return "interior_only"
    if ratio < 0.5:
        return "intermittent_boundary"
    return "persistent_boundary"


def _reliability_tier(support: int) -> str:
    if support == 0:
        return "no_support"
    if support >= 20:
        return "moderate_descriptive_support"
    if support >= 5:
        return "low_support"
    if support >= 2:
        return "very_low_support"
    return "single_unit_not_estimable"


def _descriptive_findings(frame: pd.DataFrame) -> dict[str, Any]:
    positive = frame.loc[frame["pen_minus_zero_f1"].gt(0)].sort_values(
        "pen_minus_zero_f1",
        ascending=False,
    )
    negative = frame.loc[frame["pen_minus_zero_f1"].lt(0)].sort_values(
        "pen_minus_zero_f1",
    )
    return {
        "positive_f1_delta_classes": positive[
            ["behavior_label", "support", "pen_minus_zero_f1"]
        ].to_dict(orient="records"),
        "negative_f1_delta_classes": negative[
            ["behavior_label", "support", "pen_minus_zero_f1"]
        ].to_dict(orient="records"),
        "zero_f1_classes": frame.loc[
            frame["pen_f1"].eq(0),
            "behavior_label",
        ].astype(str).tolist(),
    }


def _mode_prefix(mode: str) -> str:
    return {
        "parameter_matched_zero": "zero",
        "availability_only": "availability",
        "pen_context": "pen",
    }[mode]


def _probability_columns() -> tuple[str, ...]:
    return tuple(_probability_column(label) for label in VALID_BEHAVIORS)


def _probability_column(label: str) -> str:
    return f"prob_{label.replace('-', '_')}"


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "lineage_scope",
        "analysis_scope",
        "decision_artifact",
        "predictions",
        "pen_cache",
        "analysis_contract",
        "implementation",
        "output",
    }
    if set(config) != required:
        raise ValueError(
            f"pen diagnostic config fields={sorted(set(config) ^ required)}"
        )
    if config["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("pen diagnostic config schema drift")
    if config["lineage_scope"] != "legacy-only-unreviewed-development":
        raise ValueError("pen diagnostic lineage scope drift")
    if config["analysis_scope"] != "post_hoc_exploratory_not_promotion_evidence":
        raise ValueError("pen diagnostic analysis scope drift")
    contract = _object(config["analysis_contract"], "analysis_contract")
    expected = {
        "expected_native_units": EXPECTED_NATIVE_UNITS,
        "expected_video_clusters": EXPECTED_VIDEO_CLUSTERS,
        "expected_validation_windows": EXPECTED_VALIDATION_WINDOWS,
        "declared_frames_per_native_unit": DECLARED_FRAMES_PER_NATIVE_UNIT,
        "expected_exposed_frames_per_native_unit": (
            EXPECTED_EXPOSED_FRAMES_PER_NATIVE_UNIT
        ),
        "expected_exposed_pairs_per_native_unit": (
            EXPECTED_EXPOSED_PAIRS_PER_NATIVE_UNIT
        ),
        "near_boundary_clearance_ratio": 1.0,
        "persistent_boundary_min_fraction": 0.5,
        "bootstrap_iterations": 2000,
        "bootstrap_seed": 20260717,
    }
    if contract != expected:
        raise ValueError("pen diagnostic analysis contract drift")
    predictions = _object(config["predictions"], "predictions")
    if set(predictions) != set(MODES):
        raise ValueError("pen diagnostic prediction config modes drift")
    cache = _object(config["pen_cache"], "pen_cache")
    if set(cache) != {"manifest", "window_index", "slot_index", "pen_tensor"}:
        raise ValueError("pen diagnostic cache config fields drift")


def _validate_spec(root: Path, value: object, name: str) -> Path:
    spec = _object(value, name)
    if set(spec) != {"path", "sha256"}:
        raise ValueError(f"pen diagnostic {name} spec fields drift")
    path = _resolve_inside(root, str(spec["path"]))
    expected = str(spec["sha256"])
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"pen diagnostic {name} hash={observed}!={expected}")
    return path


def _bound_summary(root: Path, value: object) -> dict[str, str]:
    spec = _object(value, "bound spec")
    path = _resolve_inside(root, str(spec["path"]))
    return {"path": str(path), "sha256": file_sha256(path)}


def _resolve_inside(root: Path, value: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"pen diagnostic path escapes project root={value}") from error
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    return _object(json.loads(path.read_text(encoding="utf-8")), str(path))


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"pen diagnostic {name} must be an object")
    return value


def _write_csv_exclusive(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, mode="x", lineterminator="\n")


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


__all__ = [
    "build_boundary_diagnostics",
    "build_per_class_cluster_bootstrap",
    "build_per_class_diagnostic",
    "write_pen_context_utility_diagnostic",
]
