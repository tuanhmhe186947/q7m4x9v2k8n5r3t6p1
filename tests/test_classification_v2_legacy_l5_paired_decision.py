from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.evaluation.legacy_development_l5_paired_decision import (
    FIXED_RUN_MANIFEST_FIELDS,
    _compare_packets,
    _make_decision,
    _validate_semantic_pair,
)
from pig_behavior.classification_v2.evaluation.metrics import evaluate_predictions
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS


def test_paired_decision_retains_simpler_v1_on_negative_tcn_evidence() -> None:
    baseline_predictions = _predictions(error_label=None).sample(
        frac=1.0,
        random_state=11,
    )
    candidate_predictions = _predictions(error_label="drink")
    baseline = _packet(
        role="baseline",
        encoder="masked_mean",
        predictions=baseline_predictions,
        runtime=4.6,
        parameters=68_234,
        peak_vram=94_371_840,
    )
    candidate = _packet(
        role="candidate",
        encoder="masked_tcn",
        predictions=candidate_predictions,
        runtime=33.9,
        parameters=167_435,
        peak_vram=98_566_144,
    )

    _validate_semantic_pair(candidate, baseline)
    comparison = _compare_packets(candidate, baseline, _paired_contract())
    decision = _make_decision(comparison, _decision_contract())

    delta = comparison["delta_candidate_minus_baseline"]
    assert delta["macro_f1_global_10_class"] < 0.0
    assert comparison["video_cluster_bootstrap"]["ci_high"] <= 0.0
    assert comparison["rare_group"]["recall_drop_candidate_vs_baseline"] == 0.0
    assert comparison["resource_comparison"][
        "runtime_ratio_candidate_to_baseline"
    ] == pytest.approx(33.9 / 4.6)
    assert decision["candidate_promoted"] is False
    assert decision["decision"] == "RETAIN_V1_REJECT_T1_FOR_LEGACY_T16_SEARCH"
    assert decision["transformer_action"].startswith("DEFER_TRANSFORMER")
    assert decision["applies_to_merged_reviewed_data"] is False


def test_paired_decision_promotes_only_when_all_guardrails_pass() -> None:
    baseline = _packet(
        role="baseline",
        encoder="masked_mean",
        predictions=_predictions(error_label="drink"),
        runtime=10.0,
        parameters=68_234,
        peak_vram=94_371_840,
    )
    candidate = _packet(
        role="candidate",
        encoder="masked_tcn",
        predictions=_predictions(error_label=None),
        runtime=12.0,
        parameters=167_435,
        peak_vram=98_566_144,
    )

    _validate_semantic_pair(candidate, baseline)
    comparison = _compare_packets(candidate, baseline, _paired_contract())
    decision = _make_decision(comparison, _decision_contract())

    assert comparison["video_cluster_bootstrap"]["ci_low"] > 0.0
    assert all(decision["criteria"].values())
    assert decision["candidate_promoted"] is True
    assert decision["retained_temporal_length_control"] == "T1_masked_tcn"


def test_paired_decision_fails_closed_on_lineage_or_semantic_drift() -> None:
    baseline = _packet(
        role="baseline",
        encoder="masked_mean",
        predictions=_predictions(error_label=None),
        runtime=10.0,
        parameters=68_234,
        peak_vram=94_371_840,
    )
    candidate = _packet(
        role="candidate",
        encoder="masked_tcn",
        predictions=_predictions(error_label=None),
        runtime=12.0,
        parameters=167_435,
        peak_vram=98_566_144,
    )
    drifted_pair = deepcopy(candidate)
    drifted_pair["predictions"].loc[0, "video_key"] = "different-video"

    with pytest.raises(ValueError, match="candidate video clusters"):
        _compare_packets(drifted_pair, baseline, _paired_contract())

    drifted_semantics = deepcopy(candidate)
    drifted_semantics["training_config"]["optimization"]["learning_rate"] = 0.1
    with pytest.raises(ValueError, match="fixed optimization"):
        _validate_semantic_pair(drifted_semantics, baseline)


def _predictions(*, error_label: str | None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    label_to_index = {label: index for index, label in enumerate(VALID_BEHAVIORS)}
    for cluster_index in range(10):
        for label_index, label in enumerate(VALID_BEHAVIORS):
            predicted = (
                VALID_BEHAVIORS[(label_index + 1) % len(VALID_BEHAVIORS)]
                if label == error_label
                else label
            )
            probabilities = np.full(len(VALID_BEHAVIORS), 0.2 / 9.0)
            probabilities[label_to_index[predicted]] = 0.8
            row: dict[str, object] = {
                "window_id": f"window-{cluster_index}-{label_index}",
                "temporal_unit_key": f"unit-{cluster_index}-{label_index}",
                "recording_group_id": "date-a",
                "video_key": f"video-{cluster_index}",
                "source_type": "legacy_recovered",
                "dataset_id": "legacy_recovered_16f",
                "behavior_label": label,
                "target_index": label_index,
                "predicted_label": predicted,
            }
            row.update(
                {
                    "prob_" + behavior.replace("-", "_"): float(probabilities[index])
                    for index, behavior in enumerate(VALID_BEHAVIORS)
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _packet(
    *,
    role: str,
    encoder: str,
    predictions: pd.DataFrame,
    runtime: float,
    parameters: int,
    peak_vram: int,
) -> dict[str, object]:
    baseline_config_sha = "b" * 64
    fixed_manifest = {
        field: _fixed_manifest_value(field) for field in FIXED_RUN_MANIFEST_FIELDS
    }
    training_config: dict[str, object] = {
        "training_scope": "full_development_baseline",
        "lineage_scope": "legacy-only-unreviewed-development",
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "development_metrics_authorized": True,
        "base_config": {"path": "base.json", "sha256": "a" * 64},
        "consumer_parent": {"run_id": "consumer", "sha256": "d" * 64},
        "data": {"sequence_length": 16, "sampling": "centered"},
        "model": {
            "architecture": "cached_temporal_classifier",
            "temporal_encoder_name": encoder,
            "hidden_dim": 128,
        },
        "optimization": {"learning_rate": 0.003, "seed": 20260714},
    }
    if role == "candidate":
        training_config["ablation_contract"] = {
            "changed_variable": "temporal_encoder_name",
            "reference_value": "masked_mean",
            "candidate_value": "masked_tcn",
            "single_variable_only": True,
            "reference_full_config_sha256": baseline_config_sha,
        }
    return {
        "spec": {
            "role": role,
            "training_config_sha256": (
                "c" * 64 if role == "candidate" else baseline_config_sha
            ),
        },
        "training_config": training_config,
        "run_manifest": fixed_manifest,
        "run_result": {
            "runtime_seconds": runtime,
            "execution": {"peak_reserved_bytes": peak_vram},
        },
        "preflight": {"model_parameter_count": parameters},
        "predictions": predictions,
        "recomputed_metrics": _metrics(predictions),
    }


def _fixed_manifest_value(field: str) -> object:
    values: dict[str, object] = {
        "lineage_scope": "legacy-only-unreviewed-development",
        "training_scope": "full_development_baseline",
        "selection_content_sha256": "1" * 64,
        "train_native_units": 100,
        "validation_native_units": 100,
        "outer_holdout_native_units_loaded": 0,
        "cache_hash": "2" * 64,
        "feature_index_hash": "3" * 64,
        "fold_manifest_hash": "4" * 64,
        "feature_whitelist_hash": "5" * 64,
        "fold": "native_oof_006",
        "outer_holdout_fold": "native_oof_005",
        "control_id": "V1",
        "backbone_name": "resnet18",
        "pretrained_weight_enum": "ResNet18_Weights.IMAGENET1K_V1",
        "pretrained_weight_sha256": "6" * 64,
        "resolution": 224,
        "normalization_name": "imagenet_1k_rgb",
        "image_preprocessing": "aspect_preserving_letterbox",
        "temporal_view_name": "legacy_t16_centered_matched_observed_time",
        "sequence_length": 16,
        "seed": 20260714,
        "epochs": 3,
        "batch_size": 32,
        "evaluation_batch_size": 64,
        "maximum_optimizer_steps": 345,
        "precision": "float32",
        "autocast_enabled": False,
        "oom_retry_allowed": False,
    }
    return values[field]


def _metrics(predictions: pd.DataFrame) -> dict[str, object]:
    evaluated = evaluate_predictions(
        predictions,
        y_true_col="behavior_label",
        y_pred_col="predicted_label",
        label_order=list(VALID_BEHAVIORS),
    )
    probability_columns = [
        "prob_" + label.replace("-", "_") for label in VALID_BEHAVIORS
    ]
    probabilities = predictions[probability_columns].to_numpy(dtype=float)
    targets = predictions["target_index"].to_numpy(dtype=int)
    nll = float(-np.log(probabilities[np.arange(len(targets)), targets]).mean())
    return {
        "native_units": len(predictions),
        "video_clusters": predictions["video_key"].nunique(),
        "macro_f1_global_10_class": evaluated["macro_f1"],
        "accuracy": evaluated["accuracy"],
        "nll": nll,
        "per_class": evaluated["per_class"],
    }


def _paired_contract() -> dict[str, object]:
    return {
        "unit_column": "temporal_unit_key",
        "cluster_column": "video_key",
        "true_column": "behavior_label",
        "predicted_column": "predicted_label",
        "validation_fold_id": "native_oof_006",
        "expected_native_units": 100,
        "expected_clusters": 10,
        "bootstrap_iterations": 50,
        "bootstrap_seed": 17,
        "class_order": list(VALID_BEHAVIORS),
        "rare_classes": ["fight", "social-nose", "playwithtoy", "move"],
    }


def _decision_contract() -> dict[str, object]:
    return {
        "minimum_macro_f1_gain_to_override_simpler": 0.01,
        "require_positive_cluster_ci_low": True,
        "maximum_rare_group_recall_drop": 0.1,
        "maximum_runtime_ratio_to_parent": 3.0,
        "transformer_requires_tcn_promotion": True,
    }
