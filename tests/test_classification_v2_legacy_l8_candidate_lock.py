from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.classification_v2.evaluation import (
    legacy_development_l8_candidate_lock as candidate,
)
from pig_behavior.classification_v2.evaluation.metrics import evaluate_predictions
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS


def test_group_report_uses_declared_confusion_groups() -> None:
    frame = _prediction_frame(100)
    frame.loc[2, "predicted_label"] = "stand"
    metrics = evaluate_predictions(
        frame,
        y_true_col="behavior_label",
        y_pred_col="predicted_label",
        label_order=list(VALID_BEHAVIORS),
    )

    report = candidate._group_report(frame, metrics["per_class"])

    assert report["rare"]["classes"] == [
        "fight",
        "social-nose",
        "playwithtoy",
        "move",
    ]
    assert report["rare"]["support"] == 40
    assert report["feeding"]["support"] == 20
    assert report["posture"]["support"] == 30
    assert report["locomotion_exploration"]["support"] == 20
    assert report["interaction"]["predicted_inside_group_rate"] < 1.0


def test_cluster_bootstrap_is_deterministic_and_bounded() -> None:
    frame = _prediction_frame(100)

    first = candidate._cluster_bootstrap(frame, iterations=1000, seed=17)
    second = candidate._cluster_bootstrap(frame, iterations=1000, seed=17)

    assert first == second
    assert first["cluster_count"] == 33
    assert first["iterations"] == 1000
    with pytest.raises(ValueError, match="at least 1000"):
        candidate._cluster_bootstrap(frame, iterations=999, seed=17)


def test_candidate_evidence_reports_calibration_and_recording_metrics() -> None:
    evidence = candidate._candidate_evidence(
        _prediction_frame(245),
        _perfect_result(),
        bootstrap_iterations=1000,
        bootstrap_seed=17,
    )

    assert evidence["native_units"] == 245
    assert evidence["video_clusters"] == 33
    assert evidence["global"]["macro_f1_global_10_class"] == 1.0
    assert evidence["global"]["nll"] == 0.0
    assert evidence["global"]["multiclass_brier"] == 0.0
    assert evidence["global"]["top_label_ece"] == 0.0
    assert evidence["recording"]["video_count"] == 33
    assert evidence["video_cluster_bootstrap"]["iterations"] == 1000


def test_candidate_evidence_rejects_probability_mass_drift() -> None:
    frame = _prediction_frame(245)
    frame.loc[0, "prob_drink"] = 0.9

    with pytest.raises(ValueError, match="probability mass drift"):
        candidate._candidate_evidence(
            frame,
            _perfect_result(),
            bootstrap_iterations=1000,
            bootstrap_seed=17,
        )


def test_candidate_evidence_rejects_argmax_drift() -> None:
    frame = _prediction_frame(245)
    frame.loc[0, "predicted_index"] = 1
    frame.loc[0, "predicted_label"] = "eat"

    with pytest.raises(ValueError, match="prediction argmax drift"):
        candidate._candidate_evidence(
            frame,
            _perfect_result(),
            bootstrap_iterations=1000,
            bootstrap_seed=17,
        )


def test_registry_and_finalist_keep_legacy_claim_boundary(tmp_path) -> None:
    run_result = tmp_path / "run_result.json"
    run_result.write_text("{}\n", encoding="utf-8")
    (tmp_path / "best_validation_checkpoint.pt").write_bytes(b"checkpoint")
    (tmp_path / "validation_native_predictions.csv").write_text(
        "prediction_order\n0\n",
        encoding="utf-8",
    )
    (tmp_path / "validation_metrics.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    l7_path = tmp_path / "l7_decision.json"
    l7_path.write_text("{}\n", encoding="utf-8")
    l7_decision = _l7_decision()
    decision_records = [
        {
            "experiment_id": "L6_RETAINED",
            "stage": "L6",
            "principal_family": "actor_only_base",
            "path": "configs/l6.json",
            "sha256": "a" * 64,
            "status": "PASS",
            "decision": "RETAIN",
            "disposition": "retained",
        },
        {
            "experiment_id": "L6_REJECTED",
            "stage": "L6",
            "principal_family": "roi_relation",
            "path": "configs/l6_roi.json",
            "sha256": "b" * 64,
            "status": "PASS",
            "decision": "REJECT",
            "disposition": "rejected",
        },
    ]

    outputs = candidate._build_outputs(
        config={
            "full_training_config": {
                "path": "configs/full.json",
                "sha256": "9" * 64,
            }
        },
        result={
            "train_native_units": 3652,
            "train_windows": 14608,
            "validation_native_units": 245,
            "validation_windows": 980,
            "parameter_sha256": "c" * 64,
            "native_prediction_content_sha256": "d" * 64,
        },
        result_path=run_result,
        run_audit={
            "result_sha256": "e" * 64,
            "run_manifest_sha256": "f" * 64,
            "artifact_manifest_sha256": "0" * 64,
        },
        l7_decision=l7_decision,
        l7_decision_path=l7_path,
        decision_records=decision_records,
        evidence=_static_evidence(),
    )

    finalist = outputs["finalist_lock"]
    assert finalist["lineage_scope"] == "legacy-only-unreviewed-development"
    assert finalist["reviewed_or_final_claim_allowed"] is False
    assert finalist["q2_claim_allowed"] is False
    assert finalist["candidate_locked"] is True
    assert finalist["checkpoint"]["size_bytes"] == len(b"checkpoint")
    assert finalist["validation_native_predictions"]["sha256"]
    assert finalist["validation_metrics"]["sha256"]
    rejected = outputs["rejected_experiments"]["experiments"]
    assert [item["experiment_id"] for item in rejected] == [
        "L6_REJECTED",
        "L7_EFFECTIVE_NUMBER_CE",
        "L7_BALANCED_SOFTMAX",
    ]
    assert all(
        item["reassess_on_merged_reviewed_data"] is True for item in rejected
    )
    registry = outputs["ablation_registry"]
    assert registry["one_principal_family_only"].all()


def test_config_validation_requires_candidate_model_and_optimization() -> None:
    payload = {
        "schema_version": candidate.CONFIG_SCHEMA,
        "lock_id": "test",
        **candidate.CLAIM_BOUNDARY,
        "implementation_source": {"path": "src/a.py", "sha256": "a" * 64},
        "full_training_config": {"path": "configs/a.json", "sha256": "b" * 64},
        "full_candidate_result": {"path": "outputs/a.json", "sha256": "c" * 64},
        "l7_decision": {"path": "outputs/l7.json", "sha256": "d" * 64},
        "decision_artifacts": [],
        "execution_guard": {
            "allowed_dirty_paths": [],
            "required_tracked_paths": [],
        },
        "evidence_contract": {
            "validation_native_units": 245,
            "validation_video_clusters": 33,
            "class_order": list(VALID_BEHAVIORS),
            "confusion_groups": {
                name: list(labels)
                for name, labels in candidate.CONFUSION_GROUPS.items()
            },
            "bootstrap_iterations": 2000,
            "bootstrap_seed": 20260716,
        },
        "interpretation_boundary": _interpretation_boundary(),
        "output": {"root_relative_path": "outputs/l8"},
    }

    with pytest.raises(ValueError, match="L8 lock config keys mismatch"):
        candidate._validate_config(payload)


def test_checked_in_config_locks_candidate_contract() -> None:
    path = Path(
        "configs/classification_v2/"
        "legacy_development_l8_candidate_lock_v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    candidate._validate_config(payload)

    payload["model"]["parameter_count"] += 1
    with pytest.raises(ValueError, match="locked model"):
        candidate._validate_config(payload)


def test_metric_comparison_accepts_only_roundtrip_noise() -> None:
    candidate._require_close(
        1.1206917660941804,
        1.1206917636652332,
        "NLL",
    )

    with pytest.raises(ValueError, match="NLL mismatch"):
        candidate._require_close(1.1208, 1.1207, "NLL")


def _prediction_frame(rows: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for index in range(rows):
        target_index = index % len(VALID_BEHAVIORS)
        label = VALID_BEHAVIORS[target_index]
        record: dict[str, object] = {
            "temporal_unit_key": f"unit-{index:03d}",
            "recording_group_id": f"recording-{index % 5:02d}",
            "video_key": f"video-{index % 33:02d}",
            "behavior_label": label,
            "target_index": target_index,
            "predicted_index": target_index,
            "predicted_label": label,
            "training_scope": "full_development_baseline",
            "lineage_scope": "legacy-only-unreviewed-development",
            "human_review_complete": False,
            "reviewed_or_final_claim_allowed": False,
            "q2_claim_allowed": False,
        }
        record.update(
            {
                "prob_" + behavior.replace("-", "_"): float(
                    behavior == label
                )
                for behavior in VALID_BEHAVIORS
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _perfect_result() -> dict[str, object]:
    return {
        "validation_metrics": {
            "macro_f1_global_10_class": 1.0,
            "nll": 0.0,
        },
        "runtime_seconds": 1.0,
        "optimizer_steps": 1,
        "best_epoch": 1,
        "execution": {
            "peak_allocated_bytes": 1,
            "peak_reserved_bytes": 2,
            "post_cleanup_allocated_bytes": 0,
            "post_cleanup_reserved_bytes": 0,
            "oom": False,
            "oom_retry_count": 0,
        },
    }


def _l7_decision() -> dict[str, object]:
    return {
        "config_path": "configs/l7.json",
        "decision_payload_sha256": "1" * 64,
        "paired_comparisons_vs_event_balanced_ce": {
            policy: {
                "delta_candidate_minus_baseline": {
                    "macro_f1_global_10_class": -0.1,
                },
                "video_cluster_bootstrap": {
                    "ci_low": -0.2,
                    "ci_high": -0.01,
                },
            }
            for policy in ("effective_number_ce", "balanced_softmax")
        },
    }


def _static_evidence() -> dict[str, object]:
    groups = {
        name: {"macro_f1": 0.5}
        for name in candidate.CONFUSION_GROUPS
    }
    return {
        "global": {
            "macro_f1_global_10_class": 0.5,
            "accuracy": 0.6,
            "nll": 1.0,
            "multiclass_brier": 0.2,
            "top_label_ece": 0.1,
        },
        "groups": groups,
        "video_clusters": 33,
        "runtime": {
            "runtime_seconds": 1.0,
            "optimizer_steps": 1,
            "peak_reserved_bytes": 2,
        },
    }


def _interpretation_boundary() -> dict[str, object]:
    return {
        "decision_scope": "legacy-only-unreviewed-development",
        "candidate_role": "bounded_legacy_development_candidate",
        "legacy_dataset_is_legacy_16f_not_merged": True,
        "legacy_rare_support_generalizes_to_merged_data": False,
        "merged_data_has_materially_more_rare_behaviors": True,
        "merged_reviewed_reassessment_required": True,
        "local_vram_is_architecture_limit": False,
        "rented_gpu_allowed_after_target_environment_gate": True,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
    }
