from __future__ import annotations

import hashlib
import json
from pathlib import Path

AUTHORITY = Path(
    "docs/classification_v2/corrected_pooled_route_20260806/"
    "next_phase_20260806_r2/"
    "s1_control_and_pre_s1_calibration_authority.json"
)
LAUNCH_PACKET = AUTHORITY.with_name("s1_stage1_l4_launch_packet_20260810.json")


def _load() -> dict[str, object]:
    return json.loads(AUTHORITY.read_text(encoding="utf-8"))


def test_s1_budget_controls_steps_checkpoint_and_seed_contract() -> None:
    authority = _load()

    assert authority["status"] == "POST_CALIBRATION_HORIZON_FROZEN_STAGE1_CPU_PREFLIGHT_PENDING"
    budget = authority["budget_scope"]
    assert budget["stage6_autoresearch_max_trials"] == 24
    assert budget["behavior_only_s1_global_gpu_hour_ceiling"] == 48
    assert budget["registered_stage_maxima_preserved"]["stage_1_total_max"] == 8

    controls = authority["fixed_stage_1_to_4_controls"]
    assert controls["authority_classification"] == "NEW_S1_CONTROL_DECISION"
    assert {
        key: controls[key]
        for key in (
            "optimizer",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "precision",
            "scheduler",
        )
    } == {
        "optimizer": "AdamW",
        "learning_rate": 0.003,
        "weight_decay": 0,
        "batch_size": 16,
        "precision": "FP32",
        "scheduler": "none",
    }

    policy = authority["matched_training_policy"]
    assert policy["primary_training_population"] == "ALL_VALID_SINGLE_LABEL_WINDOWS"
    assert policy["training_budget_unit"] == "OPTIMIZER_STEPS"
    assert policy["fixed_training_steps"] == 4164
    assert policy["matched_stage_1_to_4_max_steps"] == 4164
    assert policy["fixed_epochs_allowed"] is False

    checkpoint = authority["checkpoint_and_seed_policy"]
    assert checkpoint["early_stopping"] == "DISABLED"
    assert checkpoint["scientific_ranking_checkpoint"] == "FIXED_STEP_ENDPOINT"
    assert checkpoint["initial_screening_seed"] == 20260804
    assert checkpoint["confirmation_seeds"] == [20260805, 20260806]


def test_s1_inner_population_weights_and_evaluator_are_bound() -> None:
    authority = _load()
    derived = authority["derived_population"]

    assert derived["allowed_roles"] == ["train", "validation"]
    assert derived["eligibility_artifact"]["outer_rows_written"] == 0
    assert derived["mixed_label_audit_only_windows"]["primary_eligible_rows"] == 0
    assert derived["mixed_label_audit_only_windows"]["training_rows"] == 0
    assert derived["common_t6_t8_t12_t16_cohort"]["native_units"] == 27378
    for view in ("T6", "T8", "T12", "T16"):
        item = derived["per_view"][view]
        assert item["eligible_train_windows"] > 0
        assert item["eligible_validation_windows"] > 0
        assert item["mixed_label_training_rows"] == 0
        assert len(item["event_weight_artifact"]["sha256"]) == 64
        assert len(item["event_weight_artifact"]["check_sha256"]) == 64

    roles = authority["inner_role_binding"]
    assert roles == {
        "inner_fold": "FOLD_3",
        "train_role": "train",
        "validation_role": "validation",
        "validation_inner_fold_id": 2,
        "binding_status": "PASS",
        "forbidden_roles": ["test", "outer", "q2_outer_00"],
    }
    evaluator = authority["primary_evaluation"]
    assert evaluator["required_path"] == "EXPLICIT_WINDOW_TO_NATIVE_COLLAPSE"
    assert evaluator["composite_key_direct_primary_metric_allowed"] is False
    assert evaluator["native_prediction_coverage"] == "COMPLETE"


def test_pre_s1_calibration_is_nonclaim_grade_and_step_bounded() -> None:
    authority = _load()
    calibration = authority["pre_s1_calibration"]

    assert calibration["status"] == "COMPLETED_VALID_HORIZON_FROZEN"
    assert calibration["reference_configuration"]["id"] == "B1_ACTOR_T6_SEQUENCE"
    assert calibration["temporal_view"] == "T6"
    assert calibration["seed"] == 20260804
    assert calibration["t6_steps_per_pass"] == 2082
    assert calibration["max_steps"] == 6246
    assert calibration["event_snapshots_at_steps"] == [2082, 4164, 6246]
    for field in (
        "scientific_trial",
        "claim_grade_result",
        "model_promotion_allowed",
        "temporal_selection_allowed",
        "feature_selection_allowed",
        "outer_access_allowed",
    ):
        assert calibration[field] is False

    outer = authority["outer_active_refusal_contract"]
    assert outer["outer_feedback"] == "FORBIDDEN"
    assert "q2_outer_00" in outer["must_reject_before_payload_open"]
    assert authority["registered_funnel_preservation"]["h5_entry_gate"] == (
        "PRESERVED_STAGE_4_ONLY"
    )
    assert authority["registered_funnel_preservation"]["c2_access"] == "BLOCKED"


def test_user_approved_horizon_and_stage1_temporal_scope_are_bound() -> None:
    authority = _load()

    decision = authority["calibration_decision_authority"]
    assert decision["selected_horizon"] == 4164
    assert decision["selection_rule"] == "REGISTERED_TRAJECTORY_REVIEW"
    assert len(decision["sha256"]) == 64

    stage1 = authority["stage_1_temporal_screening"]
    assert stage1["status"] == "CPU_PREFLIGHT_AUTHORIZED_GPU_EXECUTION_NOT_AUTHORIZED"
    assert stage1["run_kind"] == "S1_STAGE1_TEMPORAL_SCREENING"
    assert stage1["temporal_views"] == ["T6", "T8", "T12", "T16"]
    assert stage1["max_steps"] == 4164
    assert stage1["evaluation_steps"] == [4164]
    assert stage1["common_cohort"]["native_units"] == 27378
    assert stage1["outer_access_allowed"] is False
    assert stage1["gpu_execution_authorized"] is False


def test_future_l4_launch_packet_is_exact_but_not_an_execution_authorization() -> None:
    packet = json.loads(LAUNCH_PACKET.read_text(encoding="utf-8"))

    assert packet["status"] == "CPU_PREFLIGHT_PASS_FUTURE_COMMANDS_ONLY"
    boundary = packet["authority_boundary"]
    assert boundary["stage1_gpu_execution_authorized"] is False
    assert boundary["not_a_launch_authorization"] is True
    assert boundary["required_future_execution_authorization"] == (
        "EXTERNAL_SINGLE_USE_PER_ARM_PERMIT"
    )
    assert boundary["required_hardware"] == {
        "gpu_count": 1,
        "gpu_name": "NVIDIA L4",
    }
    assert packet["execution_permit_contract"] == {
        "issuer": (
            "scripts/classification_v2/04_baselines_smokes/"
            "classification_v2_authorize_s1_stage1_execution.py"
        ),
        "schema_version": "classification_v2.s1_stage1_execution_permit.v1",
        "per_arm": True,
        "single_use": True,
        "authority_gpu_flag_must_remain_false": True,
    }
    assert packet["bound_authorities"]["s1_authority_sha256"] == hashlib.sha256(
        AUTHORITY.read_bytes()
    ).hexdigest()

    prefix = packet["invocation"]["command_argv_prefix"]
    assert prefix[:6] == ["uv", "run", "--frozen", "--extra", "pt", "python"]
    assert "--max-steps" not in prefix
    assert "--execution-authorization" not in prefix
    assert prefix[prefix.index("--binding-bundle") + 1] == (
        "${S1_STAGE1_RGB_BINDING_BUNDLE}"
    )
    assert set(packet["arms"]) == {"T6", "T8", "T12", "T16"}
    for view, arm in packet["arms"].items():
        suffix = arm["command_argv_suffix"]
        assert suffix[:2] == ["--view", view]
        assert arm["execution_permit_environment_variable"] == (
            f"S1_STAGE1_{view}_EXECUTION_PERMIT"
        )
        assert suffix[2:4] == [
            "--execution-permit",
            f"${{S1_STAGE1_{view}_EXECUTION_PERMIT}}",
        ]
        assert suffix[4:6] == [
            "--data-bindings",
            f"${{S1_STAGE1_{view}_DATA_BINDINGS}}",
        ]
        assert suffix[6:] == ["--trial-id", arm["trial_id"]]
        assert arm["trial_id"].endswith("steps4164")
