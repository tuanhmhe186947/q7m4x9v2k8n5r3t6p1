from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd

import pig_behavior.classification_v2.evaluation.reviewed_q2_mixed_finalist_preflight as mixed

AGENT_RUN_ID = "c2v2_agent_mixed_finalist_fixture_v1"
HUMAN_RUN_ID = "c2v2_human_mixed_finalist_fixture_v1"


def test_mixed_finalist_requires_behavior_complete_handoff(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _fixture(tmp_path, review_stage="hidden_complete")
    _patch_p0(monkeypatch)

    result = _run(paths, tmp_path)

    assert result["valid"] is False
    assert result["short_paired_gate_authorized"] is False
    assert any("behavior_complete_handoff_required" in error for error in result["errors"])


def test_mixed_finalist_rejects_unequal_fold_bindings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _fixture(tmp_path)
    comparison = json.loads(paths["comparison"].read_text(encoding="utf-8"))
    comparison["arms"]["A128"]["bindings"]["fold_manifest_sha256"] = "wrong"
    paths["comparison"].write_text(
        json.dumps(comparison),
        encoding="utf-8",
    )
    _patch_p0(monkeypatch)

    result = _run(paths, tmp_path)

    assert result["valid"] is False
    assert "candidate_artifact_bindings_differ" in result["errors"]


def test_mixed_finalist_rejects_one_source_universe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _fixture(tmp_path, source_only="legacy_recovered")
    _patch_p0(monkeypatch)

    result = _run(paths, tmp_path)

    assert result["valid"] is False
    assert any("requires_two_sources" in error for error in result["errors"])


def test_mixed_finalist_rejects_availability_shortcut(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _fixture(tmp_path, availability_shortcut=True)
    _patch_p0(monkeypatch)

    result = _run(paths, tmp_path)

    assert result["valid"] is False
    assert "unmitigated_fixed6_source_shortcut=availability" in result["errors"]


def test_mixed_finalist_valid_pair_is_short_only_and_reports_support(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _fixture(tmp_path)
    _patch_p0(monkeypatch)

    result = _run(paths, tmp_path)

    assert result["valid"] is True
    assert result["short_paired_gate_authorized"] is True
    assert result["development_pilot_authorized"] is False
    assert result["full_oof_authorized"] is False
    assert result["checks"]["comparison_universe"]["source_labels"] == [
        "cvat_tracking_xml",
        "legacy_recovered",
    ]
    assert result["checks"]["comparison_universe"]["class_by_source_support"]
    assert result["checks"]["comparison_universe"]["missingness_support"]
    assert result["checks"]["paired_contract"][
        "attention_mechanism_claim_allowed"
    ] is False


def test_mixed_finalist_rejects_comparison_outside_agent_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _fixture(tmp_path)
    external = tmp_path / "configs" / "comparison.json"
    external.parent.mkdir(parents=True)
    external.write_text(paths["comparison"].read_text(encoding="utf-8"), encoding="utf-8")
    paths["comparison"] = external
    _patch_p0(monkeypatch)

    result = _run(paths, tmp_path)

    assert result["valid"] is False
    assert "comparison_outside_agent_derived_root" in result["errors"]


def test_mixed_finalist_writer_keeps_full_oof_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _fixture(tmp_path)
    _patch_p0(monkeypatch)

    result = _run(paths, tmp_path)
    output = (
        tmp_path
        / "outputs"
        / "classification_v2"
        / "agent_audits"
        / AGENT_RUN_ID
        / "preflight"
        / "mixed.json"
    )
    persisted = mixed.write_reviewed_q2_mixed_finalist_preflight(
        result,
        data_contract_json=paths["contract"],
        output_json=output,
        project_root=tmp_path,
        overwrite=False,
    )

    assert persisted["full_oof_authorized"] is False
    assert output.exists()


def _patch_p0(monkeypatch) -> None:
    monkeypatch.setattr(
        mixed,
        "build_reviewed_q2_p0_preflight",
        lambda *args, **kwargs: {"valid": True, "errors": []},
    )


def _run(paths: dict[str, Path], root: Path) -> dict:
    return mixed.build_reviewed_q2_mixed_finalist_preflight(
        paths["contract"],
        paths["snapshot"],
        paths["comparison"],
        paths["handoff"],
        project_root=root,
        output_json=paths["output"],
    )


def _fixture(
    root: Path,
    *,
    review_stage: str = "behavior_complete",
    source_only: str | None = None,
    availability_shortcut: bool = False,
) -> dict[str, Path]:
    agent_root = (
        root
        / "outputs"
        / "classification_v2"
        / "agent_audits"
        / AGENT_RUN_ID
    )
    human_root = (
        root
        / "human_review_workspace"
        / "classification_v2"
        / HUMAN_RUN_ID
    )
    artifact_root = agent_root / "data" / "artifacts"
    handoff = agent_root / "review_handoff" / "handoff.json"
    contract_path = agent_root / "contracts" / "data_contract.json"
    snapshot_path = agent_root / "data" / "snapshot.json"
    comparison_path = agent_root / "contracts" / "mixed_finalist.json"
    output_path = agent_root / "preflight" / "mixed_finalist.json"
    artifact_root.mkdir(parents=True)

    sources = ["legacy_recovered", "cvat_tracking_xml"]
    if source_only:
        sources = [source_only]
    units = []
    for source_index, source in enumerate(sources):
        for label_index, label in enumerate(mixed.DEFAULT_LABEL_ORDER):
            units.append(
                {
                    "temporal_unit_key": f"{source[:3]}-{label_index}",
                    "source_type": source,
                    "behavior_label": label,
                    "native_unit_valid_for_main_eval": True,
                    "recording_group_id": f"recording-{source_index}-{label_index % 2}",
                }
            )
    native_path = artifact_root / "native_units.csv"
    native = pd.DataFrame(units)
    native.to_csv(native_path, index=False)

    folds = pd.DataFrame(
        {
            "temporal_unit_key": native["temporal_unit_key"],
            "outer_fold_id": [index % 2 for index in range(len(native))],
            "oof_role": "outer_test",
        }
    )
    fold_path = artifact_root / "folds.csv"
    folds.to_csv(fold_path, index=False)

    view_rows = []
    for unit_index, unit in native.iterrows():
        for slot in range(6):
            row = {
                "temporal_view_name": mixed.FIXED_VIEW,
                "view_item_id": f"window-{unit['temporal_unit_key']}",
                "temporal_unit_key": unit["temporal_unit_key"],
                "source_type": unit["source_type"],
                "slot_index": slot,
            }
            for column in mixed.AVAILABILITY_COLUMNS:
                row[column] = not (
                    column == "actor_context_available_mask" and unit_index % 2
                )
            view_rows.append(row)
    view_path = artifact_root / "fixed6.csv"
    pd.DataFrame(view_rows).to_csv(view_path, index=False)

    whitelist_path = artifact_root / "feature_whitelist.json"
    whitelist_path.write_text(
        json.dumps(
            {
                "schema_version": "classification_v2.feature_whitelist.v1",
                "features": ["bbox_center_x_norm", "bbox_area_norm"],
            }
        ),
        encoding="utf-8",
    )
    model_input_path = artifact_root / "model_input_contract.json"
    model_input_path.write_text(
        json.dumps(_model_input_contract()),
        encoding="utf-8",
    )
    shortcut_path = artifact_root / "temporal_view_audit.json"
    shortcut_path.write_text(
        json.dumps(_shortcut_audit(availability_shortcut)),
        encoding="utf-8",
    )

    artifacts = {
        "native_temporal_unit_manifest": native_path,
        "q2_outer_fold_assignments": fold_path,
        "fixed6_observed_time_manifest": view_path,
        "feature_whitelist": whitelist_path,
        "model_input_contract": model_input_path,
        "temporal_view_audit": shortcut_path,
    }
    contract = {
        "profile": "mixed-reviewed",
        "run_id": AGENT_RUN_ID,
        "lineage_ids": {"agent_derived": AGENT_RUN_ID, "human_review": HUMAN_RUN_ID},
        "lineage_roots": {
            "agent_derived": f"outputs/classification_v2/agent_audits/{AGENT_RUN_ID}",
            "human_review": f"human_review_workspace/classification_v2/{HUMAN_RUN_ID}",
        },
        "forbidden_x_patterns": ["*label*", "*review*", "*path*", "source_type"],
        "artifacts": {
            name: {"path": path.relative_to(root).as_posix()}
            for name, path in artifacts.items()
        },
    }
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps({"snapshot": "fixture"}), encoding="utf-8")
    handoff.parent.mkdir(parents=True, exist_ok=True)
    handoff.write_text(
        json.dumps(
            {
                "run_id": HUMAN_RUN_ID,
                "review_stage": review_stage,
                "reviewer_name": "reviewer01",
                "review_code_sha": "a" * 40,
            }
        ),
        encoding="utf-8",
    )
    comparison_path.write_text(
        json.dumps(
            _comparison_contract(
                root,
                contract_path,
                snapshot_path,
                artifacts,
            )
        ),
        encoding="utf-8",
    )
    return {
        "contract": contract_path,
        "snapshot": snapshot_path,
        "comparison": comparison_path,
        "handoff": handoff,
        "output": output_path,
        "human_root": human_root,
    }


def _comparison_contract(
    root: Path,
    contract_path: Path,
    snapshot_path: Path,
    artifacts: dict[str, Path],
) -> dict:
    bindings = {
        "data_contract_sha256": _sha256(contract_path),
        "training_snapshot_sha256": _sha256(snapshot_path),
    }
    for field, artifact in mixed.ARTIFACT_BINDINGS.items():
        bindings[field] = _sha256(artifacts[artifact])
    protocol = {
        "label_order": list(mixed.DEFAULT_LABEL_ORDER),
        "preprocessing": {"normalization": "resnet18_imagenet", "crop": "letterbox"},
        "seed": 20260717,
        "loss": "event_mass_balanced_cross_entropy",
        "sampler": "frozen_native_first_selection_then_seeded_shuffle",
        "optimizer": {"name": "adamw", "learning_rate": 0.003},
        "optimizer_exposure": {"epochs": 3, "steps": 9, "batch_size": 32},
    }
    common_model = {
        "architecture": "cached_frame_feature_temporal_classifier_v1",
        "backbone_name": "resnet18",
        "input_resolution": 224,
        "hidden_dim": 128,
        "temporal_view": mixed.FIXED_VIEW,
    }
    sf_model = {
        **common_model,
        "expected_parameter_count": 68234,
        "selected_slot_indices": [2],
        "temporal_encoder_name": "masked_mean",
    }
    a_model = {
        **common_model,
        "expected_parameter_count": 68363,
        "selected_slot_indices": [0, 1, 2, 3, 4, 5],
        "temporal_encoder_name": "masked_attention",
    }
    return {
        "schema_version": mixed.CONTRACT_SCHEMA_VERSION,
        "comparison_id": "MIXED_REVIEWED_SF128_A128_SHORT_V1",
        "profile": "mixed-reviewed",
        "scientific_family": "temporal_base_finalist",
        "temporal_view": mixed.FIXED_VIEW,
        "outer_predictions_used_for_model_selection": False,
        "full_oof_requested": False,
        "evaluation_controls": {
            "paired_native_unit_evaluation": True,
            "source_stratified_evaluation": True,
            "source_matched_evaluation": True,
            "missingness_stratified_evaluation": True,
            "availability_only_control_predeclared": True,
        },
        "arms": {
            "SF128": {
                "candidate_id": "SF128",
                "role": "control",
                "bindings": bindings,
                "protocol": protocol,
                "model": sf_model,
            },
            "A128": {
                "candidate_id": "A128",
                "role": "candidate",
                "bindings": copy.deepcopy(bindings),
                "protocol": copy.deepcopy(protocol),
                "model": a_model,
            },
        },
    }


def _model_input_contract() -> dict:
    return {
        "errors": [],
        "temporal_contract": {
            "primary_view": mixed.FIXED_VIEW,
            "windows_after_harmonization": True,
        },
        "target_contract": {
            "allowed_behaviors": list(mixed.DEFAULT_LABEL_ORDER),
            "final_head_directly_supervised": True,
        },
        "inference_contract": {
            "ground_truth_only_fields_allowed": False,
            "review_fields_allowed": False,
            "missing_modalities_require_masks": True,
            "partner_selection_may_use_target_behavior": False,
        },
    }


def _shortcut_audit(availability_shortcut: bool) -> dict:
    families = {
        family: {"near_direct_source_signature": False}
        for family in ("length", "padding")
    }
    families["availability"] = {
        "near_direct_source_signature": availability_shortcut,
    }
    return {
        "schema_version": "classification_v2_temporal_shortcut_audit_v1",
        "valid": not availability_shortcut,
        "training_stop_required": availability_shortcut,
        "source_metadata_in_model_inputs": False,
        "view_reports": {
            mixed.FIXED_VIEW: {
                "source_counts": {
                    "cvat_tracking_xml": 10,
                    "legacy_recovered": 10,
                },
                "families": families,
            }
        },
        "label_shortcut_reports": {
            "fixed6_source_to_behavior": {
                "near_direct_target_signature": False,
            }
        },
        "mitigated_families": [],
        "errors": [],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
