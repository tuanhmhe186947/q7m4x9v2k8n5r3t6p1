from __future__ import annotations

import copy
import json
from decimal import Decimal
from pathlib import Path

import pytest

from pig_behavior.classification_v2.contracts.semantic_lineage import (
    ARTIFACT_MANIFEST_REQUIRED_FIELDS,
    ARTIFACT_MANIFEST_VERSION,
    RELEASE_AUTHORIZATION_FIELDS,
    STAGE_DEPENDENCIES,
    artifact_manifest_json_schema,
    build_authority_snapshot,
    build_release_authority_preflight,
    build_semantic_bundle,
    build_semantic_domain_registry,
    build_stage_authority_registry,
    build_stage_dependency_graph,
    canonical_sha256,
    change_impact_registry,
    classify_existing_artifact,
    compute_earliest_rebuild_stage,
    compute_stage_code_hash,
    decision_carry_forward_contracts,
    deterministic_topological_order,
    evaluate_behavior_decision_carry_forward,
    evaluate_hidden_decision_carry_forward,
    file_sha256,
    historical_pre_remediation_snapshot,
    load_code_contract_mapping,
    load_scientific_contract,
    promote_artifact_transactionally,
    release_authority_json_schema,
    semantic_hash_from_json_text,
    semantic_sha256,
    transitive_descendants,
    validate_artifact_manifest,
    validate_artifact_manifest_pair,
    validate_artifact_manifest_set,
    validate_release_authority_preflight,
    validate_stage_dependency_graph,
)
from pig_behavior.classification_v2.features.motion_schema import (
    MOTION_SCHEMA_HASH,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    REPO_ROOT
    / "docs"
    / "classification_v2"
    / "scientific_contract_v1"
    / "00_pipeline_contract.yaml"
)
MAPPING_PATH = (
    REPO_ROOT
    / "docs"
    / "classification_v2"
    / "scientific_contract_v1"
    / "10_code_contract_mapping.csv"
)
HEX_A = "a" * 64
HEX_B = "b" * 64


@pytest.fixture(scope="module")
def contract() -> dict:
    return load_scientific_contract(CONTRACT_PATH)


@pytest.fixture(scope="module")
def semantic_registry(contract: dict) -> dict:
    return build_semantic_domain_registry(contract)


def _manifest(output_path: Path) -> dict:
    manifest = {
        "artifact_manifest_version": ARTIFACT_MANIFEST_VERSION,
        "artifact_id": "artifact.test",
        "artifact_class": "TEST",
        "stage_id": "stage.frame_local_primitives",
        "stage_version": "v1",
        "created_by_code_authority_sha": HEX_A,
        "stage_code_hash": HEX_A,
        "stage_semantics_hash": HEX_A,
        "stage_execution_fingerprint": HEX_A,
        "semantic_bundle_hash": HEX_A,
        "contract_manifest_hash": HEX_A,
        "input_artifact_ids": [],
        "input_artifact_fingerprints": {},
        "input_file_sha256": HEX_A,
        "output_file_sha256": file_sha256(output_path),
        "output_schema_id": "schema.test",
        "output_schema_version": "v1",
        "output_schema_hash": HEX_A,
        "row_count": 1,
        "column_count": 1,
        "feature_computation_grain": "FRAME_LOCAL_PRIMITIVES",
        "pair_scope_key": "temporal_unit_key",
        "distance_metric_ids": ["image_axis_normalized_distance_v1"],
        "distance_metric_versions": ["classification_v2.distance.axis.v1"],
        "social_identity_version": "classification_v2.social_identity.v1",
        "social_tie_break_version": "classification_v2.social_tie_break.v1",
        "roi_aggregation_version": "classification_v2.roi_aggregation.v1",
        "motion_schema_id": "schema.pig_strenet_motion_v2",
        "motion_schema_version": "classification_v2.motion_tensor.v2",
        "motion_schema_hash": MOTION_SCHEMA_HASH,
        "human_decision_authority": "NONE",
        "review_key_schema_version": "classification_v2.review_key.v1",
        "status": "VALIDATED",
        "validation_errors": [],
        "validation_warnings": [],
    }
    assert set(ARTIFACT_MANIFEST_REQUIRED_FIELDS) == set(manifest)
    return manifest


def _hidden_record(**updates: object) -> dict:
    record = {
        "review_key": "hidden:1",
        "source_key": "source",
        "dataset_key": "dataset",
        "video_key": "video",
        "object_track_key": "track",
        "frame_span_key": "1:5",
        "visual_media_sha256": HEX_A,
        "crop_authority_sha256": HEX_A,
        "full_frame_authority_sha256": HEX_A,
        "review_schema_version": "review.v1",
        "decision_schema_version": "decision.v1",
        "decision": "ACCEPT",
    }
    record.update(updates)
    return record


def _behavior_record(**updates: object) -> dict:
    record = {
        "review_unit_key": "behavior:1",
        "canonical_actor_key": "actor:1",
        "temporal_unit_key": "unit:1",
        "frame_span_key": "1:5",
        "original_label_authority_sha256": HEX_A,
        "visual_media_sha256": HEX_A,
        "review_task_semantics_hash": HEX_A,
        "review_schema_version": "review.v1",
        "decision_schema_version": "decision.v1",
        "decision": "ACCEPT",
    }
    record.update(updates)
    return record


def test_canonical_dictionary_and_json_format_invariance() -> None:
    first = {"b": 2, "a": {"x": True}}
    second = {"a": {"x": True}, "b": 2}
    assert canonical_sha256(first) == canonical_sha256(second)
    assert semantic_hash_from_json_text(json.dumps(first, indent=2)) == (
        semantic_hash_from_json_text(json.dumps(second, separators=(",", ":")))
    )


def test_canonical_path_root_and_timestamp_invariance(tmp_path: Path) -> None:
    left_root = tmp_path / "user_a" / "checkout"
    right_root = tmp_path / "user_b" / "checkout"
    left = {"authority_file": str(left_root / "src" / "authority.py")}
    right = {"authority_file": str(right_root / "src" / "authority.py")}
    assert canonical_sha256(left, repo_root=left_root) == canonical_sha256(
        right,
        repo_root=right_root,
    )
    assert semantic_sha256({"value": 1, "generated_at": "yesterday"}) == (
        semantic_sha256({"value": 1, "generated_at": "tomorrow"})
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_canonical_rejects_nonfinite(value: float) -> None:
    with pytest.raises(ValueError, match="NaN and Infinity"):
        canonical_sha256({"value": value})


def test_ordered_list_sensitive_and_unordered_collection_stable() -> None:
    assert canonical_sha256({"features": ["vx", "vy"]}) != canonical_sha256(
        {"features": ["vy", "vx"]}
    )
    assert canonical_sha256({"ids": {"a", "b"}}) == canonical_sha256(
        {"ids": {"b", "a"}}
    )
    assert canonical_sha256(
        {"ids": ["a", "b"]},
        unordered_paths=(("ids",),),
    ) == canonical_sha256(
        {"ids": ["b", "a"]},
        unordered_paths=(("ids",),),
    )


def test_decimal_negative_zero_and_repeated_execution() -> None:
    payload = {"threshold": Decimal("0.0800"), "zero": -0.0}
    hashes = {canonical_sha256(payload) for _ in range(10)}
    assert len(hashes) == 1
    assert canonical_sha256(payload) != canonical_sha256(
        {"threshold": Decimal("0.081"), "zero": 0.0}
    )


def test_recursive_generated_hash_field_is_excluded() -> None:
    payload = {"domain": "x", "semantic_bundle_hash": HEX_A}
    changed = {"domain": "x", "semantic_bundle_hash": HEX_B}
    assert semantic_sha256(payload) == semantic_sha256(changed)


def test_contract_graph_has_exact_unique_acyclic_stages(contract: dict) -> None:
    result = validate_stage_dependency_graph(contract)
    assert result["valid"], result["errors"]
    assert result["declared_stage_count"] == 17
    assert result["implemented_stage_count"] == 17
    assert len(set(result["topological_order"])) == 17
    assert result["topological_order"] == deterministic_topological_order(
        STAGE_DEPENDENCIES,
        list(STAGE_DEPENDENCIES),
    )


def test_contract_graph_rejects_unknown_and_cycle() -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        deterministic_topological_order({"a": ("missing",)}, ["a"])
    with pytest.raises(ValueError, match="cycle"):
        deterministic_topological_order(
            {"a": ("b",), "b": ("a",)},
            ["a", "b"],
        )


def test_dependency_closure_is_transitive() -> None:
    closure = transitive_descendants(
        STAGE_DEPENDENCIES,
        {"stage.frame_local_primitives"},
    )
    assert "stage.frame_local_primitives" in closure
    assert "stage.model_execution" in closure
    assert "stage.legacy_cvat_source_merge" not in closure


def test_semantic_registry_and_bundle(
    contract: dict,
    semantic_registry: dict,
) -> None:
    assert semantic_registry["semantic_domain_count"] == 17
    ids = [
        domain["semantic_domain_id"]
        for domain in semantic_registry["semantic_domains"]
    ]
    assert len(ids) == len(set(ids))
    bundle = build_semantic_bundle(semantic_registry)
    assert len(bundle["semantic_domain_hashes"]) == 17
    assert len(bundle["semantic_bundle_hash"]) == 64
    modified = copy.deepcopy(semantic_registry)
    modified["semantic_domains"][0]["canonical_hash"] = HEX_A
    assert build_semantic_bundle(modified)["semantic_bundle_hash"] != (
        bundle["semantic_bundle_hash"]
    )
    assert build_stage_dependency_graph(contract)["valid"]


def test_stage_code_hash_detects_only_mapped_production_blob(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src" / "producer.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    test_file = tmp_path / "tests" / "test_producer.py"
    test_file.parent.mkdir()
    test_file.write_text("assert True\n", encoding="utf-8")
    mapping = [
        {
            "contract_item_type": "stage",
            "contract_item_id": "stage.frame_local_primitives",
            "source_file": "src/producer.py",
        }
    ]
    original = compute_stage_code_hash(
        tmp_path,
        "stage.frame_local_primitives",
        mapping,
    )
    test_file.write_text("assert False\n", encoding="utf-8")
    assert compute_stage_code_hash(
        tmp_path,
        "stage.frame_local_primitives",
        mapping,
    ) == original
    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert compute_stage_code_hash(
        tmp_path,
        "stage.frame_local_primitives",
        mapping,
    ) != original


def test_current_authority_computes_frame_local_rebuild(
    contract: dict,
    semantic_registry: dict,
) -> None:
    mapping = load_code_contract_mapping(MAPPING_PATH)
    stage_authority = build_stage_authority_registry(
        repo_root=REPO_ROOT,
        contract=contract,
        mapping_rows=mapping,
        semantic_registry=semantic_registry,
    )
    current = build_authority_snapshot(
        semantic_registry=semantic_registry,
        stage_authority_registry=stage_authority,
    )
    previous = historical_pre_remediation_snapshot(current)
    plan = compute_earliest_rebuild_stage(
        previous,
        current,
        [],
        build_stage_dependency_graph(contract),
        impact_registry=change_impact_registry(semantic_registry),
    )
    assert plan["rebuild_required"] is True
    assert plan["earliest_stage_id"] == "stage.frame_local_primitives"
    assert "stage.model_execution" in plan["invalidated_stages"]
    assert plan["human_decision_revalidation_required"] is True


@pytest.mark.parametrize(
    ("changed_domain", "expected_stage"),
    [
        ("semantic.source_parsing_selection", "stage.legacy_cvat_source_merge"),
        ("semantic.frame_local_geometry", "stage.frame_local_primitives"),
        ("semantic.hidden_selection", "stage.hidden_review_design"),
        ("semantic.temporal_harmonization", "stage.temporal_harmonization"),
        ("semantic.native_temporal_pairs", "stage.native_review_evidence"),
        (
            "semantic.behavior_review_units",
            "stage.behavior_review_unit_construction",
        ),
        ("semantic.final_view_windows", "stage.train_ready_export"),
        ("semantic.model_input_export", "stage.tensor_export"),
        ("semantic.split_leakage", "stage.model_input"),
    ],
)
def test_change_registry_derives_expected_earliest_stage(
    contract: dict,
    semantic_registry: dict,
    changed_domain: str,
    expected_stage: str,
) -> None:
    current_hashes = {
        domain["semantic_domain_id"]: domain["canonical_hash"]
        for domain in semantic_registry["semantic_domains"]
    }
    previous_hashes = dict(current_hashes)
    previous_hashes[changed_domain] = HEX_A
    from pig_behavior.classification_v2.contracts.semantic_lineage import (
        AuthoritySnapshot,
    )

    current = AuthoritySnapshot(current_hashes, {}, {})
    previous = AuthoritySnapshot(previous_hashes, {}, {})
    plan = compute_earliest_rebuild_stage(
        previous,
        current,
        [],
        build_stage_dependency_graph(contract),
        impact_registry=change_impact_registry(semantic_registry),
    )
    assert plan["earliest_stage_id"] == expected_stage


def test_docs_only_authority_produces_no_invalidation(
    contract: dict,
    semantic_registry: dict,
) -> None:
    from pig_behavior.classification_v2.contracts.semantic_lineage import (
        AuthoritySnapshot,
    )

    snapshot = AuthoritySnapshot({}, {}, {})
    plan = compute_earliest_rebuild_stage(
        snapshot,
        snapshot,
        [],
        build_stage_dependency_graph(contract),
        impact_registry=change_impact_registry(semantic_registry),
    )
    assert plan["rebuild_required"] is False
    assert plan["stale_artifact_ids"] == []


def test_multiple_changes_choose_earliest(
    contract: dict,
    semantic_registry: dict,
) -> None:
    from pig_behavior.classification_v2.contracts.semantic_lineage import (
        AuthoritySnapshot,
    )

    current_hashes = {
        domain["semantic_domain_id"]: domain["canonical_hash"]
        for domain in semantic_registry["semantic_domains"]
    }
    previous_hashes = dict(current_hashes)
    previous_hashes["semantic.final_view_windows"] = HEX_A
    previous_hashes["semantic.frame_local_geometry"] = HEX_B
    plan = compute_earliest_rebuild_stage(
        AuthoritySnapshot(previous_hashes, {}, {}),
        AuthoritySnapshot(current_hashes, {}, {}),
        [],
        build_stage_dependency_graph(contract),
        impact_registry=change_impact_registry(semantic_registry),
    )
    assert plan["earliest_stage_id"] == "stage.frame_local_primitives"


def test_artifact_and_release_schemas_are_fail_closed() -> None:
    artifact_schema = artifact_manifest_json_schema()
    release_schema = release_authority_json_schema()
    assert artifact_schema["additionalProperties"] is False
    assert release_schema["additionalProperties"] is False
    assert set(RELEASE_AUTHORIZATION_FIELDS).issubset(
        release_schema["required"]
    )


def test_artifact_manifest_validates_hash_and_schema(tmp_path: Path) -> None:
    output = tmp_path / "artifact.csv"
    output.write_text("value\n1\n", encoding="utf-8")
    manifest = _manifest(output)
    result = validate_artifact_manifest(
        manifest,
        output_path=output,
        expected_schema=("schema.test", "v1", HEX_A),
    )
    assert result["valid"], result["errors"]
    manifest["output_file_sha256"] = HEX_B
    assert "OUTPUT_HASH_MISMATCH" in validate_artifact_manifest(
        manifest,
        output_path=output,
    )["errors"]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda value: value.pop("stage_code_hash"), "MISSING_REQUIRED_FIELDS"),
        (
            lambda value: value.update(stage_code_hash=""),
            "BLANK_REQUIRED_HASH:stage_code_hash",
        ),
        (
            lambda value: value.update(input_artifact_ids=["artifact.test"]),
            "SELF_DEPENDENCY",
        ),
        (
            lambda value: value.update(output_schema_version="wrong"),
            "OUTPUT_SCHEMA_MISMATCH",
        ),
    ],
)
def test_artifact_manifest_negative_gates(
    tmp_path: Path,
    mutation,
    expected: str,
) -> None:
    output = tmp_path / "artifact.csv"
    output.write_text("value\n1\n", encoding="utf-8")
    manifest = _manifest(output)
    mutation(manifest)
    result = validate_artifact_manifest(
        manifest,
        output_path=output,
        expected_schema=("schema.test", "v1", HEX_A),
    )
    assert any(error.startswith(expected) for error in result["errors"])


def test_duplicate_artifact_id_fails(tmp_path: Path) -> None:
    output = tmp_path / "artifact.csv"
    output.write_text("value\n1\n", encoding="utf-8")
    manifest = _manifest(output)
    result = validate_artifact_manifest_set([manifest, dict(manifest)])
    assert "DUPLICATE_ARTIFACT_ID:artifact.test" in result["errors"]


def test_output_manifest_pair_requires_both(tmp_path: Path) -> None:
    output = tmp_path / "artifact.csv"
    manifest_path = tmp_path / "artifact.csv.manifest.json"
    output.write_text("value\n1\n", encoding="utf-8")
    assert validate_artifact_manifest_pair(output, manifest_path)["valid"] is False
    manifest = _manifest(output)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_artifact_manifest_pair(output, manifest_path)["valid"]
    output.unlink()
    assert validate_artifact_manifest_pair(output, manifest_path)["valid"] is False


def test_transactional_promotion_success(tmp_path: Path) -> None:
    staging = tmp_path / "staging" / "artifact.csv"
    staging.parent.mkdir()
    staging.write_text("value\n1\n", encoding="utf-8")
    final = tmp_path / "official" / "artifact.csv"
    manifest_path = final.with_suffix(".csv.manifest.json")
    result = promote_artifact_transactionally(
        staging_output=staging,
        final_output=final,
        final_manifest=manifest_path,
        candidate_manifest=_manifest(staging),
    )
    assert result["promoted"] is True
    assert validate_artifact_manifest_pair(final, manifest_path)["valid"]


def test_transactional_promotion_failure_rolls_back(tmp_path: Path) -> None:
    staging = tmp_path / "staging" / "artifact.csv"
    staging.parent.mkdir()
    staging.write_text("value\n1\n", encoding="utf-8")
    final = tmp_path / "official" / "artifact.csv"
    manifest_path = final.with_suffix(".csv.manifest.json")

    def interrupt() -> None:
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        promote_artifact_transactionally(
            staging_output=staging,
            final_output=final,
            final_manifest=manifest_path,
            candidate_manifest=_manifest(staging),
            before_manifest_promotion=interrupt,
        )
    assert staging.is_file()
    assert not final.exists()
    assert not manifest_path.exists()


def test_inventory_classifies_audits_and_missing_manifests(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "agent_audits" / "phase3" / "evidence.csv"
    audit.parent.mkdir(parents=True)
    audit.write_text("x\n", encoding="utf-8")
    official = tmp_path / "official.csv"
    official.write_text("x\n", encoding="utf-8")
    assert classify_existing_artifact(
        audit,
        current_stage_semantics={},
        current_stage_code={},
    )["classification"] == "NON_OFFICIAL_AUDIT"
    assert classify_existing_artifact(
        official,
        current_stage_semantics={},
        current_stage_code={},
    )["classification"] == "MISSING_MANIFEST"
    pre_motion = tmp_path / "old_pig_strenet" / "evidence.csv"
    pre_motion.parent.mkdir()
    pre_motion.write_text("x\n", encoding="utf-8")
    diagnostic = classify_existing_artifact(
        pre_motion,
        current_stage_semantics={},
        current_stage_code={},
    )
    assert diagnostic["classification"] == "FAILED_DIAGNOSTIC"
    assert "FAILED_DIAGNOSTIC_PRE_MOTION_FIX" in diagnostic["reason_codes"]
    assert "NOT_REUSABLE" in diagnostic["reason_codes"]
    assert "NOT_REVIEW_EVIDENCE" in diagnostic["reason_codes"]


def test_hidden_exact_carry_forward_and_visual_revalidation() -> None:
    exact = evaluate_hidden_decision_carry_forward(
        [_hidden_record()],
        [_hidden_record()],
    )
    assert exact[0]["classification"] == "EXACT_CARRY_FORWARD_CANDIDATE"
    changed = evaluate_hidden_decision_carry_forward(
        [_hidden_record()],
        [_hidden_record(visual_media_sha256=HEX_B)],
    )
    assert changed[0]["classification"] == "REQUIRES_HUMAN_REVALIDATION"
    assert changed[0]["positional_matching_used"] is False


def test_hidden_old_new_conflict_and_invalid_schema() -> None:
    old_only = evaluate_hidden_decision_carry_forward(
        [_hidden_record()],
        [],
    )
    assert old_only[0]["classification"] == "OLD_ONLY_AUDIT_EVIDENCE"
    new_only = evaluate_hidden_decision_carry_forward(
        [],
        [_hidden_record()],
    )
    assert new_only[0]["classification"] == "NEW_ONLY_REQUIRES_REVIEW"
    conflict = evaluate_hidden_decision_carry_forward(
        [_hidden_record(), _hidden_record(decision="REJECT")],
        [_hidden_record()],
    )
    assert conflict[0]["classification"] == "CONFLICT"
    invalid = evaluate_hidden_decision_carry_forward(
        [_hidden_record(review_schema_version="")],
        [_hidden_record()],
    )
    assert invalid[0]["classification"] == "INVALID_DECISION_SCHEMA"


def test_behavior_stable_key_not_position_or_pig_id() -> None:
    exact = evaluate_behavior_decision_carry_forward(
        [_behavior_record()],
        [_behavior_record()],
    )
    assert exact[0]["classification"] == "EXACT_CARRY_FORWARD_CANDIDATE"
    changed_key = evaluate_behavior_decision_carry_forward(
        [_behavior_record()],
        [_behavior_record(review_unit_key="behavior:2")],
    )
    assert {item["classification"] for item in changed_key} == {
        "OLD_ONLY_AUDIT_EVIDENCE",
        "NEW_ONLY_REQUIRES_REVIEW",
    }
    contract = decision_carry_forward_contracts()
    assert "position" in contract["forbidden_matching"]
    assert contract["new_only_auto_accepted"] is False


@pytest.mark.parametrize(
    "extra",
    [
        {"stopped_lineage": True},
        {"failed_diagnostic": True},
        {"non_official_audit_only": True},
    ],
)
def test_release_authority_negative_lineage_gates(extra: dict) -> None:
    preflight = build_release_authority_preflight(
        artifact_gate_results={"manifest": True},
        phase4_human_signoff=True,
        manual_authorizations={
            field: True for field in RELEASE_AUTHORIZATION_FIELDS
        },
        **extra,
    )
    assert not any(preflight[field] for field in RELEASE_AUTHORIZATION_FIELDS)
    assert validate_release_authority_preflight(preflight)["valid"]


def test_release_authority_unsigned_and_missing_prerequisite_fail_closed() -> None:
    preflight = build_release_authority_preflight(
        artifact_gate_results={"manifest": False},
        phase4_human_signoff=False,
        manual_authorizations={
            field: True for field in RELEASE_AUTHORIZATION_FIELDS
        },
    )
    assert not any(preflight[field] for field in RELEASE_AUTHORIZATION_FIELDS)
    assert "PHASE4_HUMAN_SIGNOFF_MISSING" in preflight["prerequisite_errors"]
    assert validate_release_authority_preflight(preflight)["valid"]


def test_phase2_motion_schema_hash_is_frozen() -> None:
    assert MOTION_SCHEMA_HASH == (
        "ec0c511b5f5198240492be49c0492e543c9e38eb4a4ff446259b958c2a59963b"
    )
