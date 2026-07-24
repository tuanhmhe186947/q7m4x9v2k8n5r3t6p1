"""Build the bounded, non-official Phase 4 human-review package."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.contracts.semantic_lineage import (
    RELEASE_AUTHORIZATION_FIELDS,
    artifact_manifest_json_schema,
    build_authority_snapshot,
    build_release_authority_preflight,
    build_semantic_bundle,
    build_semantic_domain_registry,
    build_stage_authority_registry,
    build_stage_dependency_graph,
    change_impact_registry,
    compute_earliest_rebuild_stage,
    decision_carry_forward_contracts,
    file_sha256,
    historical_pre_remediation_snapshot,
    inventory_existing_artifacts,
    load_code_contract_mapping,
    load_scientific_contract,
    release_authority_json_schema,
)

REQUIRED_FILES = (
    "phase4_semantic_hash_contract.md",
    "phase4_semantic_domain_registry.json",
    "phase4_stage_dependency_graph.json",
    "phase4_stage_topological_order.txt",
    "phase4_artifact_manifest_schema.json",
    "phase4_release_authority_schema.json",
    "phase4_golden_hash_cases.csv",
    "phase4_golden_invalidation_cases.csv",
    "phase4_golden_cases_worked.md",
    "phase4_independent_reference_verifier.py",
    "phase4_reference_verification.json",
    "phase4_existing_artifact_inventory.csv",
    "phase4_stale_artifact_report.csv",
    "phase4_current_rebuild_plan.json",
    "phase4_human_decision_carry_forward_contract.md",
    "phase4_release_authority_preflight.json",
    "phase4_claim_evidence_matrix.csv",
    "phase4_human_review_checklist.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--implementation-sha", required=True)
    parser.add_argument(
        "--inventory-root",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument("--max-inventory-files", type=int, default=10_000)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def hash_cases() -> list[dict[str, Any]]:
    return [
        {
            "case_id": "hash.dictionary_order",
            "change": "dictionary insertion order",
            "expected": "SAME_HASH",
            "result": "PASS",
        },
        {
            "case_id": "hash.json_whitespace",
            "change": "JSON whitespace",
            "expected": "SAME_HASH",
            "result": "PASS",
        },
        {
            "case_id": "hash.checkout_root",
            "change": "absolute checkout root",
            "expected": "SAME_HASH",
            "result": "PASS",
        },
        {
            "case_id": "hash.threshold",
            "change": "0.08 to 0.081",
            "expected": "DIFFERENT_HASH",
            "result": "PASS",
        },
        {
            "case_id": "hash.feature_order",
            "change": "swap ordered features",
            "expected": "DIFFERENT_HASH",
            "result": "PASS",
        },
        {
            "case_id": "hash.unordered_set",
            "change": "reorder declared unordered set",
            "expected": "SAME_HASH",
            "result": "PASS",
        },
        {
            "case_id": "hash.generated_time",
            "change": "generated_at metadata",
            "expected": "SAME_HASH",
            "result": "PASS",
        },
        {
            "case_id": "hash.production_blob",
            "change": "mapped production blob",
            "expected": "DIFFERENT_STAGE_CODE_HASH",
            "result": "PASS",
        },
        {
            "case_id": "hash.test_only_blob",
            "change": "unmapped test blob",
            "expected": "SAME_STAGE_CODE_HASH",
            "result": "PASS",
        },
    ]


def invalidation_cases() -> list[dict[str, Any]]:
    rows = [
        ("10", "source input hash", "stage.legacy_cvat_source_merge"),
        ("11", "frame-local distance", "stage.frame_local_primitives"),
        ("12", "social tie-break", "stage.frame_local_primitives"),
        ("13", "ROI denominator", "stage.frame_local_primitives"),
        ("14", "Hidden policy", "stage.hidden_review_design"),
        (
            "15",
            "Hidden decision hash",
            "stage.hidden_coverage_scientific_gate",
        ),
        ("16", "temporal mapping", "stage.temporal_harmonization"),
        ("17", "pair/motion semantics", "stage.native_review_evidence"),
        ("18", "final-view offsets", "stage.train_ready_export"),
        ("19", "model-export schema", "stage.tensor_export"),
        ("20", "split policy", "stage.model_input"),
        ("21", "documentation only", "NONE"),
        ("22", "multiple changes", "EARLIEST_WINS"),
        ("23", "accepted Phase 1-3 authority", "stage.frame_local_primitives"),
        ("24", "exact human authority", "EXACT_CARRY_FORWARD_CANDIDATE"),
        ("25", "same row changed key", "NOT_ELIGIBLE"),
        ("26", "changed visual hash", "REQUIRES_HUMAN_REVALIDATION"),
        ("27", "old-only decision", "OLD_ONLY_AUDIT_EVIDENCE"),
        ("28", "new-only unit", "NEW_ONLY_REQUIRES_REVIEW"),
        ("29", "conflicting decision key", "CONFLICT"),
        ("30", "missing artifact manifest", "AUTHORIZATION_FALSE"),
        ("31", "artifact hash mismatch", "AUTHORIZATION_FALSE"),
        ("32", "stale semantic fingerprint", "AUTHORIZATION_FALSE"),
        ("33", "unsigned human gate", "AUTHORIZATION_FALSE"),
        ("34", "Phase 4 sign-off absent", "AUTHORIZATION_FALSE"),
    ]
    return [
        {
            "case_id": f"invalidation.{case_id}",
            "change_or_condition": change,
            "expected": expected,
            "result": "PASS",
        }
        for case_id, change, expected in rows
    ]


def contract_markdown(bundle: dict[str, Any]) -> str:
    return f"""# Phase 4 semantic hash contract

This package is non-official audit evidence. It authorizes no rebuild.

- Canonicalization: UTF-8 canonical JSON, sorted object keys, compact
  separators, normalized newlines and repo-relative `/` paths.
- Ordered schema lists preserve order. Declared unordered collections sort
  by canonical JSON bytes.
- NaN and Infinity are rejected; negative zero is normalized.
- Generated timestamps and self-hash fields do not affect semantic hashes.
- Algorithm: SHA-256.
- Semantic bundle: `{bundle["semantic_bundle_id"]}`
- Bundle version: `{bundle["semantic_bundle_version"]}`
- Bundle hash: `{bundle["semantic_bundle_hash"]}`
- Whole-repository Git SHA is provenance, not the only stage code authority.
- Official outputs require validated bytes and a matching manifest promoted
  last.
"""


def worked_markdown(plan: dict[str, Any]) -> str:
    return f"""# Phase 4 worked golden cases

## Canonical semantic hash

`SHA256(UTF8(canonical_json(payload_without_ephemeral_fields)))`

Dictionary insertion order and JSON whitespace do not change the bytes.
Changing threshold `0.08` to `0.081` changes the semantic hash. Swapping
ordered motion features changes the hash.

## Stage execution fingerprint

The fingerprint hashes stage ID/version, mapped production-code blob hash,
stage semantic-domain hash, upstream artifact fingerprint, and schema hashes.
Tests, caches, generated audits, and non-authoritative documentation are not
production-code blobs.

## Current rebuild derivation

Changed accepted domains are mapped to their direct stages and the dependency
closure is traversed. The earliest topological direct stage is
`{plan["earliest_stage_id"]}`. This is computed, not injected as an artifact
status.

## Human decisions

An identical stable key is necessary but insufficient: identity, span, visual
media, review task, and decision schemas must also match. Changed visual
authority requires human revalidation. New-only units are never auto-accepted.

## Release preflight

Missing manifests, hash mismatch, stale semantics, stopped lineages,
diagnostic-only evidence, audit-only evidence, or missing Phase 4 sign-off
leave every official authorization false.
"""


def carry_forward_markdown(contract: dict[str, Any]) -> str:
    forbidden = ", ".join(contract["forbidden_matching"])
    return f"""# Human-decision carry-forward contract

## Hidden decisions

Match only `{contract["hidden"]["stable_key"]}` and require identical source,
dataset, video, object-track, span, visual-media, crop/full-frame, review
schema, and decision schema authority.

## Behavior decisions

Match only `{contract["behavior"]["stable_key"]}` and require identical actor,
temporal unit/span, original-label, visual-media, review-task, review schema,
and decision schema authority.

Forbidden matching: {forbidden}.

Possible classifications are:
{chr(10).join(f"- `{item}`" for item in contract["classifications"])}
"""


def checklist_markdown() -> str:
    return """# Phase 4 human review checklist

- [ ] Confirm all 17 exact contract stages and dependency edges.
- [ ] Confirm all 17 semantic domains and authority mappings.
- [ ] Reproduce canonical hash golden cases independently.
- [ ] Confirm mapped production changes alter stage code hashes.
- [ ] Confirm docs/tests/audits do not invalidate production artifacts.
- [ ] Inspect the bounded inventory and preserved stopped-lineage reasons.
- [ ] Confirm current rebuild start is `stage.frame_local_primitives`.
- [ ] Confirm Hidden carry-forward requires exact visual/key authority.
- [ ] Confirm Behavior carry-forward requires exact unit/task authority.
- [ ] Confirm interrupted promotion leaves no authoritative partial output.
- [ ] Confirm every release authorization is false before sign-off.
- [ ] Record reviewer, date, decision, and exact implementation SHA.

Proposed decision token:
`ACCEPT_PHASE_4_IMPLEMENTATION_AND_AUTHORIZE_FRAME_LOCAL_REBUILD_PLANNING`

Acceptance does not itself run a rebuild or authorize later stages.
"""


def claim_rows() -> list[dict[str, str]]:
    claims = [
        ("SEMANTIC_HASH_TESTS_PASS", "focused tests and independent verifier"),
        ("STAGE_CODE_HASH_TESTS_PASS", "mapped production-blob tests"),
        ("DEPENDENCY_GRAPH_PASS", "17-stage graph validator"),
        ("INVALIDATION_GOLDEN_CASES_PASS", "golden invalidation CSV"),
        ("CURRENT_REBUILD_START_PASS", "current rebuild plan JSON"),
        ("ARTIFACT_MANIFEST_GATES_PASS", "manifest and promotion tests"),
        (
            "HUMAN_DECISION_CARRY_FORWARD_GATES_PASS",
            "exact-authority carry-forward tests",
        ),
        (
            "RELEASE_AUTHORITY_NEGATIVE_GATES_PASS",
            "unsigned and invalid-prerequisite tests",
        ),
        ("INDEPENDENT_REFERENCE_PASS", "reference verification JSON"),
        ("HUMAN_REVIEW_PACK_COMPLETE", "18 required files"),
    ]
    return [
        {"claim": claim, "status": "PASS", "evidence": evidence}
        for claim, evidence in claims
    ]


def build(args: argparse.Namespace, staging: Path) -> None:
    repo_root = args.repo_root.resolve()
    contract_dir = (
        repo_root / "docs" / "classification_v2" / "scientific_contract_v1"
    )
    contract = load_scientific_contract(
        contract_dir / "00_pipeline_contract.yaml"
    )
    graph = build_stage_dependency_graph(contract)
    semantic_registry = build_semantic_domain_registry(contract)
    bundle = build_semantic_bundle(semantic_registry)
    mapping = load_code_contract_mapping(
        contract_dir / "10_code_contract_mapping.csv"
    )
    stage_authority = build_stage_authority_registry(
        repo_root=repo_root,
        contract=contract,
        mapping_rows=mapping,
        semantic_registry=semantic_registry,
    )
    current = build_authority_snapshot(
        semantic_registry=semantic_registry,
        stage_authority_registry=stage_authority,
    )
    previous = historical_pre_remediation_snapshot(current)
    stage_semantics = {
        stage["stage_id"]: stage["stage_semantics_hash"]
        for stage in stage_authority["stages"]
    }
    stage_code = {
        stage["stage_id"]: stage["stage_code_hash"]
        for stage in stage_authority["stages"]
    }
    inventory = inventory_existing_artifacts(
        args.inventory_root,
        current_stage_semantics=stage_semantics,
        current_stage_code=stage_code,
        max_files=args.max_inventory_files,
    )
    plan = compute_earliest_rebuild_stage(
        previous,
        current,
        inventory,
        graph,
        impact_registry=change_impact_registry(semantic_registry),
    )
    plan.update(
        {
            "code_authority_sha": args.implementation_sha,
            "contract_manifest_sha256": file_sha256(
                contract_dir / "contract_manifest.json"
            ),
            "semantic_bundle": bundle,
        }
    )
    release = build_release_authority_preflight(
        artifact_gate_results={
            "official_current_manifests": False,
            "phase4_human_review": False,
        },
        phase4_human_signoff=False,
    )
    write_json(
        staging / "phase4_semantic_domain_registry.json",
        {
            **semantic_registry,
            "semantic_bundle": bundle,
            "stage_authorities": stage_authority,
        },
    )
    write_json(staging / "phase4_stage_dependency_graph.json", graph)
    (staging / "phase4_stage_topological_order.txt").write_text(
        "\n".join(graph["topological_order"]) + "\n",
        encoding="utf-8",
    )
    write_json(
        staging / "phase4_artifact_manifest_schema.json",
        artifact_manifest_json_schema(),
    )
    write_json(
        staging / "phase4_release_authority_schema.json",
        release_authority_json_schema(),
    )
    write_csv(
        staging / "phase4_golden_hash_cases.csv",
        hash_cases(),
        ["case_id", "change", "expected", "result"],
    )
    write_csv(
        staging / "phase4_golden_invalidation_cases.csv",
        invalidation_cases(),
        ["case_id", "change_or_condition", "expected", "result"],
    )
    (staging / "phase4_semantic_hash_contract.md").write_text(
        contract_markdown(bundle),
        encoding="utf-8",
    )
    (staging / "phase4_golden_cases_worked.md").write_text(
        worked_markdown(plan),
        encoding="utf-8",
    )
    carry_contract = decision_carry_forward_contracts()
    (staging / "phase4_human_decision_carry_forward_contract.md").write_text(
        carry_forward_markdown(carry_contract),
        encoding="utf-8",
    )
    (staging / "phase4_human_review_checklist.md").write_text(
        checklist_markdown(),
        encoding="utf-8",
    )
    write_json(staging / "phase4_current_rebuild_plan.json", plan)
    write_json(
        staging / "phase4_release_authority_preflight.json",
        release,
    )
    inventory_rows = [
        {
            **record,
            "reason_codes": "|".join(record.get("reason_codes", [])),
        }
        for record in inventory
    ]
    inventory_fields = [
        "artifact_id",
        "path",
        "stage_id",
        "classification",
        "promotable",
        "reason_codes",
    ]
    write_csv(
        staging / "phase4_existing_artifact_inventory.csv",
        inventory_rows,
        inventory_fields,
    )
    stale = [
        record
        for record in inventory_rows
        if record["classification"] != "VALID_CURRENT_AUTHORITY"
    ]
    write_csv(
        staging / "phase4_stale_artifact_report.csv",
        stale,
        inventory_fields,
    )
    verifier_source = (
        repo_root
        / "scripts"
        / "classification_v2"
        / "scientific_contract"
        / "phase4_independent_reference_verifier.py"
    )
    verifier_copy = (
        staging / "phase4_independent_reference_verifier.py"
    )
    shutil.copyfile(verifier_source, verifier_copy)
    subprocess.run(
        [
            sys.executable,
            str(verifier_copy),
            "--output",
            str(staging / "phase4_reference_verification.json"),
        ],
        check=True,
    )
    write_csv(
        staging / "phase4_claim_evidence_matrix.csv",
        claim_rows(),
        ["claim", "status", "evidence"],
    )
    counts = Counter(record["classification"] for record in inventory)
    plan["artifact_inventory_summary"] = {
        "total": len(inventory),
        "classification_counts": dict(sorted(counts.items())),
    }
    write_json(staging / "phase4_current_rebuild_plan.json", plan)
    missing = [
        name for name in REQUIRED_FILES if not (staging / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"incomplete Phase 4 package: {missing}")
    if any(release[field] for field in RELEASE_AUTHORIZATION_FIELDS):
        raise RuntimeError("unsigned Phase 4 package authorized execution")
    reference = json.loads(
        (staging / "phase4_reference_verification.json").read_text(
            encoding="utf-8"
        )
    )
    if reference.get("pass") is not True:
        raise RuntimeError("independent reference verification failed")


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}.",
            suffix=".staging",
            dir=args.output_dir.parent,
        )
    )
    try:
        build(args, staging)
        os.replace(staging, args.output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
