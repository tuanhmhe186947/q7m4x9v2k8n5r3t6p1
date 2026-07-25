"""Deterministically render the Classification V2 scientific contract."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

from validate_scientific_contract import (
    expand_entities,
    load_contract,
    schema_hash,
    validate_contract,
)

STAGE_FIELDS = [
    "stage_id",
    "stage_name",
    "purpose",
    "input_artifacts",
    "output_artifacts",
    "input_grain",
    "output_grain",
    "grouping_keys",
    "canonical_identity_keys",
    "pair_reset_key",
    "temporal_support",
    "required_columns",
    "produced_columns",
    "forbidden_columns",
    "schema_version",
    "evidence_semantics_version",
    "model_eligibility",
    "review_eligibility",
    "deterministic_ordering",
    "missingness_policy",
    "failure_policy",
    "checker",
    "tests",
    "audit_artifact",
    "code_locations",
    "current_status",
]

FEATURE_FIELDS = [
    "feature_id",
    "feature_name",
    "feature_family",
    "producer_stage",
    "formula_latex",
    "formula_plain",
    "required_inputs",
    "output_dtype",
    "units",
    "coordinate_system",
    "normalization",
    "computation_grain",
    "grouping_key",
    "pair_reset_key",
    "validity_mask",
    "availability_mask",
    "missing_value_semantics",
    "zero_value_semantics",
    "denominator",
    "aggregation_rule",
    "minimum_valid_observations",
    "no_valid_observation_behavior",
    "physical_interpretation",
    "is_physical_measurement",
    "review_eligible",
    "model_eligible",
    "model_group",
    "leakage_risk",
    "threshold_dependencies",
    "schema_version",
    "code_locations",
    "test_locations",
    "implementation_status",
    "known_limitations",
]

INVARIANT_FIELDS = [
    "invariant_id",
    "stage_id",
    "invariant_description",
    "scientific_reason",
    "severity",
    "fatal",
    "input_scope",
    "expected_condition",
    "failure_condition",
    "checker",
    "audit_field",
    "unit_test",
    "integration_test",
    "golden_case_ids",
    "code_locations",
    "implementation_status",
    "known_gap",
]

GAP_FIELDS = [
    "gap_id",
    "category",
    "severity",
    "affected_stage",
    "affected_features",
    "evidence",
    "scientific_consequence",
    "downstream_artifacts_affected",
    "requires_rebuild_from_stage",
    "recommended_resolution",
    "blocking_before_native_evidence",
    "blocking_before_pig_strenet",
    "blocking_before_behavior_review",
    "blocking_before_training",
    "status",
]


def _display(value: Any) -> Any:
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fields,
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _display(row.get(field, "")) for field in fields})
    return buffer.getvalue()


def _stage_label(stage: dict[str, Any]) -> str:
    identity = ",".join(stage["canonical_identity_keys"]) or "none"
    grouping = ",".join(stage["grouping_keys"]) or "none"
    return (
        f"{stage['stage_name']}<br/>grain: {stage['output_grain']}"
        f"<br/>group: {grouping}<br/>identity: {identity}"
        f"<br/>reset: {stage['pair_reset_key'] or 'none'}"
        f"<br/>schema: {stage['schema_version']}"
        f"<br/>check: {stage['checker'] or 'none'}"
        f"<br/>model/review: {stage['model_eligibility']}/"
        f"{stage['review_eligibility']}"
    )


def _complete_pipeline(stages: list[dict[str, Any]]) -> list[str]:
    lines = ["flowchart LR"]
    for index, stage in enumerate(stages):
        lines.append(f'  S{index}["{_stage_label(stage)}"]')
        if index:
            lines.append(f"  S{index - 1} --> S{index}")
    return lines


def _grain_transition(stages: list[dict[str, Any]]) -> list[str]:
    lines = ["flowchart LR"]
    grains: list[str] = []
    for stage in stages:
        grain = str(stage["output_grain"])
        if not grains or grains[-1] != grain:
            grains.append(grain)
    for index, grain in enumerate(grains):
        lines.append(f'  G{index}["{grain}"]')
        if index:
            lines.append(f"  G{index - 1} --> G{index}")
    return lines


def _identity_flow() -> list[str]:
    return [
        "flowchart LR",
        '  A["source/dataset/video"] --> B["scene_frame_uid"]',
        '  B --> C["frame_uid + object_id_in_image"]',
        '  C --> D["object_track_key"]',
        '  D --> E["temporal_unit_key"]',
        '  E --> F["review_unit_id"]',
        '  F --> G["window_id"]',
        '  G --> H["tensor row/order hash"]',
        '  P["pig_id: annotation-local metadata"] -. not authority .-> D',
    ]


def _pair_flow() -> list[str]:
    return [
        "flowchart TD",
        '  A["row t"] --> B{"same temporal_unit_key?"}',
        '  P["row t-1"] --> B',
        '  B --> C{"same object_track_key?"}',
        '  C --> D{"both geometry valid?"}',
        '  D --> E{"delta_t > 0?"}',
        '  E --> F["valid_motion_pair=1"]',
        '  B -- no --> R["reset; unavailable, not measured zero"]',
        '  C -- no --> R',
        '  D -- no --> R',
        '  E -- no --> R',
    ]


def _missingness_flow() -> list[str]:
    return [
        "flowchart LR",
        '  C["cause"] --> D["detection rule"]',
        '  D --> M["availability/validity mask"]',
        '  M --> V["numeric feature: NaN or documented placeholder"]',
        '  M --> A["validity-aware aggregate denominator"]',
        '  M --> T["model-visible mask"]',
        '  M --> R["review/audit coverage"]',
    ]


def _review_flow() -> list[str]:
    return [
        "flowchart LR",
        '  H["Hidden manifest"] --> HD["stable-key decisions"]',
        '  HD --> HA["Hidden apply: row-preserving"]',
        '  HA --> N["native review units"]',
        '  N --> BD["behavior decisions by review_unit_id"]',
        '  BD --> BA["behavior apply"]',
        '  BA --> F["final-view recompute"]',
        '  F --> X["train-ready candidate"]',
    ]


def _tensor_flow(schema: dict[str, Any]) -> list[str]:
    names = schema["ordered_feature_names"]
    return [
        "flowchart LR",
        f'  P["producer: {len(names)} ordered motion features"]',
        '  P --> C{"exact ordered equality?"}',
        '  C -- yes --> T["fixed float32 tensor + masks"]',
        '  C -- no --> F["FAIL_CLOSED"]',
        f'  T --> H["schema SHA-256: {schema["schema_hash"][:16]}..."]',
    ]


def _leakage_flow() -> list[str]:
    return [
        "flowchart TD",
        '  L["labels / review / target_roi / IDs / paths / folds"]',
        '  L -->|forbidden| X["model X"]',
        '  S["source/session/video groups"] -->|grouped split| TR["train"]',
        '  S -->|disjoint| VA["validation/test"]',
        '  TR --> FT["fit transforms only here"]',
        '  FT --> VA',
    ]


def _eligibility_flow() -> list[str]:
    return [
        "flowchart LR",
        '  F["all computed features"] --> M["model-eligible whitelist"]',
        '  F --> R["review-only evidence"]',
        '  F --> A["audit/provenance only"]',
        '  R -->|forbidden| X["model tensor"]',
        '  A -->|forbidden| X',
        '  M --> X',
    ]


def _invalidation_flow() -> list[str]:
    return [
        "flowchart LR",
        '  C["formula/unit/mask/order/schema change"] --> H["new schema hash"]',
        '  H --> N["invalidate native evidence"]',
        '  N --> P["invalidate Pig-STRENet"]',
        '  P --> B["invalidate behavior evidence"]',
        '  B --> T["invalidate train-ready/tensors"]',
        '  T --> M["invalidate checkpoints/predictions"]',
    ]


def _diagrams(
    stages: list[dict[str, Any]],
    schema: dict[str, Any],
) -> list[tuple[str, list[str]]]:
    return [
        ("Complete artifact pipeline", _complete_pipeline(stages)),
        ("Grain transitions", _grain_transition(stages)),
        ("Identity and provenance propagation", _identity_flow()),
        ("Temporal pair formation and reset boundaries", _pair_flow()),
        ("Missingness and mask propagation", _missingness_flow()),
        ("Review decision propagation", _review_flow()),
        ("Model tensor construction", _tensor_flow(schema)),
        ("Leakage boundaries", _leakage_flow()),
        ("Review-only versus model-eligible", _eligibility_flow()),
        ("Artifact invalidation", _invalidation_flow()),
    ]


def _dataflow_mmd(
    stages: list[dict[str, Any]],
    schema: dict[str, Any],
) -> str:
    stage_ids = ",".join(stage["stage_id"] for stage in stages)
    lines = [f"%% stage_ids:{stage_ids}"]
    for title, diagram in _diagrams(stages, schema):
        lines.extend(["", f"%% {title}", *diagram])
    return "\n".join(lines) + "\n"


def _dataflow_md(
    stages: list[dict[str, Any]],
    schema: dict[str, Any],
) -> str:
    lines = [
        "# Classification V2 pipeline dataflow",
        "",
        "Generated from `00_pipeline_contract.yaml`. Do not edit manually.",
    ]
    for title, diagram in _diagrams(stages, schema):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "```mermaid",
                *diagram,
                "```",
            ]
        )
    return "\n".join(lines) + "\n"


def _readme(
    contract: dict[str, Any],
    stages: list[dict[str, Any]],
    features: list[dict[str, Any]],
    invariants: list[dict[str, Any]],
) -> str:
    metadata = contract["contract_metadata"]
    return "\n".join(
        [
            "# Classification V2 scientific contract v1",
            "",
            "This package is an audit authority, not approval of current code.",
            "`00_pipeline_contract.yaml` is the single source of truth. It is",
            "JSON-compatible YAML so validation requires only Python's standard",
            "library. Derived Markdown, CSV and JSON files are deterministic.",
            "",
            "## Scope",
            "",
            f"- Pipeline stages: {len(stages)}",
            f"- Feature semantics: {len(features)}",
            f"- Invariants: {len(invariants)}",
            f"- Golden numerical cases: {len(contract['golden_cases'])}",
            f"- Model schemas: {len(contract['model_schemas'])}",
            f"- Contract version: `{metadata['contract_version']}`",
            f"- Audited code SHA: `{metadata['audited_code_sha']}`",
            "",
            "Dataset counts in the contract are instance evidence only; they",
            "are never interpreted as universal mathematical semantics.",
            "",
            "## Validation",
            "",
            "```text",
            "python scripts/classification_v2/scientific_contract/"
            "validate_scientific_contract.py",
            "python scripts/classification_v2/scientific_contract/"
            "check_code_contract_mapping.py",
            "python scripts/classification_v2/scientific_contract/"
            "render_scientific_contract_docs.py --check",
            "```",
            "",
            "Any missing, reordered, duplicated or unexpected required tensor",
            "feature is a fail-closed schema violation. Image-coordinate",
            "distances and rates are not physical measurements.",
            "",
            "## Current decision",
            "",
            "`CURRENT_CODE_SCIENTIFICALLY_APPROVED=NO`. See",
            "`11_known_gaps_and_risks.csv` and",
            "`13_independent_review_report.md`.",
        ]
    ) + "\n"


def _missingness_md(contract: dict[str, Any]) -> str:
    lines = [
        "# Missingness and validity masks",
        "",
        "Generated from `00_pipeline_contract.yaml`.",
        "",
        "A missing or invalid temporal pair is unavailable evidence. It must",
        "never be interpreted as an observed zero-motion measurement.",
        "",
        "| Category | Cause | Detection | Mask | Numeric value | Zero meaning | "
        "Denominator | Model | Review | Audit coverage |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for item in contract["missingness_categories"]:
        cells = [
            item["missingness_id"],
            item["cause"],
            item["detection_rule"],
            item["mask_column"],
            item["numerical_feature_policy"],
            item["zero_semantics"],
            item["aggregate_denominator"],
            item["model_visibility"],
            item["review_visibility"],
            item["audit_coverage_field"],
        ]
        lines.append("| " + " | ".join(str(cell).replace("|", "\\|") for cell in cells) + " |")
    return "\n".join(lines) + "\n"


def _assumptions_md(contract: dict[str, Any]) -> str:
    lines = [
        "# Scientific assumptions and limits",
        "",
        "Generated from `00_pipeline_contract.yaml`.",
    ]
    for item in contract["assumptions"]:
        lines.extend(
            [
                "",
                f"## {item['assumption_id']}",
                "",
                item["statement"],
                "",
                f"- Status: `{item['implementation_status']}`",
                f"- Defensibility: {item['scientific_defensibility']}",
                f"- Limit: {item['limit']}",
                f"- Required evidence: {item['required_evidence']}",
            ]
        )
    return "\n".join(lines) + "\n"


def _golden_md(cases: list[dict[str, Any]]) -> str:
    lines = [
        "# Golden numerical cases",
        "",
        "Generated from the machine-readable golden cases in the primary",
        "contract. Expected values are recomputed by the independent validator.",
    ]
    for case in cases:
        lines.extend(
            [
                "",
                f"## {case['case_id']}",
                "",
                case["scientific_purpose"],
                "",
                f"- Pair masks: `{json.dumps(case['expected_pair_masks'])}`",
                "- Expected numeric: "
                f"`{json.dumps(case['expected_numerical_values'])}`",
                "- Expected aggregate: "
                f"`{json.dumps(case['expected_aggregate_values'])}`",
                f"- Selected neighbor: `{case['expected_selected_neighbor']}`",
                f"- Tolerance: `{case['tolerance']}`",
                "- Invariants: "
                + ", ".join(f"`{item}`" for item in case["related_invariants"]),
            ]
        )
    return "\n".join(lines) + "\n"


def _checklist_md(contract: dict[str, Any]) -> str:
    lines = [
        "# Classification V2 scientific change-impact checklist",
        "",
        "Complete this checklist for every feature or stage change. Record the",
        "answer, evidence, reviewer and invalidated artifact hashes.",
        "",
    ]
    lines.extend(
        f"- [ ] {question}" for question in contract["change_impact_questions"]
    )
    lines.extend(
        [
            "",
            "## Invalidation decision",
            "",
            "- Change ID:",
            "- Old/new schema hashes:",
            "- Earliest rebuild stage:",
            "- Invalidated artifacts:",
            "- Golden cases updated:",
            "- Tests updated:",
            "- Threshold recalibration evidence:",
            "- Reviewer and date:",
        ]
    )
    return "\n".join(lines) + "\n"


def _independent_review_md(contract: dict[str, Any]) -> str:
    review = contract["independent_review"]
    lines = [
        "# Independent adversarial review",
        "",
        f"Review pass: `{review['review_id']}`",
        "",
        "This is a falsification-oriented review. Passing tests does not imply",
        "scientific correctness or approval of the current implementation.",
        "",
        "## Method",
        "",
    ]
    lines.extend(f"- {item}" for item in review["methods"])
    lines.extend(["", "## Findings", ""])
    for finding in review["findings"]:
        lines.extend(
            [
                f"### {finding['finding_id']}: {finding['classification']}",
                "",
                finding["finding"],
                "",
                f"- Severity: `{finding['severity']}`",
                f"- Evidence: {finding['evidence']}",
                f"- Related gaps: {', '.join(finding['related_gap_ids'])}",
                f"- Disposition: {finding['disposition']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Reviewer conclusion",
            "",
            review["conclusion"],
        ]
    )
    return "\n".join(lines) + "\n"


def _symbol_range(source_path: Path, symbol: str) -> str:
    if not source_path.exists() or source_path.suffix != ".py":
        return ""
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return ""
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ) and node.name == symbol:
            end = getattr(node, "end_lineno", node.lineno)
            return f"{node.lineno}-{end}"
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == symbol:
                    end = getattr(node, "end_lineno", node.lineno)
                    return f"{node.lineno}-{end}"
    return ""


def _mapping_rows(
    project_root: Path,
    stages: list[dict[str, Any]],
    features: list[dict[str, Any]],
    invariants: list[dict[str, Any]],
    runtime_code_authority: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stage_by_id = {stage["stage_id"]: stage for stage in stages}
    entities = [
        *(("stage", item, "stage_id", "current_status") for item in stages),
        *(
            ("feature", item, "feature_id", "implementation_status")
            for item in features
        ),
        *(
            ("invariant", item, "invariant_id", "implementation_status")
            for item in invariants
        ),
    ]
    for kind, item, id_field, status_field in entities:
        locations = item.get("code_locations", [])
        if not locations and kind == "feature":
            producer = stage_by_id.get(item.get("producer_stage", ""), {})
            locations = producer.get("code_locations", [])
        if not locations and kind == "invariant":
            stage_ids = item.get("stage_id", [])
            if isinstance(stage_ids, str):
                stage_ids = [stage_ids]
            for stage_id in stage_ids:
                locations = stage_by_id.get(stage_id, {}).get(
                    "code_locations",
                    [],
                )
                if locations:
                    break
        tests = item.get("tests", item.get("test_locations", []))
        if isinstance(tests, str):
            tests = [tests] if tests else []
        if not locations:
            rows.append(
                {
                    "contract_item_id": item[id_field],
                    "contract_item_type": kind,
                    "source_file": "",
                    "symbol": "",
                    "line_range": "",
                    "test_file": "",
                    "test_name": "",
                    "audit_output": item.get(
                        "audit_artifact",
                        item.get("audit_field", ""),
                    ),
                    "current_implementation_status": item[status_field],
                }
            )
            continue
        for location in locations:
            source, _, symbol = location.partition("#")
            test_file = ""
            test_name = ""
            if tests:
                test_file, _, test_name = str(tests[0]).partition("#")
            rows.append(
                {
                    "contract_item_id": item[id_field],
                    "contract_item_type": kind,
                    "source_file": source,
                    "symbol": symbol,
                    "line_range": _symbol_range(
                        project_root / source,
                        symbol,
                    ),
                    "test_file": test_file,
                    "test_name": test_name,
                    "audit_output": item.get(
                        "audit_artifact",
                        item.get("audit_field", ""),
                    ),
                    "current_implementation_status": item[status_field],
                }
            )
    stage_ids = {str(stage["stage_id"]) for stage in stages}
    runtime_files = runtime_code_authority.get(
        "stage_runtime_dependency_files",
        {},
    )
    if set(runtime_files) != stage_ids:
        raise ValueError(
            "runtime code authority stage IDs must exactly match contract "
            f"stages: missing={sorted(stage_ids - set(runtime_files))}, "
            f"unknown={sorted(set(runtime_files) - stage_ids)}"
        )
    shared_runtime_files: dict[str, set[str]] = {
        stage_id: set() for stage_id in stage_ids
    }
    for shared in runtime_code_authority.get(
        "shared_stage_runtime_dependencies",
        [],
    ):
        applicable = set(shared.get("applicable_stage_ids", []))
        if not applicable or not applicable.issubset(stage_ids):
            raise ValueError(
                "shared runtime code authority has invalid stage IDs: "
                f"{sorted(applicable - stage_ids)}"
            )
        files = {
            str(value)
            for value in shared.get("runtime_dependency_files", [])
            if str(value).strip()
        }
        if not files:
            raise ValueError(
                "shared runtime code authority must declare files: "
                f"{shared.get('authority_id', '')}"
            )
        for stage_id in applicable:
            shared_runtime_files[stage_id].update(files)
    for stage_id in sorted(runtime_files):
        stage_files = (
            set(runtime_files[stage_id])
            | shared_runtime_files[stage_id]
        )
        for source in sorted(stage_files):
            rows.append(
                {
                    "contract_item_id": stage_id,
                    "contract_item_type": "stage",
                    "source_file": source,
                    "symbol": "",
                    "line_range": "",
                    "test_file": (
                        "tests/test_classification_v2_phase4_"
                        "runtime_dependencies.py"
                    ),
                    "test_name": "",
                    "audit_output": (
                        "phase4_runtime_dependency_completeness.json"
                    ),
                    "current_implementation_status": (
                        "IMPLEMENTED_AND_TESTED"
                    ),
                }
            )
    return rows


def _manifest_payload(
    root: Path,
    names: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for name in sorted(names):
        path = root / name
        data = path.read_bytes()
        files.append(
            {
                "path": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return {
        "manifest_schema_version": "classification_v2.scientific_contract_manifest.v1",
        "contract_version": metadata["contract_version"],
        "audited_code_sha": metadata["audited_code_sha"],
        "primary_authority": "00_pipeline_contract.yaml",
        "generated_files": files,
    }


def render_payloads(
    contract_path: Path,
    project_root: Path,
) -> dict[str, str]:
    contract = load_contract(contract_path)
    validation = validate_contract(contract_path, check_generated=False)
    if not validation["valid"]:
        raise ValueError(
            "primary contract invalid before render: "
            + "; ".join(validation["errors"])
        )
    stages = expand_entities(contract, "stages", "stage_defaults")
    features = expand_entities(contract, "features", "feature_defaults")
    invariants = expand_entities(
        contract,
        "invariants",
        "invariant_defaults",
    )
    cases = expand_entities(
        contract,
        "golden_cases",
        "golden_case_defaults",
    )
    schemas = []
    for schema in contract["model_schemas"]:
        schemas.append({**schema, "schema_hash": schema_hash(schema)})
    primary_schema = next(
        schema
        for schema in schemas
        if schema["schema_id"] == "schema.pig_strenet_motion_v2"
    )
    mapping_fields = [
        "contract_item_id",
        "contract_item_type",
        "source_file",
        "symbol",
        "line_range",
        "test_file",
        "test_name",
        "audit_output",
        "current_implementation_status",
    ]
    runtime_authority_path = (
        contract_path.parent / "stage_runtime_code_authority.json"
    )
    runtime_code_authority = json.loads(
        runtime_authority_path.read_text(encoding="utf-8")
    )
    payloads = {
        "README.md": _readme(contract, stages, features, invariants),
        "01_pipeline_dataflow.md": _dataflow_md(stages, primary_schema),
        "01_pipeline_dataflow.mmd": _dataflow_mmd(stages, primary_schema),
        "02_stage_contract_registry.csv": _csv_text(stages, STAGE_FIELDS),
        "03_feature_semantics_registry.csv": _csv_text(
            features,
            FEATURE_FIELDS,
        ),
        "04_invariant_gate_matrix.csv": _csv_text(
            invariants,
            INVARIANT_FIELDS,
        ),
        "05_tensor_schema_manifest.json": (
            json.dumps(
                {
                    "manifest_schema_version": (
                        "classification_v2.tensor_schema_manifest.v1"
                    ),
                    "schemas": schemas,
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        ),
        "06_missingness_and_masks.md": _missingness_md(contract),
        "07_scientific_assumptions_and_limits.md": _assumptions_md(
            contract
        ),
        "08_golden_numeric_cases.yaml": (
            json.dumps(
                {
                    "golden_case_schema_version": (
                        "classification_v2.golden_numeric_cases.v1"
                    ),
                    "cases": cases,
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        ),
        "09_golden_numeric_cases.md": _golden_md(cases),
        "10_code_contract_mapping.csv": _csv_text(
            _mapping_rows(
                project_root,
                stages,
                features,
                invariants,
                runtime_code_authority,
            ),
            mapping_fields,
        ),
        "11_known_gaps_and_risks.csv": _csv_text(
            contract["known_gaps"],
            GAP_FIELDS,
        ),
        "12_change_impact_checklist.md": _checklist_md(contract),
        "13_independent_review_report.md": _independent_review_md(contract),
    }
    return payloads


def write_payloads(
    contract_path: Path,
    project_root: Path,
    *,
    check: bool,
) -> list[str]:
    root = contract_path.parent
    payloads = render_payloads(contract_path, project_root)
    mismatches: list[str] = []
    for name, text in payloads.items():
        path = root / name
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                mismatches.append(name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
    manifest_names = [
        "00_pipeline_contract.yaml",
        "stage_runtime_code_authority.json",
        *sorted(payloads),
    ]
    manifest = (
        json.dumps(
            _manifest_payload(
                root,
                manifest_names,
                load_contract(contract_path)["contract_metadata"],
            ),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )
    manifest_path = root / "contract_manifest.json"
    if check:
        if (
            not manifest_path.exists()
            or manifest_path.read_text(encoding="utf-8") != manifest
        ):
            mismatches.append("contract_manifest.json")
    else:
        manifest_path.write_text(manifest, encoding="utf-8", newline="\n")
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            "docs/classification_v2/scientific_contract_v1/"
            "00_pipeline_contract.yaml"
        ),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    mismatches = write_payloads(
        args.contract,
        args.project_root.resolve(),
        check=args.check,
    )
    if mismatches:
        print("OUT_OF_SYNC")
        for name in mismatches:
            print(f"- {name}")
        return 1
    print("PASS render deterministic" if args.check else "RENDERED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
