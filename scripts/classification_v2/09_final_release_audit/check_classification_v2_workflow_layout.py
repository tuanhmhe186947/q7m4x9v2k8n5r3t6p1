"""Audit the single-namespace classification_v2 script workflow layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STAGES = (
    "00_source_feature_temporal",
    "01_review_units_gui",
    "02_train_ready_exports",
    "03_image_cache_context",
    "04_baselines_smokes",
    "05_preflight_authorization",
    "06_full_oof_training",
    "07_postrun_evaluation",
    "08_publication_reporting",
    "09_final_release_audit",
)
REMOVED_PATHS = (
    Path("scripts/behavior_review_tools"),
    Path("scripts/dev_tools"),
    Path("scripts/classification_v2/_compat.py"),
)
FORBIDDEN_REFERENCE_TOKENS = (
    "scripts/behavior_review_tools",
    "scripts\\behavior_review_tools",
    "scripts/dev_tools",
    "scripts\\dev_tools",
    "scripts.behavior_review_tools",
    "scripts.dev_tools",
)
REFERENCE_ROOTS = (
    Path("scripts/classification_v2"),
    Path("src/pig_behavior/classification_v2"),
    Path("tests"),
    Path("docs"),
    Path(".agents/memory"),
)
SELF_PATH = Path(__file__).resolve()
REQUIRED_ENTRYPOINTS = (
    Path("scripts/classification_v2/00_source_feature_temporal/classification_v2_merge_sources.py"),
    Path("scripts/classification_v2/01_review_units_gui/review_temporal_unit_gui.py"),
    Path("scripts/classification_v2/02_train_ready_exports/classification_v2_export_train_ready_windows.py"),
    Path("scripts/classification_v2/03_image_cache_context/classification_v2_build_packed_image_cache.py"),
    Path("scripts/classification_v2/04_baselines_smokes/classification_v2_run_q2_baseline_smokes.py"),
    Path("scripts/classification_v2/05_preflight_authorization/preflight_classification_v2_full_multimodal_oof.py"),
    Path("scripts/classification_v2/06_full_oof_training/classification_v2_run_full_multimodal_oof.py"),
    Path("scripts/classification_v2/07_postrun_evaluation/classification_v2_cross_fit_calibration.py"),
    Path("scripts/classification_v2/08_publication_reporting/classification_v2_register_experiment.py"),
    Path("scripts/classification_v2/09_final_release_audit/check_classification_v2_full_oof_completion_gate.py"),
)


def main() -> None:
    """Write a fail-closed audit for stage order and removed namespaces."""

    parser = argparse.ArgumentParser(description="Audit the classification_v2 numbered script workflow.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/workflow_layout_audit.json"),
    )
    args = parser.parse_args()
    audit = build_workflow_layout_audit()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit["valid"]:
        raise SystemExit(1)


def build_workflow_layout_audit() -> dict[str, object]:
    """Validate all stages, key entrypoints, and absence of legacy references."""

    workflow_root = Path("scripts/classification_v2")
    stage_counts = {stage: len(list((workflow_root / stage).glob("*.py"))) for stage in STAGES}
    missing_stages = [stage for stage, count in stage_counts.items() if count == 0]
    stale_paths = [str(path) for path in REMOVED_PATHS if path.exists()]
    missing_entrypoints = [str(path) for path in REQUIRED_ENTRYPOINTS if not path.is_file()]
    stale_references = _find_stale_references()
    errors: list[str] = []
    if missing_stages:
        errors.append(f"missing_or_empty_stages={missing_stages}")
    if stale_paths:
        errors.append(f"removed_paths_still_exist={stale_paths}")
    if missing_entrypoints:
        errors.append(f"missing_required_entrypoints={missing_entrypoints}")
    if stale_references:
        errors.append(f"stale_reference_count={len(stale_references)}")
    return {
        "schema_version": "classification_v2_workflow_layout_audit_v1",
        "workflow_root": str(workflow_root),
        "stage_order": list(STAGES),
        "stage_script_counts": stage_counts,
        "removed_paths_present": stale_paths,
        "missing_required_entrypoints": missing_entrypoints,
        "stale_references": stale_references,
        "errors": errors,
        "valid": not errors,
    }


def _find_stale_references() -> list[dict[str, object]]:
    """Return line-level references to namespaces removed by the migration."""

    findings: list[dict[str, object]] = []
    for root in REFERENCE_ROOTS:
        if not root.exists():
            continue
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in paths:
            if path.resolve() == SELF_PATH:
                continue
            if not path.is_file() or path.suffix.lower() not in {".py", ".md", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for line_number, line in enumerate(text.splitlines(), start=1):
                tokens = [token for token in FORBIDDEN_REFERENCE_TOKENS if token in line]
                if tokens:
                    findings.append(
                        {
                            "path": str(path),
                            "line": line_number,
                            "tokens": tokens,
                        }
                    )
    return findings


if __name__ == "__main__":
    main()
