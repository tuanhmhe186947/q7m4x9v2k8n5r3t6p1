from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a compact Q2 classification_v2 progress report.")
    parser.add_argument(
        "--snapshot-check-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_snapshot_check_audit.json"),
    )
    parser.add_argument(
        "--baseline-config-audit-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_baseline_config_audit.json"),
    )
    parser.add_argument(
        "--baseline-smoke-check-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_baseline_smoke_check_audit.json"),
    )
    parser.add_argument(
        "--reproducibility-audit-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_smoke/training_reproducibility_cuda_post_s0a/"
            "reproducibility_audit.json"
        ),
    )
    parser.add_argument(
        "--b4-seed-variance-check-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/b4_seed_variance_check_audit.json"),
    )
    parser.add_argument(
        "--q2-oof-metric-contract-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_oof_metric_contract_audit.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_progress_report.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_progress_report.md"),
    )
    args = parser.parse_args()

    snapshot = _load_optional_json(args.snapshot_check_json)
    baseline_configs = _load_optional_json(args.baseline_config_audit_json)
    baseline_smokes = _load_optional_json(args.baseline_smoke_check_json)
    reproducibility = _load_optional_json(args.reproducibility_audit_json)
    b4_seed_variance = _load_optional_json(args.b4_seed_variance_check_json)
    q2_oof_metric_contract = _load_optional_json(args.q2_oof_metric_contract_json)

    gates = [
        _gate("S0A snapshot/data contract", snapshot.get("valid") is True, snapshot.get("errors")),
        _gate("B2-B7 config matrix", baseline_configs.get("valid") is True, baseline_configs.get("errors")),
        _gate("B2-B7 CUDA smoke", baseline_smokes.get("valid") is True, baseline_smokes.get("errors")),
        _gate("B4 inner-validation seed variance", b4_seed_variance.get("valid") is True, b4_seed_variance.get("errors")),
        _gate(
            "S5 Q2 native-OOF metric contract",
            q2_oof_metric_contract.get("valid") is True
            and q2_oof_metric_contract.get("full_oof_execution_allowed_by_contract") is False
            and q2_oof_metric_contract.get("outer_test_used_for_threshold_tuning") is False,
            q2_oof_metric_contract.get("errors"),
        ),
        _gate(
            "Strict trainer reproducibility",
            reproducibility.get("errors") == [] and reproducibility.get("forbidden_model_input_rejected") is True,
            reproducibility.get("errors"),
        ),
    ]
    remaining = [
        "Full OOF remains blocked until explicit authorization and matching clean preflight.",
        "S5 metric contract is defined; real per-source/matched metrics still need explicitly authorized full OOF predictions.",
        "B4 seed variance is estimated from bounded validation-only smoke; full inner-validation variance still needs non-smoke OOF authorization.",
        "S7-S9 hard-negative mining, active review, final calibration, and paper package remain future work.",
    ]
    result = {
        "schema_version": "classification_v2_q2_progress_report_v1",
        "overall_status": "PASS_PARTIAL_ROADMAP" if all(gate["passed"] for gate in gates) else "FAIL",
        "claim_boundary": "Q2 internal recording-date/video-safe improvement only; no external farm/camera/cohort claim.",
        "gates": gates,
        "remaining_work": remaining,
        "evidence": {
            "snapshot": _evidence_snapshot(snapshot),
            "baseline_configs": _evidence_baseline_configs(baseline_configs),
            "baseline_smokes": _evidence_baseline_smokes(baseline_smokes),
            "b4_seed_variance": _evidence_b4_seed_variance(b4_seed_variance),
            "q2_oof_metric_contract": _evidence_q2_oof_metric_contract(q2_oof_metric_contract),
            "reproducibility": _evidence_reproducibility(reproducibility),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_render_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["overall_status"], "gates": gates}, indent=2))
    if result["overall_status"] == "FAIL":
        raise SystemExit(1)


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": str(path), "errors": [f"missing:{path}"], "valid": False}
    return json.loads(path.read_text(encoding="utf-8"))


def _gate(name: str, passed: bool, errors: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "errors": errors or []}


def _evidence_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "snapshot_id": snapshot.get("expected_snapshot_id"),
        "current_snapshot_id": snapshot.get("current_snapshot_id"),
        "valid": snapshot.get("valid"),
    }


def _evidence_baseline_configs(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_count": audit.get("baseline_count"),
        "valid": audit.get("valid"),
        "snapshot_ids": sorted({row.get("snapshot_id") for row in audit.get("baselines", [])}),
    }


def _evidence_baseline_smokes(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_count": audit.get("baseline_count"),
        "require_device": audit.get("require_device"),
        "runtime_python_executable": audit.get("runtime_python_executable"),
        "devices": sorted({row.get("device") for row in audit.get("baselines", [])}),
        "git_dirty_values": sorted({str(row.get("git_dirty")) for row in audit.get("baselines", [])}),
    }


def _evidence_reproducibility(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "errors": audit.get("errors"),
        "forbidden_model_input_rejected": audit.get("forbidden_model_input_rejected"),
        "prediction_sha256": audit.get("prediction_sha256"),
        "test_prediction_sha256": audit.get("test_prediction_sha256"),
    }


def _evidence_b4_seed_variance(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "seed_count": audit.get("seed_count"),
        "summary": audit.get("summary"),
    }


def _evidence_q2_oof_metric_contract(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": audit.get("valid"),
        "contract_version": audit.get("contract_version"),
        "statistical_unit": audit.get("statistical_unit"),
        "primary_split_policy": audit.get("primary_split_policy"),
        "label_count": audit.get("label_count"),
        "required_metric_group_count": audit.get("required_metric_group_count"),
        "confusion_pair_count": audit.get("confusion_pair_count"),
        "full_oof_execution_allowed_by_contract": audit.get("full_oof_execution_allowed_by_contract"),
        "outer_test_used_for_threshold_tuning": audit.get("outer_test_used_for_threshold_tuning"),
    }


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# classification_v2 Q2 Progress Report",
        "",
        f"Status: **{result['overall_status']}**",
        "",
        f"Claim boundary: {result['claim_boundary']}",
        "",
        "## Gates",
    ]
    for gate in result["gates"]:
        marker = "PASS" if gate["passed"] else "FAIL"
        lines.append(f"- {marker}: {gate['name']}")
        if gate["errors"]:
            lines.append(f"  - errors: `{gate['errors']}`")
    lines.extend(["", "## Remaining Work"])
    lines.extend(f"- {item}" for item in result["remaining_work"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
