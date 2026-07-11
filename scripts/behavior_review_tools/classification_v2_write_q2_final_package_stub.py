from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STATUS = "BLOCKED_PENDING_FULL_OOF"


def main() -> None:
    """Write a paper-package skeleton that cannot be mistaken for final results."""

    parser = argparse.ArgumentParser(
        description="Write classification_v2 Q2 final-package skeleton without claiming final metrics."
    )
    parser.add_argument(
        "--contract-json",
        type=Path,
        default=Path("configs/classification_v2/q2_final_calibration_paper_package_contract_v1.json"),
    )
    parser.add_argument(
        "--progress-report-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_progress_report.json"),
    )
    parser.add_argument(
        "--feature-whitelist-audit-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_feature_whitelist_audit.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/q2_final_package"),
    )
    args = parser.parse_args()
    package = build_package_stub(
        contract_json=args.contract_json,
        progress_report_json=args.progress_report_json,
        feature_whitelist_audit_json=args.feature_whitelist_audit_json,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "q2_final_report.json").write_text(json.dumps(package, indent=2), encoding="utf-8")
    (output_dir / "q2_final_report.md").write_text(_render_report_md(package), encoding="utf-8")
    (output_dir / "model_card.md").write_text(_render_model_card(package), encoding="utf-8")
    (output_dir / "data_card.md").write_text(_render_data_card(package), encoding="utf-8")
    print(json.dumps(package, indent=2))


def build_package_stub(
    *,
    contract_json: Path,
    progress_report_json: Path,
    feature_whitelist_audit_json: Path,
) -> dict[str, Any]:
    """Collect current contract evidence while preserving a no-final-claim status."""

    contract = _load_json(contract_json)
    progress = _load_json(progress_report_json)
    whitelist = _load_json(feature_whitelist_audit_json)
    artifacts = contract.get("required_package_artifacts", {})
    missing_artifacts = [
        {"name": name, "path": path}
        for name, path in sorted(artifacts.items())
        if not Path(str(path)).exists()
    ]
    return {
        "schema_version": "classification_v2_q2_final_package_stub_v1",
        "status": STATUS,
        "can_claim_q2_result": False,
        "paper_facing_metrics_available": False,
        "reason": (
            "Full native-OOF/final-test execution has not been authorized and completed; "
            "this package is a reproducible skeleton only."
        ),
        "claim_boundary": contract.get("claim_boundary", {}),
        "execution_policy": contract.get("execution_policy", {}),
        "model_selection_policy": contract.get("model_selection_policy", {}),
        "calibration_policy": {
            "default_method": contract.get("calibration_policy", {}).get("default_method"),
            "fit_inputs": contract.get("calibration_policy", {}).get("fit_inputs", []),
            "forbidden_fit_inputs": contract.get("calibration_policy", {}).get(
                "forbidden_fit_inputs",
                [],
            ),
        },
        "progress_status": progress.get("overall_status"),
        "progress_gates": progress.get("gates", []),
        "feature_whitelist_valid": whitelist.get("valid"),
        "feature_whitelist_contract_version": whitelist.get("contract_version"),
        "missing_required_package_artifacts": missing_artifacts,
        "missing_required_package_artifact_count": len(missing_artifacts),
        "next_required_actions": [
            "Build canonical actor and visual packed letterbox caches if missing.",
            "Run clean full-OOF preflight with canonical cache paths.",
            "Obtain explicit user authorization for full OOF/final-test execution.",
            "Run full native-OOF evaluation once preflight matches the committed code state.",
            "Fit calibration only on inner validation or OOF train-fold logits.",
            "Generate final metrics, confidence intervals, figures, and cards from verified outputs.",
        ],
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _render_report_md(package: dict[str, Any]) -> str:
    lines = [
        "# classification_v2 Q2 Final Package",
        "",
        f"Status: **{package['status']}**",
        "",
        f"Can claim Q2 result: **{package['can_claim_q2_result']}**",
        "",
        package["reason"],
        "",
        "## Claim Boundary",
        "",
        "- Q2 internal recording-date/video-safe improvement only.",
        "- No external farm, camera, cohort, biological identity, or SOTA claim.",
        "",
        "## Current Evidence",
        "",
        f"- Progress status: `{package.get('progress_status')}`",
        f"- Feature whitelist valid: `{package.get('feature_whitelist_valid')}`",
        f"- Missing required package artifacts: `{package.get('missing_required_package_artifact_count')}`",
        "",
        "## Next Required Actions",
    ]
    lines.extend(f"- {item}" for item in package["next_required_actions"])
    return "\n".join(lines) + "\n"


def _render_model_card(package: dict[str, Any]) -> str:
    lines = [
        "# Model Card: classification_v2 Q2 Multimodal Candidate",
        "",
        f"Status: **{package['status']}**",
        "",
        "This is a pre-result model card skeleton. It documents the intended model family and safety gates, "
        "but it does not report final performance.",
        "",
        "## Intended Use",
        "",
        "Internal pig behavior recognition under recording-date/video-safe validation.",
        "",
        "## Not Intended Use",
        "",
        "- External farm/camera/cohort generalization claims.",
        "- Persistent biological identity claims from annotation-local `pig_id`.",
        "- Model selection, threshold tuning, or calibration using outer test data.",
    ]
    return "\n".join(lines) + "\n"


def _render_data_card(package: dict[str, Any]) -> str:
    lines = [
        "# Data Card: classification_v2 Reviewed Train-Ready Dataset",
        "",
        f"Status: **{package['status']}**",
        "",
        "The data card is a pre-result skeleton. It records required leakage and audit boundaries before final "
        "OOF/final-test execution.",
        "",
        "## Required Boundaries",
        "",
        "- Review decisions apply by `review_unit_id`.",
        "- `pig_id` is annotation-local and not a cross-video identity.",
        "- Manual, review, identifier, path, policy, and label columns are not model input X.",
        "- Reviewed/excluded rows are masked or weighted; rows are not silently dropped.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
