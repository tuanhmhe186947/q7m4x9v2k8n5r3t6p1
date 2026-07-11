from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_STATUS = "BLOCKED_PENDING_FULL_OOF"


def main() -> None:
    """Validate that the Q2 final package skeleton cannot be read as final results."""

    parser = argparse.ArgumentParser(description="Check classification_v2 Q2 final-package skeleton.")
    parser.add_argument(
        "--package-json",
        type=Path,
        default=Path("outputs/classification_v2/q2_final_package/q2_final_report.json"),
    )
    parser.add_argument(
        "--package-md",
        type=Path,
        default=Path("outputs/classification_v2/q2_final_package/q2_final_report.md"),
    )
    parser.add_argument(
        "--model-card-md",
        type=Path,
        default=Path("outputs/classification_v2/q2_final_package/model_card.md"),
    )
    parser.add_argument(
        "--data-card-md",
        type=Path,
        default=Path("outputs/classification_v2/q2_final_package/data_card.md"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_final_package_stub_audit.json"),
    )
    args = parser.parse_args()
    audit = check_package_stub(
        package_json=args.package_json,
        package_md=args.package_md,
        model_card_md=args.model_card_md,
        data_card_md=args.data_card_md,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


def check_package_stub(
    *,
    package_json: Path,
    package_md: Path,
    model_card_md: Path,
    data_card_md: Path,
) -> dict[str, Any]:
    """Return no-claim and artifact-presence checks for the package skeleton."""

    errors: list[str] = []
    package = _load_json(package_json, errors)
    if package.get("status") != EXPECTED_STATUS:
        errors.append(f"status_must_be_{EXPECTED_STATUS}")
    if package.get("can_claim_q2_result") is not False:
        errors.append("can_claim_q2_result_must_be_false")
    if package.get("paper_facing_metrics_available") is not False:
        errors.append("paper_facing_metrics_available_must_be_false")
    boundary = package.get("claim_boundary", {})
    if boundary.get("external_generalization_claim") is not False:
        errors.append("external_generalization_claim_must_be_false")
    execution = package.get("execution_policy", {})
    for key in [
        "outer_test_used_for_model_selection",
        "outer_test_used_for_threshold_tuning",
        "outer_test_used_for_calibration_fit",
    ]:
        if execution.get(key) is not False:
            errors.append(f"{key}_must_be_false")
    if package.get("feature_whitelist_valid") is not True:
        errors.append("feature_whitelist_valid_must_be_true")
    _check_text_file(package_md, ["BLOCKED_PENDING_FULL_OOF", "No external farm"], errors)
    _check_text_file(model_card_md, ["pre-result model card", "Not Intended Use"], errors)
    _check_text_file(data_card_md, ["pre-result skeleton", "not model input X"], errors)
    return {
        "schema_version": "classification_v2_q2_final_package_stub_check_v1",
        "package_json": str(package_json),
        "status": package.get("status"),
        "can_claim_q2_result": package.get("can_claim_q2_result"),
        "paper_facing_metrics_available": package.get("paper_facing_metrics_available"),
        "missing_required_package_artifact_count": package.get("missing_required_package_artifact_count"),
        "feature_whitelist_valid": package.get("feature_whitelist_valid"),
        "errors": errors,
        "valid": not errors,
    }


def _load_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.exists():
        errors.append(f"missing_package_json={path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _check_text_file(path: Path, required_snippets: list[str], errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing_text_artifact={path}")
        return
    text = path.read_text(encoding="utf-8")
    missing = [snippet for snippet in required_snippets if snippet not in text]
    if missing:
        errors.append(f"text_artifact_missing_snippets={path}:{missing}")


if __name__ == "__main__":
    main()
