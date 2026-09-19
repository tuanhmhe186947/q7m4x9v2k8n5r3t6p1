from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CONTRACT_FILES = [
    Path("configs/classification_v2/data_contract_v2.json"),
    Path("configs/classification_v2/model_architecture_contract_v1.json"),
    Path("configs/classification_v2/paper_grade_protocol_v1.json"),
    Path("configs/classification_v2/q2_feature_whitelist_v1.json"),
    Path("configs/classification_v2/q2_final_calibration_paper_package_contract_v1.json"),
    Path("configs/classification_v2/q2_hard_negative_mining_contract_v1.json"),
    Path("configs/classification_v2/q2_oof_metric_contract_v1.json"),
    Path("configs/classification_v2/trainer_contract_v1.json"),
    Path("configs/classification_v2/trainer_contract_v2.json"),
]

FORBIDDEN_IDENTITY_PHRASES = [
    "biological identity across videos",
    "persistent biological identity",
    "cross-video biological identity",
]

REQUIRED_SCOPE_HINTS = [
    "annotation-local",
    "not identity",
    "never persistent",
    "not used as cross-video identity",
]


def main() -> None:
    """Check that pig_id remains annotation-local in Q2 classification contracts."""

    parser = argparse.ArgumentParser(description="Check classification_v2 pig_id locality and leakage policy.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/pig_id_locality_contract_audit.json"),
    )
    args = parser.parse_args()
    audit = check_pig_id_locality(CONTRACT_FILES)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


def check_pig_id_locality(paths: list[Path]) -> dict[str, Any]:
    """Fail if contracts omit pig_id locality or imply cross-video identity."""

    errors: list[str] = []
    file_summaries: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            errors.append(f"missing_contract={path}")
            file_summaries.append({"path": str(path), "missing": True})
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        text = json.dumps(payload, sort_keys=True).lower()
        has_pig_id = "pig_id" in text
        has_scope_hint = any(hint in text for hint in REQUIRED_SCOPE_HINTS)
        has_forbidden_allowance = _has_forbidden_allowance(text)
        if has_pig_id and not has_scope_hint:
            errors.append(f"missing_pig_id_annotation_local_scope={path}")
        if has_forbidden_allowance:
            errors.append(f"forbidden_pig_id_biological_identity_allowance={path}")
        file_summaries.append(
            {
                "path": str(path),
                "has_pig_id": has_pig_id,
                "has_scope_hint": has_scope_hint,
                "has_forbidden_allowance": has_forbidden_allowance,
            }
        )

    return {
        "schema_version": "classification_v2_pig_id_locality_contract_audit_v1",
        "contract_count": len(paths),
        "contracts_with_pig_id": sum(1 for item in file_summaries if item.get("has_pig_id")),
        "contracts_with_scope_hint": sum(1 for item in file_summaries if item.get("has_scope_hint")),
        "forbidden_identity_allowance_count": sum(1 for item in file_summaries if item.get("has_forbidden_allowance")),
        "files": file_summaries,
        "errors": errors,
        "valid": not errors,
    }


def _has_forbidden_allowance(text: str) -> bool:
    """Detect positive claims that pig_id is biological identity across videos."""

    for phrase in FORBIDDEN_IDENTITY_PHRASES:
        if phrase in text and not _near_negation(text, phrase):
            return True
    return False


def _near_negation(text: str, phrase: str) -> bool:
    index = text.find(phrase)
    if index < 0:
        return False
    prefix = text[max(0, index - 80) : index]
    return any(token in prefix for token in ["not", "never", "no ", "without"])


if __name__ == "__main__":
    main()
