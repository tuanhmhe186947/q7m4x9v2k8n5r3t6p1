"""Apply a predeclared audit gate to frozen or synthetic posture proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.posture_proposal import (
    PostureAutoValidationPolicy,
    evaluate_posture_auto_validation,
)
from pig_behavior.classification_v2.review.post_review_learning import (
    assert_not_active_behavior_ledger_path,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate posture proposals using a predeclared human audit."
    )
    parser.add_argument("--proposals-csv", type=Path, required=True)
    parser.add_argument("--human-audit-csv", type=Path, required=True)
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-audit-json", type=Path, required=True)
    args = parser.parse_args()

    paths = [
        args.proposals_csv,
        args.human_audit_csv,
        args.policy_json,
        args.output_csv,
        args.output_audit_json,
    ]
    for path in paths:
        assert_not_active_behavior_ledger_path(path)

    payload = json.loads(args.policy_json.read_text(encoding="utf-8"))
    policy = PostureAutoValidationPolicy(
        confidence_threshold=_required_number(payload, "confidence_threshold"),
        minimum_audit_rows_per_stratum=_required_integer(
            payload,
            "minimum_audit_rows_per_stratum",
        ),
        required_precision_lower_bound=_required_number(
            payload,
            "required_precision_lower_bound",
        ),
        one_sided_z=_required_number(payload, "one_sided_z"),
    )
    proposals = pd.read_csv(args.proposals_csv, low_memory=False)
    human_audit = pd.read_csv(args.human_audit_csv, low_memory=False)
    evaluated, audit = evaluate_posture_auto_validation(
        proposals,
        human_audit,
        policy=policy,
    )
    audit["inputs"] = {
        "proposals_csv": _path_record(args.proposals_csv),
        "human_audit_csv": _path_record(args.human_audit_csv),
        "policy_json": _path_record(args.policy_json),
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit_json.parent.mkdir(parents=True, exist_ok=True)
    evaluated.to_csv(args.output_csv, index=False)
    args.output_audit_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_csv": str(args.output_csv),
                "output_audit_json": str(args.output_audit_json),
                "auto_validated_rows": audit["auto_validated_rows"],
                "review_required_rows": audit["review_required_rows"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _required_number(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if value is None or isinstance(value, bool):
        raise ValueError(f"policy value must be predeclared: {key}")
    return float(value)


def _required_integer(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if value is None or isinstance(value, bool) or int(value) != float(value):
        raise ValueError(f"policy integer must be predeclared: {key}")
    return int(value)


def _path_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


if __name__ == "__main__":
    main()
