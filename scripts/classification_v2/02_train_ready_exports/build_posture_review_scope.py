"""Build a deterministic selective posture-review scope from model proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.behavior_posture import (
    SAFE_DERIVATION_BEHAVIOR_AUTHORITIES,
)
from pig_behavior.classification_v2.contracts.posture_proposal import (
    PostureReviewScopePolicy,
    build_posture_review_scope,
)
from pig_behavior.classification_v2.review.post_review_learning import (
    assert_not_active_behavior_ledger_path,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select mandatory posture reviews and deterministic upright controls."
        )
    )
    parser.add_argument("--proposals-csv", type=Path, required=True)
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument(
        "--behavior-label-authority",
        required=True,
        choices=sorted(SAFE_DERIVATION_BEHAVIOR_AUTHORITIES),
    )
    parser.add_argument("--output-review-scope-csv", type=Path, required=True)
    parser.add_argument("--output-audit-json", type=Path, required=True)
    args = parser.parse_args()

    paths = [
        args.proposals_csv,
        args.policy_json,
        args.output_review_scope_csv,
        args.output_audit_json,
    ]
    for path in paths:
        assert_not_active_behavior_ledger_path(path)

    payload = json.loads(args.policy_json.read_text(encoding="utf-8"))
    if (
        args.behavior_label_authority == "FROZEN_HUMAN_REVIEWED"
        and not _required_bool(payload, "real_execution_authorized")
    ):
        raise ValueError(
            "real posture-review scope requires real_execution_authorized=true"
        )

    policy = PostureReviewScopePolicy(
        confidence_threshold=_required_number(payload, "confidence_threshold"),
        upright_control_rows_per_stratum=_required_integer(
            payload,
            "upright_control_rows_per_stratum",
        ),
        seed=_required_integer(payload, "review_scope_seed"),
    )
    proposals = pd.read_csv(args.proposals_csv, low_memory=False)
    scope, audit = build_posture_review_scope(proposals, policy=policy)
    audit["behavior_label_authority"] = args.behavior_label_authority
    audit["inputs"] = {
        "proposals": _path_record(args.proposals_csv),
        "policy": _path_record(args.policy_json),
    }

    args.output_review_scope_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit_json.parent.mkdir(parents=True, exist_ok=True)
    scope.to_csv(args.output_review_scope_csv, index=False)
    audit["output_review_scope"] = _path_record(args.output_review_scope_csv)
    args.output_audit_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_review_scope_csv": str(args.output_review_scope_csv),
                "output_audit_json": str(args.output_audit_json),
                "scope_rows": audit["scope_rows"],
                "mandatory_rows": audit["mandatory_rows"],
                "upright_control_rows": audit["upright_control_rows"],
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


def _required_bool(payload: dict[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"policy boolean must be predeclared: {key}")
    return value


def _path_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


if __name__ == "__main__":
    main()
