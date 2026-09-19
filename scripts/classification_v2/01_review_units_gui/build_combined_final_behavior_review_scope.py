"""Combine the corrected non-interaction view with full interaction census."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA_VERSION = "classification_v2.combined_final_behavior_scope.v1"
EXPECTED_CANDIDATES = 6061
EXPECTED_NONINTERACTION = 1738
EXPECTED_INTERACTION = 991
EXPECTED_COMBINED = 2729
EXPECTED_CALIBRATION_DECISIONS = 300
FULL_CENSUS_DECISION = "DECISION_B_FULL_INTERACTION_CENSUS"
INTERACTION_BEHAVIORS = frozenset({"fight", "social-nose"})
TRUE_VALUES = frozenset({"1", "true", "yes", "y"})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key_set_sha256(values: pd.Series) -> str:
    return hashlib.sha256(
        "\n".join(sorted(values.astype(str))).encode("utf-8")
    ).hexdigest()


def _truth_series(values: pd.Series) -> pd.Series:
    return values.fillna(False).astype(str).str.strip().str.casefold().isin(
        TRUE_VALUES
    )


def validate_full_census_decision(
    decision: dict[str, Any],
    *,
    calibration_ledger_sha256: str,
) -> None:
    errors: list[str] = []
    if decision.get("post_calibration_decision") != FULL_CENSUS_DECISION:
        errors.append("interaction decision is not full census")
    if decision.get("selected_rule_id") is not None:
        errors.append("full census cannot select a screening rule")
    if bool(decision.get("confirmation_authorized")):
        errors.append("confirmation must remain closed")
    if str(decision.get("ledger_sha256", "")) != calibration_ledger_sha256:
        errors.append("calibration ledger hash mismatch")
    if errors:
        raise ValueError("; ".join(errors))


def build_combined_scope(
    candidates: pd.DataFrame,
    corrected_noninteraction: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "review_unit_id",
        "review_template",
        "requires_partner_context",
        "behavior_label",
    }
    missing_candidates = sorted(required.difference(candidates.columns))
    if missing_candidates:
        raise ValueError(
            f"candidate manifest missing columns={missing_candidates}"
        )
    if "review_unit_id" not in corrected_noninteraction:
        raise ValueError("corrected non-interaction view missing review_unit_id")

    candidate_ids = candidates["review_unit_id"].astype(str)
    noninteraction_ids = corrected_noninteraction["review_unit_id"].astype(str)
    if candidate_ids.eq("").any() or candidate_ids.duplicated().any():
        raise ValueError("candidate keys must be unique and nonblank")
    if noninteraction_ids.eq("").any() or noninteraction_ids.duplicated().any():
        raise ValueError("non-interaction keys must be unique and nonblank")
    unknown = set(noninteraction_ids).difference(candidate_ids)
    if unknown:
        raise ValueError(
            f"non-interaction keys outside candidate authority={len(unknown)}"
        )

    template_ids = set(
        candidate_ids[candidates["review_template"].astype(str).eq("interaction")]
    )
    behavior_ids = set(
        candidate_ids[candidates["behavior_label"].isin(INTERACTION_BEHAVIORS)]
    )
    partner_ids = set(
        candidate_ids[_truth_series(candidates["requires_partner_context"])]
    )
    if not (template_ids == behavior_ids == partner_ids):
        raise ValueError("interaction template/label/partner partitions differ")

    overlap = set(noninteraction_ids).intersection(template_ids)
    if overlap:
        raise ValueError(
            f"non-interaction and interaction scopes overlap={len(overlap)}"
        )

    by_id = candidates.assign(review_unit_id=candidate_ids).set_index(
        "review_unit_id",
        drop=False,
    )
    noninteraction = by_id.loc[noninteraction_ids.tolist()].copy()
    interaction = candidates.loc[candidate_ids.isin(template_ids)].copy()
    noninteraction["final_scope_component"] = (
        "ROI_DIRECTION_CORRECTED_NONINTERACTION"
    )
    interaction["final_scope_component"] = (
        "POST_CALIBRATION_FULL_INTERACTION_CENSUS"
    )
    combined = pd.concat(
        [noninteraction.reset_index(drop=True), interaction],
        ignore_index=True,
        sort=False,
    )
    combined["final_scope_schema_version"] = SCHEMA_VERSION
    combined["final_scope_interaction_decision"] = FULL_CENSUS_DECISION

    audit = {
        "candidate_count": int(len(candidates)),
        "corrected_noninteraction_count": int(len(noninteraction)),
        "full_interaction_census_count": int(len(interaction)),
        "combined_count": int(len(combined)),
        "interaction_behavior_counts": {
            str(key): int(value)
            for key, value in interaction["behavior_label"]
            .value_counts()
            .sort_index()
            .items()
        },
        "component_overlap_count": 0,
        "unknown_noninteraction_key_count": 0,
        "interaction_partition_concordant": True,
        "candidate_membership_changed": False,
        "auto_carry_membership_changed": False,
        "confirmation_authorized": False,
    }
    return combined, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest-csv", type=Path, required=True)
    parser.add_argument(
        "--corrected-noninteraction-view-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--calibration-decisions-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--post-calibration-decision-json",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-scope-csv", type=Path, required=True)
    parser.add_argument("--output-audit-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = (args.output_scope_csv, args.output_audit_json)
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite combined scope artifacts")

    candidates = pd.read_csv(args.candidate_manifest_csv, low_memory=False)
    noninteraction = pd.read_csv(
        args.corrected_noninteraction_view_csv,
        low_memory=False,
    )
    decisions = pd.read_csv(
        args.calibration_decisions_csv,
        usecols=["review_key"],
        low_memory=False,
    )
    if len(decisions) != EXPECTED_CALIBRATION_DECISIONS:
        raise ValueError(
            "calibration decision count mismatch "
            f"expected={EXPECTED_CALIBRATION_DECISIONS} actual={len(decisions)}"
        )
    decision_keys = decisions["review_key"].fillna("").astype(str).str.strip()
    if decision_keys.eq("").any() or decision_keys.duplicated().any():
        raise ValueError("calibration decision keys must be unique and nonblank")
    decision_payload = json.loads(
        args.post_calibration_decision_json.read_text(encoding="utf-8")
    )
    ledger_sha256 = _sha256(args.calibration_decisions_csv)
    validate_full_census_decision(
        decision_payload,
        calibration_ledger_sha256=ledger_sha256,
    )

    combined, audit = build_combined_scope(candidates, noninteraction)
    observed = {
        "candidate_count": len(candidates),
        "corrected_noninteraction_count": audit[
            "corrected_noninteraction_count"
        ],
        "full_interaction_census_count": audit[
            "full_interaction_census_count"
        ],
        "combined_count": len(combined),
    }
    expected = {
        "candidate_count": EXPECTED_CANDIDATES,
        "corrected_noninteraction_count": EXPECTED_NONINTERACTION,
        "full_interaction_census_count": EXPECTED_INTERACTION,
        "combined_count": EXPECTED_COMBINED,
    }
    if observed != expected:
        raise ValueError(f"frozen scope count mismatch={observed}")

    args.output_scope_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_scope_csv, index=False, lineterminator="\n")
    audit.update(
        {
            "schema_version": SCHEMA_VERSION,
            "authority_role": "COMBINED_FINAL_BEHAVIOR_REVIEW_SCOPE",
            "expected_counts": expected,
            "calibration_decision": FULL_CENSUS_DECISION,
            "calibration_decision_count": int(len(decisions)),
            "calibration_ledger_sha256": ledger_sha256,
            "post_calibration_decision_sha256": _sha256(
                args.post_calibration_decision_json
            ),
            "candidate_manifest_sha256": _sha256(
                args.candidate_manifest_csv
            ),
            "corrected_noninteraction_view_sha256": _sha256(
                args.corrected_noninteraction_view_csv
            ),
            "combined_scope_sha256": _sha256(args.output_scope_csv),
            "combined_key_set_sha256": _key_set_sha256(
                combined["review_unit_id"]
            ),
            "gui_opened": False,
            "decisions_written": False,
        }
    )
    args.output_audit_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "COMBINED_FINAL_BEHAVIOR_SCOPE "
        f"noninteraction={EXPECTED_NONINTERACTION} "
        f"interaction={EXPECTED_INTERACTION} "
        f"total={EXPECTED_COMBINED}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
