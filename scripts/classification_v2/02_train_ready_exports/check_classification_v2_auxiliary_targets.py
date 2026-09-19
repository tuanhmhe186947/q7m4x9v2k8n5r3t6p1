from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.behavior_posture import (
    BEHAVIOR_POSTURE_CONTRACT_VERSION,
    POSTURE_AUTHORITY_VALUES,
    POSTURE_LABEL_SET,
    SAFE_DERIVATION_BEHAVIOR_AUTHORITIES,
    SAFE_POSTURE_BY_BEHAVIOR,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check classification_v2 auxiliary target artifacts."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows/y_auxiliary_targets.csv"),
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows/auxiliary_targets_audit.json"),
    )
    args = parser.parse_args()

    errors: list[str] = []
    if not args.csv.exists():
        errors.append(f"missing_csv={args.csv}")
        targets = pd.DataFrame()
    else:
        targets = pd.read_csv(args.csv, low_memory=False)
    if not args.audit_json.exists():
        errors.append(f"missing_audit={args.audit_json}")
        audit = {}
    else:
        audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
        errors.extend(audit.get("errors", []))

    required_cols = {
        "window_id",
        "behavior_target",
        "behavior_label_authority",
        "posture_target",
        "posture_authority",
        "posture_authority_version",
        "posture_transition_flag",
        "motion_context_target",
        "roi_intent_target",
        "interaction_target",
        "has_posture_aux_target",
        "has_motion_context_aux_target",
        "has_roi_intent_aux_target",
        "has_interaction_aux_target",
        "aux_include_in_training",
    }
    missing_cols = sorted(required_cols.difference(targets.columns))
    if missing_cols:
        errors.append(f"missing_cols={missing_cols}")
    if "window_id" in targets and int(targets["window_id"].duplicated().sum()):
        errors.append("duplicate_window_id")
    if "behavior_target" in targets:
        valid_behavior = targets["behavior_target"].astype(str).isin(VALID_BEHAVIORS)
        for mask_col in [
            "has_motion_context_aux_target",
            "has_roi_intent_aux_target",
            "has_interaction_aux_target",
        ]:
            mask = _to_bool(targets[mask_col])
            false_valid = int((valid_behavior & ~mask).sum())
            true_invalid = int((~valid_behavior & mask).sum())
            if false_valid or true_invalid:
                errors.append(
                    f"invalid_aux_mask_semantics={mask_col}:"
                    f"false_valid={false_valid}:true_invalid={true_invalid}"
                )
        posture_mask = _to_bool(targets["has_posture_aux_target"])
        posture_target = targets["posture_target"].fillna("").astype(str)
        invalid_posture = posture_mask & ~posture_target.isin(POSTURE_LABEL_SET)
        target_while_masked = ~posture_mask & posture_target.ne("")
        if invalid_posture.any() or target_while_masked.any():
            errors.append(
                "invalid_posture_target_mask_contract:"
                f"invalid_valid={int(invalid_posture.sum())}:"
                f"target_while_masked={int(target_while_masked.sum())}"
            )
        expected_posture = targets["behavior_target"].map(SAFE_POSTURE_BY_BEHAVIOR)
        safe_rows = expected_posture.notna()
        safe_mismatch = safe_rows & (
            ~posture_mask | posture_target.ne(expected_posture.fillna(""))
        )
        if safe_mismatch.any():
            errors.append(f"safe_posture_mismatch={int(safe_mismatch.sum())}")
        invalid_authority = ~targets["posture_authority"].isin(
            POSTURE_AUTHORITY_VALUES
        )
        if invalid_authority.any():
            errors.append(
                f"invalid_posture_authority={int(invalid_authority.sum())}"
            )
        invalid_version = targets["posture_authority_version"].ne(
            BEHAVIOR_POSTURE_CONTRACT_VERSION
        )
        if invalid_version.any():
            errors.append(
                f"invalid_posture_authority_version={int(invalid_version.sum())}"
            )
        invalid_behavior_authority = ~targets["behavior_label_authority"].isin(
            SAFE_DERIVATION_BEHAVIOR_AUTHORITIES
        )
        if invalid_behavior_authority.any():
            errors.append(
                "invalid_behavior_label_authority="
                f"{int(invalid_behavior_authority.sum())}"
            )
        transition = _to_bool(targets["posture_transition_flag"])
        if (transition & posture_mask).any():
            errors.append("posture_transition_rows_must_be_masked")
        fight_in_motion = int(
            (
                targets["behavior_target"].eq("fight")
                & targets["motion_context_target"].eq("fight")
            ).sum()
        )
        playwithtoy_roi = int(
            (
                targets["behavior_target"].eq("playwithtoy")
                & targets["roi_intent_target"].eq("playwithtoy")
            ).sum()
        )
        if fight_in_motion:
            errors.append(f"fight_in_motion_context_target={fight_in_motion}")
        if playwithtoy_roi <= 0:
            errors.append("playwithtoy_missing_roi_intent_target")

    result = {
        "csv": str(args.csv),
        "audit_json": str(args.audit_json),
        "rows": int(len(targets)),
        "aux_target_active_counts": audit.get("aux_target_active_counts"),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


if __name__ == "__main__":
    main()
