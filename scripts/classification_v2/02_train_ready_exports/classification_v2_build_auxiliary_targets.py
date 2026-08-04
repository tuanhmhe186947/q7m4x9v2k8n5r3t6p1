from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.behavior_posture import (
    BEHAVIOR_POSTURE_CONTRACT_VERSION,
    SAFE_DERIVATION_BEHAVIOR_AUTHORITIES,
    SAFE_POSTURE_BY_BEHAVIOR,
    align_window_posture_authority,
)
from pig_behavior.classification_v2.review.post_review_learning import (
    assert_not_active_behavior_ledger_path,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

MOTION_LABELS = {"move", "explore", "stand"}
ROI_LABELS = {"eat", "drink", "playwithtoy"}
INTERACTION_LABELS = {"fight", "social-nose"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build classification_v2 auxiliary multi-task targets."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows"),
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--posture-window-authority-csv", type=Path, default=None)
    parser.add_argument(
        "--behavior-label-authority",
        required=True,
        choices=sorted(SAFE_DERIVATION_BEHAVIOR_AUTHORITIES),
    )
    args = parser.parse_args()

    root = args.root
    assert_not_active_behavior_ledger_path(root)
    if args.posture_window_authority_csv is not None:
        assert_not_active_behavior_ledger_path(args.posture_window_authority_csv)
    labels = pd.read_csv(root / "y_behavior.csv").iloc[:, 0].fillna("").astype(str)
    split = pd.read_csv(root / "split_manifest.csv", low_memory=False)
    train_mask = _read_bool(root / "train_mask.csv")
    if not (len(labels) == len(split) == len(train_mask)):
        raise ValueError(
            f"row count mismatch labels={len(labels)} split={len(split)} "
            f"train_mask={len(train_mask)}"
        )

    valid_behavior = labels.isin(VALID_BEHAVIORS)
    posture = _resolve_posture_authority(
        split["window_id"].astype(str),
        labels,
        args.posture_window_authority_csv,
        args.behavior_label_authority,
    )
    targets = pd.DataFrame(
        {
            "window_id": split["window_id"].astype(str),
            "behavior_target": labels,
            "behavior_label_authority": posture["behavior_label_authority"],
            "posture_target": posture["posture_target"],
            "posture_authority": posture["posture_authority"],
            "posture_authority_version": posture["posture_authority_version"],
            "posture_transition_flag": posture["posture_transition_flag"],
            "motion_context_target": labels.map(_motion_context_target),
            "roi_intent_target": labels.map(_roi_intent_target),
            "interaction_target": labels.map(_interaction_target),
            "has_posture_aux_target": posture["posture_valid_mask"],
            "has_motion_context_aux_target": valid_behavior,
            "has_roi_intent_aux_target": valid_behavior,
            "has_interaction_aux_target": valid_behavior,
            "aux_include_in_training": train_mask,
        }
    )
    audit = _audit(targets)

    output_csv = args.output_csv or (root / "y_auxiliary_targets.csv")
    audit_json = args.audit_json or (root / "auxiliary_targets_audit.json")
    assert_not_active_behavior_ledger_path(output_csv)
    assert_not_active_behavior_ledger_path(audit_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output_csv, index=False)
    audit_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_csv": str(output_csv),
                "audit_json": str(audit_json),
                "audit": audit,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if audit["errors"]:
        raise SystemExit(1)


def _resolve_posture_authority(
    window_ids: pd.Series,
    labels: pd.Series,
    authority_csv: Path | None,
    behavior_label_authority: str,
) -> pd.DataFrame:
    windows = pd.DataFrame(
        {
            "window_id": window_ids.astype(str),
            "behavior_target": labels.astype(str),
        }
    )
    if authority_csv is not None:
        assert_not_active_behavior_ledger_path(authority_csv)
        authority = pd.read_csv(authority_csv, low_memory=False)
        aligned = align_window_posture_authority(windows, authority)
        observed = set(aligned["behavior_label_authority"].astype(str))
        if observed != {behavior_label_authority}:
            raise ValueError(
                "behavior authority mismatch between CLI and posture authority: "
                f"cli={behavior_label_authority}, observed={sorted(observed)}"
            )
        return aligned

    posture = labels.map(SAFE_POSTURE_BY_BEHAVIOR)
    return pd.DataFrame(
        {
            "behavior_label_authority": behavior_label_authority,
            "posture_target": posture.fillna(""),
            "posture_valid_mask": posture.notna(),
            "posture_transition_flag": False,
            "posture_authority": posture.map(
                lambda value: "DERIVED_SAFE" if pd.notna(value) else "UNRESOLVED"
            ),
            "posture_authority_version": BEHAVIOR_POSTURE_CONTRACT_VERSION,
        }
    )


def _motion_context_target(label: str) -> str:
    return label if label in MOTION_LABELS else "other"


def _roi_intent_target(label: str) -> str:
    return label if label in ROI_LABELS else "none"


def _interaction_target(label: str) -> str:
    return label if label in INTERACTION_LABELS else "none"


def _audit(targets: pd.DataFrame) -> dict[str, object]:
    duplicate_window_id = int(targets["window_id"].duplicated().sum())
    errors: list[str] = []
    if duplicate_window_id:
        errors.append(f"duplicate_window_id={duplicate_window_id}")
    return {
        "rows": int(len(targets)),
        "duplicate_window_id": duplicate_window_id,
        "behavior_counts": targets["behavior_target"].value_counts(dropna=False).to_dict(),
        "posture_counts": targets["posture_target"].value_counts(dropna=False).to_dict(),
        "motion_context_counts": (
            targets["motion_context_target"].value_counts(dropna=False).to_dict()
        ),
        "roi_intent_counts": targets["roi_intent_target"].value_counts(dropna=False).to_dict(),
        "interaction_counts": targets["interaction_target"].value_counts(dropna=False).to_dict(),
        "aux_target_active_counts": {
            "posture": int(targets["has_posture_aux_target"].sum()),
            "motion_context": int(targets["has_motion_context_aux_target"].sum()),
            "roi_intent": int(targets["has_roi_intent_aux_target"].sum()),
            "interaction": int(targets["has_interaction_aux_target"].sum()),
        },
        "aux_target_positive_counts": {
            "posture": int(targets["has_posture_aux_target"].sum()),
            "motion_context": int(targets["behavior_target"].isin(MOTION_LABELS).sum()),
            "roi_intent": int(targets["behavior_target"].isin(ROI_LABELS).sum()),
            "interaction": int(targets["behavior_target"].isin(INTERACTION_LABELS).sum()),
        },
        "errors": errors,
        "warnings": [
            "auxiliary targets are y/mask artifacts; do not include them in model input X",
            "posture is an independent masked y target and never model input X",
            "unresolved posture rows remain behavior-eligible with posture mask false",
            "safe posture derivation is limited to lying, sitting, stand, and eat",
            "fight maps to interaction_target=fight, not motion_context_target",
        ],
    }


def _read_bool(path: Path) -> pd.Series:
    series = pd.read_csv(path).iloc[:, 0]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


if __name__ == "__main__":
    main()
