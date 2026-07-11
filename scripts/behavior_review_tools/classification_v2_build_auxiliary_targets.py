from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

POSTURE_LABELS = {"lying", "sitting"}
MOTION_LABELS = {"move", "explore", "stand"}
ROI_LABELS = {"eat", "drink", "playwithtoy"}
INTERACTION_LABELS = {"fight", "social-nose"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build classification_v2 auxiliary multi-task targets.")
    parser.add_argument("--root", type=Path, default=Path("outputs/classification_v2/train_ready_windows"))
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--audit-json", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    labels = pd.read_csv(root / "y_behavior.csv").iloc[:, 0].fillna("").astype(str)
    split = pd.read_csv(root / "split_manifest.csv", low_memory=False)
    train_mask = _read_bool(root / "train_mask.csv")
    if not (len(labels) == len(split) == len(train_mask)):
        raise ValueError(
            f"row count mismatch labels={len(labels)} split={len(split)} train_mask={len(train_mask)}"
        )

    valid_behavior = labels.isin(VALID_BEHAVIORS)
    targets = pd.DataFrame(
        {
            "window_id": split["window_id"].astype(str),
            "behavior_target": labels,
            "posture_target": labels.map(_posture_target),
            "motion_context_target": labels.map(_motion_context_target),
            "roi_intent_target": labels.map(_roi_intent_target),
            "interaction_target": labels.map(_interaction_target),
            # Every valid behavior has one well-defined class in every
            # hierarchy. Fold-local weights handle dominant none/other classes.
            "has_posture_aux_target": valid_behavior,
            "has_motion_context_aux_target": valid_behavior,
            "has_roi_intent_aux_target": valid_behavior,
            "has_interaction_aux_target": valid_behavior,
            "aux_include_in_training": train_mask,
        }
    )
    audit = _audit(targets)

    output_csv = args.output_csv or (root / "y_auxiliary_targets.csv")
    audit_json = args.audit_json or (root / "auxiliary_targets_audit.json")
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


def _posture_target(label: str) -> str:
    return label if label in POSTURE_LABELS else "standing_or_other"


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
        "motion_context_counts": targets["motion_context_target"].value_counts(dropna=False).to_dict(),
        "roi_intent_counts": targets["roi_intent_target"].value_counts(dropna=False).to_dict(),
        "interaction_counts": targets["interaction_target"].value_counts(dropna=False).to_dict(),
        "aux_target_active_counts": {
            "posture": int(targets["has_posture_aux_target"].sum()),
            "motion_context": int(targets["has_motion_context_aux_target"].sum()),
            "roi_intent": int(targets["has_roi_intent_aux_target"].sum()),
            "interaction": int(targets["has_interaction_aux_target"].sum()),
        },
        "aux_target_positive_counts": {
            "posture": int(targets["behavior_target"].isin(POSTURE_LABELS).sum()),
            "motion_context": int(targets["behavior_target"].isin(MOTION_LABELS).sum()),
            "roi_intent": int(targets["behavior_target"].isin(ROI_LABELS).sum()),
            "interaction": int(targets["behavior_target"].isin(INTERACTION_LABELS).sum()),
        },
        "errors": errors,
        "warnings": [
            "auxiliary targets are y/mask artifacts; do not include them in model input X",
            "auxiliary classes are deterministic decompositions of behavior y, not independent annotations",
            "all valid behaviors supervise every hierarchy, including none/other negative classes",
            "stand maps to motion_context_target=stand, not posture_target",
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
