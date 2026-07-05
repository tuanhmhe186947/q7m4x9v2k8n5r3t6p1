from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VALID_STRENGTH = {"strong", "medium", "weak", "boundary"}
VALID_BEHAVIORS = {
    "drink",
    "eat",
    "fight",
    "social-nose",
    "explore",
    "lying",
    "stand",
    "move",
    "sitting",
    "playwithtoy",
}
DEFAULT_WEIGHT = {"strong": 1.0, "medium": 0.75, "weak": 0.35, "boundary": 0.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply behavior strength/manual review decisions and export training-ready CSV."
    )
    parser.add_argument("--annotated-csv", type=Path, required=True, help="CSV from build_behavior_review_templates.py with auto review attributes.")
    parser.add_argument("--review-decisions-csv", type=Path, default=None, help="Filled review template or GUI decisions CSV.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument(
        "--pending-policy",
        choices=["exclude", "auto"],
        default="exclude",
        help="How to handle rows that auto require review but have no manual decision.",
    )
    parser.add_argument(
        "--include-weak-in-training",
        action="store_true",
        help="Keep weak samples in final training with low weight. Default excludes weak from main training.",
    )
    parser.add_argument(
        "--keep-all-columns",
        action="store_true",
        help="Keep all source columns. Default also keeps all columns, this flag is present for clarity/future use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.annotated_csv, low_memory=False)

    if "review_key" not in df.columns:
        raise ValueError("annotated CSV must contain review_key. Run build_behavior_review_templates.py first.")

    decisions = load_decisions(args.review_decisions_csv) if args.review_decisions_csv else pd.DataFrame()
    out = apply_decisions(
        df,
        decisions,
        pending_policy=args.pending_policy,
        include_weak_in_training=args.include_weak_in_training,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    audit = build_audit(out, decisions, args)
    audit_path = args.audit_json or args.output_csv.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"rows: {len(out)}")
    print(f"training rows: {int(out['include_in_training_final'].sum())}")
    print(f"output: {args.output_csv}")
    print(f"audit: {audit_path}")


def load_decisions(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    if "review_key" not in df.columns:
        raise ValueError("review decisions CSV must contain review_key")
    return df


def apply_decisions(
    df: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    pending_policy: str,
    include_weak_in_training: bool,
) -> pd.DataFrame:
    out = df.copy()

    # Defaults from automatic attributes.
    out["label_strength"] = clean_strength(out.get("label_strength_auto", "medium"))
    out["ambiguity_group"] = out.get("ambiguity_group_auto", "general").fillna("general").astype(str)
    out["review_reason"] = out.get("review_reason_auto", "").fillna("").astype(str)
    out["review_decision"] = np.where(
        to_bool(out.get("review_required_auto", False)),
        "pending_review",
        "auto_accept",
    )
    out["behavior_train"] = out.get("behavior", "").fillna("").astype(str)
    out["manual_review_applied"] = False

    if not decisions.empty:
        dec = normalize_decisions(decisions)
        out = out.merge(dec, on="review_key", how="left", suffixes=("", "__decision"))

        has_manual = out["manual_review_decision"].fillna("").astype(str).str.len() > 0
        out.loc[has_manual, "manual_review_applied"] = True

        strength = clean_strength(out.get("manual_label_strength", ""))
        use_strength = has_manual & strength.isin(VALID_STRENGTH)
        out.loc[use_strength, "label_strength"] = strength[use_strength]

        corrected = out.get("manual_corrected_behavior", "").fillna("").astype(str).str.strip()
        use_corrected = has_manual & corrected.isin(VALID_BEHAVIORS)
        out.loc[use_corrected, "behavior_train"] = corrected[use_corrected]

        group = out.get("manual_ambiguity_group", "").fillna("").astype(str).str.strip()
        use_group = has_manual & group.ne("")
        out.loc[use_group, "ambiguity_group"] = group[use_group]

        note = out.get("manual_note", "").fillna("").astype(str).str.strip()
        out.loc[has_manual & note.ne(""), "review_reason"] = note[has_manual & note.ne("")]

        decision = out.get("manual_review_decision", "").fillna("").astype(str).str.strip()
        out.loc[has_manual, "review_decision"] = decision[has_manual]

        manual_weight = pd.to_numeric(out.get("manual_sample_weight", np.nan), errors="coerce")
        out["manual_sample_weight_num"] = manual_weight
    else:
        out["manual_sample_weight_num"] = np.nan
        out["manual_training_action"] = ""

    out["training_weight_final"] = out["label_strength"].map(DEFAULT_WEIGHT).fillna(0.5).astype(float)
    has_manual_weight = out["manual_sample_weight_num"].notna()
    out.loc[has_manual_weight, "training_weight_final"] = out.loc[has_manual_weight, "manual_sample_weight_num"].clip(0, 1)

    review_required = to_bool(out.get("review_required_auto", False))
    pending = review_required & ~to_bool(out.get("manual_review_applied", False))

    include = to_bool(out.get("include_in_training", True))
    include &= out["behavior_train"].isin(VALID_BEHAVIORS)
    include &= out["label_strength"].isin(VALID_STRENGTH)

    if pending_policy == "exclude":
        include &= ~pending
    elif pending_policy == "auto":
        pass
    else:
        raise ValueError(pending_policy)

    reject_terms = {"reject", "exclude", "wrong_label", "remove", "boundary_exclude"}
    decision_lower = out["review_decision"].fillna("").astype(str).str.lower()
    include &= ~decision_lower.isin(reject_terms)
    include &= out["label_strength"].ne("boundary")
    if not include_weak_in_training:
        include &= out["label_strength"].ne("weak")

    out["include_in_training_final"] = include
    out.loc[~include, "training_weight_final"] = 0.0

    out["use_for_main_train_final"] = include & out["label_strength"].isin(["strong", "medium"])
    out["use_for_robust_train_final"] = include | (
        include_weak_in_training & out["label_strength"].eq("weak") & out["behavior_train"].isin(VALID_BEHAVIORS)
    )

    # ROI-specific training remains conservative for unresolved ROI conflicts.
    if "roi_consistency_status_auto" in out.columns:
        roi_conflict = out["roi_consistency_status_auto"].isin(["target_roi_far", "target_roi_unavailable"])
        unresolved = roi_conflict & ~to_bool(out.get("manual_review_applied", False))
        if "use_for_roi_training" in out.columns:
            out["use_for_roi_training_final"] = to_bool(out["use_for_roi_training"]) & ~unresolved & out["include_in_training_final"]
        else:
            out["use_for_roi_training_final"] = ~unresolved & out["include_in_training_final"]

    return out


def normalize_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "review_key",
        "manual_review_decision",
        "manual_label_strength",
        "manual_corrected_behavior",
        "manual_ambiguity_group",
        "manual_training_action",
        "manual_sample_weight",
        "manual_note",
    ]
    out = decisions.copy()
    rename_map = {
        "decision": "manual_review_decision",
        "new_label_strength": "manual_label_strength",
        "corrected_behavior": "manual_corrected_behavior",
        "new_ambiguity_group": "manual_ambiguity_group",
        "training_action": "manual_training_action",
        "sample_weight": "manual_sample_weight",
        "note": "manual_note",
    }
    for old, new in rename_map.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]
    for col in keep_cols:
        if col not in out.columns:
            out[col] = ""
    out = out[keep_cols].copy()
    out = out.drop_duplicates("review_key", keep="last")
    return out


def clean_strength(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        s = values.fillna("").astype(str).str.strip().str.lower()
    else:
        s = pd.Series(values).fillna("").astype(str).str.strip().str.lower()
    s = s.where(s.isin(VALID_STRENGTH), "medium")
    return s


def to_bool(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.fillna(False).astype(str).str.lower().isin(["true", "1", "yes", "y"])
    return pd.Series(values).fillna(False).astype(str).str.lower().isin(["true", "1", "yes", "y"])


def build_audit(out: pd.DataFrame, decisions: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "annotated_csv": str(args.annotated_csv),
        "review_decisions_csv": str(args.review_decisions_csv) if args.review_decisions_csv else "",
        "rows": int(len(out)),
        "manual_decisions": int(len(decisions)) if not decisions.empty else 0,
        "pending_policy": args.pending_policy,
        "include_weak_in_training": bool(args.include_weak_in_training),
        "include_in_training_final": value_counts(out, "include_in_training_final"),
        "label_strength": value_counts(out, "label_strength"),
        "behavior_train": value_counts(out, "behavior_train"),
        "review_decision": value_counts(out, "review_decision"),
        "ambiguity_group": value_counts(out, "ambiguity_group", top=50),
        "training_weight_final": {
            "sum": float(out["training_weight_final"].sum()),
            "mean": float(out["training_weight_final"].mean()),
        },
    }


def value_counts(df: pd.DataFrame, col: str, top: int | None = None) -> dict[str, int]:
    if col not in df.columns:
        return {}
    counts = df[col].fillna("").astype(str).value_counts(dropna=False)
    if top is not None:
        counts = counts.head(top)
    return {str(k): int(v) for k, v in counts.items()}


if __name__ == "__main__":
    main()
