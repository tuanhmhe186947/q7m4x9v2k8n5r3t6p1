"""Apply classification_v2 review-unit GUI decisions to frame-level features.

Design rules:
- Review units are human-review units, not training windows.
- legacy_recovered decisions apply to the full legacy 16-frame burst.
- cvat_tracking_xml decisions apply to the 6-frame CVAT anchor interval.
- No rows are dropped. Exclusion is represented by explicit flags/actions.
- Corrected labels update the canonical `behavior` column in the reviewed output,
  while preserving `behavior_before_review`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

CANONICAL_BEHAVIORS = {
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

VALID_DECISIONS = {"pending", "accept", "corrected", "exclude", "reject", "uncertain"}
VALID_ACTIONS = {"", "main_train", "keep", "correct_and_keep", "downweight", "exclude"}
DEFAULT_DECISION_FILES = [
    r"outputs\classification_v2\review_policy\roi_review_unit_gui_pilot\behavior_unit_review_decisions.csv",
    r"outputs\classification_v2\review_policy\motion_review_unit_gui_pilot\behavior_unit_review_decisions.csv",
    r"outputs\classification_v2\review_policy\posture_review_unit_gui_pilot\behavior_unit_review_decisions.csv",
    r"outputs\classification_v2\review_policy\interaction_review_unit_gui_pilot\behavior_unit_review_decisions.csv",
]


def _norm_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _to_bool_action(action: str, decision: str) -> bool:
    action = _norm_text(action)
    decision = _norm_text(decision)
    if decision in {"exclude", "reject"}:
        return False
    if action == "exclude":
        return False
    return True


def _default_action(decision: str) -> str:
    if decision == "accept":
        return "main_train"
    if decision == "corrected":
        return "correct_and_keep"
    if decision == "uncertain":
        return "downweight"
    if decision in {"exclude", "reject"}:
        return "exclude"
    return ""


def _default_weight(decision: str, action: str) -> float | None:
    if decision == "pending":
        return None
    if decision in {"exclude", "reject"} or action == "exclude":
        return 0.0
    if decision == "uncertain" or action == "downweight":
        return 0.5
    return 1.0


def _validate_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def load_decisions(paths: list[Path], review_unit_manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    parts: list[pd.DataFrame] = []
    missing_files: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []

    manifest_cols = [
        "review_unit_id",
        "temporal_unit_key",
        "review_template",
        "behavior_label",
        "source_type",
        "review_unit_type",
        "unit_start_frame",
        "unit_end_frame",
    ]
    manifest_cols = [c for c in manifest_cols if c in review_unit_manifest.columns]
    manifest_small = review_unit_manifest[manifest_cols].drop_duplicates("review_unit_id").copy()

    for path in paths:
        if not path.exists():
            missing_files.append(str(path))
            continue
        df = pd.read_csv(path, low_memory=False)
        df["decision_source_csv"] = str(path)

        if "review_unit_id" not in df.columns:
            errors.append(f"{path} missing review_unit_id")
            continue

        if "behavior_label" not in df.columns:
            if "original_behavior" in df.columns:
                warnings.append(f"{path} missing behavior_label; using original_behavior compatibility")
                df["behavior_label"] = df["original_behavior"]
            else:
                warnings.append(f"{path} missing behavior_label/original_behavior; filling empty")
                df["behavior_label"] = ""

        if "original_behavior" not in df.columns:
            df["original_behavior"] = df["behavior_label"]

        # Fill missing metadata from canonical review unit manifest.
        df = df.merge(
            manifest_small,
            on="review_unit_id",
            how="left",
            suffixes=("", "_manifest"),
        )
        for col in [
            "temporal_unit_key",
            "review_template",
            "behavior_label",
            "source_type",
            "review_unit_type",
            "unit_start_frame",
            "unit_end_frame",
        ]:
            mcol = f"{col}_manifest"
            if col not in df.columns:
                df[col] = df[mcol] if mcol in df.columns else ""
            elif mcol in df.columns:
                df[col] = df[col].where(df[col].notna() & df[col].astype(str).ne(""), df[mcol])

        parts.append(df)

    if parts:
        decisions = pd.concat(parts, ignore_index=True)
    else:
        decisions = pd.DataFrame()

    audit = {
        "missing_files": missing_files,
        "load_errors": errors,
        "load_warnings": warnings,
    }
    return decisions, audit


def normalize_decisions(decisions: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if decisions.empty:
        return decisions, errors, warnings

    required_defaults = {
        "review_unit_id": "",
        "temporal_unit_key": "",
        "review_template": "",
        "behavior_label": "",
        "original_behavior": "",
        "manual_review_decision": "pending",
        "manual_corrected_behavior": "",
        "manual_label_strength": "",
        "manual_training_action": "",
        "manual_sample_weight": pd.NA,
        "manual_note": "",
    }
    for col, default in required_defaults.items():
        if col not in decisions.columns:
            decisions[col] = default

    for col in [
        "review_unit_id",
        "temporal_unit_key",
        "review_template",
        "behavior_label",
        "original_behavior",
        "manual_review_decision",
        "manual_corrected_behavior",
        "manual_label_strength",
        "manual_training_action",
        "manual_note",
    ]:
        decisions[col] = decisions[col].map(_norm_text)

    decisions["manual_review_decision"] = decisions["manual_review_decision"].replace("", "pending")
    decisions["manual_sample_weight"] = pd.to_numeric(decisions["manual_sample_weight"], errors="coerce")

    # Normalize actions and weights.
    for idx, row in decisions.iterrows():
        decision = row["manual_review_decision"]
        action = row["manual_training_action"]
        if action == "":
            action = _default_action(decision)
            decisions.at[idx, "manual_training_action"] = action
        if pd.isna(row["manual_sample_weight"]):
            default_weight = _default_weight(decision, action)
            if default_weight is not None:
                decisions.at[idx, "manual_sample_weight"] = default_weight

    pending = decisions["manual_review_decision"].eq("pending")
    decisions.loc[pending, "manual_corrected_behavior"] = ""
    decisions.loc[pending, "manual_training_action"] = ""
    decisions.loc[pending, "manual_sample_weight"] = pd.NA

    bad_decisions = sorted(set(decisions["manual_review_decision"]) - VALID_DECISIONS)
    if bad_decisions:
        warnings.append(f"unknown manual_review_decision values: {bad_decisions}")

    bad_actions = sorted(set(decisions["manual_training_action"].fillna("")) - VALID_ACTIONS)
    if bad_actions:
        warnings.append(f"unknown manual_training_action values: {bad_actions}")

    active = decisions[~decisions["manual_review_decision"].eq("pending")].copy()
    corrected = active[active["manual_review_decision"].eq("corrected")].copy()
    if not corrected.empty:
        invalid_corrected = corrected[
            ~corrected["manual_corrected_behavior"].isin(CANONICAL_BEHAVIORS)
        ]
        if len(invalid_corrected):
            errors.append(
                "corrected decisions with invalid/manual missing behavior: "
                + ", ".join(invalid_corrected["review_unit_id"].astype(str).head(10).tolist())
            )

    # Dedupe: if the same review_unit_id appears more than once, keep the last active row.
    # This supports repeated GUI pilot tests while still recording a warning.
    active_dups = int(active["review_unit_id"].duplicated(keep=False).sum()) if len(active) else 0
    if active_dups:
        warnings.append(f"duplicate active decisions rows={active_dups}; keeping last per review_unit_id")

    decisions["_decision_order"] = range(len(decisions))
    return decisions, errors, warnings


def apply_decisions_to_frames(
    frames: pd.DataFrame,
    review_units: pd.DataFrame,
    decisions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _validate_columns(frames, ["temporal_unit_key", "behavior"], "frame_features_csv")
    _validate_columns(review_units, ["review_unit_id", "temporal_unit_key"], "review_unit_manifest_csv")

    out = frames.copy()
    out["behavior_before_review"] = out["behavior"].fillna("").astype(str)
    out["behavior_after_review"] = out["behavior_before_review"]
    out["review_decision_applied"] = False
    out["review_unit_id_applied"] = ""
    out["review_template_applied"] = ""
    out["review_behavior_label"] = ""
    out["review_manual_decision"] = ""
    out["review_corrected_behavior"] = ""
    out["review_label_strength"] = ""
    out["review_training_action"] = ""
    out["review_sample_weight"] = pd.NA
    out["review_include_in_training"] = True
    out["review_note"] = ""

    if decisions.empty:
        return out, {
            "decisions_loaded": 0,
            "pending_ignored": 0,
            "active_decisions": 0,
            "applied_decisions": 0,
            "accepted_units": 0,
            "corrected_units": 0,
            "excluded_units": 0,
            "decision_frame_rows_touched": 0,
            "affected_frames": 0,
            "changed_behavior_frames": 0,
            "excluded_frames": 0,
            "duplicate_active_decision_rows": 0,
            "missing_review_unit_count": 0,
            "unmatched_decisions": [],
            "decision_counts": {},
            "training_action_counts": {},
        }

    active = decisions[~decisions["manual_review_decision"].eq("pending")].copy()
    pending_ignored = int(decisions["manual_review_decision"].eq("pending").sum())
    duplicate_active_rows = int(active["review_unit_id"].duplicated(keep=False).sum()) if len(active) else 0
    if active.empty:
        return out, {
            "decisions_loaded": int(len(decisions)),
            "pending_ignored": pending_ignored,
            "active_decisions": 0,
            "applied_decisions": 0,
            "accepted_units": 0,
            "corrected_units": 0,
            "excluded_units": 0,
            "decision_frame_rows_touched": 0,
            "affected_frames": 0,
            "changed_behavior_frames": 0,
            "excluded_frames": 0,
            "duplicate_active_decision_rows": duplicate_active_rows,
            "missing_review_unit_count": 0,
            "unmatched_decisions": [],
            "decision_counts": {},
            "training_action_counts": {},
        }

    # Keep last active row per unit.
    active = active.sort_values("_decision_order").drop_duplicates("review_unit_id", keep="last")

    unit_map = review_units[["review_unit_id", "temporal_unit_key"]].drop_duplicates("review_unit_id")
    active = active.merge(unit_map, on="review_unit_id", how="left", suffixes=("", "_unit"))
    active["target_temporal_unit_key"] = active["temporal_unit_key"].where(
        active["temporal_unit_key"].notna() & active["temporal_unit_key"].astype(str).ne(""),
        active["temporal_unit_key_unit"],
    )
    active["target_temporal_unit_key"] = active["target_temporal_unit_key"].map(_norm_text)

    unmatched: list[str] = []
    touched_total = 0
    applied_count = 0

    for _, row in active.iterrows():
        unit_id = row["review_unit_id"]
        target_key = row["target_temporal_unit_key"]
        if not target_key:
            unmatched.append(f"{unit_id}:missing_temporal_unit_key")
            continue

        mask = out["temporal_unit_key"].astype(str).eq(str(target_key))
        n = int(mask.sum())
        if n == 0:
            unmatched.append(f"{unit_id}:no_matching_frames")
            continue

        decision = row["manual_review_decision"]
        corrected = row["manual_corrected_behavior"]
        action = row["manual_training_action"]
        include = _to_bool_action(action, decision)
        weight = row["manual_sample_weight"]

        out.loc[mask, "review_decision_applied"] = True
        out.loc[mask, "review_unit_id_applied"] = unit_id
        out.loc[mask, "review_template_applied"] = row.get("review_template", "")
        out.loc[mask, "review_behavior_label"] = row.get("behavior_label", "")
        out.loc[mask, "review_manual_decision"] = decision
        out.loc[mask, "review_corrected_behavior"] = corrected
        out.loc[mask, "review_label_strength"] = row.get("manual_label_strength", "")
        out.loc[mask, "review_training_action"] = action
        out.loc[mask, "review_sample_weight"] = weight
        out.loc[mask, "review_include_in_training"] = include
        out.loc[mask, "review_note"] = row.get("manual_note", "")

        if decision == "corrected":
            out.loc[mask, "behavior_after_review"] = corrected
            out.loc[mask, "behavior"] = corrected

        touched_total += n
        applied_count += 1

    audit = {
        "decisions_loaded": int(len(decisions)),
        "pending_ignored": pending_ignored,
        "active_decisions": int(len(active)),
        "applied_decisions": int(applied_count),
        "decision_frame_rows_touched": int(touched_total),
        "affected_frames": int(touched_total),
        "changed_behavior_frames": int((out["behavior_after_review"] != out["behavior_before_review"]).sum()),
        "excluded_frames": int((~out["review_include_in_training"].astype(bool)).sum()),
        "accepted_units": int(active["manual_review_decision"].eq("accept").sum()),
        "corrected_units": int(active["manual_review_decision"].eq("corrected").sum()),
        "excluded_units": int(active["manual_review_decision"].isin(["exclude", "reject"]).sum()),
        "duplicate_active_decision_rows": duplicate_active_rows,
        "missing_review_unit_count": int(len(unmatched)),
        "unmatched_decisions": unmatched,
        "decision_counts": active["manual_review_decision"].value_counts(dropna=False).to_dict(),
        "training_action_counts": active["manual_training_action"].value_counts(dropna=False).to_dict(),
        "review_include_in_training_counts": out["review_include_in_training"].value_counts(dropna=False).to_dict(),
        "review_decision_applied_counts": out["review_decision_applied"].value_counts(dropna=False).to_dict(),
    }
    return out, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame-features-csv",
        default=r"outputs\classification_v2\frame_features\spatiotemporal_frame_features_enhanced.csv",
    )
    parser.add_argument(
        "--review-unit-manifest-csv",
        default=r"outputs\classification_v2\review_units\review_unit_manifest.csv",
    )
    parser.add_argument(
        "--decisions-csv",
        nargs="*",
        default=DEFAULT_DECISION_FILES,
        help="One or more behavior_unit_review_decisions.csv files. Defaults to the 4 GUI pilot outputs.",
    )
    parser.add_argument(
        "--output-csv",
        default=r"outputs\classification_v2\review_policy\reviewed_frame_features.csv",
    )
    parser.add_argument(
        "--audit-json",
        default=r"outputs\classification_v2\review_policy\apply_review_unit_decisions_audit.json",
    )
    parser.add_argument(
        "--combined-decisions-csv",
        default=r"outputs\classification_v2\review_policy\review_unit_decisions_combined.csv",
    )
    args = parser.parse_args()

    frame_path = Path(args.frame_features_csv)
    unit_path = Path(args.review_unit_manifest_csv)
    output_path = Path(args.output_csv)
    audit_path = Path(args.audit_json)
    combined_path = Path(args.combined_decisions_csv)
    decision_paths = [Path(p) for p in args.decisions_csv]

    frames = pd.read_csv(frame_path, low_memory=False)
    review_units = pd.read_csv(unit_path, low_memory=False)

    decisions, load_audit = load_decisions(decision_paths, review_units)
    decisions, norm_errors, norm_warnings = normalize_decisions(decisions)

    apply_errors: list[str] = []
    if norm_errors:
        apply_errors.extend(norm_errors)

    reviewed, apply_audit = apply_decisions_to_frames(frames, review_units, decisions)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.parent.mkdir(parents=True, exist_ok=True)

    reviewed.to_csv(output_path, index=False, encoding="utf-8-sig")
    decisions.to_csv(combined_path, index=False, encoding="utf-8-sig")

    audit = {
        "errors": apply_errors,
        "warnings": load_audit.get("load_warnings", []) + norm_warnings,
        "inputs": {
            "frame_features_csv": str(frame_path),
            "review_unit_manifest_csv": str(unit_path),
            "decisions_csv": [str(p) for p in decision_paths],
        },
        "outputs": {
            "output_csv": str(output_path),
            "combined_decisions_csv": str(combined_path),
            "audit_json": str(audit_path),
        },
        "rows": {
            "frame_features": int(len(frames)),
            "reviewed_frame_features": int(len(reviewed)),
            "decisions_loaded": int(len(decisions)),
        },
        "load_audit": load_audit,
        "apply_audit": apply_audit,
    }

    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(json.dumps(audit, indent=2, ensure_ascii=False, default=str))
    print("\n[OK] wrote", output_path, "rows=", len(reviewed), "cols=", len(reviewed.columns))
    print("[OK] wrote", combined_path, "rows=", len(decisions))
    print("[OK] wrote", audit_path)

    if apply_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
