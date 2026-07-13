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

from pig_behavior.classification_v2.review.behavior_review_contract import (
    audit_decision_coverage,
    audit_manifest_alignment,
    audit_review_unit_contract,
    canonicalize_decisions,
    normalize_text,
    validate_decision_semantics,
)

DEFAULT_DECISION_FILES = [
    (
        r"outputs\classification_v2\review_policy\roi_review_unit_gui_pilot"
        r"\behavior_unit_review_decisions.csv"
    ),
    (
        r"outputs\classification_v2\review_policy\motion_review_unit_gui_pilot"
        r"\behavior_unit_review_decisions.csv"
    ),
    (
        r"outputs\classification_v2\review_policy\posture_review_unit_gui_pilot"
        r"\behavior_unit_review_decisions.csv"
    ),
    (
        r"outputs\classification_v2\review_policy\interaction_review_unit_gui_pilot"
        r"\behavior_unit_review_decisions.csv"
    ),
]


def _norm_text(value: Any) -> str:
    """Compatibility wrapper for callers of the former local helper."""
    return normalize_text(value)


def _to_bool_action(action: str, decision: str) -> bool:
    action = _norm_text(action)
    decision = _norm_text(decision)
    if decision in {"exclude", "reject"}:
        return False
    if action in {"exclude", "review_later"}:
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
    if decision in {"exclude", "reject"} or action in {"exclude", "review_later"}:
        return 0.0
    if decision == "uncertain" or action in {"downweight", "low_weight_train"}:
        return 0.5
    return 1.0


def _validate_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def load_decisions(
    paths: list[Path],
    review_unit_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parts: list[pd.DataFrame] = []
    missing_files: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []

    unit_contract = audit_review_unit_contract(review_unit_manifest)
    errors.extend(unit_contract["errors"])
    warnings.extend(unit_contract["warnings"])
    if errors:
        return pd.DataFrame(), {
            "missing_files": missing_files,
            "load_errors": errors,
            "load_warnings": warnings,
            "review_unit_contract": unit_contract,
        }

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
                warnings.append(
                    f"{path} missing behavior_label; "
                    "using original_behavior compatibility"
                )
                df["behavior_label"] = df["original_behavior"]
            else:
                warnings.append(f"{path} missing behavior_label/original_behavior; filling empty")
                df["behavior_label"] = ""

        if "original_behavior" not in df.columns:
            df["original_behavior"] = df["behavior_label"]

        alignment_errors, alignment_warnings = audit_manifest_alignment(
            review_unit_manifest,
            df,
            allow_blank_snapshot=True,
        )
        errors.extend(f"{path}:{error}" for error in alignment_errors)
        warnings.extend(f"{path}:{warning}" for warning in alignment_warnings)

        # Fill missing metadata from canonical review unit manifest.
        df = df.merge(
            manifest_small,
            on="review_unit_id",
            how="left",
            suffixes=("", "_manifest"),
            validate="many_to_one",
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

        helper_columns = [
            column
            for column in df.columns
            if column.endswith("_manifest")
        ]
        if helper_columns:
            df = df.drop(columns=helper_columns)

        original = df["original_behavior"].map(_norm_text)
        df["original_behavior"] = original.where(original.ne(""), df["behavior_label"])

        final_alignment_errors, _ = audit_manifest_alignment(
            review_unit_manifest,
            df,
            allow_blank_snapshot=False,
        )
        errors.extend(f"{path}:{error}" for error in final_alignment_errors)

        parts.append(df)

    if parts:
        decisions = pd.concat(parts, ignore_index=True)
    else:
        decisions = pd.DataFrame()

    audit = {
        "missing_files": missing_files,
        "load_errors": sorted(set(errors)),
        "load_warnings": sorted(set(warnings)),
        "review_unit_contract": unit_contract,
    }
    return decisions, audit


def normalize_decisions(decisions: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    if decisions.empty:
        return decisions, [], []

    normalized, warnings = canonicalize_decisions(decisions)
    errors, semantic_warnings = validate_decision_semantics(
        normalized,
        require_complete=False,
    )
    warnings.extend(semantic_warnings)
    normalized["_decision_order"] = range(len(normalized))
    return normalized, errors, warnings


def apply_decisions_to_frames(
    frames: pd.DataFrame,
    review_units: pd.DataFrame,
    decisions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _validate_columns(
        frames,
        ["temporal_unit_key", "source_type", "frame_index", "behavior"],
        "frame_features_csv",
    )
    unit_contract = audit_review_unit_contract(review_units)
    if unit_contract["errors"]:
        raise ValueError(
            "invalid review_unit_manifest_csv: "
            + "; ".join(unit_contract["errors"])
        )

    if not decisions.empty:
        semantic_errors, _ = validate_decision_semantics(
            decisions,
            require_complete=False,
        )
        alignment_errors, _ = audit_manifest_alignment(
            review_units,
            decisions,
            allow_blank_snapshot=True,
        )
        decision_errors = semantic_errors + alignment_errors
        if decision_errors:
            raise ValueError("invalid behavior decisions: " + "; ".join(decision_errors))

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
    duplicate_active_rows = (
        int(active["review_unit_id"].duplicated(keep=False).sum())
        if len(active)
        else 0
    )
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

    unit_map = review_units[
        [
            "review_unit_id",
            "temporal_unit_key",
            "source_type",
            "unit_start_frame",
            "unit_end_frame",
        ]
    ].rename(
        columns={
            "temporal_unit_key": "target_temporal_unit_key",
            "source_type": "target_source_type",
            "unit_start_frame": "target_start_frame",
            "unit_end_frame": "target_end_frame",
        }
    )
    active = active.merge(
        unit_map,
        on="review_unit_id",
        how="left",
        validate="one_to_one",
    )
    active["target_temporal_unit_key"] = active["target_temporal_unit_key"].map(
        _norm_text
    )

    unmatched: list[str] = []
    touched_total = 0
    applied_count = 0
    applied_unit_ids: list[str] = []
    frame_index = pd.to_numeric(out["frame_index"], errors="coerce")

    for _, row in active.iterrows():
        unit_id = row["review_unit_id"]
        target_key = row["target_temporal_unit_key"]
        if not target_key:
            unmatched.append(f"{unit_id}:missing_temporal_unit_key")
            continue

        start = int(row["target_start_frame"])
        end = int(row["target_end_frame"])
        key_mask = out["temporal_unit_key"].astype(str).eq(str(target_key))
        mask = (
            key_mask
            & out["source_type"].astype(str).eq(str(row["target_source_type"]))
            & frame_index.between(start, end)
        )
        n = int(mask.sum())
        if n == 0:
            unmatched.append(f"{unit_id}:no_matching_frames")
            continue
        observed_frames = frame_index[mask].dropna().astype(int).tolist()
        expected_frames = list(range(start, end + 1))
        if int(key_mask.sum()) != n:
            unmatched.append(f"{unit_id}:temporal_key_has_rows_outside_unit_scope")
            continue
        if len(observed_frames) != len(expected_frames):
            unmatched.append(
                f"{unit_id}:row_count={len(observed_frames)}:expected={len(expected_frames)}"
            )
            continue
        if sorted(observed_frames) != expected_frames:
            unmatched.append(f"{unit_id}:frame_scope_not_exact_or_has_duplicates")
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
        applied_unit_ids.append(str(unit_id))

    applied = active[active["review_unit_id"].astype(str).isin(applied_unit_ids)].copy()

    audit = {
        "decisions_loaded": int(len(decisions)),
        "pending_ignored": pending_ignored,
        "active_decisions": int(len(active)),
        "applied_decisions": int(applied_count),
        "decision_frame_rows_touched": int(touched_total),
        "affected_frames": int(touched_total),
        "changed_behavior_frames": int(
            (out["behavior_after_review"] != out["behavior_before_review"]).sum()
        ),
        "excluded_frames": int((~out["review_include_in_training"].astype(bool)).sum()),
        "accepted_units": int(applied["manual_review_decision"].eq("accept").sum()),
        "corrected_units": int(applied["manual_review_decision"].eq("corrected").sum()),
        "excluded_units": int(applied["manual_review_decision"].eq("exclude").sum()),
        "duplicate_active_decision_rows": duplicate_active_rows,
        "missing_review_unit_count": int(len(unmatched)),
        "unmatched_decisions": unmatched,
        "decision_counts": applied["manual_review_decision"].value_counts(dropna=False).to_dict(),
        "training_action_counts": applied["manual_training_action"]
        .value_counts(dropna=False)
        .to_dict(),
        "review_include_in_training_counts": out["review_include_in_training"]
        .value_counts(dropna=False)
        .to_dict(),
        "review_decision_applied_counts": out["review_decision_applied"]
        .value_counts(dropna=False)
        .to_dict(),
    }
    return out, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame-features-csv",
        default=(
            r"outputs\classification_v2\frame_features"
            r"\spatiotemporal_frame_features_enhanced.csv"
        ),
    )
    parser.add_argument(
        "--review-unit-manifest-csv",
        default=(
            r"outputs\classification_v2\review_units"
            r"\full_review_unit_manifest.csv"
        ),
    )
    parser.add_argument(
        "--decisions-csv",
        nargs="*",
        default=DEFAULT_DECISION_FILES,
        help=(
            "One or more behavior_unit_review_decisions.csv files. "
            "Defaults to the 4 GUI pilot outputs."
        ),
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
    coverage_audit = audit_decision_coverage(
        review_units,
        decisions,
        require_complete=True,
    )

    apply_errors = (
        list(load_audit.get("load_errors", []))
        + norm_errors
        + list(coverage_audit.get("errors", []))
    )
    missing_files = load_audit.get("missing_files", [])
    if missing_files:
        apply_errors.append(f"missing_decision_files={missing_files}")
    if decisions.empty:
        apply_errors.append("no_behavior_review_decisions_loaded")

    reviewed: pd.DataFrame | None = None
    apply_audit: dict[str, Any] = {"skipped": True}
    if not apply_errors:
        try:
            reviewed, apply_audit = apply_decisions_to_frames(
                frames,
                review_units,
                decisions,
            )
        except ValueError as exc:
            apply_errors.append(str(exc))
        else:
            unmatched = apply_audit.get("unmatched_decisions", [])
            if unmatched:
                apply_errors.append(f"unmatched_decisions={unmatched}")
            if len(reviewed) != len(frames):
                apply_errors.append(
                    f"reviewed_row_count_mismatch={len(frames)}:{len(reviewed)}"
                )

    audit_path.parent.mkdir(parents=True, exist_ok=True)

    audit = {
        "errors": sorted(set(apply_errors)),
        "warnings": sorted(
            set(
                load_audit.get("load_warnings", [])
                + norm_warnings
                + list(coverage_audit.get("warnings", []))
            )
        ),
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
            "reviewed_frame_features": int(len(reviewed)) if reviewed is not None else None,
            "decisions_loaded": int(len(decisions)),
        },
        "load_audit": load_audit,
        "decision_coverage_audit": coverage_audit,
        "apply_audit": apply_audit,
    }

    if apply_errors:
        audit_path.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(json.dumps(audit, indent=2, ensure_ascii=False, default=str))
        raise SystemExit(2)

    assert reviewed is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    reviewed.to_csv(output_path, index=False, encoding="utf-8-sig")
    decisions.to_csv(combined_path, index=False, encoding="utf-8-sig")
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False, default=str))
    print("\n[OK] wrote", output_path, "rows=", len(reviewed), "cols=", len(reviewed.columns))
    print("[OK] wrote", combined_path, "rows=", len(decisions))
    print("[OK] wrote", audit_path)


if __name__ == "__main__":
    main()
