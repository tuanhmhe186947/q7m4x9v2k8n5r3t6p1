"""Freeze and analyze completed blinded interaction-development decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.review.source_specific_blinded_presentation_v2 import (
    PRESENTATION_SEMANTIC_HASH,
    PRESENTATION_VERSION,
    derive_calibration_outcome,
    validate_calibration_decisions_v2,
)

DEVELOPMENT_SUBSET = "CALIBRATION_DEVELOPMENT_SET"
EXPECTED_DEVELOPMENT_COUNT = 300
MINIMUM_REVIEW_NEEDED_EVENTS = 30
RECALL_POINT_MIN = 0.95
RECALL_LCB_MIN = 0.85
MISSED_ERROR_UCB_MAX = 0.05
NPV_LCB_MIN = 0.95
ONE_SIDED_CONFIDENCE = 0.95
RULE_COLUMNS = {
    "current_991_screen": "current_interaction_candidate",
    "static_95_diagnostic_screen": "static_set_95_diagnostic",
}
TRUE_VALUES = frozenset({"1", "true", "yes", "y"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions-csv", type=Path, required=True)
    parser.add_argument("--internal-trace-csv", type=Path, required=True)
    parser.add_argument("--media-authority-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def truth(value: object) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().casefold() in TRUE_VALUES


def git_state(repo_root: Path) -> dict[str, Any]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"code_sha": sha, "code_dirty": bool(status)}


def one_sided_wilson_bounds(
    successes: int,
    total: int,
    *,
    confidence: float = ONE_SIDED_CONFIDENCE,
) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    z_value = NormalDist().inv_cdf(confidence)
    proportion = successes / total
    denominator = 1.0 + (z_value**2 / total)
    center = (proportion + (z_value**2 / (2.0 * total))) / denominator
    radius = (
        z_value
        * math.sqrt(proportion * (1.0 - proportion) / total + z_value**2 / (4.0 * total**2))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def prepare_joined_outcomes(
    decisions: pd.DataFrame,
    internal_trace: pd.DataFrame,
    media_authority: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    decision_audit = validate_calibration_decisions_v2(decisions)
    errors = list(decision_audit["errors"])

    development_trace = internal_trace.loc[
        internal_trace["frozen_subset"].eq(DEVELOPMENT_SUBSET)
    ].copy()
    development_media = media_authority.loc[media_authority["split"].eq(DEVELOPMENT_SUBSET)].copy()

    if len(decisions) != EXPECTED_DEVELOPMENT_COUNT:
        errors.append(f"decision_count={len(decisions)}")
    if len(development_trace) != EXPECTED_DEVELOPMENT_COUNT:
        errors.append(f"development_trace_count={len(development_trace)}")
    if len(development_media) != EXPECTED_DEVELOPMENT_COUNT:
        errors.append(f"development_media_count={len(development_media)}")

    decision_ids = set(decisions["calibration_item_id"].astype(str))
    trace_ids = set(development_trace["calibration_item_id"].astype(str))
    media_ids = set(development_media["calibration_item_id"].astype(str))
    if decision_ids != trace_ids:
        errors.append("decision_trace_item_set_mismatch")
    if decision_ids != media_ids:
        errors.append("decision_media_item_set_mismatch")

    confirmation_ids = set(
        internal_trace.loc[
            internal_trace["frozen_subset"].eq("BLINDED_CONFIRMATION_SET"),
            "calibration_item_id",
        ].astype(str)
    )
    confirmation_overlap = sorted(decision_ids.intersection(confirmation_ids))
    if confirmation_overlap:
        errors.append(f"confirmation_decision_overlap_count={len(confirmation_overlap)}")

    merged = decisions.merge(
        development_media[
            [
                "calibration_item_id",
                "review_key",
                "split",
                "source_type",
                "dataset_id",
                "video_key",
                "recording_date",
                "presentation_version",
                "presentation_semantic_hash",
            ]
        ],
        on=["calibration_item_id", "review_key"],
        how="left",
        validate="one_to_one",
        suffixes=("_decision", "_media"),
    )
    merged = merged.merge(
        development_trace,
        left_on=["calibration_item_id", "review_key"],
        right_on=["calibration_item_id", "review_unit_id"],
        how="left",
        validate="one_to_one",
        suffixes=("_media", "_trace"),
    )
    if merged["behavior_label"].isna().any():
        errors.append("missing_provisional_behavior_after_join")
    if not merged["split"].eq(DEVELOPMENT_SUBSET).all():
        errors.append("nondevelopment_media_joined")

    decision_versions = merged["presentation_version_decision"].astype(str)
    decision_hashes = merged["presentation_semantic_hash_decision"].astype(str)
    media_versions = merged["presentation_version_media"].astype(str)
    media_hashes = merged["presentation_semantic_hash_media"].astype(str)
    if not decision_versions.eq(PRESENTATION_VERSION).all():
        errors.append("decision_presentation_version_mismatch")
    if not decision_hashes.eq(PRESENTATION_SEMANTIC_HASH).all():
        errors.append("decision_presentation_hash_mismatch")
    if not media_versions.eq(PRESENTATION_VERSION).all():
        errors.append("media_presentation_version_mismatch")
    if not media_hashes.eq(PRESENTATION_SEMANTIC_HASH).all():
        errors.append("media_presentation_hash_mismatch")

    if errors:
        raise ValueError(";".join(errors))

    merged["calibration_outcome"] = merged.apply(
        lambda row: derive_calibration_outcome(
            provisional_behavior=str(row["behavior_label"]),
            reviewed_behavior=str(row["reviewed_behavior"]),
            visual_reviewability=str(row["visual_reviewability"]),
        ),
        axis=1,
    )
    merged["combined_review_needed"] = ~merged["calibration_outcome"].eq("LABEL_SUPPORTED")
    for column in RULE_COLUMNS.values():
        merged[column] = merged[column].map(truth)

    audit = {
        "valid": True,
        "errors": [],
        "decision_count": int(len(decisions)),
        "unique_review_key_count": int(decisions["review_key"].astype(str).nunique()),
        "duplicate_review_key_count": int(decisions["review_key"].astype(str).duplicated().sum()),
        "confirmation_decision_overlap_count": 0,
        "presentation_version": PRESENTATION_VERSION,
        "presentation_semantic_hash": PRESENTATION_SEMANTIC_HASH,
    }
    return merged, audit


def evaluate_screen(
    outcomes: pd.DataFrame,
    *,
    rule_id: str,
    screen_column: str,
) -> dict[str, Any]:
    positive = outcomes["combined_review_needed"].astype(bool)
    screen = outcomes[screen_column].astype(bool)
    true_positive = int((screen & positive).sum())
    false_negative = int((~screen & positive).sum())
    true_negative = int((~screen & ~positive).sum())
    false_positive = int((screen & ~positive).sum())
    positive_count = true_positive + false_negative
    auto_carry_count = true_negative + false_negative

    recall = true_positive / positive_count if positive_count else None
    recall_lcb, _ = one_sided_wilson_bounds(
        true_positive,
        positive_count,
    )
    missed_error_rate = false_negative / auto_carry_count if auto_carry_count else None
    _, missed_error_ucb = one_sided_wilson_bounds(
        false_negative,
        auto_carry_count,
    )
    npv = true_negative / auto_carry_count if auto_carry_count else None
    npv_lcb, _ = one_sided_wilson_bounds(
        true_negative,
        auto_carry_count,
    )

    event_gate = positive_count >= MINIMUM_REVIEW_NEEDED_EVENTS
    recall_point_gate = recall is not None and recall >= RECALL_POINT_MIN
    recall_lcb_gate = recall_lcb is not None and recall_lcb >= RECALL_LCB_MIN
    missed_error_gate = missed_error_ucb is not None and missed_error_ucb <= MISSED_ERROR_UCB_MAX
    npv_gate = npv_lcb is not None and npv_lcb >= NPV_LCB_MIN
    all_gates = all(
        [
            event_gate,
            recall_point_gate,
            recall_lcb_gate,
            missed_error_gate,
            npv_gate,
        ]
    )
    return {
        "rule_id": rule_id,
        "screen_column": screen_column,
        "sample_scope": "DEVELOPMENT_QUOTA_SAMPLE_UNWEIGHTED",
        "screen_positive_count": int(screen.sum()),
        "auto_carry_count": auto_carry_count,
        "review_needed_count": positive_count,
        "true_positive": true_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "recall_point": recall,
        "recall_one_sided_95_lcb": recall_lcb,
        "auto_carry_missed_error_rate": missed_error_rate,
        "auto_carry_missed_error_one_sided_95_ucb": missed_error_ucb,
        "auto_carry_npv": npv,
        "auto_carry_npv_one_sided_95_lcb": npv_lcb,
        "minimum_event_gate": event_gate,
        "recall_point_gate": recall_point_gate,
        "recall_lcb_gate": recall_lcb_gate,
        "missed_error_ucb_gate": missed_error_gate,
        "npv_lcb_gate": npv_gate,
        "all_development_gates_pass": all_gates,
    }


def choose_post_calibration_decision(
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    by_rule = {record["rule_id"]: record for record in evaluations}
    current = by_rule["current_991_screen"]
    static = by_rule["static_95_diagnostic_screen"]

    if current["all_development_gates_pass"]:
        decision = "DECISION_A_KEEP_CURRENT_991_WITH_REVALIDATION"
        selected_rule = current["rule_id"]
        confirmation_authorized = True
        reason = "current frozen screen passes every development safety gate"
    elif static["all_development_gates_pass"]:
        decision = "DECISION_C_NEW_CALIBRATED_SELECTIVE_PARTITION"
        selected_rule = static["rule_id"]
        confirmation_authorized = True
        reason = "static diagnostic screen passes every development safety gate"
    elif max(record["review_needed_count"] for record in evaluations) < (
        MINIMUM_REVIEW_NEEDED_EVENTS
    ):
        decision = "DECISION_E_REMAIN_INCONCLUSIVE"
        selected_rule = None
        confirmation_authorized = False
        reason = "insufficient development review-needed event count"
    else:
        decision = "DECISION_B_FULL_INTERACTION_CENSUS"
        selected_rule = None
        confirmation_authorized = False
        reason = (
            "neither predeclared selective screen supports safe auto-carry "
            "under the frozen development gates"
        )

    return {
        "post_calibration_decision": decision,
        "selected_rule_id": selected_rule,
        "reason": reason,
        "confirmation_authorized": confirmation_authorized,
        "confirmation_opened": False,
        "confirmation_decisions_accessed": False,
        "candidate_membership_changed": False,
        "auto_carry_membership_changed": False,
        "training_labels_applied": False,
        "full_interaction_census_review_started": False,
    }


def grouped_outcome_table(
    outcomes: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    grouped = (
        outcomes.groupby(group_columns + ["calibration_outcome"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = (
        outcomes.groupby(group_columns, dropna=False).size().rename("group_total").reset_index()
    )
    grouped = grouped.merge(totals, on=group_columns, validate="many_to_one")
    grouped["rate_within_group"] = grouped["count"] / grouped["group_total"]
    return grouped.sort_values(group_columns + ["calibration_outcome"]).reset_index(drop=True)


def build_final_report(
    *,
    authority: dict[str, Any],
    summary: dict[str, Any],
    decision: dict[str, Any],
    evaluations: list[dict[str, Any]],
) -> str:
    lines = [
        "# Interaction development calibration application report",
        "",
        f"- CODE_SHA: `{authority['code_sha']}`",
        f"- LEDGER_SHA256: `{authority['input_hashes']['decisions_csv']}`",
        f"- DEVELOPMENT_DECISIONS: {summary['decision_count']}",
        f"- REVIEW_NEEDED_OUTCOMES: {summary['review_needed_count']}",
        f"- LABEL_SUPPORTED: {summary['outcome_counts'].get('LABEL_SUPPORTED', 0)}",
        (f"- CORRECTION_REQUIRED: {summary['outcome_counts'].get('CORRECTION_REQUIRED', 0)}"),
        (f"- VISUALLY_UNRESOLVED: {summary['outcome_counts'].get('VISUALLY_UNRESOLVED', 0)}"),
        (
            "- TECHNICAL_AUTHORITY_DEFECT: "
            f"{summary['outcome_counts'].get('TECHNICAL_AUTHORITY_DEFECT', 0)}"
        ),
        "",
        "## Frozen screen evaluation",
        "",
    ]
    for evaluation in evaluations:
        lines.extend(
            [
                f"### {evaluation['rule_id']}",
                "",
                f"- recall: {evaluation['recall_point']}",
                (f"- recall_one_sided_95_lcb: {evaluation['recall_one_sided_95_lcb']}"),
                (
                    "- auto_carry_missed_error_one_sided_95_ucb: "
                    f"{evaluation['auto_carry_missed_error_one_sided_95_ucb']}"
                ),
                (
                    "- auto_carry_npv_one_sided_95_lcb: "
                    f"{evaluation['auto_carry_npv_one_sided_95_lcb']}"
                ),
                (f"- all_development_gates_pass: {evaluation['all_development_gates_pass']}"),
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            f"- POST_CALIBRATION_DECISION: `{decision['post_calibration_decision']}`",
            f"- REASON: {decision['reason']}",
            (
                "- BLINDED_CONFIRMATION_AUTHORIZED: "
                f"{'YES' if decision['confirmation_authorized'] else 'NO'}"
            ),
            "- CONFIRMATION_OPENED: NO",
            "- TRAINING_LABELS_APPLIED: NO",
            "",
            "The 300 decisions are development calibration evidence. They do not",
            "become final training authority and do not open confirmation unless",
            "the frozen development decision explicitly authorizes it.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    for path in (
        args.decisions_csv,
        args.internal_trace_csv,
        args.media_authority_csv,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    repo_root = Path(__file__).resolve().parents[3]
    authority = {
        **git_state(repo_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_scope": "CALIBRATION_DEVELOPMENT_SET_ONLY",
        "label_authority": "HUMAN_BLINDED_DEVELOPMENT_CALIBRATION",
        "presentation_version": PRESENTATION_VERSION,
        "presentation_semantic_hash": PRESENTATION_SEMANTIC_HASH,
        "input_paths": {
            "decisions_csv": str(args.decisions_csv.resolve()),
            "internal_trace_csv": str(args.internal_trace_csv.resolve()),
            "media_authority_csv": str(args.media_authority_csv.resolve()),
        },
        "input_hashes": {
            "decisions_csv": sha256_file(args.decisions_csv),
            "internal_trace_csv": sha256_file(args.internal_trace_csv),
            "media_authority_csv": sha256_file(args.media_authority_csv),
        },
        "confirmation_decisions_accessed": False,
        "gui_opened": False,
    }
    write_json(args.output_dir / "authority_resolution.json", authority)

    decisions = pd.read_csv(args.decisions_csv, low_memory=False)
    internal_trace = pd.read_csv(args.internal_trace_csv, low_memory=False)
    media_authority = pd.read_csv(args.media_authority_csv, low_memory=False)
    outcomes, coverage_audit = prepare_joined_outcomes(
        decisions,
        internal_trace,
        media_authority,
    )
    write_json(args.output_dir / "decision_coverage_audit.json", coverage_audit)

    frozen_ledger = args.output_dir / "frozen_development_decisions.csv"
    shutil.copyfile(args.decisions_csv, frozen_ledger)
    if sha256_file(frozen_ledger) != authority["input_hashes"]["decisions_csv"]:
        raise RuntimeError("frozen decision copy hash mismatch")

    outcome_counts = {
        str(key): int(value)
        for key, value in outcomes["calibration_outcome"].value_counts().sort_index().items()
    }
    review_needed_count = int(outcomes["combined_review_needed"].sum())
    summary = {
        "decision_count": int(len(outcomes)),
        "outcome_counts": outcome_counts,
        "review_needed_count": review_needed_count,
        "review_needed_rate": review_needed_count / len(outcomes),
        "reviewer_count": int(outcomes["reviewer"].astype(str).nunique()),
        "source_count": int(outcomes["source_type_trace"].astype(str).nunique()),
        "video_count": int(outcomes["video_key_trace"].astype(str).nunique()),
        "provisional_behavior_counts": {
            str(key): int(value)
            for key, value in outcomes["behavior_label"].value_counts().sort_index().items()
        },
        "reviewed_behavior_counts": {
            str(key): int(value)
            for key, value in outcomes["reviewed_behavior"].value_counts().sort_index().items()
        },
    }
    write_json(args.output_dir / "calibration_outcome_summary.json", summary)

    safe_outcome_columns = [
        "calibration_item_id",
        "review_key",
        "source_type_trace",
        "dataset_id_trace",
        "video_key_trace",
        "recording_date_trace",
        "behavior_label",
        "reviewed_behavior",
        "visual_reviewability",
        "review_confidence",
        "calibration_outcome",
        "combined_review_needed",
        "current_interaction_candidate",
        "static_set_95_diagnostic",
        "removed_by_static_diagnostic",
        "high_crowding",
        "lower_crowding",
        "contact_proxy_present",
        "contact_proxy_absent",
        "social_evidence_available",
        "social_evidence_unavailable_or_low_quality",
        "authority_risk_control",
    ]
    write_csv(
        args.output_dir / "calibration_outcomes.csv",
        outcomes[safe_outcome_columns],
    )
    write_csv(
        args.output_dir / "outcomes_by_source.csv",
        grouped_outcome_table(outcomes, ["source_type_trace"]),
    )
    write_csv(
        args.output_dir / "outcomes_by_video.csv",
        grouped_outcome_table(outcomes, ["video_key_trace"]),
    )
    write_csv(
        args.output_dir / "outcomes_by_provisional_behavior.csv",
        grouped_outcome_table(outcomes, ["behavior_label"]),
    )
    write_csv(
        args.output_dir / "outcomes_by_reviewed_behavior.csv",
        grouped_outcome_table(outcomes, ["reviewed_behavior"]),
    )

    evaluations = [
        evaluate_screen(
            outcomes,
            rule_id=rule_id,
            screen_column=screen_column,
        )
        for rule_id, screen_column in RULE_COLUMNS.items()
    ]
    write_csv(
        args.output_dir / "development_screening_evaluation.csv",
        pd.DataFrame(evaluations),
    )
    threshold_selection = {
        "semantic_status": "DEVELOPMENT_CALIBRATION_ANALYSIS",
        "metric_id": "combined_review_needed_probability",
        "combined_review_needed_definition": ("calibration_outcome != LABEL_SUPPORTED"),
        "one_sided_confidence": ONE_SIDED_CONFIDENCE,
        "acceptance_criteria": {
            "minimum_review_needed_events": MINIMUM_REVIEW_NEEDED_EVENTS,
            "recall_point_min": RECALL_POINT_MIN,
            "recall_one_sided_95_lcb_min": RECALL_LCB_MIN,
            "auto_carry_missed_error_one_sided_95_ucb_max": (MISSED_ERROR_UCB_MAX),
            "auto_carry_npv_one_sided_95_lcb_min": NPV_LCB_MIN,
        },
        "evaluations": evaluations,
    }
    threshold_selection["semantic_hash"] = stable_json_sha256(threshold_selection)
    write_json(
        args.output_dir / "development_threshold_selection.json",
        threshold_selection,
    )

    decision = choose_post_calibration_decision(evaluations)
    decision["ledger_sha256"] = authority["input_hashes"]["decisions_csv"]
    decision["threshold_selection_semantic_hash"] = threshold_selection["semantic_hash"]
    write_json(args.output_dir / "post_calibration_decision.json", decision)
    write_json(
        args.output_dir / "confirmation_gate.json",
        {
            "authorized": bool(decision["confirmation_authorized"]),
            "selected_rule_id": decision["selected_rule_id"],
            "development_ledger_sha256": authority["input_hashes"]["decisions_csv"],
            "confirmation_decisions_accessed": False,
            "confirmation_opened": False,
            "confirmation_use_for_threshold_design": "FORBIDDEN",
        },
    )

    exact_command = " ".join(
        [
            "python",
            str(Path(__file__).resolve()),
            "--decisions-csv",
            f'"{args.decisions_csv.resolve()}"',
            "--internal-trace-csv",
            f'"{args.internal_trace_csv.resolve()}"',
            "--media-authority-csv",
            f'"{args.media_authority_csv.resolve()}"',
            "--output-dir",
            f'"{args.output_dir.resolve()}"',
        ]
    )
    atomic_write_text(
        args.output_dir / "exact_commands.txt",
        exact_command + "\n",
    )
    atomic_write_text(
        args.output_dir / "final_interaction_calibration_report.md",
        build_final_report(
            authority=authority,
            summary=summary,
            decision=decision,
            evaluations=evaluations,
        ),
    )

    artifacts = {}
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_inventory.json":
            artifacts[path.name] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    write_json(
        args.output_dir / "artifact_inventory.json",
        {
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
        },
    )
    print(
        "INTERACTION_DEVELOPMENT_CALIBRATION_APPLIED "
        f"decision={decision['post_calibration_decision']} "
        f"confirmation_authorized={decision['confirmation_authorized']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
