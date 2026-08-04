"""Audit completed Behavior review and build a paired consistency re-review."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.review.behavior_consistency_audit import (
    add_temporal_encounter_partner_context,
    build_consistency_review_scope,
    build_effective_behavior_tables,
    build_interaction_correction_findings,
    build_interaction_pair_findings,
    build_note_findings,
    build_temporal_continuity_findings,
    combine_findings,
    decode_related_temporal_unit_keys,
)
from pig_behavior.classification_v2.review.behavior_review_contract import (
    audit_review_unit_contract,
)

FRAME_COLUMNS = [
    "source_type",
    "dataset_id",
    "video_key",
    "object_track_key",
    "frame_index",
    "pig_id",
    "track_id",
    "temporal_unit_key",
    "behavior_temporal_final",
    "nearest_partner_key",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed-decisions-csv", type=Path, required=True)
    parser.add_argument("--frame-features-csv", type=Path, required=True)
    parser.add_argument("--reference-review-view-csv", type=Path, required=True)
    parser.add_argument("--expected-decision-rows", type=int, required=True)
    parser.add_argument("--expected-decisions-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--future-review-output-dir", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--roi-coco-json", type=Path, required=True)
    parser.add_argument("--context-radius-frames", type=int, default=90)
    args = parser.parse_args()

    _require_new_output_dir(args.output_dir)
    decision_hash_before = _sha256_file(args.completed_decisions_csv)
    if decision_hash_before.casefold() != args.expected_decisions_sha256.casefold():
        raise ValueError(
            "completed decision hash mismatch: "
            f"expected={args.expected_decisions_sha256} actual={decision_hash_before}"
        )

    decisions = pd.read_csv(args.completed_decisions_csv, low_memory=False)
    if len(decisions) != args.expected_decision_rows:
        raise ValueError(
            "completed decision row mismatch: "
            f"expected={args.expected_decision_rows} actual={len(decisions)}"
        )
    frame_features = pd.read_csv(
        args.frame_features_csv,
        usecols=FRAME_COLUMNS,
        low_memory=False,
    )
    reference_view = pd.read_csv(args.reference_review_view_csv, low_memory=False)

    units, effective_frames, authority_audit = build_effective_behavior_tables(
        frame_features,
        decisions,
    )
    if authority_audit["matched_decision_units"] != args.expected_decision_rows:
        raise ValueError(
            "not every completed decision matched the frame authority: "
            f"{authority_audit}"
        )

    pair_findings, pair_stats = build_interaction_pair_findings(effective_frames)
    temporal_findings = build_temporal_continuity_findings(units)
    note_findings = build_note_findings(decisions)
    correction_findings = build_interaction_correction_findings(
        decisions,
        pair_stats,
    )

    reviewed_keys = set(decisions["temporal_unit_key"].astype(str))
    focused_pair = pair_findings.loc[
        pair_findings.apply(
            lambda row: _finding_touches_keys(row, reviewed_keys),
            axis=1,
        )
    ].copy()
    focused_temporal = temporal_findings.loc[
        temporal_findings["severity"].ne("LOW")
        & temporal_findings.apply(
            lambda row: _finding_touches_keys(row, reviewed_keys),
            axis=1,
        )
    ].copy()
    selected_findings = combine_findings(
        correction_findings,
        focused_pair,
        focused_temporal,
        note_findings,
    )
    all_findings = combine_findings(
        correction_findings,
        pair_findings,
        temporal_findings,
        note_findings,
    )
    selected_signatures = {
        _finding_signature(row) for _, row in selected_findings.iterrows()
    }
    all_findings["selected_for_rereview"] = [
        _finding_signature(row) in selected_signatures
        for _, row in all_findings.iterrows()
    ]
    selected_findings, temporal_partner_trace = (
        add_temporal_encounter_partner_context(
            selected_findings,
            effective_frames,
            units,
            context_radius_frames=args.context_radius_frames,
        )
    )

    policy = {
        "schema_version": "classification_v2.behavior_consistency_policy.v2",
        "completed_decision_rows": args.expected_decision_rows,
        "completed_decisions_sha256": decision_hash_before,
        "selection": [
            "all_interaction_label_corrections_with_temporal_partner_units",
            "all_pair_consistency_findings_touching_completed_review",
            "non_low_temporal_findings_touching_completed_review",
            "all_interaction_or_boundary_reviewer_notes",
        ],
        "partner_selection": (
            "bidirectional_nearest_history_then_synchronized_partner_unit"
        ),
        "max_temporal_partners": 3,
        "min_partner_support_frames": 2,
        "social_nose_semantics": "active_actor_only",
        "fight_semantics": "directly_involved_group",
        "general_low_islands_in_scope": False,
        "context_radius_frames": args.context_radius_frames,
    }
    selection_config_hash = _sha256_json(policy)
    scope = build_consistency_review_scope(
        units,
        reference_view,
        selected_findings,
        selection_config_hash=selection_config_hash,
        context_radius_frames=args.context_radius_frames,
    )
    contract_audit = audit_review_unit_contract(scope)
    if contract_audit["errors"]:
        raise ValueError(f"generated scope contract failed={contract_audit}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    scope_path = args.output_dir / "behavior_consistency_rereview_scope.csv"
    selected_path = args.output_dir / "selected_consistency_findings.csv"
    all_findings_path = args.output_dir / "all_consistency_findings.csv"
    pair_stats_path = args.output_dir / "interaction_pair_stats.csv"
    temporal_path = args.output_dir / "temporal_continuity_findings.csv"
    temporal_partner_path = args.output_dir / "temporal_partner_trace.csv"
    scope.to_csv(scope_path, index=False)
    selected_findings.to_csv(selected_path, index=False)
    all_findings.to_csv(all_findings_path, index=False)
    pair_stats.to_csv(pair_stats_path, index=False)
    temporal_findings.to_csv(temporal_path, index=False)
    temporal_partner_trace.to_csv(temporal_partner_path, index=False)

    review_command = _review_command(
        scope_path=scope_path,
        frame_features_csv=args.frame_features_csv,
        review_output_dir=args.future_review_output_dir,
        video_root=args.video_root,
        raw_root=args.raw_root,
        roi_coco_json=args.roi_coco_json,
    )
    command_path = args.output_dir / "exact_rereview_command.txt"
    command_path.write_text(review_command + "\n", encoding="utf-8")

    decision_hash_after = _sha256_file(args.completed_decisions_csv)
    if decision_hash_after != decision_hash_before:
        raise RuntimeError("completed decision ledger changed during read-only audit")

    manifest = {
        "schema_version": "classification_v2.behavior_consistency_rereview.v2",
        "status": "READY_FOR_TARGETED_REREVIEW",
        "code_sha": _git_head(),
        "policy": policy,
        "selection_config_hash": selection_config_hash,
        "inputs": {
            "completed_decisions": _path_record(args.completed_decisions_csv),
            "frame_features": _path_record(args.frame_features_csv),
            "reference_review_view": _path_record(args.reference_review_view_csv),
        },
        "authority_audit": authority_audit,
        "scope_contract_audit": contract_audit,
        "finding_counts": {
            "interaction_corrections": int(len(correction_findings)),
            "focused_pair": int(len(focused_pair)),
            "focused_temporal": int(len(focused_temporal)),
            "reviewer_notes": int(len(note_findings)),
            "selected_findings": int(len(selected_findings)),
            "temporal_partner_trace_rows": int(len(temporal_partner_trace)),
            "all_findings": int(len(all_findings)),
            "selected_severity": selected_findings["severity"]
            .value_counts()
            .to_dict(),
            "selected_reasons": selected_findings["finding_reason"]
            .value_counts()
            .to_dict(),
        },
        "scope_rows": int(len(scope)),
        "scope_source_counts": scope["source_type"].value_counts().to_dict(),
        "scope_behavior_counts": scope["behavior_label"].value_counts().to_dict(),
        "outputs": {
            "scope": _path_record(scope_path),
            "selected_findings": _path_record(selected_path),
            "all_findings": _path_record(all_findings_path),
            "pair_stats": _path_record(pair_stats_path),
            "temporal_findings": _path_record(temporal_path),
            "temporal_partner_trace": _path_record(temporal_partner_path),
            "exact_rereview_command": _path_record(command_path),
        },
        "completed_ledger_hash_before": decision_hash_before,
        "completed_ledger_hash_after": decision_hash_after,
        "completed_ledger_changed": False,
        "decisions_written": False,
        "source_annotations_changed": False,
        "training_started": False,
    }
    manifest_path = args.output_dir / "behavior_consistency_rereview_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "scope_rows": manifest["scope_rows"],
                "selected_findings": len(selected_findings),
                "manifest": str(manifest_path),
                "command": str(command_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _finding_touches_keys(row: pd.Series, keys: set[str]) -> bool:
    actor_key = str(row["temporal_unit_key"])
    if actor_key in keys:
        return True
    related = decode_related_temporal_unit_keys(row["related_temporal_unit_keys"])
    return bool(set(related).intersection(keys))


def _finding_signature(row: pd.Series) -> tuple[str, str, str]:
    return (
        str(row["temporal_unit_key"]),
        str(row["finding_reason"]),
        str(row["related_temporal_unit_keys"]),
    )


def _review_command(
    *,
    scope_path: Path,
    frame_features_csv: Path,
    review_output_dir: Path,
    video_root: Path,
    raw_root: Path,
    roi_coco_json: Path,
) -> str:
    python = Path(sys.executable).resolve()
    script = Path(
        "scripts/classification_v2/01_review_units_gui/"
        "review_final_behavior_gui_v1.py"
    ).resolve()
    parts = [
        f'cd /d "{Path.cwd()}"',
        f'"{python}" "{script}"',
        f'--review-units-csv "{scope_path.resolve()}"',
        f'--frame-features-csv "{frame_features_csv.resolve()}"',
        f'--output-dir "{review_output_dir.resolve()}"',
        f'--video-root "{video_root.resolve()}"',
        f'--raw-root "{raw_root.resolve()}"',
        f'--roi-coco-json "{roi_coco_json.resolve()}"',
    ]
    return " && ".join(parts[:2]) + " " + " ".join(parts[2:])


def _require_new_output_dir(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"output directory already exists={path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _path_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


if __name__ == "__main__":
    main()
