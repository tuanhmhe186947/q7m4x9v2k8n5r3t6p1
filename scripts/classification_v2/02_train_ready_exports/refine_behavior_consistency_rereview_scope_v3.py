"""Refine frozen v2 consistency review and reuse completed decisions safely."""

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
    select_targeted_consistency_scope_v3,
)
from pig_behavior.classification_v2.review.behavior_review_contract import (
    audit_review_unit_contract,
)

DECISION_FILENAME = "behavior_unit_review_decisions.csv"
STRENGTH_FILENAME = "behavior_strength_review_decisions.csv"
QUALITY_FILENAME = "behavior_label_quality_review.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed-decisions-csv", type=Path, required=True)
    parser.add_argument("--expected-completed-rows", type=int, required=True)
    parser.add_argument("--expected-completed-sha256", required=True)
    parser.add_argument("--prior-authority-dir", type=Path, required=True)
    parser.add_argument("--prior-review-output-dir", type=Path, required=True)
    parser.add_argument("--expected-prior-review-rows", type=int, required=True)
    parser.add_argument("--expected-prior-review-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--future-review-output-dir", type=Path, required=True)
    parser.add_argument("--frame-features-csv", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--roi-coco-json", type=Path, required=True)
    parser.add_argument("--fight-neighbor-radius-frames", type=int, default=24)
    parser.add_argument(
        "--min-overlapping-partner-support-frames",
        type=int,
        default=48,
    )
    parser.add_argument(
        "--required-temporal-unit-key",
        action="append",
        default=[],
    )
    args = parser.parse_args()

    _require_new_output_dir(args.output_dir)
    _require_new_output_dir(args.future_review_output_dir)

    completed_hash_before = _validate_csv_authority(
        args.completed_decisions_csv,
        expected_rows=args.expected_completed_rows,
        expected_sha256=args.expected_completed_sha256,
    )
    prior_decision_path = args.prior_review_output_dir / DECISION_FILENAME
    prior_review_hash_before = _validate_csv_authority(
        prior_decision_path,
        expected_rows=args.expected_prior_review_rows,
        expected_sha256=args.expected_prior_review_sha256,
    )

    prior_scope_path = (
        args.prior_authority_dir / "behavior_consistency_rereview_scope.csv"
    )
    prior_trace_path = args.prior_authority_dir / "temporal_partner_trace.csv"
    prior_manifest_path = (
        args.prior_authority_dir / "behavior_consistency_rereview_manifest.json"
    )
    prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
    frame_features_record = prior_manifest["inputs"]["frame_features"]
    recorded_frame_path = Path(frame_features_record["path"]).resolve()
    if str(recorded_frame_path).casefold() != str(
        args.frame_features_csv.resolve()
    ).casefold():
        raise ValueError(
            "frame features path differs from frozen v2 authority: "
            f"expected={recorded_frame_path} actual={args.frame_features_csv}"
        )
    completed_decisions = pd.read_csv(
        args.completed_decisions_csv,
        low_memory=False,
    )
    prior_scope = pd.read_csv(prior_scope_path, low_memory=False)
    prior_trace = pd.read_csv(prior_trace_path, low_memory=False)

    scope, selection_audit = select_targeted_consistency_scope_v3(
        prior_scope,
        completed_decisions,
        prior_trace,
        fight_neighbor_radius_frames=args.fight_neighbor_radius_frames,
        min_overlapping_partner_support_frames=(
            args.min_overlapping_partner_support_frames
        ),
    )
    required_keys = set(args.required_temporal_unit_key)
    selected_keys = set(scope["temporal_unit_key"].astype(str))
    missing_required = sorted(required_keys.difference(selected_keys))
    if missing_required:
        raise ValueError(f"required temporal units removed by v3={missing_required}")

    policy = {
        "schema_version": "classification_v2.behavior_consistency_policy.v3",
        "base_authority": str(args.prior_authority_dir.resolve()),
        "selection": (
            "actors plus evidence-backed non-fight temporal partners"
        ),
        "actor_policy": "retain every v2 ACTOR finding",
        "partner_policy": {
            "require_episode_partner_candidate": True,
            "require_current_behavior_not_fight": True,
            "require_bidirectional_nearest_history": True,
            "same_track_fight_neighbor_radius_frames": (
                args.fight_neighbor_radius_frames
            ),
            "alternative_target_overlap_min_support_frames": (
                args.min_overlapping_partner_support_frames
            ),
        },
        "proximity_only_context_retained": False,
        "prior_decisions_reused_by": "review_unit_id",
    }
    selection_config_hash = _sha256_json(policy)
    scope["selection_config_hash"] = selection_config_hash

    contract_audit = audit_review_unit_contract(scope)
    if contract_audit["errors"]:
        raise ValueError(f"generated v3 scope contract failed={contract_audit}")

    reuse = _prepare_reused_decisions(
        args.prior_review_output_dir,
        scope,
        expected_prior_hash=prior_review_hash_before,
    )
    if reuse["dropped_corrected_or_excluded_count"]:
        raise ValueError(
            "v3 would drop completed corrected/excluded decisions: "
            f"{reuse['dropped_corrected_or_excluded_ids']}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.future_review_output_dir.mkdir(parents=True, exist_ok=False)

    scope_path = args.output_dir / "behavior_consistency_rereview_scope.csv"
    audit_path = args.output_dir / "v3_scope_selection_audit.csv"
    trace_path = args.output_dir / "temporal_partner_trace.csv"
    scope.to_csv(scope_path, index=False)
    selection_audit.to_csv(audit_path, index=False)
    prior_trace.to_csv(trace_path, index=False)

    _write_reused_session(args.future_review_output_dir, reuse)
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

    completed_hash_after = _sha256_file(args.completed_decisions_csv)
    prior_review_hash_after = _sha256_file(prior_decision_path)
    if completed_hash_after != completed_hash_before:
        raise RuntimeError("completed primary decision ledger changed")
    if prior_review_hash_after != prior_review_hash_before:
        raise RuntimeError("prior v2 review decision ledger changed")

    kept_audit = selection_audit.loc[selection_audit["keep_in_v3"]]
    manifest = {
        "schema_version": "classification_v2.behavior_consistency_rereview.v3",
        "status": "READY_FOR_TARGETED_REREVIEW_WITH_REUSED_DECISIONS",
        "code_sha": _git_head(),
        "policy": policy,
        "selection_config_hash": selection_config_hash,
        "inputs": {
            "completed_decisions": _path_record(args.completed_decisions_csv),
            "prior_scope": _path_record(prior_scope_path),
            "prior_temporal_partner_trace": _path_record(prior_trace_path),
            "prior_manifest": _path_record(prior_manifest_path),
            "prior_review_decisions": _path_record(prior_decision_path),
            "frame_features": frame_features_record,
            "roi_coco_json": _path_record(args.roi_coco_json),
        },
        "counts": {
            "prior_scope_rows": int(len(prior_scope)),
            "v3_scope_rows": int(len(scope)),
            "removed_rows": int(len(prior_scope) - len(scope)),
            "reused_decision_rows": int(reuse["reused_decision_rows"]),
            "remaining_review_rows": int(
                len(scope) - reuse["reused_decision_rows"]
            ),
            "retained_corrected_rows": int(reuse["retained_corrected_rows"]),
        },
        "selection_reason_counts": {
            str(key): int(value)
            for key, value in selection_audit["v3_selection_reason"]
            .value_counts()
            .sort_index()
            .items()
        },
        "retained_behavior_counts": {
            str(key): int(value)
            for key, value in kept_audit["effective_behavior"]
            .value_counts()
            .sort_index()
            .items()
        },
        "scope_contract_audit": contract_audit,
        "decision_reuse_manifest": _path_record(
            args.future_review_output_dir / "reused_decisions_manifest.json"
        ),
        "protected_inputs_unchanged": {
            "completed_decisions": completed_hash_after == completed_hash_before,
            "prior_review_decisions": (
                prior_review_hash_after == prior_review_hash_before
            ),
        },
    }
    manifest_path = (
        args.output_dir / "behavior_consistency_rereview_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    inventory_paths = [
        scope_path,
        audit_path,
        trace_path,
        command_path,
        manifest_path,
        *sorted(args.future_review_output_dir.glob("*.csv")),
        *sorted(args.future_review_output_dir.glob("*.json")),
    ]
    inventory = {
        "schema_version": "classification_v2.behavior_consistency_v3_inventory.v1",
        "artifacts": [_path_record(path) for path in inventory_paths],
    }
    (args.output_dir / "artifact_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], sort_keys=True))


def _prepare_reused_decisions(
    prior_output_dir: Path,
    scope: pd.DataFrame,
    *,
    expected_prior_hash: str,
) -> dict[str, Any]:
    decisions_path = prior_output_dir / DECISION_FILENAME
    strength_path = prior_output_dir / STRENGTH_FILENAME
    quality_path = prior_output_dir / QUALITY_FILENAME
    decisions = pd.read_csv(decisions_path, low_memory=False)
    strength = pd.read_csv(strength_path, low_memory=False)
    quality = pd.read_csv(quality_path, low_memory=False)
    for name, frame in (
        ("decisions", decisions),
        ("strength", strength),
        ("quality", quality),
    ):
        if "review_unit_id" not in frame.columns:
            raise ValueError(f"{name} missing review_unit_id")
        if frame["review_unit_id"].astype(str).duplicated().any():
            raise ValueError(f"{name} contains duplicate review_unit_id")

    scope_order = {
        str(review_id): int(order)
        for review_id, order in zip(
            scope["review_unit_id"],
            scope["consistency_review_order"],
            strict=True,
        )
    }
    selected_ids = set(scope_order)
    reused_decisions = _select_in_scope_order(
        decisions,
        selected_ids,
        scope_order,
    )
    reused_strength = _select_in_scope_order(
        strength,
        selected_ids,
        scope_order,
    )
    reused_quality = _select_in_scope_order(
        quality,
        selected_ids,
        scope_order,
    )
    reused_ids = set(reused_decisions["review_unit_id"].astype(str))
    if reused_ids != set(reused_strength["review_unit_id"].astype(str)):
        raise ValueError("decision and strength reuse keys differ")
    if reused_ids != set(reused_quality["review_unit_id"].astype(str)):
        raise ValueError("decision and quality reuse keys differ")

    terminal = decisions["manual_review_decision"].astype(str).isin(
        {"corrected", "exclude"}
    )
    dropped_terminal = decisions.loc[
        terminal & ~decisions["review_unit_id"].astype(str).isin(selected_ids)
    ]
    retained_corrected = reused_decisions["manual_review_decision"].astype(
        str
    ).eq("corrected")
    scope_lookup = scope.set_index(scope["review_unit_id"].astype(str))
    lineage = reused_decisions[
        [
            "review_item_id",
            "review_unit_id",
            "temporal_unit_key",
            "manual_review_decision",
            "manual_corrected_behavior",
        ]
    ].copy()
    lineage.insert(
        0,
        "v3_consistency_review_order",
        lineage["review_unit_id"].astype(str).map(scope_order),
    )
    lineage["v3_consistency_roles"] = lineage["review_unit_id"].astype(
        str
    ).map(scope_lookup["consistency_roles"])
    lineage["source_v2_decisions_sha256"] = expected_prior_hash
    lineage["reuse_status"] = "REUSED_EXACT_HUMAN_DECISION"
    return {
        "decisions": reused_decisions,
        "strength": reused_strength,
        "quality": reused_quality,
        "lineage": lineage,
        "reused_decision_rows": int(len(reused_decisions)),
        "retained_corrected_rows": int(retained_corrected.sum()),
        "dropped_corrected_or_excluded_count": int(len(dropped_terminal)),
        "dropped_corrected_or_excluded_ids": dropped_terminal[
            "review_unit_id"
        ].astype(str).tolist(),
        "source_paths": {
            "decisions": decisions_path,
            "strength": strength_path,
            "quality": quality_path,
        },
    }


def _select_in_scope_order(
    frame: pd.DataFrame,
    selected_ids: set[str],
    scope_order: dict[str, int],
) -> pd.DataFrame:
    selected = frame.loc[
        frame["review_unit_id"].astype(str).isin(selected_ids)
    ].copy()
    selected["_v3_order"] = selected["review_unit_id"].astype(str).map(
        scope_order
    )
    selected = selected.sort_values("_v3_order", kind="mergesort")
    return selected.drop(columns="_v3_order").reset_index(drop=True)


def _write_reused_session(output_dir: Path, reuse: dict[str, Any]) -> None:
    decisions_path = output_dir / DECISION_FILENAME
    strength_path = output_dir / STRENGTH_FILENAME
    quality_path = output_dir / QUALITY_FILENAME
    lineage_path = output_dir / "reused_decision_lineage.csv"
    reuse["decisions"].to_csv(decisions_path, index=False)
    reuse["strength"].to_csv(strength_path, index=False)
    reuse["quality"].to_csv(quality_path, index=False)
    reuse["lineage"].to_csv(lineage_path, index=False)
    manifest = {
        "schema_version": "classification_v2.reused_review_decisions.v1",
        "status": "REUSED_EXACT_HUMAN_DECISIONS",
        "source_artifacts": {
            name: _path_record(path)
            for name, path in reuse["source_paths"].items()
        },
        "destination_artifacts": {
            "decisions": _path_record(decisions_path),
            "strength": _path_record(strength_path),
            "quality": _path_record(quality_path),
            "lineage": _path_record(lineage_path),
        },
        "counts": {
            "reused_decision_rows": reuse["reused_decision_rows"],
            "retained_corrected_rows": reuse["retained_corrected_rows"],
            "dropped_corrected_or_excluded_count": reuse[
                "dropped_corrected_or_excluded_count"
            ],
        },
    }
    (output_dir / "reused_decisions_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
    gui_script = (
        Path.cwd()
        / "scripts"
        / "classification_v2"
        / "01_review_units_gui"
        / "review_final_behavior_gui_v1.py"
    )
    return " ".join(
        [
            f'cd /d "{Path.cwd()}"',
            "&&",
            f'"{Path(sys.executable).resolve()}"',
            f'"{gui_script}"',
            f'--review-units-csv "{scope_path.resolve()}"',
            f'--frame-features-csv "{frame_features_csv.resolve()}"',
            f'--output-dir "{review_output_dir.resolve()}"',
            f'--video-root "{video_root.resolve()}"',
            f'--raw-root "{raw_root.resolve()}"',
            f'--roi-coco-json "{roi_coco_json.resolve()}"',
        ]
    )


def _validate_csv_authority(
    path: Path,
    *,
    expected_rows: int,
    expected_sha256: str,
) -> str:
    actual_hash = _sha256_file(path)
    if actual_hash.casefold() != expected_sha256.casefold():
        raise ValueError(
            f"authority hash mismatch path={path} "
            f"expected={expected_sha256} actual={actual_hash}"
        )
    actual_rows = len(pd.read_csv(path, low_memory=False))
    if actual_rows != expected_rows:
        raise ValueError(
            f"authority row mismatch path={path} "
            f"expected={expected_rows} actual={actual_rows}"
        )
    return actual_hash


def _require_new_output_dir(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"output directory already exists: {path}")


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
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path.cwd(),
        text=True,
    ).strip()


if __name__ == "__main__":
    main()
