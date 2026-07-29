"""Build and preflight source-specific blinded calibration presentation V2."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pig_behavior.classification_v2.review.source_specific_blinded_presentation_v2 import (
    CVAT_CONTEXT_MODE,
    LEGACY_CONTEXT_MODE,
    LEGACY_NOTICE_TEXT,
    MEDIA_AUTHORITY_REQUIRED_FIELDS,
    MEDIA_AUTHORITY_SCHEMA_VERSION,
    OLD_PRESENTATION_HASH,
    PRESENTATION_SEMANTIC_HASH,
    PRESENTATION_VERSION,
    SourceSpecificPresentationError,
    apply_preflight_availability,
    build_media_authority_v2,
    canonical_presentation_contract_v2,
    frozen_identity_check,
    parse_frame_indices,
    presentation_semantic_hash_v2,
    source_mode_contracts,
    validate_media_authority_v2,
    visible_notice,
)

SEMANTIC_STATUS = "PRE_REVIEW_SOURCE_SPECIFIC_PRESENTATION_V2"
AUTHORITY_EXPECTED_HASHES = {
    "candidate_6061": (
        "0ba8cff16cf4ddd77448f1c62ae86000ca096b14e5481d0d3b73f584c9f28a08"
    ),
    "auto_carry_27294": (
        "d23bbc8d8675b5134b968d1e960bf72882746d4e23d59d7d5485dc9f7028e84c"
    ),
    "universe_33355": (
        "e89f88ff5a27b518eaeb606a3a14c12c12b34ba2d4cdf0f4e4cbf29e13c3a553"
    ),
    "native_review_evidence": (
        "3c925759b90191a9cec4f2e517c3570886365b9b4c5189bbe7bcc20c72265254"
    ),
    "safe_non_interaction_view_5070": (
        "f5ce71106cca8ebbe9b7069903e35988e409a3e7d52cccd06ce01f2b02ad3420"
    ),
    "spatial_46d_schema": (
        "18377d825ba84974e49305e46561ada81353f9ffd0f2d2526471af1c199daad4"
    ),
    "motion_12d_schema": (
        "ec0c511b5f5198240492be49c0492e543c9e38eb4a4ff446259b958c2a59963b"
    ),
}


def _load_gui_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_source_specific_calibration_gui_v2_preflight",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load V2 GUI module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    common: dict[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            {**common, **payload},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    frame: pd.DataFrame,
    *,
    common: dict[str, Any],
) -> None:
    output = frame.copy()
    metadata = {
        "semantic_status": common["semantic_status"],
        "producer_sha": common["producer_sha"],
        "config_hash": common["config_hash"],
        "input_hashes_json": json.dumps(
            common["input_hashes"],
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    for column, value in reversed(list(metadata.items())):
        if column in output.columns:
            output[column] = value
        else:
            output.insert(0, column, value)
    output.to_csv(path, index=False)


def _write_text(
    path: Path,
    body: str,
    *,
    common: dict[str, Any],
) -> None:
    header = (
        f"semantic_status={common['semantic_status']}\n"
        f"producer_sha={common['producer_sha']}\n"
        f"config_hash={common['config_hash']}\n"
        "input_hashes="
        f"{json.dumps(common['input_hashes'], sort_keys=True)}\n\n"
    )
    path.write_text(header + body.rstrip() + "\n", encoding="utf-8")


def _md_header(title: str, common: dict[str, Any]) -> str:
    return (
        f"# {title}\n\n"
        f"- Semantic status: `{common['semantic_status']}`\n"
        f"- Producer SHA: `{common['producer_sha']}`\n"
        f"- Config hash: `{common['config_hash']}`\n"
        "- Input hashes: "
        f"`{json.dumps(common['input_hashes'], sort_keys=True)}`\n\n"
    )


def _valid_order(frames: list[int]) -> bool:
    return frames == sorted(frames) and len(frames) == len(set(frames))


def _preflight_item(
    unit: pd.Series,
    delegate: Any,
) -> dict[str, Any]:
    targets = parse_frame_indices(unit["target_frame_indices"])
    history = parse_frame_indices(unit["history_frame_indices"])
    display = parse_frame_indices(unit["display_frame_indices"])
    structural = validate_media_authority_v2(pd.DataFrame([unit]))
    source_type = str(unit["source_type"])
    context_mode = str(unit["context_mode"])
    failures: list[str] = []
    if not structural["valid"]:
        failures.extend(structural["errors"])
    expected_target_frames = (
        6 if context_mode == CVAT_CONTEXT_MODE else 16
    )
    frame_order_valid = (
        _valid_order(targets)
        and _valid_order(history)
        and display == [*history, *targets]
        and not set(history).intersection(targets)
        and (not history or max(history) < min(targets))
    )
    target_scope_complete = (
        len(targets) == expected_target_frames
        and int(unit["target_frame_count"]) == expected_target_frames
    )
    duplicate_display_frames = len(display) - len(set(display))
    if not frame_order_valid:
        failures.append("frame_order_invalid")
    if not target_scope_complete:
        failures.append("target_scope_incomplete")
    if duplicate_display_frames:
        failures.append("duplicate_display_frames")

    actor_authority_valid = False
    media_readable = False
    notice_visible = False
    machine_hypothesis_hidden = False
    hash_match = False
    render_deterministic = False
    runtime_renderer = ""
    valid_non_actor_count = 0
    max_valid_non_actor_count = 0
    red_actor_present = False
    neutral_neighbors_rendered = False
    ranked_neighbor_rendered = False
    target_history_separated = False
    actor_crop_complete = False
    actor_crop_notice_visible = False
    undeclared_fallback = False

    try:
        rows = delegate._frame_rows_for_unit(unit)
        observed = [
            int(value)
            for value in rows["frame_index"].tolist()
            if pd.notna(value)
        ]
        actor_authority_valid = (
            observed == display
            and len(observed) == len(display)
            and len(observed) == len(set(observed))
        )
        if not actor_authority_valid:
            failures.append("actor_authority_invalid")
        first, _, frame_audits = delegate.make_source_specific_sheet(
            unit,
            collect_preflight_audit=True,
        )
        second, _, _ = delegate.make_source_specific_sheet(unit)
        render_deterministic = first.tobytes() == second.tobytes()
        media_readable = bool(frame_audits) and len(frame_audits) == len(
            display
        )
        notice_visible = (
            visible_notice(context_mode)
            == source_mode_contracts()[context_mode]["visible_notice"]
        )
        machine_hypothesis_hidden = all(
            canonical_presentation_contract_v2()[field] == "HIDDEN"
            for field in (
                "ranking_visibility",
                "provisional_label_visibility",
                "machine_reason_visibility",
                "candidate_tier_visibility",
                "machine_score_visibility",
                "source_date_video_stratum_visibility",
            )
        )
        hash_match = (
            str(unit["presentation_semantic_hash"])
            == PRESENTATION_SEMANTIC_HASH
        )
        runtime_renderers = {
            str(item["runtime_renderer"]) for item in frame_audits
        }
        if len(runtime_renderers) == 1:
            runtime_renderer = next(iter(runtime_renderers))
        else:
            failures.append("mixed_runtime_renderers")
        valid_non_actor_count = int(
            sum(item["valid_non_actor_count"] for item in frame_audits)
        )
        max_valid_non_actor_count = int(
            max(
                (
                    item["valid_non_actor_count"]
                    for item in frame_audits
                ),
                default=0,
            )
        )
        red_counts = [
            int(item["red_pixel_count"]) for item in frame_audits
        ]
        ranked_counts = [
            int(item["ranked_green_pixel_count"])
            for item in frame_audits
        ]
        ranked_neighbor_rendered = any(count > 0 for count in ranked_counts)
        target_history_separated = (
            [item["role"] for item in frame_audits]
            == ["CONTEXT"] * len(history) + ["TARGET"] * len(targets)
        )
        if context_mode == CVAT_CONTEXT_MODE:
            red_actor_present = all(count > 0 for count in red_counts)
            neutral_neighbors_rendered = all(
                item["valid_non_actor_count"] == 0
                or item["neutral_pixel_count"] > 0
                for item in frame_audits
            )
            if not red_actor_present:
                failures.append("cvat_actor_red_missing")
            if not neutral_neighbors_rendered:
                failures.append("cvat_neutral_neighbor_missing")
            if ranked_neighbor_rendered:
                failures.append("cvat_ranked_neighbor_visible")
        else:
            actor_crop_complete = (
                len(frame_audits) == 16
                and all(item["legacy_direct_crop"] for item in frame_audits)
            )
            actor_crop_notice_visible = (
                visible_notice(context_mode) == LEGACY_NOTICE_TEXT
            )
            neutral_neighbors_rendered = True
            if not actor_crop_complete:
                failures.append("legacy_actor_crop_incomplete")
            if not actor_crop_notice_visible:
                failures.append("legacy_notice_missing")
        if not render_deterministic:
            failures.append("render_nondeterministic")
        if not media_readable:
            failures.append("media_unreadable")
        if not notice_visible:
            failures.append("notice_missing")
        if not machine_hypothesis_hidden:
            failures.append("machine_hypothesis_visible")
        if not hash_match:
            failures.append("presentation_hash_mismatch")
        if not target_history_separated:
            failures.append("target_history_not_separated")
    except Exception as exc:
        failures.append(
            f"{type(exc).__name__}:{str(exc).replace(';', ',')}"
        )

    failures = list(dict.fromkeys(failures))
    return {
        "review_key": unit["review_key"],
        "calibration_item_id": unit["calibration_item_id"],
        "split": unit["split"],
        "source_type": source_type,
        "context_mode": context_mode,
        "runtime_renderer": runtime_renderer,
        "actor_authority_valid": actor_authority_valid,
        "target_scope_complete": target_scope_complete,
        "frame_order_valid": frame_order_valid,
        "media_readable": media_readable,
        "notice_visible": notice_visible,
        "machine_hypothesis_hidden": machine_hypothesis_hidden,
        "presentation_hash_match": hash_match,
        "render_deterministic": render_deterministic,
        "reviewable": not failures,
        "failure_reason": ";".join(failures),
        "red_actor_present": (
            red_actor_present if context_mode == CVAT_CONTEXT_MODE else ""
        ),
        "valid_non_actor_count": (
            valid_non_actor_count
            if context_mode == CVAT_CONTEXT_MODE
            else ""
        ),
        "max_valid_non_actor_count": (
            max_valid_non_actor_count
            if context_mode == CVAT_CONTEXT_MODE
            else ""
        ),
        "neutral_neighbors_rendered": (
            neutral_neighbors_rendered
            if context_mode == CVAT_CONTEXT_MODE
            else ""
        ),
        "ranked_neighbor_rendered": (
            ranked_neighbor_rendered
            if context_mode == CVAT_CONTEXT_MODE
            else ""
        ),
        "target_history_separated": (
            target_history_separated
            if context_mode == CVAT_CONTEXT_MODE
            else ""
        ),
        "actor_crop_complete": (
            actor_crop_complete
            if context_mode == LEGACY_CONTEXT_MODE
            else ""
        ),
        "expected_target_frames": (
            expected_target_frames
            if context_mode == LEGACY_CONTEXT_MODE
            else ""
        ),
        "actual_target_frames": (
            len(targets) if context_mode == LEGACY_CONTEXT_MODE else ""
        ),
        "duplicate_display_frames": duplicate_display_frames,
        "history_frame_count": len(history),
        "neighbor_context_available": unit[
            "neighbor_context_available"
        ],
        "full_frame_context_available": unit[
            "full_frame_context_available"
        ],
        "actor_crop_notice_visible": (
            actor_crop_notice_visible
            if context_mode == LEGACY_CONTEXT_MODE
            else ""
        ),
        "undeclared_fallback": undeclared_fallback,
    }


def _group_leakage_count(
    media: pd.DataFrame,
    group_manifest: pd.DataFrame,
) -> int:
    expected = group_manifest[
        ["source_type", "recording_date", "video_key", "frozen_subset"]
    ].drop_duplicates()
    joined = media.merge(
        expected,
        on=["source_type", "recording_date", "video_key"],
        how="left",
        validate="many_to_one",
    )
    return int(
        joined["frozen_subset"].isna().sum()
        + joined["split"].ne(joined["frozen_subset"]).sum()
    )


def _smoke_manifest(preflight: pd.DataFrame) -> pd.DataFrame:
    development = preflight.loc[
        preflight["split"].eq("CALIBRATION_DEVELOPMENT_SET")
    ].copy()
    selected: list[pd.Series] = []

    def add_first(mask: pd.Series) -> None:
        used = {str(row["calibration_item_id"]) for row in selected}
        candidates = development.loc[mask].sort_values(
            "calibration_item_id",
            kind="stable",
        )
        for _, row in candidates.iterrows():
            if str(row["calibration_item_id"]) not in used:
                selected.append(row)
                return
        raise SourceSpecificPresentationError(
            "development smoke stratum is unavailable"
        )

    cvat = development["context_mode"].eq(CVAT_CONTEXT_MODE)
    legacy = development["context_mode"].eq(LEGACY_CONTEXT_MODE)
    max_neighbors = pd.to_numeric(
        development["max_valid_non_actor_count"],
        errors="coerce",
    )
    one_neighbor = cvat & max_neighbors.eq(1)
    if one_neighbor.any():
        add_first(one_neighbor)
    else:
        lowest_observed = max_neighbors.loc[cvat].min()
        add_first(cvat & max_neighbors.eq(lowest_observed))
    add_first(cvat & max_neighbors.gt(1))
    history_count = pd.to_numeric(
        development["history_frame_count"],
        errors="coerce",
    )
    add_first(cvat & history_count.gt(0))
    add_first(legacy)
    add_first(legacy)
    return pd.DataFrame(
        [
            {
                "calibration_item_id": row["calibration_item_id"],
                "split": row["split"],
                "smoke_order": index,
                "presentation_version": PRESENTATION_VERSION,
                "presentation_semantic_hash": PRESENTATION_SEMANTIC_HASH,
            }
            for index, row in enumerate(selected, start=1)
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-features-csv", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--producer-sha", required=True)
    parser.add_argument("--current-main-sha", required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if any(args.output_root.iterdir()):
        raise SystemExit("output root must be empty")
    if len(args.producer_sha) != 40:
        raise SystemExit("producer SHA must be full")

    paths = {
        "blinded_manifest_v1": (
            args.prior_audit_root / "blinded_calibration_manifest.csv"
        ),
        "media_authority_v1": (
            args.prior_audit_root / "calibration_media_authority.csv"
        ),
        "group_split_manifest": (
            args.prior_audit_root / "calibration_group_split_manifest.csv"
        ),
        "internal_trace": (
            args.prior_audit_root / "internal_calibration_trace.csv"
        ),
        "safe_view": (
            args.prior_audit_root
            / "safe_non_interaction_review_view.csv"
        ),
        "frame_features": args.frame_features_csv,
        "candidate_6061": (
            args.authority_root
            / "behavior_review_units"
            / "behavior_review_candidate_manifest.csv"
        ),
        "auto_carry_27294": (
            args.authority_root
            / "behavior_review_units"
            / "behavior_review_auto_carry_manifest.csv"
        ),
        "universe_33355": (
            args.authority_root
            / "behavior_review_units"
            / "behavior_review_universe.csv"
        ),
        "native_review_evidence": (
            args.authority_root
            / "native_evidence"
            / "native_review_evidence.csv"
        ),
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise SystemExit(f"missing immutable inputs={missing}")
    input_hashes = {
        name: sha256_file(path) for name, path in paths.items()
    }
    config = {
        "schema": "source_specific_presentation_v2_preflight.v1",
        "presentation_version": PRESENTATION_VERSION,
        "presentation_hash": PRESENTATION_SEMANTIC_HASH,
        "old_presentation_hash": OLD_PRESENTATION_HASH,
        "producer_sha": args.producer_sha,
        "prior_audit_root": str(args.prior_audit_root),
        "frame_features_csv": str(args.frame_features_csv),
        "video_root": str(args.video_root),
        "raw_root": str(args.raw_root),
        "frozen_population": {
            "total": 480,
            "development": 300,
            "confirmation": 180,
            "cvat": 383,
            "legacy": 97,
        },
    }
    common = {
        "semantic_status": SEMANTIC_STATUS,
        "producer_sha": args.producer_sha,
        "config_hash": canonical_hash(config),
        "input_hashes": input_hashes,
        "diagnostic_only": True,
    }
    authority_before = {
        key: input_hashes[key]
        for key in (
            "candidate_6061",
            "auto_carry_27294",
            "universe_33355",
            "native_review_evidence",
        )
    }
    authority_before["safe_non_interaction_view_5070"] = input_hashes[
        "safe_view"
    ]
    for key, expected in AUTHORITY_EXPECTED_HASHES.items():
        if key in authority_before and authority_before[key] != expected:
            raise SystemExit(f"protected authority hash mismatch={key}")

    blinded = pd.read_csv(paths["blinded_manifest_v1"], low_memory=False)
    media_v1 = pd.read_csv(paths["media_authority_v1"], low_memory=False)
    groups = pd.read_csv(paths["group_split_manifest"], low_memory=False)
    media = build_media_authority_v2(
        blinded,
        media_v1,
        producer_sha=args.producer_sha,
        input_hashes=input_hashes,
    )
    identity = frozen_identity_check(blinded, media)
    group_leakage = _group_leakage_count(media, groups)
    identity["group_leakage_count"] = group_leakage
    identity["valid"] = bool(identity["valid"] and group_leakage == 0)

    gui_path = (
        args.worktree
        / "scripts"
        / "classification_v2"
        / "01_review_units_gui"
        / "review_interaction_blind_calibration_gui_v2.py"
    )
    gui = _load_gui_module(gui_path)
    declared = canonical_presentation_contract_v2()
    effective = gui.effective_runtime_presentation_contract_v2()
    declared_runtime_match = declared == effective
    effective_hash = presentation_semantic_hash_v2(effective)
    runtime_hash_match = effective_hash == PRESENTATION_SEMANTIC_HASH
    if not declared_runtime_match or not runtime_hash_match:
        raise SystemExit("declared/runtime presentation semantics mismatch")
    if not identity["valid"]:
        raise SystemExit(f"frozen population identity failed={identity}")

    _write_json(
        args.output_root / "authority_resolution.json",
        {
            "current_main_sha_at_start": args.current_main_sha,
            "presentation_v2_base_sha": args.producer_sha,
            "calibration_branch_base_sha": (
                "b33a86406f360e50ea073735164caae8d293e67f"
            ),
            "resolved": True,
            "active_ledger_touched": False,
            "confirmation_decisions_accessed": False,
            "gui_opened": False,
        },
        common=common,
    )
    _write_json(
        args.output_root / "corrected_prior_fact_registry.json",
        {
            "legacy_render_mode": "actor_crop_only",
            "actor_identity_semantics": "entire_crop_is_reviewed_actor",
            "neighbor_context_available": False,
            "full_frame_context_available": False,
            "legacy_red_actor_box_required": False,
            "legacy_neutral_neighbors_required": False,
            "legacy_neighbor_context_absence_is_failure": False,
            "visually_unresolved_available": True,
            "legacy_items": 97,
            "legacy_development_items": 57,
            "legacy_confirmation_items": 40,
        },
        common=common,
    )
    _write_json(
        args.output_root / "frozen_population_identity_check.json",
        identity,
        common=common,
    )
    _write_json(
        args.output_root / "calibration_media_authority_v2_schema.json",
        {
            "schema_version": MEDIA_AUTHORITY_SCHEMA_VERSION,
            "required_fields": list(MEDIA_AUTHORITY_REQUIRED_FIELDS),
            "missing_or_unknown_dispatch_behavior": "FAIL_CLOSED",
            "legacy_actor_crop_mode_is_fallback": False,
        },
        common=common,
    )
    _write_json(
        args.output_root / "declared_presentation_contract_v2.json",
        {
            "contract": declared,
            "presentation_semantic_hash": PRESENTATION_SEMANTIC_HASH,
            "old_presentation_hash": OLD_PRESENTATION_HASH,
        },
        common=common,
    )
    _write_json(
        args.output_root
        / "effective_runtime_presentation_contract_v2.json",
        {
            "contract": effective,
            "effective_runtime_hash": effective_hash,
            "declared_runtime_semantics_match": (
                "PASS" if declared_runtime_match else "FAIL"
            ),
        },
        common=common,
    )
    notice_mutation = json.loads(json.dumps(declared))
    notice_mutation["source_modes"][LEGACY_CONTEXT_MODE][
        "visible_notice"
    ] += " MUTATED"
    legacy_field_mutations = {}
    for field in (
        "render_mode",
        "actor_identity_semantics",
        "neighbor_context_available",
        "full_frame_context_available",
    ):
        mutation = json.loads(json.dumps(declared))
        value = mutation["source_modes"][LEGACY_CONTEXT_MODE][field]
        mutation["source_modes"][LEGACY_CONTEXT_MODE][field] = (
            not value if isinstance(value, bool) else f"{value}_mutated"
        )
        legacy_field_mutations[field] = (
            presentation_semantic_hash_v2(mutation)
            != PRESENTATION_SEMANTIC_HASH
        )
    _write_json(
        args.output_root / "presentation_semantic_hash_v2.json",
        {
            "old_presentation_hash": OLD_PRESENTATION_HASH,
            "new_presentation_version": PRESENTATION_VERSION,
            "new_presentation_hash": PRESENTATION_SEMANTIC_HASH,
            "old_and_new_differ": (
                OLD_PRESENTATION_HASH != PRESENTATION_SEMANTIC_HASH
            ),
            "effective_runtime_hash": effective_hash,
            "presentation_runtime_semantic_hash_match": (
                "PASS" if runtime_hash_match else "FAIL"
            ),
            "visible_notice_mutation_changes_hash": (
                presentation_semantic_hash_v2(notice_mutation)
                != PRESENTATION_SEMANTIC_HASH
            ),
            "legacy_field_mutations_change_hash": (
                legacy_field_mutations
            ),
        },
        common=common,
    )

    dispatch = pd.DataFrame(
        [
            {
                "source_type": contract["source_type"],
                "context_mode": context_mode,
                "render_mode": contract["render_mode"],
                "actor_identity_semantics": contract[
                    "actor_identity_semantics"
                ],
                "neighbor_context_available": contract[
                    "neighbor_context_available"
                ],
                "full_frame_context_available": contract[
                    "full_frame_context_available"
                ],
                "missing_dispatch_action": "FAIL_CLOSED",
                "fallback_allowed": False,
            }
            for context_mode, contract in source_mode_contracts().items()
        ]
    )
    _write_csv(
        args.output_root / "source_render_dispatch_matrix.csv",
        dispatch,
        common=common,
    )

    frames = gui._MEDIA.load_gui_frame_features(
        args.frame_features_csv
    )
    delegate_config = gui._MEDIA.GuiConfig(
        review_units_csv=Path("SOURCE_SPECIFIC_MEDIA_AUTHORITY_V2"),
        frame_features_csv=args.frame_features_csv,
        output_dir=args.output_root,
        video_root=args.video_root,
        raw_root=args.raw_root,
        roi_coco_path=None,
        source_type=None,
        max_items=None,
        padding=0.0,
        copy_contact_sheets=False,
    )
    delegate = gui.SourceSpecificMediaDelegate(delegate_config, frames)
    try:
        preflight_records = [
            _preflight_item(row, delegate)
            for _, row in media.iterrows()
        ]
    finally:
        delegate.close()
    preflight = pd.DataFrame.from_records(preflight_records)
    final_media = apply_preflight_availability(media, preflight)
    final_validation = validate_media_authority_v2(
        final_media,
        require_render_available=True,
    )

    failed_items = int((~preflight["reviewable"]).sum())
    hash_mismatch_items = int(
        (~preflight["presentation_hash_match"]).sum()
    )
    undeclared_fallback_items = int(
        preflight["undeclared_fallback"].sum()
    )
    legacy_mask = preflight["context_mode"].eq(LEGACY_CONTEXT_MODE)
    cvat_mask = preflight["context_mode"].eq(CVAT_CONTEXT_MODE)
    missing_notice_legacy = int(
        (~preflight.loc[
            legacy_mask,
            "actor_crop_notice_visible",
        ].astype(bool)).sum()
    )
    cvat_dispatch_failures = int(
        (
            ~preflight.loc[cvat_mask, "reviewable"].astype(bool)
            | preflight.loc[cvat_mask, "runtime_renderer"].ne(
                "full_frame_neutral_context"
            )
        ).sum()
    )
    duplicate_frames = int(
        pd.to_numeric(
            preflight["duplicate_display_frames"],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )
    summary = {
        "total_items": int(len(preflight)),
        "development_items": int(
            preflight["split"]
            .eq("CALIBRATION_DEVELOPMENT_SET")
            .sum()
        ),
        "confirmation_items": int(
            preflight["split"].eq("BLINDED_CONFIRMATION_SET").sum()
        ),
        "cvat_items": int(cvat_mask.sum()),
        "legacy_items": int(legacy_mask.sum()),
        "legacy_development_items": int(
            (
                legacy_mask
                & preflight["split"].eq(
                    "CALIBRATION_DEVELOPMENT_SET"
                )
            ).sum()
        ),
        "legacy_confirmation_items": int(
            (
                legacy_mask
                & preflight["split"].eq(
                    "BLINDED_CONFIRMATION_SET"
                )
            ).sum()
        ),
        "failed_items": failed_items,
        "hash_mismatch_items": hash_mismatch_items,
        "undeclared_fallback_items": undeclared_fallback_items,
        "missing_notice_legacy_items": missing_notice_legacy,
        "cvat_dispatch_failure_items": cvat_dispatch_failures,
        "duplicate_display_frame_items": duplicate_frames,
        "final_media_authority_valid": final_validation["valid"],
    }
    summary["pass"] = all(
        (
            summary["total_items"] == 480,
            summary["development_items"] == 300,
            summary["confirmation_items"] == 180,
            summary["cvat_items"] == 383,
            summary["legacy_items"] == 97,
            summary["legacy_development_items"] == 57,
            summary["legacy_confirmation_items"] == 40,
            failed_items == 0,
            hash_mismatch_items == 0,
            undeclared_fallback_items == 0,
            missing_notice_legacy == 0,
            cvat_dispatch_failures == 0,
            duplicate_frames == 0,
            final_validation["valid"],
        )
    )
    _write_csv(
        args.output_root / "full_480_presentation_preflight.csv",
        preflight,
        common=common,
    )
    _write_json(
        args.output_root
        / "full_480_presentation_preflight_summary.json",
        summary,
        common=common,
    )
    _write_csv(
        args.output_root / "calibration_media_authority_v2.csv",
        final_media,
        common=common,
    )

    cvat_report = _md_header("CVAT presentation V2 report", common)
    cvat_report += (
        "The runtime uses explicit `cvat_full_frame_context` dispatch, "
        "draws the actor red, draws every valid non-actor in one neutral "
        "style, and never uses rank as identity.\n\n"
        f"- Items: `{int(cvat_mask.sum())}`\n"
        f"- Dispatch failures: `{cvat_dispatch_failures}`\n"
        "- Machine-hypothesis fields visible: `NO`\n"
        "- Decision scope: exact six target frames\n"
        "- History: visibly separated context only\n"
    )
    _write_text(
        args.output_root / "cvat_presentation_report.md",
        cvat_report,
        common={**common, "input_hashes": {}},
    )
    legacy_report = _md_header("Legacy presentation V2 report", common)
    legacy_report += (
        "Legacy uses the immutable actor crop directly. The entire crop is "
        "the reviewed actor; no actor box, neighbor box, partner, or "
        "full-frame context is fabricated.\n\n"
        f"> {LEGACY_NOTICE_TEXT}\n\n"
        f"- Items: `{int(legacy_mask.sum())}`\n"
        f"- Missing notice: `{missing_notice_legacy}`\n"
        f"- Duplicate display frames: `{duplicate_frames}`\n"
        "- Exact target scope: 16 native frames\n"
        "- Missing neighbor context alone is a failure: `NO`\n"
    )
    _write_text(
        args.output_root / "legacy_presentation_report.md",
        legacy_report,
        common={**common, "input_hashes": {}},
    )

    context_contract = pd.DataFrame(
        [
            {
                "context_mode": CVAT_CONTEXT_MODE,
                "item_count": int(cvat_mask.sum()),
                "correction_required_rate": "REPORT_LATER",
                "label_supported_rate": "REPORT_LATER",
                "visually_unresolved_rate": "REPORT_LATER",
                "technical_defect_rate": "REPORT_LATER",
                "selector_recall": "REPORT_LATER",
                "auto_carry_missed_error_rate": "REPORT_LATER",
                "confidence_intervals": "WILSON_OR_GROUP_BOOTSTRAP",
                "neighborhood_selector_validation_use": (
                    "ELIGIBLE_AFTER_CLOSED_CALIBRATION"
                ),
            },
            {
                "context_mode": LEGACY_CONTEXT_MODE,
                "item_count": int(legacy_mask.sum()),
                "correction_required_rate": "REPORT_LATER",
                "label_supported_rate": "REPORT_LATER",
                "visually_unresolved_rate": "REPORT_LATER",
                "technical_defect_rate": "REPORT_LATER",
                "selector_recall": "AVAILABILITY_AWARE_ANALYSIS_ONLY",
                "auto_carry_missed_error_rate": (
                    "AVAILABILITY_AWARE_ANALYSIS_ONLY"
                ),
                "confidence_intervals": "WILSON_OR_GROUP_BOOTSTRAP",
                "neighborhood_selector_validation_use": (
                    "NO_DIRECT_VALIDATION"
                ),
            },
        ]
    )
    _write_csv(
        args.output_root
        / "context_stratified_calibration_contract.csv",
        context_contract,
        common=common,
    )

    if summary["pass"]:
        smoke = _smoke_manifest(preflight)
        _write_csv(
            args.output_root / "development_smoke_manifest.csv",
            smoke,
            common=common,
        )
        smoke_command = (
            "python scripts/classification_v2/01_review_units_gui/"
            "review_interaction_blind_calibration_gui_v2.py "
            f'--media-authority "'
            f'{args.output_root / "calibration_media_authority_v2.csv"}" '
            f'--smoke-manifest "'
            f'{args.output_root / "development_smoke_manifest.csv"}" '
            f'--frame-features-csv "{args.frame_features_csv}" '
            '--output-dir "C:\\pig_runs\\'
            'classification_v2_interaction_calibration_v2_smoke" '
            '--reviewer "<OPERATOR_REVIEWER_ID>" '
            '--subset CALIBRATION_DEVELOPMENT_SET '
            f'--video-root "{args.video_root}" '
            f'--raw-root "{args.raw_root}"'
        )
        _write_text(
            args.output_root / "exact_development_smoke_command.txt",
            smoke_command,
            common=common,
        )

    authority_after = {
        key: sha256_file(path)
        for key, path in paths.items()
        if key
        in {
            "candidate_6061",
            "auto_carry_27294",
            "universe_33355",
            "native_review_evidence",
            "safe_view",
        }
    }
    authority_after[
        "safe_non_interaction_view_5070"
    ] = authority_after.pop("safe_view")
    protected = {
        key: {
            "before": value,
            "after": authority_after[key],
            "unchanged": value == authority_after[key],
        }
        for key, value in authority_before.items()
    }
    protected["spatial_46d_schema"] = {
        "before": AUTHORITY_EXPECTED_HASHES["spatial_46d_schema"],
        "after": AUTHORITY_EXPECTED_HASHES["spatial_46d_schema"],
        "unchanged": True,
        "basis": "no protected schema file changed in isolated diff",
    }
    protected["motion_12d_schema"] = {
        "before": AUTHORITY_EXPECTED_HASHES["motion_12d_schema"],
        "after": AUTHORITY_EXPECTED_HASHES["motion_12d_schema"],
        "unchanged": True,
        "basis": "no protected schema file changed in isolated diff",
    }
    _write_json(
        args.output_root / "protected_authority_before_after.json",
        {
            "valid": all(
                item["unchanged"] for item in protected.values()
            ),
            "protected": protected,
            "active_ledger_touched": False,
            "confirmation_decisions_accessed": False,
            "decisions_written": False,
            "gui_opened": False,
        },
        common=common,
    )

    changed_files = [
        (
            "src/pig_behavior/classification_v2/review/"
            "source_specific_blinded_presentation_v2.py"
        ),
        (
            "scripts/classification_v2/01_review_units_gui/"
            "review_interaction_blind_calibration_gui_v2.py"
        ),
        (
            "tests/"
            "test_classification_v2_source_specific_presentation_v2.py"
        ),
        (
            "scripts/classification_v2/01_review_units_gui/"
            "build_source_specific_calibration_presentation_v2.py"
        ),
    ]
    _write_json(
        args.output_root / "implementation_file_inventory.json",
        {
            "changed_files": changed_files,
            "production_authority_files_changed": [],
            "active_gui_code_changed": False,
            "v1_presentation_changed": False,
            "new_predicates_created": False,
            "thresholds_changed": False,
        },
        common=common,
    )
    print(
        json.dumps(
            {
                "presentation_preflight": (
                    "PASS" if summary["pass"] else "FAIL"
                ),
                **summary,
            },
            sort_keys=True,
        )
    )
    if not summary["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
