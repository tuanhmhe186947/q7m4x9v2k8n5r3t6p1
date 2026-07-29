"""Preflight the frozen legacy non-interaction review presentation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from review_temporal_unit_gui import (
    LEGACY_DECISION_TARGET_FRAME_COUNT,
    LEGACY_DECISION_TARGET_HEADING,
    LEGACY_SOURCE_TYPE,
    GuiConfig,
    ReviewUnitGui,
    legacy_noninteraction_scope_errors,
    load_gui_frame_features,
    review_scope_heading,
)

SEMANTIC_STATUS = "PRE_REVIEW_LEGACY_NONINTERACTION_PRESENTATION_FIX"
REQUIRED_OUTPUT_COLUMNS = (
    "review_key",
    "source_type",
    "expected_target_frame_count",
    "actual_target_frame_count",
    "expected_history_frame_count",
    "actual_history_frame_count",
    "display_frame_count",
    "duplicate_display_frame_count",
    "first_six_are_target",
    "chronological_order_valid",
    "media_readable",
    "heading_contract_valid",
    "render_deterministic",
    "reviewable",
    "failure_reason",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-units-csv", type=Path, required=True)
    parser.add_argument("--frame-features-csv", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--producer-sha", required=True)
    parser.add_argument("--expected-view-sha256", required=True)
    parser.add_argument("--expected-count", type=int, default=726)
    return parser.parse_args()


def _runtime_without_tk(
    args: argparse.Namespace,
    frames: pd.DataFrame,
) -> ReviewUnitGui:
    gui = ReviewUnitGui.__new__(ReviewUnitGui)
    gui.config = GuiConfig(
        review_units_csv=args.review_units_csv,
        frame_features_csv=args.frame_features_csv,
        output_dir=args.output_csv.parent,
        raw_root=args.raw_root,
    )
    gui.frames = frames
    gui.frames["frame_index"] = pd.to_numeric(
        gui.frames["frame_index"],
        errors="coerce",
    )
    if "relative_frame_index" in gui.frames.columns:
        gui.frames["relative_frame_index"] = pd.to_numeric(
            gui.frames["relative_frame_index"],
            errors="coerce",
        )
    return gui


def _expected_target_frames(unit: pd.Series) -> list[int]:
    return list(
        range(
            int(unit["unit_start_frame"]),
            int(unit["unit_end_frame"]) + 1,
        )
    )


def _preflight_item(
    gui: ReviewUnitGui,
    unit: pd.Series,
) -> dict[str, Any]:
    expected_targets = _expected_target_frames(unit)
    actual_targets = gui._display_frames(unit)
    actual_history = gui._history_display_frames(unit)
    displayed = gui._all_display_frames(unit)
    duplicate_count = len(displayed) - len(set(displayed))
    first_six_are_target = bool(
        len(displayed) >= 6
        and displayed[:6] == expected_targets[:6]
        and all(
            gui._display_frame_role(unit, frame_index) == "T"
            for frame_index in expected_targets[:6]
        )
    )
    chronological_order_valid = bool(
        displayed == sorted(displayed)
        and displayed == expected_targets
    )
    heading = review_scope_heading(unit)
    heading_contract_valid = bool(
        heading == LEGACY_DECISION_TARGET_HEADING
        and all(
            gui._display_frame_role(unit, frame_index) == "T"
            for frame_index in displayed
        )
        and "CONTEXT" not in heading
        and "HISTORY" not in heading
        and "NOT DECISION TARGET" not in heading
    )

    frame_rows = gui._frame_rows_for_unit(unit)
    observed_frames = (
        pd.to_numeric(frame_rows["frame_index"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )
    first_render, first_diagnostics = gui._make_contact_sheet(
        unit,
        frame_rows,
    )
    second_render, second_diagnostics = gui._make_contact_sheet(
        unit,
        frame_rows,
    )
    render_deterministic = bool(
        first_diagnostics == second_diagnostics
        and first_render.size == second_render.size
        and first_render.tobytes() == second_render.tobytes()
    )
    media_readable = bool(
        observed_frames == expected_targets
        and not first_diagnostics
    )

    failures: list[str] = []
    if legacy_noninteraction_scope_errors(unit):
        failures.append("legacy_scope_contract")
    if len(actual_targets) != LEGACY_DECISION_TARGET_FRAME_COUNT:
        failures.append("target_frame_count")
    if actual_targets != expected_targets:
        failures.append("target_frame_identity")
    if actual_history:
        failures.append("history_frame_count")
    if displayed != actual_targets:
        failures.append("display_target_mismatch")
    if duplicate_count:
        failures.append("duplicate_display_frames")
    if not first_six_are_target:
        failures.append("first_six_not_target")
    if not chronological_order_valid:
        failures.append("frame_order")
    if not media_readable:
        failures.append("media")
    if not heading_contract_valid:
        failures.append("heading")
    if not render_deterministic:
        failures.append("nondeterministic_render")

    return {
        "review_key": str(
            unit.get("review_key", unit.get("review_unit_id", ""))
        ),
        "source_type": str(unit.get("source_type", "")),
        "expected_target_frame_count": LEGACY_DECISION_TARGET_FRAME_COUNT,
        "actual_target_frame_count": len(actual_targets),
        "expected_history_frame_count": 0,
        "actual_history_frame_count": len(actual_history),
        "display_frame_count": len(displayed),
        "duplicate_display_frame_count": duplicate_count,
        "first_six_are_target": first_six_are_target,
        "chronological_order_valid": chronological_order_valid,
        "media_readable": media_readable,
        "heading_contract_valid": heading_contract_valid,
        "render_deterministic": render_deterministic,
        "reviewable": not failures,
        "failure_reason": ";".join(failures),
    }


def _summary(
    results: pd.DataFrame,
    *,
    config_hash: str,
    input_hashes: dict[str, str],
    producer_sha: str,
) -> dict[str, Any]:
    failed = ~results["reviewable"].astype(bool)
    summary = {
        "semantic_status": SEMANTIC_STATUS,
        "diagnostic_only": True,
        "producer_sha": producer_sha,
        "config_hash": config_hash,
        "input_hashes": input_hashes,
        "total_items": int(len(results)),
        "passed_items": int((~failed).sum()),
        "failed_items": int(failed.sum()),
        "target_frame_count_failures": int(
            (
                results["actual_target_frame_count"]
                != results["expected_target_frame_count"]
            ).sum()
        ),
        "history_frame_count_failures": int(
            (
                results["actual_history_frame_count"]
                != results["expected_history_frame_count"]
            ).sum()
        ),
        "duplicate_display_frame_items": int(
            results["duplicate_display_frame_count"].gt(0).sum()
        ),
        "frame_order_failures": int(
            (~results["chronological_order_valid"].astype(bool)).sum()
        ),
        "heading_failures": int(
            (~results["heading_contract_valid"].astype(bool)).sum()
        ),
        "media_failures": int(
            (~results["media_readable"].astype(bool)).sum()
        ),
        "render_determinism_failures": int(
            (~results["render_deterministic"].astype(bool)).sum()
        ),
    }
    summary["pass"] = bool(
        summary["total_items"] == 726
        and summary["passed_items"] == 726
        and summary["failed_items"] == 0
        and summary["target_frame_count_failures"] == 0
        and summary["history_frame_count_failures"] == 0
        and summary["duplicate_display_frame_items"] == 0
        and summary["frame_order_failures"] == 0
        and summary["heading_failures"] == 0
        and summary["media_failures"] == 0
        and summary["render_determinism_failures"] == 0
    )
    return summary


def main() -> None:
    args = _parse_args()
    if len(args.producer_sha) != 40:
        raise SystemExit("producer SHA must be full")
    if args.output_csv.exists() or args.summary_json.exists():
        raise SystemExit("preflight outputs already exist")

    actual_view_hash = _sha256(args.review_units_csv)
    if actual_view_hash != args.expected_view_sha256:
        raise SystemExit("frozen safe-view hash mismatch")

    units = pd.read_csv(args.review_units_csv, low_memory=False)
    legacy_units = units[
        units["source_type"].astype(str).eq(LEGACY_SOURCE_TYPE)
    ].copy()
    if len(legacy_units) != args.expected_count:
        raise SystemExit(
            f"legacy item count={len(legacy_units)} "
            f"expected={args.expected_count}"
        )
    keys = legacy_units["review_key"].fillna("").astype(str)
    if keys.eq("").any() or keys.duplicated().any():
        raise SystemExit("legacy review keys are blank or duplicated")

    frames = load_gui_frame_features(args.frame_features_csv)
    gui = _runtime_without_tk(args, frames)
    results = pd.DataFrame(
        [
            _preflight_item(gui, unit)
            for _, unit in legacy_units.iterrows()
        ],
        columns=REQUIRED_OUTPUT_COLUMNS,
    )

    input_hashes = {
        "safe_non_interaction_view_5070": actual_view_hash,
        "native_review_evidence": _sha256(args.frame_features_csv),
    }
    config = {
        "schema": "legacy_noninteraction_scope_preflight.v1",
        "expected_count": args.expected_count,
        "legacy_source_type": LEGACY_SOURCE_TYPE,
        "decision_target_frame_count": (
            LEGACY_DECISION_TARGET_FRAME_COUNT
        ),
        "history_frame_count": 0,
        "heading": LEGACY_DECISION_TARGET_HEADING,
    }
    summary = _summary(
        results,
        config_hash=_canonical_hash(config),
        input_hashes=input_hashes,
        producer_sha=args.producer_sha,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_csv, index=False)
    args.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not summary["pass"]:
        raise SystemExit("legacy non-interaction preflight failed")


if __name__ == "__main__":
    main()
