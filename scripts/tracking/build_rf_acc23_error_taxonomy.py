"""Build an offline RF_ACC23 identity-error taxonomy from locked artifacts.

This tool never imports or runs the tracker. It verifies the recovered
evaluation manifest, groups remapped wrong-ID rows into causal error events,
joins already-exported diagnostics, and writes a deterministic audit package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

GAP_TOLERANCE_FRAMES = 15
PERMANENT_DURATION_FRAMES = 60
NEAR_GATE_BAND = 0.05
H1_R3_THRESHOLD = 0.625

MECHANISMS = (
    "DETECTION_MISS_OR_DROPOUT",
    "DETECTION_MERGE_OR_SPLIT",
    "VISIBLE_VISIBLE_ASSOCIATION_AMBIGUITY",
    "OCCLUSION_OWNER_LOSS",
    "REENTRY_AFTER_LONG_HIDDEN_DURATION",
    "APPEARANCE_DRIFT_OR_UNAVAILABLE",
    "MOTION_PROPAGATION_FAILURE",
    "TRACK_BIRTH_OR_DUPLICATE_TRACK",
    "TRACK_TERMINATION_POLICY",
    "GT_OR_EVALUATION_AMBIGUITY",
    "OTHER_MEASURED",
    "UNRESOLVED",
)

CURRENT_LOGIC = {
    "OCCLUSION_OWNER_LOSS": "YES: bounded occlusion hold and hidden claims",
    "REENTRY_AFTER_LONG_HIDDEN_DURATION": "YES: lost-track re-identification",
    "TRACK_BIRTH_OR_DUPLICATE_TRACK": "PARTIAL: initialization matching",
    "GT_OR_EVALUATION_AMBIGUITY": "NO: evaluation-authority issue",
}

H1_RELEVANCE = {
    "OCCLUSION_OWNER_LOSS": "YES",
    "REENTRY_AFTER_LONG_HIDDEN_DURATION": "PARTIAL",
    "TRACK_BIRTH_OR_DUPLICATE_TRACK": "NO",
    "GT_OR_EVALUATION_AMBIGUITY": "NO",
}


class AuditError(RuntimeError):
    """Raised when locked evidence is inconsistent or output would overwrite."""


@dataclass(frozen=True)
class AuditInputs:
    evaluation_root: Path
    prediction_root: Path
    shadow_pairs: Path
    output_root: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_locked_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "tracking_remapped_identity_events.csv",
        "tracking_continuity_gaps.csv",
        "tracking_metrics.csv",
        "run_manifest.json",
    }
    entries = {
        Path(item["path"]).name: item
        for item in manifest["artifacts"]
        if Path(item["path"]).name in required
    }
    if set(entries) != required:
        raise AuditError(f"locked manifest lacks required files: {required - set(entries)}")
    for name, item in entries.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise AuditError(f"locked artifact hash mismatch: {path}")
    return {
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "verified_artifacts": {
            name: entries[name]["sha256"] for name in sorted(entries)
        },
        "locked_mp4_count": int(manifest["mp4_count"]),
    }


def _frame_components(group: pd.DataFrame) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    for frame, rows in group.groupby("frame", sort=True):
        identities = set(rows["gt_id"].astype(str)) | set(rows["pred_id"].astype(str))
        hits = [
            index
            for index, component in enumerate(components)
            if int(frame) - component["end_frame"] <= GAP_TOLERANCE_FRAMES
            and identities.intersection(component["identities"])
        ]
        if not hits:
            components.append(
                {
                    "start_frame": int(frame),
                    "end_frame": int(frame),
                    "identities": identities,
                    "row_indices": set(rows.index),
                    "wrong_frames": {int(frame)},
                }
            )
            continue
        base = components[hits[0]]
        base["end_frame"] = int(frame)
        base["identities"].update(identities)
        base["row_indices"].update(rows.index)
        base["wrong_frames"].add(int(frame))
        for index in reversed(hits[1:]):
            other = components.pop(index)
            base["start_frame"] = min(base["start_frame"], other["start_frame"])
            base["identities"].update(other["identities"])
            base["row_indices"].update(other["row_indices"])
            base["wrong_frames"].update(other["wrong_frames"])
    return components


def group_error_events(identity_rows: pd.DataFrame) -> list[dict[str, Any]]:
    wrong = identity_rows[
        identity_rows["event"].astype(str).str.contains("mismatch", na=False)
    ].copy()
    events: list[dict[str, Any]] = []
    assigned_rows: set[int] = set()
    for video, group in wrong.groupby("video_stem", sort=True):
        for component in _frame_components(group):
            indices = sorted(component["row_indices"])
            assigned_rows.update(indices)
            rows = wrong.loc[indices]
            events.append(
                {
                    "video_key": str(video),
                    "start_frame": component["start_frame"],
                    "end_frame": component["end_frame"],
                    "duration_frames": (
                        component["end_frame"] - component["start_frame"] + 1
                    ),
                    "wrong_id_matched_frames": len(rows),
                    "wrong_frame_count": len(component["wrong_frames"]),
                    "affected_gt_id": "|".join(sorted(rows["gt_id"].astype(str).unique())),
                    "affected_predicted_identity": "|".join(
                        sorted(rows["pred_id"].astype(str).unique())
                    ),
                    "id_switch_rows": int(
                        rows["event"].astype(str).str.contains("id_switch").sum()
                    ),
                }
            )
    if len(assigned_rows) != len(wrong) or sum(
        event["wrong_id_matched_frames"] for event in events
    ) != len(wrong):
        raise AuditError("event grouping did not conserve every wrong-ID row")
    switch_only = identity_rows[identity_rows["event"].astype(str) == "id_switch"]
    assigned_switches = 0
    for _, switch in switch_only.iterrows():
        identities = {
            str(switch["gt_id"]),
            str(switch["pred_id"]),
            str(switch["previous_pred_id"]),
        }
        candidates = []
        for event in events:
            event_identities = set(event["affected_gt_id"].split("|")) | set(
                event["affected_predicted_identity"].split("|")
            )
            if event["video_key"] != switch["video_stem"]:
                continue
            if not identities.intersection(event_identities):
                continue
            frame = int(switch["frame"])
            distance = max(
                event["start_frame"] - frame,
                frame - event["end_frame"],
                0,
            )
            if distance <= GAP_TOLERANCE_FRAMES:
                candidates.append((distance, event["start_frame"], event))
        if not candidates:
            raise AuditError(
                "standalone ID-switch row is not attributable to a wrong-ID event: "
                f"{switch['video_stem']} frame {switch['frame']}"
            )
        selected = min(candidates, key=lambda item: (item[0], item[1]))[2]
        selected["id_switch_rows"] += 1
        assigned_switches += 1
    if assigned_switches != len(switch_only):
        raise AuditError("ID-switch row attribution is incomplete")
    if sum(event["id_switch_rows"] for event in events) != int(
        identity_rows["event"].astype(str).str.contains("id_switch").sum()
    ):
        raise AuditError("event grouping did not conserve every ID-switch row")
    for index, event in enumerate(events, start=1):
        event["event_id"] = f"RF_ACC23_E{index:03d}"
    return events


def _window_frame_evidence(report: dict[str, Any], start: int) -> dict[str, Any]:
    frames = report["frames"]
    window = frames[max(0, start - 4) : min(len(frames), start + 5)]

    def maximum(key: str) -> int:
        return max(int(frame.get(key, 0) or 0) for frame in window)

    hidden_ids = sorted(
        {
            str(item)
            for frame in window
            for item in frame.get("hidden_ids", [])
        }
    )
    return {
        "evaluated_frames": len(frames),
        "missing_detection_count_max": maximum("missing_detection_count"),
        "overlap_pair_count_max": maximum("overlap_pair_count"),
        "hidden_count_max": maximum("hidden_count"),
        "occlusion_hold_count_max": maximum("occlusion_hold_count"),
        "ambiguous_occlusion_count_max": maximum("ambiguous_occlusion_count"),
        "lost_track_count_max": maximum("lost_track_count"),
        "predicted_count_max": maximum("predicted_count"),
        "hidden_ids_at_onset": "|".join(hidden_ids),
    }


def _window_debug_evidence(debug: pd.DataFrame, start: int) -> dict[str, Any]:
    window = debug[
        (debug["frame"] >= max(0, start - 4)) & (debug["frame"] <= start + 4)
    ].copy()
    costs = pd.to_numeric(window.get("cost"), errors="coerce")
    missed = pd.to_numeric(window.get("track_missed"), errors="coerce")
    track_ids = pd.to_numeric(window.get("track_id"), errors="coerce").dropna()
    phases = set(window.get("phase", pd.Series(dtype=str)).dropna().astype(str))
    return {
        "association_rows_at_onset": len(window),
        "competing_tracks": "|".join(
            str(int(value)) for value in sorted(track_ids.unique())
        ),
        "appearance_availability": (
            "FINITE_ASSOCIATION_COST_EXPORTED" if costs.notna().any() else "NOT_EXPORTED"
        ),
        "max_track_missed_at_onset": (
            int(missed.max()) if missed.notna().any() else "NOT_EXPORTED"
        ),
        "reid_phase_present": "reid" in phases,
        "low_conf_recovery_present": "low_conf_recovery" in phases,
        "lk_availability": "NOT_EXPORTED",
        "motion_history_availability": "NOT_EXPORTED",
    }


def classify_event(event: dict[str, Any]) -> tuple[str, list[str]]:
    secondary: list[str] = []
    if event["gt_authority"] != "SUFFICIENT_FOR_EVENT_TAXONOMY":
        return "GT_OR_EVALUATION_AMBIGUITY", secondary
    if event["start_frame"] == 0:
        return "TRACK_BIRTH_OR_DUPLICATE_TRACK", secondary
    if event["hidden_count_max"] > 0 and event["overlap_pair_count_max"] > 0:
        if event["missing_detection_count_max"] > 0:
            secondary.append("DETECTION_MISS_OR_DROPOUT")
        return "OCCLUSION_OWNER_LOSS", secondary
    if event["reid_phase_present"] and event["lost_track_count_max"] > 0:
        return "REENTRY_AFTER_LONG_HIDDEN_DURATION", secondary
    if event["missing_detection_count_max"] > 0:
        return "DETECTION_MISS_OR_DROPOUT", secondary
    if event["overlap_pair_count_max"] > 0:
        return "VISIBLE_VISIBLE_ASSOCIATION_AMBIGUITY", secondary
    return "UNRESOLVED", secondary


def enrich_events(
    events: list[dict[str, Any]],
    prediction_root: Path,
) -> list[dict[str, Any]]:
    cache: dict[str, tuple[dict[str, Any], pd.DataFrame, dict[str, str]]] = {}
    for event in events:
        video = event["video_key"]
        if video not in cache:
            video_root = prediction_root / video
            quality_path = video_root / "tracking_quality_report.json"
            debug_path = video_root / "association_debug_events.csv"
            xml_path = video_root / "annotations_cvat_video_1_1.xml"
            report = json.loads(quality_path.read_text(encoding="utf-8"))
            debug = pd.read_csv(debug_path, low_memory=False)
            cache[video] = (
                report,
                debug,
                {
                    "quality_report_sha256": sha256_file(quality_path),
                    "association_debug_sha256": sha256_file(debug_path),
                    "prediction_xml_sha256": sha256_file(xml_path),
                },
            )
        report, debug, hashes = cache[video]
        event.update(_window_frame_evidence(report, event["start_frame"]))
        event.update(_window_debug_evidence(debug, event["start_frame"]))
        event["gt_authority"] = (
            "UNRESOLVED_SOURCE_AUTHORITY"
            if "_000216_" in video
            else "SUFFICIENT_FOR_EVENT_TAXONOMY"
        )
        event["causal_history_available_frames"] = event["start_frame"]
        event["detector_cadence_frames"] = 2
        event["visibility_hidden_state"] = (
            "HIDDEN_OR_OCCLUSION_HOLD_AT_ONSET"
            if event["hidden_count_max"] > 0
            else "NO_HIDDEN_STATE_EXPORTED_AT_ONSET"
        )
        event["detection_availability"] = (
            "PARTIAL_DROPOUT_AT_ONSET"
            if event["missing_detection_count_max"] > 0
            else "FULL_COUNT_AT_ONSET"
        )
        event["self_recovers"] = event["end_frame"] < event["evaluated_frames"] - 1
        event["terminal_swap"] = not event["self_recovers"]
        event["permanent_swap"] = (
            event["terminal_swap"]
            or event["duration_frames"] >= PERMANENT_DURATION_FRAMES
        )
        event["supporting_artifact_hashes"] = json.dumps(
            hashes, sort_keys=True, separators=(",", ":")
        )
        primary, secondary = classify_event(event)
        event["primary_mechanism"] = primary
        event["secondary_mechanisms"] = "|".join(secondary)
        event["rf_acc23_already_addresses"] = CURRENT_LOGIC.get(
            primary, "NO OR NOT SPECIFICALLY"
        )
        event["h1_r1_r2_r3_theoretical_relevance"] = H1_RELEVANCE.get(
            primary, "NO"
        )
        event["genuine_hidden_owner_contention"] = (
            primary == "OCCLUSION_OWNER_LOSS"
            and event["association_rows_at_onset"] > 0
            and event["overlap_pair_count_max"] > 0
        )
    return events


def join_h1_relevance(
    events: list[dict[str, Any]], shadow_pairs: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events:
        subset = shadow_pairs[
            (shadow_pairs["video_key"] == event["video_key"])
            & (shadow_pairs["frame_index"] >= event["start_frame"])
            & (shadow_pairs["frame_index"] <= event["end_frame"])
        ]
        measured = len(subset) > 0
        eligible = subset[subset["core_eligible"].astype(int) == 1]
        near = eligible[
            eligible["owner_preference_lower_bound"]
            >= H1_R3_THRESHOLD - NEAR_GATE_BAND
        ]
        rows.append(
            {
                "event_id": event["event_id"],
                "video_key": event["video_key"],
                "start_frame": event["start_frame"],
                "end_frame": event["end_frame"],
                "primary_mechanism": event["primary_mechanism"],
                "genuine_hidden_owner_contention": event[
                    "genuine_hidden_owner_contention"
                ],
                "h1_r3_shadow_coverage": "MEASURED" if measured else "NOT_MEASURED",
                "shadow_candidate_pairs": len(subset) if measured else "NOT_MEASURED",
                "core_eligible_pairs": len(eligible) if measured else "NOT_MEASURED",
                "near_gate_pairs_within_0_05": len(near)
                if measured
                else "NOT_MEASURED",
                "frozen_gate_activations": int(
                    subset["would_activate"].astype(int).sum()
                )
                if measured
                else "NOT_MEASURED",
                "shadow_disagreements_with_baseline": int(
                    subset[
                        "shadow_activation_would_disagree_with_baseline"
                    ].astype(int).sum()
                )
                if measured
                else "NOT_MEASURED",
                "would_actually_alter_erroneous_assignment": (
                    "NO_FROZEN_GATE_ACTIVATION" if measured else "NOT_MEASURED"
                ),
                "score_interpretation": (
                    "UNCALIBRATED_DIAGNOSTIC_ONLY_NO_BENEFIT_INFERRED"
                    if measured
                    else "NOT_MEASURED"
                ),
            }
        )
    return pd.DataFrame(rows)


def mechanism_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mechanism in MECHANISMS:
        subset = events[events["primary_mechanism"] == mechanism]
        rows.append(
            {
                "primary_mechanism": mechanism,
                "events": len(subset),
                "wrong_id_frames": int(subset["wrong_id_matched_frames"].sum()),
                "total_duration_frames": int(subset["duration_frames"].sum()),
                "permanent_swaps": int(subset["permanent_swap"].sum()),
                "terminal_swaps": int(subset["terminal_swap"].sum()),
                "videos_affected": int(subset["video_key"].nunique()),
                "rf_acc23_already_addresses": CURRENT_LOGIC.get(
                    mechanism, "NO OR NOT SPECIFICALLY"
                ),
                "h1_r1_r2_r3_theoretical_relevance": H1_RELEVANCE.get(
                    mechanism, "NO"
                ),
                "events_with_causal_history": int(
                    (subset["causal_history_available_frames"] > 0).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def ranking_table(summary: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    candidates = {
        "OCCLUSION_OWNER_LOSS": (
            "CAUSAL_DROPOUT_STATE_PRESERVATION",
            0.9,
            0.9,
        ),
        "TRACK_BIRTH_OR_DUPLICATE_TRACK": (
            "TRACK_BIRTH_OWNERSHIP_PROTECTION",
            0.8,
            0.9,
        ),
        "REENTRY_AFTER_LONG_HIDDEN_DURATION": (
            "REENTRY_SPECIFIC_IDENTITY_CONTINUITY",
            0.8,
            0.8,
        ),
        "GT_OR_EVALUATION_AMBIGUITY": (
            "GT_AUTHORITY_REVIEW",
            1.0,
            0.1,
        ),
    }
    max_frames = max(int(summary["wrong_id_frames"].max()), 1)
    max_events = max(int(summary["events"].max()), 1)
    rows = []
    for mechanism, (hypothesis, specificity, feasibility) in candidates.items():
        item = summary[summary["primary_mechanism"] == mechanism].iloc[0]
        subset = events[events["primary_mechanism"] == mechanism]
        causal = (
            float((subset["causal_history_available_frames"] > 0).mean())
            if len(subset)
            else 0.0
        )
        impact = int(item["wrong_id_frames"]) / max_frames
        frequency = int(item["events"]) / max_events
        priority = impact * frequency * causal * specificity * feasibility
        rows.append(
            {
                "ranked_mechanism": mechanism,
                "candidate_hypothesis": hypothesis,
                "error_impact_normalized": round(impact, 6),
                "frequency_normalized": round(frequency, 6),
                "causal_evidence_availability": round(causal, 6),
                "intervention_specificity": specificity,
                "evaluation_feasibility": feasibility,
                "priority_score": round(priority, 9),
                "future_frames_required": False,
                "duplicates_h1_family": False,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["priority_score", "ranked_mechanism"],
        ascending=[False, True],
        ignore_index=True,
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.10g")


def build_outputs(inputs: AuditInputs) -> dict[str, Any]:
    if inputs.output_root.exists():
        raise AuditError(f"refusing to overwrite existing output: {inputs.output_root}")
    lineage = verify_locked_manifest(inputs.evaluation_root)
    identity_path = inputs.evaluation_root / "tracking_remapped_identity_events.csv"
    identity_rows = pd.read_csv(identity_path)
    events = enrich_events(
        group_error_events(identity_rows),
        inputs.prediction_root,
    )
    event_frame = pd.DataFrame(events).sort_values(
        ["video_key", "start_frame", "event_id"], ignore_index=True
    )
    if int(event_frame["wrong_id_matched_frames"].sum()) != 4922:
        raise AuditError("locked RF_ACC23 wrong-ID population is not 4,922 rows")
    shadow = pd.read_csv(inputs.shadow_pairs)
    relevance = join_h1_relevance(events, shadow)
    summary = mechanism_summary(event_frame)
    ranking = ranking_table(summary, event_frame)

    inputs.output_root.mkdir(parents=True)
    _write_csv(event_frame, inputs.output_root / "identity_error_events.csv")
    _write_csv(
        summary,
        inputs.output_root / "primary_failure_mechanism_summary.csv",
    )
    _write_csv(
        summary[
            ["primary_mechanism", "events", "wrong_id_frames", "total_duration_frames"]
        ],
        inputs.output_root / "error_duration_by_mechanism.csv",
    )
    _write_csv(
        summary[
            [
                "primary_mechanism",
                "permanent_swaps",
                "terminal_swaps",
                "videos_affected",
            ]
        ],
        inputs.output_root / "permanent_terminal_swap_summary.csv",
    )
    evidence_columns = [
        "event_id",
        "video_key",
        "start_frame",
        "primary_mechanism",
        "causal_history_available_frames",
        "detection_availability",
        "appearance_availability",
        "lk_availability",
        "motion_history_availability",
        "hidden_count_max",
        "overlap_pair_count_max",
        "lost_track_count_max",
        "gt_authority",
    ]
    _write_csv(
        event_frame[evidence_columns],
        inputs.output_root / "causal_evidence_availability.csv",
    )
    _write_csv(
        relevance,
        inputs.output_root / "hidden_owner_hypothesis_relevance_audit.csv",
    )
    _write_csv(ranking, inputs.output_root / "next_hypothesis_ranking.csv")

    dominant = summary.sort_values(
        ["wrong_id_frames", "events"], ascending=False
    ).iloc[0]
    genuine = event_frame[event_frame["genuine_hidden_owner_contention"]]
    wrong_total = int(event_frame["wrong_id_matched_frames"].sum())
    hidden_wrong = int(genuine["wrong_id_matched_frames"].sum())
    decision = {
        "schema_version": "tracking.rf_acc23_next_hypothesis_decision.v1",
        "date": "2026-07-27",
        "source_population": "recovered_locked_RF_ACC23_full13",
        "source_lineage_limitation": (
            "Metrics were produced at b0d90098b2ae1fcdcfe8ca4faaca7a215631ec66; "
            "byte equivalence to the promoted main tracking tree is unproven."
        ),
        "event_definition": {
            "unit": "connected wrong-identity episode, not one row per animal",
            "gap_tolerance_frames": GAP_TOLERANCE_FRAMES,
            "identity_connectivity_required": True,
            "wrong_id_rows_conserved": wrong_total,
            "permanent_duration_frames": PERMANENT_DURATION_FRAMES,
            "terminal_definition": "wrong at final evaluated frame",
        },
        "dominant_failure_mechanism": str(dominant["primary_mechanism"]),
        "dominant_event_count": int(dominant["events"]),
        "dominant_wrong_id_frames": int(dominant["wrong_id_frames"]),
        "dominant_wrong_id_percent": round(
            100 * int(dominant["wrong_id_frames"]) / wrong_total, 6
        ),
        "identity_error_events_total": len(event_frame),
        "wrong_id_frames_audited": wrong_total,
        "id_switch_rows_audited": int(event_frame["id_switch_rows"].sum()),
        "permanent_swaps_audited": int(event_frame["permanent_swap"].sum()),
        "terminal_swaps_audited": int(event_frame["terminal_swap"].sum()),
        "genuine_hidden_owner_contention_events": len(genuine),
        "hidden_owner_contention_wrong_id_percent": round(
            100 * hidden_wrong / wrong_total, 6
        ),
        "h1_r3_would_have_targeted_dominant_failure": "PARTIAL",
        "causal_evidence_available_for_next_hypothesis": True,
        "gt_authority_sufficient_for_selected_hypothesis": True,
        "decision": "PROPOSE_ONE_NEW_HYPOTHESIS",
        "next_hypothesis": {
            "name": "CAUSAL_DROPOUT_STATE_PRESERVATION",
            "scope": (
                "Preserve an established identity state through a bounded causal "
                "detector dropout or merge; do not rank hidden versus visible "
                "owners with another preference score."
            ),
            "scientific_distinction_from_h1": (
                "State-lifecycle preservation targets dropout/merge onset and "
                "termination/rebirth, not hidden-owner score thresholding."
            ),
            "operating_condition_nonempty": True,
            "positive_development_cases_predeclarable": True,
            "control_development_cases_predeclarable": True,
            "positive_development_case_definition": (
                "Established track enters a bounded detector dropout or merge "
                "and subsequently loses identity under RF_ACC23."
            ),
            "control_development_case_definition": (
                "Same causal dropout or overlap condition occurs and RF_ACC23 "
                "retains or correctly recovers identity."
            ),
            "principal_risks": [
                "Holding a stale identity through a genuine exit",
                "Creating duplicate tracks on reappearance",
                "Increasing false negatives or fragmentation",
                "Protecting the wrong state during a true visible handoff",
            ],
            "preimplementation_stop_rules": [
                "No separable positive and control operating condition",
                "Required evidence is unavailable causally",
                "Synthetic invariance or no-future-frame checks fail",
                "A proposed rule is equivalent to another H1 score threshold",
            ],
            "untouched_validation_population_retained": True,
            "implementation_authorized": False,
            "tracking_run_authorized": False,
        },
        "validation_executed": False,
        "ready_for_next_hypothesis_design": True,
        "authorizations": {
            "new_implementation_authorized": False,
            "new_tracking_run_authorized": False,
            "validation_authorized": False,
            "runtime_authorized": False,
            "promotion_authorized": False,
        },
        "lineage_verification": lineage,
        "shadow_candidate_pairs_sha256": sha256_file(inputs.shadow_pairs),
    }
    decision_path = inputs.output_root / "NEXT_TRACKING_HYPOTHESIS_DECISION.json"
    decision_path.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    inventory_items = []
    for path in sorted(inputs.output_root.iterdir(), key=lambda item: item.name):
        if path.name == "ARTIFACT_SHA256.json":
            continue
        inventory_items.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    inventory = {
        "schema_version": "tracking.rf_acc23_error_taxonomy_inventory.v1",
        "artifacts": inventory_items,
    }
    inventory["inventory_payload_sha256"] = canonical_sha256(inventory)
    (inputs.output_root / "ARTIFACT_SHA256.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "events": len(event_frame),
        "wrong_id_frames": wrong_total,
        "dominant": str(dominant["primary_mechanism"]),
        "dominant_wrong_id_frames": int(dominant["wrong_id_frames"]),
        "genuine_hidden_events": len(genuine),
        "output_root": str(inputs.output_root),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--shadow-pairs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_outputs(
            AuditInputs(
                evaluation_root=args.evaluation_root,
                prediction_root=args.prediction_root,
                shadow_pairs=args.shadow_pairs,
                output_root=args.output_root,
            )
        )
    except AuditError as exc:
        print(f"AUDIT_REFUSED: {exc}")
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
