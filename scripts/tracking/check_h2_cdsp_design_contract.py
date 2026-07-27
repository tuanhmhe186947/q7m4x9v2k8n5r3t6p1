"""Fail-closed checker for the design-only H2-CDSP scientific contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

DESIGN_DIR = Path("docs/tracking/h2_cdsp")
CONTRACT = DESIGN_DIR / "H2_CDSP_SCIENTIFIC_DESIGN_CONTRACT.md"
REGISTRY = DESIGN_DIR / "H2_CDSP_STATE_SEMANTICS_REGISTRY.csv"
MACHINE = DESIGN_DIR / "H2_CDSP_STATE_MACHINE.json"
FORMULAS = DESIGN_DIR / "H2_CDSP_PRESERVATION_FORMULAS.md"
GOLDEN = DESIGN_DIR / "H2_CDSP_GOLDEN_CASES.yaml"
PREREQUISITE = DESIGN_DIR / "H2_CDSP_CURRENT_MAIN_SHADOW_PREREQUISITE.json"
DEVELOPMENT = DESIGN_DIR / "H2_CDSP_SHADOW_DEVELOPMENT_MANIFEST.csv"
VALIDATION_POLICY = DESIGN_DIR / "H2_CDSP_VALIDATION_POLICY.md"
EVALUATION = DESIGN_DIR / "H2_CDSP_EVALUATION_GATE.json"
DECISION = DESIGN_DIR / "H2_CDSP_DESIGN_DECISION.json"
REVIEW = DESIGN_DIR / "H2_CDSP_INDEPENDENT_SCIENTIFIC_REVIEW.json"

H1_CLOSURE = Path(
    "docs/tracking/h1_r3/H1_R3_CLOSURE_DECISION_20260727.json"
)

STATES = {
    "VISIBLE_CONFIRMED",
    "DROPOUT_GRACE",
    "OCCLUSION_PRESERVED",
    "STALE_PRESERVED",
    "INVALIDATED",
    "TERMINATED",
}
EXPECTED_CLASSES = {
    "last_trusted_frame_index": "CORE_REQUIRED",
    "last_trusted_bbox": "CORE_REQUIRED",
    "dropout_age": "CORE_REQUIRED",
    "state_validity": "CORE_REQUIRED",
    "causal_frame_continuity": "CORE_REQUIRED",
    "sequence_boundary_token": "CORE_REQUIRED",
    "state_confidence": "CORE_REQUIRED",
    "uncertainty": "CORE_REQUIRED",
    "appearance_prototype": "OPTIONAL_WITH_QUALITY",
    "appearance_quality": "OPTIONAL_WITH_QUALITY",
    "velocity_history": "OPTIONAL_WITH_QUALITY",
    "lk_motion": "OPTIONAL_WITH_QUALITY",
    "occlusion_estimate": "OPTIONAL_WITH_QUALITY",
    "detection_confidence_history": "DIAGNOSTIC_ONLY",
    "gt_identity": "REJECTED",
    "runtime_context_identifiers": "REJECTED",
}
GOLDEN_IDS = {
    "one_frame_detector_dropout",
    "short_dropout_reliable_velocity",
    "short_dropout_without_appearance",
    "short_dropout_without_motion",
    "short_dropout_both_optional_missing",
    "moderate_dropout_growing_uncertainty",
    "long_dropout_requires_invalidation",
    "terminal_track_must_not_revive",
    "strong_new_trusted_match_restores_confidence",
    "weak_candidate_does_not_refresh",
    "unassigned_detection_cannot_update_appearance",
    "equivalent_motion_small_and_large_boxes",
    "camera_boundary_geometry_uncertainty",
    "lk_failure_during_dropout",
    "skipped_detector_frames_cadence_two",
    "visible_competitor_no_reservation",
    "cross_video_transition_clears_state",
    "nan_or_malformed_state_fails_closed",
}
POSITIVE_EVENTS = {
    "RF_ACC23_E003",
    "RF_ACC23_E006",
    "RF_ACC23_E007",
    "RF_ACC23_E008",
    "RF_ACC23_E009",
    "RF_ACC23_E010",
}
AUTHORIZED_MAIN_SHA = "d38ea5c804949bec40b739d9319f2e0cdff45c6c"
TRANSITION_IDS = {
    "T01_TRUSTED_MATCH_INITIALIZES",
    "T02_VISIBLE_TO_GRACE",
    "T03_GRACE_CONTINUES",
    "T04_GRACE_TO_OCCLUSION",
    "T05_GRACE_TO_STALE",
    "T06_OCCLUSION_CONTINUES",
    "T07_OCCLUSION_TO_STALE",
    "T08_STALE_CONTINUES",
    "T09_INVALIDATED_BASELINE_REACQUIRE",
    "T10_FAIL_CLOSED_INVALIDATION",
    "T11_SEQUENCE_BOUNDARY_CLEARS",
    "T12_BASELINE_TERMINATES",
    "T13_TERMINAL_ABSORBING",
}
TOLERANCE = 1e-9
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(RuntimeError):
    """Raised when the frozen H2 design packet is inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected object")
    return value


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unauthorized(payload: dict[str, Any], context: str) -> None:
    values = payload.get("authorizations", payload)
    for key in (
        "implementation_authorized",
        "shadow_execution_authorized",
        "association_evaluation_authorized",
        "validation_authorized",
        "runtime_authorized",
        "promotion_authorized",
    ):
        if values.get(key) is not False:
            raise ContractError(f"{context} authorizes {key}")


def _check_paths(pre_review: bool) -> None:
    paths = [
        CONTRACT,
        REGISTRY,
        MACHINE,
        FORMULAS,
        GOLDEN,
        PREREQUISITE,
        DEVELOPMENT,
        VALIDATION_POLICY,
        EVALUATION,
        DECISION,
    ]
    if not pre_review:
        paths.append(REVIEW)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ContractError(f"missing H2 design artifacts: {missing}")


def _check_authority_language() -> None:
    closure = _read_json(H1_CLOSURE)
    if closure.get("hidden_owner_preference_family_status") != (
        "CLOSED_FOR_CURRENT_STUDY"
    ):
        raise ContractError("H1 family is not closed")
    if closure.get("decision") != "FAIL_NO_SHADOW_ACTIVATION":
        raise ContractError("H1-r3 closure changed")
    if closure.get("authorizations", {}).get("h1_r4_authorized") is not False:
        raise ContractError("H1-r4 became authorized")
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            CONTRACT,
            FORMULAS,
            PREREQUISITE,
            VALIDATION_POLICY,
            EVALUATION,
            DECISION,
        )
    )
    for required in (
        "b0d9009",
        "MECHANISM_DISCOVERY_ONLY",
        "NOT_MEASURED",
        "000216",
    ):
        if required not in combined:
            raise ContractError(f"authority language missing: {required}")
    if "fresh current-main baseline" in combined.lower():
        raise ContractError("historical evidence is mislabeled as current")


def _check_registry() -> None:
    columns, rows = _read_csv(REGISTRY)
    required_columns = {
        "state_field",
        "evidence_class",
        "formula_or_source",
        "range_or_type",
        "units_or_normalization",
        "clipping_policy",
        "missing_behavior",
        "monotonic_direction",
        "validity_rule",
        "quality_or_age_decay",
        "maximum_influence",
        "absence_limits_duration",
        "causal",
        "rejected_runtime_use",
    }
    if set(columns) != required_columns:
        raise ContractError("state semantics registry columns changed")
    by_field = {row["state_field"]: row for row in rows}
    if set(by_field) != set(EXPECTED_CLASSES) or len(rows) != 16:
        raise ContractError("state semantics population changed")
    for field, evidence_class in EXPECTED_CLASSES.items():
        row = by_field[field]
        if row["evidence_class"] != evidence_class:
            raise ContractError(f"{field}: evidence class changed")
        if not all(
            row[key].strip()
            for key in (
                "formula_or_source",
                "range_or_type",
                "missing_behavior",
                "validity_rule",
                "maximum_influence",
            )
        ):
            raise ContractError(f"{field}: semantics incomplete")
        causal = row["causal"] == "true"
        rejected = row["rejected_runtime_use"] == "true"
        if evidence_class == "REJECTED":
            if causal or not rejected:
                raise ContractError(f"{field}: rejected input became usable")
        elif not causal or rejected:
            raise ContractError(f"{field}: causal input contract changed")


def _check_machine() -> None:
    payload = _read_json(MACHINE)
    states = payload.get("states")
    if not isinstance(states, list):
        raise ContractError("state machine states missing")
    by_name = {row.get("name"): row for row in states}
    if set(by_name) != STATES or len(states) != len(STATES):
        raise ContractError("state population changed")
    if by_name["DROPOUT_GRACE"]["maximum_duration_frames"] != 2:
        raise ContractError("dropout grace duration changed")
    if by_name["OCCLUSION_PRESERVED"]["maximum_duration_frames"] != 6:
        raise ContractError("occlusion duration changed")
    if by_name["STALE_PRESERVED"]["maximum_duration_frames"] != 10:
        raise ContractError("finite preservation duration changed")
    if any(row.get("h2_may_emit_track") is not False for row in states):
        raise ContractError("H2 state grants emission permission")
    transitions = payload.get("transitions")
    if not isinstance(transitions, list) or len(transitions) != 13:
        raise ContractError("transition population changed")
    ids = {row.get("transition_id") for row in transitions}
    if ids != TRANSITION_IDS or len(ids) != len(transitions):
        raise ContractError("transition IDs changed or duplicated")
    terminal = next(
        row
        for row in transitions
        if row["transition_id"] == "T13_TERMINAL_ABSORBING"
    )
    if terminal.get("from") != ["TERMINATED"] or terminal.get("to") != (
        "TERMINATED"
    ):
        raise ContractError("terminal state is not absorbing")
    for row in transitions:
        sources = row.get("from")
        target = row.get("to")
        if not isinstance(sources, list) or not sources:
            raise ContractError(f"{row.get('transition_id')}: sources missing")
        if any(source not in STATES for source in sources):
            raise ContractError(f"{row.get('transition_id')}: unknown source")
        if target not in STATES:
            raise ContractError(f"{row.get('transition_id')}: unknown target")
        for key in (
            "causal_trigger",
            "required_evidence",
            "state_confidence_update",
            "uncertainty_update",
            "appearance_handling",
            "motion_handling",
            "association_evidence_usable",
            "h2_may_emit_track",
            "invalidation_condition",
        ):
            if key not in row:
                raise ContractError(f"{row.get('transition_id')}: missing {key}")
        if row["h2_may_emit_track"] is not False:
            raise ContractError("transition grants H2 emission")
    reachable = {"VISIBLE_CONFIRMED"}
    changed = True
    while changed:
        changed = False
        for row in transitions:
            if reachable.intersection(row["from"]) and row["to"] not in reachable:
                reachable.add(row["to"])
                changed = True
    if reachable != STATES:
        raise ContractError(f"unreachable H2 states: {sorted(STATES - reachable)}")
    mapping = payload.get("baseline_lifecycle_mapping", {})
    if set(mapping) != {
        "live_track",
        "trusted_match",
        "VISIBLE_CONFIRMED",
        "DROPOUT_GRACE",
        "OCCLUSION_PRESERVED",
        "STALE_PRESERVED",
        "INVALIDATED",
        "TERMINATED",
        "sequence_reset",
    }:
        raise ContractError("FixedTrack lifecycle mapping is incomplete")
    mapping_text = " ".join(str(value) for value in mapping.values())
    for required in (
        "FixedTrack.state",
        "last_source",
        "last_ambiguous",
        "active baseline track dictionary",
        "LOST",
    ):
        if required not in mapping_text:
            raise ContractError(f"lifecycle mapping missing: {required}")
    precedence = payload.get("transition_precedence")
    if not isinstance(precedence, list) or len(precedence) != 8:
        raise ContractError("transition precedence is not frozen")
    precedence_text = " ".join(precedence)
    for transition_id in (
        "T11_SEQUENCE_BOUNDARY_CLEARS",
        "T12_BASELINE_TERMINATES",
        "T10_FAIL_CLOSED_INVALIDATION",
        "T01_TRUSTED_MATCH_INITIALIZES",
        "T09_INVALIDATED_BASELINE_REACQUIRE",
        "T13_TERMINAL_ABSORBING",
    ):
        if transition_id not in precedence_text:
            raise ContractError(f"transition precedence missing {transition_id}")
    positions = {
        transition_id: next(
            index
            for index, item in enumerate(precedence)
            if transition_id in item
        )
        for transition_id in (
            "T13_TERMINAL_ABSORBING",
            "T12_BASELINE_TERMINATES",
            "T09_INVALIDATED_BASELINE_REACQUIRE",
            "T10_FAIL_CLOSED_INVALIDATION",
            "T01_TRUSTED_MATCH_INITIALIZES",
        )
    }
    if not (
        positions["T13_TERMINAL_ABSORBING"]
        < positions["T12_BASELINE_TERMINATES"]
        < positions["T09_INVALIDATED_BASELINE_REACQUIRE"]
        < positions["T10_FAIL_CLOSED_INVALIDATION"]
        < positions["T01_TRUSTED_MATCH_INITIALIZES"]
    ):
        raise ContractError("transition precedence makes recovery unreachable")
    transition_cases = (
        (
            {"source": "TERMINATED"},
            "T13_TERMINAL_ABSORBING",
        ),
        (
            {"source": "INVALIDATED", "trusted_match": True},
            "T09_INVALIDATED_BASELINE_REACQUIRE",
        ),
        (
            {"source": "INVALIDATED"},
            "T10_FAIL_CLOSED_INVALIDATION",
        ),
        (
            {"source": "VISIBLE_CONFIRMED", "baseline_removed": True},
            "T12_BASELINE_TERMINATES",
        ),
        (
            {"source": "VISIBLE_CONFIRMED", "trusted_match": True},
            "T01_TRUSTED_MATCH_INITIALIZES",
        ),
        (
            {"source": "STALE_PRESERVED", "age": 8},
            "T08_STALE_CONTINUES",
        ),
    )
    for inputs, expected in transition_cases:
        if _select_transition(**inputs) != expected:
            raise ContractError(
                f"transition guards do not execute deterministically: {expected}"
            )
    non_transition = payload.get("non_transition_invariants", {})
    if (
        "never refreshes"
        not in non_transition.get(
            "weak_ambiguous_or_unassigned_detection",
            "",
        )
        or non_transition.get("baseline_lost_is_not_termination") is not True
        or non_transition.get("occlusion_support_adds_confidence") is not False
    ):
        raise ContractError("non-transition safety rules changed")
    invariants = payload.get("global_invariants", {})
    expected_false = (
        "reserves_detection",
        "directly_assigns_detection",
        "blocks_visible_assignment",
        "creates_track",
        "adds_emission",
        "future_frames_used",
        "gt_runtime_input",
        "video_or_episode_shortcut",
        "telemetry_required_for_algorithm",
    )
    if any(invariants.get(key) is not False for key in expected_false):
        raise ContractError("state machine violates intervention boundary")
    if invariants.get("output_delay_frames") != 0:
        raise ContractError("state machine delay changed")
    consumer = payload.get("association_consumer_contract", {})
    if (
        consumer.get("record_type") != "PreservedStateEvidence"
        or "one-for-one substitution"
        not in consumer.get("sole_permitted_use", "")
        or consumer.get("candidate_set_unchanged") is not True
        or consumer.get("ordinary_costs_weights_gates_solver_unchanged")
        is not True
        or consumer.get("new_pair_penalty_or_bonus") is not False
        or consumer.get("owner_preference_score") is not False
        or consumer.get("detection_reservation") is not False
        or consumer.get("candidate_removal") is not False
        or consumer.get("visible_assignment_veto") is not False
        or consumer.get("direct_assignment") is not False
        or consumer.get("no_divergence_outcome")
        != "FAIL_NO_ASSOCIATION_EFFECT"
    ):
        raise ContractError("bounded association consumer changed")
    _unauthorized(payload, "state machine")


def _select_transition(
    *,
    source: str,
    sequence_changed: bool = False,
    baseline_removed: bool = False,
    trusted_match: bool = False,
    invalid: bool = False,
    age: int | None = None,
    occlusion_support: bool = False,
) -> str:
    if source == "TERMINATED":
        return "T13_TERMINAL_ABSORBING"
    if sequence_changed:
        return "T11_SEQUENCE_BOUNDARY_CLEARS"
    if baseline_removed:
        return "T12_BASELINE_TERMINATES"
    if source == "INVALIDATED" and trusted_match:
        return "T09_INVALIDATED_BASELINE_REACQUIRE"
    if source == "INVALIDATED" or invalid:
        return "T10_FAIL_CLOSED_INVALIDATION"
    if trusted_match:
        return "T01_TRUSTED_MATCH_INITIALIZES"
    if source == "VISIBLE_CONFIRMED" and age == 1:
        return "T02_VISIBLE_TO_GRACE"
    if source == "DROPOUT_GRACE" and age is not None:
        if age <= 2:
            return "T03_GRACE_CONTINUES"
        return (
            "T04_GRACE_TO_OCCLUSION"
            if occlusion_support and age <= 6
            else "T05_GRACE_TO_STALE"
        )
    if source == "OCCLUSION_PRESERVED" and age is not None:
        if occlusion_support and age <= 6:
            return "T06_OCCLUSION_CONTINUES"
        return "T07_OCCLUSION_TO_STALE"
    if source == "STALE_PRESERVED" and age is not None and 3 <= age <= 10:
        return "T08_STALE_CONTINUES"
    return "T10_FAIL_CLOSED_INVALIDATION"


def _constants() -> dict[str, float]:
    payload = _read_json(GOLDEN)
    constants = payload.get("constants", {})
    expected = {
        "confidence_half_life_frames": 6,
        "appearance_half_life_frames": 8,
        "motion_half_life_frames": 4,
        "base_uncertainty_growth_per_frame": 0.05,
        "weak_motion_uncertainty_growth_per_frame": 0.10,
        "camera_boundary_penalty": 0.15,
        "maximum_preservation_age_frames": 10,
        "minimum_usable_confidence": 0.30,
        "maximum_usable_uncertainty": 0.75,
    }
    if constants != expected:
        raise ContractError("formula constants changed")
    return {key: float(value) for key, value in constants.items()}


def _formula_values(
    inputs: dict[str, Any],
    constants: dict[str, float],
) -> dict[str, Any]:
    if (
        inputs.get("malformed_numeric")
        or inputs.get("sequence_changed")
        or inputs.get("baseline_terminated")
    ):
        state = (
            "TERMINATED"
            if inputs.get("baseline_terminated")
            else "INVALIDATED"
        )
        return {
            "state": state,
            "confidence": 0.0,
            "uncertainty": 1.0,
            "motion_reliability": 0.0,
            "appearance_reliability": 0.0,
            "usable": False,
        }
    age = int(inputs["age"])
    confidence = float(inputs["initial_confidence"]) * 2.0 ** (
        -age / constants["confidence_half_life_frames"]
    )
    motion = (
        float(inputs["motion_quality"])
        * 2.0 ** (-age / constants["motion_half_life_frames"])
        if inputs["motion_available"]
        else 0.0
    )
    appearance = (
        float(inputs["appearance_quality"])
        * 2.0 ** (-age / constants["appearance_half_life_frames"])
        if inputs["appearance_available"]
        else 0.0
    )
    boundary = (
        constants["camera_boundary_penalty"]
        if inputs.get("boundary_seen")
        else 0.0
    )
    uncertainty = min(
        1.0,
        float(inputs["initial_uncertainty"])
        + age
        * (
            constants["base_uncertainty_growth_per_frame"]
            + constants["weak_motion_uncertainty_growth_per_frame"]
            * (1.0 - motion)
        )
        + boundary,
    )
    if inputs.get("trusted_match"):
        state = "VISIBLE_CONFIRMED"
    elif age <= 2:
        state = "DROPOUT_GRACE"
    elif age <= 6 and inputs.get("occlusion_support"):
        state = "OCCLUSION_PRESERVED"
    else:
        state = "STALE_PRESERVED"
    preserved = state in {
        "DROPOUT_GRACE",
        "OCCLUSION_PRESERVED",
        "STALE_PRESERVED",
    }
    usable = (
        preserved
        and age <= constants["maximum_preservation_age_frames"]
        and confidence >= constants["minimum_usable_confidence"]
        and uncertainty <= constants["maximum_usable_uncertainty"]
    )
    if preserved and not usable:
        state = "INVALIDATED"
    return {
        "state": state,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "motion_reliability": motion,
        "appearance_reliability": appearance,
        "usable": usable,
    }


def _normalized_center_velocity(
    previous: list[float],
    trusted: list[float],
) -> float:
    previous_center = (
        (previous[0] + previous[2]) / 2.0,
        (previous[1] + previous[3]) / 2.0,
    )
    trusted_center = (
        (trusted[0] + trusted[2]) / 2.0,
        (trusted[1] + trusted[3]) / 2.0,
    )
    diagonal = math.hypot(
        previous[2] - previous[0],
        previous[3] - previous[1],
    )
    return math.dist(previous_center, trusted_center) / diagonal


def _check_golden() -> dict[str, int]:
    payload = _read_json(GOLDEN)
    constants = _constants()
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ContractError("golden cases missing")
    by_id = {row.get("case_id"): row for row in cases}
    if set(by_id) != GOLDEN_IDS or len(cases) != 18:
        raise ContractError("golden case population changed")
    counts = {"usable_short": 0, "unsafe": 0, "nonassigning": 0}
    for case in cases:
        if (
            not case["input"].get("malformed_numeric")
            and float(case["input"]["initial_uncertainty"]) != 0.10
        ):
            raise ContractError(
                f"{case['case_id']}: trusted initial uncertainty changed"
            )
        actual = _formula_values(case["input"], constants)
        expected = case["expected"]
        for key in (
            "confidence",
            "uncertainty",
            "motion_reliability",
            "appearance_reliability",
        ):
            if not math.isclose(
                float(actual[key]),
                float(expected[key]),
                abs_tol=TOLERANCE,
            ):
                raise ContractError(f"{case['case_id']}: {key} mismatch")
        if actual["state"] != expected["state"]:
            raise ContractError(f"{case['case_id']}: state mismatch")
        if actual["usable"] is not expected["usable"]:
            raise ContractError(f"{case['case_id']}: usability mismatch")
        if (
            expected.get("direct_assignment") is not False
            or expected.get("reserves_detection") is not False
        ):
            raise ContractError(f"{case['case_id']}: H2 commands assignment")
        if expected["usable"] and int(case["input"]["age"]) <= 6:
            counts["usable_short"] += 1
        if not expected["usable"]:
            counts["unsafe"] += 1
        if case["case_id"] in {
            "unassigned_detection_cannot_update_appearance",
            "visible_competitor_no_reservation",
        }:
            counts["nonassigning"] += 1
    if counts["usable_short"] < 4:
        raise ContractError("realistic short-dropout region is too small")
    if counts["unsafe"] < 4:
        raise ContractError("unsafe invalidation region is too small")
    if counts["nonassigning"] < 2:
        raise ContractError("non-assignment golden coverage is incomplete")
    variants = by_id["equivalent_motion_small_and_large_boxes"]["input"][
        "geometry_variants"
    ]
    velocities = [
        _normalized_center_velocity(row["previous_box"], row["trusted_box"])
        for row in variants
    ]
    if not math.isclose(velocities[0], velocities[1], abs_tol=TOLERANCE):
        raise ContractError("bbox scale invariance failed")
    expected_velocity = by_id[
        "equivalent_motion_small_and_large_boxes"
    ]["expected"]["normalized_center_velocity"]
    if not math.isclose(
        velocities[0],
        expected_velocity,
        abs_tol=TOLERANCE,
    ):
        raise ContractError("normalized velocity expectation mismatch")
    return counts


def _check_formula_properties() -> None:
    constants = _constants()
    for motion_available in (False, True):
        prior_confidence = math.inf
        prior_uncertainty = -math.inf
        for age in range(0, 12):
            values = _formula_values(
                {
                    "age": age,
                    "initial_confidence": 1.0,
                    "initial_uncertainty": 0.1,
                    "motion_available": motion_available,
                    "motion_quality": 0.8 if motion_available else 0.0,
                    "appearance_available": True,
                    "appearance_quality": 0.7,
                    "occlusion_support": True,
                    "boundary_seen": age >= 2,
                },
                constants,
            )
            if values["confidence"] > prior_confidence + TOLERANCE:
                raise ContractError("confidence increases without trusted evidence")
            if values["uncertainty"] < prior_uncertainty - TOLERANCE:
                raise ContractError("uncertainty decreases during dropout")
            prior_confidence = values["confidence"]
            prior_uncertainty = values["uncertainty"]
    with_motion = _formula_values(
        {
            "age": 3,
            "initial_confidence": 1.0,
            "initial_uncertainty": 0.1,
            "motion_available": True,
            "motion_quality": 0.8,
            "appearance_available": True,
            "appearance_quality": 0.7,
            "occlusion_support": True,
        },
        constants,
    )
    without_optional = _formula_values(
        {
            "age": 3,
            "initial_confidence": 1.0,
            "initial_uncertainty": 0.1,
            "motion_available": False,
            "motion_quality": 0.0,
            "appearance_available": False,
            "appearance_quality": 0.0,
            "occlusion_support": True,
        },
        constants,
    )
    if without_optional["confidence"] > with_motion["confidence"] + TOLERANCE:
        raise ContractError("missing optional evidence increases confidence")
    if without_optional["uncertainty"] < with_motion["uncertainty"] - TOLERANCE:
        raise ContractError("missing motion reduces uncertainty")


def _check_prerequisite_and_manifest() -> tuple[int, int]:
    payload = _read_json(PREREQUISITE)
    if payload.get("status") != "FROZEN_NOT_AUTHORIZED":
        raise ContractError("shadow prerequisite is not frozen")
    if payload.get("authorized_current_main_sha") != AUTHORIZED_MAIN_SHA:
        raise ContractError("shadow prerequisite current-main SHA changed")
    historical = payload.get("historical_evidence", {})
    if (
        historical.get("source_sha") != "b0d9009"
        or historical.get("role") != "MECHANISM_DISCOVERY_ONLY"
        or historical.get("current_main_failure_prevalence") != "NOT_MEASURED"
        or historical.get("gt_authority_unresolved_excluded_events")
        != ["000216"]
    ):
        raise ContractError("historical evidence terminology changed")
    manifest = payload.get("manifest", {})
    if manifest.get("sha256") != _sha256(DEVELOPMENT):
        raise ContractError("shadow manifest hash mismatch")
    execution = payload.get("execution_contract", {})
    bindings = execution.get("required_execution_lineage_bindings", {})
    expected_bindings = {
        "code_sha": AUTHORIZED_MAIN_SHA,
        "tracking_subtree_sha256": "MUST_BE_BOUND_BEFORE_EXECUTION",
        "realtime_fast_effective_config_sha256": (
            "MUST_BE_BOUND_BEFORE_EXECUTION"
        ),
        "detector_cache_sha256_by_episode": "MUST_BE_BOUND_BEFORE_EXECUTION",
        "baseline_canonical_output_sha256_by_episode": (
            "MUST_BE_BOUND_BEFORE_SHADOW_COMPARISON"
        ),
        "source_video_sha256_by_episode": "MUST_MATCH_FROZEN_MANIFEST",
        "gt_sha256_by_episode": "MUST_MATCH_FROZEN_MANIFEST",
    }
    if bindings != expected_bindings:
        raise ContractError("future shadow lineage bindings changed")
    for key in (
        "separate_authorization_required",
        "current_main_exact_sha_required",
        "current_main_baseline_reproduction_required",
        "h2_shadow_only",
        "side_effect_free",
        "assignments_must_remain_identical",
        "cost_matrices_must_remain_identical",
        "track_states_must_remain_identical",
        "output_xml_semantically_identical",
    ):
        if execution.get(key) is not True:
            raise ContractError(f"shadow prerequisite weakened: {key}")
    if (
        execution.get("profile") != "realtime_fast"
        or execution.get("future_frames_allowed") is not False
        or execution.get("output_delay_frames") != 0
        or execution.get("detector_inference_calls") != 0
        or execution.get("gpu_inference_calls") != 0
        or execution.get("recursive_run_root_mp4_count") != 0
    ):
        raise ContractError("shadow execution contract changed")
    definitions = payload.get("operational_definitions", {})
    for key in (
        "baseline_state_loss_point",
        "preservable_state_past_loss",
        "relevant_reentry_opportunity",
        "independent_positive_event",
        "current_main_prevalence_claim",
    ):
        if not str(definitions.get(key, "")).strip():
            raise ContractError(f"shadow definition missing: {key}")
    if (
        "distinct video key and distinct recording session"
        not in definitions["independent_positive_event"]
    ):
        raise ContractError("positive-event independence is not operational")
    gates = payload.get("pass_gates", {})
    if (
        gates.get("distinct_positive_video_keys_minimum") != 2
        or gates.get("distinct_positive_recording_sessions_minimum") != 2
        or gates.get("independence_shortfall_outcome") != "INCONCLUSIVE"
    ):
        raise ContractError("shadow independence gates weakened")
    _unauthorized(payload, "shadow prerequisite")

    columns, rows = _read_csv(DEVELOPMENT)
    required_columns = {
        "window_id",
        "historical_event_id",
        "video_key",
        "recording_date_session",
        "warmup_start_frame",
        "score_start_frame",
        "score_end_frame",
        "run_end_frame",
        "development_role",
        "selection_rationale",
        "historical_evidence_sha",
        "historical_evidence_role",
        "gt_authority_status",
        "video_sha256",
        "gt_sha256",
        "validation_eligible",
    }
    if set(columns) != required_columns:
        raise ContractError("shadow manifest columns changed")
    positives = [
        row
        for row in rows
        if row["development_role"].startswith("positive_")
    ]
    controls = [
        row
        for row in rows
        if row["development_role"].startswith("control_")
    ]
    if (
        {row["historical_event_id"] for row in positives} != POSITIVE_EVENTS
        or len(positives) != 6
        or len(controls) != 4
    ):
        raise ContractError("shadow development population changed")
    for row in rows:
        if "000216" in row["video_key"]:
            raise ContractError("000216 entered mechanistic development")
        if (
            row["historical_evidence_sha"] != "b0d9009"
            or row["historical_evidence_role"] != "MECHANISM_DISCOVERY_ONLY"
            or row["validation_eligible"] != "false"
        ):
            raise ContractError("manifest authority changed")
        if not SHA256_RE.fullmatch(row["video_sha256"]):
            raise ContractError("invalid video SHA-256")
        if not SHA256_RE.fullmatch(row["gt_sha256"]):
            raise ContractError("invalid GT SHA-256")
        warmup = int(row["warmup_start_frame"])
        start = int(row["score_start_frame"])
        end = int(row["score_end_frame"])
        run_end = int(row["run_end_frame"])
        if not (0 <= warmup <= start <= end <= run_end <= 1799):
            raise ContractError(f"{row['window_id']}: invalid boundaries")
    return len(positives), len(controls)


def _check_validation_policy_and_outputs() -> None:
    text = VALIDATION_POLICY.read_text(encoding="utf-8")
    for phrase in (
        "Option 2 is selected",
        "new untouched H2 validation population",
        "disjoint",
        "video key",
        "recording date and session",
        "H2 validation outputs exist: `NO`",
    ):
        if phrase not in text:
            raise ContractError(f"validation policy missing: {phrase}")
    root = Path("outputs/tracking")
    if root.exists():
        forbidden = [
            str(path)
            for path in root.rglob("*")
            if "h2_cdsp" in str(path).lower()
            and "validation" in str(path).lower()
        ]
        if forbidden:
            raise ContractError(f"H2 validation output exists: {forbidden[:3]}")


def _check_evaluation() -> None:
    payload = _read_json(EVALUATION)
    if payload.get("status") != "FROZEN_NOT_AUTHORIZED":
        raise ContractError("evaluation gate is not frozen")
    outcomes = set(payload.get("development_outcomes", []))
    required = {
        "FAIL_NO_BASELINE_STATE_LOSS",
        "FAIL_NO_PRESERVABLE_STATE",
        "FAIL_STATE_EXPIRES_BEFORE_REENTRY",
        "FAIL_CONTROL_OVERPRESERVATION",
        "FAIL_NO_ASSOCIATION_EFFECT",
        "FAIL_NO_BENEFICIAL_EFFECT",
        "FAIL_SWAP_REGRESSION",
        "PASS_DEVELOPMENT",
        "INCONCLUSIVE",
    }
    if outcomes != required:
        raise ContractError("evaluation outcomes changed")
    gates = payload.get("required_gates", {})
    if (
        gates.get("independent_positive_events_with_real_state_preservation_minimum")
        != 2
        or gates.get("distinct_positive_video_keys_minimum") != 2
        or gates.get("distinct_positive_recording_sessions_minimum") != 2
        or gates.get("independent_beneficial_identity_outcomes_minimum") != 2
        or gates.get("new_permanent_swaps_maximum") != 0
        or gates.get("new_terminal_swaps_maximum") != 0
        or gates.get("output_delay_frames") != 0
        or gates.get("future_frame_access_allowed") is not False
        or gates.get("recursive_run_root_mp4_count") != 0
    ):
        raise ContractError("evaluation gates weakened")
    boundary = payload.get("association_change_boundary", {})
    if (
        boundary.get("sole_permitted_consumer")
        != "one-for-one track-local evidence substitution"
        or boundary.get(
            "baseline_candidate_set_costs_weights_gates_solver_unchanged"
        )
        is not True
        or boundary.get("owner_score_or_pair_penalty_allowed") is not False
        or boundary.get("reservation_or_visible_veto_allowed") is not False
        or boundary.get("direct_assignment_allowed") is not False
    ):
        raise ContractError("evaluation association boundary changed")
    _unauthorized(payload, "evaluation gate")


def _check_decision(pre_review: bool) -> None:
    payload = _read_json(DECISION)
    if payload.get("hypothesis") != "H2_CAUSAL_DROPOUT_STATE_PRESERVATION":
        raise ContractError("design hypothesis changed")
    if payload.get("historical_evidence_source_sha") != "b0d9009":
        raise ContractError("historical source changed")
    if payload.get("historical_evidence_role") != "MECHANISM_DISCOVERY_ONLY":
        raise ContractError("historical role changed")
    if payload.get("current_main_failure_prevalence") != "NOT_MEASURED":
        raise ContractError("current-main prevalence was claimed")
    if (
        payload.get("h2_reserves_detections") is not False
        or payload.get("h2_directly_assigns_detections") is not False
        or payload.get("h2_uses_future_frames") is not False
    ):
        raise ContractError("H2 intervention boundary changed")
    _unauthorized(payload, "design decision")
    paths = {
        CONTRACT.name: CONTRACT,
        REGISTRY.name: REGISTRY,
        MACHINE.name: MACHINE,
        FORMULAS.name: FORMULAS,
        GOLDEN.name: GOLDEN,
        PREREQUISITE.name: PREREQUISITE,
        DEVELOPMENT.name: DEVELOPMENT,
        VALIDATION_POLICY.name: VALIDATION_POLICY,
        EVALUATION.name: EVALUATION,
    }
    bindings = payload.get("artifact_sha256", {})
    if set(bindings) != set(paths):
        raise ContractError("design artifact binding set changed")
    for name, path in paths.items():
        if bindings[name] != _sha256(path):
            raise ContractError(f"design artifact hash mismatch: {name}")
    result = payload.get("independent_review")
    if pre_review and result not in {"PENDING", "PASS_DESIGN"}:
        raise ContractError("design is not eligible for independent review")
    if not pre_review and result != "PASS_DESIGN":
        raise ContractError("independent PASS_DESIGN review is absent")


def _check_review() -> None:
    payload = _read_json(REVIEW)
    if payload.get("review_decision") != "PASS_DESIGN":
        raise ContractError("independent review did not pass")
    if payload.get("independent_from_authoring") is not True:
        raise ContractError("review independence is absent")
    challenges = payload.get("challenges")
    if not isinstance(challenges, list) or len(challenges) < 8:
        raise ContractError("review challenge coverage is incomplete")
    if any(row.get("result") != "PASS" for row in challenges):
        raise ContractError("independent review contains a failed challenge")
    _unauthorized(payload, "independent review")
    bindings = payload.get("artifact_sha256", {})
    paths = {
        CONTRACT.name: CONTRACT,
        REGISTRY.name: REGISTRY,
        MACHINE.name: MACHINE,
        FORMULAS.name: FORMULAS,
        GOLDEN.name: GOLDEN,
        PREREQUISITE.name: PREREQUISITE,
        DEVELOPMENT.name: DEVELOPMENT,
        VALIDATION_POLICY.name: VALIDATION_POLICY,
        EVALUATION.name: EVALUATION,
    }
    for name, path in paths.items():
        if bindings.get(name) != _sha256(path):
            raise ContractError(f"review artifact hash mismatch: {name}")


def check_contract(*, pre_review: bool) -> dict[str, int]:
    _check_paths(pre_review)
    _check_authority_language()
    _check_registry()
    _check_machine()
    golden_counts = _check_golden()
    _check_formula_properties()
    positives, controls = _check_prerequisite_and_manifest()
    _check_validation_policy_and_outputs()
    _check_evaluation()
    _check_decision(pre_review)
    if not pre_review:
        _check_review()
    return {
        **golden_counts,
        "positive_events": positives,
        "control_windows": controls,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-review", action="store_true")
    args = parser.parse_args(argv)
    try:
        counts = check_contract(pre_review=args.pre_review)
    except (ContractError, ValueError, KeyError, TypeError) as exc:
        print(f"H2_CDSP_DESIGN_CHECKER=FAIL: {exc}", file=sys.stderr)
        return 1
    status = "PRE_REVIEW_PASS" if args.pre_review else "PASS"
    print(f"H2_CDSP_DESIGN_CHECKER={status}")
    print(f"STATE_COUNT={len(STATES)}")
    print(f"GOLDEN_CASES_PASS={len(GOLDEN_IDS)}/{len(GOLDEN_IDS)}")
    print(f"REALISTIC_SHORT_DROPOUT_USABLE={counts['usable_short']}")
    print(f"STALE_OR_UNSAFE_CASES={counts['unsafe']}")
    print(f"HISTORICAL_POSITIVE_EVENTS={counts['positive_events']}")
    print(f"CONTROL_WINDOWS={counts['control_windows']}")
    print("H2_RESERVES_DETECTIONS=NO")
    print("H2_DIRECTLY_ASSIGNS_DETECTIONS=NO")
    print("IMPLEMENTATION_AUTHORIZED=NO")
    print("SHADOW_EXECUTION_AUTHORIZED=NO")
    print("ASSOCIATION_EVALUATION_AUTHORIZED=NO")
    print("VALIDATION_AUTHORIZED=NO")
    print("RUNTIME_AUTHORIZED=NO")
    print("PROMOTION_AUTHORIZED=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
