"""Fail-closed validation for the design-only H1-r3 scientific contract."""

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

DESIGN_DIR = Path("docs/tracking/h1_r3")
CONTRACT = DESIGN_DIR / "H1_R3_SCIENTIFIC_DESIGN_CONTRACT.md"
FEATURES = DESIGN_DIR / "H1_R3_FEATURE_SEMANTICS_REGISTRY.csv"
ELIGIBILITY = DESIGN_DIR / "H1_R3_ELIGIBILITY_CONTRACT.json"
FEASIBILITY = DESIGN_DIR / "H1_R3_SCORE_FEASIBILITY_ANALYSIS.md"
GOLDEN = DESIGN_DIR / "H1_R3_GOLDEN_CASES.yaml"
ACTIVATION = DESIGN_DIR / "H1_R3_ACTIVATION_GATE_DECISION.json"
DEVELOPMENT = DESIGN_DIR / "H1_R3_DEVELOPMENT_MANIFEST.csv"
VALIDATION = DESIGN_DIR / "H1_R3_VALIDATION_MANIFEST.csv"
EVALUATION = DESIGN_DIR / "H1_R3_EVALUATION_GATE.json"
DECISION = DESIGN_DIR / "H1_R3_DESIGN_DECISION.json"
REVIEW = DESIGN_DIR / "H1_R3_INDEPENDENT_DESIGN_REVIEW.json"

H1_R2_DIR = Path("docs/tracking/h1_r2")
H1_R2_DEVELOPMENT = H1_R2_DIR / "H1_R2_DEVELOPMENT_MANIFEST.csv"
H1_R2_VALIDATION = H1_R2_DIR / "H1_R2_VALIDATION_MANIFEST.csv"
H1_R2_ROLE_ASSIGNMENTS = (
    H1_R2_DIR / "H1_R2_VALIDATION_ROLE_ASSIGNMENTS.json"
)
H1_R2_EVALUATION_DECISION = (
    H1_R2_DIR / "H1_R2_DEVELOPMENT_EVALUATION_DECISION_20260727.json"
)

FEATURE_CLASSES = {
    "overlap_similarity": "CORE_REQUIRED",
    "normalized_center_similarity": "DIAGNOSTIC_ONLY",
    "scale_similarity": "DIAGNOSTIC_ONLY",
    "appearance_similarity": "OPTIONAL_WITH_MISSINGNESS_MASK",
    "motion_consistency": "OPTIONAL_WITH_MISSINGNESS_MASK",
    "track_freshness": "CORE_REQUIRED",
    "appearance_available": "OPTIONAL_WITH_MISSINGNESS_MASK",
    "motion_available": "OPTIONAL_WITH_MISSINGNESS_MASK",
}
WEIGHTS = {
    "overlap_similarity": 0.6,
    "normalized_center_similarity": 0.0,
    "scale_similarity": 0.0,
    "appearance_similarity": 0.15,
    "motion_consistency": 0.1,
    "track_freshness": 0.15,
    "appearance_available": 0.0,
    "motion_available": 0.0,
}
GOLDEN_IDS = {
    "strong_hidden_geometry_weak_visible",
    "weak_hidden_geometry_strong_visible",
    "identical_evidence",
    "overlap_available_appearance_missing_both",
    "appearance_missing_hidden_only",
    "appearance_missing_visible_only",
    "recent_hidden_with_causal_motion",
    "long_hidden_stale_evidence",
    "bbox_scale_invariance_small_and_large",
    "skipped_frame_valid_lk",
    "skipped_frame_invalid_lk",
    "partial_overlap_below_former_floor",
    "high_overlap_conflicting_visible_appearance",
    "insufficient_core_evidence",
    "threshold_just_below_boundary",
    "margin_and_threshold_boundary",
}
VALIDATION_WINDOWS = {
    "V01_000085_blinded": ("Pigs281119_000085_30fps", 360, 540),
    "V02_000114_blinded": ("Pigs281119_000114_30fps", 960, 1140),
    "V03_000327_blinded": ("Pigs301119_000327_30fps", 660, 840),
    "V04_000328_blinded": ("Pigs301119_000328_30fps", 1260, 1440),
}
PROHIBITED_SCORE_NAMES = re.compile(r"\b(p_owner|owner_probability)\b")
TOLERANCE = 1e-9


class ContractError(RuntimeError):
    """Raised when the frozen design packet is inconsistent."""


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


def _bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ContractError(f"invalid boolean: {value}")


def _unauthorized(payload: dict[str, Any], context: str) -> None:
    values = payload.get("authorizations", payload)
    for key in (
        "implementation_authorized",
        "evaluation_authorized",
        "runtime_evaluation_authorized",
        "promotion_authorized",
    ):
        if values.get(key) is not False:
            raise ContractError(f"{context} authorizes {key}")


def _check_paths(pre_review: bool) -> None:
    paths = [
        CONTRACT,
        FEATURES,
        ELIGIBILITY,
        FEASIBILITY,
        GOLDEN,
        ACTIVATION,
        DEVELOPMENT,
        VALIDATION,
        EVALUATION,
        DECISION,
    ]
    if not pre_review:
        paths.append(REVIEW)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ContractError(f"missing design artifacts: {missing}")
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (CONTRACT, ELIGIBILITY, FEASIBILITY, ACTIVATION, DECISION)
    )
    match = PROHIBITED_SCORE_NAMES.search(text)
    if match:
        raise ContractError(f"prohibited score name: {match.group(0)}")
    if "owner_preference_lower_bound is a probability" in text.lower():
        raise ContractError("uncalibrated score claims probability semantics")


def _check_features() -> None:
    columns, rows = _read_csv(FEATURES)
    required = {
        "feature_id",
        "evidence_class",
        "formula",
        "range",
        "normalization",
        "monotonic_direction",
        "missing_data_behavior",
        "validity_rule",
        "quality_or_age_decay",
        "maximum_weighted_influence",
        "hidden_available",
        "visible_available",
        "symmetric_definition",
        "causal",
        "skipped_frame_dependency",
    }
    if set(columns) != required:
        raise ContractError("feature-registry columns changed")
    by_id = {row["feature_id"]: row for row in rows}
    if set(by_id) != set(FEATURE_CLASSES) or len(rows) != 8:
        raise ContractError("feature population is not exactly eight")
    for feature_id, expected_class in FEATURE_CLASSES.items():
        row = by_id[feature_id]
        if row["evidence_class"] != expected_class:
            raise ContractError(f"{feature_id}: evidence class changed")
        if _bool(row["hidden_available"]) != _bool(row["visible_available"]):
            raise ContractError(f"{feature_id}: asymmetric availability")
        if not _bool(row["symmetric_definition"]):
            raise ContractError(f"{feature_id}: asymmetric definition")
        if not _bool(row["causal"]):
            raise ContractError(f"{feature_id}: noncausal feature")
        if not all(
            row[key].strip()
            for key in (
                "formula",
                "range",
                "normalization",
                "missing_data_behavior",
                "validity_rule",
                "quality_or_age_decay",
            )
        ):
            raise ContractError(f"{feature_id}: incomplete semantics")
        influence = float(row["maximum_weighted_influence"])
        if not math.isclose(
            influence,
            WEIGHTS[feature_id],
            abs_tol=TOLERANCE,
        ):
            raise ContractError(f"{feature_id}: influence changed")


def _check_eligibility() -> None:
    payload = _read_json(ELIGIBILITY)
    if payload.get("primary_design") != (
        "symmetric_iou_recency_core_with_conservative_optional_bounds"
    ):
        raise ContractError("primary eligibility design changed")
    pair = payload.get("pair_eligibility", {})
    if pair.get("candidate_rules_identical") is not True:
        raise ContractError("candidate eligibility is asymmetric")
    if pair.get("hidden_only_overlap_floor") is not None:
        raise ContractError("hidden-only overlap floor is present")
    if pair.get("all_features_required") is not False:
        raise ContractError("all-eight policy was restored")
    if pair.get("optional_evidence_required_for_eligibility") is not False:
        raise ContractError("optional evidence became mandatory")
    if pair.get("one_side_stricter_than_other") is not False:
        raise ContractError("one side has stricter eligibility")
    if set(pair.get("core_required", [])) != {
        "overlap_similarity",
        "track_freshness",
    }:
        raise ContractError("core evidence changed")
    if set(pair.get("diagnostic_only", [])) != {
        "normalized_center_similarity",
        "scale_similarity",
    }:
        raise ContractError("correlated geometry classification changed")
    provenance = pair.get("reference_box_provenance_order")
    if not isinstance(provenance, list) or len(provenance) != 3:
        raise ContractError("reference-box provenance is incomplete")
    lk = pair.get("lk_validity_rule", {})
    expected_lk = {
        "same_for_hidden_and_visible": True,
        "current_and_past_frames_only": True,
        "minimum_attempted_points": 4,
        "minimum_valid_points": 3,
        "maximum_forward_backward_error_pixels": 1.5,
        "quality": "valid_points/attempted_points clipped to [0,1]",
    }
    if lk != expected_lk:
        raise ContractError("LK validity is not frozen symmetrically")
    for channel, width in (("appearance", 0.15), ("motion", 0.1)):
        rules = payload.get("optional_evidence", {}).get(channel, {})
        if rules.get("missing_contribution_interval") != [-width, width]:
            raise ContractError(f"{channel}: uncertainty interval changed")
        if rules.get("activation_uses") != "lower_bound":
            raise ContractError(f"{channel}: activation is not conservative")
        if rules.get("masking_cannot_increase_lower_bound") is not True:
            raise ContractError(f"{channel}: masking may create confidence")
    causality = payload.get("causality", {})
    if (
        causality.get("current_or_past_frames_only") is not True
        or causality.get("future_frames_allowed") is not False
        or causality.get("reference_box_selection_same_for_both_sides")
        is not True
    ):
        raise ContractError("causal symmetry changed")
    _unauthorized(payload.get("frozen_authority", {}), "eligibility")


def _check_activation() -> dict[str, float]:
    payload = _read_json(ACTIVATION)
    if payload.get("score_name") != "owner_preference_lower_bound":
        raise ContractError("score name changed")
    if (
        payload.get("score_calibrated") is not False
        or payload.get("score_is_probability") is not False
        or payload.get("validation_data_used") is not False
    ):
        raise ContractError("score authority changed")
    if payload.get("weights") != WEIGHTS:
        raise ContractError("weights changed")
    interval = payload.get("relative_owner_support_interval", {})
    if interval.get("range") != [-1.0, 1.0]:
        raise ContractError("support bounds changed")
    if interval.get("swap_rule") != (
        "[lower,upper] becomes [-upper,-lower]"
    ):
        raise ContractError("swap antisymmetry changed")
    gate = payload.get("activation_rule", {})
    margin = gate.get("relative_owner_support_margin")
    threshold = gate.get("owner_preference_lower_bound_threshold")
    overlap = gate.get("minimum_relative_overlap_advantage")
    if not all(
        isinstance(value, (int, float))
        for value in (margin, threshold, overlap)
    ):
        raise ContractError("activation constants are not numeric")
    if not math.isclose(
        threshold,
        0.5 + 0.5 * margin,
        abs_tol=TOLERANCE,
    ):
        raise ContractError("threshold and support margin are incompatible")
    if not (0.0 < margin < 0.5):
        raise ContractError("conservative missing-data region is empty")
    if not (0.0 < overlap < 1.0):
        raise ContractError("relative overlap requirement is infeasible")
    if gate.get("activation_uses_worst_case_lower_bound") is not True:
        raise ContractError("activation does not use worst-case evidence")
    if gate.get("second_independent_margin_gate") is not False:
        raise ContractError("independent margin gate added")
    instrumentation = payload.get("instrumentation_before_intervention", {})
    expected = {
        "required": True,
        "reservation_enabled": False,
        "score_blind_owner_labels_required": True,
        "may_change_frozen_weights_or_gate": False,
        "separate_authorization_required": True,
    }
    for key, value in expected.items():
        if instrumentation.get(key) is not value:
            raise ContractError(f"instrumentation contract changed: {key}")
    _unauthorized(payload, "activation gate")
    return {
        "margin": float(margin),
        "threshold": float(threshold),
        "overlap": float(overlap),
    }


def _valid_box(box: Any) -> bool:
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(value, (int, float)) for value in box)
        and all(math.isfinite(float(value)) for value in box)
        and box[2] > box[0]
        and box[3] > box[1]
    )


def _box_area(box: list[float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = _box_area(first) + _box_area(second) - intersection
    return intersection / union


def _center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _diag(box: list[float]) -> float:
    return math.hypot(box[2] - box[0], box[3] - box[1])


def _center_similarity(first: list[float], second: list[float]) -> float:
    first_center = _center(first)
    second_center = _center(second)
    distance = math.dist(first_center, second_center)
    normalized = 2.0 * distance / (_diag(first) + _diag(second))
    return 1.0 - min(max(normalized, 0.0), 1.0)


def _scale_similarity(first: list[float], second: list[float]) -> float:
    ratio = _box_area(second) / _box_area(first)
    residual = abs(math.log(ratio)) / math.log(4.0)
    return 1.0 - min(max(residual, 0.0), 1.0)


def _hist_similarity(first: Any, second: Any) -> float | None:
    if (
        not isinstance(first, list)
        or not isinstance(second, list)
        or len(first) != len(second)
        or not first
    ):
        return None
    if any(value < 0 or not math.isfinite(value) for value in first + second):
        return None
    first_sum = sum(first)
    second_sum = sum(second)
    if first_sum <= 0 or second_sum <= 0:
        return None
    first_norm = [value / first_sum for value in first]
    second_norm = [value / second_sum for value in second]
    squared = sum(
        (math.sqrt(left) - math.sqrt(right)) ** 2
        for left, right in zip(first_norm, second_norm, strict=True)
    )
    return 1.0 - min(max(math.sqrt(0.5 * squared), 0.0), 1.0)


def _candidate_features(
    candidate: dict[str, Any],
    detection: dict[str, Any],
) -> dict[str, float | bool]:
    reference = candidate.get("reference_box")
    detection_box = detection.get("box")
    eligible = _valid_box(reference) and _valid_box(detection_box)
    age = candidate.get("age")
    eligible = (
        eligible
        and isinstance(age, int)
        and 0 <= age <= 8
    )
    if not eligible:
        return {"eligible": False}
    reference = [float(value) for value in reference]
    detection_box = [float(value) for value in detection_box]
    appearance = _hist_similarity(
        candidate.get("hist"),
        detection.get("hist"),
    )
    descriptor_age = candidate.get("descriptor_age")
    appearance_available = (
        appearance is not None
        and isinstance(descriptor_age, int)
        and descriptor_age >= 0
    )
    appearance_quality = (
        2.0 ** (-descriptor_age / 4.0)
        if appearance_available
        else 0.0
    )
    prediction = candidate.get("predicted_box")
    motion_available = _valid_box(prediction)
    prediction_age = candidate.get("prediction_age")
    base_quality = candidate.get("prediction_base_quality")
    motion_available = (
        motion_available
        and isinstance(prediction_age, int)
        and prediction_age >= 0
        and isinstance(base_quality, (int, float))
        and 0.0 <= base_quality <= 1.0
    )
    motion = 0.5
    motion_quality = 0.0
    if motion_available:
        prediction = [float(value) for value in prediction]
        motion = _center_similarity(prediction, detection_box)
        motion_quality = base_quality * 2.0 ** (-prediction_age / 2.0)
    return {
        "eligible": True,
        "overlap": _iou(reference, detection_box),
        "center": _center_similarity(reference, detection_box),
        "scale": _scale_similarity(reference, detection_box),
        "freshness": 1.0 - age / 8.0,
        "appearance": appearance if appearance is not None else 0.5,
        "appearance_available": appearance_available,
        "appearance_quality": appearance_quality,
        "motion": motion,
        "motion_available": motion_available,
        "motion_quality": motion_quality,
    }


def _support_interval(
    hidden: dict[str, float | bool],
    visible: dict[str, float | bool],
) -> tuple[float, float, float]:
    overlap_delta = float(hidden["overlap"]) - float(visible["overlap"])
    freshness_delta = (
        float(hidden["freshness"]) - float(visible["freshness"])
    )
    core = 0.6 * overlap_delta + 0.15 * freshness_delta
    if hidden["appearance_available"] and visible["appearance_available"]:
        quality = min(
            float(hidden["appearance_quality"]),
            float(visible["appearance_quality"]),
        )
        appearance = (
            0.15
            * quality
            * (
                float(hidden["appearance"])
                - float(visible["appearance"])
            )
        )
        appearance_bounds = (appearance, appearance)
    else:
        appearance_bounds = (-0.15, 0.15)
    if hidden["motion_available"] and visible["motion_available"]:
        quality = min(
            float(hidden["motion_quality"]),
            float(visible["motion_quality"]),
        )
        motion = (
            0.1
            * quality
            * (float(hidden["motion"]) - float(visible["motion"]))
        )
        motion_bounds = (motion, motion)
    else:
        motion_bounds = (-0.1, 0.1)
    return (
        overlap_delta,
        core + appearance_bounds[0] + motion_bounds[0],
        core + appearance_bounds[1] + motion_bounds[1],
    )


def _decision(
    eligible: bool,
    overlap_delta: float,
    lower: float,
    upper: float,
    constants: dict[str, float],
) -> str:
    if not eligible:
        return "ABSTAIN_INELIGIBLE"
    if (
        overlap_delta + TOLERANCE >= constants["overlap"]
        and lower + TOLERANCE >= constants["margin"]
    ):
        return "ACTIVATE_HIDDEN_OWNER"
    if (
        overlap_delta - TOLERANCE <= -constants["overlap"]
        and upper - TOLERANCE <= -constants["margin"]
    ):
        return "RETAIN_VISIBLE_COMPETITOR"
    return "ABSTAIN_AMBIGUOUS"


def _check_lk_state(candidate: dict[str, Any]) -> None:
    if candidate.get("reference_source") != "causal_lk":
        return
    if (
        candidate.get("lk_attempted", 0) < 4
        or candidate.get("lk_valid", 0) < 3
        or candidate.get("lk_fb_error", math.inf) > 1.5
    ):
        raise ContractError("golden case uses invalid LK as causal reference")


def _scaled_case(case: dict[str, Any], factor: float) -> dict[str, Any]:
    clone = json.loads(json.dumps(case))
    for section in ("detection", "hidden", "visible"):
        for key in ("box", "reference_box", "predicted_box"):
            box = clone[section].get(key)
            if box is not None:
                clone[section][key] = [factor * value for value in box]
    return clone


def _case_result(
    case: dict[str, Any],
    constants: dict[str, float],
) -> tuple[str, tuple[float, float, float] | None]:
    _check_lk_state(case["hidden"])
    _check_lk_state(case["visible"])
    hidden = _candidate_features(case["hidden"], case["detection"])
    visible = _candidate_features(case["visible"], case["detection"])
    eligible = bool(hidden["eligible"]) and bool(visible["eligible"])
    if not eligible:
        return _decision(False, 0.0, 0.0, 0.0, constants), None
    interval = _support_interval(hidden, visible)
    return _decision(True, *interval, constants), interval


def _check_golden(constants: dict[str, float]) -> dict[str, int]:
    payload = _read_json(GOLDEN)
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ContractError("golden cases missing")
    by_id = {case.get("case_id"): case for case in cases}
    if set(by_id) != GOLDEN_IDS or len(cases) != 16:
        raise ContractError("golden population changed")
    counts = {"activate": 0, "visible": 0, "abstain": 0, "realistic": 0}
    intervals: dict[str, tuple[float, float, float]] = {}
    for case in cases:
        decision, interval = _case_result(case, constants)
        if decision != case.get("expected_decision"):
            raise ContractError(f"{case['case_id']}: decision mismatch")
        if interval is not None:
            intervals[case["case_id"]] = interval
        if decision == "ACTIVATE_HIDDEN_OWNER":
            counts["activate"] += 1
            if case.get("realistic_nonperfect") is True:
                counts["realistic"] += 1
        elif decision == "RETAIN_VISIBLE_COMPETITOR":
            counts["visible"] += 1
        else:
            counts["abstain"] += 1
        for factor in case.get("uniform_scale_variants", []):
            scaled_decision, scaled_interval = _case_result(
                _scaled_case(case, factor),
                constants,
            )
            if scaled_decision != decision or scaled_interval is None:
                raise ContractError("bbox scale invariance failed")
            assert interval is not None
            if any(
                not math.isclose(left, right, abs_tol=TOLERANCE)
                for left, right in zip(
                    interval,
                    scaled_interval,
                    strict=True,
                )
            ):
                raise ContractError("normalized geometry changed with scale")
    if counts["realistic"] < 2:
        raise ContractError("realistic activation region is empty")
    if counts["visible"] < 2 or counts["abstain"] < 2:
        raise ContractError("decision regions are degenerate")
    partial = by_id["partial_overlap_below_former_floor"]
    hidden = _candidate_features(partial["hidden"], partial["detection"])
    if (
        float(hidden["overlap"]) >= 0.5
        or partial["expected_decision"] != "ACTIVATE_HIDDEN_OWNER"
    ):
        raise ContractError("former overlap floor still controls activation")
    strong = intervals["strong_hidden_geometry_weak_visible"]
    swapped = intervals["weak_hidden_geometry_strong_visible"]
    if not all(
        math.isclose(left, right, abs_tol=TOLERANCE)
        for left, right in zip(
            strong,
            (-swapped[0], -swapped[2], -swapped[1]),
            strict=True,
        )
    ):
        raise ContractError("role-swap complement failed")
    return counts


def _check_masking_counterfactuals() -> None:
    for width in (0.15, 0.1):
        missing_lower = -width
        for step in range(21):
            observed = -width + 2.0 * width * step / 20.0
            if missing_lower > observed + TOLERANCE:
                raise ContractError("masking increases hidden lower bound")


def _check_manifests() -> tuple[int, int]:
    _, development = _read_csv(DEVELOPMENT)
    _, validation = _read_csv(VALIDATION)
    if _sha256(DEVELOPMENT) != _sha256(H1_R2_DEVELOPMENT):
        raise ContractError("development population changed")
    if _sha256(VALIDATION) != _sha256(H1_R2_VALIDATION):
        raise ContractError("validation population or boundaries changed")
    development_videos = {row["video_key"] for row in development}
    validation_videos = {row["video_key"] for row in validation}
    development_sessions = {
        row["recording_date_session"] for row in development
    }
    validation_sessions = {
        row["recording_date_session"] for row in validation
    }
    if development_videos & validation_videos:
        raise ContractError("development and validation overlap by video")
    if development_sessions & validation_sessions:
        raise ContractError("development and validation overlap by session")
    if any(row["validation_eligible"] != "false" for row in development):
        raise ContractError("development episode became validation eligible")
    windows = {
        row["episode_id"]: (
            row["video_key"],
            int(row["start_frame"]),
            int(row["end_frame"]),
        )
        for row in validation
    }
    if windows != VALIDATION_WINDOWS:
        raise ContractError("validation boundaries changed")
    if _sha256(H1_R2_ROLE_ASSIGNMENTS) != (
        "ae867355ff5ee04693451a52121e31606364d96def69dc7c0a03a585dfac3f0f"
    ):
        raise ContractError("validation role artifact changed")
    return len(development), len(validation)


def _check_no_validation_outputs() -> None:
    root = Path("outputs/tracking")
    if not root.exists():
        return
    forbidden = [
        str(path)
        for path in root.rglob("*")
        if "h1_r3" in str(path).lower()
        and "validation" in str(path).lower()
    ]
    if forbidden:
        raise ContractError(f"H1-r3 validation output exists: {forbidden[:3]}")


def _check_decisions(pre_review: bool) -> None:
    h1_r2 = _read_json(H1_R2_EVALUATION_DECISION)
    if h1_r2.get("decision") != "FAIL_NO_ACTIVATION":
        raise ContractError("H1-r2 was reopened")
    evaluation = _read_json(EVALUATION)
    _unauthorized(evaluation, "evaluation gate")
    causality = evaluation.get("causality_gates", {})
    if (
        causality.get("execution_contract") != "causal_framewise"
        or causality.get("output_delay_frames") != 0
        or causality.get("prefix_invariance_required") is not True
        or causality.get("future_frame_access_allowed") is not False
    ):
        raise ContractError("causality gate changed")
    if evaluation.get("artifact_gates", {}).get(
        "recursive_run_root_mp4_count"
    ) != 0:
        raise ContractError("zero-MP4 gate missing")
    decision = _read_json(DECISION)
    _unauthorized(decision, "design decision")
    primary = decision.get("primary_design", {})
    if (
        primary.get("hidden_only_overlap_gate_present") is not False
        or primary.get("calibrated") is not False
        or primary.get("is_probability") is not False
    ):
        raise ContractError("design score authority changed")
    bindings = decision.get("artifact_bindings", {})
    paths = {
        CONTRACT.name: CONTRACT,
        FEATURES.name: FEATURES,
        ELIGIBILITY.name: ELIGIBILITY,
        FEASIBILITY.name: FEASIBILITY,
        GOLDEN.name: GOLDEN,
        ACTIVATION.name: ACTIVATION,
        DEVELOPMENT.name: DEVELOPMENT,
        VALIDATION.name: VALIDATION,
        EVALUATION.name: EVALUATION,
    }
    if set(bindings) != set(paths):
        raise ContractError("artifact binding set changed")
    for name, path in paths.items():
        if bindings[name] != _sha256(path):
            raise ContractError(f"artifact hash mismatch: {name}")
    result = decision.get("independent_review_result")
    if pre_review and result not in {"PENDING", "PASS_DESIGN"}:
        raise ContractError("design is not eligible for review")
    if not pre_review and result != "PASS_DESIGN":
        raise ContractError("PASS_DESIGN review is absent")


def _check_review() -> None:
    review = _read_json(REVIEW)
    if review.get("review_result") != "PASS_DESIGN":
        raise ContractError("independent review did not pass")
    if review.get("independent_from_authoring_process") is not True:
        raise ContractError("review independence is absent")
    _unauthorized(review, "independent review")
    checks = review.get("checks")
    if not isinstance(checks, list) or len(checks) < 7:
        raise ContractError("review challenge coverage is incomplete")
    if any(check.get("result") != "PASS" for check in checks):
        raise ContractError("review contains a failed challenge")
    bindings = review.get("artifact_sha256", {})
    paths = {
        CONTRACT.name: CONTRACT,
        FEATURES.name: FEATURES,
        ELIGIBILITY.name: ELIGIBILITY,
        FEASIBILITY.name: FEASIBILITY,
        GOLDEN.name: GOLDEN,
        ACTIVATION.name: ACTIVATION,
        DEVELOPMENT.name: DEVELOPMENT,
        VALIDATION.name: VALIDATION,
        EVALUATION.name: EVALUATION,
    }
    for name, path in paths.items():
        if bindings.get(name) != _sha256(path):
            raise ContractError(f"review hash mismatch: {name}")


def check_contract(*, pre_review: bool) -> dict[str, int]:
    _check_paths(pre_review)
    _check_features()
    _check_eligibility()
    constants = _check_activation()
    counts = _check_golden(constants)
    _check_masking_counterfactuals()
    development, validation = _check_manifests()
    _check_no_validation_outputs()
    _check_decisions(pre_review)
    if not pre_review:
        _check_review()
    return {**counts, "development": development, "validation": validation}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-review", action="store_true")
    args = parser.parse_args(argv)
    try:
        counts = check_contract(pre_review=args.pre_review)
    except (ContractError, ValueError, KeyError, TypeError) as exc:
        print(f"H1_R3_DESIGN_CHECKER=FAIL: {exc}", file=sys.stderr)
        return 1
    status = "PRE_REVIEW_PASS" if args.pre_review else "PASS"
    print(f"H1_R3_DESIGN_CHECKER={status}")
    print(f"FEATURE_COUNT={len(FEATURE_CLASSES)}")
    print(f"GOLDEN_CASE_COUNT={len(GOLDEN_IDS)}")
    print(f"REALISTIC_NONPERFECT_ACTIVATIONS={counts['realistic']}")
    print(f"VISIBLE_SUPPORT_CASES={counts['visible']}")
    print(f"AMBIGUOUS_OR_INELIGIBLE_CASES={counts['abstain']}")
    print(f"DEVELOPMENT_EPISODE_COUNT={counts['development']}")
    print(f"VALIDATION_EPISODE_COUNT={counts['validation']}")
    print("IMPLEMENTATION_AUTHORIZED=NO")
    print("EVALUATION_AUTHORIZED=NO")
    print("RUNTIME_EVALUATION_AUTHORIZED=NO")
    print("PROMOTION_AUTHORIZED=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
